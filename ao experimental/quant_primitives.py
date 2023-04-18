import torch

# copy-pasta of https://www.internalfb.com/intern/anp/view/?id=3350736
def dynamically_quantize_per_tensor(x, quant_min, quant_max, target_dtype):
    # assumes affine quantization

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    # TODO(future): make torch.aminmax work on cpu-half
    # min_val, max_val = torch.aminmax(x)
    min_val = torch.min(x)
    max_val = torch.max(x)

    # calculate scale and zero point based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
    # TODO(future): make torch.clamp with scalar work on cpu-half
    scale = torch.clamp(scale, min=eps).reshape(1)
    zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
    zero_point = torch.clamp(zero_point, quant_min, quant_max)

    # quantize based on qmin/qmax/scale/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    quant = torch.clamp(torch.round(x / scale) + zero_point, quant_min, quant_max).to(target_dtype)

    return quant, scale, zero_point

def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scale and zero point based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scale = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scale is the same dtype as the original tensor
    scale = torch.clamp(scale, min=eps).to(x.dtype)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scale/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x.transpose(0, 1) / scale
    x_round = torch.round(x_div)
    x_zp = x_round + zero_point
    x_zp = x_zp.transpose(0, 1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scale, zero_point

# reference: https://fburl.com/code/vfsygwd0
def dequantize_per_tensor(int_repr, scale, zero_point, out_dtype=torch.float32):
    return (int_repr.to(out_dtype) - zero_point) * scale

# reference: https://fburl.com/code/org0fmi3
def dequantize_per_channel(int_repr, scales, zero_points, out_dtype=torch.float32):
    # assumes axis is 0
    y = int_repr.transpose(0, 1)
    y = y.to(out_dtype)
    y = y - zero_points
    y = y * scales
    y = y.transpose(0, 1)
    return y

def quant_int8_dynamic_linear(
    x,
    x_quant_min,
    x_quant_max,
    x_q_dtype,
    w_vals_int8_t,
    w_scales,
    bias,
    out_dtype=torch.float32,
):
    # like F.linear, but with int8 dynamic quantization of activation,
    # and a quantized weight
    x_vals_int8, x_scale, x_zp = dynamically_quantize_per_tensor(
        x, x_quant_min, x_quant_max, x_q_dtype)
    mm_out = quant_int8_matmul(
        x_vals_int8, x_scale, x_zp, w_vals_int8_t, w_scales, out_dtype)
    if bias is not None:
        mm_out += bias
    return mm_out

def quant_int8_matmul(
    x_vals_int8,
    x_scale,
    x_zp,
    w_vals_int8,
    w_scales,
    out_dtype=torch.float32,
):
    # Quantized matmul of int8 operands that accumulates to int32 and returns
    # out_dtype. For now, this is written for approximate numerical
    # correctness, and things like aligning accumulation behaviors and
    # performance optimizations are left for a future PR.
    # Assumes that weight quantization is symmetric, i.e. w_zp is 0.
    # Assumes that weight quantization is per-channel.

    # see
    # https://github.com/google/gemmlowp/blob/master/doc/quantization.md
    # for an overview of quantized matmul compute

    # in scalar form, assuming out_dtype is fp32 and zw == 0:
    #
    #   Y_i_j_fp32 = sx * sw (dot(X_i, W_j) - zx * sum(W_j))
    #

    assert x_vals_int8.dtype in (torch.uint8, torch.int8), \
        f'x dtype {x_vals_int8.dtype} not yet supported'
    assert w_vals_int8.dtype == torch.int8, \
        f'w dtype {w_vals_int8.dtype} not yet supported'
    assert w_scales.dtype == out_dtype, \
        f'{w_scales.dtype} does not match {out_dtype}'

    #
    # 1. do the matrix form of dot(X_i, W_j)
    #

    if x_vals_int8.is_cuda:
        # TODO(before land): add test case for input with bsz
        tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
        y_dot_int32 = torch._int_mm(tmp, w_vals_int8)
        y_dot_int32 = y_dot_int32.reshape(*x_vals_int8.shape[:-1], -1)
    else:
        x_vals_int32 = x_vals_int8.to(torch.int32)
        w_vals_int32 = w_vals_int8.to(torch.int32)
        y_dot_int32 = torch.matmul(x_vals_int32, w_vals_int32)
    # TODO(future): consider using integer arithmetic throughout, although
    # TBD if that is actually faster on GPUs
    # need to use 32 bits here to prevent overflow for large shapes,
    # 16 bits is not enough
    y_dot_float32 = y_dot_int32.to(torch.float32)

    #
    # 2. do the matrix form of zx * sum(W_j)
    #

    w_sums_int64 = w_vals_int8.sum(dim=0)
    x_zp_times_w_sums_int64 = x_zp * w_sums_int64

    #
    # 3. connect it all together
    #

    # mm_unscaled has to stay in float32 for the next two lines to prevent overflow
    mm_unscaled_float32 = (y_dot_float32 - (x_zp * w_sums_int64))
    y = x_scale * w_scales * mm_unscaled_float32
    # can downcast only at the very end
    y = y.to(out_dtype)
    return y
