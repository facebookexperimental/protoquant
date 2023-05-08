import torch
from torch import _dynamo
from typing import Union

# copy-pasta of https://www.internalfb.com/intern/anp/view/?id=3350736
def dynamically_quantize_per_tensor(x: torch.Tensor, quant_min: int = -128, quant_max: int=127, target_dtype: torch.dtype = torch.int8):
    """
    This function dynamically quantizes the tensor x similar to torch.quantize_per_tensor_dynamic but returns the 
    int tensor, scale and zero_point separately to more easily enable int8 gpu quantization.

    Assumes affine quantization

    Args:
        x (Tensor): the tensor being quantized
        quant_min (int): minimum integer value desired for quantized output
        quant_max (int): maximum integer value desired for quantized output
        target_dtype (dtype): desired dtype for output tensor

    Return:
        x_q (Tensor): the resulting integer tensor with dtype of target_dtype
        scale (float64): the dynamically calculated scale
        zero_point (int32): the dynamically calculated zero_point
    """
    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max

    # min_val, max_val = torch.aminmax(x) # compiled triton code is the same for min/max and aminmax
    min_val = torch.min(x)
    max_val = torch.max(x)

    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    
    # calculate scale and zero point based on min and max
    # reference: https://github.com/pytorch/pytorch/blob/e779a30d5097714acea011da6a554e43810b5d0e/aten/src/ATen/native/quantized/cpu/QuantUtils.h#L107
    scale = (max_val_pos.to(torch.float64) - min_val_neg) / torch.tensor([quant_max - quant_min], dtype=torch.float64).to(x.device)
    scale = torch.clamp(scale, min=eps)

    zero_point = quant_min -  torch.round(min_val_neg / scale).to(torch.int32)
    zero_point = torch.clamp(zero_point, quant_min, quant_max)

    # quantize based on qmin/qmax/scale/zp
    # reference: https://github.com/pytorch/pytorch/blob/e779a30d5097714acea011da6a554e43810b5d0e/aten/src/ATen/native/quantized/cuda/AffineQuantizer.cu#L60
    x_q = torch.clamp(torch.round(x / scale) + zero_point, quant_min, quant_max).to(target_dtype)

    return x_q, scale.item(), zero_point.item()

def dynamically_quantize_per_channel(x: torch.Tensor, quant_min: int=-128, quant_max: int=127, target_dtype: torch.dtype = torch.int8, axis: int = 0):
    """
    This function dynamically quantizes the tensor x by channel but returns the 
    int tensor, scale and zero_point separately to more easily enable int8 gpu quantization.

    Assumes symmetric quantization

    Args:
        x (Tensor): the tensor being quantized
        quant_min (int): minimum integer value desired for quantized output
        quant_max (int): maximum integer value desired for quantized output
        target_dtype (dtype): desired dtype for output tensor
        axis (int): the channel axis

    Return:
        x_q (Tensor): the resulting integer tensor with dtype of target_dtype
        scale (FloatTensor): the dynamically calculated scale (float64)
        zero_point (IntTensor): the dynamically calculated zero_point (int64)
    """

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    def get_min_max_per_channel(x: torch.Tensor, axis: int):
        new_axis_list = [i for i in range(len(x.shape))]
        new_axis_list[axis] = 0
        new_axis_list[0] = axis
        x2 = x.permute(new_axis_list)
        x2 = torch.flatten(x2, start_dim = 1)
        mins = x2.min(dim=1).values
        maxs = x2.max(dim=1).values
        return mins, maxs

    min_val, max_val = get_min_max_per_channel(x, axis=axis)

    # calculate scales and zero point based on min and max
    # reference: https://github.com/pytorch/pytorch/blob/a3989b2802a5b32d8793557ddb5aba36298ef2be/torch/ao/quantization/observer.py#L330
    max_val_pos = torch.max(max_val, -min_val)
    
    scales = 2*max_val_pos.to(torch.float64) / torch.tensor([quant_max - quant_min], device=x.device).to(torch.float64)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps)
    zero_points = torch.zeros(max_val_pos.size(), dtype=torch.int64, device=x.device)+128+quant_min

    # quantize based on qmin/qmax/scales/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x.transpose(axis, -1) / scales
    # note: quantize_per_channel uses inv_scale method of calculation with a float32 but thats slightly less accurate
    # inv_scales = 1/scales
    # x_div = x.transpose(axis, -1) * inv_scales
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points
    x_zp = x_zp.transpose(axis, -1)
    x_q = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return x_q, scales, zero_points

# reference: https://fburl.com/code/vfsygwd0
def dequantize_per_tensor(int_repr: torch.IntTensor, scale: Union[torch.Tensor, float], zero_point: Union[torch.Tensor, int], out_dtype=torch.float32):
    """This function works alongside dynamically_quantize_per_tensor to obtain a floating point tensor from a quantized tensor
    
    Args:
        int_repr (Tensor): the integer representation of the quantized tensor being dequantized
        scale (Union[torch.Tensor, float]): scale value for quantized tensor (can be a tensor or scalar)
        zero_point (Union[torch.Tensor, int]): zero point for quantized tensor (can be a tensor or scalar)
        out_dtype (dtype): desired dtype for output tensor

    Return:
        x (Tensor): the resulting float tensor with dtype of out_dtype
    """
    return (int_repr.to(out_dtype) - zero_point) * scale

# reference: https://fburl.com/code/org0fmi3
def dequantize_per_channel(int_repr: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, out_dtype=torch.float32, axis: int=0):
    """This function works alongside dynamically_quantize_per_tensor to obtain a floating point tensor from a quantized tensor
    
    Args:
        int_repr (Tensor): the integer representation of the quantized tensor being dequantized
        scales (Tensor): float tensor of scales for each channel
        zero_points (Tensor): integer tensor for zero point for each channel
        out_dtype (dtype): desired dtype for output tensor
        axis (int): the channel axis

    Return:
        x (Tensor): the resulting float tensor with dtype of out_dtype
    """
    y = int_repr.transpose(-1, axis)
    y = y.to(out_dtype)
    y = y - zero_points
    y = y * scales.to(out_dtype)
    y = y.transpose(-1, axis)
    return y

class DynamicallyQuantizedLinear(torch.nn.Module):
    def __init__(
        self,    
        w_int8_t,
        w_scales,
        x_quant_min = -128,
        x_quant_max = 127,
        x_q_dtype = torch.int8,
        out_dtype=torch.float32,
    ):
        super().__init__()
        self.register_buffer('w_int8_t', w_int8_t)
        self.register_buffer('w_int8_t_sums_int64', w_int8_t.sum(dim=0).to(torch.int64))
        self.register_buffer('w_scales', w_scales)
        self.register_buffer('bias', None)
        self.out_dtype = out_dtype
        self.x_quant_min = x_quant_min
        self.x_quant_max = x_quant_max
        self.x_q_dtype = x_q_dtype

    def forward(self, x):
        return quant_int8_dynamic_linear(
            x, 
            self.x_quant_min,
            self.x_quant_max,
            self.x_q_dtype,
            self.w_int8_t,
            self.w_int8_t_sums_int64,
            self.w_scales,
            self.bias,
            self.out_dtype
        )
    
    @classmethod
    def from_float(cls, mod, w_quant_min = -128, w_quant_max = 127, w_q_dtype = torch.int8 , w_axis = 0, x_quant_min = -128, x_quant_max = 127, x_q_dtype = torch.int8, out_dtype = torch.float32):
        assert isinstance(mod, torch.nn.Linear), f"need mod to be type torch.nn.Linear but got {type(mod)}"
        w_int8, w_scales, _ = dynamically_quantize_per_channel(mod.weight, quant_min = w_quant_min, quant_max = w_quant_max, target_dtype=w_q_dtype, axis = w_axis)
        new_qlinear = cls(w_int8.transpose(0,1), w_scales, x_quant_min, x_quant_max, x_q_dtype, out_dtype)
        if mod.bias is not None:
            new_qlinear.bias = mod.bias
        return new_qlinear

def quant_int8_dynamic_linear(
    x,
    x_quant_min,
    x_quant_max,
    x_q_dtype,
    w_int8_t,
    w_int8_t_sums_int64,
    w_scales,
    bias,
    out_dtype=torch.float32,
):
    # like F.linear, but with int8 dynamic quantization of activation,
    # and a quantized weight
    x_int8, x_scale, x_zp = dynamically_quantize_per_tensor(
        x, x_quant_min, x_quant_max, x_q_dtype)
    # w_int8_t_sums_int64 = w_int8_t.sum(dim=0)
    mm_out = quant_int8_matmul(
        x_int8, x_scale, x_zp, w_int8_t, w_int8_t_sums_int64,
        w_scales, out_dtype)
    if bias is not None:
        mm_out += bias
    return mm_out.to(out_dtype)

def quant_int8_matmul(
    x_vals_int8,
    x_scale,
    x_zp,
    w_int8_t,
    w_int8_t_sums_int64,
    w_scales,
    out_dtype=torch.float32,
):
    # Quantized matmul of int8 operands that accumulates to int32 and returns
    # out_dtype. For now, this is written for approximate numerical
    # correctness, and things like aligning accumulation behaviors and
    # performance optimizations are left for a future PR.
    # Assumes that weight quantization is symmetric, i.e. w_zp is 0.
    # Assumes that weight quantization is per-channel.
    # Assumes that weight channel axis is 0

    # see
    # https://github.com/google/gemmlowp/blob/master/doc/quantization.md
    # for an overview of quantized matmul compute

    # assuming zw == 0, . is matmul, * is element wise mult:
    # Y = X.W'                                                      # float form
    #   = ([X_int-xz] * xs) . ( [W_int'] * [ws'])                   # dequantize both
    #   = (xs * [ws']) * ([X_int . W_int'] - xz * [1_mat . W_int])  # rearrange
    # note: [1_mat . W_int] is w_int8_t_sums_int64 where 1_mat
    # note: ws is a rank 1 tensor so ws' just indicates aligning it correctly

    assert x_vals_int8.dtype == torch.int8, \
        f'x dtype {x_vals_int8.dtype} not yet supported'
    assert w_int8_t.dtype == torch.int8, \
        f'w dtype {w_int8_t.dtype} not yet supported'
    # assert w_scales.dtype == out_dtype, \
    #     f'{w_scales.dtype} does not match {out_dtype}'

    #
    # 1. calculate [X_int . W_int]
    #

    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    XW_int32 = safe_int_mm(tmp, w_int8_t)
    XW_int32 = XW_int32.reshape(*x_vals_int8.shape[:-1], -1)


    # TODO(future): consider using integer arithmetic throughout, although
    # TBD if that is actually faster on GPUs
    # need to use 32 bits here to prevent overflow for large shapes,
    # 16 bits is not enough
    XW_float32 = XW_int32.to(torch.float32)

    #
    # 2. connect it all together
    #

    # mm_unscaled has to stay in float32 for the next two lines to prevent overflow
    mm_unscaled_float32 = (XW_float32 - (x_zp * w_int8_t_sums_int64))
    y = x_scale * w_scales * mm_unscaled_float32
    # can downcast only at the very end
    y = y.to(out_dtype)
    return y

def safe_int_mm(x_int8: torch.Tensor, w_int8: torch.Tensor):
    """
    This function wraps torch._int_mm and avoids several undesirable behaviors of the function for certain inputs while still 
    returning correct results and being torch.compiled in a performant way.

    Assumes both tensors have dimension of 2.
    
    Note: no error checking for torch.compiled path, if x_int8.shape = [i, j] and j<=16 then the triton kernel
    will silently give incorrect results

    Args:
        x_int8 (Tensor, torch.int8): the first tensor to be multiplied
        w_int8 (Tensor, torch.int8): the second tensor to be multiplied

    Return:
        out (Tensor, torch.int32): the result of the matmul with device matching that of the inputs
    """

    # torch.compile path
    if torch._dynamo.is_compiling():
        return torch._int_mm(x_int8, w_int8)
    
    # error checking for cublas path    
    device_cpu = 'cpu' in [w_int8.device.type, x_int8.device.type]
    # with x_int8.shape = [i,j] and w_int8.shape = [j,k]
    i_is_strictly_greater_than_16 = (x_int8.shape[0] > 16)
    j_is_nonzero_multiple_of_8 = ((x_int8.shape[1] % 8 == 0) and (x_int8.shape[1] > 0)) 
    k_is_nonzero_multiple_of_8 = ((w_int8.shape[1] % 8 == 0) and (w_int8.shape[1] > 0))
    bad_dimensions_for_cublas = not (i_is_strictly_greater_than_16 and j_is_nonzero_multiple_of_8 and k_is_nonzero_multiple_of_8)

    if device_cpu or bad_dimensions_for_cublas:
        # fallback path
        return torch.matmul(x_int8.cpu().to(torch.int32), w_int8.cpu().to(torch.int32)).to(x_int8.device.type)

    # cublas paths
    if not w_int8.is_contiguous(): # silently gives incorrect result without this
        w_int8 = w_int8.contiguous()
    if (not x_int8.is_contiguous()) and (x_int8.shape[0] % 8 != 0): # gives cryptic error without this           
        x_int8 = x_int8.contiguous() # (it seems the transpose makes cublas check the above j constraint on i)
    return torch._int_mm(x_int8, w_int8)