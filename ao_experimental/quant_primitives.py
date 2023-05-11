import torch
from torch._dynamo import is_compiling as dynamo_is_compiling


# copy-pasta of https://www.internalfb.com/intern/anp/view/?id=3350736
def dynamically_quantize_per_tensor(
    x: torch.Tensor,
    quant_min: int = -128,
    quant_max: int = 127,
    target_dtype: torch.dtype = torch.int8,
):
    r"""
    This function dynamically quantizes the tensor x similar to torch.quantize_per_tensor_dynamic but returns the
    int tensor, scale and zero_point separately to more easily enable int8 gpu quantization.

    Assumes affine quantization

    Args:
        x (Tensor, float): the tensor being quantized
        quant_min (int): minimum integer value desired for quantized output
        quant_max (int): maximum integer value desired for quantized output
        target_dtype (dtype): desired dtype for output tensor

    Return:
        x_q (Tensor, int): the resulting integer tensor with dtype of target_dtype
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
    # we choose to match the scale and zero_point dtypes of the above reference function, i.e.
    # fp64 scale and int64 zero_point for ease of debugging, this may change subject to analysis
    # of performance
    scale = (max_val_pos.to(torch.float64) - min_val_neg) / torch.tensor(
        [quant_max - quant_min], dtype=torch.float64
    ).to(x.device)
    scale = torch.clamp(scale, min=eps)

    zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int32)
    zero_point = torch.clamp(zero_point, quant_min, quant_max)

    # quantize based on qmin/qmax/scale/zp
    # reference: https://github.com/pytorch/pytorch/blob/e779a30d5097714acea011da6a554e43810b5d0e/aten/src/ATen/native/quantized/cuda/AffineQuantizer.cu#L60
    x_q = torch.clamp(torch.round(x / scale) + zero_point, quant_min, quant_max).to(
        target_dtype
    )

    return x_q, scale.item(), zero_point.item()


def dynamically_quantize_per_channel(
    x: torch.Tensor,
    quant_min: int = -128,
    quant_max: int = 127,
    target_dtype: torch.dtype = torch.int8,
    axis: int = 0,
):
    r"""
    This function dynamically quantizes the tensor x by channel but returns the
    int tensor, scale and zero_point separately to more easily enable int8 gpu quantization.

    Assumes symmetric quantization

    Args:
        x (Tensor, float): the tensor being quantized
        quant_min (int): minimum integer value desired for quantized output
        quant_max (int): maximum integer value desired for quantized output
        target_dtype (dtype): desired dtype for output tensor
        axis (int): the channel axis

    Return:
        x_q (Tensor, int): the resulting integer tensor with dtype of target_dtype
        scale (Tensor, float64): the dynamically calculated scale (float64)
        zero_point (Tensor, int64): the dynamically calculated zero_point (int64)
    """

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    def get_min_max_per_channel(x: torch.Tensor, axis: int):
        new_axis_list = [i for i in range(len(x.shape))]
        new_axis_list[axis] = 0
        new_axis_list[0] = axis
        x2 = x.permute(new_axis_list)
        x2 = torch.flatten(x2, start_dim=1)
        mins = x2.min(dim=1).values
        maxs = x2.max(dim=1).values
        return mins, maxs

    min_val, max_val = get_min_max_per_channel(x, axis=axis)

    # calculate scales and zero point based on min and max
    # reference: https://github.com/pytorch/pytorch/blob/a3989b2802a5b32d8793557ddb5aba36298ef2be/torch/ao/quantization/observer.py#L330
    # here we choose the scale and zero_point dtypes to be float64 and int32 to match the reference
    # implementation in the link above since there is no per channel dynamically quantized function as of now.
    # This choice of precision may change subect to performance consideration in the future.
    max_val_pos = torch.max(max_val, -min_val)

    scales = (
        2
        * max_val_pos.to(torch.float64)
        / torch.tensor([quant_max - quant_min], device=x.device).to(torch.float64)
    )
    scales = torch.clamp(scales, min=eps)
    zero_points = (
        torch.zeros(max_val_pos.size(), dtype=torch.int64, device=x.device)
        + 128
        + quant_min
    )

    # quantize based on qmin/qmax/scales/zp
    # reference: https://github.com/pytorch/pytorch/blob/bb7d9886fbd7d058146c76aa428e227d15f67e53/torch/ao/quantization/fx/_decomposed.py#L325
    x_div = x.transpose(axis, -1) / scales
    # note: certain implementations of quantize_per_channel uses inv_scale method of calculation with a float32
    # which is slightly less accurate
    # inv_scales = 1/scales
    # x_div = x.transpose(axis, -1) * inv_scales
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points
    x_zp = x_zp.transpose(axis, -1)
    x_q = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return x_q, scales, zero_points


def safe_int_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    r"""
    This function wraps torch._int_mm and avoids several undesirable behaviors of the function for certain inputs while still
    returning correct results and being torch.compiled in a performant way.

    Assumes both tensors have dimension of 2.

    Note: no error checking for torch.compiled path, if input.shape = [i, j] and j<=16 then the triton kernel
    will error.

    Args:
        input (Tensor, int8): the first tensor to be multiplied
        mat2 (Tensor, int8): the second tensor to be multiplied

    Return:
        out (Tensor, int32): the result of the matmul with device matching that of the inputs
    """

    # torch.compile path
    if dynamo_is_compiling():
        return torch._int_mm(input, mat2)

    # error checking for cublas path
    assert (
        mat2.device == input.device
    ), f"need both tensors to be on the same device but got {mat2.device} and {input.device}"
    device_cpu = "cpu" in [mat2.device.type, input.device.type]
    # with input.shape = [i,j] and mat2.shape = [j,k]
    i_is_strictly_greater_than_16 = input.shape[0] > 16
    j_is_nonzero_multiple_of_8 = (input.shape[1] % 8 == 0) and (input.shape[1] > 0)
    k_is_nonzero_multiple_of_8 = (mat2.shape[1] % 8 == 0) and (mat2.shape[1] > 0)
    bad_dimensions_for_cublas = not (
        i_is_strictly_greater_than_16
        and j_is_nonzero_multiple_of_8
        and k_is_nonzero_multiple_of_8
    )

    if device_cpu or bad_dimensions_for_cublas:
        # fallback path
        return torch.matmul(input.cpu().to(torch.int32), mat2.cpu().to(torch.int32)).to(
            input.device.type
        )

    # cublas paths
    if not mat2.is_contiguous():  # silently gives incorrect result without this
        mat2 = mat2.contiguous()
    if (not input.is_contiguous()) and (
        input.shape[0] % 8 != 0
    ):  # gives cryptic error without this
        input = (
            input.contiguous()
        )  # (it seems the transpose makes cublas check the above j constraint on i)
    return torch._int_mm(input, mat2)
