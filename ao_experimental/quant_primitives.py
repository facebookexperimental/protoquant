import torch
from torch import _dynamo

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