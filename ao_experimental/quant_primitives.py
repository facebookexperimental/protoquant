import torch
from torch import _dynamo

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