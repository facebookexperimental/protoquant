import contextlib

import torch

import triton
import triton.language as tl
from torch._dynamo import optimize
from torch._inductor import compile_fx
from torch._inductor.decomposition import decompositions
from torch.fx.experimental.proxy_tensor import make_fx


@contextlib.contextmanager
def _reenter_functionalization():
    # See: note [Fake Tensor Dispatch Keys]
    func_excluded = torch._C._dispatch_tls_local_exclude_set().has(
        torch._C.DispatchKey.Functionalize
    )
    torch._C._dispatch_tls_set_dispatch_key_excluded(
        torch._C.DispatchKey.Functionalize, False
    )
    try:
        yield
    finally:
        torch._C._dispatch_tls_set_dispatch_key_excluded(
            torch._C.DispatchKey.Functionalize, func_excluded
        )


# @torch.compile()
def dequant_kernel(
    inputs,
    other,
    mat1_scales,
    mat1_zeros,
    mat1_sums,
    mat2_scales,
    mat2_zeros,
    mat2_sums,
):

    temp = (mat1_zeros * mat2_sums).to(torch.int32)
    temp = temp + (mat1_sums * mat2_zeros).to(torch.int32)

    scale = mat1_scales * mat2_scales

    temp = ((inputs - temp) * (scale)).to(other.dtype)
    result = other + temp
    return (result,)


def dequant(
    inputs,
    other,
    mat1_scales,
    mat1_zeros,
    mat1_sums,
    mat1_rowwise,
    mat1_transpose,
    mat2_scales,
    mat2_zeros,
    mat2_sums,
    mat2_rowwise,
    mat2_transpose,
):

    assert (mat1_rowwise and not mat1_transpose) and (
        not mat2_rowwise and mat2_transpose
    ), "Expected mat1 to be quantized rowwise, non-transposed and mat2 to be quantized colwise, transposed!"
    n_rows, n_cols = inputs.shape
    assert inputs.is_contiguous()

    s0 = inputs.size(0)
    s1 = inputs.size(1)

    m, n = inputs.size()

    all_inputs = [
        inputs,
        other,
        mat1_scales.view(s0, 1),
        mat1_zeros.view(s0, 1),
        mat1_sums.view(s0, 1),
        mat2_scales.view(1, s1),
        mat2_zeros.view(1, s1),
        mat2_sums.view(1, s1),
    ]

    # with _reenter_functionalization():
    return dequant_kernel(*all_inputs)[0]
