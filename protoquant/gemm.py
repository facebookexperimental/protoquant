from typing import Optional

import torch

try:
    from .src.triton.matmul import matmul
except ImportError:
    matmul = None


def gemm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    input: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    # pyre-fixme[7]: Expected `Tensor` but got implicit return value of `None`.
) -> torch.Tensor:
    assert (input is None) or (out is None)

    if matmul is not None and input is None and out is None:
        return matmul(mat1, mat2.t())

    if (input is None) and (out is None):
        return torch.ops.protoquant.ngemm(mat1, mat2)  # mat1 @ mat2

    if input is None:
        return torch.ops.protoquant.gemm_out(out, mat1, mat2)  # out = mat1 @ mat2

    if out is None:
        return torch.ops.protoquant.gemm(
            input, mat1, mat2
        )  # input = input + mat1 @ mat2


def pad(x: int) -> int:
    PADDING = 8
    if x % PADDING:
        return x + PADDING - (x % PADDING)
    return x
