from collections import namedtuple
from typing import Optional, Tuple

import torch

import triton

from protoquant import pad

from .src.triton.dequant import dequant
from .src.triton.quant import quant

QParams = namedtuple(
    "QParams",
    ["scales", "zeros", "sums", "rowwise", "transpose", "dtype", "pad_0", "pad_1"],
)


def qntz(
    input: torch.Tensor,
    is_a: bool,
    do_pad: bool | None = True,
    minimize_error=True,
) -> tuple[torch.Tensor, QParams]:
    assert input.dim() == 2

    dtype = input.dtype

    rowwise = is_a
    transpose = not is_a

    m, n = input.size()
    batch = m if rowwise else n

    pad_0 = pad(m) - m if do_pad else 0
    pad_1 = pad(n) - n if do_pad else 0

    if rowwise and not transpose and pad_0 == 0 and pad_1 == 0:
        mins, maxs, scales, zeros, sums, out = quant(input, 1, minimize_error)
        params = QParams(scales, zeros, sums, rowwise, transpose, dtype, pad_0, pad_1)
        return (out, params)

    assert minimize_error
    out = torch.empty(
        [n + pad_1, m + pad_0] if transpose else [m + pad_0, n + pad_1],
        dtype=torch.int8,
        device=input.device,
    )
    zeros = torch.empty(batch, dtype=torch.int32, device=input.device)
    sums = torch.empty(batch, dtype=torch.int32, device=input.device)
    scales = torch.empty(batch, dtype=torch.float64, device=input.device)
    params = QParams(scales, zeros, sums, rowwise, transpose, dtype, pad_0, pad_1)
    torch.ops.protoquant.qntz_out(
        out,
        input,
        params.scales,
        params.zeros,
        params.sums,
        params.rowwise,
        params.transpose,
    )
    return (out, params)


def dqntz(
    input: torch.Tensor,
    mat1_params: QParams,
    mat2_params: QParams | None = None,
    other: torch.Tensor | None = None,
) -> torch.Tensor:
    assert input.dim() == 2

    if mat2_params is None:
        m, n = input.size()[::-1] if mat1_params.transpose else input.size()

        out = torch.empty(
            [m - mat1_params.pad_0, n - mat1_params.pad_1],
            dtype=mat1_params.dtype,
            device=input.device,
        )

        torch.ops.protoquant.dqntz_out(
            out,
            input,
            mat1_params.scales,
            mat1_params.zeros,
            mat1_params.sums,
            mat1_params.rowwise,
            mat1_params.transpose,
        )
        return out

    if mat1_params.pad_0 == 0 and mat2_params.pad_1 == 0:
        return dequant(
            input,
            other,
            mat1_params.scales,
            mat1_params.zeros,
            mat1_params.sums,
            mat1_params.rowwise,
            mat1_params.transpose,
            mat2_params.scales,
            mat2_params.zeros,
            mat2_params.sums,
            mat2_params.rowwise,
            mat2_params.transpose,
        )

    m, n = input.size()
    out = torch.empty(
        [m - mat1_params.pad_0, n - mat2_params.pad_1],
        dtype=mat1_params.dtype,
        device=input.device,
    )

    if other is None:
        torch.ops.protoquant.dqntz_mm_out(
            out,
            input,
            mat1_params.scales,
            mat1_params.zeros,
            mat1_params.sums,
            mat1_params.rowwise,
            mat1_params.transpose,
            mat2_params.scales,
            mat2_params.zeros,
            mat2_params.sums,
            mat2_params.rowwise,
            mat2_params.transpose,
        )
        return out

    torch.ops.protoquant.dqntz_mm_add_out(
        out,
        input,
        other,
        mat1_params.scales,
        mat1_params.zeros,
        mat1_params.sums,
        mat1_params.rowwise,
        mat1_params.transpose,
        mat2_params.scales,
        mat2_params.zeros,
        mat2_params.sums,
        mat2_params.rowwise,
        mat2_params.transpose,
    )
    return out
