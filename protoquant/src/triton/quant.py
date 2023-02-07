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
def quant_kernel(inputs):
    # Writing out these values as constants
    # precision = 8
    # qmin = ((1 << (precision - 1)) * -1)
    # qmax = ((1 << (precision - 1)) - 1)
    qmin = -128
    qmax = 127
    row_min_val = inputs.amin(dim=1)
    row_max_val = inputs.amax(dim=1)
    # row_sum_val = inputs.sum(dim=1)

    mins = row_min_val
    maxs = row_max_val

    # We extend the [min, max] interval to ensure that it contains 0.
    # Otherwise, we would not meet the requirement that 0 be an exactly
    # representable value.
    row_min_val = torch.minimum(row_min_val, torch.tensor(0)).to(torch.float32)
    row_max_val = torch.maximum(row_max_val, torch.tensor(0)).to(torch.float32)

    # Use double precision for intermediate computation but use single precision
    # in final number to reflect the actual number used during quantization.
    scale = ((row_max_val.to(torch.float64) - row_min_val) / (qmax - qmin)).to(
        torch.float32
    )

    # If scale is 0 or too small so its reciprocal is infinity, we arbitrary
    # adjust the scale to 0.1 . We want to avoid scale's reciprocal being
    # infinity because some of fbgemm code pre-computes scale's reciprocal to do
    # multiplication instead of division in the time critical part of code.
    isinf_scales = torch.isinf(1.0 / scale)
    scale = torch.where(scale == 0.0 + isinf_scales, 0.1, scale)

    # Cut off small scale
    SMALL_SCALE_THRESHOLD = 6.1e-5
    is_small_scale = scale < SMALL_SCALE_THRESHOLD
    amplifier = SMALL_SCALE_THRESHOLD / scale
    scale = torch.where(is_small_scale, SMALL_SCALE_THRESHOLD, scale)

    # Unconditionally create amplified variant for small scales
    row_max_val_amplified = torch.where(
        is_small_scale, amplifier * row_max_val, row_max_val
    ).to(torch.float32)
    row_min_val_amplified = torch.where(
        is_small_scale, amplifier * row_min_val, row_min_val
    ).to(torch.float32)

    # TODO: This doesn't match quantize.cu yet! Revisit if there are accuracy issues.
    # is_row_min_val_zero = (row_min_val == 0.0)
    # is_row_max_val_zero = (row_max_val == 0.0)
    # # Fill amplified values with regular values for special cases
    # row_max_val_amplified = tl.where(tl.minimum(is_small_scale, is_row_min_val_zero),
    #                        SMALL_SCALE_THRESHOLD * (qmax_val - qmin_val), row_max_val)
    # row_min_val_amplified = tl.where(tl.minimum(is_small_scale, is_row_max_val_zero),
    #                        -SMALL_SCALE_THRESHOLD * (qmax_val - qmin_val), row_min_val)

    row_max_val = row_max_val_amplified
    row_min_val = row_min_val_amplified

    # Zero-point computation.
    # First the initial floating-point computation. The zero-point can be
    # determined from solving an affine equation for any known pair
    # (real value, corresponding quantized value).
    # We know two such pairs: (rmin, qmin) and (rmax, qmax).
    # The arithmetic error on the zero point computed from either pair
    # will be roughly machine_epsilon * (sum of absolute values of terms)
    # so we want to use the variant that adds the smaller terms.
    scale_fp64 = scale.to(torch.float64)
    zero_point_from_min = qmin - (row_min_val / scale_fp64)
    zero_point_from_max = qmax - (row_max_val / scale_fp64)
    zero_point_from_min_error = abs(qmin) + torch.abs(row_min_val / scale_fp64)
    zero_point_from_max_error = abs(qmax) + torch.abs(row_max_val / scale_fp64)

    initial_zero_point = torch.where(
        zero_point_from_min_error < zero_point_from_max_error,
        zero_point_from_min,
        zero_point_from_max,
    )

    # Now we need to nudge the zero point to be an integer
    # (our zero points are integer, and this is motivated by the requirement
    # to be able to represent the real value "0" exactly as a quantized value,
    # which is required in multiple places, for example in Im2col with zero
    # padding).
    # TODO: Using torch.round for nearbyint. If there are accuracy issues, this
    # might be worth investigating in more depth.
    # nudged_zero_point = tl.libdevice.nearbyint(initial_zero_point).to(tl.int32)
    nudged_zero_point = torch.round(initial_zero_point).to(torch.int32)
    nudged_zero_point = torch.where(initial_zero_point < qmin, qmin, nudged_zero_point)
    nudged_zero_point = torch.where(initial_zero_point > qmax, qmax, nudged_zero_point)

    inv_scale = 1.0 / scale
    precomputed_sum = torch.round(inputs.sum(dim=1) * inv_scale).to(torch.int32)

    transformed_val = (inputs * inv_scale.unsqueeze(1)).to(torch.float32)
    transformed_val = torch.round(transformed_val) + nudged_zero_point.unsqueeze(1)

    output = torch.clamp(transformed_val, min=qmin, max=qmax).to(torch.int8)
    return mins, maxs, scale_fp64, nudged_zero_point, precomputed_sum, output


def quant(inputs, dim):
    assert dim == 1
    n_rows, n_cols = inputs.shape

    assert inputs.is_contiguous()

    # with _reenter_functionalization():
    return quant_kernel(inputs)
