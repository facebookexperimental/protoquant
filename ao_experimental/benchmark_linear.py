import torch
import torch.utils.benchmark as benchmark

import itertools
from tabulate import tabulate
from quant_primitives import quant_int8_matmul, dynamically_quantize_per_tensor
from quantized_modules import DynamicallyQuantizedLinear

torch._inductor.config.epilogue_fusion = False


# taken from https://fburl.com/spo9gm31
def benchmark_fn_in_ms(f, *args, **kwargs):
    # Manual warmup
    f(*args, **kwargs)
    f(*args, **kwargs)

    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    measurement = t0.blocked_autorange()
    return measurement.mean * 1e6


# adapted from
torch._inductor.config.epilogue_fusion = False


@torch.inference_mode()
def run():
    device = "cuda"
    float_dtype = torch.half

    n_vals = (512, 2048, 16384)
    k_vals = (512, 2048, 16384)
    m_vals = (512, 2048, 16384)

    results = []
    for n, k, m in itertools.product(n_vals, k_vals, m_vals):
        shape_x = (n, k)
        shape_w = (k, m)
        print(f"shapes: {shape_x}, {shape_w}")
        X0 = torch.randn(*shape_x, device=device, dtype=float_dtype)

        lin = torch.nn.Linear(k, m, bias=True, device=device, dtype=float_dtype)

        lin_ms = benchmark_fn_in_ms(lin, X0)

        qlin = DynamicallyQuantizedLinear.from_float(lin, out_dtype=torch.half)
        del lin

        qlin_ms = benchmark_fn_in_ms(qlin, X0)

        trit_qlin = torch.compile(qlin, mode="max-autotune")
        trit_qlin(X0)
        trit_qlin_ms = benchmark_fn_in_ms(trit_qlin, X0)

        qlin_speedup = lin_ms / qlin_ms
        trit_qlin_speedup = lin_ms / trit_qlin_ms

        x_vals_int8, x_scale, x_zp = dynamically_quantize_per_tensor(X0)

        w_int8_t = qlin.w_int8_t
        w_int8_t_sums_int64 = qlin.w_int8_t_sums_int64
        w_scales = qlin.w_scales
        del X0, qlin

        matmul_ms = benchmark_fn_in_ms(
            quant_int8_matmul,
            x_vals_int8,
            x_scale,
            x_zp,
            w_int8_t,
            w_int8_t_sums_int64,
            w_scales,
            out_dtype=float_dtype,
        )

        trit_int8_matmul = torch.compile(quant_int8_matmul, mode="max-autotune")
        trit_int8_matmul(
            x_vals_int8,
            x_scale,
            x_zp,
            w_int8_t,
            w_int8_t_sums_int64,
            w_scales,
            out_dtype=float_dtype,
        )
        trit_matmul_ms = benchmark_fn_in_ms(
            trit_int8_matmul,
            x_vals_int8,
            x_scale,
            x_zp,
            w_int8_t,
            w_int8_t_sums_int64,
            w_scales,
            out_dtype=float_dtype,
        )
        del x_vals_int8, x_scale, x_zp, w_int8_t, w_int8_t_sums_int64, w_scales

        trit_matmul_speedup = matmul_ms / trit_matmul_ms

        results.append(
            [
                shape_x,
                shape_w,
                lin_ms,
                qlin_ms,
                trit_qlin_ms,
                qlin_speedup,
                trit_qlin_speedup,
                matmul_ms,
                trit_matmul_ms,
                trit_matmul_speedup,
            ]
        )

    print(
        tabulate(
            results,
            headers=[
                "shape_x",
                "shape_w",
                "lin_ms",
                "qlin_ms",
                "trit_qlin_ms",
                "qlin_speedup",
                "trit_qlin_speedup",
                "matmul_ms",
                "trit_matmul_ms",
                "trit_matmul_speedup",
            ],
        )
    )


if __name__ == "__main__":
    run()
