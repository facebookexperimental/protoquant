import itertools

import torch
import torch.utils.benchmark as benchmark
from quant_primitives import dequantize_per_tensor, dynamically_quantize_per_tensor
from tabulate import tabulate


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


torch._inductor.config.epilogue_fusion = False


@torch.inference_mode()
def run():
    device = "cuda"
    float_dtype = torch.half

    n_vals = (512, 2048, 16384)
    k_vals = (512, 2048, 16384)

    results = []
    for n, k in itertools.product(n_vals, k_vals):
        shape_x = (n, k)
        print(f"shape: {shape_x}")
        X0 = torch.randn(*shape_x, device=device, dtype=float_dtype)

        q_per_tensor_int8_ms = benchmark_fn_in_ms(
            dynamically_quantize_per_tensor, X0, -128, 127, torch.int8
        )

        trit_q = torch.compile(dynamically_quantize_per_tensor, mode="max-autotune")
        trit_q(X0, -128, 127, torch.int8)  # autotune it
        trit_q_per_tensor_int8_ms = benchmark_fn_in_ms(
            trit_q, X0, -128, 127, torch.int8
        )

        trit_q_per_tensor_int8_speedup = (
            q_per_tensor_int8_ms / trit_q_per_tensor_int8_ms
        )

        X0_q, X0_scales, X0_zero_points = dynamically_quantize_per_tensor(
            X0, -128, 127, torch.int8
        )
        del X0

        dq_per_tensor_int8_ms = benchmark_fn_in_ms(
            dequantize_per_tensor, X0_q, X0_scales, X0_zero_points, torch.half
        )
        trit_dq = torch.compile(dequantize_per_tensor, mode="max-autotune")
        trit_dq(X0_q, X0_scales, X0_zero_points, torch.half)

        trit_dq_per_tensor_int8_ms = benchmark_fn_in_ms(
            trit_dq, X0_q, X0_scales, X0_zero_points, torch.half
        )

        trit_dq_per_tensor_int8_speedup = (
            dq_per_tensor_int8_ms / trit_dq_per_tensor_int8_ms
        )

        results.append(
            [
                shape_x,
                q_per_tensor_int8_ms,
                trit_q_per_tensor_int8_ms,
                dq_per_tensor_int8_ms,
                trit_dq_per_tensor_int8_ms,
                trit_q_per_tensor_int8_speedup,
                trit_dq_per_tensor_int8_speedup,
            ]
        )

        print(
            f" quantize per tensor, eager:{q_per_tensor_int8_ms:0.2f} triton:{trit_q_per_tensor_int8_ms:0.2f} \
              , dequantize per tensor, eager:{dq_per_tensor_int8_ms:0.2f} triton:{trit_dq_per_tensor_int8_ms:0.2f}"
        )
        print("\n")

    print(
        tabulate(
            results,
            headers=[
                "shape_x",
                "q_per_tensor_int8_ms",
                "trit_q_per_tensor_int8_ms",
                "dq_per_tensor_int8_ms",
                "trit_dq_per_tensor_int8_ms",
                "trit_q_per_tensor_int8_speedup",
                "trit_dq_per_tensor_int8_speedup",
            ],
        )
    )


if __name__ == "__main__":
    run()
