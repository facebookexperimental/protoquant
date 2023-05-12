import torch
import torch.utils.benchmark as benchmark

import itertools
from tabulate import tabulate
from quant_primitives import dynamically_quantize_per_tensor, dynamically_quantize_per_channel, safe_int_mm

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
def run():
    device = 'cuda'
    float_dtype = torch.half

    n_vals = (32, 512, 8192)
    k_vals = (32, 512, 8192)
    m_vals = (32, 512, 8192)
    # n_vals = (2048, 16384)
    # k_vals = (2048, 16384)
    # m_vals = (2048, 16384)
    # n_vals = (32,)
    # k_vals = (32,)
    # m_vals = (32,)

    results = []
    for n, k, m in itertools.product(n_vals, k_vals, m_vals):
        shape_x = (n, k)
        shape_w = (k, m)
        print(f"shapes: {shape_x}, {shape_w}")
        X0 = torch.randn(*shape_x, device=device, dtype=float_dtype)
        W0 = torch.randn(*shape_w, device=device, dtype=float_dtype)

        mm_half_ms = benchmark_fn_in_ms(torch.mm, X0, W0)
        del X0, W0

        X1 = torch.randint(-128, 127, shape_x, device=device, dtype=torch.int8)
        W1 = torch.randint(-128, 127, shape_w, device=device, dtype=torch.int8)
        
        int_mm_ms = benchmark_fn_in_ms(torch._int_mm, X1, W1)
        safe_int_mm_ms = benchmark_fn_in_ms(safe_int_mm, X1, W1)
        trit_mm = torch.compile(safe_int_mm, mode='max-autotune')
        trit_mm(X1, W1)  # autotune it
        trit_int_mm_ms = benchmark_fn_in_ms(trit_mm, X1, W1)

        # now do it with transposed weight, which is faster
        W1_t = W1.t().contiguous()
        W1_tt = W1_t.t()
        assert(not W1_tt.is_contiguous())
        trit_mm(X1, W1_tt)
        trit_int_t_mm_ms = benchmark_fn_in_ms(trit_mm, X1, W1_tt)

        int_mm_speedup = mm_half_ms / int_mm_ms
        safe_int_mm_speedup = mm_half_ms / safe_int_mm_ms
        int_mm_trit_speedup = mm_half_ms / trit_int_mm_ms
        int_mm_trit_t_speedup = mm_half_ms / trit_int_t_mm_ms
        del X1, W1

        results.append([
            shape_x, shape_w, mm_half_ms, int_mm_ms, safe_int_mm_ms, trit_int_mm_ms,
            trit_int_t_mm_ms, int_mm_speedup, safe_int_mm_speedup, int_mm_trit_speedup, int_mm_trit_t_speedup])

        print(f"  half_eag_ms: {mm_half_ms:0.2f}, int8_eag_ms: {int_mm_ms:0.2f}, int8_safe_eag_ms: {safe_int_mm_ms:0.2f}, int8_trit_ms: {trit_int_mm_ms:0.2f}, int8_trit_t_ms: {trit_int_t_mm_ms:0.2f}")
        print(f"  eag_speedup: {int_mm_speedup:0.2f}, safe_eag_speedup: {safe_int_mm_speedup:0.2f}, trit_speedup: {int_mm_trit_speedup:0.2f}, trit_t_speedup: {int_mm_trit_t_speedup:0.2f}")
        print('\n')

    print(tabulate(results,
        headers=["shape_x", "shape_w", "mm_half_ms", "int_mm_ms", "safe_int_mm_ms", "trit_int_mm_ms", 
                 "trit_int_t_mm_ms", "int_mm_speedup", "safe_int_mm_speedup", "int_mm_trit_speedup", "int_mm_trit_t_speedup"]))

if __name__ == '__main__':
    run()