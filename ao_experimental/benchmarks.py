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

    # n_vals = (32, 512, 8192)
    # k_vals = (32, 512, 8192)
    # m_vals = (32, 512, 8192)
    # n_vals = (2048, 16384)
    # k_vals = (2048, 16384)
    # m_vals = (2048, 16384)
    n_vals = (32,)
    k_vals = (32,)
    m_vals = (32,)

    results = []
    for n, k, m in itertools.product(n_vals, k_vals, m_vals):
        shape_x = (n, k)
        shape_w = (k, m)
        print(f"shapes: {shape_x}, {shape_w}")
        X0 = torch.randn(*shape_x, device=device, dtype=float_dtype)
        W0 = torch.randn(*shape_w, device=device, dtype=float_dtype)

        mm_half_ms = benchmark_fn_in_ms(torch.mm, X0, W0)        
        int_mm_ms = benchmark_fn_in_ms(torch._int_mm, X0, W0)
        safe_int_mm_ms = benchmark_fn_in_ms(safe_int_mm, X0, W0)

        int_mm_opt = torch.compile(int_mm_wrap, mode='max-autotune')
        int_mm_opt(X0, W0)  # autotune it
        int_mm_opt_ms = benchmark_fn_in_ms(int_mm_opt, X0, W0)



        # now do it with transposed weight, which is faster
        W1 = W1.t().contiguous()
        int_mm_opt(X1, W1.t())
        int_mm_opt_t_ms = benchmark_fn_in_ms(int_mm_opt, X1, W1.t())

        int_mm_speedup = mm_half_ms / int_mm_ms
        int_mm_opt_speedup = mm_half_ms / int_mm_opt_ms
        int_mm_opt_t_speedup = mm_half_ms / int_mm_opt_t_ms
        del X1, W1

        results.append([
            shape_x, shape_w, mm_half_ms, int_mm_ms, int_mm_opt_ms, int_mm_opt_t_ms,
            int_mm_speedup, int_mm_opt_speedup, int_mm_opt_t_speedup])

        print(f"  half_eag_ms: {mm_half_ms:0.2f}, int8_eag_ms: {int_mm_ms:0.2f}, int8_opt_ms: {int_mm_opt_ms:0.2f}, int8_opt_t_ms: {int_mm_opt_t_ms:0.2f}")
        print(f"  eag_speedup: {int_mm_speedup:0.2f}, opt_speedup: {int_mm_opt_speedup:0.2f}, opt_t_speedup: {int_mm_opt_t_speedup:0.2f}")
        print('\n')

    print(tabulate(results,
        headers=['shape_x', 'shape_w', 'half_ms', 'int8_eag_ms', 'int8_opt_ms', 'int8_opt_t_ms',
                 'eag_speedup', 'opt_speedup', 'opt_t_speedup']))

if __name__ == '__main__':
    run()