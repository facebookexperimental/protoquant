import argparse
import csv
import itertools
import sys
import time
from functools import partial

import protoquant

import torch

import torch.utils.benchmark as benchmark


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    # Manual warmup
    f(*args, **kwargs)
    f(*args, **kwargs)

    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    measurement = t0.blocked_autorange()
    return measurement.mean * 1e6


class FFN(torch.nn.Module):
    def __init__(self, d_model, dim_feedforward, device, dtype):
        super(FFN, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.activation = torch.nn.functional.relu
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model, **factory_kwargs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return self.linear2(x)


# torch._inductor.config.implicit_fallbacks = False
# torch._dynamo.config.verbose = True
# torch._inductor.config.debug = True
# torch._inductor.triton.cudagraphs=True


def run_benchmark(
    use_q, d_model, dim_feedforward, batch_size, seq_len, minimize_error=True
):
    inp = torch.randn(batch_size, seq_len, d_model)
    inp = inp.half().cuda()
    ffn = FFN(
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        device="cuda",
        dtype=torch.float16,
    )
    ffn = ffn.half().cuda().eval()
    fp16_ref = ffn(inp).detach().clone().float()
    if use_q:
        ffn.linear1 = protoquant.qlinear_from_linear(ffn.linear1, minimize_error)
        ffn.linear2 = protoquant.qlinear_from_linear(ffn.linear2, minimize_error)
        # ffn = torch.compile(ffn, options={"max-autotune": True})
        fp8_ref = ffn(inp).detach().clone().float()
        torch.testing.assert_close(fp16_ref, fp8_ref, atol=3e-2, rtol=3e-2)
    return benchmark_torch_function_in_microseconds(ffn, inp)


def get_default_shapes():
    for i, (d_model, dim_feedforward) in enumerate(
        itertools.product([1024, 2048, 4096, 8192], [1024, 2048, 4096, 8192])
    ):
        yield (d_model, dim_feedforward, f"default{i}")


def get_big_shapes():
    for i, (d_model, dim_feedforward) in enumerate(
        itertools.product([8192, 16384], [8192, 16384])
    ):
        yield (d_model, dim_feedforward, f"big_zucchini{i}")
    for i, (d_model, dim_feedforward) in enumerate(
        itertools.product([10240, 20480], [20480, 10240])
    ):
        yield (d_model, dim_feedforward, f"big_genesis{i}")
    for i, (d_model, dim_feedforward) in enumerate(
        itertools.product([6144, 12288], [12288, 6144])
    ):
        yield (d_model, dim_feedforward, f"big_opt{i}")


def get_opt_shapes():
    d_model = [
        1536,
        2048,
        2560,
        4096,
        5120,
        7168,
        9216,
        12288,
    ]

    dim_feedforward = [
        6144,
        8192,
        10240,
        16384,
        20480,
        28672,
        36864,
        49152,
    ]

    annotation = [
        "760M",
        "1.3B",
        "2.7B",
        "6.7B",
        "13B",
        "30B",
        "66B",
        "175B",
    ]

    for d, f, a in zip(d_model, dim_feedforward, annotation):
        yield (d, f, a)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("batchsize")
    parser.add_argument("seq_len")
    parser.add_argument("--opt-shapes", action="store_true")
    parser.add_argument("--big-shapes", action="store_true")
    args = parser.parse_args()

    headers = [
        "bs",
        "seq_len",
        "kind",
        "d_model",
        "dim_feedforward",
        "with_q(μs)",
        "without_q(μs)",
        "minimize_error",
        "speedup",
    ]
    shape_gen = get_default_shapes
    if args.opt_shapes:
        shape_gen = get_opt_shapes
    if args.big_shapes:
        shape_gen = get_big_shapes
    print(",".join(headers))
    bs = int(args.batchsize)
    seq_len = int(args.seq_len)
    for d_model, dim_feedforward, annotation in shape_gen():
        for minimize_error in [True, False]:
            with_q = run_benchmark(
                True, d_model, dim_feedforward, bs, seq_len, minimize_error
            )
            without_q = run_benchmark(False, d_model, dim_feedforward, bs, seq_len)
            print(
                ",".join(
                    map(
                        str,
                        [
                            bs,
                            seq_len,
                            annotation,
                            d_model,
                            dim_feedforward,
                            f"{with_q:.0f}",
                            f"{without_q:.0f}",
                            minimize_error,
                            f"{without_q / with_q:.2f}",
                        ],
                    )
                )
            )
