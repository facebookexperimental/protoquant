import copy
import logging
import math
import platform
import time
from typing import Optional, Tuple

import protoquant

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch._dynamo import config
from torch.backends.cuda import sdp_kernel, SDPBackend
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

torch.set_float32_matmul_precision("high")
# config.log_level = logging.DEBUG
# config.verbose = True


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    import torch.utils.benchmark as benchmark

    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


def evaluate(
    model: nn.Module,
    eval_data: Tensor,
    eval_seq_len,
    device,
    ntokens,
    criterion,
) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.0
    src_mask = generate_square_subsequent_mask(eval_seq_len).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, eval_seq_len):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            if seq_len != eval_seq_len:
                src_mask = src_mask[:seq_len, :seq_len]
            #            output = model(data, src_mask)
            output = model(data, causal_mask=True)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


backend_map = {
    SDPBackend.MATH: {
        "enable_math": True,
        "enable_flash": False,
        "enable_mem_efficient": False,
    },
    SDPBackend.FLASH_ATTENTION: {
        "enable_math": False,
        "enable_flash": True,
        "enable_mem_efficient": False,
    },
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False,
        "enable_flash": False,
        "enable_mem_efficient": True,
    },
}


def data_process(raw_text_iter: dataset.IterableDataset, vocab, tokenizer) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [
        torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter
    ]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def toy_data(num_tokens, d_model, dtype):
    return torch.rand((num_tokens, d_model), device=torch.device("cuda"), dtype=dtype)


def batchify(data: Tensor, bsz: int, device: str) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N, emsize]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz, emsize]
    """
    seq_len = data.size(0) // bsz
    emsize = data.size(1)
    data = data[: seq_len * bsz]
    data = data.view(bsz, seq_len, emsize)
    return data.to(device)


def get_batch(source: Tensor, i: int, min_seq_len=35) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(min_seq_len, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cpu = platform.processor()
    gpu = torch.cuda.get_device_name(device)

    print(f"torch version: {torch.__version__}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    print(f"CPU type: {cpu}")
    print(f"GPU type: {gpu}")
    print(f"Training will be performed on device {device}")

    d_model = 1024  # embedding dimension

    # SET MODEL DTYPE
    dtype = torch.half

    batch_size = 256
    d_hid = 4096  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 8  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 16  # number of heads in nn.MultiheadAttention
    dropout = 0  # 0..2  # dropout probability
    # model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    encoder_layers = TransformerEncoderLayer(
        d_model, nhead, d_hid, dropout, batch_first=True
    )
    model = TransformerEncoder(encoder_layers, nlayers)
    model = model.to(device)
    train_seq_len = 1024
    eval_seq_len = 256
    epochs = 8

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    val_data = toy_data(256 * batch_size, d_model, dtype)
    val_data = batchify(val_data, batch_size, device)

    model = model.to(dtype).eval()

    t0 = benchmark_torch_function_in_microseconds(model, val_data)
    # t1 = benchmark_torch_function_in_microseconds(torch.compile(model), val_data)
    # with torch.no_grad():
    #     t2 = benchmark_torch_function_in_microseconds(torch.compile(model), val_data)

    for i in range(nlayers):
        model.layers[i].linear1 = protoquant.qlinear_from_linear(
            model.layers[i].linear1
        )
        model.layers[i].linear2 = protoquant.qlinear_from_linear(
            model.layers[i].linear2
        )

    t3 = benchmark_torch_function_in_microseconds(model, val_data)

    with torch.no_grad():
        t4 = benchmark_torch_function_in_microseconds(model, val_data)

    print("val_data: ", val_data.size())
    print(f"Baseline: {t0}")
    # print(f"Compiled: {t1}")
    # print(f"Compiled + no_grad: {t2}")
    print(f"Quantized: {t3}")
    print(f"Quantized + no_grad: {t4}")


if __name__ == "__main__":
    main()
