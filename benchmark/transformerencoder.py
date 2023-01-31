import copy
import math
import platform
import time
from typing import Optional, Tuple

import torch
from torch.backends.cuda import sdp_kernel, SDPBackend
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import logging
from torch._dynamo import config

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
    SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
    SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
}


def data_process(raw_text_iter: dataset.IterableDataset, vocab, tokenizer) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [
        torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter
    ]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def toy_data(num_tokens, d_model, dtype):
    return torch.rand((num_tokens, d_model), device=torch.device('cuda'), dtype=dtype)


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
    dtype = torch.bfloat16
    # dtype = torch.half
    # dtype = torch.float32

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_data = toy_data(1000, d_model, dtype)
    val_data = toy_data(1000, d_model, dtype)
    test_data = toy_data(1000, d_model, dtype)

    batch_size = 32
    eval_batch_size = 32
    d_hid = 1024  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 16  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 16  # number of heads in nn.MultiheadAttention
    dropout = 0  # 0..2  # dropout probability
    # model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
    model = TransformerEncoder(encoder_layers, nlayers)
    model = model.to(device)
    train_seq_len = 1024
    eval_seq_len = 256
    epochs = 8

    # Select which kernel to use
    # kernel = SDPBackend.MATH
    # kernel = SDPBackend.FLASH_ATTENTION
    kernel = SDPBackend.EFFICIENT_ATTENTION

    profile_path = f"/scratch/drisspg/work/scripts/data/profiles/xlmr_train_{kernel.name}_batch_size_"\
    f"{batch_size}_d_hid{d_hid}_nlayers{nlayers}_nhead{nhead}_seq_len_{train_seq_len}.json"
    # profile_path=None

    lr = 3  # learning rate


    val_data = batchify(val_data, eval_batch_size, device)

    print("val_data.size(): ", train_data.size())

    model = model.to(dtype)

    # Pick which dtype to use
    # model = torch.compile(model, fullgraph=True)
    model = model

    if profile_path is not None:
        print(f"Saving profile to:{profile_path}")
    print(f"Using SDP backed by {kernel.name} ")
    print(f"Eval time: {benchmark_torch_function_in_microseconds(model, val_data)}")


if __name__ == "__main__":
    main()
