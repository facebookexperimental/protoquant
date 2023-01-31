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
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
import logging
from torch._dynamo import config

torch.set_float32_matmul_precision("high")
# config.log_level = logging.DEBUG
# config.verbose = True


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, src: Tensor, src_mask: Optional[Tensor] = None, causal_mask: bool = False
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def train_step(model, data, targets, optimizer, criterion, ntokens):
    # Zero_grads
    optimizer.zero_grad()
    output = model(data, causal_mask=True)
    loss = criterion(output.view(-1, ntokens), targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return loss


def train(
    model: nn.Module,
    train_data,
    train_seq_len,
    device,
    ntokens,
    criterion,
    optimizer,
    scheduler,
    epoch,
    profile_path=None,
) -> None:
    model.train()  # turn on train mode
    total_loss = 0.0
    log_interval = 15
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(train_seq_len).to(device)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    num_batches = len(train_data) // train_seq_len
    for batch, i in enumerate(range(0, train_data.size(0) - 1, train_seq_len)):
        data, targets = get_batch(train_data, i, train_seq_len)
        seq_len = data.size(0)
        if seq_len != train_seq_len:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]

        if profile_path is not None and batch == 4:
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_stack=True,
            ) as profiler:
                loss = train_step(model, data, targets, optimizer, criterion, ntokens)
            profiler.export_chrome_trace(profile_path)
        else:
            loss = train_step(model, data, targets, optimizer, criterion, ntokens)

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(
                f"| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | "
                f"lr {lr:02.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}"
            )
            total_loss = 0
            start_time = time.time()


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


def batchify(data: Tensor, bsz: int, device: str) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[: seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
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

    train_iter = WikiText2(split="train")
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter, vocab, tokenizer)
    val_data = data_process(val_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)

    batch_size = 32
    eval_batch_size = 32
    ntokens = len(vocab)  # size of vocabulary
    emsize = 1024  # embedding dimension
    d_hid = 1024  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 16  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 16  # number of heads in nn.MultiheadAttention
    dropout = 0  # 0..2  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
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

    # SET MODEL DTYPE
    dtype = torch.bfloat16
    # dtype = torch.half
    # dtype = torch.float32

    train_data = batchify(train_data, batch_size, device)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size, device)
    test_data = batchify(test_data, eval_batch_size, device)

    model = model.to(dtype)

    # Pick which dtype to use
    # model = torch.compile(model, fullgraph=True)
    model = model

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, foreach=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 300, gamma=0.1)

    if profile_path is not None:
        print(f"Saving profile to:{profile_path}")
    # profile_path=None
    print(f"Using SDP backed by {kernel.name} ")
    print(f"Training a model of dtype {dtype}")
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        with torch.backends.cuda.sdp_kernel(**backend_map[kernel]):
            train(
                model,
                train_data,
                train_seq_len,
                device,
                ntokens,
                criterion,
                optimizer,
                scheduler,
                epoch,
                profile_path,
            )
            val_loss = evaluate(
                model, val_data, eval_seq_len, device, ntokens, criterion
            )
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"val loss {val_loss:5.2f} | val ppl {val_ppl:8.2f}"
        )
        print("-" * 89)

        scheduler.step()

    print("We are now evalutating the model")
    test_loss = evaluate(model, test_data, train_seq_len, device, ntokens, criterion)
    test_ppl = math.exp(test_loss)
    print("=" * 89)
    print(
        f"| End of training | test loss {test_loss:5.2f} | " f"test ppl {test_ppl:8.2f}"
    )
    print("=" * 89)


if __name__ == "__main__":
    main()
