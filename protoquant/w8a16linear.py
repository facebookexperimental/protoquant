import torch
from protoquant.quantization import dqntz, qntz
from protoquant.src.triton.matmul import matmul as matmul_int8
from torch.nn.parameter import Parameter
from typing import Callable, Optional, Tuple



class W8A16QLinear(torch.nn.Module):
    def __init__(self, qweight, qscales, bias):
        super(W8A16QLinear, self).__init__()
        assert isinstance(bias, Parameter)
        self.qweight = qweight
        self.qscales = qscales
        self.bias = bias
        self.in_features = qweight.size(1)
        self.out_features = qweight.size(1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        assert inp.dim() == 3
        assert inp.dtype == torch.float16
        return torch.nn.functional.linear(inp,
                self.qweight.mul(self.qscales).to(torch.float16))

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

half_range_lookup = {
    8: torch.full((1,), (1 << (8 - 1)) - 1, dtype=torch.float16, device="cuda"),
}
full_range_lookup = {
    8: torch.full((1,), 1 << 8, dtype=torch.float16, device="cuda"),
}

inv_half_range_lookup = {
    8: torch.full((1,), 1 / ((1 << (8 - 1)) - 1), dtype=torch.float16, device="cuda"),
}
inv_full_range_lookup = {
    8: torch.full((1,), 1 / (1 << 8), dtype=torch.float16, device="cuda"),
}

def scales_from_point(input: torch.Tensor, dim: Optional[int], qdtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    input_abs = torch.abs(input)
    if dim is None:
        input_abs_maxs = torch.max(input_abs)
    else:
        input_abs_maxs = torch.max(input_abs, dim, keepdim=True).values
    scales = torch.mul(input_abs_maxs, inv_half_range_lookup[torch.iinfo(qdtype).bits]).to(torch.float32)
    inv_scales = torch.div(half_range_lookup[torch.iinfo(qdtype).bits], input_abs_maxs).to(torch.float32)
    return scales, inv_scales

def per_channel_scaled(input: torch.Tensor, qdtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    scales, inv_scales = scales_from_point(input, 0, qdtype)
    qinput = torch.mul(input, inv_scales).to(qdtype)
    return qinput, scales

def per_token_scaled(input: torch.Tensor, qdtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    scales, inv_scales = scales_from_point(input, 1, qdtype)
    qinput = torch.mul(input, inv_scales).to(qdtype)
    return qinput, scales


def w8a16_qlinear_from_linear(
    linear: torch.nn.Module, minimize_error=True
) -> torch.nn.Module:
    import protoquant

    assert isinstance(linear, torch.nn.Linear)
    assert linear.weight.dtype == torch.float16
    assert linear.bias.dtype == torch.float16
    qweight, qscales = per_token_scaled(linear.weight, torch.int8)
    return W8A16QLinear(qweight, qscales, linear.bias)
