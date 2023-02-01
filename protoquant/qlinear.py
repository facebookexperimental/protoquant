import torch


class QLinear(torch.nn.Module):

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(inp, self.weight, self.bias)


def qlinear_from_linear(linear: torch.nn.Module) -> torch.nn.Module:
    assert isinstance(linear, torch.nn.Linear)
    qweight = protoquant.QTensor(linear.weight).force_quantize(is_a=False)
    return QLinear(qweight, linear.bias)
