import torch
from torch.nn.parameter import Parameter
from protoquant.quantization import dqntz, qntz
from protoquant.src.triton.matmul import _matmul_call
from protoquant.src.triton.dequant import dequant_kernel
from protoquant.src.triton.quant import quant_kernel


class QLinear(torch.nn.Module):

    def __init__(self, qweight, wparams, bias):
        super(QLinear, self).__init__()
        assert isinstance(bias, Parameter)
        self.qweight = qweight
        self.wparams = wparams
        self.bias = bias
        self.in_features = self.qweight.size(1)
        self.out_features = self.qweight.size(1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        assert inp.dim() == 3
        inp_size0 = inp.size(0)
        inp_size1 = inp.size(1)
        inp_size2 = inp.size(2)
        inp = inp.reshape(inp_size0 * inp_size1, inp_size2)

        # qinp, iparams = qntz(inp, is_a=True)

        # row wise and without transpose
        _, _, scales, zeros, sums, qinp = quant_kernel(inp)  # dim: 1

        bias = self.bias
        qweight = self.qweight
        wparams = self.wparams

        # d = torch.ops.protoquant._triton_gemm(qinp, qweight.t())
        d = torch.ops.aten._int_mm(qinp, qweight.t())
        # d = torch.mm(qinp.float(), qweight.t().float()).int()
        # d = _matmul_call(qinp, qweight.t())
        # return dqntz(d, iparams, wparams, bias).view(inp_size0, inp_size1, -1)
        d_size0 = d.size(0)
        d_size1 = d.size(1)
        return dequant_kernel(
            d,
            bias,
            scales.view(d_size0, 1),
            zeros.view(d_size0, 1),
            sums.view(d_size0, 1),
            wparams.scales.view(1, d_size1),
            wparams.zeros.view(1, d_size1),
            wparams.sums.view(1, d_size1),
        )[0].view(inp_size0, inp_size1, -1)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def qlinear_from_linear(linear: torch.nn.Module) -> torch.nn.Module:
    import protoquant
    assert isinstance(linear, torch.nn.Linear)
    qw = protoquant.QTensor(linear.weight).force_quantize(is_a=False)
    qweight, wparams = qw.wrapped_qntzd, qw.wrapped_params
    assert linear.weight.dtype == torch.float16
    assert linear.bias.dtype == torch.float16
    return QLinear(qweight, wparams, linear.bias)
