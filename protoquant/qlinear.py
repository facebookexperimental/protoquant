import torch
from protoquant.quantization import dqntz, qntz
from protoquant.src.triton.matmul import matmul as matmul_int8
from torch.nn.parameter import Parameter


class QLinear(torch.nn.Module):
    def __init__(self, qweight, wparams, bias, minimize_error, use_int_mm):
        super(QLinear, self).__init__()
        assert isinstance(bias, Parameter)
        # Need to store in transposed form due to cuBLAS
        self.qweight = qweight
        self.wparams = wparams
        self.bias = bias
        self.in_features = qweight.size(1)
        self.out_features = qweight.size(1)
        self.minimize_error = minimize_error
        self.use_int_mm = use_int_mm

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        assert inp.dim() == 3
        inp_size0 = inp.size(0)
        inp_size1 = inp.size(1)
        inp_size2 = inp.size(2)
        inp = inp.reshape(inp_size0 * inp_size1, inp_size2)
        qinp, iparams = qntz(
            inp, is_a=True, minimize_error=self.minimize_error
        )
        if self.use_int_mm:
            d = torch.ops.aten._int_mm(qinp, self.qweight.t())
        else:
            d = (qinp.unsqueeze(-1).to(torch.int32) * self.qweight.t().to(torch.int32)).sum(1).to(torch.int32)
        return dqntz(d, iparams, self.wparams, self.bias).view(
            inp_size0, inp_size1, -1
        )

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


def qlinear_from_linear(
    linear: torch.nn.Module, minimize_error=True, use_int_mm=True,
) -> torch.nn.Module:
    import protoquant

    assert isinstance(linear, torch.nn.Linear)
    qw = protoquant.QTensor(linear.weight).force_quantize(is_a=False)
    qweight, wparams = qw.wrapped_qntzd, qw.wrapped_params
    assert linear.weight.dtype == torch.float16
    assert linear.bias.dtype == torch.float16
    return QLinear(qweight, wparams, linear.bias, minimize_error=minimize_error, use_int_mm=use_int_mm)
