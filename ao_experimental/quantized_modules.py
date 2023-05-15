import torch
from quant_primitives import dynamically_quantize_per_channel, quant_int8_dynamic_linear


class DynamicallyQuantizedLinear(torch.nn.Module):
    r"""
    This function is similar to cpu-only torch.ao.nn.quantized.dynamic.modules.linear.Linear
    but is implemented in a way that can be triton traced to run gpu cuda.

    note1: in order for this to be triton compilable and runnable the in_channels, aka w_int8_t.shape[0]
    must be greater than 16

    note2: This is not a final API and may change without warning

    Attributes:
        w_int8_t (Tensor, int8): the integer representation of the per-channel symmetrically quantized and transposed weight tensor
        w_scales (Tensor, float64): the per_channel scales of the quantized weight tensor
        x_quant_min (int): the minimum quantized x integer
        x_quant_max (int): the maximum quantized x integer
        x_q_dtype (dtype): the desired integer type to quantize x to (only int8 currently supported)
        out_dtype (dtype): the dtype of the output

        w_int8_t_sums_int64 (Tensor, int8): a preprocessed tensor derived from w_int8_t needed for the matmul
        bias (Tensor, float32): a float tensor that gets added on to the final result

    Examples::

        >>> lin = torch.nn.Linear(32, 64).to('cuda')
        >>> qlin = DynamicallyQuantizedLinear.from_float(lin)
        >>> trit_qlin = torch.compile(qlin, mode='max-autotune')
        >>> out = trit_qlin(torch.randn([24, 32], device='cuda'))
    """

    def __init__(
        self,
        w_int8_t,
        w_scales,
        x_quant_min=-128,
        x_quant_max=127,
        x_q_dtype=torch.int8,
        out_dtype=torch.float32,
    ):
        super().__init__()
        self.register_buffer("w_int8_t", w_int8_t)
        self.register_buffer("w_int8_t_sums_int64", w_int8_t.sum(dim=0).to(torch.int64))
        self.register_buffer("w_scales", w_scales)
        self.register_buffer("bias", None)
        self.out_dtype = out_dtype
        self.x_quant_min = x_quant_min
        self.x_quant_max = x_quant_max
        self.x_q_dtype = x_q_dtype

    def forward(self, x):
        return quant_int8_dynamic_linear(
            x,
            self.x_quant_min,
            self.x_quant_max,
            self.x_q_dtype,
            self.w_int8_t,
            self.w_int8_t_sums_int64,
            self.w_scales,
            self.bias,
            self.out_dtype,
        )

    @classmethod
    def from_float(
        cls,
        mod,
        w_quant_min=-128,
        w_quant_max=127,
        w_q_dtype=torch.int8,
        w_axis=0,
        x_quant_min=-128,
        x_quant_max=127,
        x_q_dtype=torch.int8,
        out_dtype=torch.float32,
    ):
        assert isinstance(
            mod, torch.nn.Linear
        ), f"need mod to be type torch.nn.Linear but got {type(mod)}"
        assert (
            w_axis == 0
        ), f"only weight per-channel quantization axis of 0 currently supported but got {w_axis}"
        w_int8, w_scales, _ = dynamically_quantize_per_channel(
            mod.weight,
            quant_min=w_quant_min,
            quant_max=w_quant_max,
            target_dtype=w_q_dtype,
            axis=w_axis,
        )
        new_qlinear = cls(
            w_int8.transpose(0, 1),
            w_scales,
            x_quant_min,
            x_quant_max,
            x_q_dtype,
            out_dtype,
        )
        if mod.bias is not None:
            new_qlinear.bias = mod.bias
        return new_qlinear
