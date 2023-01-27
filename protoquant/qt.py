import torch

from .gemm import gemm
from .quantization import dqntz, qntz


# Implementation
# https://www.internalfb.com/code/fbsource/[1a00a08b25b91f7f114ee7bcfbc1eb462001e922]/fbcode/protoquant/quantization/quantization.cc?lines=38


class QTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        wrapped_data,
        is_quantized=False,
        qntzd=None,
        params=None,
        custom_dtype=None,
        custom_shape=None,
    ):
        kwargs = {}
        kwargs["device"] = qntzd.device if wrapped_data is None else wrapped_data.device
        kwargs["dtype"] = wrapped_data.dtype if custom_dtype is None else custom_dtype
        kwargs["layout"] = qntzd.layout if wrapped_data is None else wrapped_data.layout
        kwargs["requires_grad"] = (
            qntzd.requires_grad if wrapped_data is None else wrapped_data.requires_grad
        )
        return torch.Tensor._make_wrapper_subclass(
            cls, wrapped_data.shape if custom_shape is None else custom_shape, **kwargs
        )

    def __init__(
        self,
        wrapped_data,
        is_quantized=False,
        qntzd=None,
        params=None,
        custom_dtype=None,
        custom_shape=None,
    ):
        # NOTE: .data is a protected member. So we have to use wrapped_data.
        # This is a regular torch.Tensor if the input is not quantized, but could
        # be either
        self.wrapped_data = wrapped_data
        # NOTE: is_quantized is already defined, so we use wrapped_data_is_quantized.
        self.wrapped_data_is_quantized = is_quantized
        self.wrapped_qntzd = qntzd
        self.wrapped_params = params
        self.custom_dtype = custom_dtype
        self.custom_shape = custom_shape

    def __repr__(self):
        return f"QTensor(shape={self.wrapped_data.shape}, data={self.wrapped_data})"

    def is_q(self):
        # Note this implementation never persists quantized data for computation.
        # We do support quantized storage format for quantized weights.
        # it just use QTs to indicate matmul style float ops should be done in quantized domain
        return self.wrapped_data_is_quantized

    def force_quantize(self, is_a=True):
        qntzd, params = qntz(self.wrapped_data.detach().t().contiguous(), is_a=is_a)
        self.wrapped_data_is_quantized = True
        self.wrapped_qntzd = qntzd
        self.wrapped_params = params
        del self.wrapped_data
        self.wrapped_data = None
        return self

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # two scenarios where we currently fall back to vanilla mm:
        # 1 - when tensor is on CPU: we are missing qmm for CPU, but we should have a CPU implementation
        #     for consistency and to allow people to test
        # 2 - we need to define what happens when we're given non-floats - quantizing long to int8 is probs craxy
        if (
            func is torch.ops.aten.mm.default
            and args[0].is_floating_point()
            and args[0].is_cuda
        ):
            # This is where the actual efficient kernels will be called.
            # A given argument might see itself quantized before being passed into
            # an efficient kernel.
            # Assuming the operation supports this (e.g. via qmm) the result will also
            # be a quantized decomposition and converted back to a floating type
            # (add laxzy de-quantize later)

            # FIXME: wer have to handle where the wrong dimension is pre-quantized!
            inp = args[0]
            weight = args[1]

            qinp, iparams = (
                (
                    qntz(inp.wrapped_data, is_a=True)
                    if not inp.is_q()
                    else (inp.wrapped_qntzd, inp.wrapped_params)
                )
                if isinstance(inp, QTensor)
                else qntz(inp, is_a=True)
            )

            qweight, wparams = (
                (
                    qntz(weight.wrapped_data, is_a=False)
                    if not weight.is_q()
                    else (weight.wrapped_qntzd, weight.wrapped_params)
                )
                if isinstance(weight, QTensor)
                else qntz(weight, is_a=False)
            )

            # Pass the integer matrices to matrix multiple, the dequant args to dequantize
            # aways return dequantized results for now
            qntzd = gemm(qinp, qweight)

            out = dqntz(qntzd, iparams, wparams)
            return out

        if (
            func is torch.ops.aten.addmm.default
            and args[0].is_floating_point()
            and args[0].is_cuda
        ):
            # FIXME: wer have to handle where the wrong dimension is pre-quantized!
            bias = args[0]
            inp = args[1]
            weight = args[2]

            qinp, iparams = (
                (
                    qntz(inp.wrapped_data, is_a=True)
                    if not inp.is_q()
                    else (inp.wrapped_qntzd, inp.wrapped_params)
                )
                if isinstance(inp, QTensor)
                else qntz(inp, is_a=True)
            )

            qweight, wparams = (
                (
                    qntz(weight.wrapped_data, is_a=False)
                    if not weight.is_q()
                    else (weight.wrapped_qntzd, weight.wrapped_params)
                )
                if isinstance(weight, QTensor)
                else qntz(weight, is_a=False)
            )

            d = gemm(qinp, qweight)
            ret = dqntz(d, iparams, wparams, bias)
            return ret

        if func is torch.ops.aten.detach.default:
            if args[0].is_q():
                # print("000 args[0].shape: ", args[0].shape, " args[0].wrapped_qntzd.size(): ", args[0].wrapped_qntzd.size())
                return QTensor(
                    args[0].wrapped_data,
                    args[0].wrapped_data_is_quantized,
                    args[0].wrapped_qntzd,
                    args[0].wrapped_params,
                    args[0].dtype,
                    args[0].shape,
                )
            return QTensor(args[0].wrapped_data.detach(), args[0].is_q())

        if func is torch.ops.aten.t.default:
            if args[0].is_q() and args[0].wrapped_params.transpose:
                new_shape = (args[0].shape[1], args[0].shape[0])
                ret = QTensor(
                    args[0].wrapped_data,
                    args[0].wrapped_data_is_quantized,
                    args[0].wrapped_qntzd,
                    args[0].wrapped_params,
                    args[0].dtype,
                    new_shape,
                )
                # print("111 new_shape: ", new_shape, " args[0].shape: ", args[0].shape, " args[0].wrapped_qntzd.size(): ", args[0].wrapped_qntzd.size(), " ret.shape: ", ret.shape)
                return ret
            return QTensor(args[0].wrapped_data.t(), args[0].is_q())

        return NotImplemented
