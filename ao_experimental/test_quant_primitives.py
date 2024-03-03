import copy
import unittest
from itertools import cycle as cycle

import torch
import torch.nn as nn
from quant_primitives import (
    dequantize_per_channel,
    dequantize_per_tensor,
    dynamically_quantize_per_channel,
    dynamically_quantize_per_tensor,
    quant_int8_dynamic_linear,
    quant_int8_matmul,
    safe_int_mm,
)
from quantized_modules import DynamicallyQuantizedLinear

torch.manual_seed(0)


def SQNR(x, y):
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


class TwoLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(24, 32).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(32, 64).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.subm = TwoLayerLinearModel()
        self.fc = nn.Linear(64, 32)

    def forward(self, x):
        x = self.subm(x)
        x = self.fc(x)
        return x

    def get_example_input(self):
        return torch.randn(2, 5, 24)


class EndToEndTest(unittest.TestCase):
    r"""
    Tests that end to end performance of DynamicallyQuantizedLinear is good when used in a toy model
    """

    def _test_end_to_end_impl(self, device):
        model = TestModel().to(device).eval()
        q_model = copy.deepcopy(model)

        def convert_modules(model, targets, convert_function):
            for name, module in model.named_children():
                if type(module) in targets:
                    new_mod = convert_function(module)
                    setattr(model, name, new_mod)
                else:
                    convert_modules(module, targets, convert_function)

        convert_modules(
            q_model, [torch.nn.Linear], DynamicallyQuantizedLinear.from_float
        )
        if device == "cuda":
            trit_model = torch.compile(q_model, mode="max-autotune")
        for _ in range(5):
            x = model.get_example_input().to(device)
            y_ref = model(x)
            y = q_model(x)
            self.assertGreater(SQNR(y_ref, y), 35)
            if device == "cuda":
                trit_model.eval()
                y_trit = trit_model(x)
                self.assertGreater(SQNR(y_ref, y_trit), 35)

    def test_end_to_end_cuda(self):
        self._test_end_to_end_impl("cuda")

    def test_end_to_end_cpu(self):
        self._test_end_to_end_impl("cpu")


class TestDequantizePerChannel(unittest.TestCase):
    """
    Tests the dequantize_per_tensor function across a variety of input cases and ensures numerics match ao version
    """

    shapes = (
        (5, 5),
        (2, 8, 32, 32),
        (32, 16, 64, 64),
        (1, 200, 200),
    )

    def _test_dequantize_per_channel_impl(
        self, device, quant_min=-128, quant_max=127, target_dtype=torch.int8
    ):
        out_dtypes = cycle([torch.float16, torch.float32, torch.float64])
        for x_shape in self.shapes:
            for axis in range(len(x_shape)):
                out_dtype = next(out_dtypes)
                x = torch.randn(x_shape, device=device) * 1000

                _, scales, zero_points = dynamically_quantize_per_channel(
                    x, quant_min, quant_max, target_dtype, axis
                )

                q_dtype = torch.quint8 if target_dtype == torch.uint8 else torch.qint8
                ref_q = torch.quantize_per_channel(
                    x, scales, zero_points, axis, q_dtype
                )
                ref_int = ref_q.int_repr()

                self.assertEqual(ref_int.dtype, target_dtype)

                x_dq = dequantize_per_channel(
                    ref_int, scales, zero_points, out_dtype, axis
                )
                ref_dq = ref_q.dequantize().to(out_dtype)
                self.assertEqual(x_dq.dtype, out_dtype)
                torch.testing.assert_close(x_dq, ref_dq)

                if device == "cuda":
                    trit_dequantize_per_channel = torch.compile(
                        dequantize_per_channel, mode="max-autotune"
                    )
                    trit_dq = trit_dequantize_per_channel(
                        ref_int, scales, zero_points, out_dtype, axis
                    )
                    self.assertEqual(trit_dq.dtype, out_dtype)
                    torch.testing.assert_close(trit_dq, ref_dq)

    def test_dequantize_per_channel_cuda_int8(self):
        self._test_dequantize_per_channel_impl(
            device="cuda", quant_min=-128, quant_max=127, target_dtype=torch.int8
        )

    def test_dequantize_per_channel_cuda_uint8(self):
        self._test_dequantize_per_channel_impl(
            device="cuda", quant_min=0, quant_max=255, target_dtype=torch.uint8
        )

    def test_dequantize_per_channel_cpu_int8(self):
        self._test_dequantize_per_channel_impl(
            device="cpu", quant_min=-128, quant_max=127, target_dtype=torch.int8
        )

    def test_dequantize_per_channel_cpu_uint8(self):
        self._test_dequantize_per_channel_impl(
            device="cpu", quant_min=0, quant_max=255, target_dtype=torch.uint8
        )


class TestDequantizePerTensor(unittest.TestCase):
    """
    Tests the dequantize_per_tensor function across a variety of input cases and ensures numerics match ao version
    """

    shapes = (
        (5, 5),
        (2, 8, 32, 32),
        (32, 16, 64, 64),
        (1, 200, 200),
    )

    def _test_dequantize_per_tensor_impl(
        self, device, quant_min=-128, quant_max=127, target_dtype=torch.int8
    ):
        out_dtypes = [torch.float16, torch.float32, torch.float64]
        for x_shape in self.shapes:
            for out_dtype in out_dtypes:
                x = torch.randn(x_shape, device=device) * 1000

                _, scale, zero_point = dynamically_quantize_per_tensor(
                    x, quant_min, quant_max, target_dtype
                )

                q_dtype = torch.quint8 if target_dtype == torch.uint8 else torch.qint8
                ref_q = torch.quantize_per_tensor_dynamic(
                    x, q_dtype, reduce_range=False
                )
                ref_int = ref_q.int_repr()

                self.assertEqual(ref_int.dtype, target_dtype)

                x_dq = dequantize_per_tensor(
                    ref_int, ref_q.q_scale(), ref_q.q_zero_point(), out_dtype
                )  # scalar args
                x_dq2 = dequantize_per_tensor(
                    ref_int, scale, zero_point, out_dtype
                )  # tensor args
                ref_dq = ref_q.dequantize().to(out_dtype)
                self.assertEqual(x_dq.dtype, out_dtype)
                self.assertEqual(x_dq2.dtype, out_dtype)
                torch.testing.assert_close(x_dq, ref_dq)
                torch.testing.assert_close(x_dq2, ref_dq)

                if device == "cuda":
                    trit_dequantize_per_tensor = torch.compile(
                        dequantize_per_tensor, mode="max-autotune"
                    )
                    trit_dq = trit_dequantize_per_tensor(
                        ref_int, scale, zero_point, out_dtype
                    )  # tensor args
                    self.assertEqual(trit_dq.dtype, out_dtype)
                    torch.testing.assert_close(trit_dq, ref_dq)

    def test_dequantize_per_tensor_cuda_int8(self):
        self._test_dequantize_per_tensor_impl(
            device="cuda", quant_min=-128, quant_max=127, target_dtype=torch.int8
        )

    def test_dequantize_per_tensor_cuda_uint8(self):
        self._test_dequantize_per_tensor_impl(
            device="cuda", quant_min=0, quant_max=255, target_dtype=torch.uint8
        )

    def test_dequantize_per_tensor_cpu_int8(self):
        self._test_dequantize_per_tensor_impl(
            device="cpu", quant_min=-128, quant_max=127, target_dtype=torch.int8
        )

    def test_dequantize_per_tensor_cpu_uint8(self):
        self._test_dequantize_per_tensor_impl(
            device="cpu", quant_min=0, quant_max=255, target_dtype=torch.uint8
        )


class TestPerChannelQuantization(unittest.TestCase):
    r"""
    Tests the dynamically_quantize_per_channel function across a variety of input cases and ensures numerics match ao version
    """

    shapes = (
        (1, 200, 200),
        (5, 5),
        (2, 8, 32, 32),
        (32, 16, 64, 64),
    )

    def _test_dynamically_quantize_per_channel_impl(
        self, device, quant_min=-128, quant_max=127, target_dtype=torch.int8
    ):
        f_dtypes = cycle([torch.float16, torch.float32, torch.float64])
        transposes = cycle([True, False])
        for x_shape in self.shapes:
            for axis in range(len(x_shape)):
                transp = next(transposes)
                f_dtype = next(f_dtypes)
                x = torch.randn(x_shape, device=device, dtype=f_dtype) * 1000
                if transp:
                    x = x.transpose(0, -1)

                x_int8, scales, zero_points = dynamically_quantize_per_channel(
                    x, quant_min, quant_max, target_dtype, axis=axis
                )

                q_dtype = torch.quint8 if target_dtype == torch.uint8 else torch.qint8

                obs = torch.ao.quantization.PerChannelMinMaxObserver(
                    ch_axis=axis,
                    dtype=q_dtype,
                    qscheme=torch.per_channel_symmetric,
                    reduce_range=False,
                )
                obs(x)
                ref_scales, ref_zero_points = obs.calculate_qparams()
                ref_scales, ref_zero_points = ref_scales.to(
                    x.device
                ), ref_zero_points.to(x.device)
                torch.testing.assert_close(scales.to(torch.float32), ref_scales)
                torch.testing.assert_close(zero_points, ref_zero_points, atol=0, rtol=0)

                x_q_int_repr = torch.quantize_per_channel(
                    x.to(torch.float32),
                    ref_scales,
                    ref_zero_points,
                    axis=axis,
                    dtype=q_dtype,
                ).int_repr()
                torch.testing.assert_close(x_int8, x_q_int_repr, atol=1, rtol=100)

                if device == "cuda":
                    trit_dynamic_quant = torch.compile(
                        dynamically_quantize_per_channel, mode="max-autotune"
                    )
                    trit_x_int8, trit_scales, trit_zps = trit_dynamic_quant(
                        x, quant_min, quant_max, target_dtype, axis=axis
                    )
                    torch.testing.assert_close(
                        trit_scales.to(torch.float32), ref_scales
                    )
                    torch.testing.assert_close(
                        trit_zps, ref_zero_points, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        trit_x_int8, x_q_int_repr, atol=1, rtol=100
                    )

    def test_dynamically_quantize_per_channel_cuda_int8(self):
        self._test_dynamically_quantize_per_channel_impl(
            device="cuda", quant_min=-128, quant_max=127, target_dtype=torch.int8
        )

    def test_dynamically_quantize_per_channel_cuda_uint8(self):
        self._test_dynamically_quantize_per_channel_impl(
            device="cuda", quant_min=0, quant_max=255, target_dtype=torch.uint8
        )

    def test_dynamically_quantize_per_channel_cpu_int8(self):
        self._test_dynamically_quantize_per_channel_impl(
            device="cpu", quant_min=-128, quant_max=127, target_dtype=torch.int8
        )

    def test_dynamically_quantize_per_channel_cpu_uint8(self):
        self._test_dynamically_quantize_per_channel_impl(
            device="cpu", quant_min=0, quant_max=255, target_dtype=torch.uint8
        )


class TestPerTensorQuantization(unittest.TestCase):
    r"""
    Tests the dynamically_quantize_per_tensor function across a variety of input cases and ensures numerics match ao version
    """

    shapes = (
        (1, 1, 32, 32),
        (32, 16, 64, 64),
        (100, 100),
        (1, 200, 200),
    )

    def _test_dynamically_quantize_per_tensor_impl(
        self, device, quant_min=-128, quant_max=127, target_dtype=torch.int8, tol=0
    ):
        f_dtypes = cycle([torch.float16, torch.float32, torch.float64])
        for x_shape in self.shapes:
            for transp in [False, True]:
                f_dtype = next(f_dtypes)
                x = torch.randn(x_shape, device=device, dtype=f_dtype) * 1000
                if transp:
                    x = x.transpose(0, -1)

                x_int8, scale, zero_point = dynamically_quantize_per_tensor(
                    x, quant_min, quant_max, target_dtype
                )

                q_dtype = torch.quint8 if target_dtype == torch.uint8 else torch.qint8
                x_q = torch.quantize_per_tensor_dynamic(
                    x.to(torch.float32), dtype=q_dtype, reduce_range=False
                )

                torch.testing.assert_close(scale, x_q.q_scale())
                torch.testing.assert_close(
                    zero_point, x_q.q_zero_point(), atol=0, rtol=0
                )
                torch.testing.assert_close(
                    x_int8.to(torch.int32),
                    x_q.int_repr().to(torch.int32),
                    atol=tol,
                    rtol=100,
                )

                if device == "cuda":
                    trit_dynamic_quant = torch.compile(
                        dynamically_quantize_per_tensor, mode="max-autotune"
                    )
                    trit_x_int8, trit_scale, trit_zp = trit_dynamic_quant(
                        x, quant_min, quant_max, target_dtype
                    )
                    torch.testing.assert_close(trit_scale, x_q.q_scale())
                    torch.testing.assert_close(
                        trit_zp, x_q.q_zero_point(), atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        trit_x_int8.to(torch.int32),
                        x_q.int_repr().to(torch.int32),
                        atol=tol,
                        rtol=100,
                    )

    def test_dynamically_quantize_per_tensor_cuda_int8(self):
        self._test_dynamically_quantize_per_tensor_impl(
            device="cuda", quant_min=-128, quant_max=127, target_dtype=torch.int8
        )

    def test_dynamically_quantize_per_tensor_cuda_uint8(self):
        self._test_dynamically_quantize_per_tensor_impl(
            device="cuda", quant_min=0, quant_max=255, target_dtype=torch.uint8
        )

    # CPU quantization has slightly different numerics than cuda, we chose to match cuda and
    # have all int values be within 1 of cpu
    def test_dynamically_quantize_per_tensor_cpu_int8(self):
        self._test_dynamically_quantize_per_tensor_impl(
            device="cpu", quant_min=-128, quant_max=127, target_dtype=torch.int8, tol=1
        )

    def test_dynamically_quantize_per_tensor_cpu_uint8(self):
        self._test_dynamically_quantize_per_tensor_impl(
            device="cpu", quant_min=0, quant_max=255, target_dtype=torch.uint8, tol=1
        )


class TestQuantInt8MatMul(unittest.TestCase):
    r"""
    Tests that quant_int8_matmul has good numerical accuracy
    """

    shapes = (
        # ((x_shape), (w_shape))
        ((3, 2, 32, 32), (20, 32)),
        ((32, 16, 64, 64), (16, 64)),
        ((100, 100), (2, 100)),
        ((3, 200, 200), (100, 200)),
    )

    def _test_quant_int8_matmul_impl(self, device):
        out_dtypes = cycle([torch.float16, torch.float32, torch.float64])
        for x_shape, w_shape in self.shapes:
            for contiguous_x in [True, False]:
                for contiguous_wt in [True, False]:
                    out_dtype = next(out_dtypes)

                    x = torch.randn(x_shape, device=device)
                    if not contiguous_x:
                        x = x.transpose(0, 1)
                        assert not x.is_contiguous()
                    else:
                        assert x.is_contiguous()

                    x_vals_int8, x_scale, x_zp = dynamically_quantize_per_tensor(x)

                    lin = torch.nn.Linear(w_shape[1], w_shape[0], False).to(device)
                    w = lin.weight
                    w_int8, w_scales, _ = dynamically_quantize_per_channel(w)
                    w_int8_t = w_int8.transpose(0, 1)
                    if contiguous_wt:
                        w_int8_t = w_int8_t.contiguous()
                        assert w_int8_t.is_contiguous()
                    else:
                        assert not w_int8_t.is_contiguous()
                    w_int8_t_sums_int64 = w_int8_t.sum(dim=0)

                    y = quant_int8_matmul(
                        x_vals_int8,
                        x_scale,
                        x_zp,
                        w_int8_t,
                        w_int8_t_sums_int64,
                        w_scales,
                        out_dtype,
                    )
                    y_ref = lin(x)
                    self.assertGreater(SQNR(y_ref, y), 37)

                    if device == "cuda":
                        trit_fn = torch.compile(quant_int8_matmul, mode="max-autotune")
                        y_triton = trit_fn(
                            x_vals_int8,
                            x_scale,
                            x_zp,
                            w_int8_t,
                            w_int8_t_sums_int64,
                            w_scales,
                            out_dtype,
                        )
                        self.assertGreater(SQNR(y_ref, y_triton), 37)

    def test_quant_int8_matmul_cuda(self):
        self._test_quant_int8_matmul_impl(device="cuda")

    def test_quant_int8_matmul_cpu(self):
        self._test_quant_int8_matmul_impl(device="cpu")


class TestQuantInt8DynamicLinearOp(unittest.TestCase):
    r"""
    Tests that quant_int8_dynamic_linear has good numerical accuracy
    """

    shapes = (
        # ((x_shape), (w_shape))
        ((3, 2, 32, 32), (20, 32)),
        ((32, 16, 64, 64), (16, 64)),
        ((100, 100), (2, 100)),
        ((3, 200, 200), (100, 200)),
    )

    def _test_quant_int8_dynamic_linear_impl(self, device):
        out_dtypes = cycle([torch.float16, torch.float32, torch.float64])
        biases = cycle([True, False])
        for x_shape, w_shape in self.shapes:
            use_bias = next(biases)
            for contiguous_x in [True, False]:
                for contiguous_wt in [True, False]:
                    out_dtype = next(out_dtypes)

                    x = torch.randn(x_shape, device=device)
                    if not contiguous_x:
                        x = x.transpose(0, 1)
                        assert not x.is_contiguous()
                    else:
                        assert x.is_contiguous()
                    lin = torch.nn.Linear(w_shape[1], w_shape[0], bias=use_bias).to(
                        device
                    )
                    bias = lin.bias
                    w = lin.weight
                    w_int8, w_scales, _ = dynamically_quantize_per_channel(w)
                    w_int8_t = w_int8.transpose(0, 1)
                    if contiguous_wt:
                        w_int8_t = w_int8_t.contiguous()
                        assert w_int8_t.is_contiguous()
                    else:
                        assert not w_int8_t.is_contiguous()
                    w_int8_t_sums_int64 = w_int8_t.sum(dim=0)

                    y = quant_int8_dynamic_linear(
                        x,
                        -128,
                        127,
                        torch.int8,
                        w_int8_t,
                        w_int8_t_sums_int64,
                        w_scales,
                        bias,
                        out_dtype,
                    )
                    y_ref = lin(x)
                    self.assertGreater(SQNR(y_ref, y), 37)

                    if device == "cuda":
                        trit_fn = torch.compile(
                            quant_int8_dynamic_linear, mode="max-autotune"
                        )
                        y_triton = trit_fn(
                            x,
                            -128,
                            127,
                            torch.int8,
                            w_int8_t,
                            w_int8_t_sums_int64,
                            w_scales,
                            bias,
                            out_dtype,
                        )
                        self.assertGreater(SQNR(y_ref, y_triton), 37)

    def test_quant_int8_dynamic_linear_cuda(self):
        self._test_quant_int8_dynamic_linear_impl(device="cuda")

    def test_quant_int8_dynamic_linear_cpu(self):
        self._test_quant_int8_dynamic_linear_impl(device="cpu")


class TestDynamicallyQuantizedLinear(unittest.TestCase):
    r"""
    Tests that DynamicallyQuantizedlinear has good numerical accuracy
    """

    shapes = (
        # ((x_shape), (w_shape))
        ((3, 2, 32, 32), (20, 32)),
        ((32, 16, 64, 64), (16, 64)),
        ((100, 100), (2, 100)),
        ((3, 200, 200), (100, 200)),
    )

    def _test_dynamically_quantized_linear_impl(self, device):
        out_dtypes = cycle([torch.float16, torch.float32, torch.float64])
        biases = cycle([True, False])
        for x_shape, w_shape in self.shapes:
            use_bias = next(biases)
            for contiguous_x in [True, False]:
                for contiguous_wt in [True, False]:
                    out_dtype = next(out_dtypes)

                    x = torch.randn(x_shape, device=device)
                    if not contiguous_x:
                        x = x.transpose(0, 1)
                        assert not x.is_contiguous()
                    else:
                        assert x.is_contiguous()
                    lin = torch.nn.Linear(w_shape[1], w_shape[0], bias=use_bias).to(
                        device
                    )
                    qlin = DynamicallyQuantizedLinear.from_float(
                        lin, out_dtype=out_dtype
                    )

                    if contiguous_wt:
                        qlin.w_int8_t = qlin.w_int8_t.contiguous()
                        assert qlin.w_int8_t.is_contiguous()
                    else:
                        assert not qlin.w_int8_t.is_contiguous()

                    y = qlin(x)
                    y_ref = lin(x)
                    self.assertGreater(SQNR(y_ref, y), 37)

                    if device == "cuda":
                        trit_lin = torch.compile(qlin, mode="max-autotune")
                        y_triton = trit_lin(x)
                        self.assertGreater(SQNR(y_ref, y_triton), 37)

    def test_quant_int8_dynamic_linea_cuda(self):
        self._test_dynamically_quantized_linear_impl(device="cuda")

    def test_quant_int8_dynamic_linea_cpu(self):
        self._test_dynamically_quantized_linear_impl(device="cpu")


class TestSafeIntMM(unittest.TestCase):
    r"""
    Tests the safe_int_mm functionality/correctness across a variety of input cases
    """

    shapes = (
        # ((x_shape), (w_shape))
        ((8, 17), (17, 8)),  # break cublas but not triton (fallback)
        ((17, 24), (24, 8)),  # smallest test that doesn't need fallback
        ((1536, 1536), (1536, 1536)),
        ((17, 4096), (4096, 1536)),
        # ((17, 8), (8, 8)), # breaks triton but not cublas
        # note: this last isn't tested since triton path doesn't have the fallback option for perf reasons,
        # so the error is expected
    )

    def _test_safe_int_mm_impl(self, x, w):
        y = safe_int_mm(x, w)
        y_ref = torch.matmul(x.to(torch.int32).cpu(), w.to(torch.int32).cpu()).to(
            x.device
        )
        torch.testing.assert_close(
            y_ref,
            y,
            atol=0,
            rtol=0,
            msg=r"failed for shape {} and {}".format(x.shape, w.shape),
        )

        if x.device.type == "cuda" and w.device.type == "cuda":
            trit_safe_int_mm = torch.compile(safe_int_mm, mode="max-autotune")
            y_triton = trit_safe_int_mm(x, w)
            torch.testing.assert_close(
                y_ref,
                y_triton,
                atol=0,
                rtol=0,
                msg=r"failed for shape {} and {}".format(x.shape, w.shape),
            )

    def test_safe_int_mm_cuda(self):
        for x_shape, w_shape in self.shapes:
            x = torch.randint(-128, 127, x_shape, dtype=torch.int8, device="cuda")
            w = torch.randint(-128, 127, w_shape, dtype=torch.int8, device="cuda")
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cpu(self):
        for x_shape, w_shape in self.shapes:
            x = torch.randint(-128, 127, x_shape, dtype=torch.int8, device="cpu")
            w = torch.randint(-128, 127, w_shape, dtype=torch.int8, device="cpu")
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cuda_non_contiguous_w(self):
        for x_shape, w_shape in self.shapes:
            x = torch.randint(-128, 127, x_shape, dtype=torch.int8, device="cuda")
            w = torch.randint(
                -128, 127, w_shape[::-1], dtype=torch.int8, device="cuda"
            ).transpose(0, 1)
            assert not w.is_contiguous()
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cpu_non_contiguous_w(self):
        for x_shape, w_shape in self.shapes:
            x = torch.randint(-128, 127, x_shape, dtype=torch.int8, device="cpu")
            w = torch.randint(
                -128, 127, w_shape[::-1], dtype=torch.int8, device="cpu"
            ).transpose(0, 1)
            assert not w.is_contiguous()
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cuda_non_contiguous_x(self):
        for x_shape, w_shape in self.shapes:
            x = torch.randint(
                -128, 127, x_shape[::-1], dtype=torch.int8, device="cuda"
            ).transpose(0, 1)
            w = torch.randint(-128, 127, w_shape, dtype=torch.int8, device="cuda")
            assert not x.is_contiguous()
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cpu_non_contiguous_x(self):
        for x_shape, w_shape in self.shapes:
            x = torch.randint(
                -128, 127, x_shape[::-1], dtype=torch.int8, device="cpu"
            ).transpose(0, 1)
            w = torch.randint(-128, 127, w_shape, dtype=torch.int8, device="cpu")
            assert not x.is_contiguous()
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_device_mismatch_error(self):
        x_shape, w_shape = self.shapes[0]
        x = torch.randint(-128, 127, x_shape, dtype=torch.int8, device="cuda")
        w = torch.randint(-128, 127, w_shape, dtype=torch.int8, device="cpu")
        with self.assertRaisesRegex(
            AssertionError, "need both tensors to be on the same device but got.*"
        ):
            self._test_safe_int_mm_impl(x, w)


if __name__ == "__main__":
    unittest.main()
