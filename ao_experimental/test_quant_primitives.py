import unittest
import torch
from quant_primitives import (
    safe_int_mm,
    dynamically_quantize_per_tensor,
    dynamically_quantize_per_channel,
)
from itertools import cycle as cycle

torch.manual_seed(0)


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
