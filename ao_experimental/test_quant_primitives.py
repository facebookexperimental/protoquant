import unittest

import torch
from quant_primitives import safe_int_mm


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
