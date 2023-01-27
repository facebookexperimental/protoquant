#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

import numpy
import torch

from protoquant import dqntz, qntz


def rtol_95_percentile(actual, expected):
    rtols = abs(actual - expected) / expected
    return numpy.percentile(rtols.cpu(), 95)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "QUANTIZATION_E2E_CUDA_TESTS require available CUDA device, none found.",
)
class QUANTIZATION_E2E_CUDA_TESTS(unittest.TestCase):
    device = "cuda"

    def qntz_dqntz_e2e_cuda(self, is_a, do_pad, dtype, m, n, max_rtol_95=0.01):
        numpy.random.seed(0)
        torch.manual_seed(0)

        input = torch.testing.make_tensor([m, n], dtype=dtype, device=self.device)

        qntzd, params = qntz(input, is_a=is_a, do_pad=do_pad)
        dqntzd = dqntz(qntzd, mat1_params=params)

        expected = input

        assert dqntzd.shape == expected.shape
        assert dqntzd.dtype == expected.dtype
        assert dqntzd.device == expected.device
        val = rtol_95_percentile(dqntzd, expected)
        assert val < max_rtol_95, f"val: {val}"

    def test_qntz_dqntz_e2e_cuda_float16(self):
        is_a, do_pad = False, False
        self.qntz_dqntz_e2e_cuda(is_a, do_pad, torch.float16, 4, 4, 0.011)

    def test_qntz_dqntz_e2e_cuda_float32(self):
        is_a, do_pad = False, False
        self.qntz_dqntz_e2e_cuda(is_a, do_pad, torch.float32, 4, 4, 0.011)

    def test_qntz_dqntz_e2e_cuda_float64(self):
        is_a, do_pad = False, False
        self.qntz_dqntz_e2e_cuda(is_a, do_pad, torch.float64, 4, 4, 0.013)

    def test_qntz_dqntz_is_a_e2e_cuda_float16(self):
        is_a, do_pad = True, False
        self.qntz_dqntz_e2e_cuda(is_a, do_pad, torch.float16, 4, 4, 0.025)

    def test_qntz_dqntz_is_a_e2e_cuda_float32(self):
        is_a, do_pad = True, False
        self.qntz_dqntz_e2e_cuda(is_a, do_pad, torch.float32, 4, 4, 0.022)

    def test_qntz_dqntz_is_a_e2e_cuda_float64(self):
        is_a, do_pad = True, False
        self.qntz_dqntz_e2e_cuda(is_a, do_pad, torch.float64, 4, 4)

    def test_qntz_dqntz_do_pad_e2e_cuda_float16(self):
        is_a, do_pad = False, True
        self.qntz_dqntz_e2e_cuda(is_a, do_pad, torch.float16, 4, 4, 0.011)

    def test_qntz_dqntz_do_pad_e2e_cuda_float32(self):
        is_a, do_pad = False, True
        self.qntz_dqntz_e2e_cuda(is_a, do_pad, torch.float32, 4, 4, 0.011)

    def test_qntz_dqntz_do_pad_e2e_cuda_float64(self):
        is_a, do_pad = False, True
        self.qntz_dqntz_e2e_cuda(is_a, do_pad, torch.float64, 4, 4, 0.013)

    def test_qntz_dqntz_is_a_do_pad_e2e_cuda_float16(self):
        is_a, do_pad = True, True
        self.qntz_dqntz_e2e_cuda(is_a, do_pad, torch.float16, 4, 4, 0.025)

    def test_qntz_dqntz_is_a_do_pad_e2e_cuda_float32(self):
        is_a, do_pad = True, True
        self.qntz_dqntz_e2e_cuda(is_a, do_pad, torch.float32, 4, 4, 0.022)

    def test_qntz_dqntz_is_a_do_pad_e2e_cuda_float64(self):
        is_a, do_pad = True, True
        self.qntz_dqntz_e2e_cuda(is_a, do_pad, torch.float64, 4, 4)

    def _test_qntz_mm_dqntz_e2e_cuda(self, dtype, m, k, n):
        numpy.random.seed(0)
        torch.manual_seed(0)

        input = torch.testing.make_tensor([m, k], dtype=dtype, device=self.device)
        mat2 = torch.testing.make_tensor([k, n], dtype=dtype, device=self.device)

        input_qntzd, input_params = qntz(input, is_a=True, do_pad=True)
        mat2_qntzd, mat2_params = qntz(mat2, is_a=False, do_pad=True)

        qntzd = torch.mm(
            input_qntzd.to(torch.int32).cpu(),
            mat2_qntzd.to(torch.int32).transpose(0, 1).cpu(),
        ).cuda()
        dqntzd = dqntz(qntzd, mat1_params=input_params, mat2_params=mat2_params)

        expected = torch.mm(input, mat2)

        assert dqntzd.dtype == expected.dtype
        assert dqntzd.device == expected.device
        val = rtol_95_percentile(dqntzd, expected)
        assert val < 0.05, f"val: {val}"

    def test_qntz_mm_dqntz_e2e_cuda_float16(self):
        self._test_qntz_mm_dqntz_e2e_cuda(torch.float16, 1, 2, 1)
        self._test_qntz_mm_dqntz_e2e_cuda(torch.float16, 2, 2, 3)

    def test_qntz_mm_dqntz_e2e_cuda_float32(self):
        self._test_qntz_mm_dqntz_e2e_cuda(torch.float32, 1, 2, 1)
        self._test_qntz_mm_dqntz_e2e_cuda(torch.float32, 2, 2, 3)

    def test_qntz_mm_dqntz_e2e_cuda_float64(self):
        self._test_qntz_mm_dqntz_e2e_cuda(torch.float64, 1, 2, 1)
        self._test_qntz_mm_dqntz_e2e_cuda(torch.float64, 2, 2, 3)

    def _test_qntz_mm_dqntz_add_e2e_cuda(self, dtype, m, k, n):
        numpy.random.seed(0)
        torch.manual_seed(0)

        input = torch.testing.make_tensor([n], dtype=dtype, device=self.device)
        mat1 = torch.testing.make_tensor([m, k], dtype=dtype, device=self.device)
        mat2 = torch.testing.make_tensor([k, n], dtype=dtype, device=self.device)

        mat1_qntzd, mat1_params = qntz(mat1, is_a=True, do_pad=True)
        mat2_qntzd, mat2_params = qntz(mat2, is_a=False, do_pad=True)

        qntzd = torch.mm(
            mat1_qntzd.to(torch.int32).cpu(),
            mat2_qntzd.to(torch.int32).transpose(0, 1).cpu(),
        ).cuda()
        dqntzd = dqntz(
            qntzd, mat1_params=mat1_params, mat2_params=mat2_params, other=input
        )

        expected = torch.mm(mat1, mat2) + input

        assert dqntzd.dtype == expected.dtype
        assert dqntzd.device == expected.device
        val = rtol_95_percentile(dqntzd, expected)
        assert val < 0.05, f"val: {val}"

    def test_qntz_mm_dqntz_add_e2e_cuda_float16(self):
        self._test_qntz_mm_dqntz_add_e2e_cuda(torch.float16, 1, 2, 1)
        self._test_qntz_mm_dqntz_add_e2e_cuda(torch.float16, 2, 2, 3)

    def test_qntz_mm_dqntz_add_e2e_cuda_float32(self):
        self._test_qntz_mm_dqntz_add_e2e_cuda(torch.float32, 1, 2, 1)
        self._test_qntz_mm_dqntz_add_e2e_cuda(torch.float32, 2, 2, 3)

    def test_qntz_mm_dqntz_add_e2e_cuda_float64(self):
        self._test_qntz_mm_dqntz_add_e2e_cuda(torch.float64, 1, 2, 1)
        self._test_qntz_mm_dqntz_add_e2e_cuda(torch.float64, 2, 2, 3)
