#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# Owner(s): ["cpuhrsch"]

# pyre-unsafe

import unittest

import numpy
import torch

from protoquant import gemm, pad

numpy.random.seed(0)
torch.manual_seed(0)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "GEMM_E2E_CUDA_TESTS require available CUDA device, none found.",
)
class GEMM_E2E_CUDA_TESTS(unittest.TestCase):
    device, iterations, max_size = "cuda", 10, 100

    def test_gemm_e2e_cuda(self):
        for _ in range(self.iterations):
            m = pad(numpy.random.randint(low=1, high=self.max_size))
            k = pad(numpy.random.randint(low=1, high=self.max_size))
            n = pad(numpy.random.randint(low=1, high=self.max_size))

            input = torch.testing.make_tensor(
                [m, n], dtype=torch.int32, device=self.device
            )
            mat1 = torch.testing.make_tensor(
                [m, k], dtype=torch.int8, device=self.device
            )
            mat2 = torch.testing.make_tensor(
                [n, k], dtype=torch.int8, device=self.device
            )

            input_copy = input.clone().cpu()
            mat1_copy = mat1.clone().cpu()
            mat2_copy = mat2.clone().cpu()

            out = gemm(mat1, mat2, input=input)

            out_copy = torch.addmm(
                input_copy,
                mat1_copy.to(torch.int32),
                mat2_copy.to(torch.int32).transpose(0, 1),
            )

            # torch.testing.assert_close(input.cpu(), input_copy)
            torch.testing.assert_close(mat1.cpu(), mat1_copy)
            torch.testing.assert_close(mat2.cpu(), mat2_copy)
            torch.testing.assert_close(out.cpu(), out_copy)

    def test_gemm_out_e2e_cuda(self):
        for _ in range(self.iterations):
            m = pad(numpy.random.randint(low=1, high=self.max_size))
            k = pad(numpy.random.randint(low=1, high=self.max_size))
            n = pad(numpy.random.randint(low=1, high=self.max_size))

            mat1 = torch.testing.make_tensor(
                [m, k], dtype=torch.int8, device=self.device
            )
            mat2 = torch.testing.make_tensor(
                [n, k], dtype=torch.int8, device=self.device
            )
            out = torch.testing.make_tensor(
                [m, n], dtype=torch.int32, device=self.device
            )

            mat1_copy = mat1.clone().cpu()
            mat2_copy = mat2.clone().cpu()
            out_copy = out.clone().cpu()

            gemm(mat1, mat2, out=out)

            torch.mm(
                mat1_copy.to(torch.int32),
                mat2_copy.to(torch.int32).transpose(0, 1),
                out=out_copy,
            )

            torch.testing.assert_close(mat1.cpu(), mat1_copy)
            torch.testing.assert_close(mat2.cpu(), mat2_copy)
            torch.testing.assert_close(out.cpu(), out_copy)

    def test_ngemm_e2e_cuda(self):
        for _ in range(self.iterations):
            m = pad(numpy.random.randint(low=1, high=self.max_size))
            k = pad(numpy.random.randint(low=1, high=self.max_size))
            n = pad(numpy.random.randint(low=1, high=self.max_size))

            mat1 = torch.testing.make_tensor(
                [m, k], dtype=torch.int8, device=self.device
            )
            mat2 = torch.testing.make_tensor(
                [n, k], dtype=torch.int8, device=self.device
            )

            mat1_copy = mat1.clone().cpu()
            mat2_copy = mat2.clone().cpu()

            out = gemm(mat1, mat2)

            out_copy = torch.mm(
                mat1_copy.to(torch.int32),
                mat2_copy.to(torch.int32).transpose(0, 1),
            )

            torch.testing.assert_close(mat1.cpu(), mat1_copy)
            torch.testing.assert_close(mat2.cpu(), mat2_copy)
            torch.testing.assert_close(out.cpu(), out_copy)
