#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary
# Owner(s): ["cpuhrsch"]

# pyre-unsafe

import unittest

import protoquant

import torch

m = k = n = 2

DEVICE = "cuda"


@unittest.skipIf(
    not torch.cuda.is_available(),
    "test_qt_mm requires available CUDA device, none found.",
)
def test_qt_mm(input_dtype=torch.float16):
    mat1 = torch.randn(m, k, device=DEVICE, dtype=input_dtype)
    qmat1 = protoquant.QTensor(mat1)
    mat2 = torch.randn(k, n, device=DEVICE, dtype=input_dtype)
    qmat2 = protoquant.QTensor(mat2)

    rslt = torch.mm(qmat1, qmat2)

    # actl = torch.mm(Tmat1, mat2)
    # assert is_close_batch_aware(rslt, actl)


class TestQT(unittest.TestCase):
    def test_qt(self):
        test_qt_mm()


if __name__ == "__main__":
    unittest.main()
