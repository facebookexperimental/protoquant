import torch

import unittest
from quant_primitives import dynamically_quantize_per_tensor, safe_int_mm

torch.manual_seed(0)

class TestPerTensorQuantization(unittest.TestCase):
    """ 
    Tests the dynamically_quantize_per_tensor function across a variety of input cases and ensures numerics match ao version
    """
    shapes = (
        (1, 1, 32 , 32),
        (32, 16, 64, 64),
        (100, 100),
        (1, 200, 200),
    )
    def _test_dynamically_quantize_per_tensor_impl(self, device, quant_min = -128, quant_max = 127, target_dtype = torch.int8, tol = 0):
        for x_shape in self.shapes:
            x = torch.randn(x_shape, device=device)*1000
            
            x_int8, scale, zero_point = dynamically_quantize_per_tensor(x, quant_min, quant_max, target_dtype)
            
            q_dtype = torch.quint8 if target_dtype == torch.uint8 else torch.qint8
            x_q = torch.quantize_per_tensor_dynamic(x, dtype = q_dtype, reduce_range = False)
            
            self.assertEqual(scale.item(), x_q.q_scale())
            self.assertEqual(zero_point.item(), x_q.q_zero_point())
            self.assertGreaterEqual(tol, (x_int8.to(torch.int32)-x_q.int_repr().to(torch.int32)).abs().max())

            if device == 'cuda':
                trit_dynamic_quant = torch.compile(dynamically_quantize_per_tensor, mode='max-autotune')
                trit_x_int8, trit_scale, trit_zp = trit_dynamic_quant(x, quant_min, quant_max, target_dtype)
                self.assertEqual(trit_scale.item(), scale.item())
                self.assertEqual(trit_zp.item(), x_q.q_zero_point())
                self.assertGreaterEqual(0, (trit_x_int8.to(torch.int32)-x_q.int_repr().to(torch.int32)).abs().max())


    def test_dynamically_quantize_per_tensor_cuda_int8(self):
            self._test_dynamically_quantize_per_tensor_impl(
                device = 'cuda', quant_min = -128, quant_max = 127, target_dtype = torch.int8
            )

    def test_dynamically_quantize_per_tensor_cuda_uint8(self):
        self._test_dynamically_quantize_per_tensor_impl(
           device = 'cuda', quant_min = 0, quant_max = 255, target_dtype = torch.uint8
        )

    # CPU quantization has slightly different numerics than cuda, we chose to match cuda and
    # have all int values be within 1 of cpu
    def test_dynamically_quantize_per_tensor_cpu_int8(self):
            self._test_dynamically_quantize_per_tensor_impl(
                device = 'cpu', quant_min = -128, quant_max = 127, target_dtype = torch.int8, tol = 1
            )

    def test_dynamically_quantize_per_tensor_cpu_uint8(self):
        self._test_dynamically_quantize_per_tensor_impl(
           device = 'cpu', quant_min = 0, quant_max = 255, target_dtype = torch.uint8, tol = 1
        )

class TestSafeIntMM(unittest.TestCase):
    """ 
    Tests the safe_int_mm functionality/correctness across a variety of input cases  
    """
    test_shapes = (
        # ((x_shape), (w_shape))
        ((8, 17), (17, 8)), # break cublas but not triton (fallback)
        ((17, 24), (24, 8)), # smallest test that doesn't need fallback
        ((1536, 1536), (1536, 1536)),
        ((17, 4096), (4096, 1536)),
        # ((17, 8), (8, 8)), # breaks triton but not cublas 
        # note: this last isn't tested since triton path doesn't have the fallback option for perf reasons,
        # so the error is expected
    )

    def _test_safe_int_mm_impl(self, x, w):
        y = safe_int_mm(x, w)
        y_ref = torch.matmul(x.to(torch.int32).cpu(),w.to(torch.int32).cpu()).to(x.device)
        torch.testing.assert_close(
            y_ref, y, atol=0, rtol=0,
            msg = r"failed for shape {} and {}".format(x.shape, w.shape)
        )

        if x.device.type == 'cuda':
            trit_safe_int_mm = torch.compile(safe_int_mm, mode='max-autotune')
            y_triton = trit_safe_int_mm(x, w)
            torch.testing.assert_close(
                y_ref, y_triton, atol=0, rtol=0, 
                msg = r"failed for shape {} and {}".format(x.shape, w.shape)
                )

    def test_safe_int_mm_cuda(self):
        for x_shape, w_shape in self.test_shapes:
            x = torch.randint(-128, 127, x_shape, dtype = torch.int8, device='cuda')
            w = torch.randint(-128, 127, w_shape, dtype = torch.int8, device='cuda')
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cpu(self):
        for x_shape, w_shape in self.test_shapes:
            x = torch.randint(-128, 127, x_shape, dtype = torch.int8, device='cpu')
            w = torch.randint(-128, 127, w_shape, dtype = torch.int8, device='cpu')
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cuda_non_contiguous_w(self):
        for x_shape, w_shape in self.test_shapes:
            x = torch.randint(-128, 127, x_shape, dtype = torch.int8, device='cuda')
            w = torch.randint(-128, 127, w_shape[::-1], dtype = torch.int8, device='cuda').transpose(0,1)
            assert not w.is_contiguous()
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cpu_non_contiguous_w(self):
        for x_shape, w_shape in self.test_shapes:
            x = torch.randint(-128, 127, x_shape, dtype = torch.int8, device='cpu')
            w = torch.randint(-128, 127, w_shape[::-1], dtype = torch.int8, device='cpu').transpose(0,1)
            assert not w.is_contiguous()
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cuda_non_contiguous_x(self):
        for x_shape, w_shape in self.test_shapes:
            x = torch.randint(-128, 127, x_shape[::-1], dtype = torch.int8, device='cuda').transpose(0,1)
            w = torch.randint(-128, 127, w_shape, dtype = torch.int8, device='cuda')
            assert not x.is_contiguous()
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cpu_non_contiguous_x(self):
        for x_shape, w_shape in self.test_shapes:
            x = torch.randint(-128, 127, x_shape[::-1], dtype = torch.int8, device='cpu').transpose(0,1)
            w = torch.randint(-128, 127, w_shape, dtype = torch.int8, device='cpu')
            assert not x.is_contiguous()
            self._test_safe_int_mm_impl(x, w)


if __name__ == "__main__":
    unittest.main()
