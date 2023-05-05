import torch

import unittest
from itertools import cycle
from quant_primitives import dynamically_quantize_per_tensor, safe_int_mm, dynamically_quantize_per_channel, dequantize_per_tensor, dequantize_per_channel, quant_int8_matmul

torch.manual_seed(0)

class TestDequantizePerChannel(unittest.TestCase):
    shapes = (
        (5, 5),
        (2, 8, 32 , 32),
        (32, 16, 64, 64),
        (1, 200, 200),
    )
    def _test_dequantize_per_channel_impl(self, device, quant_min = -128, quant_max = 127, target_dtype = torch.int8):
        out_dtypes = cycle([torch.float16, torch.float32, torch.float64])
        for x_shape in self.shapes:
            for axis in range(len(x_shape)):
                out_dtype = next(out_dtypes)
                x = torch.randn(x_shape, device=device)*1000
                
                _, scales, zero_points = dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype, axis)

                q_dtype = torch.quint8 if target_dtype == torch.uint8 else torch.qint8
                ref_q = torch.quantize_per_channel(x, scales, zero_points, axis, q_dtype)
                ref_int = ref_q.int_repr()

                self.assertEqual(ref_int.dtype, target_dtype)

                x_dq = dequantize_per_channel(ref_int, scales, zero_points, out_dtype, axis)
                ref_dq = ref_q.dequantize().to(out_dtype)
                self.assertEqual(x_dq.dtype, out_dtype)
                torch.testing.assert_close(x_dq, ref_dq)

                if device == 'cuda':
                    trit_dequantize_per_channel = torch.compile(dequantize_per_channel, mode='max-autotune')
                    trit_dq = trit_dequantize_per_channel(ref_int, scales, zero_points, out_dtype, axis)
                    self.assertEqual(trit_dq.dtype, out_dtype)
                    torch.testing.assert_close(trit_dq, ref_dq)
  
    def test_dequantize_per_channel_cuda_int8(self):
        self._test_dequantize_per_channel_impl(
            device = 'cuda', quant_min = -128, quant_max = 127, target_dtype = torch.int8
        )

    def test_dequantize_per_channel_cuda_uint8(self):
        self._test_dequantize_per_channel_impl(
           device = 'cuda', quant_min = 0, quant_max = 255, target_dtype = torch.uint8
        )

    def test_dequantize_per_channel_cpu_int8(self):
        self._test_dequantize_per_channel_impl(
            device = 'cpu', quant_min = -128, quant_max = 127, target_dtype = torch.int8
        )

    def test_dequantize_per_channel_cpu_uint8(self):
        self._test_dequantize_per_channel_impl(
           device = 'cpu', quant_min = 0, quant_max = 255, target_dtype = torch.uint8
        )
class TestDequantizePerTensor(unittest.TestCase):
    shapes = (
        (5, 5),
        (2, 8, 32 , 32),
        (32, 16, 64, 64),
        (1, 200, 200),
    )
    def _test_dequantize_per_tensor_impl(self, device, quant_min = -128, quant_max = 127, target_dtype = torch.int8):
        out_dtypes = [torch.float16, torch.float32, torch.float64]
        for x_shape in self.shapes:
            for out_dtype in out_dtypes:
                x = torch.randn(x_shape, device=device)*1000
                
                _, scale, zero_point = dynamically_quantize_per_tensor(x, quant_min, quant_max, target_dtype)

                q_dtype = torch.quint8 if target_dtype == torch.uint8 else torch.qint8
                ref_q = torch.quantize_per_tensor_dynamic(x, q_dtype, reduce_range = False)
                ref_int = ref_q.int_repr()

                self.assertEqual(ref_int.dtype, target_dtype)

                x_dq = dequantize_per_tensor(ref_int, ref_q.q_scale(), ref_q.q_zero_point(), out_dtype) # scalar args
                x_dq2 = dequantize_per_tensor(ref_int, scale, zero_point, out_dtype) # tensor args
                ref_dq = ref_q.dequantize().to(out_dtype)
                self.assertEqual(x_dq.dtype, out_dtype)
                self.assertEqual(x_dq2.dtype, out_dtype)
                torch.testing.assert_close(x_dq, ref_dq)
                torch.testing.assert_close(x_dq2, ref_dq)

                if device == 'cuda':
                    trit_dequantize_per_tensor = torch.compile(dequantize_per_tensor, mode='max-autotune')
                    # x_dq = dequantize_per_tensor(ref_int, ref_q.q_scale(), ref_q.q_zero_point(), out_dtype) # scalar args, not working
                    trit_dq = trit_dequantize_per_tensor(ref_int, scale, zero_point, out_dtype) # tensor args
                    self.assertEqual(trit_dq.dtype, out_dtype)
                    torch.testing.assert_close(trit_dq, ref_dq)
  
    def test_dequantize_per_tensor_cuda_int8(self):
        self._test_dequantize_per_tensor_impl(
            device = 'cuda', quant_min = -128, quant_max = 127, target_dtype = torch.int8
        )

    def test_dequantize_per_tensor_cuda_uint8(self):
        self._test_dequantize_per_tensor_impl(
           device = 'cuda', quant_min = 0, quant_max = 255, target_dtype = torch.uint8
        )

    def test_dequantize_per_tensor_cpu_int8(self):
        self._test_dequantize_per_tensor_impl(
            device = 'cpu', quant_min = -128, quant_max = 127, target_dtype = torch.int8
        )

    def test_dequantize_per_tensor_cpu_uint8(self):
        self._test_dequantize_per_tensor_impl(
           device = 'cpu', quant_min = 0, quant_max = 255, target_dtype = torch.uint8
        )


class TestPerChannelQuantization(unittest.TestCase):
    """ 
    Tests the dynamically_quantize_per_channel function across a variety of input cases and ensures numerics match ao version
    """
    shapes = (
        (1, 200, 200),
        (5, 5),
        (2, 8, 32 , 32),
        (32, 16, 64, 64),
    )
    def _test_dynamically_quantize_per_channel_impl(self, device, quant_min = -128, quant_max = 127, target_dtype = torch.int8):
        f_dtypes = cycle([torch.float16, torch.float32, torch.float64])
        transposes = cycle([True, False])
        for x_shape in self.shapes:
            for axis in range(len(x_shape)):
                transp = next(transposes)
                f_dtype = next(f_dtypes)
                x = torch.randn(x_shape, device=device, dtype = f_dtype)*1000
                if transp:
                    x = x.transpose(0, -1)
                
                x_int8, scales, zero_points = dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype, axis=axis)
            

                q_dtype = torch.quint8 if target_dtype == torch.uint8 else torch.qint8

                obs = torch.ao.quantization.PerChannelMinMaxObserver(ch_axis = axis, dtype = q_dtype, qscheme = torch.per_channel_symmetric, reduce_range = False)
                obs(x)
                ref_scales, ref_zero_points = obs.calculate_qparams()
                ref_scales, ref_zero_points = ref_scales.to(x.device), ref_zero_points.to(x.device)
                torch.testing.assert_close(scales.to(torch.float32), ref_scales)
                torch.testing.assert_close(zero_points, ref_zero_points, atol=0, rtol=0)

                x_q_int_repr = torch.quantize_per_channel(x.to(torch.float32), ref_scales, ref_zero_points, axis=axis, dtype=q_dtype).int_repr()
                torch.testing.assert_close(x_int8, x_q_int_repr, atol=1, rtol=100)

                if device == 'cuda':
                    trit_dynamic_quant = torch.compile(dynamically_quantize_per_channel, mode='max-autotune')
                    trit_x_int8, trit_scales, trit_zps = trit_dynamic_quant(x, quant_min, quant_max, target_dtype, axis=axis)
                    torch.testing.assert_close(trit_scales.to(torch.float32), ref_scales)
                    torch.testing.assert_close(trit_zps, ref_zero_points, atol=0, rtol=0)
                    torch.testing.assert_close(trit_x_int8, x_q_int_repr, atol=1, rtol=100)


    def test_dynamically_quantize_per_channel_cuda_int8(self):
            self._test_dynamically_quantize_per_channel_impl(
                device = 'cuda', quant_min = -128, quant_max = 127, target_dtype = torch.int8
            )

    def test_dynamically_quantize_per_channel_cuda_uint8(self):
        self._test_dynamically_quantize_per_channel_impl(
           device = 'cuda', quant_min = 0, quant_max = 255, target_dtype = torch.uint8
        )

    def test_dynamically_quantize_per_channel_cpu_int8(self):
            self._test_dynamically_quantize_per_channel_impl(
                device = 'cpu', quant_min = -128, quant_max = 127, target_dtype = torch.int8
            )

    def test_dynamically_quantize_per_channel_cpu_uint8(self):
        self._test_dynamically_quantize_per_channel_impl(
           device = 'cpu', quant_min = 0, quant_max = 255, target_dtype = torch.uint8
        )

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
        f_dtypes = cycle([torch.float16, torch.float32, torch.float64])
        for x_shape in self.shapes:
            for transp in [False, True]:
                f_dtype = next(f_dtypes)
                x = torch.randn(x_shape, device=device, dtype=f_dtype)*1000
                if transp:
                    x = x.transpose(0,-1)
                
                
                x_int8, scale, zero_point = dynamically_quantize_per_tensor(x, quant_min, quant_max, target_dtype)
                
                q_dtype = torch.quint8 if target_dtype == torch.uint8 else torch.qint8
                x_q = torch.quantize_per_tensor_dynamic(x.to(torch.float32), dtype = q_dtype, reduce_range = False)
                
                torch.testing.assert_close(scale, x_q.q_scale())
                torch.testing.assert_close(zero_point, x_q.q_zero_point(), atol=0, rtol=0)
                torch.testing.assert_close(x_int8.to(torch.int32),x_q.int_repr().to(torch.int32), atol = tol, rtol = 100)

                if device == 'cuda':
                    trit_dynamic_quant = torch.compile(dynamically_quantize_per_tensor, mode='max-autotune')
                    trit_x_int8, trit_scale, trit_zp = trit_dynamic_quant(x, quant_min, quant_max, target_dtype)
                    torch.testing.assert_close(trit_scale, x_q.q_scale())
                    torch.testing.assert_close(trit_zp, x_q.q_zero_point(), atol=0, rtol=0)
                    torch.testing.assert_close(trit_x_int8.to(torch.int32),x_q.int_repr().to(torch.int32), atol = tol, rtol = 100)


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

class TestQuantInt8MatMul(unittest.TestCase):
    shapes = (
        # ((x_shape), (w_shape))
        ((1, 1, 32 , 32),(20, 32)),
        ((32, 16, 64, 64),(16, 64)),
        ((100, 100),(2, 100)),
        ((1, 200, 200),(100, 200)),
    )
    def _test_quant_int8_matmul_impl(self, device):
        # x_vals_int8,
        # x_scale,
        # x_zp,
        # w_int8_t,
        # w_int8_t_sums_int64,
        # w_scales,
        # out_dtype=torch.float32,
        out_dtypes = cycle([torch.float16, torch.float32, torch.float64])
        for x_shape, w_shape in self.shapes:
            for contiguous_x in [True]:#, False]:
                for contiguous_w in [True]:#, False]:
                    out_dtype = next(out_dtypes)
                    
                    x = torch.randn(x_shape, device=device)
                    x_vals_int8, x_scale, x_zp = dynamically_quantize_per_tensor(x)

                    lin = torch.nn.Linear(in_features = w_shape[1], out_features = w_shape[0], bias=False).to(device)
                    lin.qconfig = torch.ao.quantization.default_per_channel_qconfig
                    lin.activation_post_process = lin.qconfig.activation(dtype = torch.quint8, quant_min = 0, quant_max=255)
                    lin.activation_post_process(lin(x))
                    qlinear = torch.ao.nn.quantized.modules.Linear.from_float(lin)

            
                    w_int8_t, w_scales, _ = dynamically_quantize_per_channel(lin.weight.transpose(0,1))
                    w_int8_t_sums_int64 = w_int8_t.sum(d=0)
                    y = quant_int8_matmul(x_vals_int8, x_scale, x_zp, w_int8_t, w_int8_t_sums_int64, w_scales, out_dtype)
                    y_ref = torch.nn.functional.linear(x, w)

                    torch.testing.assert_close()



class TestSafeIntMM(unittest.TestCase):
    """ 
    Tests the safe_int_mm functionality/correctness across a variety of input cases  
    """
    shapes = (
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
        for x_shape, w_shape in self.shapes:
            x = torch.randint(-128, 127, x_shape, dtype = torch.int8, device='cuda')
            w = torch.randint(-128, 127, w_shape, dtype = torch.int8, device='cuda')
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cpu(self):
        for x_shape, w_shape in self.shapes:
            x = torch.randint(-128, 127, x_shape, dtype = torch.int8, device='cpu')
            w = torch.randint(-128, 127, w_shape, dtype = torch.int8, device='cpu')
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cuda_non_contiguous_w(self):
        for x_shape, w_shape in self.shapes:
            x = torch.randint(-128, 127, x_shape, dtype = torch.int8, device='cuda')
            w = torch.randint(-128, 127, w_shape[::-1], dtype = torch.int8, device='cuda').transpose(0,1)
            assert not w.is_contiguous()
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cpu_non_contiguous_w(self):
        for x_shape, w_shape in self.shapes:
            x = torch.randint(-128, 127, x_shape, dtype = torch.int8, device='cpu')
            w = torch.randint(-128, 127, w_shape[::-1], dtype = torch.int8, device='cpu').transpose(0,1)
            assert not w.is_contiguous()
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cuda_non_contiguous_x(self):
        for x_shape, w_shape in self.shapes:
            x = torch.randint(-128, 127, x_shape[::-1], dtype = torch.int8, device='cuda').transpose(0,1)
            w = torch.randint(-128, 127, w_shape, dtype = torch.int8, device='cuda')
            assert not x.is_contiguous()
            self._test_safe_int_mm_impl(x, w)

    def test_safe_int_mm_cpu_non_contiguous_x(self):
        for x_shape, w_shape in self.shapes:
            x = torch.randint(-128, 127, x_shape[::-1], dtype = torch.int8, device='cpu').transpose(0,1)
            w = torch.randint(-128, 127, w_shape, dtype = torch.int8, device='cpu')
            assert not x.is_contiguous()
            self._test_safe_int_mm_impl(x, w)


if __name__ == "__main__":
    unittest.main()
