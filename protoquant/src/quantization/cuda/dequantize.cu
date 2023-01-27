#include <protoquant/src/quantization/utilities.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <ATen/cuda/Atomic.cuh>

#include <cmath>
#include <limits>

// TODO: Optimize by architecture?
#define MAX_BLOCKS 2048
#define MAX_THREADS 1024

// MIN_BLOCK_WORK is the minimum number of batch elements that each block will
// be responsible for within the quantization/dequantization kernels
#define MIN_BLOCK_WORK 4

// MIN_THREAD_WORK is the minimum number of entries within a batch element that
// each thread will be responsible for within the quantization/dequantization
// kernels
#define MIN_THREAD_WORK 8

#define CALL_DQNTZ_OUT_KERNEL(out_type, input_type)               \
  auto out_accessor = out.packed_accessor32<out_type, 2>();       \
  auto input_accessor = input.packed_accessor32<input_type, 2>(); \
  dqntz_out_kernel<<<blocks, threads>>>(                          \
      out_accessor,                                               \
      input_accessor,                                             \
      scales_accessor,                                            \
      zeros_accessor,                                             \
      out_rows,                                                   \
      out_cols,                                                   \
      rowwise,                                                    \
      transpose);                                                 \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define CALL_DQNTZ_MM_OUT_KERNEL(dst_t, src_t)             \
  auto dst_accessor = out.packed_accessor32<dst_t, 2>();   \
  auto src_accessor = input.packed_accessor32<src_t, 2>(); \
  dqntz_mm_out_kernel<<<blocks, threads>>>(                \
      dst_accessor,                                        \
      src_accessor,                                        \
      quantized_rows,                                      \
      quantized_cols,                                      \
      input_scales_accessor,                               \
      input_zero_points_accessor,                          \
      input_precomputed_sums_accessor,                     \
      weight_scales_accessor,                              \
      weight_zero_points_accessor,                         \
      weight_precomputed_sums_accessor);                   \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define CALL_DQNTZ_MM_ADD_OUT_KERNEL(dst_t, src_t)           \
  auto dst_accessor = out.packed_accessor32<dst_t, 2>();     \
  auto src_accessor = input.packed_accessor32<src_t, 2>();   \
  auto other_accessor = other.packed_accessor32<dst_t, 1>(); \
  dqntz_mm_add_out_kernel<<<blocks, threads>>>(              \
      dst_accessor,                                          \
      src_accessor,                                          \
      other_accessor,                                        \
      quantized_rows,                                        \
      quantized_cols,                                        \
      input_scales_accessor,                                 \
      input_zero_points_accessor,                            \
      input_precomputed_sums_accessor,                       \
      weight_scales_accessor,                                \
      weight_zero_points_accessor,                           \
      weight_precomputed_sums_accessor);                     \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

namespace at {
namespace protoquant {

// __global__ kernels for dequantization
namespace {

template <typename out_type, typename input_type>
__global__ void dqntz_out_kernel(
    PackedTensorAccessor32<out_type, 2> out,
    PackedTensorAccessor32<input_type, 2> input,
    PackedTensorAccessor32<double, 1> scales,
    PackedTensorAccessor32<int32_t, 1> zeros,
    int64_t out_rows,
    int64_t out_cols,
    bool rowwise,
    bool transpose) {
  int temp;
  if (rowwise && transpose) {
    for (int64_t row = blockIdx.x; row < out_cols; row += gridDim.x) {
      for (int64_t col = threadIdx.x; col < out_rows; col += blockDim.x) {
        temp = input[row][col];
        temp = temp - zeros[col];
        out[col][row] = temp * scales[col];
      }
    }
  } else if (rowwise) {
    for (int64_t row = blockIdx.x; row < out_rows; row += gridDim.x) {
      for (int64_t col = threadIdx.x; col < out_cols; col += blockDim.x) {
        temp = input[row][col];
        temp = temp - zeros[row];
        out[row][col] = temp * scales[row];
      }
    }
  } else if (transpose) {
    for (int64_t row = blockIdx.x; row < out_cols; row += gridDim.x) {
      for (int64_t col = threadIdx.x; col < out_rows; col += blockDim.x) {
        temp = input[row][col];
        temp = temp - zeros[row];
        out[col][row] = temp * scales[row];
      }
    }
  } else {
    for (int64_t row = blockIdx.x; row < out_rows; row += gridDim.x) {
      for (int64_t col = threadIdx.x; col < out_cols; col += blockDim.x) {
        temp = input[row][col];
        temp = temp - zeros[col];
        out[row][col] = temp * scales[col];
      }
    }
  }
}

template <typename dst_type, typename src_type>
__global__ void dqntz_mm_out_kernel(
    PackedTensorAccessor32<dst_type, 2> dst_accessor,
    PackedTensorAccessor32<src_type, 2> src_accessor,
    int64_t quantized_rows,
    int64_t quantized_cols,
    PackedTensorAccessor32<double, 1> input_scales_accessor,
    PackedTensorAccessor32<int32_t, 1> input_zero_points_accessor,
    PackedTensorAccessor32<int32_t, 1> input_precomputed_sums_accessor,
    PackedTensorAccessor32<double, 1> weight_scales_accessor,
    PackedTensorAccessor32<int32_t, 1> weight_zero_points_accessor,
    PackedTensorAccessor32<int32_t, 1> weight_precomputed_sums_accessor) {
  int temp;
  for (int64_t row = blockIdx.x; row < quantized_rows; row += gridDim.x) {
    for (int64_t col = threadIdx.x; col < quantized_cols; col += blockDim.x) {
      temp = src_accessor[row][col];
      temp -= input_zero_points_accessor[row] *
          weight_precomputed_sums_accessor[col];
      temp -= input_precomputed_sums_accessor[row] *
          weight_zero_points_accessor[col];
      dst_accessor[row][col] =
          temp * input_scales_accessor[row] * weight_scales_accessor[col];
    }
  }
}

template <typename dst_type, typename src_type>
__global__ void dqntz_mm_add_out_kernel(
    PackedTensorAccessor32<dst_type, 2> dst_accessor,
    PackedTensorAccessor32<src_type, 2> src_accessor,
    PackedTensorAccessor32<dst_type, 1> other_accessor,
    int64_t quantized_rows,
    int64_t quantized_cols,
    PackedTensorAccessor32<double, 1> input_scales_accessor,
    PackedTensorAccessor32<int32_t, 1> input_zero_points_accessor,
    PackedTensorAccessor32<int32_t, 1> input_precomputed_sums_accessor,
    PackedTensorAccessor32<double, 1> weight_scales_accessor,
    PackedTensorAccessor32<int32_t, 1> weight_zero_points_accessor,
    PackedTensorAccessor32<int32_t, 1> weight_precomputed_sums_accessor) {
  int temp;
  for (int64_t row = blockIdx.x; row < quantized_rows; row += gridDim.x) {
    for (int64_t col = threadIdx.x; col < quantized_cols; col += blockDim.x) {
      temp = src_accessor[row][col];
      temp -= input_zero_points_accessor[row] *
          weight_precomputed_sums_accessor[col];
      temp -= input_precomputed_sums_accessor[row] *
          weight_zero_points_accessor[col];
      dst_accessor[row][col] = other_accessor[col] +
          temp * input_scales_accessor[row] * weight_scales_accessor[col];
    }
  }
}

} // namespace

Tensor dqntz_out_cuda(
    Tensor& out,
    const Tensor& input,
    const Tensor& scales,
    const Tensor& zeros,
    const Tensor& sums,
    const bool rowwise,
    const bool transpose) {
  cuda_check(out);
  auto device_guard(out.get_device());
  dqntz_out_check(out, input, scales, zeros, sums, rowwise, transpose);

  auto out_rows = out.sizes()[0];
  auto out_cols = out.sizes()[1];

  auto blocks = std::min(
      (out_rows + MIN_BLOCK_WORK - 1) / MIN_BLOCK_WORK,
      static_cast<int64_t>(MAX_BLOCKS));
  auto threads = std::min(
      (out_cols + MIN_THREAD_WORK - 1) / MIN_THREAD_WORK,
      static_cast<int64_t>(MAX_THREADS));

  auto out_type = out.scalar_type();
  auto input_type = input.scalar_type();

  auto scales_accessor = scales.packed_accessor32<double, 1>();
  auto zeros_accessor = zeros.packed_accessor32<int32_t, 1>();

  if (input_type == kChar && out_type == kHalf) {
    CALL_DQNTZ_OUT_KERNEL(Half, int8_t);
  } else if (input_type == kChar && out_type == kFloat) {
    CALL_DQNTZ_OUT_KERNEL(float, int8_t);
  } else if (input_type == kChar && out_type == kDouble) {
    CALL_DQNTZ_OUT_KERNEL(double, int8_t);
  } else if (input_type == kChar && out_type == kBFloat16) {
    CALL_DQNTZ_OUT_KERNEL(BFloat16, int8_t);
  } else if (input_type == kInt && out_type == kHalf) {
    CALL_DQNTZ_OUT_KERNEL(Half, int32_t);
  } else if (input_type == kInt && out_type == kFloat) {
    CALL_DQNTZ_OUT_KERNEL(float, int32_t);
  } else if (input_type == kInt && out_type == kDouble) {
    CALL_DQNTZ_OUT_KERNEL(double, int32_t);
  } else if (input_type == kInt && out_type == kBFloat16) {
    CALL_DQNTZ_OUT_KERNEL(BFloat16, int32_t);
  } else {
    TORCH_CHECK(false, "unsupported type combination");
  }

  return out;
}

Tensor dqntz_mm_out_cuda(
    Tensor& out,
    const Tensor& input,
    const Tensor& mat1_scales,
    const Tensor& mat1_zeros,
    const Tensor& mat1_sums,
    const bool mat1_rowwise,
    const bool mat1_transpose,
    const Tensor& mat2_scales,
    const Tensor& mat2_zeros,
    const Tensor& mat2_sums,
    const bool mat2_rowwise,
    const bool mat2_transpose) {
  cuda_check(out);
  auto device_guard(out.get_device());
  dqntz_mm_out_check(
      out,
      input,
      mat1_scales,
      mat1_zeros,
      mat1_sums,
      mat1_rowwise,
      mat1_transpose,
      mat2_scales,
      mat2_zeros,
      mat2_sums,
      mat2_rowwise,
      mat2_transpose);

  int64_t quantized_rows = mat1_scales.sizes()[0];
  int64_t quantized_cols = mat2_scales.sizes()[0];

  int64_t blocks = std::min(
      (quantized_rows + MIN_BLOCK_WORK - 1) / MIN_BLOCK_WORK,
      static_cast<int64_t>(MAX_BLOCKS));
  int64_t threads = std::min(
      (quantized_cols + MIN_THREAD_WORK - 1) / MIN_THREAD_WORK,
      static_cast<int64_t>(MAX_THREADS));

  ScalarType dst_type = out.scalar_type();
  ScalarType src_type = input.scalar_type();

  auto input_scales_accessor = mat1_scales.packed_accessor32<double, 1>();
  auto input_zero_points_accessor = mat1_zeros.packed_accessor32<int32_t, 1>();
  auto input_precomputed_sums_accessor =
      mat1_sums.packed_accessor32<int32_t, 1>();
  auto weight_scales_accessor = mat2_scales.packed_accessor32<double, 1>();
  auto weight_zero_points_accessor = mat2_zeros.packed_accessor32<int32_t, 1>();
  auto weight_precomputed_sums_accessor =
      mat2_sums.packed_accessor32<int32_t, 1>();

  if (src_type == at::kChar && dst_type == at::kHalf) {
    CALL_DQNTZ_MM_OUT_KERNEL(Half, int8_t);
  } else if (src_type == at::kChar && dst_type == at::kFloat) {
    CALL_DQNTZ_MM_OUT_KERNEL(float, int8_t);
  } else if (src_type == at::kChar && dst_type == at::kDouble) {
    CALL_DQNTZ_MM_OUT_KERNEL(double, int8_t);
  } else if (src_type == at::kChar && dst_type == at::kBFloat16) {
    CALL_DQNTZ_MM_OUT_KERNEL(BFloat16, int8_t);
  } else if (src_type == at::kInt && dst_type == at::kHalf) {
    CALL_DQNTZ_MM_OUT_KERNEL(Half, int32_t);
  } else if (src_type == at::kInt && dst_type == at::kFloat) {
    CALL_DQNTZ_MM_OUT_KERNEL(float, int32_t);
  } else if (src_type == at::kInt && dst_type == at::kDouble) {
    CALL_DQNTZ_MM_OUT_KERNEL(double, int32_t);
  } else if (src_type == at::kInt && dst_type == at::kBFloat16) {
    CALL_DQNTZ_MM_OUT_KERNEL(BFloat16, int32_t);
  } else {
    TORCH_CHECK(false, "unsupported type combination");
  }

  return out;
}

Tensor dqntz_mm_add_out_cuda(
    Tensor& out,
    const Tensor& input,
    const Tensor& other,
    const Tensor& mat1_scales,
    const Tensor& mat1_zeros,
    const Tensor& mat1_sums,
    const bool mat1_rowwise,
    const bool mat1_transpose,
    const Tensor& mat2_scales,
    const Tensor& mat2_zeros,
    const Tensor& mat2_sums,
    const bool mat2_rowwise,
    const bool mat2_transpose) {
  cuda_check(out);
  auto device_guard(out.get_device());
  dqntz_mm_out_check(
      out,
      input,
      mat1_scales,
      mat1_zeros,
      mat1_sums,
      mat1_rowwise,
      mat1_transpose,
      mat2_scales,
      mat2_zeros,
      mat2_sums,
      mat2_rowwise,
      mat2_transpose);

  int64_t quantized_rows = mat1_scales.sizes()[0];
  int64_t quantized_cols = mat2_scales.sizes()[0];

  int64_t blocks = std::min(
      (quantized_rows + MIN_BLOCK_WORK - 1) / MIN_BLOCK_WORK,
      static_cast<int64_t>(MAX_BLOCKS));
  int64_t threads = std::min(
      (quantized_cols + MIN_THREAD_WORK - 1) / MIN_THREAD_WORK,
      static_cast<int64_t>(MAX_THREADS));

  ScalarType dst_type = out.scalar_type();
  ScalarType src_type = input.scalar_type();

  auto input_scales_accessor = mat1_scales.packed_accessor32<double, 1>();
  auto input_zero_points_accessor = mat1_zeros.packed_accessor32<int32_t, 1>();
  auto input_precomputed_sums_accessor =
      mat1_sums.packed_accessor32<int32_t, 1>();
  auto weight_scales_accessor = mat2_scales.packed_accessor32<double, 1>();
  auto weight_zero_points_accessor = mat2_zeros.packed_accessor32<int32_t, 1>();
  auto weight_precomputed_sums_accessor =
      mat2_sums.packed_accessor32<int32_t, 1>();

  if (src_type == at::kChar && dst_type == at::kHalf) {
    CALL_DQNTZ_MM_ADD_OUT_KERNEL(Half, int8_t);
  } else if (src_type == at::kChar && dst_type == at::kFloat) {
    CALL_DQNTZ_MM_ADD_OUT_KERNEL(float, int8_t);
  } else if (src_type == at::kChar && dst_type == at::kDouble) {
    CALL_DQNTZ_MM_ADD_OUT_KERNEL(double, int8_t);
  } else if (src_type == at::kChar && dst_type == at::kBFloat16) {
    CALL_DQNTZ_MM_ADD_OUT_KERNEL(BFloat16, int8_t);
  } else if (src_type == at::kInt && dst_type == at::kHalf) {
    CALL_DQNTZ_MM_ADD_OUT_KERNEL(Half, int32_t);
  } else if (src_type == at::kInt && dst_type == at::kFloat) {
    CALL_DQNTZ_MM_ADD_OUT_KERNEL(float, int32_t);
  } else if (src_type == at::kInt && dst_type == at::kDouble) {
    CALL_DQNTZ_MM_ADD_OUT_KERNEL(double, int32_t);
  } else if (src_type == at::kInt && dst_type == at::kBFloat16) {
    CALL_DQNTZ_MM_ADD_OUT_KERNEL(BFloat16, int32_t);
  } else {
    TORCH_CHECK(false, "unsupported type combination");
  }

  return out;
}

TORCH_LIBRARY_IMPL(protoquant, CUDA, m) {
  m.impl("dqntz_out", dqntz_out_cuda);
  m.impl("dqntz_mm_out", dqntz_mm_out_cuda);
  m.impl("dqntz_mm_add_out", dqntz_mm_add_out_cuda);
}

} // namespace protoquant
} // namespace at
