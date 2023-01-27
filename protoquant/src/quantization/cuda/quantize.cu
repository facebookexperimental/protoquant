#include <protoquant/src/quantization/utilities.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <ATen/cuda/Atomic.cuh>

#include <cmath>
#include <limits>

#define SMALL_SCALE_THRESHOLD 6.1e-5f

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

#define CALL_QUANTIZE_KERNEL(scalar_t)                                 \
  auto src_accessor = input.packed_accessor32<scalar_t, 2>();          \
  auto min_accessor = mins.packed_accessor32<scalar_t, 1>();           \
  auto max_accessor = maxs.packed_accessor32<scalar_t, 1>();           \
  auto sum_accessor = _sums.packed_accessor32<scalar_t, 1>();          \
  auto func = [](bool _row_wise, bool _transpose) {                    \
    return _row_wise ? _transpose                                      \
            ? batch_aware_quantize_row_wise_transpose_kernel<scalar_t> \
            : batch_aware_quantize_row_wise_kernel<scalar_t>           \
        : _transpose                                                   \
        ? batch_aware_quantize_col_wise_transpose_kernel<scalar_t>     \
        : batch_aware_quantize_col_wise_kernel<scalar_t>;              \
  };                                                                   \
  auto kernel = func(rowwise, transpose);                              \
  kernel<<<blocks, threads>>>(                                         \
      dst_accessor,                                                    \
      dst_sizes[0],                                                    \
      dst_sizes[1],                                                    \
      src_accessor,                                                    \
      src_sizes[0],                                                    \
      src_sizes[1],                                                    \
      min_accessor,                                                    \
      max_accessor,                                                    \
      sum_accessor,                                                    \
      scales_accessor,                                                 \
      zero_points_accessor,                                            \
      precomputed_sums_accessor);                                      \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

namespace at {
namespace protoquant {

// __device__ helper functions
namespace {

__device__ void choose_quantization_params_device(
    float min,
    float max,
    int32_t qmin,
    int32_t qmax,
    double* _scale,
    int32_t* _zero_point) {
  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  // Use double precision for intermediate computation but use single precision
  // in final number to reflect the actual number used during quantization.
  float scale = (static_cast<double>(max) - min) / (qmax - qmin);
  // If scale is 0 or too small so its reciprocal is infinity, we arbitrary
  // adjust the scale to 0.1 . We want to avoid scale's reciprocal being
  // infinity because some of fbgemm code pre-computes scale's reciprocal to do
  // multiplication instead of division in the time critical part of code.
  if (scale == 0.0f || std::isinf(1.0f / scale)) {
    scale = 0.1;
  }

  // Cut off small scale
  if (scale < SMALL_SCALE_THRESHOLD) {
    float org_scale = scale;
    scale = SMALL_SCALE_THRESHOLD;
    // Adjust the min and max based on the new scale
    if (min == 0.0f) {
      max = SMALL_SCALE_THRESHOLD * (qmax - qmin);
    } else if (max == 0.0f) {
      min = -SMALL_SCALE_THRESHOLD * (qmax - qmin);
    } else {
      float amplifier = SMALL_SCALE_THRESHOLD / org_scale;
      min *= amplifier;
      max *= amplifier;
    }
  }

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  double zero_point_from_min = qmin - min / static_cast<double>(scale);
  double zero_point_from_max = qmax - max / static_cast<double>(scale);
  double zero_point_from_min_error =
      std::abs(qmin) + std::abs(min / static_cast<double>(scale));
  double zero_point_from_max_error =
      std::abs(qmax) + std::abs(max / static_cast<double>(scale));
  double initial_zero_point =
      zero_point_from_min_error < zero_point_from_max_error
      ? zero_point_from_min
      : zero_point_from_max;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with zero
  // padding).
  int32_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = nearbyint(initial_zero_point);
  }

  *_scale = scale;
  *_zero_point = nudged_zero_point;
}

template <typename T1, typename T2 = std::uint8_t>
NO_SANITIZE("signed-integer-overflow")
__device__ T2 clamp_device(T1 src, int precision, bool is_signed = false) {
  std::int32_t min = is_signed ? -(1LL << (precision - 1)) : 0;
  std::int32_t max =
      is_signed ? ((1LL << (precision - 1)) - 1) : (1LL << precision) - 1;
  return std::min<T1>(std::max<T1>(src, min), max);
}

} // namespace

// __global__ kernels for quantize
namespace {

template <typename scalar_t>
__global__ void batch_aware_quantize_row_wise_kernel(
    PackedTensorAccessor32<int8_t, 2> dst,
    int64_t dst_rows,
    int64_t dst_cols,
    PackedTensorAccessor32<scalar_t, 2> src,
    int64_t src_rows,
    int64_t src_cols,
    PackedTensorAccessor32<scalar_t, 1> min_accessor,
    PackedTensorAccessor32<scalar_t, 1> max_accessor,
    PackedTensorAccessor32<scalar_t, 1> sum_accessor,
    PackedTensorAccessor32<double, 1> scales,
    PackedTensorAccessor32<int32_t, 1> zero_points,
    PackedTensorAccessor32<int32_t, 1> precomputed_sums) {
  int64_t row;
  int64_t col;

  double scale;
  int32_t zero_point;

  float x_min;
  float x_max;
  float inv_scale;
  float transformed_val;

  constexpr int precision = 8;
  constexpr int q_min = ((1 << (precision - 1)) * -1);
  constexpr int q_max = ((1 << (precision - 1)) - 1);

  for (row = blockIdx.x; row < src_rows; row += gridDim.x) {
    x_min = min_accessor[row];
    x_max = max_accessor[row];
    choose_quantization_params_device(
        x_min, x_max, q_min, q_max, &scale, &zero_point);
    inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) {
      scales[row] = scale;
      zero_points[row] = zero_point;
      precomputed_sums[row] = std::nearbyint(sum_accessor[row] * inv_scale);
    }

    for (col = threadIdx.x; col < src_cols; col += blockDim.x) {
      transformed_val = src[row][col] * inv_scale;
      transformed_val = std::nearbyint(transformed_val) + zero_point;
      dst[row][col] = clamp_device<double, int8_t>(
          transformed_val, precision, /* is signed */ true);
    }

    for (; col < dst_cols; col += blockDim.x) {
      dst[row][col] = 0;
    }
  }

  for (; row < dst_rows; row += gridDim.x) {
    for (col = threadIdx.x; col < dst_cols; col += blockDim.x) {
      dst[row][col] = 0;
    }
  }
}

template <typename scalar_t>
__global__ void batch_aware_quantize_col_wise_kernel(
    PackedTensorAccessor32<int8_t, 2> dst,
    int64_t dst_rows,
    int64_t dst_cols,
    PackedTensorAccessor32<scalar_t, 2> src,
    int64_t src_rows,
    int64_t src_cols,
    PackedTensorAccessor32<scalar_t, 1> min_accessor,
    PackedTensorAccessor32<scalar_t, 1> max_accessor,
    PackedTensorAccessor32<scalar_t, 1> sum_accessor,
    PackedTensorAccessor32<double, 1> scales,
    PackedTensorAccessor32<int32_t, 1> zero_points,
    PackedTensorAccessor32<int32_t, 1> precomputed_sums) {
  int64_t row;
  int64_t col;

  double scale;
  int32_t zero_point;

  float x_min;
  float x_max;
  float inv_scale;
  float transformed_val;

  constexpr int precision = 8;
  constexpr int q_min = ((1 << (precision - 1)) * -1);
  constexpr int q_max = ((1 << (precision - 1)) - 1);

  for (col = blockIdx.x; col < src_cols; col += gridDim.x) {
    x_min = min_accessor[col];
    x_max = max_accessor[col];
    choose_quantization_params_device(
        x_min, x_max, q_min, q_max, &scale, &zero_point);
    inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) {
      scales[col] = scale;
      zero_points[col] = zero_point;
      precomputed_sums[col] = std::nearbyint(sum_accessor[col] * inv_scale) +
          (zero_point * src_rows);
    }

    for (row = threadIdx.x; row < src_rows; row += blockDim.x) {
      transformed_val = src[row][col] * inv_scale;
      transformed_val = std::nearbyint(transformed_val) + zero_point;
      dst[row][col] =
          clamp_device<double, int8_t>(transformed_val, precision, true);
    }

    for (; row < dst_rows; row += blockDim.x) {
      dst[row][col] = 0;
    }
  }

  for (; col < dst_cols; col += gridDim.x) {
    for (row = threadIdx.x; row < dst_rows; row += blockDim.x) {
      dst[row][col] = 0;
    }
  }
}

template <typename scalar_t>
__global__ void batch_aware_quantize_row_wise_transpose_kernel(
    PackedTensorAccessor32<int8_t, 2> dst,
    int64_t dst_rows,
    int64_t dst_cols,
    PackedTensorAccessor32<scalar_t, 2> src,
    int64_t src_rows,
    int64_t src_cols,
    PackedTensorAccessor32<scalar_t, 1> min_accessor,
    PackedTensorAccessor32<scalar_t, 1> max_accessor,
    PackedTensorAccessor32<scalar_t, 1> sum_accessor,
    PackedTensorAccessor32<double, 1> scales,
    PackedTensorAccessor32<int32_t, 1> zero_points,
    PackedTensorAccessor32<int32_t, 1> precomputed_sums) {
  int64_t row;
  int64_t col;

  double scale;
  int32_t zero_point;

  float x_min;
  float x_max;
  float inv_scale;
  float transformed_val;

  constexpr int precision = 8;
  constexpr int q_min = ((1 << (precision - 1)) * -1);
  constexpr int q_max = ((1 << (precision - 1)) - 1);

  for (row = blockIdx.x; row < src_rows; row += gridDim.x) {
    x_min = min_accessor[row];
    x_max = max_accessor[row];
    choose_quantization_params_device(
        x_min, x_max, q_min, q_max, &scale, &zero_point);
    inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) {
      scales[row] = scale;
      zero_points[row] = zero_point;
      precomputed_sums[row] = std::nearbyint(sum_accessor[row] * inv_scale);
    }

    for (col = threadIdx.x; col < src_cols; col += blockDim.x) {
      transformed_val = src[row][col] * inv_scale;
      transformed_val = std::nearbyint(transformed_val) + zero_point;
      dst[col][row] = clamp_device<double, int8_t>(
          transformed_val, precision, /* is signed */ true);
    }

    for (; col < dst_rows; col += blockDim.x) {
      dst[col][row] = 0;
    }
  }

  for (; row < dst_cols; row += gridDim.x) {
    for (col = threadIdx.x; col < dst_rows; col += blockDim.x) {
      dst[col][row] = 0;
    }
  }
}

template <typename scalar_t>
__global__ void batch_aware_quantize_col_wise_transpose_kernel(
    PackedTensorAccessor32<int8_t, 2> dst,
    int64_t dst_rows,
    int64_t dst_cols,
    PackedTensorAccessor32<scalar_t, 2> src,
    int64_t src_rows,
    int64_t src_cols,
    PackedTensorAccessor32<scalar_t, 1> min_accessor,
    PackedTensorAccessor32<scalar_t, 1> max_accessor,
    PackedTensorAccessor32<scalar_t, 1> sum_accessor,
    PackedTensorAccessor32<double, 1> scales,
    PackedTensorAccessor32<int32_t, 1> zero_points,
    PackedTensorAccessor32<int32_t, 1> precomputed_sums) {
  int64_t row;
  int64_t col;

  double scale;
  int32_t zero_point;

  float x_min;
  float x_max;
  float inv_scale;
  float transformed_val;

  constexpr int precision = 8;
  constexpr int q_min = ((1 << (precision - 1)) * -1);
  constexpr int q_max = ((1 << (precision - 1)) - 1);

  for (col = blockIdx.x; col < src_cols; col += gridDim.x) {
    x_min = min_accessor[col];
    x_max = max_accessor[col];
    choose_quantization_params_device(
        x_min, x_max, q_min, q_max, &scale, &zero_point);
    inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) {
      scales[col] = scale;
      zero_points[col] = zero_point;
      precomputed_sums[col] = std::nearbyint(sum_accessor[col] * inv_scale) +
          (zero_point * src_rows);
    }

    for (row = threadIdx.x; row < src_rows; row += blockDim.x) {
      transformed_val = src[row][col] * inv_scale;
      transformed_val = std::nearbyint(transformed_val) + zero_point;
      dst[col][row] =
          clamp_device<double, int8_t>(transformed_val, precision, true);
    }

    for (; row < dst_cols; row += blockDim.x) {
      dst[col][row] = 0;
    }
  }

  for (; col < dst_rows; col += gridDim.x) {
    for (row = threadIdx.x; row < dst_cols; row += blockDim.x) {
      dst[col][row] = 0;
    }
  }
}

} // namespace

Tensor qntz_out_cuda(
    Tensor& out,
    const Tensor& input,
    Tensor& scales,
    Tensor& zeros,
    Tensor& sums,
    bool rowwise,
    bool transpose) {
  cuda_check(out);
  auto device_guard(out.get_device());
  qntz_out_check(out, input, scales, zeros, sums, rowwise, transpose);

  IntArrayRef dst_sizes = out.sizes();
  IntArrayRef src_sizes = input.sizes();
  IntArrayRef scales_sizes = scales.sizes();
  IntArrayRef zero_points_sizes = zeros.sizes();
  IntArrayRef precomputed_sums_sizes = sums.sizes();

  // dst_sizes must be at least as large as src_sizes if transpose=false,
  // or the transposed of src_sizes if transpose=true. any index in dst not
  // corresponding to an index in src is considered padding and thus zeroed
  TORCH_CHECK(
      (dst_sizes[transpose ? 1 : 0] >= src_sizes[0]) &&
          (dst_sizes[transpose ? 0 : 1] >= src_sizes[1]),
      transpose ? "dst must be at least as large as src transposed"
                : "dst must be at least as large as src");

  // parameter tensors must be as large as the size of batch elements,
  // in the case of row-wise batching this is the number of columns,
  // and in the case of col-wise batching this is the number of rows
  TORCH_CHECK(
      (scales_sizes[0] == src_sizes[rowwise ? 0 : 1]),
      rowwise ? "scales size must match the number of rows in src"
              : "scales size must match the number of cols in src");
  TORCH_CHECK(
      (zero_points_sizes[0] == src_sizes[rowwise ? 0 : 1]),
      rowwise ? "zero_points size must match the number of rows in src"
              : "zero_points size must match the number of cols in src");
  TORCH_CHECK(
      (precomputed_sums_sizes[0] == src_sizes[rowwise ? 0 : 1]),
      rowwise ? "precomputed_sums size must match the number of rows in src"
              : "precomputed_sums size must match the number of cols in src");

  int64_t blocks;
  int64_t threads;

  if (rowwise) {
    // blocks assigned to rows, threads assigned to columns
    blocks = std::min(
        (src_sizes[0] + MIN_BLOCK_WORK - 1) / MIN_BLOCK_WORK,
        static_cast<int64_t>(MAX_BLOCKS));
    threads = std::min(
        (src_sizes[1] + MIN_THREAD_WORK - 1) / MIN_THREAD_WORK,
        static_cast<int64_t>(MAX_THREADS));
  } else {
    // blocks assigned to columns, threads assigned to rows
    blocks = std::min(
        (src_sizes[1] + MIN_BLOCK_WORK - 1) / MIN_BLOCK_WORK,
        static_cast<int64_t>(MAX_BLOCKS));
    threads = std::min(
        (src_sizes[0] + MIN_THREAD_WORK - 1) / MIN_THREAD_WORK,
        static_cast<int64_t>(MAX_THREADS));
  }

  auto min_max_tensors =
      aminmax(input, rowwise ? 1 : 0, /* no preserve dimension */ false);

  Tensor mins = std::get<0>(min_max_tensors);
  Tensor maxs = std::get<1>(min_max_tensors);
  Tensor _sums =
      sum(input, transpose ? 0 : 1, /* no preserve dimension */ false);

  auto dst_accessor = out.packed_accessor32<int8_t, 2>();

  auto scales_accessor = scales.packed_accessor32<double, 1>();
  auto zero_points_accessor = zeros.packed_accessor32<int32_t, 1>();
  auto precomputed_sums_accessor = sums.packed_accessor32<int32_t, 1>();

  if (input.dtype() == kHalf) {
    CALL_QUANTIZE_KERNEL(Half);
  } else if (input.dtype() == kFloat) {
    CALL_QUANTIZE_KERNEL(float);
  } else if (input.dtype() == kDouble) {
    CALL_QUANTIZE_KERNEL(double);
  } else if (input.dtype() == kBFloat16) {
    CALL_QUANTIZE_KERNEL(BFloat16);
  } else {
    TORCH_CHECK(
        false,
        "input type must be one of kHalf, kFloat, kBFloat16, or kDouble");
  }

  return out;
}

TORCH_LIBRARY_FRAGMENT(protoquant, m) {
  m.def(
      "qntz_out(Tensor(a!) out, Tensor input, Tensor(a!) scales, Tensor(a!) zeros, Tensor(a!) sums, bool rowwise, bool transpose) -> Tensor");
}

TORCH_LIBRARY_IMPL(protoquant, CUDA, m) {
  m.impl("qntz_out", qntz_out_cuda);
}

} // namespace protoquant
} // namespace at
