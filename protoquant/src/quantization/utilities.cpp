#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#define PADDING 8

namespace at {
namespace protoquant {

int64_t pad(int64_t x) {
  return (x % PADDING) ? x + PADDING - (x % PADDING) : x;
}

void same_device_check(const Tensor& a, const Tensor& b) {
  TORCH_CHECK(
      a.device() == b.device(),
      "Expected all tensors to be on the same device, but found at least two devices, ",
      a.device(),
      " and ",
      b.device(),
      "!");
}

void contiguous_check(const Tensor& a) {
  TORCH_CHECK(a.is_contiguous(), "Expected ", a, " to be contiguous!");
}

void dtype_check(const Tensor& a, const ScalarType dtype) {
  TORCH_CHECK(
      (a.dtype() == dtype),
      "Expected ",
      a,
      " to have dtype ",
      dtype,
      ", but found ",
      a.dtype(),
      "!");
}

void float_check(const Tensor& a) {
  TORCH_CHECK(
      (a.dtype() == kHalf) || (a.dtype() == kFloat) || (a.dtype() == kDouble),
      "Expected ",
      a,
      " to have float dtype, but found ",
      a.dtype(),
      "!");
}

void dim_check(const Tensor& a, int64_t dim) {
  TORCH_CHECK(
      (a.dim() == dim),
      "Expected ",
      a,
      " to have dim of ",
      dim,
      ", but found dim of ",
      a.dim(),
      "!");
}

void sizes_check(const Tensor& input, const Tensor& mat2) {
  int64_t m = input.sizes()[0];
  int64_t k = input.sizes()[1];
  int64_t n = mat2.sizes()[0];

  TORCH_CHECK(
      (m == pad(m)) && (k == pad(k)) && (n == pad(n)),
      "Expected dimensions to be padded to ",
      PADDING,
      ", but found at least one unpadded dimension!");
  TORCH_CHECK(
      (mat2.sizes()[1] == k),
      "Expected mat2 to have sizes {",
      n,
      ", ",
      k,
      "}, but found ",
      mat2.sizes(),
      "!");
}

void sizes_check(const Tensor& out, const Tensor& input, const Tensor& mat2) {
  int64_t m = input.sizes()[0];
  int64_t k = input.sizes()[1];
  int64_t n = mat2.sizes()[0];

  TORCH_CHECK(
      (m == pad(m)) && (k == pad(k)) && (n == pad(n)),
      "Expected dimensions to be padded to ",
      PADDING,
      ", but found at least one unpadded dimension!");
  TORCH_CHECK(
      (mat2.sizes()[1] == k),
      "Expected mat2 to have sizes {",
      n,
      ", ",
      k,
      "}, but found ",
      mat2.sizes(),
      "!");
  TORCH_CHECK(
      (out.sizes()[0] == m) && (out.sizes()[1] == n),
      "Expected out to have sizes {",
      m,
      ", ",
      n,
      "}, but found ",
      out.sizes(),
      "!");
}

void sizes_geq_check(
    const Tensor& out,
    const Tensor& input,
    const bool transpose) {
  int64_t m_out = out.sizes()[0];
  int64_t n_out = out.sizes()[1];

  int64_t m = transpose ? input.sizes()[1] : input.sizes()[0];
  int64_t n = transpose ? input.sizes()[0] : input.sizes()[1];

  TORCH_CHECK(
      (m_out >= m) && (n_out >= n),
      "Expected out to be at least as large as input, but found at least one smaller dimension!");
}

void sizes_leq_check(
    const Tensor& out,
    const Tensor& input,
    const bool transpose) {
  int64_t m_out = out.sizes()[0];
  int64_t n_out = out.sizes()[1];

  int64_t m = transpose ? input.sizes()[1] : input.sizes()[0];
  int64_t n = transpose ? input.sizes()[0] : input.sizes()[1];

  TORCH_CHECK(
      (m_out <= m) && (n_out <= n),
      "Expected input to be at least as large as out, but found at least one smaller dimension!");
}

void params_check(
    const Tensor& out,
    const Tensor& scales,
    const Tensor& zeros,
    const Tensor& sums,
    const bool rowwise) {
  int64_t m = out.sizes()[0];
  int64_t n = out.sizes()[1];

  int64_t batch = rowwise ? m : n;

  TORCH_CHECK(
      (batch == scales.sizes()[0]),
      "Expected scales length to be equal to the length of the quantized dimension, ",
      batch,
      " but found ",
      scales.sizes()[0],
      "!");
  TORCH_CHECK(
      (batch == zeros.sizes()[0]),
      "Expected zeros length to be equal to the length of the quantized dimension, ",
      batch,
      " but found ",
      zeros.sizes()[0],
      "!");
  TORCH_CHECK(
      (batch == sums.sizes()[0]),
      "Expected sums length to be equal to the length of the quantized dimension, ",
      batch,
      " but found ",
      sums.sizes()[0],
      "!");
}

void cpu_check(const Tensor& a) {
  TORCH_CHECK(
      a.is_cpu(),
      "Expected tensor ",
      a,
      " to use CPU, but found ",
      a.device(),
      "!");
}

void cuda_check(const Tensor& a) {
  TORCH_CHECK(
      a.is_cuda(),
      "Expected tensor ",
      a,
      " to use CUDA device, but found ",
      a.device(),
      "!");
}

void qntz_out_check(
    const Tensor& out,
    const Tensor& input,
    const Tensor& scales,
    const Tensor& zeros,
    const Tensor& sums,
    const bool rowwise,
    const bool transpose) {
  same_device_check(out, input);
  same_device_check(out, scales);
  same_device_check(out, zeros);
  same_device_check(out, sums);
  contiguous_check(out);
  contiguous_check(input);
  contiguous_check(scales);
  contiguous_check(zeros);
  contiguous_check(sums);
  dtype_check(out, kChar);
  float_check(input);
  dtype_check(scales, kDouble);
  dtype_check(zeros, kInt);
  dtype_check(sums, kInt);
  dim_check(out, 2);
  dim_check(input, 2);
  dim_check(scales, 1);
  dim_check(zeros, 1);
  dim_check(sums, 1);
  sizes_geq_check(out, input, transpose);
  params_check(input, scales, zeros, sums, rowwise);
}

void dqntz_out_check(
    const Tensor& out,
    const Tensor& input,
    const Tensor& scales,
    const Tensor& zeros,
    const Tensor& sums,
    const bool rowwise,
    const bool transpose) {
  same_device_check(out, input);
  same_device_check(out, scales);
  same_device_check(out, zeros);
  same_device_check(out, sums);
  contiguous_check(out);
  contiguous_check(input);
  contiguous_check(scales);
  contiguous_check(zeros);
  contiguous_check(sums);
  float_check(out);
  dtype_check(input, kChar);
  dtype_check(scales, kDouble);
  dtype_check(zeros, kInt);
  dtype_check(sums, kInt);
  dim_check(out, 2);
  dim_check(input, 2);
  dim_check(scales, 1);
  dim_check(zeros, 1);
  dim_check(sums, 1);
  sizes_leq_check(out, input, transpose);
  params_check(out, scales, zeros, sums, rowwise);
}

void dqntz_mm_out_check(
    const Tensor& out,
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
  same_device_check(out, input);
  same_device_check(out, mat1_scales);
  same_device_check(out, mat1_zeros);
  same_device_check(out, mat1_sums);
  same_device_check(out, mat2_scales);
  same_device_check(out, mat2_zeros);
  same_device_check(out, mat2_sums);
  contiguous_check(out);
  contiguous_check(input);
  contiguous_check(mat1_scales);
  contiguous_check(mat1_zeros);
  contiguous_check(mat1_sums);
  contiguous_check(mat2_scales);
  contiguous_check(mat2_zeros);
  contiguous_check(mat2_sums);
  float_check(out);
  dtype_check(input, kInt);
  dtype_check(mat1_scales, kDouble);
  dtype_check(mat1_zeros, kInt);
  dtype_check(mat1_sums, kInt);
  dtype_check(mat2_scales, kDouble);
  dtype_check(mat2_zeros, kInt);
  dtype_check(mat2_sums, kInt);
  dim_check(out, 2);
  dim_check(input, 2);
  dim_check(mat1_scales, 1);
  dim_check(mat1_zeros, 1);
  dim_check(mat1_sums, 1);
  dim_check(mat2_scales, 1);
  dim_check(mat2_zeros, 1);
  dim_check(mat2_sums, 1);
  sizes_leq_check(out, input, false);
  params_check(out, mat1_scales, mat1_zeros, mat1_sums, mat1_rowwise);
  params_check(out, mat2_scales, mat2_zeros, mat2_sums, mat2_rowwise);

  TORCH_CHECK(
      (mat1_rowwise && !mat1_transpose) && (!mat2_rowwise && mat2_transpose),
      "Expected mat1 to be quantized rowwise, non-transposed and mat2 to be quantized colwise, transposed!");
}

} // namespace protoquant
} // namespace at
