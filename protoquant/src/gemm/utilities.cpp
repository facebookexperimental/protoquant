#include <ATen/ATen.h>
#include "c10/core/ScalarType.h"

#define PADDING 8

namespace at {
namespace fb {

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

void cuda_check(const Tensor& a) {
  TORCH_CHECK(
      a.is_cuda(),
      "Expected tensor ",
      a,
      " to use CUDA device, but found ",
      a.device(),
      "!");
}

void ngemm_check(const Tensor& input, const Tensor& mat2) {
  same_device_check(input, mat2);
  contiguous_check(input);
  contiguous_check(mat2);
  dtype_check(input, kChar);
  dtype_check(mat2, kChar);
  dim_check(input, 2);
  dim_check(mat2, 2);
  sizes_check(input, mat2);
}

void ngemm_out_check(
    const Tensor& out,
    const Tensor& input,
    const Tensor& mat2) {
  same_device_check(out, input);
  same_device_check(out, mat2);
  contiguous_check(out);
  contiguous_check(input);
  contiguous_check(mat2);
  dtype_check(out, kInt);
  dtype_check(input, kChar);
  dtype_check(mat2, kChar);
  dim_check(out, 2);
  dim_check(input, 2);
  dim_check(mat2, 2);
  sizes_check(out, input, mat2);
}

void gemm_check(const Tensor& input, const Tensor& mat1, const Tensor& mat2) {
  same_device_check(input, mat1);
  same_device_check(input, mat2);
  contiguous_check(input);
  contiguous_check(mat1);
  contiguous_check(mat2);
  dtype_check(input, kInt);
  dtype_check(mat1, kChar);
  dtype_check(mat2, kChar);
  dim_check(input, 2);
  dim_check(mat1, 2);
  dim_check(mat2, 2);
  sizes_check(input, mat1, mat2);
}

void gemm_out_check(const Tensor& out, const Tensor& mat1, const Tensor& mat2) {
  same_device_check(out, mat1);
  same_device_check(out, mat2);
  contiguous_check(out);
  contiguous_check(mat1);
  contiguous_check(mat2);
  dtype_check(out, kInt);
  dtype_check(mat1, kChar);
  dtype_check(mat2, kChar);
  dim_check(out, 2);
  dim_check(mat1, 2);
  dim_check(mat2, 2);
  sizes_check(out, mat1, mat2);
}

} // namespace fb
} // namespace at
