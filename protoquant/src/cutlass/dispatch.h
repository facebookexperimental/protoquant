#pragma once

#include <protoquant/src/cutlass/templates.h>
#include <protoquant/src/cutlass/utilities.h>

namespace at {
namespace fb {

namespace {

template <typename architecture>
void dispatch_impl(
    int m,
    int n,
    int k,
    const at::Tensor& a,
    int lda,
    const at::Tensor& b,
    int ldb,
    const at::Tensor& c,
    int ldc,
    at::Tensor& d,
    int ldd,
    const float alpha,
    const float beta) {
  at::ScalarType a_type = a.scalar_type();
  at::ScalarType b_type = b.scalar_type();
  at::ScalarType c_type = c.scalar_type();
  at::ScalarType d_type = d.scalar_type();

  if ((a_type == at::kHalf) && (b_type == at::kHalf) && (c_type == at::kHalf) &&
      (d_type == at::kHalf)) {
    // fp16@fp16 + fp16 => fp16
    const Half* a_ptr = const_cast<Half*>(a.data_ptr<Half>());
    const Half* b_ptr = const_cast<Half*>(b.data_ptr<Half>());
    const Half* c_ptr = const_cast<Half*>(c.data_ptr<Half>());
    Half* d_ptr = d.data_ptr<Half>();
    gemm<architecture>(
        m, n, k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd, alpha, beta);
  } else if (
      (a_type == at::kFloat) && (b_type == at::kFloat) &&
      (c_type == at::kFloat) && (d_type == at::kFloat)) {
    // fp32@fp32 + fp32 => fp32
    const float* a_ptr = const_cast<float*>(a.data_ptr<float>());
    const float* b_ptr = const_cast<float*>(b.data_ptr<float>());
    const float* c_ptr = const_cast<float*>(c.data_ptr<float>());
    float* d_ptr = d.data_ptr<float>();
    gemm<architecture>(
        m, n, k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd, alpha, beta);
  } else if (
      (a_type == at::kDouble) && (b_type == at::kDouble) &&
      (c_type == at::kDouble) && (d_type == at::kDouble)) {
    // fp64@fp64 + fp64 => fp64
    const double* a_ptr = const_cast<double*>(a.data_ptr<double>());
    const double* b_ptr = const_cast<double*>(b.data_ptr<double>());
    const double* c_ptr = const_cast<double*>(c.data_ptr<double>());
    double* d_ptr = d.data_ptr<double>();
    gemm<architecture>(
        m, n, k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd, alpha, beta);
  } else if (
      (a_type == at::kChar) && (b_type == at::kChar) && (c_type == at::kInt) &&
      (d_type == at::kInt)) {
    // s8@s8 + s32 => s32
    const int8_t* a_ptr = const_cast<int8_t*>(a.data_ptr<int8_t>());
    const int8_t* b_ptr = const_cast<int8_t*>(b.data_ptr<int8_t>());
    const int32_t* c_ptr = const_cast<int32_t*>(c.data_ptr<int32_t>());
    int32_t* d_ptr = d.data_ptr<int32_t>();
    gemm<architecture>(
        m, n, k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd, alpha, beta);
  } else {
    AT_ERROR("not supported on any architecture");
  }
}

} // namespace

void dispatch(
    int m,
    int n,
    int k,
    const at::Tensor& a,
    int lda,
    const at::Tensor& b,
    int ldb,
    const at::Tensor& c,
    int ldc,
    at::Tensor& d,
    int ldd,
    const at::Scalar alpha,
    const at::Scalar beta) {
  const float alpha_val = alpha.to<float>();
  const float beta_val = beta.to<float>();
  int device = d.get_device();
  switch (get_architecture()) {
    case 60:
      dispatch_impl<cutlass::arch::Sm60>(
          m, n, k, a, lda, b, ldb, c, ldc, d, ldd, alpha_val, beta_val);
      break;
    case 61:
      dispatch_impl<cutlass::arch::Sm61>(
          m, n, k, a, lda, b, ldb, c, ldc, d, ldd, alpha_val, beta_val);
      break;
    case 70:
      dispatch_impl<cutlass::arch::Sm70>(
          m, n, k, a, lda, b, ldb, c, ldc, d, ldd, alpha_val, beta_val);
      break;
    case 75:
      dispatch_impl<cutlass::arch::Sm75>(
          m, n, k, a, lda, b, ldb, c, ldc, d, ldd, alpha_val, beta_val);
      break;
    case 80:
      dispatch_impl<cutlass::arch::Sm80>(
          m, n, k, a, lda, b, ldb, c, ldc, d, ldd, alpha_val, beta_val);
      break;
    default:
      AT_ERROR("device architecture is not supported");
      break;
  }
}

} // namespace fb
} // namespace at
