#include <protoquant/src/gemm/utilities.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace at {
namespace fb {

Tensor gemm(
    Tensor& input,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  auto m = input.sizes()[0];
  auto n = input.sizes()[1];
  auto k = mat1.sizes()[1];

  auto _beta = beta.to<int64_t>();
  auto _alpha = alpha.to<int64_t>();

  TORCH_CHECK(
      (_beta == 0) || (_beta == 1),
      "Expected beta to be 0 or 1, but found ",
      _beta,
      "!");
  TORCH_CHECK(
      (_alpha == 0) || (_alpha == 1),
      "Expected alpha to be 0 or 1, but found ",
      _alpha,
      "!");

  auto beta_ptr = (void*)&_beta;
  auto alpha_ptr = (void*)&_alpha;

  auto input_ptr = (void*)input.data_ptr<int32_t>();
  auto mat1_ptr = (void*)mat1.data_ptr<int8_t>();
  auto mat2_ptr = (void*)mat2.data_ptr<int8_t>();

  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      at::cuda::getCurrentCUDABlasHandle(),
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      alpha_ptr,
      mat2_ptr,
      CUDA_R_8I,
      k,
      mat1_ptr,
      CUDA_R_8I,
      k,
      beta_ptr,
      input_ptr,
      CUDA_R_32I,
      n,
      CUBLAS_COMPUTE_32I,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  return input;
}

Tensor gemm_cuda(Tensor& input, const Tensor& mat1, const Tensor& mat2) {
  cuda_check(input);
  // auto device_guard(input.get_device());
  gemm_check(input, mat1, mat2);

  auto beta = 1;
  auto alpha = 1;

  return gemm(input, mat1, mat2, beta, alpha);
}

Tensor gemm_out_cuda(Tensor& out, const Tensor& mat1, const Tensor& mat2) {
  cuda_check(out);
  // auto device_guard(out.get_device());
  gemm_out_check(out, mat1, mat2);

  auto beta = 0;
  auto alpha = 1;

  return gemm(out, mat1, mat2, beta, alpha);
}

Tensor ngemm_cuda(const Tensor& input, const Tensor& mat2) {
  cuda_check(input);
  auto device_guard(input.get_device());
  ngemm_check(input, mat2);

  auto m = input.sizes()[0];
  auto n = mat2.sizes()[0];

  auto out = empty({m, n}, input.options().dtype(kInt));

  auto beta = 0;
  auto alpha = 1;

  return gemm(out, input, mat2, beta, alpha);
}

Tensor ngemm_out_cuda(Tensor& out, const Tensor& input, const Tensor& mat2) {
  cuda_check(out);
  auto device_guard(out.get_device());
  ngemm_out_check(out, input, mat2);

  auto m = out.sizes()[0];
  auto n = out.sizes()[1];

  auto beta = 0;
  auto alpha = 1;

  return gemm(out, input, mat2, beta, alpha);
}

TORCH_LIBRARY_IMPL(protoquant, CUDA, m) {
  m.impl("gemm", gemm_cuda);
  m.impl("gemm_out", gemm_out_cuda);
  m.impl("ngemm", ngemm_cuda);
  m.impl("ngemm_out", ngemm_out_cuda);
}

} // namespace fb
} // namespace at
