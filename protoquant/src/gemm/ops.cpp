#include <protoquant/src/gemm/gemm.h>
#include <protoquant/src/gemm/utilities.h>

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/DispatchKey.h>
#include <torch/library.h>

namespace at {
namespace fb {

Tensor gemm_cpu(Tensor& input, const Tensor& mat1, const Tensor& mat2) {
  gemm_check(input, mat1, mat2);
  return addmm(input, mat1.to(kInt), mat2.to(kInt).transpose(0, 1));
}

Tensor gemm_out_cpu(Tensor& out, const Tensor& mat1, const Tensor& mat2) {
  gemm_out_check(out, mat1, mat2);
  return mm_out(out, mat1.to(kInt), mat2.to(kInt).transpose(0, 1));
}

Tensor ngemm_cpu(const Tensor& input, const Tensor& mat2) {
  ngemm_check(input, mat2);
  return mm(input.to(kInt), mat2.to(kInt).transpose(0, 1));
}

Tensor ngemm_out_cpu(Tensor& out, const Tensor& input, const Tensor& mat2) {
  ngemm_out_check(out, input, mat2);
  return mm_out(out, input.to(kInt), mat2.to(kInt).transpose(0, 1));
}

class PadKernel final : public c10::OperatorKernel {
 public:
  int64_t operator()(const int64_t x) {
    return pad(x);
  }
};

static auto pad_registry = c10::RegisterOperators().op(
    c10::RegisterOperators::options().schema("fb::pad").kernel<PadKernel>(
        c10::DispatchKey::CatchAll));

TORCH_LIBRARY(protoquant, m) {
  m.def("gemm(Tensor input, Tensor mat1, Tensor mat2) -> Tensor");
  m.def("gemm_out(Tensor out, Tensor mat1, Tensor mat2) -> Tensor");
  m.def("ngemm(Tensor out, Tensor mat2) -> Tensor");
  m.def("ngemm_out(Tensor(a!) out, Tensor input, Tensor mat2) -> Tensor");
  m.def("_triton_gemm(Tensor mat1, Tensor mat2) -> Tensor");
}
TORCH_LIBRARY_IMPL(protoquant, CPU, m) {
  m.impl("gemm", gemm_cpu);
  m.impl("gemm_out", gemm_out_cpu);
  m.impl("ngemm", ngemm_cpu);
  m.impl("ngemm_out", ngemm_out_cpu);
}
} // namespace fb
} // namespace at
