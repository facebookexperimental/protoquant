#include <ATen/ATen.h>
#include <torch/library.h>

namespace at {
namespace protoquant {

Tensor dqntz_out_cpu(
    Tensor& out,
    const Tensor& input,
    const Tensor& scales,
    const Tensor& zeros,
    const Tensor& sums,
    const bool rowwise,
    const bool transpose) {
  return out;
}

Tensor dqntz_mm_out_cpu(
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
  return out;
}

Tensor dqntz_mm_add_out_cpu(
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
  return out;
}

TORCH_LIBRARY_FRAGMENT(protoquant, m) {
  m.def(
      "dqntz_out(Tensor out, Tensor input, Tensor scales, Tensor zeros, Tensor sums, bool rowwise, bool transpose) -> Tensor");
  m.def(
      "dqntz_mm_out(Tensor out, Tensor input, Tensor mat1_scales, Tensor mat1_zeros, Tensor mat1_sums, bool mat1_rowwise, bool mat1_transpose, Tensor mat2_scales, Tensor mat2_zeros, Tensor mat2_sums, bool mat2_rowwise, bool mat2_transpose) -> Tensor");
  m.def(
      "dqntz_mm_add_out(Tensor out, Tensor input, Tensor other, Tensor mat1_scales, Tensor mat1_zeros, Tensor mat1_sums, bool mat1_rowwise, bool mat1_transpose, Tensor mat2_scales, Tensor mat2_zeros, Tensor mat2_sums, bool mat2_rowwise, bool mat2_transpose) -> Tensor");
}

TORCH_LIBRARY_IMPL(protoquant, CPU, m) {
  m.impl("dqntz_out", dqntz_out_cpu);
  m.impl("dqntz_mm_out", dqntz_mm_out_cpu);
  m.impl("dqntz_mm_add_out", dqntz_mm_add_out_cpu);
}

} // namespace protoquant
} // namespace at
