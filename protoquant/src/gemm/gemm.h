#pragma once

#include <ATen/ATen.h>

namespace at {
namespace fb {

Tensor gemm(
    Tensor& input,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha);

} // namespace fb
} // namespace at
