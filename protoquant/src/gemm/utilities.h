#pragma once

#include <ATen/ATen.h>
#include "c10/core/ScalarType.h"

namespace at {
namespace fb {

int64_t pad(int64_t x);

void same_device_check(const Tensor& a, const Tensor& b);

void contiguous_check(const Tensor& a);

void dtype_check(const Tensor& a, const ScalarType dtype);

void dim_check(const Tensor& a, int64_t dim);

void sizes_check(const Tensor& input, const Tensor& mat2);

void sizes_check(const Tensor& out, const Tensor& input, const Tensor& mat2);

void cuda_check(const Tensor& a);

void ngemm_check(const Tensor& input, const Tensor& mat2);

void ngemm_out_check(
    const Tensor& out,
    const Tensor& input,
    const Tensor& mat2);

void gemm_check(const Tensor& input, const Tensor& mat1, const Tensor& mat2);

void gemm_out_check(const Tensor& out, const Tensor& mat1, const Tensor& mat2);

} // namespace fb
} // namespace at
