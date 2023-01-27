#pragma once

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

// Sanitizers annotations
#if defined(__has_attribute)
#if __has_attribute(no_sanitize)
#define NO_SANITIZE(what) __attribute__((no_sanitize(what)))
#endif
#endif
#if !defined(NO_SANITIZE)
#define NO_SANITIZE(what)
#endif

namespace at {
namespace protoquant {

void same_device_check(const Tensor& a, const Tensor& b);

void contiguous_check(const Tensor& a);

void dtype_check(const Tensor& a, const ScalarType dtype);

void float_check(const Tensor& a);

void dim_check(const Tensor& a, int64_t dim);

void sizes_check(const Tensor& input, const Tensor& mat2);

void sizes_check(const Tensor& out, const Tensor& input, const Tensor& mat2);

void sizes_leq_check(
    const Tensor& out,
    const Tensor& input,
    const bool transpose);

void params_check(
    const Tensor& out,
    const Tensor& scales,
    const Tensor& zeros,
    const Tensor& sums,
    const bool rowwise);

void cpu_check(const Tensor& a);

void cuda_check(const Tensor& a);

void qntz_out_check(
    const Tensor& out,
    const Tensor& input,
    const Tensor& scales,
    const Tensor& zeros,
    const Tensor& sums,
    const bool rowwise,
    const bool transpose);

void dqntz_out_check(
    const Tensor& out,
    const Tensor& input,
    const Tensor& scales,
    const Tensor& zeros,
    const Tensor& sums,
    const bool rowwise,
    const bool transpose);

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
    const bool mat2_transpose);

} // namespace protoquant
} // namespace at
