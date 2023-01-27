#pragma once

#include <ATen/ATen.h>
#include <cutlass/cutlass.h>

namespace at {
namespace fb {

int get_architecture();

void get_padding(
    const at::ScalarType& mat1_type,
    const at::ScalarType& mat2_type,
    const at::ScalarType& mat3_type,
    int device,
    int* m_padding,
    int* n_padding,
    int* k_padding);

void get_transpose(
    const at::ScalarType& mat1_type,
    const at::ScalarType& mat2_type,
    const at::ScalarType& mat3_type,
    int device,
    bool* transpose_mat1,
    bool* transpose_mat2,
    bool* transpose_mat3);

void check_cutlass_status(cutlass::Status status);

bool is_supported(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    at::ScalarType* r_type);

} // namespace fb
} // namespace at
