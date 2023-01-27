#include <protoquant/src/cutlass/utilities.h>

#include <ATen/ATen.h>
#include <cutlass/arch/arch.h>

#define COMPUTE_CAPABILITY 80

namespace at {
namespace fb {

namespace {

template <typename architecture>
void get_padding(
    at::ScalarType mat1_type,
    at::ScalarType mat2_type,
    at::ScalarType mat3_type,
    int* m_padding,
    int* n_padding,
    int* k_padding) {
  *m_padding = 1;
  *n_padding = 1;
  *k_padding = 1;
}

template <>
void get_padding<cutlass::arch::Sm70>(
    at::ScalarType mat1_type,
    at::ScalarType mat2_type,
    at::ScalarType mat3_type,
    int* m_padding,
    int* n_padding,
    int* k_padding) {
  if ((mat1_type == at::kHalf) && (mat2_type == at::kHalf) &&
      (mat3_type == at::kHalf)) {
    *m_padding = 128;
    *n_padding = 128;
    *k_padding = 32;
  } else if (
      (mat1_type == at::kChar) && (mat2_type == at::kChar) &&
      (mat3_type == at::kInt)) {
    *m_padding = 4;
    *n_padding = 4;
    *k_padding = 4;
  } else {
    *m_padding = 1;
    *n_padding = 1;
    *k_padding = 1;
  }
}

template <>
void get_padding<cutlass::arch::Sm75>(
    at::ScalarType mat1_type,
    at::ScalarType mat2_type,
    at::ScalarType mat3_type,
    int* m_padding,
    int* n_padding,
    int* k_padding) {
  if ((mat1_type == at::kHalf) && (mat2_type == at::kHalf) &&
      (mat3_type == at::kHalf)) {
    *m_padding = 128;
    *n_padding = 256;
    *k_padding = 32;
  } else if (
      (mat1_type == at::kChar) && (mat2_type == at::kChar) &&
      (mat3_type == at::kInt)) {
    *m_padding = 128;
    *n_padding = 256;
    *k_padding = 64;
  } else {
    *m_padding = 1;
    *n_padding = 1;
    *k_padding = 1;
  }
}

template <>
void get_padding<cutlass::arch::Sm80>(
    at::ScalarType mat1_type,
    at::ScalarType mat2_type,
    at::ScalarType mat3_type,
    int* m_padding,
    int* n_padding,
    int* k_padding) {
  if ((mat1_type == at::kHalf) && (mat2_type == at::kHalf) &&
      (mat3_type == at::kHalf)) {
    *m_padding = 128;
    *n_padding = 256;
    *k_padding = 64;
  } else if (
      (mat1_type == at::kChar) && (mat2_type == at::kChar) &&
      (mat3_type == at::kInt)) {
    *m_padding = 128;
    *n_padding = 256;
    *k_padding = 64;
  } else {
    *m_padding = 1;
    *n_padding = 1;
    *k_padding = 1;
  }
}

template <typename architecture>
void get_transpose(
    at::ScalarType mat1_type,
    at::ScalarType mat2_type,
    at::ScalarType mat3_type,
    bool* transpose_mat1,
    bool* transpose_mat2,
    bool* transpose_mat3) {
  *transpose_mat1 = false;
  *transpose_mat2 = false;
  *transpose_mat3 = false;
}

template <>
void get_transpose<cutlass::arch::Sm61>(
    at::ScalarType mat1_type,
    at::ScalarType mat2_type,
    at::ScalarType mat3_type,
    bool* transpose_mat1,
    bool* transpose_mat2,
    bool* transpose_mat3) {
  if ((mat1_type == at::kChar) && (mat2_type == at::kChar) &&
      (mat3_type == at::kInt)) {
    *transpose_mat1 = false;
    *transpose_mat2 = true;
    *transpose_mat3 = false;
  } else {
    *transpose_mat1 = false;
    *transpose_mat2 = false;
    *transpose_mat3 = false;
  }
}

template <>
void get_transpose<cutlass::arch::Sm70>(
    at::ScalarType mat1_type,
    at::ScalarType mat2_type,
    at::ScalarType mat3_type,
    bool* transpose_mat1,
    bool* transpose_mat2,
    bool* transpose_mat3) {
  if ((mat1_type == at::kChar) && (mat2_type == at::kChar) &&
      (mat3_type == at::kInt)) {
    *transpose_mat1 = false;
    *transpose_mat2 = true;
    *transpose_mat3 = false;
  } else {
    *transpose_mat1 = false;
    *transpose_mat2 = false;
    *transpose_mat3 = false;
  }
}

template <>
void get_transpose<cutlass::arch::Sm75>(
    at::ScalarType mat1_type,
    at::ScalarType mat2_type,
    at::ScalarType mat3_type,
    bool* transpose_mat1,
    bool* transpose_mat2,
    bool* transpose_mat3) {
  if ((mat1_type == at::kChar) && (mat2_type == at::kChar) &&
      (mat3_type == at::kInt)) {
    *transpose_mat1 = false;
    *transpose_mat2 = true;
    *transpose_mat3 = false;
  } else {
    *transpose_mat1 = false;
    *transpose_mat2 = false;
    *transpose_mat3 = false;
  }
}

template <>
void get_transpose<cutlass::arch::Sm80>(
    at::ScalarType mat1_type,
    at::ScalarType mat2_type,
    at::ScalarType mat3_type,
    bool* transpose_mat1,
    bool* transpose_mat2,
    bool* transpose_mat3) {
  if ((mat1_type == at::kChar) && (mat2_type == at::kChar) &&
      (mat3_type == at::kInt)) {
    *transpose_mat1 = false;
    *transpose_mat2 = true;
    *transpose_mat3 = false;
  } else {
    *transpose_mat1 = false;
    *transpose_mat2 = false;
    *transpose_mat3 = false;
  }
}

} // namespace

int get_architecture() {
  return COMPUTE_CAPABILITY;
}

void get_padding(
    const at::ScalarType& mat1_type,
    const at::ScalarType& mat2_type,
    const at::ScalarType& mat3_type,
    int device,
    int* m_padding,
    int* n_padding,
    int* k_padding) {
  switch (get_architecture()) {
    case 60:
      get_padding<cutlass::arch::Sm60>(
          mat1_type, mat2_type, mat3_type, m_padding, n_padding, k_padding);
      break;
    case 61:
      get_padding<cutlass::arch::Sm61>(
          mat1_type, mat2_type, mat3_type, m_padding, n_padding, k_padding);
      break;
    case 70:
      get_padding<cutlass::arch::Sm70>(
          mat1_type, mat2_type, mat3_type, m_padding, n_padding, k_padding);
      break;
    case 75:
      get_padding<cutlass::arch::Sm75>(
          mat1_type, mat2_type, mat3_type, m_padding, n_padding, k_padding);
      break;
    case 80:
      get_padding<cutlass::arch::Sm80>(
          mat1_type, mat2_type, mat3_type, m_padding, n_padding, k_padding);
      break;
    default:
      AT_ERROR("device architecture is not supported");
  }
}

void get_transpose(
    const at::ScalarType& mat1_type,
    const at::ScalarType& mat2_type,
    const at::ScalarType& mat3_type,
    int device,
    bool* transpose_mat1,
    bool* transpose_mat2,
    bool* transpose_mat3) {
  switch (get_architecture()) {
    case 60:
      get_transpose<cutlass::arch::Sm60>(
          mat1_type,
          mat2_type,
          mat3_type,
          transpose_mat1,
          transpose_mat2,
          transpose_mat3);
      break;
    case 61:
      get_transpose<cutlass::arch::Sm61>(
          mat1_type,
          mat2_type,
          mat3_type,
          transpose_mat1,
          transpose_mat2,
          transpose_mat3);
      break;
    case 70:
      get_transpose<cutlass::arch::Sm70>(
          mat1_type,
          mat2_type,
          mat3_type,
          transpose_mat1,
          transpose_mat2,
          transpose_mat3);
      break;
    case 75:
      get_transpose<cutlass::arch::Sm75>(
          mat1_type,
          mat2_type,
          mat3_type,
          transpose_mat1,
          transpose_mat2,
          transpose_mat3);
      break;
    case 80:
      get_transpose<cutlass::arch::Sm80>(
          mat1_type,
          mat2_type,
          mat3_type,
          transpose_mat1,
          transpose_mat2,
          transpose_mat3);
      break;
    default:
      AT_ERROR("device architecture is not supported");
  }
}

void check_cutlass_status(cutlass::Status status) {
  TORCH_CHECK(
      (status == cutlass::Status::kSuccess), cutlassGetStatusString(status));
}

bool is_supported(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    at::ScalarType* r_type) {
  at::ScalarType mat1_type = mat1.scalar_type();
  at::ScalarType mat2_type = mat2.scalar_type();
  int device = mat1.get_device();
  switch (get_architecture()) {
    case 60:
      if ((mat1_type == at::kFloat) && (mat2_type == at::kFloat)) {
        *r_type = at::kFloat;
        return true;
      } else if ((mat1_type == at::kDouble) && (mat2_type == at::kDouble)) {
        *r_type = at::kDouble;
        return true;
      } else {
        return false;
      }
    case 61:
      if ((mat1_type == at::kFloat) && (mat2_type == at::kFloat)) {
        *r_type = at::kFloat;
        return true;
      } else if ((mat1_type == at::kDouble) && (mat2_type == at::kDouble)) {
        *r_type = at::kDouble;
        return true;
      } else if ((mat1_type == at::kChar) && (mat2_type == at::kChar)) {
        *r_type = at::kInt;
        return true;
      } else {
        return false;
      }
    case 70:
      if ((mat1_type == at::kHalf) && (mat2_type == at::kHalf)) {
        *r_type = at::kHalf;
        return true;
      } else if ((mat1_type == at::kFloat) && (mat2_type == at::kFloat)) {
        *r_type = at::kFloat;
        return true;
      } else if ((mat1_type == at::kDouble) && (mat2_type == at::kDouble)) {
        *r_type = at::kDouble;
        return true;
      } else if ((mat1_type == at::kChar) && (mat2_type == at::kChar)) {
        *r_type = at::kInt;
        return true;
      } else {
        return false;
      }
    case 75:
      if ((mat1_type == at::kHalf) && (mat2_type == at::kHalf)) {
        *r_type = at::kHalf;
        return true;
      } else if ((mat1_type == at::kFloat) && (mat2_type == at::kFloat)) {
        *r_type = at::kFloat;
        return true;
      } else if ((mat1_type == at::kDouble) && (mat2_type == at::kDouble)) {
        *r_type = at::kDouble;
        return true;
      } else if ((mat1_type == at::kChar) && (mat2_type == at::kChar)) {
        *r_type = at::kInt;
        return true;
      } else {
        return false;
      }
    case 80:
      if ((mat1_type == at::kHalf) && (mat2_type == at::kHalf)) {
        *r_type = at::kHalf;
        return true;
      } else if ((mat1_type == at::kFloat) && (mat2_type == at::kFloat)) {
        *r_type = at::kFloat;
        return true;
      } else if ((mat1_type == at::kDouble) && (mat2_type == at::kDouble)) {
        *r_type = at::kDouble;
        return true;
      } else if ((mat1_type == at::kChar) && (mat2_type == at::kChar)) {
        *r_type = at::kInt;
        return true;
      } else {
        return false;
      }
    default:
      return false;
  }
}

} // namespace fb
} // namespace at
