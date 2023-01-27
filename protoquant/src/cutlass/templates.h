#pragma once

#include <protoquant/src/cutlass/utilities.h>

#include <ATen/cuda/Exceptions.h>
#include <cutlass/gemm/device/gemm.h>

namespace at {
namespace fb {

// f16*f16 + f16 => f16
template <typename architecture>
void gemm(
    int m,
    int n,
    int k,
    const Half* a,
    int lda,
    const Half* b,
    int ldb,
    const Half* c,
    int ldc,
    Half* d,
    int ldd,
    const float alpha,
    const float beta) {
  AT_ERROR("not supported on this architecture");
}

// f32*f32 + f32 => f32
template <typename architecture>
void gemm(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    const float* c,
    int ldc,
    float* d,
    int ldd,
    const float alpha,
    const float beta) {
  AT_ERROR("not supported on this architecture");
}

// f64*f64 + f64 => f64
template <typename architecture>
void gemm(
    int m,
    int n,
    int k,
    const double* a,
    int lda,
    const double* b,
    int ldb,
    const double* c,
    int ldc,
    double* d,
    int ldd,
    const float alpha,
    const float beta) {
  AT_ERROR("not supported on this architecture");
}

// s8*s8 + s32 => s32
template <typename architecture>
void gemm(
    int m,
    int n,
    int k,
    const int8_t* a,
    int lda,
    const int8_t* b,
    int ldb,
    const int32_t* c,
    int ldc,
    int32_t* d,
    int ldd,
    const float alpha,
    const float beta) {
  AT_ERROR("not supported on this architecture");
}

} // namespace fb
} // namespace at

// cutlass::arch::Sm60
namespace at {
namespace fb {

// f32*f32 + f32 => f32
template <>
void gemm<cutlass::arch::Sm60>(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    const float* c,
    int ldc,
    float* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = float;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = float;
  using B_layout = cutlass::layout::RowMajor;

  using C_type = float;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = float;

  using target = cutlass::arch::OpClassSimt;
  using architecture = cutlass::arch::Sm60;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k}, {a, lda}, {b, ldb}, {c, ldc}, {d, ldd}, {alpha, beta}});

  check_cutlass_status(status);
}

// f64*f64 + f64 => f64
template <>
void gemm<cutlass::arch::Sm60>(
    int m,
    int n,
    int k,
    const double* a,
    int lda,
    const double* b,
    int ldb,
    const double* c,
    int ldc,
    double* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = double;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = double;
  using B_layout = cutlass::layout::RowMajor;

  using C_type = double;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = double;

  using target = cutlass::arch::OpClassSimt;
  using architecture = cutlass::arch::Sm60;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k}, {a, lda}, {b, ldb}, {c, ldc}, {d, ldd}, {alpha, beta}});

  check_cutlass_status(status);
}

} // namespace fb
} // namespace at

// cutlass::arch::Sm61
namespace at {
namespace fb {

// f32*f32 + f32 => f32
template <>
void gemm<cutlass::arch::Sm61>(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    const float* c,
    int ldc,
    float* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = float;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = float;
  using B_layout = cutlass::layout::RowMajor;

  using C_type = float;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = float;

  using target = cutlass::arch::OpClassSimt;
  using architecture = cutlass::arch::Sm61;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k}, {a, lda}, {b, ldb}, {c, ldc}, {d, ldd}, {alpha, beta}});

  check_cutlass_status(status);
}

// f64*f64 + f64 => f64
template <>
void gemm<cutlass::arch::Sm61>(
    int m,
    int n,
    int k,
    const double* a,
    int lda,
    const double* b,
    int ldb,
    const double* c,
    int ldc,
    double* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = double;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = double;
  using B_layout = cutlass::layout::RowMajor;

  using C_type = double;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = double;

  using target = cutlass::arch::OpClassSimt;
  using architecture = cutlass::arch::Sm61;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k}, {a, lda}, {b, ldb}, {c, ldc}, {d, ldd}, {alpha, beta}});

  check_cutlass_status(status);
}

// s8*s8 + s32 => s32
template <>
void gemm<cutlass::arch::Sm61>(
    int m,
    int n,
    int k,
    const int8_t* a,
    int lda,
    const int8_t* b,
    int ldb,
    const int32_t* c,
    int ldc,
    int32_t* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = int8_t;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = int8_t;
  using B_layout = cutlass::layout::ColumnMajor;

  using C_type = int32_t;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = int32_t;

  using target = cutlass::arch::OpClassSimt;
  using architecture = cutlass::arch::Sm61;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k}, {a, lda}, {b, ldb}, {c, ldc}, {d, ldd}, {alpha, beta}});

  check_cutlass_status(status);
}

} // namespace fb
} // namespace at

// cutlass::arch::Sm70
namespace at {
namespace fb {

// f32*f32 + f32 => f32
template <>
void gemm<cutlass::arch::Sm70>(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    const float* c,
    int ldc,
    float* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = float;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = float;
  using B_layout = cutlass::layout::RowMajor;

  using C_type = float;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = float;

  using target = cutlass::arch::OpClassSimt;
  using architecture = cutlass::arch::Sm70;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k}, {a, lda}, {b, ldb}, {c, ldc}, {d, ldd}, {alpha, beta}});

  check_cutlass_status(status);
}

// f64*f64 + f64 => f64
template <>
void gemm<cutlass::arch::Sm70>(
    int m,
    int n,
    int k,
    const double* a,
    int lda,
    const double* b,
    int ldb,
    const double* c,
    int ldc,
    double* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = double;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = double;
  using B_layout = cutlass::layout::RowMajor;

  using C_type = double;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = double;

  using target = cutlass::arch::OpClassSimt;
  using architecture = cutlass::arch::Sm70;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k}, {a, lda}, {b, ldb}, {c, ldc}, {d, ldd}, {alpha, beta}});

  check_cutlass_status(status);
}

// s8*s8 + s32 => s32
template <>
void gemm<cutlass::arch::Sm70>(
    int m,
    int n,
    int k,
    const int8_t* a,
    int lda,
    const int8_t* b,
    int ldb,
    const int32_t* c,
    int ldc,
    int32_t* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = int8_t;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = int8_t;
  using B_layout = cutlass::layout::ColumnMajor;

  using C_type = int32_t;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = int32_t;

  using target = cutlass::arch::OpClassSimt;
  using architecture = cutlass::arch::Sm70;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k}, {a, lda}, {b, ldb}, {c, ldc}, {d, ldd}, {alpha, beta}});

  check_cutlass_status(status);
}

} // namespace fb
} // namespace at

// cutlass::arch::Sm75
namespace at {
namespace fb {

// fp16*fp16 + fp16 => fp16
template <>
void gemm<cutlass::arch::Sm75>(
    int m,
    int n,
    int k,
    const Half* a,
    int lda,
    const Half* b,
    int ldb,
    const Half* c,
    int ldc,
    Half* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = cutlass::half_t;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = cutlass::half_t;
  using B_layout = cutlass::layout::RowMajor;

  using C_type = cutlass::half_t;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = cutlass::half_t;

  using target = cutlass::arch::OpClassTensorOp;
  using architecture = cutlass::arch::Sm75;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 32>;

  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;

  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      128 / cutlass::sizeof_bits<cutlass::half_t>::value,
      cutlass::half_t,
      cutlass::half_t>;

  constexpr int NumStages = 2;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture,
      ShapeMMAThreadBlock,
      ShapeMMAWarp,
      ShapeMMAOp,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k},
       {const_cast<cutlass::half_t*>((cutlass::half_t*)a), lda},
       {const_cast<cutlass::half_t*>((cutlass::half_t*)b), ldb},
       {const_cast<cutlass::half_t*>((cutlass::half_t*)c), ldc},
       {(cutlass::half_t*)d, ldd},
       {(cutlass::half_t)alpha, (cutlass::half_t)beta}});

  check_cutlass_status(status);
}

// f32*f32 + f32 => f32
template <>
void gemm<cutlass::arch::Sm75>(
    int m,
    int n,
    int k,
    const float* a,
    int lda,
    const float* b,
    int ldb,
    const float* c,
    int ldc,
    float* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = float;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = float;
  using B_layout = cutlass::layout::RowMajor;

  using C_type = float;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = float;

  using target = cutlass::arch::OpClassSimt;
  using architecture = cutlass::arch::Sm75;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k}, {a, lda}, {b, ldb}, {c, ldc}, {d, ldd}, {alpha, beta}});

  check_cutlass_status(status);
}

// f64*f64 + f64 => f64
template <>
void gemm<cutlass::arch::Sm75>(
    int m,
    int n,
    int k,
    const double* a,
    int lda,
    const double* b,
    int ldb,
    const double* c,
    int ldc,
    double* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = double;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = double;
  using B_layout = cutlass::layout::RowMajor;

  using C_type = double;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = double;

  using target = cutlass::arch::OpClassSimt;
  using architecture = cutlass::arch::Sm75;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k}, {a, lda}, {b, ldb}, {c, ldc}, {d, ldd}, {alpha, beta}});

  check_cutlass_status(status);
}

// s8*s8 + s32 => s32
template <>
void gemm<cutlass::arch::Sm75>(
    int m,
    int n,
    int k,
    const int8_t* a,
    int lda,
    const int8_t* b,
    int ldb,
    const int32_t* c,
    int ldc,
    int32_t* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = int8_t;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = int8_t;
  using B_layout = cutlass::layout::ColumnMajor;

  using C_type = int32_t;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = int32_t;

  using target = cutlass::arch::OpClassTensorOp;
  using architecture = cutlass::arch::Sm75;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>;

  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;

  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>;

  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      C_type,
      128 / cutlass::sizeof_bits<C_type>::value,
      acccumulator_type,
      acccumulator_type>;

  constexpr int NumStages = 2;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture,
      ShapeMMAThreadBlock,
      ShapeMMAWarp,
      ShapeMMAOp,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k},
       {a, lda},
       {b, ldb},
       {c, ldc},
       {d, ldd},
       {(int32_t)alpha, (int32_t)beta}});

  check_cutlass_status(status);
}

} // namespace fb
} // namespace at

// cutlass::arch::Sm80
namespace at {
namespace fb {

// fp16*fp16 + fp16 => fp16
template <>
void gemm<cutlass::arch::Sm80>(
    int m,
    int n,
    int k,
    const Half* a,
    int lda,
    const Half* b,
    int ldb,
    const Half* c,
    int ldc,
    Half* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = cutlass::half_t;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = cutlass::half_t;
  using B_layout = cutlass::layout::RowMajor;

  using C_type = cutlass::half_t;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = cutlass::half_t;

  using target = cutlass::arch::OpClassTensorOp;
  using architecture = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>;

  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;

  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      128 / cutlass::sizeof_bits<cutlass::half_t>::value,
      cutlass::half_t,
      cutlass::half_t>;

  constexpr int NumStages = 2;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture,
      ShapeMMAThreadBlock,
      ShapeMMAWarp,
      ShapeMMAOp,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k},
       {const_cast<cutlass::half_t*>((cutlass::half_t*)a), lda},
       {const_cast<cutlass::half_t*>((cutlass::half_t*)b), ldb},
       {const_cast<cutlass::half_t*>((cutlass::half_t*)c), ldc},
       {(cutlass::half_t*)d, ldd},
       {(cutlass::half_t)alpha, (cutlass::half_t)beta}});

  check_cutlass_status(status);
}

// s8*s8 + s32 => s32
template <>
void gemm<cutlass::arch::Sm80>(
    int m,
    int n,
    int k,
    const int8_t* a,
    int lda,
    const int8_t* b,
    int ldb,
    const int32_t* c,
    int ldc,
    int32_t* d,
    int ldd,
    const float alpha,
    const float beta) {
  using A_type = int8_t;
  using A_layout = cutlass::layout::RowMajor;

  using B_type = int8_t;
  using B_layout = cutlass::layout::ColumnMajor;

  using C_type = int32_t;
  using C_layout = cutlass::layout::RowMajor;

  using acccumulator_type = int32_t;

  using target = cutlass::arch::OpClassTensorOp;
  using architecture = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>;

  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;

  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>;

  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      C_type,
      128 / cutlass::sizeof_bits<C_type>::value,
      acccumulator_type,
      acccumulator_type>;

  constexpr int NumStages = 2;

  cutlass::gemm::device::Gemm<
      A_type,
      A_layout,
      B_type,
      B_layout,
      C_type,
      C_layout,
      acccumulator_type,
      target,
      architecture,
      ShapeMMAThreadBlock,
      ShapeMMAWarp,
      ShapeMMAOp,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages>
      gemm_op;

  cutlass::Status status = gemm_op(
      {{m, n, k},
       {a, lda},
       {b, ldb},
       {c, ldc},
       {d, ldd},
       {(int32_t)alpha, (int32_t)beta}});

  check_cutlass_status(status);
}

} // namespace fb
} // namespace at
