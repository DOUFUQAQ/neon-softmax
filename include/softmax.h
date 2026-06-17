#ifndef KUNPENG_SOFTMAX_H
#define KUNPENG_SOFTMAX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Detect/define half-precision API type for portability
#if defined(__clang__) || defined(__GNUC__)
#  if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__aarch64__)
#    define KUNPENG_HAS_NATIVE_FP16 1
#  endif
#endif

#if defined(KUNPENG_HAS_NATIVE_FP16)
  typedef __fp16 SOFTMAX_HALF_T;
#else
  // Portable fallback: store IEEE754 half in uint16_t
  typedef uint16_t SOFTMAX_HALF_T;
#endif

// Compute numerically stable softmax over a length-n vector x into y.
// Requirements:
// - x and y must be non-null and n > 0
// - In-place is allowed (y == x)
// - Baseline correctness implementation (portable)
void softmax_f32_scalar(const float* input, float* output, int n);

// AArch64 NEON implementation with polynomial-approximated exp.
// Falls back to scalar on unsupported platforms or when disabled.
void softmax_f32_neon(const float* input, float* output, int n);

// FP16 interface. Implementation may use mixed precision internally
// (FP32 accumulation and normalization) to maintain accuracy.
// If FP16 vector arithmetic is unavailable, it will convert to FP32,
// compute, then convert back to FP16.
void softmax_f16_neon(const SOFTMAX_HALF_T* input, SOFTMAX_HALF_T* output, int n);

#ifdef __cplusplus
}
#endif

#endif // KUNPENG_SOFTMAX_H


