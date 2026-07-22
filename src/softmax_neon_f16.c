#include <stddef.h>
#include <math.h>
#include <string.h>
#include "softmax.h"
#include "exp_poly.h"

#if defined(USE_AARCH64_NEON)
#include <arm_neon.h>
#endif

// Current implementation: mixed precision path that converts FP16 to FP32,
// computes softmax in FP32 scalar/NEON, then converts back. This provides
// correctness now; pure FP16 vector arithmetic kernel can replace later.
static inline float half_to_float(SOFTMAX_HALF_T h) {
#if defined(KUNPENG_HAS_NATIVE_FP16)
    return (float)h;
#else
    // Convert IEEE754 half (binary16) in uint16_t to float
    uint16_t u = h;
    uint32_t sign = (u & 0x8000u) << 16;
    uint32_t exp  = (u & 0x7C00u) >> 10;
    uint32_t mant = (u & 0x03FFu);
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            // subnormal
            exp = 127 - 15 + 1; // bias adjust with normalization step
            while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x03FFu;
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        // inf/NaN
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        // normalized
        uint32_t exp32 = exp - 15 + 127;
        out = sign | (exp32 << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
#endif
}

static inline SOFTMAX_HALF_T float_to_half(float f) {
#if defined(KUNPENG_HAS_NATIVE_FP16)
    return (SOFTMAX_HALF_T)f;
#else
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp   = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3FFu;
    uint16_t out;
    if (exp <= 0) {
        if (exp < -10) {
            out = (uint16_t)sign; // underflow to zero
        } else {
            // subnormal
            uint32_t m = (u & 0x7FFFFFu) | 0x800000u;
            int shift = 14 - exp;
            out = (uint16_t)(sign | (m >> shift));
        }
    } else if (exp >= 0x1F) {
        // overflow to inf
        out = (uint16_t)(sign | 0x7C00u);
    } else {
        out = (uint16_t)(sign | (exp << 10) | mant);
    }
    return out;
#endif
}

void softmax_f16_neon(const SOFTMAX_HALF_T* input, SOFTMAX_HALF_T* output, int n) {
    if (n <= 0 || input == NULL || output == NULL) {
        return;
    }
    // Temporary: allocate on stack per call (simple baseline); for very large n,
    // users should call the FP32 API directly or we can stream-convert in blocks.
    // Here we do a simple two-pass in FP32.
    // 1) find max
    float max_val = -INFINITY;
    for (int i = 0; i < n; ++i) {
        float v = half_to_float(input[i]);
        if (v > max_val) max_val = v;
    }

    double sum = 0.0;
    // We store exp values into output as FP16 later; keep a temporary sum in FP32
    for (int i = 0; i < n; ++i) {
        float v = half_to_float(input[i]) - max_val;
        float e = expf(v);
        // Temporarily stash FP32 in place via union trick avoided; use sum only
        sum += (double)e;
        output[i] = float_to_half(e); // will be renormalized in second pass
    }
    if (sum == 0.0) {
        SOFTMAX_HALF_T uniform = float_to_half(1.0f / (float)n);
        for (int i = 0; i < n; ++i) output[i] = uniform;
        return;
    }
    float inv_sum = (float)(1.0 / sum);
    for (int i = 0; i < n; ++i) {
        float e = half_to_float(output[i]);
        output[i] = float_to_half(e * inv_sum);
    }
}


