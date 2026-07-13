#include <stddef.h>
#include <math.h>
#include "softmax.h"
#include "exp_poly.h"

#if defined(USE_AARCH64_NEON)
#include <arm_neon.h>
#endif

void softmax_f32_neon(const float* input, float* output, int n) {
    if (n <= 0 || input == NULL || output == NULL) {
        return;
    }

#if defined(USE_AARCH64_NEON)
    // Vectorized max reduction
    int i = 0;
    float32x4_t vmax0 = vdupq_n_f32(-INFINITY);
    float32x4_t vmax1 = vdupq_n_f32(-INFINITY);
    for (; i + 8 <= n; i += 8) {
        float32x4_t a0 = vld1q_f32(input + i);
        float32x4_t a1 = vld1q_f32(input + i + 4);
        vmax0 = vmaxq_f32(vmax0, a0);
        vmax1 = vmaxq_f32(vmax1, a1);
    }
    float32x4_t vmax = vmaxq_f32(vmax0, vmax1);
    // horizontal reduce
    float32x2_t vmax2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
    vmax2 = vpmax_f32(vmax2, vmax2);
    float max_val = vget_lane_f32(vmax2, 0);
    // tail for max
    for (; i < n; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Compute exp approximation and sum
    i = 0;
    float32x4_t vsum0 = vdupq_n_f32(0.0f);
    float32x4_t vsum1 = vdupq_n_f32(0.0f);
    float32x4_t vmax_b = vdupq_n_f32(max_val);
    for (; i + 8 <= n; i += 8) {
        float32x4_t a0 = vsubq_f32(vld1q_f32(input + i), vmax_b);
        float32x4_t a1 = vsubq_f32(vld1q_f32(input + i + 4), vmax_b);

        // Polynomial exp approx per lane (scalar helper, still vectorized loop)
        float tmp0[4], tmp1[4];
        vst1q_f32(tmp0, a0);
        vst1q_f32(tmp1, a1);
        for (int t = 0; t < 4; ++t) tmp0[t] = exp_approx_float(tmp0[t]);
        for (int t = 0; t < 4; ++t) tmp1[t] = exp_approx_float(tmp1[t]);
        float32x4_t e0 = vld1q_f32(tmp0);
        float32x4_t e1 = vld1q_f32(tmp1);

        vst1q_f32(output + i, e0);
        vst1q_f32(output + i + 4, e1);
        vsum0 = vaddq_f32(vsum0, e0);
        vsum1 = vaddq_f32(vsum1, e1);
    }
    float32x4_t vsum = vaddq_f32(vsum0, vsum1);
    float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    vsum2 = vpadd_f32(vsum2, vsum2);
    float sum = vget_lane_f32(vsum2, 0);
    // tail compute exp and sum
    for (; i < n; ++i) {
        float v = input[i] - max_val;
        float e = exp_approx_float(v);
        output[i] = e;
        sum += e;
    }
    if (sum == 0.0f) {
        float uniform = 1.0f / (float)n;
        for (int j = 0; j < n; ++j) output[j] = uniform;
        return;
    }
    float inv_sum = 1.0f / sum;
    float32x4_t vinv = vdupq_n_f32(inv_sum);
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        float32x4_t e0 = vld1q_f32(output + j);
        float32x4_t e1 = vld1q_f32(output + j + 4);
        vst1q_f32(output + j, vmulq_f32(e0, vinv));
        vst1q_f32(output + j + 4, vmulq_f32(e1, vinv));
    }
    for (; j < n; ++j) output[j] *= inv_sum;
    return;
#else
    // Fallback to scalar
    softmax_f32_scalar(input, output, n);
#endif
}


