#ifndef KUNPENG_EXP_POLY_H
#define KUNPENG_EXP_POLY_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Constants for range reduction
#define LOG2E_F 1.4426950408889634f   // log2(e)
#define LN2_F   0.6931471805599453f   // ln(2)

// Simple 5th-order Taylor polynomial for exp(r) with r in ~[-ln2/2, ln2/2]
// P(r) = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
static inline float exp_poly5_float(float r) {
    float r2 = r * r;
    float r3 = r2 * r;
    float r4 = r2 * r2;
    float r5 = r4 * r;
    return 1.0f + r + 0.5f * r2 + (1.0f/6.0f) * r3 + (1.0f/24.0f) * r4 + (1.0f/120.0f) * r5;
}

// Compute exp(x) approximately using range reduction by ln2 and polynomial for the remainder.
// exp(x) = 2^k * P(r), with k = round(x * log2(e)), r = x - k*ln2
static inline float exp_approx_float(float x) {
    float y = x * LOG2E_F;
    // nearest integer to y for lower error accumulation vs floor
    int k = (int)(y > 0.0f ? y + 0.5f : y - 0.5f);
    float r = x - (float)k * LN2_F;
    float p = exp_poly5_float(r);
    // scale by 2^k via ldexpf
    extern float ldexpf(float, int);
    return ldexpf(p, k);
}

#ifdef __cplusplus
}
#endif

#endif // KUNPENG_EXP_POLY_H


