/*
 * neon-softmax 单元测试。
 *
 * 设计要点：
 * 1. 每个用例都真的断言，失败返回非 0 —— ctest / CI 才能拦住回归。
 * 2. 参考实现用 double 算，避免"拿自己验证自己"。
 * 3. 覆盖向量循环的边界（n = 7/8/9、1023/1024/1025），这是最容易写出 tail bug 的地方。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "softmax.h"

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                                 \
    do {                                                                 \
        ++g_checks;                                                      \
        if (!(cond)) {                                                   \
            ++g_failures;                                                \
            printf("  FAIL: " __VA_ARGS__);                              \
            printf("\n");                                                \
        }                                                                \
    } while (0)

/* ------------------------------------------------------------------ 工具 */

static void reference_softmax(const float *x, double *y, int n)
{
    double maxv = -INFINITY;
    for (int i = 0; i < n; ++i) {
        if ((double)x[i] > maxv) maxv = (double)x[i];
    }
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        y[i] = exp((double)x[i] - maxv);
        s += y[i];
    }
    for (int i = 0; i < n; ++i) y[i] /= s;
}

static double max_abs_diff(const float *a, const double *b, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static double sum_of(const float *a, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)a[i];
    return s;
}

/* 可移植的 binary16 编解码：native fp16 可用时直接用，否则按位拼 */
#if defined(KUNPENG_HAS_NATIVE_FP16)
static SOFTMAX_HALF_T half_from_float(float f) { return (SOFTMAX_HALF_T)f; }
static float half_to_float(SOFTMAX_HALF_T h) { return (float)h; }
#else
static SOFTMAX_HALF_T half_from_float(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)(((u >> 23) & 0xFFu)) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3FFu;
    if (exp <= 0) return (SOFTMAX_HALF_T)sign;
    if (exp >= 0x1F) return (SOFTMAX_HALF_T)(sign | 0x7C00u);
    return (SOFTMAX_HALF_T)(sign | ((uint32_t)exp << 10) | mant);
}

static float half_to_float(SOFTMAX_HALF_T h)
{
    uint16_t u = (uint16_t)h;
    uint32_t sign = (uint32_t)(u & 0x8000u) << 16;
    uint32_t exp = (uint32_t)(u & 0x7C00u) >> 10;
    uint32_t mant = (uint32_t)(u & 0x03FFu);
    uint32_t out;
    float f;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        exp = 127 - 15 + 1;
        while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
        mant &= 0x03FFu;
        out = sign | (exp << 23) | (mant << 13);
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    memcpy(&f, &out, sizeof(f));
    return f;
}
#endif

/* ------------------------------------------------------------------ 用例 */

static void test_null_and_empty(void)
{
    printf("case: null / empty input is a no-op\n");
    float x[4] = {1.f, 2.f, 3.f, 4.f};
    float y[4] = {9.f, 9.f, 9.f, 9.f};

    softmax_f32_scalar(NULL, y, 4);
    softmax_f32_scalar(x, NULL, 4);
    softmax_f32_scalar(x, y, 0);
    softmax_f32_scalar(x, y, -3);
    softmax_f32_neon(NULL, y, 4);
    softmax_f32_neon(x, NULL, 4);
    softmax_f32_neon(x, y, 0);

    /* 全部输入非法时不应该动输出 */
    for (int i = 0; i < 4; ++i) CHECK(y[i] == 9.f, "y[%d] was modified: %g", i, y[i]);
}

static void test_sum_to_one(int n)
{
    printf("case: sum-to-one, n=%d (vector boundary)\n", n);
    float *x = (float *)malloc(sizeof(float) * (size_t)n);
    float *ys = (float *)malloc(sizeof(float) * (size_t)n);
    float *yn = (float *)malloc(sizeof(float) * (size_t)n);
    if (!x || !ys || !yn) { free(x); free(ys); free(yn); CHECK(0, "malloc failed"); return; }

    for (int i = 0; i < n; ++i) {
        x[i] = sinf(0.017f * (float)i) * 6.0f;
    }

    softmax_f32_scalar(x, ys, n);
    softmax_f32_neon(x, yn, n);

    CHECK(fabs(sum_of(ys, n) - 1.0) < 1e-5, "scalar sum = %.9f", sum_of(ys, n));
    CHECK(fabs(sum_of(yn, n) - 1.0) < 1e-5, "neon   sum = %.9f", sum_of(yn, n));

    for (int i = 0; i < n; ++i) {
        CHECK(ys[i] >= 0.0f, "scalar produced negative probability at %d", i);
        CHECK(yn[i] >= 0.0f, "neon produced negative probability at %d", i);
    }

    free(x); free(ys); free(yn);
}

static void test_against_double_reference(void)
{
    const int n = 1024;
    printf("case: accuracy against double reference, n=%d\n", n);
    float *x = (float *)malloc(sizeof(float) * (size_t)n);
    float *ys = (float *)malloc(sizeof(float) * (size_t)n);
    float *yn = (float *)malloc(sizeof(float) * (size_t)n);
    double *ref = (double *)malloc(sizeof(double) * (size_t)n);
    if (!x || !ys || !yn || !ref) { free(x); free(ys); free(yn); free(ref); CHECK(0, "malloc failed"); return; }

    for (int i = 0; i < n; ++i) {
        x[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 20.0f;
    }
    reference_softmax(x, ref, n);
    softmax_f32_scalar(x, ys, n);
    softmax_f32_neon(x, yn, n);

    double es = max_abs_diff(ys, ref, n);
    double en = max_abs_diff(yn, ref, n);
    printf("  max|err| scalar=%.3e neon=%.3e\n", es, en);

    /* 标量走 libm expf，误差应在 1e-6 内 */
    CHECK(es < 1e-6, "scalar error %.3e exceeds 1e-6", es);
    /* NEON 走 5 阶多项式 + ldexpf，放宽到 1e-4 */
    CHECK(en < 1e-4, "neon error %.3e exceeds 1e-4", en);

    free(x); free(ys); free(yn); free(ref);
}

static void test_shift_invariance(void)
{
    const int n = 300;
    printf("case: shift invariance  softmax(x + c) == softmax(x)\n");
    float *x = (float *)malloc(sizeof(float) * (size_t)n);
    float *xs = (float *)malloc(sizeof(float) * (size_t)n);
    float *a = (float *)malloc(sizeof(float) * (size_t)n);
    float *b = (float *)malloc(sizeof(float) * (size_t)n);
    if (!x || !xs || !a || !b) { free(x); free(xs); free(a); free(b); CHECK(0, "malloc failed"); return; }

    for (int i = 0; i < n; ++i) {
        x[i] = cosf(0.05f * (float)i) * 3.0f;
        xs[i] = x[i] + 37.5f;
    }
    softmax_f32_scalar(x, a, n);
    softmax_f32_scalar(xs, b, n);

    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > worst) worst = d;
    }
    CHECK(worst < 1e-6, "shift changed output by %.3e (max-subtraction is broken)", worst);

    free(x); free(xs); free(a); free(b);
}

static void test_uniform_input(void)
{
    const int n = 64;
    printf("case: uniform input -> uniform output\n");
    float x[64], y[64];
    for (int i = 0; i < n; ++i) x[i] = 2.5f;
    softmax_f32_scalar(x, y, n);
    for (int i = 0; i < n; ++i) {
        CHECK(fabsf(y[i] - 1.0f / (float)n) < 1e-7f,
              "y[%d]=%.9f expected %.9f", i, (double)y[i], (double)(1.0f / (float)n));
    }
}

static void test_extreme_magnitude(void)
{
    const int n = 128;
    printf("case: extreme magnitudes do not overflow\n");
    float x[128], y[128];
    for (int i = 0; i < n; ++i) x[i] = (i % 2 == 0) ? 100.0f : -100.0f;
    softmax_f32_scalar(x, y, n);

    for (int i = 0; i < n; ++i) {
        CHECK(isfinite(y[i]), "y[%d] is not finite: %g", i, (double)y[i]);
    }
    CHECK(fabs(sum_of(y, n) - 1.0) < 1e-5, "sum = %.9f", sum_of(y, n));
}

static void test_single_element(void)
{
    printf("case: n=1 -> probability 1.0\n");
    float x = 42.0f, y = 0.0f;
    softmax_f32_scalar(&x, &y, 1);
    CHECK(fabsf(y - 1.0f) < 1e-7f, "scalar n=1 -> %g", (double)y);
    y = 0.0f;
    softmax_f32_neon(&x, &y, 1);
    CHECK(fabsf(y - 1.0f) < 1e-7f, "neon n=1 -> %g", (double)y);
}

static void test_in_place(void)
{
    const int n = 100;
    printf("case: in-place (y == x) is allowed\n");
    float x[100], ref[100], buf[100];
    double worst = 0.0;
    for (int i = 0; i < n; ++i) x[i] = (float)(i % 7) - 3.0f;
    memcpy(buf, x, sizeof(x));
    softmax_f32_scalar(buf, ref, n);

    memcpy(buf, x, sizeof(x));
    softmax_f32_scalar(buf, buf, n);
    for (int i = 0; i < n; ++i) {
        double d = fabs((double)buf[i] - (double)ref[i]);
        if (d > worst) worst = d;
    }
    CHECK(worst < 1e-7, "in-place differs from out-of-place by %.3e", worst);
}

static void test_f16_roundtrip(void)
{
    const int n = 5;
    printf("case: fp16 path keeps ordering and sums to ~1\n");
    /* 这些值在 binary16 下可精确表示 */
    const float vals[5] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    SOFTMAX_HALF_T x[5], y[5];
    float decoded[5];
    double s = 0.0;
    for (int i = 0; i < n; ++i) x[i] = half_from_float(vals[i]);

    softmax_f16_neon(x, y, n);

    for (int i = 0; i < n; ++i) decoded[i] = half_to_float(y[i]);

    for (int i = 0; i < n; ++i) {
        CHECK(isfinite(decoded[i]) && decoded[i] >= 0.0f,
              "fp16 out[%d] invalid: %g", i, (double)decoded[i]);
        if (i > 0) {
            CHECK(decoded[i] >= decoded[i - 1] - 1e-3f,
                  "fp16 ordering broken at %d: %g < %g", i, (double)decoded[i], (double)decoded[i - 1]);
        }
    }
    for (int i = 0; i < n; ++i) s += (double)decoded[i];
    /* fp16 只有 1e-3 量级分辨率，再叠加"先量化 exp 再归一化"的损失 */
    CHECK(fabs(s - 1.0) < 5e-3, "fp16 sum = %.6f", s);
}

int main(void)
{
    printf("=== neon-softmax tests ===\n");
    srand(0);

    test_null_and_empty();
    test_single_element();
    test_sum_to_one(1);
    test_sum_to_one(7);
    test_sum_to_one(8);
    test_sum_to_one(9);
    test_sum_to_one(1023);
    test_sum_to_one(1024);
    test_sum_to_one(1025);
    test_against_double_reference();
    test_shift_invariance();
    test_uniform_input();
    test_extreme_magnitude();
    test_in_place();
    test_f16_roundtrip();

    printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
