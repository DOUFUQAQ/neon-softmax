#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "softmax.h"

static double now_sec(void) {
    struct timespec ts;
#ifdef _WIN32
    timespec_get(&ts, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void run_once(const char* name,
                     void (*fn)(const float*, float*, int),
                     const float* x, float* y, int n,
                     int iters) {
    // warmup
    for (int i = 0; i < 3; ++i) fn(x, y, n);
    double t0 = now_sec();
    for (int i = 0; i < iters; ++i) fn(x, y, n);
    double t1 = now_sec();
    double dt = (t1 - t0) / iters;
    printf("impl=%s n=%d time=%.6g s per call, %.3f ns/elem\n",
           name, n, dt, dt * 1e9 / n);
}

int main(int argc, char** argv) {
    int n = 16384;
    int iters = 100;
    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) iters = atoi(argv[2]);
    float* x = (float*)malloc(sizeof(float) * n);
    float* y = (float*)malloc(sizeof(float) * n);
    if (!x || !y) return 1;
    for (int i = 0; i < n; ++i) {
        x[i] = sinf(0.001f * (float)i) * 8.0f; // deterministic pattern
    }
    run_once("f32_scalar", softmax_f32_scalar, x, y, n, iters);
    run_once("f32_neon", softmax_f32_neon, x, y, n, iters);
    free(x); free(y);
    return 0;
}


