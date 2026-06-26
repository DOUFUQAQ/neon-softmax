#include <math.h>
#include <stddef.h>
#include <float.h>
#include "softmax.h"

void softmax_f32_scalar(const float* input, float* output, int n) {
    if (n <= 0 || input == NULL || output == NULL) {
        return;
    }
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < n; ++i) {
        float v = input[i];
        if (v > max_val) max_val = v;
    }
    // Compute exp(x - max) and sum
    double sum = 0.0; // accumulate in higher precision to reduce error
    for (int i = 0; i < n; ++i) {
        float ex = expf(input[i] - max_val);
        output[i] = ex;
        sum += (double)ex;
    }
    if (sum == 0.0) {
        // Degenerate case: all -inf; distribute uniformly
        float uniform = 1.0f / (float)n;
        for (int i = 0; i < n; ++i) output[i] = uniform;
        return;
    }
    float inv_sum = (float)(1.0 / sum);
    for (int i = 0; i < n; ++i) {
        output[i] *= inv_sum;
    }
}


