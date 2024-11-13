#include <stdio.h>
#include <math.h>

#define TOLERANCE 1e-6
#define MAX_ITERATIONS 1000

__device__ float newton(float x, float a) {
    return x - (x * x - a) / (2 * x);
}

__global__ void newton_kernel(float *x, float a) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float xi = x[tid];

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        float xi1 = newton(xi, a);
        if (fabs(xi1 - xi) < TOLERANCE) {
            x[tid] = xi1;
            break;
        }
        xi = xi1;
    }
}

int main() {
    int N = 1000; // Number of values to find the square root for
    float *h_values = (float*)malloc(N * sizeof(float));
    float *d_values;
    float a = 2.0f; // The number to find the square root of

    for (int i = 0; i < N; i++) {
        h_values[i] = 1.0f + i; // Initial guesses
    }

    cudaMalloc((void**)&d_values, N * sizeof(float));
    cudaMemcpy(d_values, h_values, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    newton_kernel<<<numBlocks, threadsPerBlock>>>(d_values, a);
    cudaMemcpy(h_values, d_values, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("sqrt(%f) = %f\n", a, h_values[i]);
    }

    free(h_values);
    cudaFree(d_values);

    return 0;
}
