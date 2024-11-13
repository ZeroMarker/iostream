#include <stdio.h>

#define N 10000000

// CUDA kernel for vector addition
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *a, *b, *out; // Host arrays
    float *d_a, *d_b, *d_out; // Device arrays

    // Allocate memory for host arrays
    a = (float *)malloc(sizeof(float) * N);
    b = (float *)malloc(sizeof(float) * N);
    out = (float *)malloc(sizeof(float) * N);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate memory for device arrays
    cudaMalloc((void **)&d_a, sizeof(float) * N);
    cudaMalloc((void **)&d_b, sizeof(float) * N);
    cudaMalloc((void **)&d_out, sizeof(float) * N);

    // Copy host data to device
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;

    // Launch the CUDA kernel
    vector_add<<<num_blocks, block_size>>>(d_out, d_a, d_b, N);

    // Copy the result back to the host
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Free host memory
    free(a);
    free(b);
    free(out);

    return 0;
}
