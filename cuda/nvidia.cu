#include <stdio.h>

// CUDA kernel to calculate the square of each element
__global__ void squareKernel(float *d_out, float *d_in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] * d_in[idx];
    }
}

int main() {
    const int n = 10; // Number of elements in the array
    const int size = n * sizeof(float);

    float h_in[n];     // Input array on the host
    float h_out[n];    // Output array on the host
    float *d_in, *d_out; // Pointers to data on the device (GPU)

    // Initialize input data on the host
    for (int i = 0; i < n; i++) {
        h_in[i] = float(i);
    }

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    // Copy data from the host to the GPU
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 grid(2); // 2 blocks
    dim3 block(5); // 5 threads per block

    // Launch the kernel on the GPU
    squareKernel<<<grid, block>>>(d_out, d_in, n);

    // Copy the result back from the GPU to the host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < n; i++) {
        printf("%f squared is %f\n", h_in[i], h_out[i]);
    }

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
