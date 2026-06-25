#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void reluKernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = fmaxf(src[idx], 0.0f);
}

__global__ void reluInPlace(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = fmaxf(x[idx], 0.0f);
}

bool checkReluResult(const float* input, const float* output, int n) {
    for (int i = 0; i < n; i++) {
        float expected = input[i] > 0.0f ? input[i] : 0.0f;
        if (output[i] != expected) {
            fprintf(stderr, "MISMATCH at %d: input=%f expected=%f got=%f\n",
                    i, input[i], expected, output[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int n       = 1 << 20;
    const size_t bytes = n * sizeof(float);
    const int threads  = 256;
    const int blocks   = (n + threads - 1) / threads;

    float* h_in  = new float[n];
    float* h_out = new float[n];

    for (int i = 0; i < n; i++)
        h_in[i] = (i % 2 == 0) ? (float)i : -(float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    reluKernel<<<blocks, threads>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("out-of-place ReLU: %s\n", checkReluResult(h_in, h_out, n) ? "PASS" : "FAIL");

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    reluInPlace<<<blocks, threads>>>(d_in, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_in, bytes, cudaMemcpyDeviceToHost));
    printf("in-place ReLU:     %s\n", checkReluResult(h_in, h_out, n) ? "PASS" : "FAIL");

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}