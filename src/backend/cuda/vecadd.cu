#include "stakml/cuda/kernels.cuh"
#include <cuda_runtime.h>

namespace stakml::cuda {

namespace {
__global__ void vecadd_kernel(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}
} // namespace

void vecadd(const float* d_A, const float* d_B, float* d_C, int n) {
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vecadd_kernel<<<blocks, threads>>>(d_A, d_B, d_C, n);
}

} // namespace stakml::cuda
