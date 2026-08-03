#include "stakml/cuda/kernels.cuh"
#include <cuda_runtime.h>

namespace stakml::cuda {

namespace {
__global__ void relu_forward_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx];
        out[idx] = v > 0.0f ? v : 0.0f;
    }
}
} // namespace

void relu_forward(const float* d_in, float* d_out, int n) {
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(d_in, d_out, n);
}

} // namespace stakml::cuda
