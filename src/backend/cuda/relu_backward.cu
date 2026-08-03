#include "stakml/cuda/kernels.cuh"
#include <cuda_runtime.h>

namespace stakml::cuda {

namespace {
// Gradient only flows through where the forward input was > 0.
__global__ void relu_backward_kernel(const float* in, const float* grad_out,
                                      float* grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = (in[idx] > 0.0f) ? grad_out[idx] : 0.0f;
    }
}
} // namespace

void relu_backward(const float* d_in, const float* d_grad_out, float* d_grad_in, int n) {
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(d_in, d_grad_out, d_grad_in, n);
}

} // namespace stakml::cuda
