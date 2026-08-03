#include "stakml/cuda/kernels.cuh"
#include <cuda_runtime.h>

namespace stakml::cuda {

namespace {
// in/out are (rows x cols) row-major, bias is (cols).
// One thread per output element; bias is broadcast across rows.
__global__ void add_bias_forward_kernel(const float* in, const float* bias,
                                         float* out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;
        out[idx] = in[idx] + bias[col];
    }
}
} // namespace

void add_bias_forward(const float* d_in, const float* d_bias, float* d_out,
                       int rows, int cols) {
    constexpr int threads = 256;
    int total = rows * cols;
    int blocks = (total + threads - 1) / threads;
    add_bias_forward_kernel<<<blocks, threads>>>(d_in, d_bias, d_out, rows, cols);
}

} // namespace stakml::cuda
