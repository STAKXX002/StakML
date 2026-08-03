#include "stakml/cuda/kernels.cuh"
#include <cuda_runtime.h>

namespace stakml::cuda {

namespace {
constexpr int BLOCK_SIZE = 256;

// grad_output is (rows x cols) row-major.
// One block per output feature (column j). Each block reduces that
// column across all `rows` using shared memory (tree reduction),
// same pattern as the tiled matmul's shared-memory usage.
__global__ void add_bias_backward_kernel(const float* grad_output,
                                          float* grad_bias,
                                          int rows, int cols) {
    __shared__ float partial[BLOCK_SIZE];

    int col = blockIdx.x;
    int tid = threadIdx.x;
    if (col >= cols) return;

    float sum = 0.0f;
    for (int row = tid; row < rows; row += BLOCK_SIZE) {
        sum += grad_output[row * cols + col];
    }
    partial[tid] = sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_bias[col] = partial[0];
    }
}
} // namespace

void add_bias_backward(const float* d_grad_out, float* d_grad_bias, int rows, int cols) {
    // grad_input w.r.t. add_bias is identity (grad_in = grad_out), so the
    // autograd graph can just reuse the same buffer/pointer for that side —
    // no kernel needed for it.
    dim3 grid(cols);
    dim3 block(BLOCK_SIZE);
    add_bias_backward_kernel<<<grid, block>>>(d_grad_out, d_grad_bias, rows, cols);
}

} // namespace stakml::cuda
