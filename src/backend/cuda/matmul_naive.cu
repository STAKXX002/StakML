#include "stakml/cuda/kernels.cuh"
#include <cuda_runtime.h>

namespace stakml::cuda {

namespace {
// A: (M x K), B: (K x N), C: (M x N). All row-major.
// One thread computes one C[row][col].
//
// NOTE: this indexing matters as soon as A and B aren't square —
// A is indexed by K (its width), B is indexed by N (its width).
// Mixing those up silently produces wrong results only when M != K != N,
// which is exactly the bug that hid in earlier square-matrix testing.
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C,
                                     int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
} // namespace

void matmul_naive(const float* d_A, const float* d_B, float* d_C, int M, int K, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);
    matmul_naive_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
}

} // namespace stakml::cuda
