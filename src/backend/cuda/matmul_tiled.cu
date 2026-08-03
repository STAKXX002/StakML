#include "stakml/cuda/kernels.cuh"
#include <cuda_runtime.h>

namespace stakml::cuda {

namespace {
constexpr int TILE = 16;

// A: (M x K), B: (K x N), C: (M x N). Row-major.
// Each block computes a TILE x TILE tile of C by sliding a TILE x TILE
// window across the K dimension, staging each window of A and B into
// shared memory before the inner-product accumulation. Two barriers
// per iteration: one after the load (so no thread reads a stale/half-
// written tile), one after the compute (so no thread overwrites shared
// memory the next iteration's load needs while others are still using it).
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                     int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads(); // barrier 1: whole tile loaded before anyone reads it

        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads(); // barrier 2: everyone's done reading before next load overwrites
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
} // namespace

void matmul_tiled(const float* d_A, const float* d_B, float* d_C, int M, int K, int N) {
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
}

} // namespace stakml::cuda
