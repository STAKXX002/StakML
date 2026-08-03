#pragma once
// StakML hand-written CUDA kernels (Phase 2).
// Each .cu file below owns one kernel + a thin host-side launch wrapper.
// These are deliberately separate from the cuBLAS Phase-1 path so they
// can be benchmarked and swapped in independently.

namespace stakml::cuda {

// C[i] = A[i] + B[i], all length n
void vecadd(const float* d_A, const float* d_B, float* d_C, int n);

// out[i] = max(0, in[i]), length n
void relu_forward(const float* d_in, float* d_out, int n);

// grad_in[i] = (in[i] > 0) ? grad_out[i] : 0, length n
void relu_backward(const float* d_in, const float* d_grad_out, float* d_grad_in, int n);

// out[i][j] = in[i][j] + bias[j]   -- in is (rows x cols), bias is (cols)
void add_bias_forward(const float* d_in, const float* d_bias, float* d_out, int rows, int cols);

// grad_bias[j] = sum_i grad_out[i][j]  -- reduction down each column
void add_bias_backward(const float* d_grad_out, float* d_grad_bias, int rows, int cols);

// C = A @ B, naive one-thread-per-output-element (A: MxK, B: KxN, C: MxN)
void matmul_naive(const float* d_A, const float* d_B, float* d_C, int M, int K, int N);

// C = A @ B, shared-memory tiled version (A: MxK, B: KxN, C: MxN)
void matmul_tiled(const float* d_A, const float* d_B, float* d_C, int M, int K, int N);

} // namespace stakml::cuda
