#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// cuda/matmul.cuh — cuBLAS wrappers for StakML's three matmul variants
//
// PHASE 1 (current): cuBLAS wrappers — correct, fast, minimal CUDA code.
// PHASE 2 (next):    replace with a hand-written tiled SGEMM kernel so that
//                    this file has zero external GPU library dependencies.
//
// KEY CONCEPT: cuBLAS is column-major (Fortran order).
// Our tensors are row-major (C order).
//
// The identity that makes this painless:
//   Row-major:    C      = A   @  B      (what we want)
//   Column-major: C.T    = B.T @  A.T   (what cuBLAS computes natively)
//
// So we swap A↔B in the cublasSgemm call and flip the op flags.
// No data is ever transposed in memory — just the argument order changes.
//
// MEMORY STRATEGY: "copy-per-call"
//   Each matmul allocates device buffers, copies host→device, runs SGEMM,
//   copies device→host, then frees. This keeps ALL tensor data on the host
//   (the existing std::vector<float> storage stays untouched).
//
//   Cost:    2 memcpy round-trips per matmul call.
//   Benefit: zero changes to Tensor's memory model, autograd, or any other
//            part of the codebase.
//
//   This is the correct Phase 1 step. Phase 2 will keep tensors device-resident
//   across an entire forward+backward pass to eliminate these round-trips.
// ─────────────────────────────────────────────────────────────────────────────

// ── Error-checking macros ─────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess)                                                 \
            throw std::runtime_error(std::string("CUDA error: ")               \
                + cudaGetErrorString(err) + " at " __FILE__ ":"                \
                + std::to_string(__LINE__));                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st = (call);                                             \
        if (st != CUBLAS_STATUS_SUCCESS)                                        \
            throw std::runtime_error(std::string("cuBLAS error: ")             \
                + std::to_string(st) + " at " __FILE__ ":"                     \
                + std::to_string(__LINE__));                                    \
    } while (0)

// ── Global cuBLAS handle (singleton) ─────────────────────────────────────────
// Created once on first use, destroyed at program exit.
// A single handle is correct for single-threaded training.
inline cublasHandle_t& cublas_handle() {
    static cublasHandle_t handle = []() {
        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        return h;
    }();
    return handle;
}

// ── Device buffer helpers ─────────────────────────────────────────────────────
inline float* cuda_to_device(const float* host, size_t n) {
    float* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev, host, n * sizeof(float), cudaMemcpyHostToDevice));
    return dev;
}

inline void cuda_from_device(float* host, const float* dev, size_t n) {
    CUDA_CHECK(cudaMemcpy(host, dev, n * sizeof(float), cudaMemcpyDeviceToHost));
}

// ─────────────────────────────────────────────────────────────────────────────
// cublas_matmul: C = A @ B
//   A: {M, K}   B: {K, N}   C: {M, N}
//
// Row-major trick:
//   We tell cuBLAS to compute:  C.T = B @ A   (both no-transpose in col-major)
//   which gives us C = A @ B in row-major — exactly what we want.
// ─────────────────────────────────────────────────────────────────────────────
inline void cublas_matmul(
    const float* A, const float* B, float* C,
    size_t M, size_t K, size_t N)
{
    float* dA = cuda_to_device(A, M * K);
    float* dB = cuda_to_device(B, K * N);
    float* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));

    const float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(cublas_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        dB, (int)N,
        dA, (int)K,
        &beta,
        dC, (int)N));

    cuda_from_device(C, dC, M * N);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

// ─────────────────────────────────────────────────────────────────────────────
// cublas_matmul_A_BT: C = A @ B.T
//   A: {M, K}   B: {N, K}   C: {M, N}
//
// Used in backward: dA = dC @ W.T
// ─────────────────────────────────────────────────────────────────────────────
inline void cublas_matmul_A_BT(
    const float* A, const float* B, float* C,
    size_t M, size_t K, size_t N)
{
    float* dA = cuda_to_device(A, M * K);
    float* dB = cuda_to_device(B, N * K);
    float* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));

    const float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(cublas_handle(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        dB, (int)K,
        dA, (int)K,
        &beta,
        dC, (int)N));

    cuda_from_device(C, dC, M * N);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

// ─────────────────────────────────────────────────────────────────────────────
// cublas_matmul_AT_B: C = A.T @ B
//   A: {M, K}   B: {M, N}   C: {K, N}
//
// Used in backward: dW = X.T @ dC
// ─────────────────────────────────────────────────────────────────────────────
inline void cublas_matmul_AT_B(
    const float* A, const float* B, float* C,
    size_t M, size_t K, size_t N)
{
    float* dA = cuda_to_device(A, M * K);
    float* dB = cuda_to_device(B, M * N);
    float* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dC, K * N * sizeof(float)));

    const float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(cublas_handle(),
        CUBLAS_OP_N, CUBLAS_OP_T,
        (int)N, (int)K, (int)M,
        &alpha,
        dB, (int)N,
        dA, (int)K,
        &beta,
        dC, (int)N));

    cuda_from_device(C, dC, K * N);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}