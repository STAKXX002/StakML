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

#define TILE_SIZE 16

__global__ void matmulTiled(const float* A, const float* B, float* C,
                             int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // loop over tiles along the K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {

        // each thread loads one element into shared memory
        // guard against out-of-bounds for non-multiple-of-TILE_SIZE sizes
        tileA[threadIdx.y][threadIdx.x] = A[row * K + (t * TILE_SIZE + threadIdx.x)];   // A[row][t * TILE_SIZE + threadIdx.x]
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];   // B[t * TILE_SIZE + threadIdx.y][col]

        _syncthreads();   // sync before using the tile - why?

        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        _syncthreads();   // sync after using the tile - why?
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

bool checkResult(const float* A, const float* B, const float* C,
                 int M, int K, int N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float expected = 0.0f;
            for (int k = 0; k < K; k++)
                expected += A[row * K + k] * B[k * N + col];
            if (fabsf(C[row * N + col] - expected) > 1e-3f) {
                fprintf(stderr, "MISMATCH at [%d,%d]: got %f expected %f\n",
                        row, col, C[row * N + col], expected);
                return false;
            }
        }
    }
    return true;
}

int main() {
    int M = 512, K = 512, N = 512;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];

    for (int i = 0; i < M * K; i++) h_A[i] = (float)i / (M * K);
    for (int i = 0; i < K * N; i++) h_B[i] = (float)i / (K * N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmulTiled<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("matmul tiled: %s  |  time: %.2f ms\n",
           checkResult(h_A, h_B, h_C, M, K, N) ? "PASS" : "FAIL", ms);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}