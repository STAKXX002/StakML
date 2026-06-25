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

// input:  [rows x cols]  — think batch_size x out_features
// bias:   [cols]         — one value per column
// output: [rows x cols]  — input + bias broadcast across rows
__global__ void addBias(const float* input, const float* bias, float* output,
                        int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        output[idx] = input[idx] + bias[col];
    }
}

bool checkResult(const float* input, const float* bias, const float* output,
                 int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        float expected = input[i] + bias[i % cols];
        if (output[i] != expected) {
            fprintf(stderr, "MISMATCH at %d: got %f expected %f\n",
                    i, output[i], expected);
            return false;
        }
    }
    return true;
}

int main() {
    int rows = 1024;   // batch size
    int cols = 256;    // out_features
    int n    = rows * cols;
    size_t bytes      = n * sizeof(float);
    size_t bias_bytes = cols * sizeof(float);

    float* h_input  = new float[n];
    float* h_bias   = new float[cols];
    float* h_output = new float[n];

    for (int i = 0; i < n;    i++) h_input[i] = (float)i;
    for (int i = 0; i < cols; i++) h_bias[i]  = (float)i * 0.1f;

    float *d_input, *d_bias, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  bytes));
    CUDA_CHECK(cudaMalloc(&d_bias,   bias_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes,      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias,  h_bias,  bias_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    addBias<<<blocks, threads>>>(d_input, d_bias, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    printf("add_bias: %s\n", checkResult(h_input, h_bias, h_output, rows, cols) ? "PASS" : "FAIL");

    cudaFree(d_input); cudaFree(d_bias); cudaFree(d_output);
    delete[] h_input; delete[] h_bias; delete[] h_output;
    return 0;
}