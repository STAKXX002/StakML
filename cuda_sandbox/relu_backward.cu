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

__global__ void reluBackward(const float* grad_output, const float* input,
                              float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = (input[idx] > 0) ? grad_output[idx] : 0.0f;
    }
}

bool checkResult(const float* grad_output, const float* input,
                  const float* grad_input, int n) {
    for (int i = 0; i < n; i++) {
        float expected = (input[i] > 0) ? grad_output[i] : 0.0f;
        if (grad_input[i] != expected) {
            fprintf(stderr, "MISMATCH at %d: got %f expected %f\n",
                    i, grad_input[i], expected);
            return false;
        }
    }
    return true;
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    float* h_input       = new float[n];
    float* h_grad_output = new float[n];
    float* h_grad_input  = new float[n];

    for (int i = 0; i < n; i++) {
        h_input[i]       = (i % 2 == 0) ? (float)i : -(float)i;
        h_grad_output[i] = (float)i * 0.01f;
    }

    float *d_input, *d_grad_output, *d_grad_input;
    CUDA_CHECK(cudaMalloc(&d_input,       bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_output, bytes));
    CUDA_CHECK(cudaMalloc(&d_grad_input,  bytes));
    CUDA_CHECK(cudaMemcpy(d_input,       h_input,       bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    reluBackward<<<blocks, threads>>>(d_grad_output, d_input, d_grad_input, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_grad_input, d_grad_input, bytes, cudaMemcpyDeviceToHost));

    printf("relu_backward: %s\n",
           checkResult(h_grad_output, h_input, h_grad_input, n) ? "PASS" : "FAIL");

    cudaFree(d_input); cudaFree(d_grad_output); cudaFree(d_grad_input);
    delete[] h_input; delete[] h_grad_output; delete[] h_grad_input;
    return 0;
}