# Matrix Multiply

Matrix multiply is the single most important operation in a neural network. Every Linear layer is one matmul. The speed of matmul determines the speed of training. Understanding *why* the naive version is slow - and how tiling fixes it - is the foundation of GPU programming.

## The naive algorithm

```
C[i][j] = sum_k( A[i][k] * B[k][j] )
```

Three nested loops, O(M·K·N) multiply-adds. For a 1024×1024 square multiply: ~1 billion operations. This is trivially correct but pathologically slow on any modern CPU because of memory access patterns.

## Why the naive version is slow: memory hierarchy

Modern CPUs have three levels of cache:

```
L1 cache:  32 KB,   ~4 cycles latency
L2 cache:  256 KB,  ~12 cycles latency
L3 cache:  8–32 MB, ~40 cycles latency
DRAM:      ∞,       ~200 cycles latency
```

The naive algorithm's inner loop iterates over column `j` of B:

```cpp
for (j = 0; j < N; ++j)
    C[i][j] += A[i][k] * B[k][j];   // B[k][j], B[k][j+1], B[k][j+2]...
```

`B[k][j]` and `B[k][j+1]` are adjacent in memory (row-major). But when we move to the next `k`, we jump a full row - `N` floats - forward in memory. For large N, that row doesn't fit in L1. We get a cache miss on every single `k` iteration, paying 200 cycles for every multiply. The CPU computes at 1 cycle but waits 200. Utilization: 0.5%.

## The fix: blocking (tiling)

The idea: instead of accessing one element of B per inner iteration, load a *tile* (a small submatrix) of B into cache and reuse it as many times as possible before it gets evicted.

```
                  ┌──────┐
                  │ tile │ ← load once, use BLOCK times
                  │  of B│
                  └──────┘
 ┌──────┐  ┌────────────────────────────────┐
 │ tile │  │                                │
 │ of A │  │  partial result for C tile     │
 └──────┘  └────────────────────────────────┘
```

StakML's CPU implementation:

```cpp
const size_t BLOCK = 64;   // tile side - tuned to fit in L1

#pragma omp parallel for schedule(static) collapse(2)
for (size_t i = 0; i < M; i += BLOCK) {
    for (size_t j = 0; j < N; j += BLOCK) {
        for (size_t k = 0; k < K; k += BLOCK) {
            // process the BLOCK×BLOCK tile
            for (size_t ii = i; ii < min(i+BLOCK, M); ++ii) {
                for (size_t kk = k; kk < min(k+BLOCK, K); ++kk) {
                    float a_ik = A[ii*K + kk];   // scalar - stays in register
                    #pragma omp simd
                    for (size_t jj = j; jj < min(j+BLOCK, N); ++jj)
                        C[ii*N + jj] += a_ik * B[kk*N + jj];
                }
            }
        }
    }
}
```

Key details:

- **BLOCK=64**: 64×64 floats = 16 KB - fits comfortably in L1
- **`a_ik` hoisted**: the compiler keeps this in a register across the inner `jj` loop - zero memory reads for A in the inner loop
- **`#pragma omp simd`**: tells the compiler to emit SIMD (AVX2) instructions for the `jj` loop - 8 multiply-adds per cycle instead of 1
- **`collapse(2)`**: OpenMP parallelizes both `i` and `j` loops together - better load balancing across cores than parallelizing just `i`

## The three matmul variants

The backward pass of a Linear layer needs not just `C = A @ B` but also `dA = dC @ W.T` and `dW = X.T @ dC`. Rather than calling `.transpose()` (which would make the tensor non-contiguous and kill cache performance), StakML has three separate blocked kernels that handle the access patterns directly:

| Function | Computes | Used for |
|---|---|---|
| `matmul(A, B)` | `A @ B` | forward pass |
| `matmul_A_BT(A, B)` | `A @ B.T` | backward: `dInput = dOutput @ W.T` |
| `matmul_AT_B(A, B)` | `A.T @ B` | backward: `dW = Input.T @ dOutput` |

Each has its own loop order optimized for the transposed access pattern - no runtime transpose, no non-contiguous strides in the inner loop.

## The CUDA plan

On GPU, the same principle applies but the numbers are different:

```
Global memory (DRAM): ~400 cycle latency, ~900 GB/s bandwidth
Shared memory (SRAM): ~20 cycle latency,  ~19 TB/s bandwidth
Registers:            ~1 cycle
```

The naive GPU kernel (one thread per output element, reads A and B from global memory) issues `K` global reads per thread × `M·N` threads = `M·N·K` global reads total. For 1024³: 1 billion global memory transactions.

The tiled CUDA kernel (Phase 2) follows the same logic as the CPU blocker but using on-chip **shared memory** as the tile buffer:

```
Phase 1 (current):   cuBLAS handles the tiling for us
Phase 2 (planned):   write it ourselves

__shared__ float As[TILE][TILE];   // tile of A in shared mem
__shared__ float Bs[TILE][TILE];   // tile of B in shared mem

// All threads in a block cooperatively load one tile from global mem
As[ty][tx] = A[row * K + (tile * TILE + tx)];
Bs[ty][tx] = B[(tile * TILE + ty) * N + col];
__syncthreads();

// Each thread computes its partial dot product from the tile (fast)
for (int k = 0; k < TILE; ++k)
    acc += As[ty][k] * Bs[k][tx];
__syncthreads();
```

Each global memory byte is loaded once per tile but used `TILE` times. For TILE=16: 16× fewer global memory reads. For TILE=32: 32× fewer. This is the core of all GPU linear algebra.

The Phase 2 implementation will replace `cublas_matmul` in `include/stakml/cuda/matmul.cuh` with a hand-written kernel that does exactly this.