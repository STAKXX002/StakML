# StakML CUDA backend

All CUDA-specific headers live here. Nothing in this directory is included
unless the project is built with `-DSTAKML_CUDA=ON`.

## Files

| File | Purpose |
|---|---|
| `matmul.cuh` | Three matmul variants used by `tensor.hpp` (forward, `A@B.T`, `A.T@B`) - cuBLAS-backed |
| `kernels.cuh` | Host-wrapper declarations for the hand-written kernels below |

Hand-written kernel implementations live outside this header directory, in
`src/backend/cuda/`, and are built into the `stakml_cuda_kernels` static lib:

| File | Purpose |
|---|---|
| `vecadd.cu` | Element-wise vector add |
| `relu.cu` / `relu_backward.cu` | ReLU forward and backward |
| `add_bias.cu` / `add_bias_backward.cu` | Bias broadcast forward, and backward (shared-memory column reduction) |
| `matmul_naive.cu` | One-thread-per-output-element matmul |
| `matmul_tiled.cu` | Shared-memory tiled matmul |

## Roadmap

### Phase 1 — cuBLAS wrappers (shipped)
`matmul.cuh` delegates to `cublasSgemm`. Tensors live on the host; each call
does host→device, compute, device→host. Correct and fast. Requires `libcublas`.

### Phase 2 — hand-written kernels (shipped)
`src/backend/cuda/` - no cuBLAS dependency. Covers vecadd, relu (+backward),
add_bias (+backward), and both naive and shared-memory-tiled matmul. Built
whenever `-DSTAKML_CUDA=ON` is set; not yet swapped in as the default matmul
path used by `tensor.hpp` (still Phase 1/cuBLAS for that).

### Phase 3 — elementwise kernels + fused ops
Add `elementwise.cuh` (ReLU, sigmoid, tanh, scalar mul/add) and `reduce.cuh`
(row-softmax, cross-entropy). Eventually fuse activation into the matmul epilogue
to eliminate round-trips between layers.

## Building

```bash
# CPU only (default)
cmake -B build -DSTAKML_CUDA=OFF && cmake --build build

# CUDA backend
cmake -B build -DSTAKML_CUDA=ON && cmake --build build
```

## Row-major ↔ cuBLAS column-major

cuBLAS assumes column-major storage. StakML tensors are row-major.
The identity used throughout `matmul.cuh`:

```
Row-major:    C     = A  @  B
Col-major:    C.T   = B.T @ A.T
```

So we swap A↔B in the `cublasSgemm` argument order and set the op flags
accordingly. No data is transposed in memory — only argument order changes.