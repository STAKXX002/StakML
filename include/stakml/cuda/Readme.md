# StakML CUDA backend

All CUDA-specific headers live here. Nothing in this directory is included
unless the project is built with `-DSTAKML_CUDA=ON`.

## Files

| File | Purpose |
|---|---|
| `matmul.cuh` | Three matmul variants used by `tensor.hpp` (forward, `A@B.T`, `A.T@B`) |

## Roadmap

### Phase 1 — cuBLAS wrappers (current)
`matmul.cuh` delegates to `cublasSgemm`. Tensors live on the host; each call
does host→device, compute, device→host. Correct and fast. Requires `libcublas`.

### Phase 2 — hand-written tiled SGEMM
Replace `cublas_matmul` / `cublas_matmul_A_BT` / `cublas_matmul_AT_B` with a
shared-memory tiled kernel. No cuBLAS dependency. Goal: understand the memory
hierarchy, tile size tuning, and occupancy.

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