# CUDA Sandbox Notes

Personal log of CUDA concepts learned, one section per kernel.

---

## vecadd.cu

**Concept:** thread/block/grid indexing, host↔device memory model

**Key formula:**
idx = blockIdx.x * blockDim.x + threadIdx.x

**Why it works:** GPU and CPU have separate memory. Data must be
cudaMalloc'd on device, cudaMemcpy'd over from host, then processed
by ~1M threads running the same kernel code simultaneously - each
one figures out which single element it owns using only its own
blockIdx/threadIdx (no loop hands out work). Result is copied back
to host memory to read/verify, since host code can't see device
memory directly.

**Result:** PASS (index 5: got 15.000000, expected 15)

---

## relu.cu

**Concept:** elementwise conditional, in-place vs out-of-place writes

**Key formula:**
output[idx] = fmaxf(input[idx], 0.0f)

**Why it works:** Same indexing/bounds-check skeleton as vecadd, but
the per-thread work is a comparison instead of a combine. Two
variants: writing to a separate output buffer vs overwriting the
input buffer directly - same formula, different memory target.

**Result:** PASS (both variants)

---

## add_bias.cu

**Concept:** broadcasting a smaller vector across a larger 2D array;
splitting a flat index back into row/column

**Key formula:**
col = idx % cols
output[idx] = input[idx] + bias[col]

**Why it works:** bias has only `cols` entries (one per output
feature) but gets reused across every row. idx / cols recovers the
row, idx % cols recovers the column - same flat-index pattern as
matmul below, just with one of the two coordinates discarded since
bias only needs the column.

**Result:** PASS

---

## matmul_naive.cu

**Concept:** 2D thread indexing (row via blockIdx.y/threadIdx.y, col
via blockIdx.x/threadIdx.x), dot product per output element,
flattening 2D indices into 1D

**Key formula:**
C[row][col] = sum over k of A[row*K+k] * B[k*N+col]

**Why it works:** Each thread owns one output element and loops
over k itself to accumulate the dot product (the only kernel so far
that needs an internal loop). Flattening rule is the same one used
in add_bias - row_index * num_columns + column_index - applied
separately to A (K columns wide) and B (N columns wide).

**Result:** PASS with square dims (M=K=N=512) - found a B[k*K+col]
vs B[k*N+col] bug that's invisible when K=N, fix pending.

---

## matmul_tiled.cu

**Concept:**

**Key formula:**

**Why it works:**

**Result:**

---