# CUDA Sandbox Notes

Personal log of CUDA concepts learned, one section per kernel.

---

## vecadd.cu

**Concept:** thread/block/grid indexing, host↔device memory model

**Key formula:**
idx = blockIdx.x * blockDim.x + threadIdx.x

**Why it works:** GPU and CPU have separate memory. Data must be
cudaMalloc'd on device, cudaMemcpy'd over from host, then processed
by ~1M threads running the same kernel code simultaneously — each
one figures out which single element it owns using only its own
blockIdx/threadIdx (no loop hands out work). Result is copied back
to host memory to read/verify, since host code can't see device
memory directly.

**Result:** PASS (index 5: got 15.000000, expected 15)

---

## relu.cu

**Concept:**

**Key formula:**

**Why it works:**

**Result:**

---