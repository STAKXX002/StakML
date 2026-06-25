# Tensors

## What is a tensor?

A tensor is just an N-dimensional array with some extra bookkeeping. A scalar is 0-D, a vector is 1-D, a matrix is 2-D, an image batch is 4-D `{batch, channels, height, width}`. The word is borrowed from physics but the concept is straightforward: one flat buffer of floats, plus metadata telling you how to interpret it.

In StakML, every tensor stores its data in a single `std::vector<float>` regardless of how many dimensions it has. The shape and strides are what give that flat array its structure.

## Shape and strides

Consider a 2×3 matrix:

```
  1  2  3
  4  5  6
```

In memory, row-major order, this is just:

```
data = [1, 2, 3, 4, 5, 6]
```

To get element `[i][j]`, you compute:

```
index = i * 3 + j
```

The `3` is the **stride** of the first dimension - how many elements you skip to advance one step along that axis. The `1` is the stride of the second dimension.

StakML stores this as:

```cpp
shape_   = {2, 3}
strides_ = {3, 1}
```

And element access is:

```cpp
element(i, j) = data_[ offset_ + i * strides_[0] + j * strides_[1] ]
              = data_[ offset_ + i * 3            + j * 1           ]
```

This generalizes to any number of dimensions automatically.

## Why strides?

The payoff is zero-copy views. Operations that logically change the layout don't need to move any data - they just change the metadata.

**Reshape:** A 2×3 matrix reshaped to 6×1 or 3×2 keeps the same `data_` pointer. Only `shape_` and `strides_` change.

```cpp
Tensor a({2, 3}, {1,2,3,4,5,6});
Tensor b = a.reshape({6});      // no copy
Tensor c = a.reshape({3, 2});   // no copy
```

**Transpose:** Swapping `shape_[0]↔shape_[1]` and `strides_[0]↔strides_[1]` gives you the transpose. Element `[i][j]` of the transpose is `[j][i]` of the original, and the stride swap makes the indexing arithmetic work out exactly right.

```cpp
// Original: shape={2,3}, strides={3,1}
// Transposed: shape={3,2}, strides={1,3}
// element [i][j] = data_[i*1 + j*3]  ← same as original [j][i] ✓
```

## Shared storage

`data_` is a `shared_ptr<vector<float>>`. Multiple tensors can point at the same underlying buffer. When you reshape or transpose, the result shares the original's data. The last tensor holding a reference to that buffer is the one that frees it - `shared_ptr` handles this automatically.

This is exactly how PyTorch's storage model works. A PyTorch "storage" is the flat buffer; a "tensor" is the view on top.

## Contiguity

A tensor is **contiguous** if its strides match what you'd get from `compute_strides(shape_)` - i.e., standard row-major with no gaps. After a transpose, the tensor is *not* contiguous: strides are `{1, 3}` instead of `{3, 1}`.

This matters for matmul and other operations that want to iterate over the data linearly. If you need a contiguous copy:

```cpp
Tensor t_contig = t.contiguous();  // copies data into a fresh row-major layout
```

## In StakML

```cpp
// Construct
Tensor a({2, 3});                    // zeros
Tensor b({2, 3}, 1.0f);             // fill value
Tensor c({2, 3}, {1,2,3,4,5,6});    // from data

// Zero-copy ops
Tensor r = c.reshape({6});
Tensor t = c.transpose();            // strides swap, no copy

// Element access
float v = c.at({0, 2});             // = 3.0f

// Raw pointer (for loops/matmul)
float* p = c.raw_ptr();

// Shape info
c.shape_          // {2, 3}
c.strides_        // {3, 1}
c.num_elements()  // 6
c.ndim()          // 2
```

## Static factories

```cpp
Tensor::zeros({3, 4})       // all 0.0f
Tensor::ones({3, 4})        // all 1.0f
Tensor::randn({3, 4})       // N(0, 1)
Tensor::xavier({128, 64})   // Glorot init: std = sqrt(2 / (fan_in + fan_out))
```

Xavier init is the right default for Linear layer weights. It keeps the variance of activations stable as signals flow through many layers - without it, activations in deep nets either explode or vanish.