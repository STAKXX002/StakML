# Convolutions

## The key insight: convolution is matrix multiply

A 2D convolution slides a small kernel (say 3×3) over an image and computes a dot product at every position. The naive implementation is six nested loops. The StakML implementation is zero lines of custom math - it reuses the existing `matmul` by rearranging the input first.

This rearrangement is called **im2col** ("image to columns"). It's how PyTorch's CPU backend works, how cuDNN's implicit GEMM works, and how Caffe originally implemented convolutions.

## im2col

For each position where the kernel will land, extract the patch it covers and flatten it into a row. Stack all rows into a matrix.

```
Input: {N, C_in, H, W}
Kernel: {kH, kW}

For each image n, for each output position (h_out, w_out):
  → extract the patch that the kernel covers: {C_in, kH, kW}
  → flatten it into a row of C_in * kH * kW elements

Result "col": {N * H_out * W_out,  C_in * kH * kW}
```

Where:
```
H_out = (H + 2*padding - kH) / stride + 1
W_out = (W + 2*padding - kW) / stride + 1
```

Out-of-bounds positions (from padding) contribute 0 - zero padding.

Then reshape the weights:

```
W_flat: {C_out,  C_in * kH * kW}
```

The entire convolution becomes:

```
out_flat = col @ W_flat.T     {N*H_out*W_out,  C_out}
```

Reshape `out_flat` back to `{N, C_out, H_out, W_out}` and you're done. One matmul.

## Why this works

A dot product of the flattened kernel with a flattened patch is exactly what a convolution computes at one output position. Stacking all patches as rows and all kernels as rows of `W_flat` lets a single `matmul` compute all output positions simultaneously.

The cost is memory: `col` can be large (`N * H_out * W_out * C_in * kH * kW` floats). For a typical CIFAR-10 first layer: `32 * 32 * 32 * 3 * 3 * 3 = 884,736` floats = 3.4 MB per batch. Acceptable. For very large inputs with large kernels, alternatives like Winograd or FFT-based convolution may be preferable.

## Conv2d

```cpp
nn::Conv2d conv(
    /*C_in=*/  3,
    /*C_out=*/ 32,
    /*kH=*/    3,
    /*kW=*/    3,
    /*stride=*/   1,
    /*padding=*/  1   // 'same' padding for 3×3
);

// Input: {N, 3, 32, 32}
Tensor out = conv.forward(x_ptr);
// Output: {N, 32, 32, 32}  (same spatial size with padding=1)
```

Internally:

```cpp
Tensor col     = im2col(input, kH, kW, stride, padding);
Tensor out_flat = ops::matmul(col_ptr, W_flat_ptr);   // col @ W.T
// + bias broadcast
// reshape to {N, C_out, H_out, W_out}
```

**Weight init:** Xavier on `{C_out, C_in * kH * kW}` reshaped.

**Backward:** three gradients to compute:

```
d_W   = d_out_flat.T @ col          {C_out, C_in*kH*kW}
d_col = d_out_flat  @ W_flat        {N*H_out*W_out, C_in*kH*kW}
d_input = col2im(d_col)             {N, C_in, H, W}
d_bias  = d_out_flat.sum(axis=0)    {C_out}
```

`col2im` is the inverse of `im2col`: scatter values back to their original input positions, accumulating where patches overlap.

## MaxPool2d

Downsamples spatial dimensions by taking the max over non-overlapping windows:

```cpp
nn::MaxPool2d pool(/*kH=*/2, /*kW=*/2, /*stride=*/2);

// Input:  {N, C, H, W}
// Output: {N, C, H/2, W/2}
```

No learnable parameters. Backward just routes the gradient to the position that held the maximum value (all other positions get zero gradient - max is a non-smooth function, but the subgradient convention is to put all mass at the argmax).

## Flatten

Reshapes `{N, C, H, W}` → `{N, C*H*W}` for the transition from conv layers to fully connected layers:

```cpp
nn::Flatten flatten;
Tensor flat = flatten.forward(x_ptr);   // {N, C*H*W}
```

## Typical CNN pattern

```cpp
nn::Sequential model({
    std::make_shared<nn::Conv2d>(1, 32, 3, 3, 1, 1),   // {N,1,28,28} → {N,32,28,28}
    std::make_shared<nn::ReLU>(),
    std::make_shared<nn::MaxPool2d>(2, 2, 2),           // → {N,32,14,14}
    std::make_shared<nn::Conv2d>(32, 64, 3, 3, 1, 1),  // → {N,64,14,14}
    std::make_shared<nn::ReLU>(),
    std::make_shared<nn::MaxPool2d>(2, 2, 2),           // → {N,64,7,7}
    std::make_shared<nn::Flatten>(),                    // → {N,3136}
    std::make_shared<nn::Linear>(3136, 128),
    std::make_shared<nn::ReLU>(),
    std::make_shared<nn::Linear>(128, 10),
});
```

See `examples/cifar_cnn.cpp` for a working CIFAR-10 training loop.

## im2col parallelism

The patch extraction loop is parallelized with OpenMP:

```cpp
#pragma omp parallel for schedule(static) collapse(3)
for (n ...) for (h_out ...) for (w_out ...) {
    // extract patch into col row - independent, no races
}
```

Each output position is independent (reads from input, writes to a distinct row of `col`), so this parallelizes perfectly with no synchronization.