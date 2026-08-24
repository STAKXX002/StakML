#pragma once
#include "ops.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// conv_ops.hpp — im2col, col2im, conv2d_forward
//
// Split out of conv.hpp: this file holds the raw convolution MATH (free
// functions, stakml::ops namespace) — the same category as ops.hpp. The
// nn::Conv2d / nn::MaxPool2d / nn::Flatten LAYER classes that build on top
// of these live in nn.hpp, next to Linear/Dropout/Sequential, since they're
// all Module subclasses. conv.hpp remains as a compatibility include that
// pulls in both, so no existing #include "conv.hpp" needs to change.
//
// CONCEPT: Im2col
//
//   The key insight: convolution IS matrix multiplication in disguise.
//   Instead of a custom conv loop, we REARRANGE the input into a matrix
//   (im2col), then call our existing optimized matmul.
//
//   For each output position (h_out, w_out), the kernel "sees" a patch
//   of the input of shape {C_in, kH, kW}. Im2col flattens each patch
//   into a row. Stack all rows → one big matrix.
//
//   Input:  {N, C_in, H, W}
//   col:    {N*H_out*W_out,  C_in*kH*kW}   ← im2col output
//   W_flat: {C_out,          C_in*kH*kW}   ← weights reshaped
//   out:    {N*H_out*W_out,  C_out}         ← col @ W_flat.T
//   →reshape {N, C_out, H_out, W_out}
//
//   This lets us reuse all the AVX2/cache-optimized matmul code we
//   already wrote. PyTorch's CPU backend does exactly this.
//
// BACKWARD:
//   d_W    = d_out_flat.T @ col            {C_out, C_in*kH*kW}
//   d_col  = d_out_flat  @ W_flat          {N*H_out*W_out, C_in*kH*kW}
//   d_input = col2im(d_col)                {N, C_in, H, W}
//   d_bias  = d_out_flat.sum(axis=0)       {C_out}
//
// ─────────────────────────────────────────────────────────────────────────────

namespace stakml {

// ─────────────────────────────────────────────────────────────────────────────
// NHWC <-> NCHW permute helpers
//
// conv2d_forward's matmul produces NHWC layout (channels last, since that's
// what falls out of col @ W_flat.T), but the layer's public shape is NCHW
// (channels first, matching PyTorch). The forward pass permutes NHWC->NCHW
// on the way out; the backward pass permutes the incoming NCHW gradient
// back to NHWC to do the matching matmul. Same index arithmetic, opposite
// direction — factored out once instead of duplicated in both places.
// ─────────────────────────────────────────────────────────────────────────────
inline void permute_nhwc_to_nchw(const float* src, float* dst,
                                  size_t N, size_t C, size_t H, size_t W)
{
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t n = 0; n < N; ++n)
        for (size_t h = 0; h < H; ++h)
            for (size_t w = 0; w < W; ++w)
                for (size_t c = 0; c < C; ++c) {
                    size_t src_idx = n*(H*W*C) + h*(W*C) + w*C + c;
                    size_t dst_idx = n*(C*H*W) + c*(H*W) + h*W + w;
                    dst[dst_idx] = src[src_idx];
                }
}

inline void permute_nchw_to_nhwc(const float* src, float* dst,
                                  size_t N, size_t C, size_t H, size_t W)
{
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < C; ++c)
            for (size_t h = 0; h < H; ++h)
                for (size_t w = 0; w < W; ++w) {
                    size_t src_idx = n*(C*H*W) + c*(H*W) + h*W + w;
                    size_t dst_idx = n*(H*W*C) + h*(W*C) + w*C + c;
                    dst[dst_idx] = src[src_idx];
                }
}

// ─────────────────────────────────────────────────────────────────────────────
// im2col
//
// Converts a 4D input tensor {N, C, H, W} into a 2D column matrix
// {N*H_out*W_out, C*kH*kW} where each row is one flattened receptive field.
//
// Parameters:
//   input   : {N, C_in, H, W}
//   kH, kW  : kernel height and width
//   stride  : step size (we use 1 throughout, standard for 3x3 conv)
//   padding : zero-padding applied to H and W dimensions
//
// Output shape: {N * H_out * W_out,  C_in * kH * kW}
//   where H_out = (H + 2*padding - kH) / stride + 1
//         W_out = (W + 2*padding - kW) / stride + 1
//
// Memory layout of each row (C_in * kH * kW elements):
//   [c=0,kh=0,kw=0], [c=0,kh=0,kw=1], ..., [c=0,kh=kH-1,kw=kW-1],
//   [c=1,kh=0,kw=0], ..., [c=C_in-1,kh=kH-1,kw=kW-1]
//
// Out-of-bounds accesses (from padding) write 0.0f — zero padding.
// ─────────────────────────────────────────────────────────────────────────────
inline Tensor im2col(const Tensor& input,
                     size_t kH, size_t kW,
                     size_t stride  = 1,
                     size_t padding = 0)
{
    // input must be 4D: {N, C, H, W}
    if (input.ndim() != 4)
        throw std::runtime_error("im2col: input must be 4-D {N, C, H, W}");

    size_t N  = input.shape_[0];
    size_t C  = input.shape_[1];
    size_t H  = input.shape_[2];
    size_t W  = input.shape_[3];

    // Output spatial dimensions
    size_t H_out = (H + 2 * padding - kH) / stride + 1;
    size_t W_out = (W + 2 * padding - kW) / stride + 1;

    // Each row of the column matrix = one flattened receptive field
    size_t row_len = C * kH * kW;
    size_t num_rows = N * H_out * W_out;

    Tensor col({num_rows, row_len}, 0.0f);

    const float* inp = input.raw_ptr();
    float* out = col.raw_ptr();

    // ── NEW: Parallelize the patch extraction ───────────────────────────────
    #pragma omp parallel for schedule(static) collapse(3)
    for (size_t n = 0; n < N; ++n) {
        for (size_t h_out = 0; h_out < H_out; ++h_out) {
            for (size_t w_out = 0; w_out < W_out; ++w_out) {

                size_t row_idx = n * (H_out * W_out) + h_out * W_out + w_out;
                float* col_row = out + row_idx * row_len;

                size_t col_idx = 0;
                for (size_t c = 0; c < C; ++c) {
                    for (size_t kh = 0; kh < kH; ++kh) {
                        for (size_t kw = 0; kw < kW; ++kw) {

                            long h_in = static_cast<long>(h_out * stride + kh) - static_cast<long>(padding);
                            long w_in = static_cast<long>(w_out * stride + kw) - static_cast<long>(padding);

                            if (h_in >= 0 && h_in < static_cast<long>(H) &&
                                w_in >= 0 && w_in < static_cast<long>(W))
                            {
                                size_t inp_idx = n * (C * H * W)
                                               + c * (H * W)
                                               + static_cast<size_t>(h_in) * W
                                               + static_cast<size_t>(w_in);
                                col_row[col_idx] = inp[inp_idx];
                            }
                            ++col_idx;
                        }
                    }
                }
            }
        }
    }

    return col;
}

// ─────────────────────────────────────────────────────────────────────────────
// col2im
//
// Inverse of im2col — used in the backward pass to convert d_col back to
// the input gradient shape {N, C_in, H, W}.
//
// IMPORTANT: col2im is NOT a true inverse. Multiple col rows can overlap
// (when stride < kernel_size), so we ACCUMULATE (+=) contributions.
// This is correct for gradient computation: overlapping receptive fields
// each contribute to the same input element's gradient.
// ─────────────────────────────────────────────────────────────────────────────
inline Tensor col2im(const Tensor& col,
                     size_t N, size_t C, size_t H, size_t W,
                     size_t kH, size_t kW,
                     size_t stride  = 1,
                     size_t padding = 0)
{
    size_t H_out = (H + 2 * padding - kH) / stride + 1;
    size_t W_out = (W + 2 * padding - kW) / stride + 1;
    size_t row_len = C * kH * kW;

    Tensor d_input({N, C, H, W}, 0.0f);  // accumulate into this

    const float* col_ptr = col.raw_ptr();
    float* din_ptr = d_input.raw_ptr();

    // ── NEW: Parallelize over the batch dimension only (thread-safe) ────────
    #pragma omp parallel for schedule(static)
    for (size_t n = 0; n < N; ++n) {
        for (size_t h_out = 0; h_out < H_out; ++h_out) {
            for (size_t w_out = 0; w_out < W_out; ++w_out) {

                size_t row_idx = n * (H_out * W_out) + h_out * W_out + w_out;
                const float* col_row = col_ptr + row_idx * row_len;

                size_t col_idx = 0;
                for (size_t c = 0; c < C; ++c) {
                    for (size_t kh = 0; kh < kH; ++kh) {
                        for (size_t kw = 0; kw < kW; ++kw) {

                            long h_in = static_cast<long>(h_out * stride + kh) - static_cast<long>(padding);
                            long w_in = static_cast<long>(w_out * stride + kw) - static_cast<long>(padding);

                            if (h_in >= 0 && h_in < static_cast<long>(H) &&
                                w_in >= 0 && w_in < static_cast<long>(W))
                            {
                                size_t din_idx = n * (C * H * W)
                                               + c * (H * W)
                                               + static_cast<size_t>(h_in) * W
                                               + static_cast<size_t>(w_in);
                                din_ptr[din_idx] += col_row[col_idx];
                            }

                            ++col_idx;
                        }
                    }
                }
            }
        }
    }

    return d_input;
}


// ─────────────────────────────────────────────────────────────────────────────
// ops wrappers for Conv2d forward/backward
// ─────────────────────────────────────────────────────────────────────────────
namespace ops {

// conv2d_forward:
//   x    : shared_ptr to {N, C_in, H, W}
//   W    : shared_ptr to {C_out, C_in, kH, kW}  (kernel weights)
//   bias : shared_ptr to {C_out}
//   Returns Tensor of shape {N, C_out, H_out, W_out}
//
inline Tensor conv2d_forward(std::shared_ptr<Tensor> x,
                             std::shared_ptr<Tensor> W,
                             std::shared_ptr<Tensor> bias,
                             size_t stride  = 1,
                             size_t padding = 0)
{
    size_t N    = x->shape_[0];
    size_t C_in = x->shape_[1];
    size_t H    = x->shape_[2];
    size_t Ww   = x->shape_[3];  // input width (Ww to avoid shadowing W param)

    size_t C_out = W->shape_[0];
    size_t kH    = W->shape_[2];
    size_t kW    = W->shape_[3];

    size_t H_out = (H + 2 * padding - kH) / stride + 1;
    size_t W_out = (Ww + 2 * padding - kW) / stride + 1;

    // ── 1. im2col: rearrange input patches into rows ──────────────────────────
    // col shape: {N*H_out*W_out,  C_in*kH*kW}
    Tensor col = im2col(*x, kH, kW, stride, padding);

    // ── 2. Flatten W: {C_out, C_in, kH, kW} → {C_out, C_in*kH*kW} ──────────
    // W is already contiguous (we init it that way), so reshape is zero-copy.
    size_t kernel_flat = C_in * kH * kW;
    Tensor W_flat = W->reshape({C_out, kernel_flat});

    // ── 3. Matmul: col @ W_flat.T → {N*H_out*W_out, C_out} ──────────────────
    // col:    {N*H_out*W_out, C_in*kH*kW}
    // W_flat: {C_out,         C_in*kH*kW}   ← we need its transpose
    // Use matmul_A_BT: this @ B.T where B=W_flat
    Tensor out_flat = col.matmul_A_BT(W_flat);
    // out_flat shape: {N*H_out*W_out, C_out}

    // ── 4. Add bias: broadcast {C_out} across all rows ───────────────────────
    {
        float*       op  = out_flat.raw_ptr();
        const float* bp  = bias->raw_ptr();
        size_t rows = N * H_out * W_out;
        #pragma omp parallel for schedule(static) collapse(2)
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < C_out; ++c)
                op[r * C_out + c] += bp[c];
    }

    // ── 5. Reshape to {N, C_out, H_out, W_out} ───────────────────────────────
    // out_flat is {N*H_out*W_out, C_out} — we need {N, C_out, H_out, W_out}.
    // But standard row-major layout of {N*H_out*W_out, C_out} is actually
    // {N, H_out, W_out, C_out} logically — channels are the LAST dim.
    // PyTorch uses NCHW (channels first), so we need to permute.
    //
    // We do this with a shared helper (no in-place permute yet):
    //   NHWC[n,h,w,c] → NCHW[n,c,h,w]
    Tensor result({N, C_out, H_out, W_out}, 0.0f);
    permute_nhwc_to_nchw(out_flat.raw_ptr(), result.raw_ptr(), N, C_out, H_out, W_out);

    // ── 6. Stamp graph node ───────────────────────────────────────────────────
    result.op_name_ = "conv2d";
    result.inputs_  = {x, W, bias};

    result.grad_ = std::make_shared<Tensor>(result.shape_, 0.0f);
    auto grad_out = result.grad_;

    // Capture everything the backward needs
    result.backward_fn_ = [x, W, bias, grad_out,
                           col, W_flat,               // saved for backward
                           N, C_in, H, Ww,
                           C_out, kH, kW,
                           H_out, W_out,
                           kernel_flat,
                           stride, padding]() mutable
    {
        // grad_out is {N, C_out, H_out, W_out} in NCHW
        // Convert back to NHWC flat {N*H_out*W_out, C_out} for math

        Tensor d_out_flat({N * H_out * W_out, C_out}, 0.0f);
        permute_nchw_to_nhwc(grad_out->raw_ptr(), d_out_flat.raw_ptr(), N, C_out, H_out, W_out);

        // ── dW: {C_out, C_in*kH*kW} = d_out_flat.T @ col ────────────────────
        // d_out_flat: {N*H_out*W_out, C_out}
        // col:        {N*H_out*W_out, C_in*kH*kW}
        // dW_flat = d_out_flat.T @ col → {C_out, kernel_flat}
        Tensor dW_flat = d_out_flat.matmul_AT_B(col);

        // Accumulate into W->grad (W is {C_out, C_in, kH, kW} = same elements)
        float*       gW = W->grad().raw_ptr();
        const float* dw = dW_flat.raw_ptr();
        size_t nW = W->num_elements();
        accumulate(gW, dw, nW);

        // ── dbias: sum d_out_flat over all rows → {C_out} ────────────────────
        // Parallelized over c (each thread owns one output channel's sum),
        // not over r — summing over r into a shared gb[c] across threads
        // would race, same hazard as ops::add_bias's backward.
        float*       gb   = bias->grad().raw_ptr();
        const float* dof  = d_out_flat.raw_ptr();
        size_t rows = N * H_out * W_out;
        #pragma omp parallel for schedule(static)
        for (size_t c = 0; c < C_out; ++c) {
            float sum = 0.0f;
            for (size_t r = 0; r < rows; ++r)
                sum += dof[r * C_out + c];
            gb[c] += sum;
        }

        // ── d_col: {N*H_out*W_out, C_in*kH*kW} = d_out_flat @ W_flat ────────
        // W_flat: {C_out, kernel_flat}
        // d_col  = d_out_flat @ W_flat  (NOT transposed)
        Tensor d_col = d_out_flat.matmul(W_flat);

        // ── d_input: col2im(d_col) → {N, C_in, H, W} ────────────────────────
        Tensor d_input = col2im(d_col, N, C_in, H, Ww, kH, kW, stride, padding);

        float*       gx  = x->grad().raw_ptr();
        const float* di  = d_input.raw_ptr();
        size_t nX = x->num_elements();
        accumulate(gx, di, nX);
    };

    return result;
}

} // namespace ops
} // namespace stakml
