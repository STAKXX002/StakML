#pragma once
#include "tensor.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// ops.hpp — Functional wrappers around Tensor methods.
//
// WHY THIS FILE EXISTS:
//   tensor.hpp knows how to compute things (matmul, relu, etc.).
//   ops.hpp knows how to RECORD what was computed and from what.
//
//   Every function here does two things:
//     1. Calls the underlying Tensor method to get the result data.
//     2. Stamps op_name_ and inputs_ on the result so the graph exists.
//
//   In Week 3, backward_fn_ gets attached here too — this is where
//   autograd lives. The Tensor methods themselves stay graph-unaware.
//
// OWNERSHIP MODEL:
//   Inputs are passed as shared_ptr<Tensor>. The result's inputs_ vector
//   holds those same pointers — no copies, just shared ownership.
//   This means the graph keeps inputs alive as long as any output refers
//   to them, which is exactly what you need for backprop.
// ─────────────────────────────────────────────────────────────────────────────

namespace stakml::ops {

// ── matmul ───────────────────────────────────────────────────────────────────
// (M×K) @ (K×N) → (M×N)
inline Tensor matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    Tensor result = a->matmul(*b);   // compute (existing Tensor logic)
    result.op_name_ = "matmul";
    result.inputs_  = {a, b};        // reference — no copy
    return result;
}

// ── add ──────────────────────────────────────────────────────────────────────
// Element-wise add, identical shapes required.
inline Tensor add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    Tensor result = *a + *b;
    result.op_name_ = "add";
    result.inputs_  = {a, b};
    return result;
}

// ── add_bias ─────────────────────────────────────────────────────────────────
// Broadcast add: x is {batch, N}, bias is {1, N}.
// Adds the same bias row to every row of x.
//
// CONCEPT: this is the most common broadcast pattern in MLPs.
//   x    shape {batch, out}
//   bias shape {1,     out}
//   out  shape {batch, out}   ← bias row repeated batch times
//
// We handle this manually because the general broadcast rules are
// complex; we'll generalise in a later week if needed.
inline Tensor add_bias(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> bias) {
    if (x->ndim() != 2 || bias->ndim() != 2)
        throw std::runtime_error("add_bias: both inputs must be 2-D");
    if (bias->shape_[0] != 1)
        throw std::runtime_error("add_bias: bias must have shape {1, N}");
    if (x->shape_[1] != bias->shape_[1])
        throw std::runtime_error("add_bias: feature dimension mismatch");

    size_t batch = x->shape_[0];
    size_t cols  = x->shape_[1];

    Tensor result({batch, cols});
    const float* xp = x->raw_ptr();
    const float* bp = bias->raw_ptr();
    float*       rp = result.raw_ptr();
    for (size_t i = 0; i < batch; ++i)
        for (size_t j = 0; j < cols; ++j)
            rp[i*cols + j] = xp[i*cols + j] + bp[j];

    result.op_name_ = "add_bias";
    result.inputs_  = {x, bias};
    return result;
}

// ── relu ─────────────────────────────────────────────────────────────────────
inline Tensor relu(std::shared_ptr<Tensor> x) {
    Tensor result = x->relu();
    result.op_name_ = "relu";
    result.inputs_  = {x};
    return result;
}

// ── sigmoid ──────────────────────────────────────────────────────────────────
inline Tensor sigmoid(std::shared_ptr<Tensor> x) {
    Tensor result = x->sigmoid();
    result.op_name_ = "sigmoid";
    result.inputs_  = {x};
    return result;
}

// ── tanh ─────────────────────────────────────────────────────────────────────
inline Tensor tanh_act(std::shared_ptr<Tensor> x) {
    Tensor result = x->tanh_act();
    result.op_name_ = "tanh";
    result.inputs_  = {x};
    return result;
}

// ── softmax ──────────────────────────────────────────────────────────────────
inline Tensor softmax(std::shared_ptr<Tensor> x) {
    Tensor result = x->softmax();
    result.op_name_ = "softmax";
    result.inputs_  = {x};
    return result;
}

// ── mul_scalar ───────────────────────────────────────────────────────────────
// Scalar multiply — used in loss scaling, learning rate application, etc.
inline Tensor mul_scalar(std::shared_ptr<Tensor> x, float scalar) {
    Tensor result = *x * scalar;
    result.op_name_ = "mul_scalar";
    result.inputs_  = {x};
    return result;
}

// ── sub ──────────────────────────────────────────────────────────────────────
// Element-wise subtract, identical shapes required.
inline Tensor sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    Tensor result = *a - *b;
    result.op_name_ = "sub";
    result.inputs_  = {a, b};
    return result;
}

// ── Graph inspection helper ──────────────────────────────────────────────────
// Prints the op chain from a result node back to leaves.
// Useful for debugging — run this after a forward pass to see the graph.
//
//   ops::print_graph(out);
//   // matmul → relu → matmul → [leaf] [leaf] → [leaf] [leaf]
//
inline void print_graph(const Tensor& t, int depth = 0) {
    std::string indent(depth * 2, ' ');
    if (t.op_name_.empty()) {
        std::cout << indent << "[leaf " << t.shape_str() << "]\n";
        return;
    }
    std::cout << indent << t.op_name_ << " → " << t.shape_str() << "\n";
    for (const auto& inp : t.inputs_)
        print_graph(*inp, depth + 1);
}

} // namespace stakml::ops