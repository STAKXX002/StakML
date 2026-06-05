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
    Tensor result = a->matmul(*b);
    result.op_name_ = "matmul";
    result.inputs_  = {a, b};

    result.grad();
    auto grad_out = result.grad_;

    result.backward_fn_ = [a, b, grad_out]() {
        // dC shape: {M, N}
        // dA = dC @ B.T  — use matmul_A_BT: no copy of B
        // dB = A.T @ dC  — use matmul_AT_B: no copy of A

        const Tensor& dC = *grad_out;

        Tensor dA = dC.matmul_A_BT(*b);   // dC:{M,N}, b:{K,N} → dA:{M,K}
        Tensor dB = a->matmul_AT_B(dC);   // a:{M,K}, dC:{M,N} → dB:{K,N}

        // accumulate into inputs' grads
        size_t nA = a->num_elements();
        size_t nB = b->num_elements();
        float* gaP = a->grad().raw_ptr();
        float* gbP = b->grad().raw_ptr();
        const float* daP = dA.raw_ptr();
        const float* dbP = dB.raw_ptr();
        for (size_t i = 0; i < nA; ++i) gaP[i] += daP[i];
        for (size_t i = 0; i < nB; ++i) gbP[i] += dbP[i];
    };

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

    // ── backward ─────────────────────────────────────────────────────────────
    result.grad();                    // force-create grad_ before capture
    auto grad_out = result.grad_;

    result.backward_fn_ = [x, bias, grad_out]() {
        size_t batch = x->shape_[0], cols = x->shape_[1];
        const float* gop = grad_out->raw_ptr();   // dL/d_result
        float*       gxp = x->grad().raw_ptr();   // accumulate into x
        float*       gbp = bias->grad().raw_ptr(); // accumulate into bias

        for (size_t i = 0; i < batch; ++i)
            for (size_t j = 0; j < cols; ++j) {
                gxp[i*cols + j] += gop[i*cols + j];  // d_x = d_out (pass-through)
                gbp[j]          += gop[i*cols + j];  // d_bias = sum over batch
            }
    };

    return result;
}

// ── relu ─────────────────────────────────────────────────────────────────────
inline Tensor relu(std::shared_ptr<Tensor> x) {
    Tensor result = x->relu();
    result.op_name_ = "relu";
    result.inputs_  = {x};

    result.grad();                      // ensure grad_ exists before capture
    auto grad_out = result.grad_;       // shared_ptr, safe to capture

    result.backward_fn_ = [x, grad_out]() {
        // d_x += d_out * (x > 0)
        // grad_out holds dL/d_result, we push dL/dx into x->grad()
        size_t n = x->num_elements();
        const float* xp  = x->raw_ptr();
        const float* gop = grad_out->raw_ptr();
        float*       gxp = x->grad().raw_ptr();   // accumulate
        for (size_t i = 0; i < n; ++i)
            gxp[i] += gop[i] * (xp[i] > 0.0f ? 1.0f : 0.0f);
    };

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