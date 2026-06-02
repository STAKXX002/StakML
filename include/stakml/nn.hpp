#pragma once
#include "ops.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// nn.hpp — neural network layers
//
// CONCEPT: What is a Layer?
//
//   A layer owns its parameters (W, b) as shared_ptr<Tensor>.
//   shared_ptr because:
//     - The graph nodes (inputs_) also hold ptrs to the same W, b
//     - When Week 3 accumulates gradients into W->grad_, both the
//       layer and the graph node see the same object — no sync needed
//
//   forward() takes shared_ptr<Tensor> in, returns Tensor out.
//   The returned Tensor has op_name_ and inputs_ set (via ops::).
//   That's the computation graph being built — one node per forward().
//
// ─────────────────────────────────────────────────────────────────────────────

namespace stakml {
namespace nn {

// ── Linear ───────────────────────────────────────────────────────────────────
// y = x @ W + b
//
//   x : {batch, in_features}
//   W : {in_features, out_features}
//   b : {1, out_features}
//   y : {batch, out_features}
//
// NOTE on W shape:
//   W is {in, out}, so x @ W works directly as {batch,in} @ {in,out} → {batch,out}.
//   PyTorch stores W as {out, in} and does x @ W.T — same math, different convention.
//   Ours is simpler for now; we'll revisit if BLAS tiling matters later.
//
struct Linear {
    size_t in_features;
    size_t out_features;

    std::shared_ptr<Tensor> W;   // {in_features, out_features}
    std::shared_ptr<Tensor> b;   // {1, out_features}

    Linear(size_t in, size_t out)
        : in_features(in), out_features(out)
    {
        W = std::make_shared<Tensor>(Tensor::xavier({in, out}));
        b = std::make_shared<Tensor>(Tensor::zeros({1, out}));
    }

    // x: {batch, in_features} → {batch, out_features}
    Tensor forward(std::shared_ptr<Tensor> x) const {
        if (x->ndim() != 2)
            throw std::runtime_error("Linear::forward: input must be 2-D {batch, in}");
        if (x->shape_[1] != in_features)
            throw std::runtime_error("Linear::forward: input cols must equal in_features");

        // x @ W  →  {batch, out_features}
        auto xW = std::make_shared<Tensor>(ops::matmul(x, W));

        // xW + b  →  {batch, out_features}  (broadcast over batch dim)
        return ops::add_bias(xW, b);
    }

    // Collect parameters — Week 4 optimizer will iterate these
    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return {W, b};
    }

    void print_info() const {
        std::cout << "Linear(" << in_features << " → " << out_features << ")"
                  << "  W:" << W->shape_str()
                  << "  b:" << b->shape_str() << "\n";
    }
};

// ── Sequential ───────────────────────────────────────────────────────────────
// CONCEPT: chains layers so you can write:
//
//   Sequential model({ fc1, relu_layer, fc2, softmax_layer });
//   Tensor out = model.forward(x);
//
// For now this holds Linear layers + named activations.
// A proper design uses a base Layer interface — that's fine for Week 4.
// For Week 2 we just need shape flow to work end-to-end.
//
// Usage:
//   auto out = fc1.forward(x_ptr);               // manual chaining
//   auto h   = ops::relu(make_shared<Tensor>(out));
//   auto out2 = fc2.forward(make_shared<Tensor>(h));
//
// Sequential is optional sugar — build the MLP manually first.
//

} // namespace nn
} // namespace stakml