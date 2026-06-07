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

// ── Base Module ──────────────────────────────────────────────────────────────
struct Module {
    virtual ~Module() = default;
    
    // Every layer takes a shared pointer and returns a new Tensor node
    virtual Tensor forward(std::shared_ptr<Tensor> x) = 0;
    
    // By default, a layer has no parameters (e.g., ReLU)
    virtual std::vector<std::shared_ptr<Tensor>> parameters() const {
        return {}; 
    }
};

// ── Linear ───────────────────────────────────────────────────────────────────
// y = x @ W + b
//
//   x : {batch, in_features}
//   W : {in_features, out_features}
//   b : {1, out_features}
//   y : {batch, out_features}
//
struct Linear : public Module {
    size_t in_features;
    size_t out_features;

    std::shared_ptr<Tensor> W;
    std::shared_ptr<Tensor> b;

    Linear(size_t in, size_t out) : in_features(in), out_features(out) {
        W = std::make_shared<Tensor>(Tensor::xavier({in, out}));
        b = std::make_shared<Tensor>(Tensor::zeros({1, out}));
        W->requires_grad_ = true;
        b->requires_grad_ = true;
    }

    Tensor forward(std::shared_ptr<Tensor> x) override {
        if (x->ndim() != 2)
            throw std::runtime_error("Linear::forward: input must be 2-D {batch, in}");
        if (x->shape_[1] != in_features)
            throw std::runtime_error("Linear::forward: input cols must equal in_features");

        auto xW = std::make_shared<Tensor>(ops::matmul(x, W));
        return ops::add_bias(xW, b);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const override {
        return {W, b};
    }

    void print_info() const {
        std::cout << "Linear(" << in_features << " → " << out_features << ")"
                  << "  W:" << W->shape_str()
                  << "  b:" << b->shape_str() << "\n";
    }
};

// ── Activations ──────────────────────────────────────────────────────────────
struct ReLU : public Module {
    Tensor forward(std::shared_ptr<Tensor> x) override {
        return ops::relu(x);
    }
};

struct Sigmoid : public Module {
    Tensor forward(std::shared_ptr<Tensor> x) override {
        return ops::sigmoid(x);
    }
};

// ── Dropout ──────────────────────────────────────────────────────────────────
// Inverted dropout: during training, randomly zeros activations with
// probability p, and scales survivors by 1/(1-p) so the expected
// magnitude is unchanged at test time.
//
// At inference (training_ = false), forward() is a pure pass-through.
// You must call set_training(false) before your test loop.
//
// BACKWARD: gradient flows through only the positions that weren't
// zeroed in the forward pass — same mask, same scale.
//
struct Dropout : public Module {
    float p_;           // drop probability (e.g. 0.5)
    bool  training_;
    std::mt19937 rng_;

    explicit Dropout(float p = 0.5f)
        : p_(p), training_(true), rng_(std::random_device{}()) {}

    void set_training(bool mode) { training_ = mode; }

    Tensor forward(std::shared_ptr<Tensor> x) override {
        if (!training_) return *x;   // test mode: identity

        size_t n = x->num_elements();
        float scale = 1.0f / (1.0f - p_);

        // Build the mask once — save it for backward
        auto mask = std::make_shared<std::vector<float>>(n);
        std::bernoulli_distribution dist(1.0f - p_);
        const float* xp = x->raw_ptr();
        float*       mp = mask->data();
        for (size_t i = 0; i < n; ++i)
            mp[i] = dist(rng_) ? scale : 0.0f;

        Tensor result(x->shape_);
        float* rp = result.raw_ptr();
        for (size_t i = 0; i < n; ++i)
            rp[i] = xp[i] * mp[i];

        result.op_name_ = "dropout";
        result.inputs_  = {x};
        result.grad();
        auto grad_out = result.grad_;

        result.backward_fn_ = [x, grad_out, mask]() {
            size_t n = x->num_elements();
            const float* gop = grad_out->raw_ptr();
            const float* mp  = mask->data();
            float*       gxp = x->grad().raw_ptr();
            for (size_t i = 0; i < n; ++i)
                gxp[i] += gop[i] * mp[i];  // same mask, same scale
        };

        return result;
    }
};

// ── Sequential ───────────────────────────────────────────────────────────────
// CONCEPT: chains layers so you can write:
//
//   Sequential model({ fc1, relu_layer, fc2, softmax_layer });
//   Tensor out = model.forward(x);
//
struct Sequential : public Module {
    std::vector<std::shared_ptr<Module>> layers;

    // Constructor takes an initializer list of shared_ptr<Module>
    Sequential(std::initializer_list<std::shared_ptr<Module>> init) 
        : layers(init) {}

    Tensor forward(std::shared_ptr<Tensor> x) override {
        if (layers.empty()) throw std::runtime_error("Sequential model is empty");

        std::shared_ptr<Tensor> current_input = x;
        
        for (size_t i = 0; i < layers.size(); ++i) {
            Tensor out = layers[i]->forward(current_input);
            
            // If it's the last layer, just return the final Tensor
            if (i == layers.size() - 1) {
                return out;
            }
            // Otherwise, wrap it in a shared_ptr to pass to the next layer
            current_input = std::make_shared<Tensor>(out);
        }
        return *current_input; // Fallback
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const override {
        std::vector<std::shared_ptr<Tensor>> all_params;
        for (const auto& layer : layers) {
            auto layer_params = layer->parameters();
            all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
        }
        return all_params;
    }

    void set_training(bool mode) {
        for (auto& layer : layers) {
            // dynamic_cast returns nullptr if the layer isn't a Dropout — safe
            if (auto* d = dynamic_cast<Dropout*>(layer.get()))
                d->set_training(mode);
        }
    }
};

} // namespace nn
} // namespace stakml