#pragma once
#include "ops.hpp"
#include "conv_ops.hpp"   // ops::conv2d_forward, used by nn::Conv2d below

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

// ── Conv2d ───────────────────────────────────────────────────────────────────
// Standard 2D convolution layer.
//
//   in_channels  : C_in  (e.g. 3 for RGB)
//   out_channels : C_out (number of filters)
//   kernel_size  : kH = kW (square kernels only for now)
//   stride       : step size (default 1)
//   padding      : zero-padding (default 0; use 1 with 3×3 for "same" padding)
//
// Weights: {C_out, C_in, kH, kW}  — Xavier init
// Bias:    {C_out}
//
struct Conv2d : public Module {
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;

    std::shared_ptr<Tensor> W;     // {C_out, C_in, kH, kW}
    std::shared_ptr<Tensor> bias;  // {C_out}

    Conv2d(size_t in_ch, size_t out_ch, size_t ksize,
           size_t stride = 1, size_t padding = 0)
        : in_channels(in_ch), out_channels(out_ch),
          kernel_size(ksize), stride(stride), padding(padding)
    {
        // Xavier init for 4D kernel
        // fan_in  = C_in  * kH * kW
        // fan_out = C_out * kH * kW
        W    = std::make_shared<Tensor>(
                   Tensor::xavier({out_ch, in_ch, ksize, ksize}));
        bias = std::make_shared<Tensor>(
                   Tensor::zeros({out_ch}));

        W->requires_grad_    = true;
        bias->requires_grad_ = true;
    }

    Tensor forward(std::shared_ptr<Tensor> x) override {
        if (x->ndim() != 4)
            throw std::runtime_error("Conv2d::forward: input must be 4-D {N, C, H, W}");
        if (x->shape_[1] != in_channels)
            throw std::runtime_error("Conv2d::forward: input channel mismatch");

        return ops::conv2d_forward(x, W, bias, stride, padding);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const override {
        return {W, bias};
    }

    void print_info() const {
        std::cout << "Conv2d(" << in_channels << " → " << out_channels
                  << ", kernel=" << kernel_size << "x" << kernel_size
                  << ", stride=" << stride << ", padding=" << padding << ")\n";
    }
};


// ── MaxPool2d ─────────────────────────────────────────────────────────────────
// 2×2 max pooling with stride=2 (halves spatial dimensions).
//
// Forward: for each 2×2 window, take the maximum value.
// Backward: gradient flows only through the position that held the max.
//           (other positions get zero gradient — they didn't contribute)
//
// We save the "argmax mask" during forward so backward knows where to route.
//
struct MaxPool2d : public Module {
    size_t pool_size;   // typically 2
    size_t pool_stride; // typically 2

    MaxPool2d(size_t size = 2, size_t stride = 2)
        : pool_size(size), pool_stride(stride) {}

    Tensor forward(std::shared_ptr<Tensor> x) override {
        if (x->ndim() != 4)
            throw std::runtime_error("MaxPool2d::forward: input must be 4-D {N,C,H,W}");

        size_t N    = x->shape_[0];
        size_t C    = x->shape_[1];
        size_t H    = x->shape_[2];
        size_t W    = x->shape_[3];

        size_t H_out = (H - pool_size) / pool_stride + 1;
        size_t W_out = (W - pool_size) / pool_stride + 1;

        Tensor result({N, C, H_out, W_out}, 0.0f);

        // mask stores the flat index in x->data_ of the winning element
        // per output position — needed for backward
        auto mask = std::make_shared<std::vector<size_t>>(N * C * H_out * W_out);

        const float* inp = x->raw_ptr();
        float*       out = result.raw_ptr();
        size_t*      msk = mask->data();

        for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < C; ++c)
        for (size_t h = 0; h < H_out; ++h)
        for (size_t w = 0; w < W_out; ++w)
        {
            size_t out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + h*W_out + w;

            // Scan the pool_size × pool_size window
            float  best_val = -std::numeric_limits<float>::infinity();
            size_t best_inp = 0;

            for (size_t ph = 0; ph < pool_size; ++ph)
            for (size_t pw = 0; pw < pool_size; ++pw)
            {
                size_t h_in = h * pool_stride + ph;
                size_t w_in = w * pool_stride + pw;
                size_t inp_idx = n*(C*H*W) + c*(H*W) + h_in*W + w_in;

                if (inp[inp_idx] > best_val) {
                    best_val = inp[inp_idx];
                    best_inp = inp_idx;
                }
            }

            out[out_idx] = best_val;
            msk[out_idx] = best_inp;
        }

        result.op_name_ = "maxpool2d";
        result.inputs_  = {x};

        result.grad_ = std::make_shared<Tensor>(result.shape_, 0.0f);
        auto grad_out = result.grad_;

        result.backward_fn_ = [x, grad_out, mask,
                                N, C, H_out, W_out]() {
            const float*  gop = grad_out->raw_ptr();
            float*        gxp = x->grad().raw_ptr();
            const size_t* msk = mask->data();

            size_t total = N * C * H_out * W_out;
            for (size_t i = 0; i < total; ++i)
                gxp[msk[i]] += gop[i];  // route gradient to the winner
        };

        return result;
    }

    // No learnable parameters
    std::vector<std::shared_ptr<Tensor>> parameters() const override { return {}; }
};


// ── Flatten ───────────────────────────────────────────────────────────────────
// Reshapes {N, C, H, W} → {N, C*H*W}
// Bridges the conv stack and the linear stack.
//
// Backward: just reshape the gradient back to {N, C, H, W}.
//
struct Flatten : public Module {
    Tensor forward(std::shared_ptr<Tensor> x) override {
        size_t N    = x->shape_[0];
        size_t rest = x->num_elements() / N;

        // contiguous() ensures row-major layout before reshape
        Tensor flat = x->contiguous().reshape({N, rest});

        flat.op_name_ = "flatten";
        flat.inputs_  = {x};

        flat.grad_ = std::make_shared<Tensor>(flat.shape_, 0.0f);
        auto grad_out = flat.grad_;
        auto x_shape  = x->shape_;

        flat.backward_fn_ = [x, grad_out, x_shape]() {
            // gradient has same elements, just different shape
            const float* gop = grad_out->raw_ptr();
            float*       gxp = x->grad().raw_ptr();
            size_t n = x->num_elements();
            for (size_t i = 0; i < n; ++i)
                gxp[i] += gop[i];
        };

        return flat;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const override { return {}; }
};

} // namespace nn
} // namespace stakml