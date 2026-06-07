#pragma once
#include "tensor.hpp"
#include <vector>
#include <memory>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// optim.hpp — Optimizers for updating neural network parameters
//
// CONCEPT: What is an Optimizer?
//
//   An optimizer takes a list of parameters (shared_ptr<Tensor>) and a 
//   learning rate. After a backward pass populates the .grad() buffers 
//   of these parameters, the optimizer's step() function updates the 
//   underlying weights.
//
//   It also provides zero_grad() to clear out the old gradients before 
//   the next forward-backward cycle.
// ─────────────────────────────────────────────────────────────────────────────

namespace stakml {
namespace optim {

// ── Base Optimizer ───────────────────────────────────────────────────────────
class Optimizer {
protected:
    std::vector<std::shared_ptr<Tensor>> parameters_;

public:
    explicit Optimizer(const std::vector<std::shared_ptr<Tensor>>& params)
        : parameters_(params) {}

    virtual ~Optimizer() = default;

    // Clears the gradients of all tracked parameters
    void zero_grad() {
        for (auto& p : parameters_) {
            p->zero_grad();
        }
    }

    // Updates the parameters based on their gradients (implemented by subclasses)
    virtual void step() = 0;
};

// ── Stochastic Gradient Descent (SGD) ────────────────────────────────────────
class SGD : public Optimizer {
public:
    float lr_; // Learning rate

    SGD(const std::vector<std::shared_ptr<Tensor>>& params, float lr)
        : Optimizer(params), lr_(lr) {
        if (lr_ <= 0.0f) {
            throw std::invalid_argument("SGD: learning rate must be positive");
        }
    }

    void step() override {
        for (auto& p : parameters_) {
            // If the parameter has no gradient (e.g., requires_grad_ was false 
            // or backward hasn't been called), skip it.
            if (!p->grad_) continue;

            float* param_ptr = p->raw_ptr();
            const float* grad_ptr = p->grad_->raw_ptr();
            size_t n = p->num_elements();

            // In-place update: W_new = W_old - lr * grad
            // We use raw pointers here for maximum performance, matching the 
            // style of your matrix multiplication loops.
            for (size_t i = 0; i < n; ++i) {
                param_ptr[i] -= lr_ * grad_ptr[i];
            }
        }
    }
};

// ── Adam ─────────────────────────────────────────────────────────────────────
// Adaptive Moment Estimation (Kingma & Ba, 2014)
//
// Maintains two moment buffers per parameter:
//   m  — first moment  (exponential moving average of gradients)
//   v  — second moment (exponential moving average of squared gradients)
//
// Update rule:
//   m_t = β1 * m_{t-1} + (1 - β1) * g
//   v_t = β2 * v_{t-1} + (1 - β2) * g²
//   m̂   = m_t / (1 - β1^t)          ← bias correction (important early on)
//   v̂   = v_t / (1 - β2^t)
//   w   = w - lr * m̂ / (√v̂ + ε)
//
// WHY BIAS CORRECTION?
//   m and v are initialised to zero. Without correction, early estimates
//   are heavily biased toward zero — the network barely moves on step 1.
//   Dividing by (1 - β^t) rescales them back to their true magnitude.
//   By step ~1000 the correction term → 1 and stops mattering.
//
// WHY ε?
//   Prevents division by zero when v̂ is tiny (e.g. a dead neuron
//   whose gradient has been zero for many steps).
//
class Adam : public Optimizer {
public:
    float lr_;
    float beta1_;   // default 0.9
    float beta2_;   // default 0.999
    float eps_;     // default 1e-8
    float weight_decay_;  
    size_t t_;      // step counter (starts at 0, incremented each step())

    // Per-parameter moment buffers — same shape as each parameter
    std::vector<std::vector<float>> m_;  // first moments
    std::vector<std::vector<float>> v_;  // second moments

    Adam(const std::vector<std::shared_ptr<Tensor>>& params,
        float lr           = 1e-3f,
        float beta1        = 0.9f,
        float beta2        = 0.999f,
        float eps          = 1e-8f,
        float weight_decay = 0.0f)
        : Optimizer(params), lr_(lr), beta1_(beta1), beta2_(beta2),
        eps_(eps), weight_decay_(weight_decay), t_(0)
    {
        // Allocate moment buffers — zero-initialised, same size as each param
        m_.resize(params.size());
        v_.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            size_t n = params[i]->num_elements();
            m_[i].assign(n, 0.0f);
            v_[i].assign(n, 0.0f);
        }
    }

    void step() override {
        ++t_;  // increment before update so bias correction uses t=1 on first step

        // Precompute bias-corrected denominators (same for all params this step)
        float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));  // 1 - β1^t
        float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));  // 1 - β2^t

        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto& p = parameters_[i];
            if (!p->grad_) continue;

            float*       w  = p->raw_ptr();
            const float* g  = p->grad_->raw_ptr();
            float*       mi = m_[i].data();
            float*       vi = v_[i].data();
            size_t       n  = p->num_elements();

            for (size_t j = 0; j < n; ++j) {
                // Update moments
                mi[j] = beta1_ * mi[j] + (1.0f - beta1_) * g[j];
                vi[j] = beta2_ * vi[j] + (1.0f - beta2_) * g[j] * g[j];

                // Bias-corrected estimates
                float m_hat = mi[j] / bc1;
                float v_hat = vi[j] / bc2;

                // Parameter update
                w[j] -= lr_ * (m_hat / (std::sqrt(v_hat) + eps_) + weight_decay_ * w[j]);
            }
        }
    }
};

} // namespace optim
} // namespace stakml