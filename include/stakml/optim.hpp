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

} // namespace optim
} // namespace stakml