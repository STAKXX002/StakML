#pragma once
#include "ops.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// loss.hpp — loss functions for classification
//
// TWO-PIECE DESIGN (PyTorch-style):
//
//   ops::log_softmax(x)        {batch, C} → {batch, C}
//   ops::nll_loss(log_p, y)    {batch, C} + labels → scalar float
//
// WHY LOG-SOFTMAX SEPARATELY?
//   softmax(x)[i] = exp(x[i]) / sum(exp(x))
//   log(softmax(x)[i]) = x[i] - log(sum(exp(x)))          ← naive: two passes
//
//   log_softmax(x)[i] = x[i] - max(x) - log(sum(exp(x-max)))
//                                         ↑ shift by max for numerical stability
//
//   Fusing log and softmax in one pass avoids:
//     1. Computing softmax (which saturates at large logits)
//     2. Taking log of near-zero values → -inf / NaN
//
// WHY NLL_LOSS SEPARATELY?
//   nll_loss just picks log_probs[i][y[i]] and negates+averages.
//   No math, no instability — but keeping it separate makes each piece
//   independently testable and grad-checkable.
//
// BACKWARD DERIVATION (log_softmax):
//   Let s = log_softmax(x),  p = softmax(x) = exp(s)
//   L = nll_loss(s, y) = -mean( s[i][y[i]] )
//
//   dL/ds[i][j] = -1/batch  if j == y[i], else 0    (from nll_loss)
//   dL/dx[i][j] = dL/ds[i][j] - p[i][j] * sum_k(dL/ds[i][k])
//              = dL/ds[i][j] - p[i][j] * (-1/batch)   (only y[i] contributes)
//
//   But we compute the JOINT backward through the combined op here.
//   nll_loss seeds grad into log_probs, log_softmax backward uses it.
//
//   log_softmax backward:
//     dL/dx[i][j] = grad_lp[i][j] - softmax(x)[i][j] * sum_k(grad_lp[i][k])
//
//   This is correct for any upstream grad (not just from nll_loss), so
//   log_softmax is reusable for other losses too.
// ─────────────────────────────────────────────────────────────────────────────

namespace stakml::ops {

// ── log_softmax ───────────────────────────────────────────────────────────────
// x: {batch, C}  →  {batch, C}
// out[i][j] = x[i][j] - max(x[i]) - log( sum_k exp(x[i][k] - max(x[i])) )
//
inline Tensor log_softmax(std::shared_ptr<Tensor> x) {
    if (x->ndim() != 2)
        throw std::runtime_error("log_softmax: input must be 2-D {batch, C}");

    size_t batch = x->shape_[0];
    size_t C     = x->shape_[1];

    Tensor result({batch, C});
    const float* xp  = x->raw_ptr();
    float*       rp  = result.raw_ptr();

    for (size_t i = 0; i < batch; ++i) {
        const float* row = xp + i*C;
        float*       out = rp + i*C;

        // max for numerical stability
        float max_val = row[0];
        for (size_t j = 1; j < C; ++j)
            max_val = std::max(max_val, row[j]);

        // sum of exp(x - max)
        float sum_exp = 0.0f;
        for (size_t j = 0; j < C; ++j)
            sum_exp += std::exp(row[j] - max_val);

        float log_sum = std::log(sum_exp);

        // log_softmax[j] = x[j] - max - log_sum
        for (size_t j = 0; j < C; ++j)
            out[j] = row[j] - max_val - log_sum;
    }

    result.op_name_ = "log_softmax";
    result.inputs_  = {x};

    // ── backward ─────────────────────────────────────────────────────────────
    // dL/dx[i][j] = grad_out[i][j] - softmax(x)[i][j] * sum_k(grad_out[i][k])
    //
    // We need softmax(x) = exp(log_softmax(x)), which we already computed.
    // Save a shared_ptr to result's data so the closure captures it correctly.
    //
    result.grad();                        // allocate grad_ before capture
    auto grad_out = result.grad_;

    // Capture the log-softmax values (= result's data) for use in backward.
    // result.data_ is a shared_ptr<vector<float>> — safe to capture.
    auto lsp_data = result.data_;         // log-softmax values, shape {batch,C}

    result.backward_fn_ = [x, grad_out, lsp_data, batch, C]() {
        const float* lsp = lsp_data->data();   // log_softmax output
        const float* gop = grad_out->raw_ptr(); // dL/d(log_softmax output)
        float*       gxp = x->grad().raw_ptr(); // accumulate into x

        for (size_t i = 0; i < batch; ++i) {
            const float* lsp_row = lsp + i*C;
            const float* gop_row = gop + i*C;
            float*       gx_row  = gxp + i*C;

            // sum of upstream grad over this row
            float sum_g = 0.0f;
            for (size_t j = 0; j < C; ++j)
                sum_g += gop_row[j];

            // dL/dx[j] = gop[j] - exp(lsp[j]) * sum_g
            for (size_t j = 0; j < C; ++j)
                gx_row[j] += gop_row[j] - std::exp(lsp_row[j]) * sum_g;
        }
    };

    return result;
}

// ── nll_loss ──────────────────────────────────────────────────────────────────
// Negative log-likelihood loss.
//
//   log_probs : {batch, C}   — output of log_softmax
//   labels    : {batch}      — integer class indices in [0, C)
//
// Returns scalar loss = -mean( log_probs[i][labels[i]] )
//
// Also seeds the backward gradient into log_probs->grad_ so that calling
// log_probs_tensor.backward() propagates through the whole graph.
//
// USAGE:
//   auto lp_tensor = ops::log_softmax(logits_ptr);
//   auto lp_ptr    = std::make_shared<Tensor>(lp_tensor);
//   float loss     = ops::nll_loss(*lp_ptr, labels);
//   lp_ptr->backward();   ← propagates through log_softmax → fc3 → ... → fc1
//
inline float nll_loss(Tensor& log_probs,
                      const std::vector<int>& labels)
{
    if (log_probs.ndim() != 2)
        throw std::runtime_error("nll_loss: log_probs must be 2-D {batch, C}");

    size_t batch = log_probs.shape_[0];
    size_t C     = log_probs.shape_[1];

    if (labels.size() != batch)
        throw std::runtime_error("nll_loss: labels.size() must equal batch size");

    const float* lp = log_probs.raw_ptr();

    // Forward: loss = -1/batch * sum_i log_probs[i][labels[i]]
    float loss = 0.0f;
    for (size_t i = 0; i < batch; ++i) {
        int y = labels[i];
        if (y < 0 || static_cast<size_t>(y) >= C)
            throw std::runtime_error("nll_loss: label out of range");
        loss -= lp[i*C + y];
    }
    loss /= static_cast<float>(batch);

    // Backward seed: dL/d(log_probs[i][j]) = -1/batch if j==labels[i], else 0
    // We write this directly into log_probs.grad_ so that log_probs.backward()
    // will pick it up immediately — no extra wrapper node needed.
    float* glp = log_probs.grad().raw_ptr();
    float  inv_batch = -1.0f / static_cast<float>(batch);
    // zero first (grad() lazily allocates, but may be reused across steps)
    size_t n = log_probs.num_elements();
    for (size_t k = 0; k < n; ++k) glp[k] = 0.0f;
    for (size_t i = 0; i < batch; ++i)
        glp[i*C + labels[i]] = inv_batch;

    return loss;
}

// ── accuracy ──────────────────────────────────────────────────────────────────
// Given logits or log_probs {batch, C} and integer labels {batch},
// returns fraction of correctly predicted samples.
// Works on raw logits (argmax is the same whether or not you apply softmax).
//
inline float accuracy(const Tensor& logits,
                      const std::vector<int>& labels)
{
    size_t batch = logits.shape_[0];
    size_t C     = logits.shape_[1];
    const float* lp = logits.raw_ptr();

    int correct = 0;
    for (size_t i = 0; i < batch; ++i) {
        const float* row = lp + i*C;
        size_t pred = 0;
        for (size_t j = 1; j < C; ++j)
            if (row[j] > row[pred]) pred = j;
        if (static_cast<int>(pred) == labels[i]) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(batch);
}

} // namespace stakml::ops