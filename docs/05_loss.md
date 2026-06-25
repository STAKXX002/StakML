# Loss Functions

## The classification setup

A classifier's output is a vector of raw scores (logits) - one per class. To turn them into a probability distribution and compute a loss, you need two steps:

1. `log_softmax(logits)` → log-probabilities
2. `nll_loss(log_probs, labels)` → scalar loss

StakML follows PyTorch's convention of separating these.

## Why log-softmax instead of softmax + log?

The naive approach:

```cpp
probs = softmax(logits);     // exp(x[i]) / sum(exp(x))
loss  = -log(probs[y]);      // negative log probability of correct class
```

This is numerically unstable. For large logits (say, `x = [1000, 1001, 999]`), `exp(1001)` overflows to `inf`, and `log(inf/inf)` is `nan`.

The standard fix is to subtract the row maximum before exponentiating:

```
log_softmax(x)[i] = x[i] - max(x) - log( sum_k exp(x[k] - max(x)) )
```

The `max(x)` shift doesn't change the result mathematically (it cancels in numerator and denominator) but keeps all the `exp` arguments ≤ 0, preventing overflow. The largest `exp` term is always `exp(0) = 1`.

StakML implementation:

```cpp
float max_val = *max_element(row, row + C);

float sum_exp = 0.0f;
for (size_t j = 0; j < C; ++j)
    sum_exp += std::exp(row[j] - max_val);

float log_sum = std::log(sum_exp);

for (size_t j = 0; j < C; ++j)
    out[j] = row[j] - max_val - log_sum;
```

## nll_loss

Negative log-likelihood: pick the log-probability of the correct class and negate it. Average over the batch.

```
L = -1/N * sum_i log_probs[i][labels[i]]
```

```cpp
float loss = ops::nll_loss(*lp_tensor, labels);
```

This also seeds the backward gradient directly into `log_probs->grad_`:

```
dL/d(log_probs[i][j]) = -1/N  if j == labels[i]
                        =  0   otherwise
```

So calling `lp_tensor->backward()` immediately after is all you need - the loss gradient is already in place.

## Backward through log_softmax

The backward of `log_softmax` is:

```
dL/dx[i][j] = grad_out[i][j] - softmax(x)[i][j] * sum_k( grad_out[i][k] )
```

Derivation: let `s = log_softmax(x)` and `p = exp(s) = softmax(x)`. The Jacobian of `s` w.r.t. `x` is `I - p * 1^T` (identity minus outer product with the softmax vector). Multiplying by the upstream gradient `g = dL/ds` gives:

```
dL/dx = g - p * (g · 1)
      = g - p * sum(g)
```

In code:

```cpp
result.backward_fn_ = [x, grad_out, lsp_data, batch, C]() {
    for (size_t i = 0; i < batch; ++i) {
        // sum of upstream gradient over this row
        float sum_g = 0.0f;
        for (size_t j = 0; j < C; ++j)
            sum_g += gop_row[j];

        // dL/dx[j] = gop[j] - exp(lsp[j]) * sum_g
        for (size_t j = 0; j < C; ++j)
            gx_row[j] += gop_row[j] - std::exp(lsp_row[j]) * sum_g;
    }
};
```

`exp(lsp_row[j])` recovers `softmax(x)[j]` from the saved log-softmax output - no need to recompute softmax.

## The combined backward

When `nll_loss` seeds `grad_out[i][labels[i]] = -1/N` and zero elsewhere, `sum_g = -1/N` for each row, and the log_softmax backward simplifies to:

```
dL/dx[i][j] = -1/N          if j == labels[i]
            = softmax(x)[i][j] / N   otherwise
```

Which is exactly the well-known result for cross-entropy loss: gradient is `(softmax - one_hot) / N`.

## Accuracy

```cpp
float acc = ops::accuracy(logits, labels);
```

Takes raw logits (or log-probs - argmax is the same either way). Returns the fraction of correctly predicted samples. No backward is attached - this is metrics only.

## Usage

```cpp
// Forward
auto logits = std::make_shared<Tensor>(model.forward(x));
auto lp     = std::make_shared<Tensor>(ops::log_softmax(logits));
float loss  = ops::nll_loss(*lp, labels);   // seeds lp->grad_

// Backward
lp->backward();

// Metrics
float acc = ops::accuracy(*logits, labels);
```