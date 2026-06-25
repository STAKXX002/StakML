# Optimizers

## What an optimizer does

After `backward()` fills every parameter's `grad_` buffer with `dL/dW`, the optimizer's job is to move `W` in the direction that reduces the loss. Then `zero_grad()` clears the buffers for the next iteration.

```cpp
lp->backward();   // fills W->grad_, b->grad_, etc.
opt.step();       // updates W, b, etc. from their gradients
opt.zero_grad();  // clears all grad_ buffers
```

## SGD

Stochastic Gradient Descent. The simplest possible update rule:

```
W_new = W_old - lr * dL/dW
```

```cpp
optim::SGD opt(model.parameters(), /*lr=*/0.01f);
```

"Stochastic" because in practice you compute the gradient on a random mini-batch of data, not the full dataset. This is noisy but much faster per update, and the noise can actually help escape shallow local minima.

**When to use SGD:** large-scale vision tasks, when you have time to tune the learning rate and momentum carefully. SGD with momentum and weight decay is still competitive with Adam on ImageNet-scale problems. It generalizes slightly better in some settings.

**Learning rate:** the most important hyperparameter. Too large → loss oscillates or diverges. Too small → slow convergence. Typical range: 1e-2 to 1e-4. Usually needs a scheduler (decay over time) for best results.

## Adam

Adaptive Moment Estimation (Kingma & Ba, 2014). Maintains a per-parameter learning rate adapted from the history of gradients.

```
m_t = β1 * m_{t-1} + (1 - β1) * g         ← first moment (EMA of gradients)
v_t = β2 * v_{t-1} + (1 - β2) * g²        ← second moment (EMA of squared gradients)

m̂ = m_t / (1 - β1^t)                      ← bias-corrected first moment
v̂ = v_t / (1 - β2^t)                      ← bias-corrected second moment

W = W - lr * m̂ / (√v̂ + ε)
```

```cpp
optim::Adam opt(model.parameters(),
    /*lr=*/     1e-3f,
    /*beta1=*/  0.9f,
    /*beta2=*/  0.999f,
    /*eps=*/    1e-8f,
    /*weight_decay=*/ 0.0f);
```

**First moment `m`:** a smoothed version of the gradient. Reduces oscillation in high-curvature directions. Acts like momentum.

**Second moment `v`:** tracks how large gradients have been historically per parameter. Parameters with consistently large gradients get a small effective learning rate. Parameters with small or sporadic gradients get a large effective learning rate. This is the "adaptive" part - learning rate adapts per parameter.

**Why bias correction?** `m` and `v` are initialized to zero. At step 1, `m_1 = (1 - β1) * g`. For `β1 = 0.9`, that's only 10% of the actual gradient - heavily biased toward zero. Dividing by `(1 - β1^t) = (1 - 0.9^1) = 0.1` rescales it back to `g`. By step ~100, `(1 - 0.9^100) ≈ 1` and the correction term stops mattering.

**ε (epsilon):** prevents division by zero when `v̂` is tiny. The default `1e-8` is almost never the right thing to tune, but `1e-6` or `1e-7` can sometimes help with very sparse gradients.

**Weight decay:** L2 regularization. Adds `weight_decay * W` to the gradient before the Adam update, which decays weights toward zero and reduces overfitting. Note that this is the "coupled" implementation - for the "decoupled" version (AdamW, slightly better in practice), the weight decay should be applied to `W` directly, not folded into the gradient before the adaptive scaling. StakML's implementation is coupled Adam, which is what the original paper describes.

**When to use Adam:** most of the time. It's robust to learning rate choice, works well with default hyperparameters, and converges quickly. `lr=1e-3` is usually a good starting point.

## Moment buffers

Adam allocates two float vectors per parameter at construction time:

```cpp
m_.resize(params.size());   // first moments
v_.resize(params.size());   // second moments
for (size_t i = 0; i < params.size(); ++i) {
    size_t n = params[i]->num_elements();
    m_[i].assign(n, 0.0f);
    v_[i].assign(n, 0.0f);
}
```

These are indexed in parallel with `parameters_`. The `t_` counter tracks the step number for bias correction.

## zero_grad

```cpp
void zero_grad() {
    for (auto& p : parameters_)
        p->zero_grad();   // sets all elements of grad_ to 0.0f
}
```

This must be called after every `step()`, before the next `backward()`. Forgetting it causes gradients to accumulate across iterations - the update in step N will include gradients from steps 1..N-1, which is almost never what you want (unless you're intentionally doing gradient accumulation over multiple mini-batches).

## Hyperparameter guide

| | SGD | Adam |
|---|---|---|
| Learning rate | 0.01–0.1 | 0.001 |
| Momentum | 0.9 (if adding) | β1=0.9 (built-in) |
| Weight decay | 1e-4 | 1e-4 |
| Typical use | CV with schedule | Everything else |