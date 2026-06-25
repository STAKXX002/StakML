# Layers

## The Module abstraction

Every layer in StakML inherits from `nn::Module`:

```cpp
struct Module {
    virtual Tensor forward(std::shared_ptr<Tensor> x) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> parameters() const { return {}; }
};
```

`forward()` takes a `shared_ptr<Tensor>` and returns a `Tensor`. The input is a shared pointer because the resulting graph node needs to hold a reference to it - if you passed by value, the graph couldn't keep the input alive for the backward pass.

`parameters()` returns the layer's learnable weights. The optimizer calls this to know what to update.

## Linear

```
y = x @ W + b

x : {batch, in_features}
W : {in_features, out_features}   ← Xavier init
b : {1, out_features}             ← zeros
y : {batch, out_features}
```

```cpp
auto fc = std::make_shared<nn::Linear>(128, 64);
Tensor out = fc->forward(x_ptr);
```

Internally, `forward()` is:

```cpp
auto xW = std::make_shared<Tensor>(ops::matmul(x, W));
return ops::add_bias(xW, b);
```

Both `ops::matmul` and `ops::add_bias` attach backward functions, so the full backward chain (including gradient into `W` and `b`) is built automatically.

**Weight init:** Xavier/Glorot - `std = sqrt(2 / (fan_in + fan_out))`. This keeps the variance of activations roughly constant across layers. Too large → activations explode. Too small → gradients vanish. Xavier is the stable middle ground for layers with symmetric activations (tanh, linear). For ReLU layers, He init (`std = sqrt(2 / fan_in)`) is technically more correct, but Xavier works fine in practice.

## Activations

### ReLU

```
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```

The most widely used activation. Fast, no saturation on the positive side, sparse (many zeros). The backward function is a simple mask: gradient passes through where `x > 0`, is zeroed where `x ≤ 0`.

```cpp
auto relu = std::make_shared<nn::ReLU>();
Tensor out = relu->forward(x_ptr);
```

### Sigmoid

```
f(x) = 1 / (1 + exp(-x))
f'(x) = f(x) * (1 - f(x))
```

Squashes output to (0, 1). Used for binary classification outputs. Saturates at large |x|, causing vanishing gradients in deep nets - this is why ReLU replaced it in hidden layers. Still useful in output layers for probabilities.

### Tanh

```
f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
f'(x) = 1 - f(x)²
```

Squashes to (-1, 1). Zero-centred unlike sigmoid, which makes optimization slightly easier. Same saturation problem at large |x|.

## Dropout

Inverted dropout: during training, each activation is zeroed with probability `p` and the survivors are scaled by `1/(1-p)`.

```cpp
auto drop = std::make_shared<nn::Dropout>(0.5f);
drop->set_training(true);   // training mode
Tensor out = drop->forward(x_ptr);

drop->set_training(false);  // inference mode - pure pass-through
```

**Why inverted?** The `1/(1-p)` scale ensures that the expected value of each activation is unchanged at training time. This means inference runs without any modification - you don't need to multiply by `(1-p)` at test time. PyTorch uses this convention; older literature sometimes scales at test time instead.

**Backward:** The same binary mask built during forward is captured in the closure and applied to the upstream gradient. Positions that were zeroed in the forward pass get zero gradient. Survivors get their gradient scaled by the same `1/(1-p)`.

**Why it works:** By randomly disabling neurons, dropout prevents co-adaptation - neurons can't learn to rely on specific other neurons always being present. This acts as a regularizer, roughly equivalent to training an ensemble of `2^n` thinned networks and averaging them.

## Sequential

Chains layers into a single `Module`:

```cpp
nn::Sequential model({
    std::make_shared<nn::Linear>(784, 256),
    std::make_shared<nn::ReLU>(),
    std::make_shared<nn::Dropout>(0.3f),
    std::make_shared<nn::Linear>(256, 10),
});

Tensor out = model.forward(x_ptr);
auto params = model.parameters();   // W1, b1, W2, b2 - no ReLU/Dropout params
model.set_training(false);          // propagates to Dropout layers
```

`parameters()` collects from all child layers - this is what you pass to the optimizer.

## Typical MLP pattern

```cpp
// Build
auto model = nn::Sequential({
    std::make_shared<nn::Linear>(784, 512),
    std::make_shared<nn::ReLU>(),
    std::make_shared<nn::Dropout>(0.2f),
    std::make_shared<nn::Linear>(512, 256),
    std::make_shared<nn::ReLU>(),
    std::make_shared<nn::Linear>(256, 10),
});
optim::Adam opt(model.parameters(), 1e-3f);

// Train loop
model.set_training(true);
for (auto& [x_data, labels] : train_loader) {
    auto x      = std::make_shared<Tensor>(Tensor({batch, 784}, x_data));
    auto logits = std::make_shared<Tensor>(model.forward(x));
    auto lp     = std::make_shared<Tensor>(ops::log_softmax(logits));
    float loss  = ops::nll_loss(*lp, labels);
    lp->backward();
    opt.step();
    opt.zero_grad();
}

// Eval
model.set_training(false);
float acc = ops::accuracy(model.forward(x_test), test_labels);
```