# Autograd

## The problem

Training a neural network means computing `dL/dW` for every weight `W` - how much does the loss change if we nudge `W` slightly? With a tiny network you could derive this by hand. With a real one (dozens of layers, millions of parameters) you need the computer to do it automatically.

That's what autograd is: automatic differentiation via a dynamically-built computation graph.

## The computation graph

Every time you call an `ops::` function, it does two things:

1. Computes the output value (the forward pass result)
2. Records *what operation produced this output* and *which tensors were its inputs*

This recording is just two fields on the result `Tensor`:

```cpp
result.op_name_ = "matmul";       // what happened
result.inputs_  = {a, b};         // who the parents are (shared_ptr)
```

After a full forward pass, these links form a directed acyclic graph (DAG) - the computation graph. Leaf tensors (weights, inputs) have empty `inputs_`. Intermediate tensors point back to whoever created them.

```
   [W]    [x]
     \    /
     matmul         op_name_="matmul", inputs_={W, x}
        |
     add_bias       op_name_="add_bias", inputs_={matmul_out, b}
        |
      relu          op_name_="relu", inputs_={add_bias_out}
        |
     log_softmax
        |
  nll_loss (scalar)
```

## The backward pass

`tensor.backward()` does three things:

**Step 1: Seed the gradient.** The loss tensor gets `grad_ = ones(shape)`. This seeds `dL/dL = 1`, which is trivially correct and starts the chain.

**Step 2: Topological sort.** Starting from the loss, walk the graph recursively following `inputs_`. Collect nodes in visit order, then reverse it. The result is a sequence where every node appears *before* its parents - i.e., consumers before producers.

```cpp
std::function<void(Tensor*)> build_topo = [&](Tensor* t) {
    if (visited.count(t)) return;
    visited.insert(t);
    for (auto& inp : t->inputs_)
        build_topo(inp.get());
    topo.push_back(t);
};
build_topo(this);
std::reverse(topo.begin(), topo.end());
```

**Step 3: Fire backward functions.** Walk the sorted list and call each node's `backward_fn_()`. Each function reads the node's own `grad_` (which has already been filled in by the node ahead of it) and *accumulates* the gradient into its inputs' `grad_` buffers.

```cpp
for (Tensor* t : topo)
    if (t->backward_fn_) t->backward_fn_();
```

The accumulation (`+=` not `=`) is important: a parameter used in multiple places (shared weights, like in an RNN) receives gradient contributions from all of them, summed up correctly.

## How a backward function is attached

Inside `ops::relu`:

```cpp
inline Tensor relu(std::shared_ptr<Tensor> x) {
    Tensor result = x->relu();          // forward computation
    result.op_name_ = "relu";
    result.inputs_  = {x};

    result.grad();                      // force-allocate grad_ before capture
    auto grad_out = result.grad_;       // shared_ptr to result's gradient buffer

    result.backward_fn_ = [x, grad_out]() {
        // The chain rule for ReLU:
        //   dL/dx[i] += dL/d_result[i]  *  (x[i] > 0 ? 1 : 0)
        size_t n = x->num_elements();
        const float* xp  = x->raw_ptr();
        const float* gop = grad_out->raw_ptr();  // dL/d_result, already filled
        float*       gxp = x->grad().raw_ptr();  // accumulate into x
        for (size_t i = 0; i < n; ++i)
            gxp[i] += gop[i] * (xp[i] > 0.0f ? 1.0f : 0.0f);
    };

    return result;
}
```

The closure captures `x` (the input, to read its values) and `grad_out` (the result's gradient, which will be filled by whoever is above this node in the graph). Both are `shared_ptr`, so they stay alive as long as the graph exists.

## Why shared_ptr for inputs_?

`inputs_` holds `shared_ptr<Tensor>`. This means:

- The graph keeps its inputs alive. If you discard the intermediate tensors after the forward pass, they don't get freed - the graph is still holding references.
- When the graph is eventually destroyed (after `step()` and `zero_grad()`), all intermediate nodes are freed in the right order.

This is the same ownership model as PyTorch's autograd engine, just without a separate C++ object for "grad_fn".

## Gradient accumulation vs assignment

Gradients always `+=`, never `=`. This is correct because:

1. A parameter might be used in multiple places in the graph. Each use adds its contribution.
2. You might call `backward()` multiple times before `zero_grad()` (useful for gradient accumulation over mini-batches).

After each optimizer step, call `zero_grad()` to clear the buffers before the next forward pass.

## Usage pattern

```cpp
// 1. Forward pass (builds graph implicitly)
auto logits = std::make_shared<Tensor>(model.forward(x));
auto lp     = std::make_shared<Tensor>(ops::log_softmax(logits));
float loss  = ops::nll_loss(*lp, labels);  // seeds lp->grad_

// 2. Backward pass (topological sort + fire backward_fn_s)
lp->backward();

// 3. Parameter update
opt.step();

// 4. Clear gradients for next iteration
opt.zero_grad();
```

## Debugging the graph

```cpp
ops::print_graph(result);
// relu → (3, 10)
//   add_bias → (3, 10)
//     matmul → (3, 10)
//       [leaf (3, 4)]
//       [leaf (4, 10)]
//     [leaf (1, 10)]
```