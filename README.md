# StakML

A from-scratch machine learning library in C++17. Tensors, autograd, and layers built up from first principles, with an optional CUDA backend for matrix multiply.

```cpp
#include <stakml/nn.hpp>
#include <stakml/loss.hpp>
#include <stakml/optim.hpp>

using namespace stakml;

auto fc1 = std::make_shared<nn::Linear>(784, 256);
auto fc2 = std::make_shared<nn::Linear>(256, 10);
nn::Sequential model({ fc1, std::make_shared<nn::ReLU>(), fc2 });

optim::Adam opt(model.parameters(), 1e-3f);

// forward
auto x  = std::make_shared<Tensor>(Tensor({batch, 784}, data));
auto lp = std::make_shared<Tensor>(ops::log_softmax(
              std::make_shared<Tensor>(model.forward(x))));
float loss = ops::nll_loss(*lp, labels);

// backward + update
lp->backward();
opt.step();
opt.zero_grad();
```

## What's inside

| Header | What it gives you |
|---|---|
| `tensor.hpp` | N-D tensor, row-major strides, zero-copy reshape/transpose, element-wise ops, blocked OpenMP matmul |
| `ops.hpp` | Autograd-aware functional ops - matmul, relu, add_bias, log_softmax, graph tracing |
| `nn.hpp` | Linear, ReLU, Sigmoid, Dropout, Sequential |
| `loss.hpp` | log_softmax + nll_loss + accuracy |
| `optim.hpp` | SGD, Adam (with bias correction and weight decay) |
| `conv.hpp` | Conv2d, MaxPool2d, Flatten via im2col → matmul |
| `dataset.hpp` | MNIST/CIFAR binary loader |
| `serialize.hpp` | Save/load weights |
| `cuda/matmul.cuh` | cuBLAS matmul backend (opt-in, Phase 1) |

## Building

Requires: CMake ≥ 3.18, C++17, OpenMP.

```bash
git clone https://github.com/STAKXX002/StakML.git
cd StakML
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**With CUDA backend** (requires CUDA Toolkit ≥ 11):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSTAKML_CUDA=ON
cmake --build build -j$(nproc)
```

When `STAKML_CUDA=ON`, the three matmul variants (`A@B`, `A@B.T`, `A.T@B`) route through cuBLAS instead of the OpenMP loop. Everything else - autograd, layers, loss, optimizer - is untouched.

## Running examples

```bash
cd build
./mnist_mlp          # trains a 784→256→10 MLP on MNIST
./cifar_cnn          # trains a Conv→Pool→FC net on CIFAR-10
./world_cup          # group stage prediction demo
```

## Running tests

```bash
cd build
./test_tensor        # 17 tensor primitive tests
./test_backward      # 7 autograd / grad-check tests
./test_graph         # 12 op graph and layer tests
./test_loss          # loss function tests
./test_optim         # SGD + Adam step tests
./test_conv          # im2col, Conv2d, MaxPool2d tests
```

## Architecture

```
stakml::Tensor          - data storage, shape/strides, raw matmul
    ↓
stakml::ops::*          - functional layer: attaches op_name_, inputs_, backward_fn_
    ↓
stakml::nn::*           - stateful layers (own parameters as shared_ptr<Tensor>)
    ↓
stakml::optim::*        - reads .grad_ buffers, updates weights in-place
```

The graph is built implicitly during the forward pass. Each `ops::` call stamps the result with its input pointers. `tensor.backward()` does a topological sort and fires each `backward_fn_` in reverse order. `shared_ptr` ownership keeping nodes alive.

## CUDA backend roadmap

```
Phase 1 (current)   cuBLAS wrappers in include/stakml/cuda/matmul.cuh
                    Host↔device copies per call, zero changes to Tensor layout

Phase 2 (next)      Hand-written tiled SGEMM - removes cuBLAS dependency
                    Teaches shared memory, tile size tuning, occupancy

Phase 3             Element-wise kernels (relu, softmax) stay on device
                    between layers - eliminates per-layer round-trips
```

See [`include/stakml/cuda/README.md`](include/stakml/cuda/README.md) for details.

## Docs

Concept explainers live in [`docs/`](docs/):

- [`docs/01_tensors.md`](docs/01_tensors.md) - what a tensor is, strides, zero-copy views
- [`docs/02_autograd.md`](docs/02_autograd.md) - how the computation graph is built and walked
- [`docs/03_matmul.md`](docs/03_matmul.md) - CPU blocked matmul, why tiling exists, CUDA plan
- [`docs/04_layers.md`](docs/04_layers.md) - Linear, activations, Dropout, Sequential
- [`docs/05_loss.md`](docs/05_loss.md) - log_softmax stability, nll_loss backward derivation
- [`docs/06_optimizers.md`](docs/06_optimizers.md) - SGD, Adam bias correction, weight decay
- [`docs/07_conv.md`](docs/07_conv.md) - im2col, Conv2d forward/backward, MaxPool2d