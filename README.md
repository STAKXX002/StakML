# StakML

A from-scratch machine learning library in C++17. Tensors, autograd, and layers built up from first principles, with optional CUDA and Vulkan GPU backends.

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

When `STAKML_CUDA=ON`, the three matmul variants (`A@B`, `A@B.T`, `A.T@B`) route through cuBLAS instead of the OpenMP loop. Everything else stays untouched.

## Datasets

`mnist_mlp` and `cifar_cnn` expect raw binary datasets in `../data/` (relative to `build/`), gitignored due to size.

```bash
mkdir -p data && cd data

curl -LO https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
curl -LO https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
curl -LO https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
curl -LO https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz
gunzip *.gz

curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
```

`world_cup`, `predict_match`, `group_stage`, and `full_tournament` use a bundled dataset (`examples/football_dataset.hpp`) - no download needed.

## Running examples

```bash
cd build
./mnist_mlp
./cifar_cnn
./world_cup
./predict_match
./group_stage
./full_tournament
```

## Running tests

```bash
cd build
./test_tensor
./test_backward
./test_graph
./test_loss
./test_optim
./test_conv
```

## Architecture

```
stakml::Tensor    - data storage, shape/strides, raw matmul
    ↓
stakml::ops::*    - functional layer: op_name_, inputs_, backward_fn_
    ↓
stakml::nn::*     - stateful layers (own parameters as shared_ptr<Tensor>)
    ↓
stakml::optim::*  - reads .grad_ buffers, updates weights in-place
```

The graph is built implicitly during the forward pass. `tensor.backward()` does a topological sort and fires each `backward_fn_` in reverse.

## GPU backends

### CUDA (Nvidia)
- Phase 1 (shipped, `main`): cuBLAS wrappers, `-DSTAKML_CUDA=ON`
- Phase 2 (in progress, [`cuda-backend` branch](https://github.com/STAKXX002/StakML/tree/cuda-backend)): hand-written tiled SGEMM kernels in `cuda_sandbox/`
- Phase 3 (planned): keep element-wise kernels on-device between layers

See [`include/stakml/cuda/Readme.md`](include/stakml/cuda/Readme.md).

### Vulkan (any GPU, tested on AMD Radeon 780M / RADV)
```bash
mkdir build && cd build
cmake .. && make test_vulkan && ./test_vulkan
```
Status: vecadd verified on 780M. matmul, relu, conv in progress.

## Docs

- [`docs/01_tensors.md`](docs/01_tensors.md)
- [`docs/02_autograd.md`](docs/02_autograd.md)
- [`docs/03_matmul.md`](docs/03_matmul.md)
- [`docs/04_layers.md`](docs/04_layers.md)
- [`docs/05_loss.md`](docs/05_loss.md)
- [`docs/06_optimizers.md`](docs/06_optimizers.md)
- [`docs/07_conv.md`](docs/07_conv.md)