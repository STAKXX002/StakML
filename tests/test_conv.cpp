#include "stakml/tensor.hpp"
#include "stakml/ops.hpp"
#include "stakml/nn.hpp"
#include "stakml/conv.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace stakml;

// ─────────────────────────────────────────────────────────────────────────────
// Numerical gradient checker
//
// For each parameter element p[i]:
//   f(p + h)[i] - f(p - h)[i]
//   ──────────────────────────  ≈  analytical_grad[i]
//            2h
//
// If max relative error < 1e-3, gradients are correct.
// ─────────────────────────────────────────────────────────────────────────────
float numerical_grad(std::function<float()> loss_fn,
                     float* param, size_t idx, float h = 1e-4f)
{
    float orig = param[idx];
    param[idx] = orig + h;
    float f_plus = loss_fn();
    param[idx] = orig - h;
    float f_minus = loss_fn();
    param[idx] = orig;
    return (f_plus - f_minus) / (2.0f * h);
}

bool close(float a, float b, float tol = 1e-2f) {
    float diff = std::abs(a - b);
    float scale = std::max(1e-5f, std::max(std::abs(a), std::abs(b)));
    return (diff / scale) < tol;
}

int passed = 0, failed = 0;

void check(bool cond, const std::string& name) {
    if (cond) { std::cout << "  PASS  " << name << "\n"; ++passed; }
    else       { std::cout << "  FAIL  " << name << "\n"; ++failed; }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: im2col output shape
// ─────────────────────────────────────────────────────────────────────────────
void test_im2col_shape() {
    std::cout << "\n[im2col shape]\n";

    // {2, 3, 4, 4} input, 3x3 kernel, stride=1, padding=1
    // H_out = (4 + 2 - 3)/1 + 1 = 4
    // W_out = 4
    // col shape = {2*4*4, 3*3*3} = {32, 27}
    Tensor x({2, 3, 4, 4}, 1.0f);
    Tensor col = im2col(x, 3, 3, 1, 1);

    check(col.shape_[0] == 32, "im2col rows = N*H_out*W_out = 32");
    check(col.shape_[1] == 27, "im2col cols = C*kH*kW = 27");

    // No padding: H_out = (4-3)/1+1 = 2
    Tensor col2 = im2col(x, 3, 3, 1, 0);
    check(col2.shape_[0] == 2*2*2, "im2col no-pad rows = 8");
    check(col2.shape_[1] == 27,    "im2col no-pad cols = 27");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Conv2d output shape
// ─────────────────────────────────────────────────────────────────────────────
void test_conv2d_shape() {
    std::cout << "\n[Conv2d shape]\n";

    // CIFAR-style: {4, 3, 32, 32} → Conv2d(3,16,3,pad=1) → {4,16,32,32}
    nn::Conv2d conv(3, 16, 3, 1, 1);
    auto x = std::make_shared<Tensor>(Tensor::randn({4, 3, 32, 32}));
    Tensor out = conv.forward(x);

    check(out.shape_[0] == 4,  "N=4");
    check(out.shape_[1] == 16, "C_out=16");
    check(out.shape_[2] == 32, "H_out=32 (same padding)");
    check(out.shape_[3] == 32, "W_out=32 (same padding)");

    // No padding: {4, 3, 32, 32} → Conv2d(3,16,3,pad=0) → {4,16,30,30}
    nn::Conv2d conv2(3, 16, 3, 1, 0);
    Tensor out2 = conv2.forward(x);
    check(out2.shape_[2] == 30, "H_out=30 (no padding)");
    check(out2.shape_[3] == 30, "W_out=30 (no padding)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: MaxPool2d shape
// ─────────────────────────────────────────────────────────────────────────────
void test_maxpool_shape() {
    std::cout << "\n[MaxPool2d shape]\n";

    nn::MaxPool2d pool(2, 2);
    auto x = std::make_shared<Tensor>(Tensor::randn({4, 16, 32, 32}));
    Tensor out = pool.forward(x);

    check(out.shape_[0] == 4,  "N=4");
    check(out.shape_[1] == 16, "C=16");
    check(out.shape_[2] == 16, "H_out=16 (halved)");
    check(out.shape_[3] == 16, "W_out=16 (halved)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Flatten shape
// ─────────────────────────────────────────────────────────────────────────────
void test_flatten_shape() {
    std::cout << "\n[Flatten shape]\n";

    nn::Flatten flat;
    auto x = std::make_shared<Tensor>(Tensor::randn({4, 16, 8, 8}));
    Tensor out = flat.forward(x);

    check(out.shape_[0] == 4,    "N=4");
    check(out.shape_[1] == 1024, "features=16*8*8=1024");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: Conv2d correctness — single known filter
//
// 1×1×3×3 input, 1×1×2×2 kernel (no padding, stride 1)
// Expected output: 1×1×2×2, each element = sum of 2x2 patch * kernel
// ─────────────────────────────────────────────────────────────────────────────
void test_conv2d_values() {
    std::cout << "\n[Conv2d values]\n";

    // Input: 1 sample, 1 channel, 3x3
    // [[1,2,3],
    //  [4,5,6],
    //  [7,8,9]]
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{1,1,3,3},
        std::vector<float>{1,2,3, 4,5,6, 7,8,9}
    );

    // 1 filter, 1 channel, 2x2 kernel
    // [[1,0],
    //  [0,1]]  → identity-ish: picks top-left + bottom-right of each 2x2 patch
    nn::Conv2d conv(1, 1, 2, 1, 0);
    // Override weights to known values
    float* wptr = conv.W->raw_ptr();
    wptr[0]=1; wptr[1]=0; wptr[2]=0; wptr[3]=1;
    // Zero bias
    conv.bias->raw_ptr()[0] = 0.0f;

    Tensor out = conv.forward(x);
    // H_out=W_out=2
    // out[0,0,0,0] = 1*1 + 0*2 + 0*4 + 1*5 = 6
    // out[0,0,0,1] = 1*2 + 0*3 + 0*5 + 1*6 = 8
    // out[0,0,1,0] = 1*4 + 0*5 + 0*7 + 1*8 = 12
    // out[0,0,1,1] = 1*5 + 0*6 + 0*8 + 1*9 = 14
    const float* op = out.raw_ptr();
    check(std::abs(op[0] -  6.0f) < 1e-4f, "conv val [0,0,0,0]=6");
    check(std::abs(op[1] -  8.0f) < 1e-4f, "conv val [0,0,0,1]=8");
    check(std::abs(op[2] - 12.0f) < 1e-4f, "conv val [0,0,1,0]=12");
    check(std::abs(op[3] - 14.0f) < 1e-4f, "conv val [0,0,1,1]=14");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: MaxPool2d correctness
// ─────────────────────────────────────────────────────────────────────────────
void test_maxpool_values() {
    std::cout << "\n[MaxPool2d values]\n";

    // 1 sample, 1 channel, 4x4:
    // [[ 1, 3, 2, 4],
    //  [ 5, 6, 7, 8],
    //  [ 9,10,11,12],
    //  [13,14,15,16]]
    // Pool 2x2 stride 2: top-left max=6, top-right max=8, bot-left=14, bot-right=16
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{1,1,4,4},
        std::vector<float>{1,3,2,4, 5,6,7,8, 9,10,11,12, 13,14,15,16}
    );

    nn::MaxPool2d pool(2, 2);
    Tensor out = pool.forward(x);
    const float* op = out.raw_ptr();

    check(std::abs(op[0] -  6.0f) < 1e-4f, "pool [0,0,0,0]=6");
    check(std::abs(op[1] -  8.0f) < 1e-4f, "pool [0,0,0,1]=8");
    check(std::abs(op[2] - 14.0f) < 1e-4f, "pool [0,0,1,0]=14");
    check(std::abs(op[3] - 16.0f) < 1e-4f, "pool [0,0,1,1]=16");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: Gradient check — Conv2d weights
//
// Use a tiny conv (1 sample, 1 channel, 4x4, 1 filter, 2x2 kernel)
// and verify dW numerically.
// ─────────────────────────────────────────────────────────────────────────────
void test_conv2d_grad() {
    std::cout << "\n[Conv2d grad check]\n";

    // Fixed input
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{1,1,4,4},
        std::vector<float>{
            0.1f,0.2f,0.3f,0.4f,
            0.5f,0.6f,0.7f,0.8f,
            0.9f,1.0f,1.1f,1.2f,
            1.3f,1.4f,1.5f,1.6f
        }
    );

    nn::Conv2d conv(1, 1, 2, 1, 0);

    // loss = sum of all output elements
    auto run_loss = [&]() -> float {
        auto out = conv.forward(x);
        float s = 0.0f;
        const float* op = out.raw_ptr();
        for (size_t i = 0; i < out.num_elements(); ++i) s += op[i];
        return s;
    };

    // Analytical backward
    {
        auto out = conv.forward(x);
        // seed grad = all ones (d(sum)/d(out) = 1 everywhere)
        float* gop = out.grad().raw_ptr();
        for (size_t i = 0; i < out.num_elements(); ++i) gop[i] = 1.0f;
        out.backward();
    }

    // Compare analytical vs numerical for each weight
    float* W_ptr  = conv.W->raw_ptr();
    float* gW_ptr = conv.W->grad_->raw_ptr();
    size_t nW = conv.W->num_elements();

    bool all_ok = true;
    for (size_t i = 0; i < nW; ++i) {
        float num = numerical_grad(run_loss, W_ptr, i);
        if (!close(gW_ptr[i], num, 1e-2f)) {
            std::cout << "    W[" << i << "]: analytical=" << gW_ptr[i]
                      << " numerical=" << num << "\n";
            all_ok = false;
        }
    }
    check(all_ok, "Conv2d dW matches numerical gradient");

    // Also check d_bias
    float* b_ptr  = conv.bias->raw_ptr();
    float* gb_ptr = conv.bias->grad_->raw_ptr();
    float num_b = numerical_grad(run_loss, b_ptr, 0);
    check(close(gb_ptr[0], num_b, 1e-2f), "Conv2d d_bias matches numerical gradient");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: Gradient check — MaxPool2d
// ─────────────────────────────────────────────────────────────────────────────
void test_maxpool_grad() {
    std::cout << "\n[MaxPool2d grad check]\n";

    auto x_data = std::vector<float>{
        1,3,2,4, 5,6,7,8, 9,10,11,12, 13,14,15,16
    };
    auto x = std::make_shared<Tensor>(
        std::vector<size_t>{1,1,4,4}, x_data
    );

    nn::MaxPool2d pool(2, 2);

    auto run_loss = [&]() -> float {
        // rebuild x from x_data each time (numerical grad modifies raw ptr)
        auto out = pool.forward(x);
        float s = 0.0f;
        const float* op = out.raw_ptr();
        for (size_t i = 0; i < out.num_elements(); ++i) s += op[i];
        return s;
    };

    // Analytical
    {
        auto out = pool.forward(x);
        float* gop = out.grad().raw_ptr();
        for (size_t i = 0; i < out.num_elements(); ++i) gop[i] = 1.0f;
        out.backward();
    }

    float* xp  = x->raw_ptr();
    float* gxp = x->grad_->raw_ptr();
    size_t nX = x->num_elements();

    bool all_ok = true;
    for (size_t i = 0; i < nX; ++i) {
        float num = numerical_grad(run_loss, xp, i);
        // non-max positions have grad=0, max positions have grad=1
        if (!close(gxp[i], num, 5e-2f)) {
            std::cout << "    x[" << i << "]: analytical=" << gxp[i]
                      << " numerical=" << num << "\n";
            all_ok = false;
        }
    }
    check(all_ok, "MaxPool2d d_input matches numerical gradient");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: End-to-end forward shape through a mini CNN
// ─────────────────────────────────────────────────────────────────────────────
void test_e2e_shape() {
    std::cout << "\n[End-to-end CNN shape]\n";

    // Simulate CIFAR architecture
    nn::Conv2d  conv1(3, 32, 3, 1, 1);
    nn::ReLU    relu1;
    nn::MaxPool2d pool1(2, 2);
    nn::Conv2d  conv2(32, 64, 3, 1, 1);
    nn::ReLU    relu2;
    nn::MaxPool2d pool2(2, 2);
    nn::Flatten flat;
    nn::Linear  fc(64*8*8, 10);

    auto x = std::make_shared<Tensor>(Tensor::randn({2, 3, 32, 32}));

    auto h1 = std::make_shared<Tensor>(conv1.forward(x));
    check(h1->shape_[1]==32 && h1->shape_[2]==32, "after conv1: {2,32,32,32}");

    auto h2 = std::make_shared<Tensor>(relu1.forward(h1));
    auto h3 = std::make_shared<Tensor>(pool1.forward(h2));
    check(h3->shape_[2]==16, "after pool1: {2,32,16,16}");

    auto h4 = std::make_shared<Tensor>(conv2.forward(h3));
    check(h4->shape_[1]==64, "after conv2: {2,64,16,16}");

    auto h5 = std::make_shared<Tensor>(relu2.forward(h4));
    auto h6 = std::make_shared<Tensor>(pool2.forward(h5));
    check(h6->shape_[2]==8, "after pool2: {2,64,8,8}");

    auto h7 = std::make_shared<Tensor>(flat.forward(h6));
    check(h7->shape_[1]==4096, "after flatten: {2,4096}");

    auto out = fc.forward(h7);
    check(out.shape_[0]==2 && out.shape_[1]==10, "after fc: {2,10}");
}

int main() {
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  StakML — Conv2d / MaxPool2d / Flatten Tests\n";
    std::cout << "══════════════════════════════════════════\n";

    test_im2col_shape();
    test_conv2d_shape();
    test_maxpool_shape();
    test_flatten_shape();
    test_conv2d_values();
    test_maxpool_values();
    test_conv2d_grad();
    test_maxpool_grad();
    test_e2e_shape();

    std::cout << "\n───────────────────────────────────────────\n";
    std::cout << "  " << passed << " passed, " << failed << " failed\n";
    std::cout << "───────────────────────────────────────────\n";
    return failed > 0 ? 1 : 0;
}