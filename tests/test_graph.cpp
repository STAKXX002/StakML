#include "stakml/tensor.hpp"
#include "stakml/ops.hpp"
#include "stakml/nn.hpp"
#include <cassert>
#include <iostream>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Test harness (same pattern as test_tensor.cpp)
// ─────────────────────────────────────────────────────────────────────────────

static int pass_count = 0;
static int fail_count = 0;

#define RUN_TEST(fn) do {                           \
    try {                                           \
        fn();                                       \
        std::cout << "  [PASS] " #fn "\n";          \
        ++pass_count;                               \
    } catch (std::exception& e) {                   \
        std::cout << "  [FAIL] " #fn "\n"           \
                  << "        " << e.what() <<"\n"; \
        ++fail_count;                               \
    }                                               \
} while(0)

#define ASSERT(cond) do {                           \
    if (!(cond)) throw std::runtime_error(          \
        "Assertion failed: " #cond                  \
        " at line " + std::to_string(__LINE__));    \
} while(0)

#define ASSERT_EQ(a, b) do {                                        \
    if ((a) != (b)) throw std::runtime_error(                       \
        std::string("Expected ") + std::to_string(b) +              \
        " got " + std::to_string(a) +                               \
        " at line " + std::to_string(__LINE__));                    \
} while(0)

#define ASSERT_NEAR(a, b, eps) do {                                 \
    if (std::abs((a)-(b)) > (eps)) throw std::runtime_error(        \
        std::string("Expected ~") + std::to_string(b) +             \
        " got " + std::to_string(a) +                               \
        " at line " + std::to_string(__LINE__));                    \
} while(0)

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

using namespace stakml;

// ── ops::matmul tags the result correctly ────────────────────────────────────
void test_ops_matmul_tags() {
    auto a = std::make_shared<Tensor>(Tensor::ones({2, 3}));
    auto b = std::make_shared<Tensor>(Tensor::ones({3, 4}));

    Tensor c = ops::matmul(a, b);

    ASSERT(c.op_name_ == "matmul");
    ASSERT(c.inputs_.size() == 2);
    // inputs_ point to the originals, not copies
    ASSERT(c.inputs_[0] == a);
    ASSERT(c.inputs_[1] == b);
}

// ── ops::matmul shape is correct ─────────────────────────────────────────────
void test_ops_matmul_shape() {
    auto a = std::make_shared<Tensor>(Tensor::ones({2, 3}));
    auto b = std::make_shared<Tensor>(Tensor::ones({3, 5}));
    Tensor c = ops::matmul(a, b);
    ASSERT_EQ(c.shape_[0], 2u);
    ASSERT_EQ(c.shape_[1], 5u);
}

// ── ops::matmul values unchanged from tensor.matmul ─────────────────────────
void test_ops_matmul_values() {
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2,2},
                                      std::vector<float>{1,2,3,4});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{2,2},
                                      std::vector<float>{5,6,7,8});
    Tensor c = ops::matmul(a, b);
    ASSERT_NEAR(c.at({0,0}), 19.0f, 1e-4f);
    ASSERT_NEAR(c.at({0,1}), 22.0f, 1e-4f);
    ASSERT_NEAR(c.at({1,0}), 43.0f, 1e-4f);
    ASSERT_NEAR(c.at({1,1}), 50.0f, 1e-4f);
}

// ── ops::relu tags + values ───────────────────────────────────────────────────
void test_ops_relu_tags_and_values() {
    auto x = std::make_shared<Tensor>(std::vector<size_t>{1,4},
                                      std::vector<float>{-2, 0, 1, 3});
    Tensor r = ops::relu(x);

    ASSERT(r.op_name_ == "relu");
    ASSERT(r.inputs_.size() == 1);
    ASSERT(r.inputs_[0] == x);

    ASSERT_NEAR(r.at({0,0}), 0.0f, 1e-4f);
    ASSERT_NEAR(r.at({0,1}), 0.0f, 1e-4f);
    ASSERT_NEAR(r.at({0,2}), 1.0f, 1e-4f);
    ASSERT_NEAR(r.at({0,3}), 3.0f, 1e-4f);
}

// ── ops::add_bias shape and values ───────────────────────────────────────────
void test_ops_add_bias() {
    // x: 3 rows, each will get bias [10, 20] added
    auto x = std::make_shared<Tensor>(std::vector<size_t>{3,2},
                                      std::vector<float>{1,2, 3,4, 5,6});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{1,2},
                                      std::vector<float>{10, 20});
    Tensor r = ops::add_bias(x, b);

    ASSERT(r.op_name_ == "add_bias");
    ASSERT_EQ(r.shape_[0], 3u);
    ASSERT_EQ(r.shape_[1], 2u);

    ASSERT_NEAR(r.at({0,0}), 11.0f, 1e-4f);
    ASSERT_NEAR(r.at({0,1}), 22.0f, 1e-4f);
    ASSERT_NEAR(r.at({1,0}), 13.0f, 1e-4f);
    ASSERT_NEAR(r.at({2,1}), 26.0f, 1e-4f);
}

// ── ops::add_bias wrong bias shape throws ────────────────────────────────────
void test_ops_add_bias_bad_shape() {
    auto x = std::make_shared<Tensor>(Tensor::zeros({3, 4}));
    auto b = std::make_shared<Tensor>(Tensor::zeros({1, 5}));  // wrong cols
    bool threw = false;
    try { ops::add_bias(x, b); }
    catch (const std::runtime_error&) { threw = true; }
    ASSERT(threw);
}

// ── Linear: output shape ─────────────────────────────────────────────────────
void test_linear_shape() {
    nn::Linear fc(4, 8);
    auto x = std::make_shared<Tensor>(Tensor::randn({3, 4}));  // batch=3
    Tensor out = fc.forward(x);
    ASSERT_EQ(out.shape_[0], 3u);
    ASSERT_EQ(out.shape_[1], 8u);
}

// ── Linear: graph structure ───────────────────────────────────────────────────
// The output node should be add_bias, whose first input is matmul
void test_linear_graph() {
    nn::Linear fc(4, 8);
    auto x = std::make_shared<Tensor>(Tensor::randn({3, 4}));
    Tensor out = fc.forward(x);

    ASSERT(out.op_name_ == "add_bias");
    ASSERT(out.inputs_.size() == 2);
    ASSERT(out.inputs_[0]->op_name_ == "matmul");
}

// ── Linear: W and b are the same objects as in graph ─────────────────────────
// This is the key Week-3 property: gradients written to W->grad_ from
// the backward pass are immediately visible via fc.W->grad_.
void test_linear_shared_params() {
    nn::Linear fc(4, 8);
    auto x = std::make_shared<Tensor>(Tensor::randn({3, 4}));
    Tensor out = fc.forward(x);

    // out = add_bias(xW, b)
    // out.inputs_[1] should BE fc.b (same shared_ptr target)
    ASSERT(out.inputs_[1] == fc.b);

    // out.inputs_[0] = matmul node; its inputs_[1] should BE fc.W
    ASSERT(out.inputs_[0]->inputs_[1] == fc.W);
}

// ── Two-layer MLP: shape flow ─────────────────────────────────────────────────
// fc1: 4→8, relu, fc2: 8→3, softmax
void test_mlp_shape_flow() {
    nn::Linear fc1(4, 8), fc2(8, 3);
    auto x = std::make_shared<Tensor>(Tensor::randn({5, 4}));  // batch=5

    // layer 1
    Tensor h = fc1.forward(x);
    ASSERT_EQ(h.shape_[0], 5u);
    ASSERT_EQ(h.shape_[1], 8u);

    // relu
    auto h_ptr = std::make_shared<Tensor>(h);
    Tensor h_relu = ops::relu(h_ptr);
    ASSERT_EQ(h_relu.shape_[0], 5u);
    ASSERT_EQ(h_relu.shape_[1], 8u);

    // layer 2
    auto h_relu_ptr = std::make_shared<Tensor>(h_relu);
    Tensor out = fc2.forward(h_relu_ptr);
    ASSERT_EQ(out.shape_[0], 5u);
    ASSERT_EQ(out.shape_[1], 3u);

    // softmax
    auto out_ptr = std::make_shared<Tensor>(out);
    Tensor probs = ops::softmax(out_ptr);
    ASSERT_EQ(probs.shape_[0], 5u);
    ASSERT_EQ(probs.shape_[1], 3u);
}

// ── Two-layer MLP: graph depth ────────────────────────────────────────────────
// Walk the graph from softmax back to x and check op names
void test_mlp_graph_depth() {
    nn::Linear fc1(4, 8), fc2(8, 3);
    auto x = std::make_shared<Tensor>(Tensor::randn({2, 4}));

    Tensor h       = fc1.forward(x);
    Tensor h_relu  = ops::relu(std::make_shared<Tensor>(h));
    Tensor out     = fc2.forward(std::make_shared<Tensor>(h_relu));
    Tensor probs   = ops::softmax(std::make_shared<Tensor>(out));

    // probs → softmax
    ASSERT(probs.op_name_ == "softmax");
    // → out (add_bias from fc2)
    ASSERT(probs.inputs_[0]->op_name_ == "add_bias");
    // → matmul (fc2)
    ASSERT(probs.inputs_[0]->inputs_[0]->op_name_ == "matmul");
    // → relu
    ASSERT(probs.inputs_[0]->inputs_[0]->inputs_[0]->op_name_ == "relu");
    // → add_bias (fc1)
    ASSERT(probs.inputs_[0]->inputs_[0]->inputs_[0]->inputs_[0]->op_name_ == "add_bias");
}

// ── Softmax rows sum to 1 ─────────────────────────────────────────────────────
void test_softmax_row_sums() {
    nn::Linear fc(4, 3);
    auto x = std::make_shared<Tensor>(Tensor::randn({5, 4}));
    Tensor out   = fc.forward(x);
    Tensor probs = ops::softmax(std::make_shared<Tensor>(out));

    for (size_t i = 0; i < 5; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < 3; ++j)
            row_sum += probs.at({i, j});
        ASSERT_NEAR(row_sum, 1.0f, 1e-5f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "══════════════════════════════════\n";
    std::cout << "  StakML — Graph / Forward Tests  \n";
    std::cout << "══════════════════════════════════\n";

    RUN_TEST(test_ops_matmul_tags);
    RUN_TEST(test_ops_matmul_shape);
    RUN_TEST(test_ops_matmul_values);
    RUN_TEST(test_ops_relu_tags_and_values);
    RUN_TEST(test_ops_add_bias);
    RUN_TEST(test_ops_add_bias_bad_shape);
    RUN_TEST(test_linear_shape);
    RUN_TEST(test_linear_graph);
    RUN_TEST(test_linear_shared_params);
    RUN_TEST(test_mlp_shape_flow);
    RUN_TEST(test_mlp_graph_depth);
    RUN_TEST(test_softmax_row_sums);

    std::cout << "══════════════════════════════════\n";
    std::cout << "  " << pass_count << " / " << (pass_count + fail_count) << " passed\n";
    std::cout << "══════════════════════════════════\n";

    return fail_count > 0 ? 1 : 0;
}