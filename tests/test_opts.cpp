#include "stakml/tensor.hpp"
#include "stakml/ops.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include <string>

using namespace stakml;

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

#define ASSERT_NEAR(a, b, eps) do {                              \
    if (std::abs((float)(a)-(float)(b)) > (float)(eps))          \
        throw std::runtime_error(                                \
            std::string("Expected ~") + std::to_string((float)b) \
            + " got " + std::to_string((float)a)                 \
            + " at line " + std::to_string(__LINE__));           \
} while(0)

// ─────────────────────────────────────────────────────────────────────────────
// add_bias
// ─────────────────────────────────────────────────────────────────────────────

// basic correctness already in test_graph — these add edge cases

void test_add_bias_single_row() {
    auto x = std::make_shared<Tensor>(std::vector<size_t>{1,3},
                                      std::vector<float>{1,2,3});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{1,3},
                                      std::vector<float>{10,20,30});
    Tensor r = ops::add_bias(x, b);
    ASSERT_NEAR(r.at({0,0}), 11.0f, 1e-5f);
    ASSERT_NEAR(r.at({0,1}), 22.0f, 1e-5f);
    ASSERT_NEAR(r.at({0,2}), 33.0f, 1e-5f);
}

void test_add_bias_large() {
    // 128 rows, 256 cols — catches any stride/index mistake
    size_t rows = 128, cols = 256;
    std::vector<float> xdata(rows*cols), bdata(cols);
    for (size_t i = 0; i < rows*cols; ++i) xdata[i] = (float)i;
    for (size_t j = 0; j < cols; ++j)      bdata[j]  = (float)j * 0.1f;

    auto x = std::make_shared<Tensor>(std::vector<size_t>{rows, cols}, xdata);
    auto b = std::make_shared<Tensor>(std::vector<size_t>{1,    cols}, bdata);
    Tensor r = ops::add_bias(x, b);

    // spot-check every row gets the same bias
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float expected = xdata[i*cols + j] + bdata[j];
            ASSERT_NEAR(r.at({i,j}), expected, 1e-4f);
        }
    }
}

void test_add_bias_negative_values() {
    auto x = std::make_shared<Tensor>(std::vector<size_t>{2,3},
                                      std::vector<float>{-1,-2,-3, 4,5,6});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{1,3},
                                      std::vector<float>{1,1,1});
    Tensor r = ops::add_bias(x, b);
    ASSERT_NEAR(r.at({0,0}),  0.0f, 1e-5f);
    ASSERT_NEAR(r.at({0,1}), -1.0f, 1e-5f);
    ASSERT_NEAR(r.at({1,2}),  7.0f, 1e-5f);
}

// ─────────────────────────────────────────────────────────────────────────────
// softmax
// ─────────────────────────────────────────────────────────────────────────────

void test_softmax_rows_sum_to_one() {
    Tensor x(std::vector<size_t>{4,5});
    // fill with arbitrary values
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 5; ++j)
            x.at({i,j}) = (float)(i*5+j) - 10.0f;
    Tensor s = x.softmax();
    for (size_t i = 0; i < 4; ++i) {
        float sum = 0;
        for (size_t j = 0; j < 5; ++j) sum += s.at({i,j});
        ASSERT_NEAR(sum, 1.0f, 1e-5f);
    }
}

void test_softmax_all_same_input() {
    // equal logits → uniform distribution
    Tensor x(std::vector<size_t>{2,4}, 1.0f);
    Tensor s = x.softmax();
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j)
            ASSERT_NEAR(s.at({i,j}), 0.25f, 1e-5f);
}

void test_softmax_large_values_stable() {
    // without max subtraction this would overflow to inf
    Tensor x(std::vector<size_t>{1,3}, std::vector<float>{1000.f, 1001.f, 1002.f});
    Tensor s = x.softmax();
    float sum = 0;
    for (size_t j = 0; j < 3; ++j) sum += s.at({0,j});
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
    // largest logit should have largest probability
    ASSERT(s.at({0,2}) > s.at({0,1}));
    ASSERT(s.at({0,1}) > s.at({0,0}));
}

void test_softmax_known_values() {
    // softmax([0, 1, 2]) = [0.0900, 0.2447, 0.6652] (approx)
    Tensor x(std::vector<size_t>{1,3}, std::vector<float>{0.f, 1.f, 2.f});
    Tensor s = x.softmax();
    ASSERT_NEAR(s.at({0,0}), 0.0900f, 1e-3f);
    ASSERT_NEAR(s.at({0,1}), 0.2447f, 1e-3f);
    ASSERT_NEAR(s.at({0,2}), 0.6652f, 1e-3f);
}

// ─────────────────────────────────────────────────────────────────────────────
// sum(axis)
// ─────────────────────────────────────────────────────────────────────────────

void test_sum_axis0_basic() {
    // [[1,2,3],[4,5,6]] sum axis=0 → [5,7,9]
    Tensor x(std::vector<size_t>{2,3}, std::vector<float>{1,2,3,4,5,6});
    Tensor s = x.sum(0);
    ASSERT(s.shape_[0] == 3);
    ASSERT_NEAR(s.at({0}), 5.0f, 1e-5f);
    ASSERT_NEAR(s.at({1}), 7.0f, 1e-5f);
    ASSERT_NEAR(s.at({2}), 9.0f, 1e-5f);
}

void test_sum_axis1_basic() {
    // [[1,2,3],[4,5,6]] sum axis=1 → [6,15]
    Tensor x(std::vector<size_t>{2,3}, std::vector<float>{1,2,3,4,5,6});
    Tensor s = x.sum(1);
    ASSERT(s.shape_[0] == 2);
    ASSERT_NEAR(s.at({0}), 6.0f,  1e-5f);
    ASSERT_NEAR(s.at({1}), 15.0f, 1e-5f);
}

void test_sum_axis0_large() {
    // 128 rows, 64 cols — each col sum = 128 * col_index (if row i, col j = j)
    size_t rows = 128, cols = 64;
    std::vector<float> data(rows*cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            data[i*cols + j] = (float)j;
    Tensor x(std::vector<size_t>{rows, cols}, data);
    Tensor s = x.sum(0);
    for (size_t j = 0; j < cols; ++j)
        ASSERT_NEAR(s.at({j}), (float)(rows * j), 1e-3f);
}

void test_sum_axis1_large() {
    // 64 rows, 128 cols — row sum = sum(0..127) = 8128, same for each row
    size_t rows = 64, cols = 128;
    std::vector<float> data(rows*cols);
    float expected_row_sum = 0;
    for (size_t j = 0; j < cols; ++j) expected_row_sum += (float)j;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            data[i*cols + j] = (float)j;
    Tensor x(std::vector<size_t>{rows, cols}, data);
    Tensor s = x.sum(1);
    for (size_t i = 0; i < rows; ++i)
        ASSERT_NEAR(s.at({i}), expected_row_sum, 1e-2f);
}

void test_sum_axis_out_of_range() {
    Tensor x(std::vector<size_t>{2,3});
    bool threw = false;
    try { x.sum(2); } catch (const std::runtime_error&) { threw = true; }
    ASSERT(threw);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  StakML — Optimization Correctness Tests \n";
    std::cout << "══════════════════════════════════════════\n";

    std::cout << "\n── add_bias ──────────────────────────────\n";
    RUN_TEST(test_add_bias_single_row);
    RUN_TEST(test_add_bias_large);
    RUN_TEST(test_add_bias_negative_values);

    std::cout << "\n── softmax ───────────────────────────────\n";
    RUN_TEST(test_softmax_rows_sum_to_one);
    RUN_TEST(test_softmax_all_same_input);
    RUN_TEST(test_softmax_large_values_stable);
    RUN_TEST(test_softmax_known_values);

    std::cout << "\n── sum(axis) ─────────────────────────────\n";
    RUN_TEST(test_sum_axis0_basic);
    RUN_TEST(test_sum_axis1_basic);
    RUN_TEST(test_sum_axis0_large);
    RUN_TEST(test_sum_axis1_large);
    RUN_TEST(test_sum_axis_out_of_range);

    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "  " << pass_count << " / " << (pass_count+fail_count) << " passed\n";
    std::cout << "══════════════════════════════════════════\n";
    return fail_count > 0 ? 1 : 0;
}