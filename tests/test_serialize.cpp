#include "stakml/tensor.hpp"
#include "stakml/ops.hpp"
#include "stakml/nn.hpp"
#include "stakml/serialize.hpp"
#include <iostream>
#include <string>
#include <cstdio>

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
    }                                                \
} while(0)

#define ASSERT(cond) do {                           \
    if (!(cond)) throw std::runtime_error(          \
        "Assertion failed: " #cond                  \
        " at line " + std::to_string(__LINE__));    \
} while(0)

static nn::Sequential make_mlp() {
    return nn::Sequential({
        std::make_shared<nn::Linear>(4, 8),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(8, 3)
    });
}

// Save then load into a model with identical architecture but different
// (freshly Xavier-initialized) weights: after load, every parameter must
// match the saved model exactly, since save/load is a raw byte copy with
// no lossy conversion involved.
void test_save_load_round_trip_exact() {
    const std::string path = "/tmp/stakml_test_model.bin";

    nn::Sequential model_a = make_mlp();
    serialize::save_model(model_a, path);

    nn::Sequential model_b = make_mlp(); // different random init
    auto params_a = model_a.parameters();
    auto params_b = model_b.parameters();
    ASSERT(params_a.size() == params_b.size());

    // Sanity: the two independently-initialized models should actually
    // differ before loading, otherwise this test can't tell us anything.
    bool any_diff = false;
    for (size_t i = 0; i < params_a.size(); ++i) {
        for (size_t j = 0; j < params_a[i]->num_elements(); ++j) {
            if (params_a[i]->raw_ptr()[j] != params_b[i]->raw_ptr()[j]) {
                any_diff = true;
                break;
            }
        }
    }
    ASSERT(any_diff);

    serialize::load_model(model_b, path);

    for (size_t i = 0; i < params_a.size(); ++i) {
        ASSERT(params_a[i]->num_elements() == params_b[i]->num_elements());
        for (size_t j = 0; j < params_a[i]->num_elements(); ++j) {
            ASSERT(params_a[i]->raw_ptr()[j] == params_b[i]->raw_ptr()[j]);
        }
    }

    std::remove(path.c_str());
}

// After loading matching weights, forward-pass output on the same input
// must be identical, not just the raw parameter bytes.
void test_save_load_preserves_forward_output() {
    const std::string path = "/tmp/stakml_test_model_fwd.bin";

    nn::Sequential model_a = make_mlp();
    auto x = std::make_shared<Tensor>(Tensor::randn({2, 4}));
    Tensor out_a = model_a.forward(x);

    serialize::save_model(model_a, path);

    nn::Sequential model_b = make_mlp();
    serialize::load_model(model_b, path);
    Tensor out_b = model_b.forward(x);

    ASSERT(out_a.num_elements() == out_b.num_elements());
    for (size_t i = 0; i < out_a.num_elements(); ++i)
        ASSERT(out_a.raw_ptr()[i] == out_b.raw_ptr()[i]);

    std::remove(path.c_str());
}

// Loading from a path that doesn't exist should throw, not silently
// leave the model's parameters untouched or crash.
void test_load_missing_file_throws() {
    nn::Sequential model = make_mlp();
    bool threw = false;
    try {
        serialize::load_model(model, "/tmp/stakml_this_file_does_not_exist.bin");
    } catch (const std::exception&) {
        threw = true;
    }
    ASSERT(threw);
}

int main() {
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  StakML — Serialize (save/load) Tests\n";
    std::cout << "══════════════════════════════════════════\n\n";

    RUN_TEST(test_save_load_round_trip_exact);
    RUN_TEST(test_save_load_preserves_forward_output);
    RUN_TEST(test_load_missing_file_throws);

    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "  " << pass_count << " / "
              << (pass_count + fail_count) << " passed\n";
    std::cout << "══════════════════════════════════════════\n";

    return fail_count > 0 ? 1 : 0;
}