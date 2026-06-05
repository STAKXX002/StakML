#include "stakml/tensor.hpp"
#include "stakml/ops.hpp"
#include "stakml/nn.hpp"
#include "stakml/loss.hpp"
#include "stakml/optim.hpp"
#include "stakml/dataset.hpp"
#include "stakml/serialize.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>
#include <sys/stat.h>

using namespace stakml;

// ── Visual Helpers ───────────────────────────────────────────────────────────

void print_ascii_digit(const float* pixels, size_t rows = 28, size_t cols = 28) {
    const char* gradient = " .:-=+*#%@";
    std::cout << "┌────────────────────────────┐\n";
    for (size_t r = 0; r < rows; ++r) {
        std::cout << "│";
        for (size_t c = 0; c < cols; ++c) {
            float val = pixels[r * cols + c];
            int idx = static_cast<int>(val * 9.0f);
            if (idx < 0) idx = 0;
            if (idx > 9) idx = 9;
            std::cout << gradient[idx];
        }
        std::cout << "│\n";
    }
    std::cout << "└────────────────────────────┘\n";
}

void print_probability_bars(const Tensor& log_probs, int actual_label) {
    std::cout << "Network Confidence:\n";
    for (int i = 0; i < 10; ++i) {
        float prob = std::exp(log_probs.at({0, static_cast<size_t>(i)}));
        int bar_len = static_cast<int>(prob * 20.0f);
        
        std::cout << "  [" << i << "] ";
        if (i == actual_label) std::cout << "\033[1;32m"; 
        
        for (int b = 0; b < 20; ++b) {
            if (b < bar_len) std::cout << "█";
            else std::cout << "░";
        }
        
        if (i == actual_label) std::cout << "\033[0m"; 
        std::cout << " " << std::fixed << std::setprecision(1) << (prob * 100.0f) << "%\n";
    }
}

inline bool file_exists(const std::string& name) {
    struct stat buffer;   
    return (stat(name.c_str(), &buffer) == 0); 
}

// ── Main Demo ────────────────────────────────────────────────────────────────

int main() {
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  StakML — Visual Inference Demo\n";
    std::cout << "══════════════════════════════════════════\n\n";

    nn::Sequential model({
        std::make_shared<nn::Linear>(784, 128), std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(128, 64),  std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(64, 10)
    });

    std::string model_file = "mnist_model.stak";

    if (file_exists(model_file)) {
        std::cout << "1. Found saved weights! Loading model directly...\n";
        serialize::load_model(model, model_file);
    } else {
        std::cout << "1. No saved model found. Booting up and fast-training for 1 Epoch...\n";
        dataset::MNIST train_data = dataset::MNIST::load("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte");
        optim::SGD opt(model.parameters(), 0.1f);
        size_t batch_size = 128;
        
        for (size_t b = 0; b < train_data.num_samples / batch_size; ++b) {
            auto X_batch = std::make_shared<Tensor>(std::vector<size_t>{batch_size, 784});
            std::vector<int> Y_batch(batch_size);
            size_t offset = b * batch_size;
            std::copy(train_data.images.raw_ptr() + offset * 784, train_data.images.raw_ptr() + (offset + batch_size) * 784, X_batch->raw_ptr());
            for (size_t i = 0; i < batch_size; ++i) Y_batch[i] = train_data.labels[offset + i];

            opt.zero_grad();
            auto log_probs = ops::log_softmax(std::make_shared<Tensor>(model.forward(X_batch)));
            ops::nll_loss(log_probs, Y_batch);
            log_probs.backward();
            opt.step();
        }
        std::cout << "   Training complete. Saving weights to disk...\n";
        serialize::save_model(model, model_file);
    }

    std::cout << "2. Loading Test Images...\n\n";
    dataset::MNIST test_data  = dataset::MNIST::load("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");

    // ── Interactive Demo Loop ──
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, test_data.num_samples - 1);

    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  ENTERING VISUAL INFERENCE MODE\n";
    std::cout << "══════════════════════════════════════════\n";

    for (int i = 0; i < 3; ++i) { 
        size_t idx = dist(rng);
        
        std::cout << "\nTest Subject #" << idx << ":\n";
        print_ascii_digit(test_data.images.raw_ptr() + (idx * 784));

        auto X_single = std::make_shared<Tensor>(std::vector<size_t>{1, 784});
        std::copy(test_data.images.raw_ptr() + (idx * 784), test_data.images.raw_ptr() + ((idx + 1) * 784), X_single->raw_ptr());

        auto logits = model.forward(X_single);
        auto log_probs = ops::log_softmax(std::make_shared<Tensor>(logits));

        int actual_label = test_data.labels[idx];
        print_probability_bars(log_probs, actual_label);
        
        std::cout << "──────────────────────────────────────────\n";
    }

    return 0;
}