#include "stakml/tensor.hpp"
#include "stakml/ops.hpp"
#include "stakml/nn.hpp"
#include "stakml/conv.hpp"
#include "stakml/loss.hpp"
#include "stakml/optim.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>
#include <stdexcept>

using namespace stakml;

// ─────────────────────────────────────────────────────────────────────────────
// CIFAR-10 loader
//
// Binary format (cifar-10-batches-bin):
//   Each file contains 10000 records.
//   Each record = 1 label byte + 3072 pixel bytes (3×32×32, RGB, channel-first)
//   Pixel values are uint8 in [0,255].
//
// We load all 5 training batches + the test batch.
// We normalise to [0,1] (divide by 255) — simple but effective.
// ─────────────────────────────────────────────────────────────────────────────
struct CIFARDataset {
    Tensor images;           // {N, 3, 32, 32}  float32 in [0,1]
    std::vector<int> labels; // {N}
    size_t num_samples;

    static CIFARDataset load(const std::vector<std::string>& bin_paths) {
        constexpr size_t RECORD_SIZE = 1 + 3 * 32 * 32; // 3073 bytes
        constexpr size_t PIXELS      = 3 * 32 * 32;     // 3072

        // Count total samples across all files
        size_t total = 0;
        for (const auto& p : bin_paths) {
            std::ifstream f(p, std::ios::binary | std::ios::ate);
            if (!f) throw std::runtime_error("Cannot open: " + p);
            size_t bytes = f.tellg();
            if (bytes % RECORD_SIZE != 0)
                throw std::runtime_error("Bad file size: " + p);
            total += bytes / RECORD_SIZE;
        }

        CIFARDataset ds;
        ds.num_samples = total;
        ds.images  = Tensor({total, 3, 32, 32}, 0.0f);
        ds.labels.resize(total);

        float* img_ptr = ds.images.raw_ptr();
        size_t offset  = 0;

        for (const auto& p : bin_paths) {
            std::ifstream f(p, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open: " + p);

            // Read whole file in one shot
            f.seekg(0, std::ios::end);
            size_t file_bytes = f.tellg();
            f.seekg(0, std::ios::beg);
            std::vector<uint8_t> buf(file_bytes);
            f.read(reinterpret_cast<char*>(buf.data()), file_bytes);
            size_t n = file_bytes / RECORD_SIZE;

            for (size_t i = 0; i < n; ++i) {
                const uint8_t* rec = buf.data() + i * RECORD_SIZE;
                ds.labels[offset + i] = static_cast<int>(rec[0]);

                // Pixels follow: [R plane 1024][G plane 1024][B plane 1024]
                // Already channel-first — just normalise and copy
                for (size_t px = 0; px < PIXELS; ++px)
                    img_ptr[(offset + i) * PIXELS + px] =
                        static_cast<float>(rec[1 + px]) / 255.0f;
            }
            offset += n;
        }

        return ds;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Per-channel mean/std normalisation (computed on training set, applied to both)
//
// Standard CIFAR-10 stats (pre-computed from the training set):
//   mean: R=0.4914  G=0.4822  B=0.4465
//   std:  R=0.2470  G=0.2435  B=0.2616
//
// Normalising helps Adam converge faster — without it, channels with
// different magnitudes require very different effective learning rates.
// ─────────────────────────────────────────────────────────────────────────────
void normalize_cifar(CIFARDataset& ds) {
    static const float mean[3] = {0.4914f, 0.4822f, 0.4465f};
    static const float std_[3] = {0.2470f, 0.2435f, 0.2616f};

    float* p = ds.images.raw_ptr();
    size_t N = ds.num_samples;
    constexpr size_t plane = 32 * 32; // 1024 pixels per channel

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < 3; ++c) {
            float* ch = p + n * 3 * plane + c * plane;
            float m = mean[c], s = std_[c];
            for (size_t px = 0; px < plane; ++px)
                ch[px] = (ch[px] - m) / s;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Build a batch tensor {batch_size, 3, 32, 32} from a dataset + index list
// ─────────────────────────────────────────────────────────────────────────────
std::pair<std::shared_ptr<Tensor>, std::vector<int>>
make_batch(const CIFARDataset& ds,
           const std::vector<size_t>& indices,
           size_t start, size_t batch_size)
{
    constexpr size_t PIXELS = 3 * 32 * 32;

    auto X = std::make_shared<Tensor>(
        std::vector<size_t>{batch_size, 3, 32, 32}, 0.0f);
    std::vector<int> Y(batch_size);

    float*       xp  = X->raw_ptr();
    const float* src = ds.images.raw_ptr();

    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx = indices[start + i];
        std::copy(src + idx * PIXELS, src + (idx + 1) * PIXELS,
                  xp + i * PIXELS);
        Y[i] = ds.labels[idx];
    }
    return {X, Y};
}

int main() {
    std::cout << "══════════════════════════════════════════\n";
    std::cout << "  StakML — CIFAR-10 CNN Training\n";
    std::cout << "══════════════════════════════════════════\n\n";

    // ── 1. Load data ──────────────────────────────────────────────────────────
    // Expects CIFAR-10 binary batches in ../data/cifar-10-batches-bin/
    const std::string base = "../data/cifar-10-batches-bin/";
    std::vector<std::string> train_files = {
        base + "data_batch_1.bin",
        base + "data_batch_2.bin",
        base + "data_batch_3.bin",
        base + "data_batch_4.bin",
        base + "data_batch_5.bin",
    };
    std::string test_file = base + "test_batch.bin";

    std::cout << "Loading CIFAR-10...\n";
    CIFARDataset train_ds, test_ds;
    try {
        train_ds = CIFARDataset::load(train_files);
        test_ds  = CIFARDataset::load({test_file});
    } catch (const std::exception& e) {
        std::cerr << "Failed to load CIFAR-10: " << e.what() << "\n";
        std::cerr << "Download from: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz\n";
        std::cerr << "Extract to: ../data/cifar-10-batches-bin/\n";
        return 1;
    }
    std::cout << "Loaded " << train_ds.num_samples << " training images.\n";
    std::cout << "Loaded " << test_ds.num_samples  << " test images.\n";

    // ── 2. Normalise ──────────────────────────────────────────────────────────
    std::cout << "Normalising (per-channel mean/std)...\n\n";
    normalize_cifar(train_ds);
    normalize_cifar(test_ds);

    // ── 3. Model ──────────────────────────────────────────────────────────────
    //
    // Architecture:
    //   Conv2d(3→32,  k=3, pad=1) → ReLU → MaxPool(2)   {N,32,16,16}
    //   Conv2d(32→64, k=3, pad=1) → ReLU → MaxPool(2)   {N,64,8,8}
    //   Flatten                                           {N,4096}
    //   Linear(4096→256) → ReLU
    //   Linear(256→10)
    //
    // Target: ~70% test accuracy (respectable for a from-scratch CNN with no
    //         data augmentation or BatchNorm)
    //
    nn::Sequential model({
        std::make_shared<nn::Conv2d>(3, 32, 3, 1, 1),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::MaxPool2d>(2, 2),

        std::make_shared<nn::Conv2d>(32, 64, 3, 1, 1),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::MaxPool2d>(2, 2),

        std::make_shared<nn::Flatten>(),

        std::make_shared<nn::Linear>(64 * 8 * 8, 256),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(256, 10),
    });

    // ── 4. Optimizer ──────────────────────────────────────────────────────────
    // lr=1e-3 is the standard Adam default.
    // We step down to 1e-4 at epoch 15 (see training loop).
    optim::Adam opt(model.parameters(), 1e-3f);

    // ── 5. Hyperparameters ────────────────────────────────────────────────────
    constexpr size_t BATCH_SIZE  = 64;   // smaller than MNIST due to 4D tensors
    constexpr size_t EPOCHS      = 20;
    const size_t     N_TRAIN     = train_ds.num_samples;
    const size_t     N_TEST      = test_ds.num_samples;
    const size_t     TRAIN_STEPS = N_TRAIN / BATCH_SIZE;
    const size_t     TEST_STEPS  = N_TEST  / BATCH_SIZE;

    // Shuffle index arrays — we reshuffle each epoch
    std::vector<size_t> train_idx(N_TRAIN), test_idx(N_TEST);
    std::iota(train_idx.begin(), train_idx.end(), 0);
    std::iota(test_idx.begin(),  test_idx.end(),  0);
    std::mt19937 rng(42);

    std::cout << "Training (epochs=" << EPOCHS
              << ", batch=" << BATCH_SIZE
              << ", steps/epoch=" << TRAIN_STEPS << ")\n";
    std::cout << "──────────────────────────────────────────────────────────────\n";

    // ── 6. Training loop ──────────────────────────────────────────────────────
    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Learning rate schedule: drop at epoch 15
        if (epoch == 15) {
            opt.lr_ = 1e-4f;
            std::cout << "  [lr → 1e-4]\n";
        }

        // Shuffle training indices each epoch
        std::shuffle(train_idx.begin(), train_idx.end(), rng);

        // ── A. Train ──────────────────────────────────────────────────────────
        float total_loss  = 0.0f;
        int   train_corr  = 0;

        for (size_t step = 0; step < TRAIN_STEPS; ++step) {
            auto [X, Y] = make_batch(train_ds, train_idx, step * BATCH_SIZE, BATCH_SIZE);

            opt.zero_grad();

            Tensor logits   = model.forward(X);
            auto   lp_ptr   = std::make_shared<Tensor>(
                                  ops::log_softmax(
                                      std::make_shared<Tensor>(logits)));

            float loss = ops::nll_loss(*lp_ptr, Y);
            total_loss += loss;
            train_corr += static_cast<int>(
                ops::accuracy(logits, Y) * static_cast<float>(BATCH_SIZE));

            lp_ptr->backward();
            opt.step();
        }

        // ── B. Test (no grad) ─────────────────────────────────────────────────
        int test_corr = 0;
        for (size_t step = 0; step < TEST_STEPS; ++step) {
            auto [X, Y] = make_batch(test_ds, test_idx, step * BATCH_SIZE, BATCH_SIZE);
            Tensor logits = model.forward(X);
            test_corr += static_cast<int>(
                ops::accuracy(logits, Y) * static_cast<float>(BATCH_SIZE));
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        float avg_loss  = total_loss / static_cast<float>(TRAIN_STEPS);
        float train_acc = 100.0f * static_cast<float>(train_corr) /
                          static_cast<float>(TRAIN_STEPS * BATCH_SIZE);
        float test_acc  = 100.0f * static_cast<float>(test_corr)  /
                          static_cast<float>(TEST_STEPS  * BATCH_SIZE);

        std::cout << "Epoch " << std::setw(2) << epoch + 1 << "/" << EPOCHS
                  << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | Train: " << std::setprecision(2) << train_acc << "%"
                  << " | Test: "  << std::setprecision(2) << test_acc  << "%"
                  << " | " << std::setprecision(1) << elapsed << "s\n";
    }

    std::cout << "──────────────────────────────────────────────────────────────\n";
    std::cout << "Done.\n";
    return 0;
}