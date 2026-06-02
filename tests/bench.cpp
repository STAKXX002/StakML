#include "stakml/tensor.hpp"
#include "stakml/ops.hpp"
#include "stakml/nn.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

using namespace stakml;
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ── timer helper ─────────────────────────────────────────────────────────────
struct Timer {
    std::chrono::time_point<Clock> t0;
    void start() { t0 = Clock::now(); }
    double ms() const { return Ms(Clock::now() - t0).count(); }
};

// ── bench helper ─────────────────────────────────────────────────────────────
// runs fn() `warmup` times (discarded), then `runs` times, reports stats
template<typename Fn>
void bench(const std::string& label, int warmup, int runs, Fn fn) {
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> times(runs);
    for (int i = 0; i < runs; ++i) {
        Timer t; t.start();
        fn();
        times[i] = t.ms();
    }

    double sum = 0, mn = times[0], mx = times[0];
    for (double v : times) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
    double avg = sum / runs;

    std::cout << std::left << std::setw(40) << label
              << "  avg=" << std::fixed << std::setprecision(3) << avg   << "ms"
              << "  min=" << mn << "ms"
              << "  max=" << mx << "ms"
              << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. matmul at various sizes
// ─────────────────────────────────────────────────────────────────────────────
void bench_matmul() {
    std::cout << "\n── matmul (naive triple loop) ──────────────────────────\n";

    for (size_t N : {32u, 64u, 128u, 256u, 512u}) {
        auto a = std::make_shared<Tensor>(Tensor::randn({N, N}));
        auto b = std::make_shared<Tensor>(Tensor::randn({N, N}));
        std::string label = "matmul " + std::to_string(N) + "x" + std::to_string(N);
        bench(label, 2, 10, [&]{ ops::matmul(a, b); });
    }

    // Rectangular: typical MLP shapes
    std::cout << "\n── matmul (MLP-realistic shapes) ───────────────────────\n";
    struct Shape { size_t M, K, N; std::string name; };
    for (auto s : std::vector<Shape>{
        {32,  784, 128, "batch=32  784→128"},
        {32,  128,  64, "batch=32  128→64"},
        {32,   64,  10, "batch=32   64→10"},
        {128, 784, 128, "batch=128 784→128"},
    }) {
        auto a = std::make_shared<Tensor>(Tensor::randn({s.M, s.K}));
        auto b = std::make_shared<Tensor>(Tensor::randn({s.K, s.N}));
        bench(s.name, 2, 20, [&]{ ops::matmul(a, b); });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. element-wise ops
// ─────────────────────────────────────────────────────────────────────────────
void bench_elementwise() {
    std::cout << "\n── element-wise ops (1024 elements) ────────────────────\n";
    auto x = std::make_shared<Tensor>(Tensor::randn({32, 32}));
    bench("relu   32x32", 2, 100, [&]{ ops::relu(x); });
    bench("sigmoid 32x32", 2, 100, [&]{ ops::sigmoid(x); });

    std::cout << "\n── element-wise ops (100352 elements = 784*128) ────────\n";
    auto y = std::make_shared<Tensor>(Tensor::randn({784, 128}));
    bench("relu   784x128", 2, 100, [&]{ ops::relu(y); });
    bench("sigmoid 784x128", 2, 100, [&]{ ops::sigmoid(y); });
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. full MLP forward pass
// ─────────────────────────────────────────────────────────────────────────────
void bench_mlp_forward() {
    std::cout << "\n── MLP forward pass (784→128→64→10) ────────────────────\n";

    nn::Linear fc1(784, 128), fc2(128, 64), fc3(64, 10);

    for (size_t batch : {1u, 8u, 32u, 128u}) {
        auto x = std::make_shared<Tensor>(Tensor::randn({batch, 784}));

        std::string label = "batch=" + std::to_string(batch);
        bench(label, 2, 50, [&]{
            auto h1    = fc1.forward(x);
            auto h1r   = ops::relu(std::make_shared<Tensor>(h1));
            auto h2    = fc2.forward(std::make_shared<Tensor>(h1r));
            auto h2r   = ops::relu(std::make_shared<Tensor>(h2));
            auto out   = fc3.forward(std::make_shared<Tensor>(h2r));
            auto probs = ops::softmax(std::make_shared<Tensor>(out));
            (void)probs;
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. allocation count proxy
//    Count how many Tensor objects are constructed per forward pass
//    by temporarily overriding — we can't hook the ctor, so we
//    just count ops manually and report expected allocs.
// ─────────────────────────────────────────────────────────────────────────────
void report_allocs() {
    std::cout << "\n── allocation analysis (784→128→64→10, batch=32) ──────\n";
    std::cout << "  Each op allocates one new Tensor (new vector<float>)\n";
    std::cout << "  fc1.forward : matmul + add_bias          = 2 allocs\n";
    std::cout << "  relu        :                             = 1 alloc\n";
    std::cout << "  fc2.forward : matmul + add_bias          = 2 allocs\n";
    std::cout << "  relu        :                             = 1 alloc\n";
    std::cout << "  fc3.forward : matmul + add_bias          = 2 allocs\n";
    std::cout << "  softmax     :                             = 1 alloc\n";
    std::cout << "  ─────────────────────────────────────────────────────\n";
    std::cout << "  total per forward pass                   = 9 allocs\n";
    std::cout << "  each alloc touches heap + zero-inits the buffer\n";
    std::cout << "  at batch=32, 784→128: alloc = 32*128*4 = 16384 bytes\n";
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << "  StakML — Performance Benchmark\n";
    std::cout << "══════════════════════════════════════════════════════\n";

    bench_matmul();
    bench_elementwise();
    bench_mlp_forward();
    report_allocs();

    std::cout << "\n══════════════════════════════════════════════════════\n";
    return 0;
}