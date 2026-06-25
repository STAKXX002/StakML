#pragma once
#include "tensor.hpp"
#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// dataset.hpp — Standard dataset loaders
//
// Contains loaders for benchmark datasets used to validate the library.
// Domain-specific loaders (e.g. football, tabular CSV) belong in the
// consuming application, not here.
//
// Efficiency note:
//   MNIST files are binary. We read the entire file in a single I/O call
//   into a std::vector<uint8_t>, then convert to float. Avoids per-sample
//   fstream::read() overhead.
// ─────────────────────────────────────────────────────────────────────────────

namespace stakml {
namespace dataset {

// Big-endian → little-endian (MNIST headers are big-endian)
inline uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

// ── MNIST ─────────────────────────────────────────────────────────────────────
// Loads the standard MNIST binary format produced by Yann LeCun's IDX files.
//
// Usage:
//   auto train = dataset::MNIST::load(
//       "../data/train-images-idx3-ubyte",
//       "../data/train-labels-idx1-ubyte");
//   // train.images : Tensor {60000, 784}, values in [0, 1]
//   // train.labels : std::vector<int> {60000}, values in [0, 9]
//
struct MNIST {
    Tensor images;            // {num_samples, 784}, float in [0, 1]
    std::vector<int> labels;  // {num_samples}, class index in [0, 9]
    size_t num_samples;

    static MNIST load(const std::string& image_path, const std::string& label_path) {
        MNIST ds;

        // ── Labels ────────────────────────────────────────────────────────────
        std::ifstream lf(label_path, std::ios::binary);
        if (!lf) throw std::runtime_error("Cannot open: " + label_path);

        uint32_t magic, num_labels;
        lf.read(reinterpret_cast<char*>(&magic),      4);
        lf.read(reinterpret_cast<char*>(&num_labels), 4);
        magic      = swap_endian(magic);
        num_labels = swap_endian(num_labels);
        if (magic != 2049) throw std::runtime_error("Invalid MNIST label magic");

        ds.num_samples = num_labels;
        ds.labels.resize(num_labels);
        std::vector<uint8_t> raw_labels(num_labels);
        lf.read(reinterpret_cast<char*>(raw_labels.data()), num_labels);
        for (size_t i = 0; i < num_labels; ++i)
            ds.labels[i] = static_cast<int>(raw_labels[i]);

        // ── Images ────────────────────────────────────────────────────────────
        std::ifstream imf(image_path, std::ios::binary);
        if (!imf) throw std::runtime_error("Cannot open: " + image_path);

        uint32_t num_images, rows, cols;
        imf.read(reinterpret_cast<char*>(&magic),      4);
        imf.read(reinterpret_cast<char*>(&num_images), 4);
        imf.read(reinterpret_cast<char*>(&rows),       4);
        imf.read(reinterpret_cast<char*>(&cols),       4);
        magic      = swap_endian(magic);
        num_images = swap_endian(num_images);
        rows       = swap_endian(rows);
        cols       = swap_endian(cols);
        if (magic != 2051) throw std::runtime_error("Invalid MNIST image magic");
        if (num_images != num_labels)
            throw std::runtime_error("MNIST image/label count mismatch");

        size_t image_size = rows * cols;  // 784 for standard MNIST
        ds.images = Tensor({num_images, image_size});

        std::vector<uint8_t> raw_images(num_images * image_size);
        imf.read(reinterpret_cast<char*>(raw_images.data()), num_images * image_size);

        float* p = ds.images.raw_ptr();
        for (size_t i = 0; i < num_images * image_size; ++i)
            p[i] = static_cast<float>(raw_images[i]) / 255.0f;

        return ds;
    }
};

} // namespace dataset
} // namespace stakml