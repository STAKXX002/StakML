#pragma once
#include "tensor.hpp"
#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// dataset.hpp — Data loading utilities
//
// Efficiency Note:
//   MNIST files are binary. To save time and CPU cycles, we read the entire
//   file block into a std::vector<uint8_t> in a single I/O operation, rather 
//   than looping over fstream::read(). We then convert bytes to floats.
// ─────────────────────────────────────────────────────────────────────────────

namespace stakml {
namespace dataset {

// Helper to swap endianness (MNIST files are Big-Endian, most modern CPUs are Little-Endian)
inline uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

struct MNIST {
    Tensor images;           // Shape: {num_images, 784}
    std::vector<int> labels; // Shape: {num_images}
    size_t num_samples;

    static MNIST load(const std::string& image_path, const std::string& label_path) {
        MNIST dataset;

        // ─── 1. Load Labels ──────────────────────────────────────────────────
        std::ifstream label_file(label_path, std::ios::binary);
        if (!label_file) throw std::runtime_error("Cannot open " + label_path);

        uint32_t magic, num_labels;
        label_file.read(reinterpret_cast<char*>(&magic), 4);
        label_file.read(reinterpret_cast<char*>(&num_labels), 4);
        magic = swap_endian(magic);
        num_labels = swap_endian(num_labels);

        if (magic != 2049) throw std::runtime_error("Invalid MNIST label file magic number");

        dataset.num_samples = num_labels;
        dataset.labels.resize(num_labels);
        
        // Single block read for all labels
        std::vector<uint8_t> raw_labels(num_labels);
        label_file.read(reinterpret_cast<char*>(raw_labels.data()), num_labels);
        for (size_t i = 0; i < num_labels; ++i) {
            dataset.labels[i] = static_cast<int>(raw_labels[i]);
        }

        // ─── 2. Load Images ──────────────────────────────────────────────────
        std::ifstream image_file(image_path, std::ios::binary);
        if (!image_file) throw std::runtime_error("Cannot open " + image_path);

        uint32_t num_images, rows, cols;
        image_file.read(reinterpret_cast<char*>(&magic), 4);
        image_file.read(reinterpret_cast<char*>(&num_images), 4);
        image_file.read(reinterpret_cast<char*>(&rows), 4);
        image_file.read(reinterpret_cast<char*>(&cols), 4);

        magic = swap_endian(magic);
        num_images = swap_endian(num_images);
        rows = swap_endian(rows);
        cols = swap_endian(cols);

        if (magic != 2051) throw std::runtime_error("Invalid MNIST image file magic number");
        if (num_images != num_labels) throw std::runtime_error("Mismatch between image and label counts");

        size_t image_size = rows * cols; // Should be 28x28 = 784
        dataset.images = Tensor({num_images, image_size});

        // Single massive block read for all image data
        std::vector<uint8_t> raw_images(num_images * image_size);
        image_file.read(reinterpret_cast<char*>(raw_images.data()), num_images * image_size);

        // Convert to float in [0.0, 1.0] range for neural network stability
        float* img_ptr = dataset.images.raw_ptr();
        for (size_t i = 0; i < num_images * image_size; ++i) {
            img_ptr[i] = static_cast<float>(raw_images[i]) / 255.0f;
        }

        return dataset;
    }
};

} // namespace dataset
} // namespace stakml