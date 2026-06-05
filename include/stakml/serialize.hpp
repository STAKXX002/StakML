#pragma once
#include "nn.hpp"
#include <string>
#include <fstream>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// serialize.hpp — Model Saving and Loading
// ─────────────────────────────────────────────────────────────────────────────

namespace stakml {
namespace serialize {

inline void save_model(const nn::Sequential& model, const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for saving: " + filepath);
    }

    auto params = model.parameters();
    for (const auto& p : params) {
        // Write the raw float data of each parameter directly to disk
        out.write(reinterpret_cast<const char*>(p->raw_ptr()), p->num_elements() * sizeof(float));
    }
}

inline void load_model(nn::Sequential& model, const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for loading: " + filepath);
    }

    auto params = model.parameters();
    for (auto& p : params) {
        // Read the raw float data directly back into the parameter's memory buffer
        in.read(reinterpret_cast<char*>(p->raw_ptr()), p->num_elements() * sizeof(float));
    }
}

} // namespace serialize
} // namespace stakml