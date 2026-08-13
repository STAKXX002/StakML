#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// conv.hpp — compatibility shim
//
// The contents of this file were split for maintainability as the codebase
// grew:
//   - im2col, col2im, ops::conv2d_forward   → moved to conv_ops.hpp
//   - nn::Conv2d, nn::MaxPool2d, nn::Flatten → moved to nn.hpp
//     (they're Module subclasses, same category as Linear/Dropout/Sequential)
//
// This file is kept so any existing #include "conv.hpp" keeps working
// unchanged. New code should prefer including conv_ops.hpp / nn.hpp
// directly for what it actually needs.
// ─────────────────────────────────────────────────────────────────────────────

#include "conv_ops.hpp"
#include "nn.hpp"