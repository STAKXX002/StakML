#pragma once
#include <cstddef>

// ─────────────────────────────────────────────────────────────────────────────
// grad_utils.hpp — small numeric helpers shared across backward passes
//
// WHY THIS FILE EXISTS:
//   Every backward_fn_ in the codebase ends the same way: accumulate a
//   locally-computed gradient into the input's persistent grad buffer,
//   element by element (`dst[i] += src[i]`). That loop was duplicated
//   verbatim in ops.hpp (matmul), conv_ops.hpp (dW, d_input), nn.hpp
//   (Dropout, Flatten), and tensor.hpp (operator+=) — seven copies of
//   the same three lines.
//
//   Centralizing it means:
//     - one place to add OpenMP/SIMD tuning that benefits every backward
//       pass, instead of hunting down seven call sites
//     - one place to add e.g. bounds-checked/debug variants later
//
//   No dependency on Tensor — this operates on raw float* and stays
//   usable from any file, including tensor.hpp itself, with zero risk
//   of circular includes.
// ─────────────────────────────────────────────────────────────────────────────

namespace stakml {

// accumulate: dst[i] += src[i] for i in [0, n)
// The gradient-accumulation primitive used by every backward_fn_.
inline void accumulate(float* dst, const float* src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] += src[i];
}

} // namespace stakml
