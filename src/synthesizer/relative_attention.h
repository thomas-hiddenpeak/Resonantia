#pragma once

#include <vector>

namespace voxmutatio::synthesizer::ops {

// Private synchronous VITS attention primitive. q/k/v and output are
// [heads * head_dim, frames]. Shared relative embeddings are
// [2 * window + 1, head_dim]; positions outside the window contribute zero.
// Keeps one frames x frames device score matrix, reused across heads.
std::vector<float> relative_attention(const float* q, const float* k, const float* v,
                                     const float* relative_k, const float* relative_v,
                                     int frames, int heads, int head_dim, int window);

}  // namespace voxmutatio::synthesizer::ops
