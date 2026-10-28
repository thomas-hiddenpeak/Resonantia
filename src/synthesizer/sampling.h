// Copyright 2024 Resonantia Authors. Licensed under the MIT License.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace voxmutatio::synthesizer::detail {

inline constexpr std::uint64_t kPriorStream = 0x243f6a8885a308d3ULL;
inline constexpr std::uint64_t kNsfStream = 0x13198a2e03707344ULL;

inline std::uint64_t sampling_hash(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

// Stateless Box-Muller pair. The 32-bit midpoint mapping excludes 0 and 1.
// Indexing is independent of matrix shape, call order, and library RNG versions.
inline float sampling_normal(std::uint64_t seed, std::uint64_t stream,
                             std::uint64_t index) noexcept {
    const auto key = sampling_hash(seed ^ stream);
    const auto counter = (index / 2) * 2;
    const double u = (double(sampling_hash(key + counter) >> 32) + 0.5) / 4294967296.0;
    const double v = (double(sampling_hash(key + counter + 1) >> 32) + 0.5) / 4294967296.0;
    const double radius = std::sqrt(-2.0 * std::log(u));
    constexpr double tau = 6.283185307179586476925286766559;
    return static_cast<float>(radius * ((index & 1) ? std::sin(tau * v) : std::cos(tau * v)));
}

inline bool valid_sampling_range(std::uint64_t frame_offset, int frames) noexcept {
    constexpr auto limit = std::numeric_limits<std::uint64_t>::max() / 400;
    return frames > 0 && frame_offset <= limit &&
           static_cast<std::uint64_t>(frames) <= limit - frame_offset;
}

inline bool sample_prior(std::vector<float>& mean, const std::vector<float>& logs,
                         int frames, std::uint64_t seed, std::uint64_t frame_offset = 0) {
    if (!valid_sampling_range(frame_offset, frames) ||
        mean.size() != static_cast<std::size_t>(frames) * 192 ||
        logs.size() != mean.size()) return false;
    for (int channel = 0; channel < 192; ++channel) {
        for (int frame = 0; frame < frames; ++frame) {
            const auto offset = static_cast<std::size_t>(channel) * frames + frame;
            if (!std::isfinite(mean[offset]) || !std::isfinite(logs[offset])) return false;
            const float draw = sampling_normal(seed, kPriorStream,
                (frame_offset + static_cast<std::uint64_t>(frame)) * 192 + channel);
            mean[offset] += std::exp(logs[offset]) * draw * 0.66666f;
            if (!std::isfinite(mean[offset])) return false;
        }
    }
    return true;
}

}  // namespace voxmutatio::synthesizer::detail
