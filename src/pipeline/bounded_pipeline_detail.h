// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#pragma once

#include "../io/bounded_audio_io.h"
#include "voxmutatio/core/types.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>

namespace voxmutatio::pipeline::detail {

inline constexpr std::uint64_t kPitchHop = 160;
inline constexpr std::uint64_t kF0CoreSamples = 8 * 16'000;
inline constexpr std::uint64_t kRmvpeContextSamples = 4 * 16'000;
inline constexpr std::uint64_t kFcpeContextSamples = 2 * 16'000;

struct F0WindowPlan {
    std::uint64_t first_frame = 0;
    std::uint64_t end_frame = 0;
    std::uint64_t input_start = 0;
    std::uint64_t input_samples = 0;
    std::uint64_t crop_first_frame = 0;
    std::uint64_t crop_frame_count = 0;
    std::uint64_t expected_local_frames = 0;
};

[[nodiscard]] inline constexpr std::uint64_t f0_context_samples(
        F0Method method) noexcept {
    return method == F0Method::kFcpe
        ? kFcpeContextSamples : kRmvpeContextSamples;
}

// Plan one raw-F0 call on the global 160-sample lattice. Only the true clip
// edges are allowed to omit profile context; artificial window edges are
// cropped before any interpolation, median filtering, or pitch shift.
[[nodiscard]] inline bool plan_f0_window(
        std::uint64_t total_samples, std::uint64_t first_frame,
        std::uint64_t context_samples, F0WindowPlan& plan) noexcept {
    plan = {};
    if (total_samples == 0 || context_samples % kPitchHop != 0) return false;
    const std::uint64_t total_frames = total_samples / kPitchHop + 1;
    if (first_frame >= total_frames) return false;
    constexpr std::uint64_t frames_per_core = kF0CoreSamples / kPitchHop;
    const std::uint64_t remaining_frames = total_frames - first_frame;
    const std::uint64_t frame_count = std::min(frames_per_core, remaining_frames);
    const std::uint64_t end_frame = first_frame + frame_count;
    if (first_frame > std::numeric_limits<std::uint64_t>::max() / kPitchHop) {
        return false;
    }
    const std::uint64_t core_start = first_frame * kPitchHop;
    const std::uint64_t core_end = end_frame < total_frames
        ? end_frame * kPitchHop : total_samples;
    const std::uint64_t input_start = core_start > context_samples
        ? core_start - context_samples : 0;
    const std::uint64_t available_right = total_samples - core_end;
    const std::uint64_t input_end = core_end +
        std::min(context_samples, available_right);
    if (input_start % kPitchHop != 0 || input_end <= input_start) return false;

    const std::uint64_t input_samples = input_end - input_start;
    const std::uint64_t expected_local_frames = input_samples / kPitchHop + 1;
    const std::uint64_t crop_first = first_frame - input_start / kPitchHop;
    if (crop_first > expected_local_frames ||
        frame_count > expected_local_frames - crop_first ||
        input_samples > kF0CoreSamples + 2 * context_samples) {
        return false;
    }
    plan.first_frame = first_frame;
    plan.end_frame = end_frame;
    plan.input_start = input_start;
    plan.input_samples = input_samples;
    plan.crop_first_frame = crop_first;
    plan.crop_frame_count = frame_count;
    plan.expected_local_frames = expected_local_frames;
    return true;
}

// Private deterministic seams used by production and low-level numerical
// contracts. They intentionally stay outside include/voxmutatio and do not
// expand the supported public API.
bool resample_wav_to_stable_snapshot(
    io::detail::WavChunkReader& reader,
    const std::filesystem::path& spool_directory,
    io::detail::RawFloatSpool& snapshot,
    std::uint64_t expected_samples,
    std::string& error);

bool filter_stable_source(
    io::detail::RawFloatSpool& source,
    const std::filesystem::path& spool_directory,
    bool apply_highpass,
    io::detail::RawFloatSpool& filtered,
    std::string& error);

bool process_f0(
    io::detail::RawFloatSpool& raw,
    const std::filesystem::path& spool_directory,
    const VCConfig& config,
    io::detail::RawFloatSpool& processed,
    std::string& error, double pad_seconds = 0, double duration_seconds = 0);

bool plan_first_quiet_core(
    io::detail::RawFloatSpool& source,
    std::uint64_t source_samples,
    std::uint64_t chunk_samples,
    std::uint64_t query_samples,
    std::uint64_t hard_core_samples,
    std::uint64_t& core_start,
    std::uint64_t& core_end,
    bool& done,
    std::string& error);

bool write_global_rms_output(
    io::detail::RawFloatSpool& filtered_source,
    io::detail::RawFloatSpool& raw_output,
    const std::filesystem::path& spool_directory,
    double rms_mix_rate,
    io::detail::AtomicWavStreamWriter& writer,
    std::string& error);

}  // namespace voxmutatio::pipeline::detail
