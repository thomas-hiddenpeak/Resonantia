#pragma once

#include "pipeline/bounded_pipeline_detail.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace resonantia_test {

struct F0WindowProfile {
    bool valid = false;
    std::string error;
    std::size_t frames = 0;
    std::size_t both_voiced = 0;
    std::size_t uv_agree = 0;
    std::size_t boundary_frames = 0;
    std::size_t boundary_uv_agree = 0;
    std::size_t boundary_both_voiced = 0;
    std::uint64_t maximum_input_samples = 0;
    double voiced_mean_hz = 0.0;
    double voiced_p95_hz = 0.0;
    double voiced_max_hz = 0.0;
    double boundary_max_hz = 0.0;
};

template <typename Extractor>
F0WindowProfile profile_windowed_f0(Extractor& extractor,
                                    const std::vector<float>& audio,
                                    int context_samples) {
    constexpr int kHop = static_cast<int>(
        voxmutatio::pipeline::detail::kPitchHop);
    constexpr int kCoreFrames = static_cast<int>(
        voxmutatio::pipeline::detail::kF0CoreSamples /
        voxmutatio::pipeline::detail::kPitchHop);
    constexpr int kBoundaryRadius = 10;
    F0WindowProfile profile;
    if (audio.empty() || context_samples < 0 || context_samples % kHop != 0 ||
        audio.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        profile.error = "invalid profile input";
        return profile;
    }

    const auto full = extractor.infer(audio.data(), static_cast<int>(audio.size()));
    const int total_frames = static_cast<int>(audio.size() / kHop) + 1;
    if (full.size() != static_cast<std::size_t>(total_frames)) {
        profile.error = "unexpected full F0 frame count";
        return profile;
    }
    std::vector<float> windowed(full.size(), 0.0f);
    for (std::uint64_t first_frame = 0;
         first_frame < static_cast<std::uint64_t>(total_frames);) {
        voxmutatio::pipeline::detail::F0WindowPlan plan;
        if (!voxmutatio::pipeline::detail::plan_f0_window(
                audio.size(), first_frame, context_samples, plan)) {
            profile.error = "production F0 window planner rejected the profile";
            return profile;
        }
        profile.maximum_input_samples = std::max(
            profile.maximum_input_samples, plan.input_samples);
        const auto local = extractor.infer(
            audio.data() + static_cast<std::size_t>(plan.input_start),
            static_cast<int>(plan.input_samples));
        if (local.size() != plan.expected_local_frames) {
            profile.error = "profile F0 frame/crop mismatch";
            return profile;
        }
        std::copy(
            local.begin() + static_cast<std::size_t>(plan.crop_first_frame),
            local.begin() + static_cast<std::size_t>(
                plan.crop_first_frame + plan.crop_frame_count),
            windowed.begin() + static_cast<std::size_t>(plan.first_frame));
        first_frame = plan.end_frame;
    }

    std::vector<double> voiced_errors;
    for (int frame = 0; frame < total_frames; ++frame) {
        const bool full_uv = full[static_cast<std::size_t>(frame)] <= 0.0f;
        const bool window_uv = windowed[static_cast<std::size_t>(frame)] <= 0.0f;
        if (full_uv == window_uv) ++profile.uv_agree;
        if (!full_uv && !window_uv) {
            ++profile.both_voiced;
            const double error = std::abs(
                static_cast<double>(full[static_cast<std::size_t>(frame)]) -
                static_cast<double>(windowed[static_cast<std::size_t>(frame)]));
            voiced_errors.push_back(error);
            profile.voiced_mean_hz += error;
            profile.voiced_max_hz = std::max(profile.voiced_max_hz, error);
        }

        bool boundary = false;
        for (int split = kCoreFrames; split < total_frames;
             split += kCoreFrames) {
            if (std::abs(frame - split) <= kBoundaryRadius) {
                boundary = true;
                break;
            }
        }
        if (boundary) {
            ++profile.boundary_frames;
            if (full_uv == window_uv) ++profile.boundary_uv_agree;
            if (!full_uv && !window_uv) {
                ++profile.boundary_both_voiced;
                profile.boundary_max_hz = std::max(
                    profile.boundary_max_hz,
                    std::abs(static_cast<double>(
                                 full[static_cast<std::size_t>(frame)]) -
                             static_cast<double>(
                                 windowed[static_cast<std::size_t>(frame)])));
            }
        }
    }
    profile.frames = full.size();
    if (voiced_errors.empty() || profile.boundary_frames == 0 ||
        profile.boundary_both_voiced == 0) {
        profile.error =
            "profile lacks voiced internal-boundary comparison evidence";
        return profile;
    }
    profile.voiced_mean_hz /= voiced_errors.size();
    std::sort(voiced_errors.begin(), voiced_errors.end());
    const std::size_t p95_index = static_cast<std::size_t>(
        std::floor(0.95 * static_cast<double>(voiced_errors.size() - 1)));
    profile.voiced_p95_hz = voiced_errors[p95_index];
    profile.valid = true;
    return profile;
}

}  // namespace resonantia_test
