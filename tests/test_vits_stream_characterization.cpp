#include <catch2/catch_test_macros.hpp>

#include "voxmutatio/synthesizer/synthesizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

voxmutatio::synthesizer::Synthesizer make_synthesizer() {
    voxmutatio::synthesizer::SynthesizerConfig config;
    config.model_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    config.version = voxmutatio::ModelVersion::kV2;

    voxmutatio::synthesizer::Synthesizer synthesizer;
    REQUIRE(synthesizer.init(config));
    REQUIRE(synthesizer.sample_rate() == 40'000);
    REQUIRE(synthesizer.feature_dim() == 768);
    return synthesizer;
}

double max_abs_difference(const std::vector<float>& lhs,
                          const std::vector<float>& rhs) {
    REQUIRE(lhs.size() == rhs.size());
    double maximum = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        maximum = std::max(maximum,
                           std::abs(static_cast<double>(lhs[i]) -
                                    static_cast<double>(rhs[i])));
    }
    return maximum;
}

}  // namespace

TEST_CASE("Legacy infer_stream is full infer followed by a 400-sample frame crop",
          "[realtime][characterization][vits]") {
    auto synthesizer = make_synthesizer();
    constexpr int kFrames = 4;
    constexpr int kSamplesPerFrame = 400;

    std::vector<float> features(kFrames * synthesizer.feature_dim());
    for (std::size_t i = 0; i < features.size(); ++i) {
        features[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.001f;
    }
    const std::vector<float> pitch = {100.0f, 110.0f, 120.0f, 130.0f};
    const std::vector<float> f0 = {110.0f, 146.0f, 196.0f, 246.0f};

    const auto full = synthesizer.infer(features.data(), kFrames, pitch.data(), f0.data(), 0);
    REQUIRE(full.sample_rate == 40'000);
    REQUIRE(full.data.size() == kFrames * kSamplesPerFrame);

    const auto middle = synthesizer.infer_stream(
        features.data(), kFrames, pitch.data(), f0.data(), 0, 1, 2);
    REQUIRE(middle.sample_rate == full.sample_rate);
    REQUIRE(middle.data.size() == 2 * kSamplesPerFrame);
    const std::vector<float> expected(
        full.data.begin() + kSamplesPerFrame,
        full.data.begin() + 3 * kSamplesPerFrame);
    CHECK(max_abs_difference(middle.data, expected) == 0.0);

    // This makes an off-by-one-frame crop observably different instead of only
    // asserting the output length.
    const std::vector<float> wrong_crop(
        full.data.begin(), full.data.begin() + 2 * kSamplesPerFrame);
    CHECK(max_abs_difference(middle.data, wrong_crop) > 1e-7);
}

TEST_CASE("Legacy infer_stream rejects every out-of-range window",
          "[realtime][characterization][vits]") {
    auto synthesizer = make_synthesizer();
    constexpr int kFrames = 2;
    std::vector<float> features(kFrames * synthesizer.feature_dim(), 0.0f);
    std::vector<float> pitch(kFrames, 100.0f);
    std::vector<float> f0(kFrames, 220.0f);

    const auto invoke = [&](int skip_head, int return_length) {
        return synthesizer.infer_stream(features.data(), kFrames, pitch.data(), f0.data(),
                                        0, skip_head, return_length);
    };
    CHECK(invoke(-1, 1).data.empty());
    CHECK(invoke(0, -1).data.empty());
    CHECK(invoke(0, 0).data.empty());
    CHECK(invoke(kFrames + 1, 1).data.empty());
    CHECK(invoke(kFrames, 1).data.empty());
    CHECK(invoke(1, 2).data.empty());
    CHECK(invoke(std::numeric_limits<int>::max(), 1).data.empty());
    CHECK(invoke(0, std::numeric_limits<int>::max()).data.empty());
}

TEST_CASE("Repeated legacy calls reset all call-local output and NSF phase",
          "[realtime][characterization][vits]") {
    auto synthesizer = make_synthesizer();
    std::vector<float> features(synthesizer.feature_dim(), 0.0f);
    const float pitch = 100.0f;
    const float f0 = 223.0f;

    const auto first = synthesizer.infer_stream(
        features.data(), 1, &pitch, &f0, 0, 0, 1);
    const auto repeated = synthesizer.infer_stream(
        features.data(), 1, &pitch, &f0, 0, 0, 1);
    REQUIRE(first.data.size() == 400);
    REQUIRE(repeated.data.size() == 400);
    CHECK(max_abs_difference(first.data, repeated.data) == 0.0);

    const auto one_frame_phase = synthesizer.debug_har(&f0, 1);
    const std::vector<float> two_frame_f0 = {f0, f0};
    const auto continuous_phase = synthesizer.debug_har(two_frame_f0.data(), 2);
    REQUIRE(one_frame_phase.size() == 400);
    REQUIRE(continuous_phase.size() == 800);
    const std::vector<float> continuous_second(
        continuous_phase.begin() + 400, continuous_phase.end());
    const auto restarted_phase = synthesizer.debug_har(&f0, 1);
    CHECK(max_abs_difference(one_frame_phase, restarted_phase) == 0.0);
    CHECK(max_abs_difference(continuous_second, restarted_phase) > 1e-7);
}
