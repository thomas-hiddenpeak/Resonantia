// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/catch_approx.hpp>

#include "voxmutatio/f0/rmvpe.h"
#include "f0/rmvpe_frontend.h"
#include "voxmutatio/f0/fcpe.h"
#include "test_artifacts.h"

#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

static_assert(std::is_copy_constructible_v<voxmutatio::f0::RmvpeExtractor>);
static_assert(std::is_copy_assignable_v<voxmutatio::f0::RmvpeExtractor>);

namespace {

void append_u64_le(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
    }
}

bool write_fcpe_head_only_checkpoint(const std::filesystem::path& path) {
    const std::string header =
        R"({"cent_table":{"dtype":"F32","shape":[360],"data_offsets":[0,1440]},)"
        R"("output_proj.bias":{"dtype":"F32","shape":[360],"data_offsets":[1440,2880]},)"
        R"("output_proj.parametrizations.weight.original0":{"dtype":"F32","shape":[360,1],"data_offsets":[2880,4320]},)"
        R"("output_proj.parametrizations.weight.original1":{"dtype":"F32","shape":[360,512],"data_offsets":[4320,741600]}})";

    std::vector<std::uint8_t> bytes;
    bytes.reserve(sizeof(std::uint64_t) + header.size() + 741'600);
    append_u64_le(bytes, header.size());
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.resize(bytes.size() + 741'600, 0);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return false;
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(output);
}

}  // namespace

TEST_CASE("Pitch shift calculation", "[f0][pitch-shift]") {
    std::vector<float> f0 = {100.0f, 200.0f, 0.0f, 300.0f};  // 0.0 = unvoiced
    
    // +12 semitones = octave up (2x frequency)
    auto shifted_up = voxmutatio::f0::RmvpeExtractor::pitch_shift(f0, 12);
    
    REQUIRE(shifted_up[0] == Catch::Approx(200.0f).epsilon(1e-4));
    REQUIRE(shifted_up[1] == Catch::Approx(400.0f).epsilon(1e-4));
    REQUIRE(shifted_up[2] == Catch::Approx(0.0f).epsilon(1e-4));  // Unvoiced stays 0
    REQUIRE(shifted_up[3] == Catch::Approx(600.0f).epsilon(1e-4));
    
    // -12 semitones = octave down (0.5x frequency)
    auto shifted_down = voxmutatio::f0::RmvpeExtractor::pitch_shift(f0, -12);
    
    REQUIRE(shifted_down[0] == Catch::Approx(50.0f).epsilon(1e-4));
    REQUIRE(shifted_down[1] == Catch::Approx(100.0f).epsilon(1e-4));
    REQUIRE(shifted_down[2] == Catch::Approx(0.0f).epsilon(1e-4));
    REQUIRE(shifted_down[3] == Catch::Approx(150.0f).epsilon(1e-4));
}

TEST_CASE("Pitch shift zero semitones", "[f0][pitch-shift]") {
    std::vector<float> f0 = {100.0f, 200.0f, 300.0f};
    
    auto shifted = voxmutatio::f0::RmvpeExtractor::pitch_shift(f0, 0);
    
    for (size_t i = 0; i < f0.size(); ++i) {
        REQUIRE(shifted[i] == Catch::Approx(f0[i]).epsilon(1e-6));
    }
}

TEST_CASE("Pitch shift +7 semitones (perfect fifth)", "[f0][pitch-shift]") {
    std::vector<float> f0 = {100.0f};
    
    auto shifted = voxmutatio::f0::RmvpeExtractor::pitch_shift(f0, 7);
    
    // 100 * 2^(7/12) ≈ 149.83 Hz
    REQUIRE(shifted[0] == Catch::Approx(149.83f).epsilon(1e-2));
}

TEST_CASE("FCPE pitch shift matches RMVPE", "[f0][pitch-shift]") {
    std::vector<float> f0 = {100.0f, 150.0f, 200.0f, 0.0f, 250.0f};
    
    auto rmvpe_shifted = voxmutatio::f0::RmvpeExtractor::pitch_shift(f0, 5);
    auto fcpe_shifted = voxmutatio::f0::FcpeExtractor::pitch_shift(f0, 5);
    
    REQUIRE(rmvpe_shifted.size() == fcpe_shifted.size());
    
    for (size_t i = 0; i < f0.size(); ++i) {
        REQUIRE(rmvpe_shifted[i] == Catch::Approx(fcpe_shifted[i]).epsilon(1e-6));
    }
}

TEST_CASE("Pitch shift rejects non-finite input, unsupported range, and overflow",
          "[f0][pitch-shift][contract]") {
    using voxmutatio::f0::FcpeExtractor;
    using voxmutatio::f0::RmvpeExtractor;

    CHECK(RmvpeExtractor::pitch_shift(
              {std::numeric_limits<float>::quiet_NaN()}, 0).empty());
    CHECK(FcpeExtractor::pitch_shift(
              {std::numeric_limits<float>::infinity()}, 0).empty());

    CHECK(RmvpeExtractor::pitch_shift({100.0f}, 49).empty());
    CHECK(FcpeExtractor::pitch_shift({100.0f}, -49).empty());

    const auto rmvpe_boundary = RmvpeExtractor::pitch_shift({100.0f}, 48);
    REQUIRE(rmvpe_boundary.size() == 1);
    CHECK(std::isfinite(rmvpe_boundary.front()));
    const auto fcpe_boundary = FcpeExtractor::pitch_shift({100.0f}, -48);
    REQUIRE(fcpe_boundary.size() == 1);
    CHECK(std::isfinite(fcpe_boundary.front()));

    CHECK(RmvpeExtractor::pitch_shift(
              {std::numeric_limits<float>::max()}, 48).empty());
    CHECK(FcpeExtractor::pitch_shift(
              {std::numeric_limits<float>::max()}, 48).empty());
}

TEST_CASE("F0 extractors reject missing and incomplete checkpoints",
          "[f0][contract][init]") {
    using voxmutatio::f0::FcpeConfig;
    using voxmutatio::f0::FcpeExtractor;
    using voxmutatio::f0::RmvpeConfig;
    using voxmutatio::f0::RmvpeExtractor;

    RmvpeExtractor missing_rmvpe;
    RmvpeConfig missing_rmvpe_config;
    missing_rmvpe_config.model_path = "/tmp/resonantia-missing-rmvpe.safetensors";
    CHECK_FALSE(missing_rmvpe.init(missing_rmvpe_config));

    resonantia_test::ScopedArtifact incomplete(
        resonantia_test::unique_artifact_path("fcpe-head-only", ".safetensors"));
    REQUIRE(write_fcpe_head_only_checkpoint(incomplete.path()));

    FcpeExtractor fcpe;
    FcpeConfig fcpe_config;
    fcpe_config.model_path = incomplete.path().string();
    CHECK_FALSE(fcpe.init(fcpe_config));

    RmvpeExtractor rmvpe;
    RmvpeConfig rmvpe_config;
    rmvpe_config.model_path = incomplete.path().string();
    CHECK_FALSE(rmvpe.init(rmvpe_config));
}

TEST_CASE("F0 extractors reject invalid configuration and public input",
          "[f0][contract][boundary]") {
    using voxmutatio::f0::FcpeConfig;
    using voxmutatio::f0::FcpeExtractor;
    using voxmutatio::f0::RmvpeConfig;
    using voxmutatio::f0::RmvpeExtractor;

    RmvpeConfig rmvpe_config;
    rmvpe_config.model_path = "/tmp/resonantia-missing-rmvpe.safetensors";
    rmvpe_config.threshold = std::numeric_limits<float>::quiet_NaN();
    RmvpeExtractor rmvpe;
    CHECK_FALSE(rmvpe.init(rmvpe_config));

    FcpeConfig fcpe_config;
    fcpe_config.model_path = "/tmp/resonantia-missing-fcpe.safetensors";
    fcpe_config.threshold = std::numeric_limits<float>::infinity();
    FcpeExtractor fcpe;
    CHECK_FALSE(fcpe.init(fcpe_config));

    int frames = 17;
    CHECK(rmvpe.infer(nullptr, 0).empty());
    CHECK(rmvpe.infer_salience(nullptr, 0, frames).empty());
    CHECK(frames == 0);

    frames = 17;
    CHECK(fcpe.infer(nullptr, 0).empty());
    CHECK(fcpe.compute_mel(nullptr, 0, frames).empty());
    CHECK(frames == 0);
}

TEST_CASE("RMVPE frontend uses HTK spacing with Slaney area normalization",
          "[f0][frontend][contract]") {
    const auto basis = voxmutatio::f0::detail::rmvpe_mel_basis();
    REQUIRE(basis.size() == 128 * 513);
    // Independent goldens from upstream RVC/librosa 0.11.0 (htk=True).
    // Bands span the spectrum; normalization must not become peak-normalization.
    struct Golden { int band; int bin; float weight; };
    const Golden expected[] = {
        {0, 3, 0.0567850396f}, {1, 4, 0.0503917262f},
        {15, 19, 0.0409837179f}, {32, 43, 0.0318208821f},
        {64, 118, 0.0201660637f}, {100, 280, 0.00963796210f},
        {127, 501, 0.00586089073f},
    };
    for (const auto& value : expected) {
        CAPTURE(value.band, value.bin);
        CHECK(basis[value.band * 513 + value.bin] ==
              Catch::Approx(value.weight).margin(1e-8));
    }
    for (int band = 0; band < 128; ++band) {
        double sum = 0.0;
        for (int bin = 0; bin < 513; ++bin) {
            const float value = basis[band * 513 + bin];
            REQUIRE(std::isfinite(value));
            REQUIRE(value >= 0.0f);
            sum += value;
        }
        CHECK(sum > 0.05);
        CHECK(sum < 0.08);
        CHECK(basis[band * 513] == 0.0f);
        // Inverting the HTK endpoint can land a few ulps above 8000 Hz.
        CHECK(basis[band * 513 + 512] == Catch::Approx(0.0f).margin(1e-12));
    }
}
