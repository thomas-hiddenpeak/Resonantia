/**
 * @file test_fcpe_alignment.cpp
 * @brief FCPE F0 extractor numerically aligned with torchfcpe.
 *
 * Compares the C++ FCPE (mel + fully-conv CFNaiveMelPE + cent decode) against
 * the torchfcpe reference exported by tools/export_fcpe.py. Constitution VI
 * requires F0 within 0.5 Hz on voiced frames. The reference files are produced
 * locally (models/fcpe/, tests/fixtures/fcpe_ref_*.bin); the test is skipped
 * with a clear message if they are absent.
 */
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "voxmutatio/f0/fcpe.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/io/safetensors.h"
#include "f0_checkpoint_fixture.h"
#include "f0_window_profile.h"
#include "test_artifacts.h"

namespace {

std::vector<float> load_bin1d(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {};
    int32_t nd;
    f.read(reinterpret_cast<char*>(&nd), 4);
    int64_t tot = 1;
    for (int i = 0; i < nd; ++i) { int32_t s; f.read(reinterpret_cast<char*>(&s), 4); tot *= s; }
    std::vector<float> d(tot);
    f.read(reinterpret_cast<char*>(d.data()), tot * 4);
    return d;
}

bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

}  // namespace

TEST_CASE("FCPE aligns with torchfcpe reference", "[fcpe][f0][integration]") {
    using namespace voxmutatio;

    const std::string model = "../models/fcpe/model.safetensors";
    const std::string ref_mel = "../tests/fixtures/fcpe_ref_mel.bin";
    const std::string ref_f0 = "../tests/fixtures/fcpe_ref_f0.bin";
    if (!file_exists(model) || !file_exists(ref_f0)) {
        WARN("FCPE reference missing; run tools/export_fcpe.py. Skipping.");
        return;
    }

    f0::FcpeConfig cfg;
    cfg.model_path = model;
    cfg.threshold = 0.006f;
    f0::FcpeExtractor fcpe;
    REQUIRE(fcpe.init(cfg));

    auto audio = io::read_audio("../tests/fixtures/speech_librispeech.wav", 16000);
    REQUIRE(audio.has_value());

    // --- Mel alignment ---
    if (file_exists(ref_mel)) {
        int T = 0;
        auto mel = fcpe.compute_mel(audio->data.data(),
                                    static_cast<int>(audio->data.size()), T);
        auto rmel = load_bin1d(ref_mel);
        REQUIRE(!rmel.empty());
        size_t n = std::min(mel.size(), rmel.size());
        double msum = 0, mmax = 0;
        for (size_t i = 0; i < n; ++i) {
            double d = std::abs(mel[i] - rmel[i]);
            msum += d;
            mmax = std::max(mmax, d);
        }
        std::cout << "mel frames=" << T << " mean|d|=" << (msum / n)
                  << " max|d|=" << mmax << std::endl;
        CHECK(msum / n < 0.02);
    }

    // --- F0 alignment ---
    auto f0 = fcpe.infer(audio->data.data(), static_cast<int>(audio->data.size()));
    auto rf0 = load_bin1d(ref_f0);
    REQUIRE(!rf0.empty());
    size_t n = std::min(f0.size(), rf0.size());
    REQUIRE(n > 0);

    int both_voiced = 0, uv_agree = 0;
    std::vector<double> diffs;
    for (size_t i = 0; i < n; ++i) {
        bool va = f0[i] > 0.0f, vb = rf0[i] > 0.0f;
        if (va == vb) ++uv_agree;
        if (va && vb) { ++both_voiced; diffs.push_back(std::abs(f0[i] - rf0[i])); }
    }
    std::sort(diffs.begin(), diffs.end());
    double mean = 0;
    for (double d : diffs) mean += d;
    mean = diffs.empty() ? 0 : mean / diffs.size();
    double median = diffs.empty() ? 0 : diffs[diffs.size() / 2];
    double uv_frac = static_cast<double>(uv_agree) / n;
    std::cout << "F0 frames=" << n << " both_voiced=" << both_voiced
              << " uv_agree=" << uv_frac << " meanΔ=" << mean
              << "Hz medianΔ=" << median << "Hz p90="
              << (diffs.empty() ? 0 : diffs[static_cast<size_t>(diffs.size() * 0.9)])
              << "Hz" << std::endl;

    CHECK(uv_frac > 0.95);
    CHECK(median < 0.5);   // Constitution VI: F0 < 0.5 Hz on voiced frames
    CHECK(mean < 1.0);
}

TEST_CASE("FCPE bounded-window profile preserves global-lattice centers",
          "[fcpe][f0][bounded-window][integration]") {
    using namespace voxmutatio;

    auto fixture = io::read_audio(
        "../tests/fixtures/speech_librispeech.wav", 16'000);
    REQUIRE(fixture.has_value());
    std::vector<float> tiled;
    tiled.reserve(fixture->data.size() * 3);
    for (int repeat = 0; repeat < 3; ++repeat) {
        tiled.insert(tiled.end(), fixture->data.begin(), fixture->data.end());
    }

    f0::FcpeConfig config;
    config.model_path = "../models/fcpe/model.safetensors";
    f0::FcpeExtractor extractor;
    REQUIRE(extractor.init(config));
    const auto profile = resonantia_test::profile_windowed_f0(
        extractor, tiled, static_cast<int>(
            pipeline::detail::f0_context_samples(F0Method::kFcpe)));
    INFO(profile.error);
    REQUIRE(profile.valid);
    INFO("frames=" << profile.frames << " both_voiced=" << profile.both_voiced
         << " uv=" << static_cast<double>(profile.uv_agree) / profile.frames
         << " mean=" << profile.voiced_mean_hz
         << " p95=" << profile.voiced_p95_hz
         << " global_max=" << profile.voiced_max_hz
         << " boundary_both_voiced=" << profile.boundary_both_voiced
         << " maximum_input_samples=" << profile.maximum_input_samples
         << " boundary_max=" << profile.boundary_max_hz);
    std::cout << "fcpe_bounded_profile frames=" << profile.frames
              << " both_voiced=" << profile.both_voiced
              << " uv_agreement="
              << static_cast<double>(profile.uv_agree) / profile.frames
              << " mean_hz=" << profile.voiced_mean_hz
              << " p95_hz=" << profile.voiced_p95_hz
              << " global_max_hz=" << profile.voiced_max_hz
              << " boundary_both_voiced=" << profile.boundary_both_voiced
              << " boundary_max_hz=" << profile.boundary_max_hz
              << " maximum_input_samples=" << profile.maximum_input_samples
              << std::endl;
    REQUIRE(profile.boundary_both_voiced > 0);
    CHECK(profile.maximum_input_samples <=
          pipeline::detail::kF0CoreSamples +
              2 * pipeline::detail::kFcpeContextSamples);
    CHECK(static_cast<double>(profile.uv_agree) / profile.frames > 0.95);
    CHECK(profile.voiced_mean_hz < 0.5);
    CHECK(profile.voiced_p95_hz < 0.5);
    CHECK(profile.boundary_max_hz < 0.5);
}

TEST_CASE("FCPE initialization and inference failure boundaries",
          "[fcpe][integration][contract]") {
    using namespace voxmutatio;

    f0::FcpeConfig config;
    config.model_path = "../models/fcpe/model.safetensors";
    f0::FcpeExtractor extractor;
    REQUIRE(extractor.init(config));

    float finite_sample = 0.0f;
    float nonfinite_sample = std::numeric_limits<float>::infinity();
    int frames = 23;
    CHECK(extractor.infer(nullptr, 1).empty());
    CHECK(extractor.infer(&finite_sample, 0).empty());
    CHECK(extractor.infer(&finite_sample, std::numeric_limits<int>::max()).empty());
    CHECK(extractor.infer(&nonfinite_sample, 1).empty());
    CHECK(extractor.compute_mel(&nonfinite_sample, 1, frames).empty());
    CHECK(frames == 0);

    resonantia_test::ScopedArtifact sparse_checkpoint(
        resonantia_test::unique_artifact_path("fcpe-complete-sparse", ".safetensors"));
    REQUIRE(resonantia_test::f0_fixture::create_zero_payload_sparse_clone(
        "../models/fcpe/model.safetensors", sparse_checkpoint.path()));
    REQUIRE(resonantia_test::f0_fixture::prepare_fcpe_sparse_payload(
        sparse_checkpoint.path()));
    {
        f0::FcpeExtractor manifest_probe;
        f0::FcpeConfig sparse_config;
        sparse_config.model_path = sparse_checkpoint.path().string();
        REQUIRE(manifest_probe.init(sparse_config));
    }
    REQUIRE(resonantia_test::f0_fixture::replace_header_bytes(
        sparse_checkpoint.path(),
        R"("input_stack.0.weight":{"dtype":"F32","shape":[512,128,3])",
        R"("input_stack.0.weight":{"dtype":"F32","shape":[512,384,1])"));
    {
        io::SafetensorsLoader generic_loader;
        REQUIRE(generic_loader.load(sparse_checkpoint.path().string()));
    }

    config.model_path = sparse_checkpoint.path().string();
    CHECK_FALSE(extractor.init(config));
    CHECK(extractor.infer(&finite_sample, 1).empty());
    frames = 23;
    CHECK(extractor.compute_mel(&finite_sample, 1, frames).empty());
    CHECK(frames == 0);
}
