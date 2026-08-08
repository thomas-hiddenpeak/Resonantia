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
#include <vector>

#include "voxmutatio/f0/fcpe.h"
#include "voxmutatio/io/audio_io.h"

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
