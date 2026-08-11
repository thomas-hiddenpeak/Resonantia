/**
 * @file test_wavlm_alignment.cpp
 * @brief WavLM-Base+ content encoder aligned with HuggingFace transformers.
 *
 * Compares the C++/CUDA WavLM encoder (wav2vec2 front-end + gated relative
 * position bias) against microsoft/wavlm-base-plus's last hidden state exported
 * by tools/export_wavlm.py. Constitution VI requires content features within a
 * tight L2 tolerance. Reference files are produced locally (models/wavlm/,
 * tests/fixtures/wavlm_ref_*.bin); the test is skipped if they are absent.
 */
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "voxmutatio/content/wavlm_encoder.h"
#include "voxmutatio/io/audio_io.h"

namespace {

std::vector<float> load_bin(const std::string& path) {
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

TEST_CASE("WavLM-Base+ aligns with transformers reference", "[wavlm][content][integration]") {
    using namespace voxmutatio;

    const std::string model = "../models/wavlm/model.safetensors";
    const std::string ref = "../tests/fixtures/wavlm_ref_last.bin";
    if (!file_exists(model) || !file_exists(ref)) {
        WARN("WavLM reference missing; run tools/export_wavlm.py. Skipping.");
        return;
    }

    content::WavlmConfig cfg;
    cfg.model_path = model;
    cfg.output_dim = 768;
    cfg.num_layers = 12;
    content::WavlmEncoder enc;
    REQUIRE(enc.init(cfg));

    auto audio = io::read_audio("../tests/fixtures/speech_librispeech.wav", 16000);
    REQUIRE(audio.has_value());

    auto feats = enc.extract(audio->data.data(), static_cast<int>(audio->data.size()));
    REQUIRE(!feats.empty());
    auto rf = load_bin(ref);
    REQUIRE(!rf.empty());

    size_t n = std::min(feats.size(), rf.size());
    REQUIRE(n > 0);
    for (float v : feats) { REQUIRE(!std::isnan(v)); REQUIRE(!std::isinf(v)); }

    // Per-element error + cosine similarity (phase-free content match).
    double sabs = 0, smax = 0, dot = 0, na = 0, nb = 0, sse = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = static_cast<double>(feats[i]) - rf[i];
        sabs += std::abs(d);
        sse += d * d;
        smax = std::max(smax, std::abs(d));
        dot += static_cast<double>(feats[i]) * rf[i];
        na += static_cast<double>(feats[i]) * feats[i];
        nb += static_cast<double>(rf[i]) * rf[i];
    }
    double cos = dot / (std::sqrt(na * nb) + 1e-12);
    double rmse = std::sqrt(sse / n);
    std::cout << "WavLM frames*dim=" << n << " mean|d|=" << (sabs / n)
              << " max|d|=" << smax << " rmse=" << rmse << " cos=" << cos << std::endl;

    CHECK(cos > 0.999);
    CHECK(sabs / n < 1e-3);
    CHECK(smax < 5e-2);

    auto invalid = cfg;
    invalid.model_path.clear();
    CHECK_FALSE(enc.init(invalid));
    CHECK_FALSE(enc.initialized());
    CHECK(enc.extract(audio->data.data(), static_cast<int>(audio->data.size())).empty());
}
