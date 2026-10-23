/**
 * @file test_wavlm_alignment.cpp
 * @brief WavLM-Base+ content encoder aligned with HuggingFace transformers.
 *
 * Compares the C++/CUDA WavLM encoder (wav2vec2 front-end + gated relative
 * position bias) against microsoft/wavlm-base-plus's last hidden state exported
 * by tools/export_wavlm.py. Constitution VI requires content features within a
 * tight L2 tolerance. Reference files are produced locally (models/wavlm/,
 * tests/fixtures/wavlm_ref_*.bin); CMake requires these assets for registration.
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

std::vector<float> load_bin(const std::string& path, std::size_t elements, int dimension) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.is_open());
    int32_t nd = 0;
    f.read(reinterpret_cast<char*>(&nd), 4);
    REQUIRE(nd == 2);
    int32_t shape[2] = {};
    f.read(reinterpret_cast<char*>(shape), sizeof(shape));
    REQUIRE(f.good());
    REQUIRE(shape[0] > 0);
    REQUIRE(shape[1] == dimension);
    REQUIRE(static_cast<std::uint64_t>(shape[0]) * shape[1] == elements);
    std::vector<float> d(elements);
    f.read(reinterpret_cast<char*>(d.data()), elements * sizeof(float));
    REQUIRE(f.good());
    REQUIRE(f.peek() == std::char_traits<char>::eof());
    return d;
}

}  // namespace

TEST_CASE("WavLM-Base+ aligns with transformers reference", "[wavlm][content][integration]") {
    using namespace voxmutatio;

    const std::string model = "../models/wavlm/model.safetensors";
    const std::string ref = "../tests/fixtures/wavlm_ref_last.bin";

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
    REQUIRE(feats.size() % cfg.output_dim == 0);
    auto rf = load_bin(ref, feats.size(), cfg.output_dim);
    REQUIRE(!rf.empty());

    REQUIRE(feats.size() == rf.size());
    const size_t n = feats.size();
    REQUIRE(n > 0);
    for (float v : feats) { REQUIRE(!std::isnan(v)); REQUIRE(!std::isinf(v)); }
    for (float v : rf) REQUIRE(std::isfinite(v));

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
    CHECK(rmse < 1e-4);  // Constitution VI: normalized L2 error over all elements.
    CHECK(sabs / n < 1e-3);
    CHECK(smax < 5e-2);

    auto invalid = cfg;
    invalid.model_path.clear();
    CHECK_FALSE(enc.init(invalid));
    CHECK_FALSE(enc.initialized());
    CHECK(enc.extract(audio->data.data(), static_cast<int>(audio->data.size())).empty());
}
