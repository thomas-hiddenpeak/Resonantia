/**
 * @file test_e2e_conversion.cpp
 * @brief End-to-end voice conversion integration test.
 *
 * Runs the complete C++ pipeline (HuBERT -> RMVPE -> VITS) on real speech
 * audio and compares the output against the Python RVC reference waveform.
 * This is the highest-priority integration test (Constitution Principle IX).
 */
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "voxmutatio/pipeline/pipeline.h"
#include "voxmutatio/io/audio_io.h"

namespace {

std::vector<float> load_bin1d(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {};
    int32_t nd; f.read(reinterpret_cast<char*>(&nd), 4);
    int64_t tot = 1;
    for (int i = 0; i < nd; ++i) { int32_t s; f.read(reinterpret_cast<char*>(&s), 4); tot *= s; }
    std::vector<float> d(tot);
    f.read(reinterpret_cast<char*>(d.data()), tot * 4);
    return d;
}

double correlation(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = std::min(a.size(), b.size());
    if (n == 0) return 0;
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double c = 0, va = 0, vb = 0;
    for (size_t i = 0; i < n; ++i) {
        double da = a[i] - ma, db = b[i] - mb;
        c += da * db; va += da * da; vb += db * db;
    }
    return c / (std::sqrt(va * vb) + 1e-12);
}

}  // namespace

TEST_CASE("End-to-end voice conversion (real audio)", "[e2e][integration]") {
    using namespace voxmutatio;

    VCConfig cfg;
    cfg.hubert_model_path = "../models/hubert_base/model.safetensors";
    cfg.rmvpe_model_path = "../models/rmvpe.safetensors";
    cfg.synthesizer_model_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    cfg.version = ModelVersion::kV2;
    cfg.has_f0 = true;
    cfg.model_sample_rate = 40000;
    cfg.num_speakers = 109;
    cfg.rms_mix_rate = 1.0;  // no RMS mixing (match reference)
    cfg.f0_up_key = 0;

    pipeline::VoiceConversionPipeline pipe;
    REQUIRE(pipe.init(cfg));

    // Load real speech audio
    auto audio = io::read_audio("../tests/fixtures/speech_librispeech.wav", 16000);
    REQUIRE(audio.has_value());
    std::cout << "Input: " << audio->data.size() << " samples @ 16kHz" << std::endl;

    // Run full conversion
    auto result = pipe.convert_buffer(*audio, /*speaker_id=*/0);
    REQUIRE(result.success);
    std::cout << "Output: " << result.audio.data.size() << " samples @ "
              << result.audio.sample_rate << " Hz" << std::endl;
    std::cout << "Timing: HuBERT " << result.hubert_ms << "ms, F0 "
              << result.f0_ms << "ms, Synth " << result.synth_ms
              << "ms, Total " << result.total_ms << "ms" << std::endl;

    // Output should be valid audio
    REQUIRE(!result.audio.data.empty());
    REQUIRE(result.audio.sample_rate == 40000);

    // No NaN/Inf
    for (float v : result.audio.data) {
        REQUIRE(!std::isnan(v));
        REQUIRE(!std::isinf(v));
    }

    // Compare against Python RVC reference waveform
    auto ref = load_bin1d("../tests/fixtures/vits_ref_audio.bin");
    REQUIRE(!ref.empty());
    std::cout << "Reference: " << ref.size() << " samples" << std::endl;

    double corr = correlation(result.audio.data, ref);
    std::cout << "End-to-end waveform correlation: " << corr << std::endl;

    // Write the output for manual inspection
    io::write_audio("../tests/fixtures/e2e_cpp_output.wav",
                    result.audio.data.data(),
                    static_cast<int>(result.audio.data.size()),
                    result.audio.sample_rate, "wav");

    // The full C++ chain should closely match the Python reference.
    CHECK(corr > 0.99);
}
