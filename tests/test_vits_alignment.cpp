/**
 * @file test_vits_alignment.cpp
 * @brief VITS synthesizer numerical alignment vs deterministic Python reference.
 *
 * Stage-by-stage: TextEncoder (m_p) -> Flow (z) -> Generator (audio).
 * SC-003 relates to final audio SRCC.
 */
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "voxmutatio/synthesizer/synthesizer.h"

namespace {

struct Ref {
    std::vector<int> shape;
    std::vector<float> data;
    bool ok = false;
};

Ref load_ref(const std::string& path) {
    Ref r;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return r;
    int32_t nd = 0; ifs.read(reinterpret_cast<char*>(&nd), 4);
    if (nd <= 0 || nd > 8) return r;
    int64_t total = 1;
    for (int i = 0; i < nd; ++i) { int32_t s; ifs.read(reinterpret_cast<char*>(&s), 4); r.shape.push_back(s); total *= s; }
    r.data.resize(total);
    ifs.read(reinterpret_cast<char*>(r.data.data()), total * sizeof(float));
    r.ok = ifs.good() || ifs.eof();
    return r;
}

double rms(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return -1;
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) { double d = (double)a[i] - b[i]; s += d * d; }
    return std::sqrt(s / a.size());
}

double srcc_pearson(const std::vector<float>& a, const std::vector<float>& b) {
    // Pearson correlation as a proxy for waveform similarity
    size_t n = std::min(a.size(), b.size());
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double cov = 0, va = 0, vb = 0;
    for (size_t i = 0; i < n; ++i) {
        double da = a[i] - ma, db = b[i] - mb;
        cov += da * db; va += da * da; vb += db * db;
    }
    return cov / (std::sqrt(va * vb) + 1e-12);
}

}  // namespace

TEST_CASE("VITS TextEncoder alignment (m_p)", "[vits][alignment][encp]") {
    using namespace voxmutatio::synthesizer;

    auto phone = load_ref("../tests/fixtures/vits_ref_phone.bin");    // [T,768]
    auto pitch = load_ref("../tests/fixtures/vits_ref_pitch.bin");    // [T,1]
    auto m_ref = load_ref("../tests/fixtures/vits_ref_m_p.bin");      // [192,T]
    REQUIRE(phone.ok); REQUIRE(pitch.ok); REQUIRE(m_ref.ok);

    int T = phone.shape[0];
    std::cout << "T=" << T << " m_p ref=[" << m_ref.shape[0] << "," << m_ref.shape[1] << "]" << std::endl;

    std::vector<int> pitch_i(T);
    for (int t = 0; t < T; ++t) pitch_i[t] = (int)std::lround(pitch.data[t]);

    SynthesizerConfig cfg;
    cfg.model_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    cfg.version = voxmutatio::ModelVersion::kV2;
    cfg.synthesis_mode = voxmutatio::SynthesisMode::kMean;
    Synthesizer synth;
    REQUIRE(synth.init(cfg));

    auto m_p = synth.debug_text_encoder(phone.data.data(), T, pitch_i.data());
    REQUIRE(m_p.size() == m_ref.data.size());
    double e = rms(m_p, m_ref.data);
    std::cout << "m_p RMS error: " << e << std::endl;
    CHECK(e < 1e-3);
}

TEST_CASE("VITS Flow alignment (z)", "[vits][alignment][flow]") {
    using namespace voxmutatio::synthesizer;

    auto phone = load_ref("../tests/fixtures/vits_ref_phone.bin");
    auto pitch = load_ref("../tests/fixtures/vits_ref_pitch.bin");
    auto z_ref = load_ref("../tests/fixtures/vits_ref_z.bin");
    REQUIRE(phone.ok); REQUIRE(pitch.ok); REQUIRE(z_ref.ok);

    int T = phone.shape[0];
    std::vector<int> pitch_i(T);
    for (int t = 0; t < T; ++t) pitch_i[t] = (int)std::lround(pitch.data[t]);

    SynthesizerConfig cfg;
    cfg.model_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    cfg.version = voxmutatio::ModelVersion::kV2;
    cfg.synthesis_mode = voxmutatio::SynthesisMode::kMean;
    Synthesizer synth;
    REQUIRE(synth.init(cfg));

    auto z = synth.debug_flow(phone.data.data(), T, pitch_i.data(), 0);
    REQUIRE(z.size() == z_ref.data.size());
    double e = rms(z, z_ref.data);
    std::cout << "z RMS error: " << e << std::endl;
    CHECK(e < 1e-2);
}

TEST_CASE("VITS full inference alignment (audio, SC-003)", "[vits][alignment][audio]") {
    using namespace voxmutatio::synthesizer;

    auto phone = load_ref("../tests/fixtures/vits_ref_phone.bin");
    auto pitch = load_ref("../tests/fixtures/vits_ref_pitch.bin");
    auto nsff0 = load_ref("../tests/fixtures/vits_ref_nsff0.bin");
    auto audio_ref = load_ref("../tests/fixtures/vits_ref_audio.bin");
    REQUIRE(phone.ok); REQUIRE(pitch.ok); REQUIRE(nsff0.ok); REQUIRE(audio_ref.ok);

    int T = phone.shape[0];

    SynthesizerConfig cfg;
    cfg.model_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    cfg.version = voxmutatio::ModelVersion::kV2;
    cfg.synthesis_mode = voxmutatio::SynthesisMode::kMean;
    Synthesizer synth;
    REQUIRE(synth.init(cfg));

    auto out = synth.infer(phone.data.data(), T, pitch.data.data(), nsff0.data.data(), 0);
    std::cout << "C++ audio: " << out.data.size() << " (ref " << audio_ref.data.size() << ")" << std::endl;
    REQUIRE(out.data.size() == audio_ref.data.size());
    REQUIRE_FALSE(out.data.empty());

    double corr = srcc_pearson(out.data, audio_ref.data);
    double e = rms(out.data, audio_ref.data);
    std::cout << "audio correlation: " << corr << ", RMS: " << e << std::endl;

    // SC-003: high waveform correlation
    CHECK(corr > 0.999);
    // Correlation alone missed using slope 0.1 instead of the reference's
    // default 0.01 immediately before conv_post (about 4% relative L2 error).
    double reference_energy = 0.0;
    for (float value : audio_ref.data) reference_energy += double(value) * value;
    REQUIRE(reference_energy > 0.0);
    const double relative_l2 = e / std::sqrt(reference_energy / audio_ref.data.size());
    std::cout << "audio relative L2: " << relative_l2 << std::endl;
    CHECK(relative_l2 < 1e-3);
}

TEST_CASE("VITS default sampled inference matches upstream supplied Gaussian draws",
          "[vits][alignment][sampled]") {
    using namespace voxmutatio::synthesizer;
    const auto phone = load_ref("../tests/fixtures/vits_ref_phone.bin");
    const auto pitch = load_ref("../tests/fixtures/e2e_htk_pitch.bin");
    const auto f0 = load_ref("../tests/fixtures/e2e_htk_nsff0.bin");
    const auto reference = load_ref("../tests/fixtures/vits_sampled_audio.bin");
    REQUIRE(phone.ok); REQUIRE(pitch.ok); REQUIRE(f0.ok); REQUIRE(reference.ok);
    const int frames = phone.shape[0];
    REQUIRE(pitch.data.size() == static_cast<std::size_t>(frames));
    REQUIRE(f0.data.size() == static_cast<std::size_t>(frames));
    SynthesizerConfig config;
    config.model_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    config.version = voxmutatio::ModelVersion::kV2;
    Synthesizer synth;
    REQUIRE(synth.init(config));
    const auto output = synth.infer(phone.data.data(), frames, pitch.data.data(), f0.data.data(), 0);
    REQUIRE(output.data.size() == reference.data.size());
    double energy = 0.0;
    for (float value : reference.data) energy += double(value) * value;
    REQUIRE(energy > 0.0);
    const double corr = srcc_pearson(output.data, reference.data);
    const double relative_l2 = rms(output.data, reference.data) /
        std::sqrt(energy / reference.data.size());
    std::cout << "sampled correlation=" << corr << " relative_l2=" << relative_l2 << '\n';
    CHECK(corr > 0.999);
    CHECK(relative_l2 < 1e-3);
    // Repeatable samples at fixed seed; another seed changes the realized audio.
    constexpr int short_frames = 40;
    const auto first = synth.infer(phone.data.data(), short_frames, pitch.data.data(), f0.data.data(), 0);
    const auto repeated = synth.infer(phone.data.data(), short_frames, pitch.data.data(), f0.data.data(), 0);
    REQUIRE(first.data.size() == short_frames * 400);
    CHECK(first.data == repeated.data);
    config.synthesis_seed = 1;
    REQUIRE(synth.init(config));
    const auto changed = synth.infer(phone.data.data(), short_frames, pitch.data.data(), f0.data.data(), 0);
    REQUIRE(changed.data.size() == first.data.size());
    CHECK(changed.data != first.data);
    // UV samples must contain NSF excitation, while legacy mean mode has none.
    const std::vector<float> unvoiced(40, 0.0f);
    const auto excited = synth.debug_har(unvoiced.data(), 40);
    config.synthesis_mode = voxmutatio::SynthesisMode::kMean;
    REQUIRE(synth.init(config));
    const auto quiet = synth.debug_har(unvoiced.data(), 40);
    REQUIRE(excited.size() == 16000);
    REQUIRE(quiet.size() == excited.size());
    CHECK(excited != quiet);
    CHECK(*std::max_element(quiet.begin(), quiet.end()) == *std::min_element(quiet.begin(), quiet.end()));

}
