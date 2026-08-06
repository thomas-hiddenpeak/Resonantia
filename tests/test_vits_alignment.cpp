/**
 * @file test_vits_alignment.cpp
 * @brief VITS synthesizer numerical alignment vs deterministic Python reference.
 *
 * Stage-by-stage: TextEncoder (m_p) -> Flow (z) -> Generator (audio).
 * SC-003 relates to final audio SRCC.
 */
#include <catch2/catch_test_macros.hpp>

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
    Synthesizer synth;
    REQUIRE(synth.init(cfg));

    auto out = synth.infer(phone.data.data(), T, pitch.data.data(), nsff0.data.data(), 0);
    std::cout << "C++ audio: " << out.data.size() << " (ref " << audio_ref.data.size() << ")" << std::endl;

    double corr = srcc_pearson(out.data, audio_ref.data);
    double e = rms(out.data, audio_ref.data);
    std::cout << "audio correlation: " << corr << ", RMS: " << e << std::endl;

    // SC-003: high waveform correlation
    CHECK(corr > 0.999);
}
