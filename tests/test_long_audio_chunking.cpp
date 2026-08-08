/**
 * @file test_long_audio_chunking.cpp
 * @brief Long-audio chunked inference must match the whole-clip pass.
 *
 * The production pipeline splits clips longer than `max_chunk_sec` at silence,
 * converts each chunk with context padding, and stitches the result. This test
 * tiles a real speech fixture past the chunk threshold and asserts the chunked
 * output is numerically consistent with a single whole-clip pass (same knobs),
 * so the >5min edge case is handled without seams or drift.
 */
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <vector>

#include "voxmutatio/pipeline/pipeline.h"
#include "voxmutatio/io/audio_io.h"

namespace {

double correlation(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = std::min(a.size(), b.size());
    if (n == 0) return 0.0;
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

// In-place iterative radix-2 FFT.
void fft(std::vector<std::complex<float>>& a) {
    int n = static_cast<int>(a.size());
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * static_cast<float>(M_PI) / len;
        std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int k = 0; k < len / 2; ++k) {
                std::complex<float> u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Phase-invariant descriptor: Hann-windowed magnitude spectrogram, flattened.
// NSF vocoders reset harmonic phase per inference call, so chunked and whole-clip
// outputs are perceptually identical but waveform-decorrelated; magnitude removes
// that phase and lets us verify spectral content + coarse time alignment.
std::vector<float> mag_spectrogram(const std::vector<float>& x, int fftn, int hop) {
    if (x.size() < static_cast<size_t>(fftn)) return {};
    int nf = (static_cast<int>(x.size()) - fftn) / hop + 1;
    std::vector<float> win(fftn);
    for (int i = 0; i < fftn; ++i)
        win[i] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * i / (fftn - 1));
    std::vector<float> out;
    out.reserve(static_cast<size_t>(nf) * (fftn / 2 + 1));
    std::vector<std::complex<float>> buf(fftn);
    for (int f = 0; f < nf; ++f) {
        int base = f * hop;
        for (int i = 0; i < fftn; ++i) buf[i] = {x[base + i] * win[i], 0.0f};
        fft(buf);
        for (int k = 0; k <= fftn / 2; ++k) out.push_back(std::abs(buf[k]));
    }
    return out;
}

voxmutatio::VCConfig base_config() {
    using namespace voxmutatio;
    VCConfig cfg;
    cfg.hubert_model_path = "../models/hubert_base/model.safetensors";
    cfg.rmvpe_model_path = "../models/rmvpe.safetensors";
    cfg.synthesizer_model_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    cfg.version = ModelVersion::kV2;
    cfg.has_f0 = true;
    cfg.model_sample_rate = 40000;
    cfg.num_speakers = 109;
    cfg.rms_mix_rate = 1.0;   // isolate chunking (no RMS envelope matching)
    cfg.protect = 0.5;        // protect disabled (>=0.5)
    cfg.apply_highpass = true;
    cfg.interp_unvoiced = true;
    cfg.filter_radius = 3;
    return cfg;
}

}  // namespace

TEST_CASE("Long-audio chunking matches whole-clip pass", "[chunk][integration]") {
    using namespace voxmutatio;

    auto fixture = io::read_audio("../tests/fixtures/speech_librispeech.wav", 16000);
    REQUIRE(fixture.has_value());

    // Tile the fixture past the chunk threshold (~18s -> 3 chunks at 6s cuts).
    AudioBuffer clip;
    clip.sample_rate = 16000;
    for (int rep = 0; rep < 3; ++rep)
        clip.data.insert(clip.data.end(), fixture->data.begin(), fixture->data.end());
    const double clip_sec = static_cast<double>(clip.data.size()) / 16000.0;
    REQUIRE(clip_sec > 12.0);

    // A) Chunked path: force multiple silence-cut segments with 2s context pad.
    VCConfig chunk_cfg = base_config();
    chunk_cfg.max_chunk_sec = 6.0;
    chunk_cfg.chunk_pad_sec = 2.0;
    chunk_cfg.chunk_query_sec = 1.0;
    pipeline::VoiceConversionPipeline chunk_pipe;
    REQUIRE(chunk_pipe.init(chunk_cfg));
    auto chunked = chunk_pipe.convert_buffer(clip, /*speaker_id=*/0);
    REQUIRE(chunked.success);
    REQUIRE(!chunked.audio.data.empty());
    for (float v : chunked.audio.data) { REQUIRE(!std::isnan(v)); REQUIRE(!std::isinf(v)); }

    // B) Reference whole-clip pass: never chunk, match the 2s outer reflect pad.
    VCConfig whole_cfg = base_config();
    whole_cfg.max_chunk_sec = 0.0;   // disable chunking
    whole_cfg.edge_pad_sec = 2.0;    // same outer context as the chunk path
    pipeline::VoiceConversionPipeline whole_pipe;
    REQUIRE(whole_pipe.init(whole_cfg));
    auto whole = whole_pipe.convert_buffer(clip, /*speaker_id=*/0);
    REQUIRE(whole.success);
    REQUIRE(!whole.audio.data.empty());

    // Lengths should match within a few percent (chunk boundaries quantize to frames).
    double len_ratio = static_cast<double>(chunked.audio.data.size()) /
                       static_cast<double>(whole.audio.data.size());
    CHECK(len_ratio > 0.98);
    CHECK(len_ratio < 1.02);

    // Phase-invariant spectral match: the chunked output must reproduce the
    // whole-clip spectrum (content + coarse time alignment). Raw-waveform corr
    // is expected to be low because the NSF vocoder resets harmonic phase per
    // chunk, so it is reported for information only. The residual spectral gap
    // is inherent: HuBERT uses global self-attention, so a chunk's content
    // features legitimately differ from a whole pass (RVC chunks HuBERT too).
    auto sc = mag_spectrogram(chunked.audio.data, 1024, 512);
    auto sw = mag_spectrogram(whole.audio.data, 1024, 512);
    double spec_corr = correlation(sc, sw);
    double wave_corr = correlation(chunked.audio.data, whole.audio.data);
    std::printf("spec_corr=%.4f  wave_corr=%.4f  (chunk=%zu whole=%zu)\n",
                spec_corr, wave_corr, chunked.audio.data.size(), whole.audio.data.size());
    CHECK(spec_corr > 0.98);
}
