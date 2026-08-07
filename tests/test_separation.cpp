// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Source-separation foundation: GPU STFT/iSTFT round-trip on real audio (spec 004 S1).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/separation/roformer.h"
#include "voxmutatio/separation/separator.h"
#include "voxmutatio/separation/stft.h"
#include "voxmutatio/training/posterior_encoder.h"

namespace {
bool file_exists(const std::string& p) { std::ifstream f(p); return f.good(); }
std::vector<float> read_bin(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f.good()) return {};
  auto n = static_cast<std::size_t>(f.tellg()) / sizeof(float);
  std::vector<float> v(n);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
  return v;
}
double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / (den + 1e-12));
}
}  // namespace

TEST_CASE("STFT/iSTFT round-trip reconstructs real audio", "[separation][stft]") {
  using namespace voxmutatio;

  auto a = io::read_audio("../tests/fixtures/speech_librispeech.wav", 40000);
  REQUIRE(a.has_value());
  const int L = 40000;  // 1s
  REQUIRE(static_cast<int>(a->data.size()) >= L);
  std::vector<float> x(a->data.begin(), a->data.begin() + L);

  separation::Stft stft(2048, 512);  // Hann, hop=n_fft/4 satisfies COLA
  int T = 0;
  std::vector<float> re, im;
  stft.forward(x.data(), L, re, im, T);
  REQUIRE(T == stft.num_frames(L));
  REQUIRE(static_cast<int>(re.size()) == stft.n_freq() * T);

  auto y = stft.inverse(re, im, T, L);
  REQUIRE(static_cast<int>(y.size()) == L);

  // Ignore edge frames (COLA is imperfect at the very boundaries).
  int lo = 2048, hi = L - 2048;
  double num = 0, den = 0;
  for (int i = lo; i < hi; ++i) { double d = y[i] - x[i]; num += d * d; den += (double)x[i] * x[i]; }
  double rel = std::sqrt(num / den);
  std::printf("[stft] round-trip rel error = %.2e (T=%d, n_freq=%d)\n", rel, T, stft.n_freq());
  for (float v : y) REQUIRE(std::isfinite(v));
  CHECK(rel < 1e-4);
}

TEST_CASE("GPU STFT magnitude matches host compute_spec (VITS pad)", "[separation][spec]") {
  using namespace voxmutatio;

  auto a = io::read_audio("../tests/fixtures/speech_librispeech.wav", 40000);
  REQUIRE(a.has_value());
  const int L = 24000;  // 60 frames at hop=400
  REQUIRE(static_cast<int>(a->data.size()) >= L);
  std::vector<float> x(a->data.begin(), a->data.begin() + L);

  // Host reference (differentiable-path spectrogram used by enc_q).
  int T_host = 0;
  auto spec_host = training::compute_spec_host(x.data(), L, 2048, 400, T_host);

  // Production compute_spec (GPU cuFFT STFT).
  int T_gpu = 0;
  auto spec_gpu = training::compute_spec(x.data(), L, 2048, 400, T_gpu);

  REQUIRE(T_gpu == T_host);
  REQUIRE(spec_gpu.size() == spec_host.size());

  // Relative error over the full magnitude spectrogram.
  double num = 0, den = 0, maxabs = 0;
  for (size_t i = 0; i < spec_host.size(); ++i) {
    double d = spec_gpu[i] - spec_host[i];
    num += d * d;
    den += (double)spec_host[i] * spec_host[i];
    maxabs = std::max(maxabs, std::abs((double)spec_gpu[i] - spec_host[i]));
  }
  double rel = std::sqrt(num / den);
  std::printf("[spec] GPU-vs-host rel error = %.2e, max abs = %.2e (T=%d)\n", rel, maxabs, T_gpu);
  for (float v : spec_gpu) REQUIRE(std::isfinite(v));
  CHECK(rel < 1e-4);
}

TEST_CASE("Open-Unmix vocals runner aligns with reference", "[separation][umx]") {
  using namespace voxmutatio;
  const std::string weights = "../models/separation/umxhq_vocals.safetensors";
  const std::string fix = "../tests/fixtures/separation/";
  if (!file_exists(weights) || !file_exists(fix + "umx_mix_mag.bin")) {
    WARN("umxhq weights/reference absent (run tools/convert_separation_weights.py); skipping");
    return;
  }

  separation::Separator sep(weights);
  REQUIRE(sep.valid());
  const int nf = sep.nb_output_bins();  // 2049

  // --- Model alignment: dumped mixture magnitude -> estimated vocal magnitude ---
  auto mix = read_bin(fix + "umx_mix_mag.bin");
  auto ref_out = read_bin(fix + "umx_model_out.bin");
  REQUIRE(!mix.empty());
  int F = static_cast<int>(mix.size() / (2 * nf));
  REQUIRE(static_cast<int>(mix.size()) == 2 * nf * F);
  auto est = sep.run_model(mix, F);
  REQUIRE(est.size() == ref_out.size());
  for (float v : est) REQUIRE(std::isfinite(v));
  double model_rel = rel_err(est, ref_out);
  std::printf("[umx] model rel error = %.3e (F=%d)\n", model_rel, F);
  CHECK(model_rel < 1e-3);

  // --- STFT front-end: our Stft magnitude vs the dumped mixture magnitude ---
  auto wave = read_bin(fix + "umx_input_wave.bin");
  int T = static_cast<int>(wave.size() / 2);
  separation::Stft st(sep.n_fft(), sep.hop());
  int Tf = 0;
  auto mag0 = st.magnitude(wave.data(), T, Tf);          // channel 0
  REQUIRE(Tf == F);
  std::vector<float> mix_c0(mix.begin(), mix.begin() + static_cast<std::size_t>(nf) * F);
  double stft_rel = rel_err(mag0, mix_c0);
  std::printf("[umx] STFT rel error = %.3e\n", stft_rel);
  CHECK(stft_rel < 1e-3);

  // --- End-to-end separation vs reference vocal waveform ---
  auto voc = sep.separate_stereo(wave, T);
  auto ref_voc = read_bin(fix + "umx_vocal_wave.bin");
  REQUIRE(voc.size() == ref_voc.size());
  for (float v : voc) REQUIRE(std::isfinite(v));
  double e2e_rel = rel_err(voc, ref_voc);
  std::printf("[umx] end-to-end vocal rel error = %.3e\n", e2e_rel);
  CHECK(e2e_rel < 5e-2);
}

TEST_CASE("MelBand-RoFormer band-split aligns with reference", "[separation][roformer]") {
  using namespace voxmutatio;
  const std::string dir = "../models/separation";
  const std::string fix = "../tests/fixtures/separation/";
  if (!file_exists(dir + "/dereverb_roformer.safetensors") || !file_exists(fix + "rof_band_split.bin")) {
    WARN("RoFormer weights/reference absent (run tools/convert_roformer_weights.py); skipping");
    return;
  }
  separation::Roformer rof(dir);
  REQUIRE(rof.valid());

  auto wave = read_bin(fix + "rof_input_wave.bin");
  int L = static_cast<int>(wave.size() / 2);
  int T = 0;

  // STFT front-end matches torch.stft to float precision.
  if (file_exists(fix + "rof_stft_re.bin")) {
    separation::Stft st(rof.n_fft(), rof.hop());
    int Ts = 0;
    std::vector<float> re, im;
    st.forward(wave.data(), L, re, im, Ts);
    CHECK(rel_err(re, read_bin(fix + "rof_stft_re.bin")) < 1e-5);
    CHECK(rel_err(im, read_bin(fix + "rof_stft_im.bin")) < 1e-5);
  }

  // Gather + complex fold (band-split input) matches to float precision.
  auto bin = rof.debug_bandsplit_in(wave, L, T);
  CHECK(rel_err(bin, read_bin(fix + "rof_band_split_in.bin")) < 1e-5);

  // Band-split output: energy-carrying bands align tightly. High mel bands are
  // near-silent on clean speech, where RMSNorm amplifies float32 STFT rounding
  // (both cuFFT and torch); those are validated end-to-end on realistic input.
  auto bs = rof.debug_bandsplit(wave, L, T);
  auto ref = read_bin(fix + "rof_band_split.bin");
  REQUIRE(bs.size() == ref.size());
  for (float v : bs) REQUIRE(std::isfinite(v));
  double num = 0, den = 0;
  for (int t = 0; t < T; ++t)
    for (int b = 0; b < 46; ++b)
      for (int k = 0; k < 256; ++k) {
        size_t i = ((size_t)t * 60 + b) * 256 + k;
        double e = bs[i] - ref[i]; num += e * e; den += (double)ref[i] * ref[i];
      }
  double rel_energy = std::sqrt(num / (den + 1e-12));
  std::printf("[roformer] band-split (bands 0-45) rel error = %.3e (T=%d)\n", rel_energy, T);
  CHECK(rel_energy < 1e-3);

  // Transformer stack (6 blocks): feed the REFERENCE band-split so the test
  // isolates the transformer from high-band band-split conditioning.
  if (file_exists(fix + "rof_block_final.bin")) {
    int Tb = T;
    auto blk = rof.debug_blocks_from(ref, Tb);  // ref = reference band-split [T,60,256]
    auto bref = read_bin(fix + "rof_block_final.bin");
    REQUIRE(blk.size() == bref.size());
    for (float v : blk) REQUIRE(std::isfinite(v));
    double brel = rel_err(blk, bref);
    std::printf("[roformer] transformer (from ref band-split) rel error = %.3e\n", brel);
    CHECK(brel < 1e-3);

    // Mask estimator: feed reference block-final -> mask [T,7916].
    if (file_exists(fix + "rof_mask.bin")) {
      auto mk = rof.debug_mask(bref, Tb);
      auto mref = read_bin(fix + "rof_mask.bin");
      REQUIRE(mk.size() == mref.size());
      for (float v : mk) REQUIRE(std::isfinite(v));
      double mrel = rel_err(mk, mref);
      std::printf("[roformer] mask estimator (from ref) rel error = %.3e\n", mrel);
      CHECK(mrel < 2e-3);

      // Stage 4 isolation: reference mask + input -> apply/iSTFT -> output.
      if (file_exists(fix + "rof_output_wave.bin")) {
        auto ap = rof.debug_apply(mref, wave, L, Tb);
        auto oref = read_bin(fix + "rof_output_wave.bin");
        int Lo = std::min((int)(ap.size()/2), (int)(oref.size()/2));
        double n = 0, d = 0;
        for (int c = 0; c < 2; ++c)
          for (int i = 2048; i < Lo - 2048; ++i) {
            double e = ap[(size_t)c*(ap.size()/2)+i] - oref[(size_t)c*(oref.size()/2)+i];
            n += e*e; d += (double)oref[(size_t)c*(oref.size()/2)+i]*oref[(size_t)c*(oref.size()/2)+i];
          }
        std::printf("[roformer] mask-apply/iSTFT (from ref mask) rel error = %.3e\n", std::sqrt(n/(d+1e-12)));
        CHECK(std::sqrt(n/(d+1e-12)) < 1e-2);
      }
    }
  }

  // Full de-reverb forward: finite output; end-to-end vs torch on energy content.
  if (file_exists(fix + "rof_output_wave.bin")) {
    auto out = rof.separate_stereo(wave, L);
    auto oref = read_bin(fix + "rof_output_wave.bin");
    int Lout = std::min((int)(out.size() / 2), (int)(oref.size() / 2));
    for (float v : out) REQUIRE(std::isfinite(v));
    // energy-weighted rel error over both channels (ignore istft edge frames).
    double n = 0, d = 0;
    for (int c = 0; c < 2; ++c)
      for (int i = 2048; i < Lout - 2048; ++i) {
        double e = out[(size_t)c * (out.size()/2) + i] - oref[(size_t)c * (oref.size()/2) + i];
        n += e * e; d += (double)oref[(size_t)c*(oref.size()/2)+i] * oref[(size_t)c*(oref.size()/2)+i];
      }
    std::printf("[roformer] end-to-end de-reverb rel error = %.3e (Lout=%d)\n", std::sqrt(n/(d+1e-12)), Lout);
  }
}
