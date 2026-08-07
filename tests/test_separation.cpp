// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Source-separation foundation: GPU STFT/iSTFT round-trip on real audio (spec 004 S1).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/separation/stft.h"
#include "voxmutatio/training/posterior_encoder.h"

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
