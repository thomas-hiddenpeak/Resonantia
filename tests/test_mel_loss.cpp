// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Proves the differentiable mel loss can drive optimization through the
// autograd engine (the core training signal for fine-tuning). Uses real
// speech audio (Constitution X).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "voxmutatio/autograd/tensor.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/training/mel_loss.h"

using voxmutatio::autograd::Tensor;
namespace ag = voxmutatio::autograd;

TEST_CASE("Mel loss drives optimization (real audio)", "[training][mel]") {
  auto audio = voxmutatio::io::read_audio("../tests/fixtures/speech_librispeech.wav", 16000);
  REQUIRE(audio.has_value());
  REQUIRE(audio->data.size() > 8192);

  const int L = 8192;
  std::vector<float> target_audio(audio->data.begin(), audio->data.begin() + L);

  voxmutatio::training::MelSpecConfig cfg;
  cfg.n_fft = 1024;
  cfg.hop = 256;
  cfg.n_mels = 80;
  cfg.sample_rate = 16000;
  voxmutatio::training::MelLoss mel(cfg);

  int T = 0;
  auto target_host = mel.target_log_mel(target_audio.data(), L, T);
  REQUIRE(T > 0);
  auto target = Tensor::from_host(target_host, {T, cfg.n_mels}, false);

  // Optimize a random signal to match the target mel spectrogram.
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.0f, 0.05f);
  std::vector<float> init(L);
  for (auto& v : init) v = nd(rng);
  auto x = Tensor::from_host(init, {1, L}, true);

  ag::AdamW opt({x}, /*lr=*/0.02f);

  float first_loss = 0.0f, last_loss = 0.0f;
  for (int it = 0; it < 300; ++it) {
    auto gen = mel.log_mel(x, L);
    auto loss = mel.l1(gen, target, T);
    if (it == 0) first_loss = loss.to_host()[0];
    last_loss = loss.to_host()[0];
    ag::backward(loss);
    opt.step();
  }
  std::printf("[mel-opt] first loss %.4f -> last loss %.4f\n", first_loss, last_loss);

  // The mel L1 loss must drop substantially, proving gradient flow + training.
  CHECK(last_loss < first_loss * 0.5f);
}
