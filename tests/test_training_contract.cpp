// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <array>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "voxmutatio/training/mel_loss.h"
#include "voxmutatio/training/posterior_encoder.h"

TEST_CASE("RVC mel profile keeps crop frame counts and rejects incompatible geometry", "[training][joint]") {
  using namespace voxmutatio::training;
  MelSpecConfig config;
  config.rvc_profile = true;
  REQUIRE_THROWS_AS(MelLoss(config), std::invalid_argument);
  config.n_fft = 2048; config.hop = 400; config.n_mels = 125;
  MelLoss mel(config);
  CHECK(mel.num_frames(12800) == 32);
  CHECK(mel.num_frames(240000) == 600);
  CHECK_THROWS_AS(mel.num_frames(824), std::invalid_argument);
}

TEST_CASE("Spectrogram host and GPU agree at odd padding boundaries", "[training][bounds]") {
  using namespace voxmutatio::training;
  for (const auto [length, fft, hop] :
       {std::array<int, 3>{6, 8, 3}, {8, 7, 2}, {8, 8, 2}, {9, 8, 3}}) {
    const std::vector<float> audio(length, 0.25f);
    int host_frames = -1, gpu_frames = -1;
    const auto host = compute_spec_host(audio.data(), length, fft, hop, host_frames);
    const int expected = 1 + (length + 2 * ((fft - hop) / 2) - fft) / hop;
    REQUIRE(host_frames == expected);
    const auto gpu = compute_spec(audio.data(), length, fft, hop, gpu_frames);
    REQUIRE(gpu_frames == expected);
    REQUIRE(host.size() == gpu.size());
    for (std::size_t i = 0; i < host.size(); ++i)
      CHECK(std::abs(host[i] - gpu[i]) < 1e-5f);
  }
  const float audio[] = {0.25f, 0.25f, 0.25f};
  int frames = -1;
  REQUIRE_THROWS_AS(compute_spec_host(audio, 3, 8, 3, frames), std::invalid_argument);
  REQUIRE_THROWS_AS(compute_spec(audio, 3, 8, 3, frames), std::invalid_argument);
  REQUIRE_THROWS_AS(compute_spec_host(audio, 3, std::numeric_limits<int>::max(), 1, frames),
                    std::invalid_argument);
}

TEST_CASE("MelLoss rejects invalid configuration before creating a graph", "[training][bounds]") {
  using namespace voxmutatio::training;
  MelSpecConfig valid;
  valid.n_fft = 8; valid.hop = 2; valid.n_mels = 3; valid.sample_rate = 16000;
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  const auto inf = std::numeric_limits<float>::infinity();
  const std::vector<std::function<void(MelSpecConfig&)>> invalid = {
      [](auto& c) { c.n_fft = 1; }, [](auto& c) { c.n_fft = 0; },
      [](auto& c) { c.hop = 0; }, [](auto& c) { c.hop = -1; },
      [](auto& c) { c.n_mels = 0; }, [](auto& c) { c.sample_rate = 0; },
      [](auto& c) { c.fmin = -1; }, [](auto& c) { c.fmax = -1; },
      [](auto& c) { c.fmin = 8000; }, [](auto& c) { c.fmax = 8001; },
      [](auto& c) { c.fmin = 1000; c.fmax = 1000; },
      [nan](auto& c) { c.fmin = nan; }, [inf](auto& c) { c.fmin = inf; },
      [nan](auto& c) { c.fmax = nan; }, [inf](auto& c) { c.fmax = inf; },
      [](auto& c) { c.n_fft = std::numeric_limits<int>::max(); },
      [](auto& c) { c.n_mels = std::numeric_limits<int>::max(); },
  };
  for (const auto& corrupt : invalid) {
    auto config = valid;
    corrupt(config);
    CHECK_THROWS_AS(MelLoss(config), std::invalid_argument);
  }
  MelLoss mel(valid);
  CHECK(mel.num_frames(7) == 0);
  CHECK(mel.num_frames(8) == 1);
  CHECK(mel.num_frames(10) == 2);
  CHECK_THROWS_AS(mel.num_frames(-1), std::invalid_argument);
  using voxmutatio::autograd::Tensor;
  CHECK_THROWS_AS(mel.log_mel(Tensor{}, 8), std::invalid_argument);
  CHECK_THROWS_AS(mel.log_mel(Tensor::from_host(std::vector<float>(7), {1, 7}), 7),
                  std::invalid_argument);
  CHECK_THROWS_AS(mel.log_mel(Tensor::from_host(std::vector<float>(8), {2, 4}), 8),
                  std::invalid_argument);
  int frames = 99;
  CHECK_THROWS_AS(mel.target_log_mel(nullptr, 8, frames), std::invalid_argument);
  CHECK(frames == 99);
  const std::vector<float> silence(8, 0);
  CHECK_THROWS_AS(mel.target_log_mel(silence.data(), std::numeric_limits<int>::max(), frames),
                  std::invalid_argument);
  CHECK(frames == 99);
  const auto target = mel.target_log_mel(silence.data(), 8, frames);
  CHECK(frames == 1);
  for (float value : target) CHECK(std::isfinite(value));
  auto expected = Tensor::from_host(target, {1, 3});
  CHECK(mel.l1(expected, expected, 1).to_host()[0] == 0);
  CHECK_THROWS_AS(mel.l1(expected, expected, 0), std::invalid_argument);
  CHECK_THROWS_AS(mel.l1(expected, Tensor::from_host(target, {3, 1}), 1),
                  std::invalid_argument);
}
