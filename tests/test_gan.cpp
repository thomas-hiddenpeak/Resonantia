// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// GAN component tests (spec 002): discriminator forward/backward on real audio.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#include "voxmutatio/autograd/tensor.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/training/discriminator.h"

using voxmutatio::autograd::Tensor;
namespace ag = voxmutatio::autograd;

TEST_CASE("Discriminator forward/backward on real audio", "[gan][discriminator]") {
  using namespace voxmutatio;

  std::string d_path = "../models/pretrained_v2/pretrained_v2/f0D40k.safetensors";
  training::Discriminator D;
  REQUIRE(D.init(d_path));
  REQUIRE(D.params().size() > 0);

  // Real audio segment @ 40k.
  auto a = io::read_audio("../tests/fixtures/speech_librispeech.wav", 40000);
  REQUIRE(a.has_value());
  const int L = 25600;  // 0.64s
  REQUIRE(static_cast<int>(a->data.size()) >= L);
  std::vector<float> seg(a->data.begin(), a->data.begin() + L);
  auto audio = Tensor::from_host(seg, {1, L}, false);

  auto results = D.forward(audio, L);
  REQUIRE(results.size() == 9);  // 1 DiscriminatorS + 8 DiscriminatorP

  // LSGAN-style scalar: sum of mean(score^2) over all sub-discriminators.
  Tensor loss;
  bool first = true;
  int total_fmaps = 0;
  for (auto& r : results) {
    for (auto& v : r.score.to_host()) REQUIRE(std::isfinite(v));
    total_fmaps += static_cast<int>(r.fmaps.size());
    auto sq = ag::sum(ag::mul(r.score, r.score));
    loss = first ? sq : ag::add(loss, sq);
    first = false;
  }
  std::printf("[disc] sub-Ds=%zu total feature maps=%d, loss=%.4f\n",
              results.size(), total_fmaps, loss.to_host()[0]);
  REQUIRE(std::isfinite(loss.to_host()[0]));

  ag::backward(loss);

  // Every discriminator parameter must receive a finite gradient.
  int checked = 0;
  for (auto& p : D.params()) {
    auto g = p.grad_to_host();
    for (float v : g) REQUIRE(std::isfinite(v));
    checked += !g.empty();
  }
  std::printf("[disc] params with finite grad: %d/%zu\n", checked, D.params().size());
  CHECK(checked == static_cast<int>(D.params().size()));
}
