// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// GAN component tests (spec 002): discriminator forward/backward on real audio.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "voxmutatio/autograd/tensor.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/synthesizer/synthesizer.h"
#include "voxmutatio/training/discriminator.h"
#include "voxmutatio/training/gan_trainer.h"

using voxmutatio::autograd::Tensor;
namespace ag = voxmutatio::autograd;

namespace {
struct Ref { std::vector<int> shape; std::vector<float> data; bool ok = false; };
Ref load_bin(const std::string& p) {
  Ref r; std::ifstream f(p, std::ios::binary); if (!f.is_open()) return r;
  int32_t nd; f.read(reinterpret_cast<char*>(&nd), 4); if (nd <= 0 || nd > 8) return r;
  int64_t tot = 1;
  for (int i = 0; i < nd; ++i) { int32_t s; f.read(reinterpret_cast<char*>(&s), 4); r.shape.push_back(s); tot *= s; }
  r.data.resize(tot); f.read(reinterpret_cast<char*>(r.data.data()), tot * 4);
  r.ok = f.good() || f.eof(); return r;
}
}  // namespace

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

TEST_CASE("GAN fine-tune step runs on real audio", "[gan][finetune]") {
  using namespace voxmutatio;

  auto z = load_bin("../tests/fixtures/vits_ref_z.bin");
  auto nsff0 = load_bin("../tests/fixtures/vits_ref_nsff0.bin");
  REQUIRE(z.ok); REQUIRE(nsff0.ok);
  int Tfull = z.shape[1];

  std::string g_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
  std::string d_path = "../models/pretrained_v2/pretrained_v2/f0D40k.safetensors";

  synthesizer::SynthesizerConfig scfg;
  scfg.model_path = g_path; scfg.version = ModelVersion::kV2;
  scfg.sample_rate = 40000; scfg.spk_embed_dim = 109;
  synthesizer::Synthesizer synth; REQUIRE(synth.init(scfg));
  auto har_full = synth.debug_har(nsff0.data.data(), Tfull);

  // Real target voice @ native 40k.
  auto a40 = io::read_audio("../tests/fixtures/speech_librispeech.wav", 40000);
  REQUIRE(a40.has_value());

  const int Tseg = 32, Lseg = Tseg * 400;
  REQUIRE(static_cast<int>(a40->data.size()) >= Lseg);
  std::vector<float> z_seg(192 * Tseg), har_seg(har_full.begin(), har_full.begin() + Lseg);
  for (int c = 0; c < 192; ++c)
    for (int t = 0; t < Tseg; ++t) z_seg[c * Tseg + t] = z.data[c * Tfull + t];
  std::vector<float> target(a40->data.begin(), a40->data.begin() + Lseg);

  training::MelSpecConfig mcfg;
  mcfg.n_fft = 1024; mcfg.hop = 256; mcfg.n_mels = 80; mcfg.sample_rate = 40000;

  training::GANTrainer gan;
  REQUIRE(gan.init(g_path, d_path, 0, mcfg, /*g_lr=*/1e-4f, /*d_lr=*/1e-4f));

  float first_mel = 0.0f, last_mel = 0.0f;
  const int steps = 8;
  for (int it = 0; it < steps; ++it) {
    auto ls = gan.train_step(z_seg, har_seg, target, Tseg, Lseg);
    if (it == 0) first_mel = ls.mel;
    last_mel = ls.mel;
    std::printf("[gan] step %d  D=%.4f G=%.4f (mel=%.4f fm=%.4f adv=%.4f)\n",
                it + 1, ls.d, ls.g, ls.mel, ls.fm, ls.adv);
    REQUIRE(std::isfinite(ls.d));
    REQUIRE(std::isfinite(ls.g));
    REQUIRE(std::isfinite(ls.mel));
    REQUIRE(std::isfinite(ls.fm));
  }
  std::printf("[gan] mel %.4f -> %.4f\n", first_mel, last_mel);
  // Adversarial + fm + mel training must stay finite and reduce reconstruction.
  CHECK(last_mel < first_mel);
}
