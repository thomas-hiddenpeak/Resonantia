// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// GAN component tests (spec 002): discriminator forward/backward on real audio.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

#include "voxmutatio/autograd/tensor.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/synthesizer/synthesizer.h"
#include "voxmutatio/training/discriminator.h"
#include "voxmutatio/training/gan_trainer.h"
#include "voxmutatio/training/posterior_encoder.h"

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

TEST_CASE("PosteriorEncoder reconstructs real audio (enc_q + dec)", "[gan][encq]") {
  using namespace voxmutatio;

  auto nsff0 = load_bin("../tests/fixtures/vits_ref_nsff0.bin");
  REQUIRE(nsff0.ok);
  int Tfull = nsff0.shape[0];

  std::string g_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";

  synthesizer::SynthesizerConfig scfg;
  scfg.model_path = g_path; scfg.version = ModelVersion::kV2;
  scfg.sample_rate = 40000; scfg.spk_embed_dim = 109;
  synthesizer::Synthesizer synth; REQUIRE(synth.init(scfg));
  auto har_full = synth.debug_har(nsff0.data.data(), Tfull);

  // Real audio @ native 40k, aligned to a whole number of frames.
  auto a40 = io::read_audio("../tests/fixtures/speech_librispeech.wav", 40000);
  REQUIRE(a40.has_value());
  const int Tseg = 128, Lseg = Tseg * 400;
  REQUIRE(static_cast<int>(a40->data.size()) >= Lseg);
  std::vector<float> y(a40->data.begin(), a40->data.begin() + Lseg);
  std::vector<float> har_seg(har_full.begin(), har_full.begin() + Lseg);

  // Linear spectrogram (VITS: n_fft=2048, hop=400).
  int Tspec = 0;
  auto spec_host = training::compute_spec(y.data(), Lseg, 2048, 400, Tspec);
  REQUIRE(Tspec == Tseg);
  auto spec = Tensor::from_host(spec_host, {1025, Tseg}, false);

  // enc_q (mean) -> z -> decoder should reconstruct the target audio.
  training::PosteriorEncoder enc_q; REQUIRE(enc_q.init(g_path, 0));
  training::GeneratorTrainer dec; REQUIRE(dec.init(g_path, 0));

  Tensor m_q, logs_q;
  auto z = enc_q.forward(spec, Tseg, /*sample=*/false, m_q, logs_q);
  REQUIRE(z.shape()[0] == 192);
  REQUIRE(z.shape()[1] == Tseg);

  auto har_t = Tensor::from_host(har_seg, {1, Lseg}, false);
  auto y_hat = dec.decode(z, har_t, Tseg).to_host();
  REQUIRE(static_cast<int>(y_hat.size()) == Lseg);
  for (float v : y_hat) REQUIRE(std::isfinite(v));

  // Vocoder reconstruction is phase-agnostic, so compare in the mel domain.
  training::MelSpecConfig mc;
  mc.n_fft = 1024; mc.hop = 256; mc.n_mels = 80; mc.sample_rate = 40000;
  training::MelLoss mel(mc);
  int Ta = 0, Tb = 0;
  auto ma_host = mel.target_log_mel(y_hat.data(), Lseg, Ta);
  auto mb_host = mel.target_log_mel(y.data(), Lseg, Tb);
  int nn = std::min(ma_host.size(), mb_host.size());
  double ma = 0, mb = 0;
  for (int i = 0; i < nn; ++i) { ma += ma_host[i]; mb += mb_host[i]; }
  ma /= nn; mb /= nn;
  double c = 0, va = 0, vb = 0;
  for (int i = 0; i < nn; ++i) { double da = ma_host[i]-ma, db = mb_host[i]-mb; c += da*db; va += da*da; vb += db*db; }
  double corr = c / (std::sqrt(va*vb) + 1e-12);
  std::printf("[encq] mel-domain reconstruction corr = %.4f\n", corr);
  // Pretrained enc_q+dec is an autoencoder: mel content must track the target.
  CHECK(corr > 0.5);
}

TEST_CASE("Flow forward inverts inference flow_reverse", "[gan][flow]") {
  using namespace voxmutatio;

  std::string g_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
  synthesizer::SynthesizerConfig scfg;
  scfg.model_path = g_path; scfg.version = ModelVersion::kV2;
  scfg.sample_rate = 40000; scfg.spk_embed_dim = 109;
  synthesizer::Synthesizer synth; REQUIRE(synth.init(scfg));

  training::Flow flow; REQUIRE(flow.init(g_path, 0));

  // Random latent; flow forward then the proven inference reverse must round-trip.
  const int T = 64;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> zq(192 * T);
  for (auto& v : zq) v = nd(rng);
  auto z = Tensor::from_host(zq, {192, T}, false);

  auto zp = flow.forward(z, T).to_host();
  for (float v : zp) REQUIRE(std::isfinite(v));
  auto zback = synth.debug_flow_reverse(zp.data(), T, 0);
  REQUIRE(zback.size() == zq.size());

  double num = 0, den = 0;
  for (size_t i = 0; i < zq.size(); ++i) { double d = zback[i] - zq[i]; num += d*d; den += (double)zq[i]*zq[i]; }
  double rel = std::sqrt(num / den);
  std::printf("[flow] round-trip rel error = %.2e\n", rel);
  // flow.forward must be the exact inverse of flow_reverse.
  CHECK(rel < 1e-3);
}

TEST_CASE("Full GAN step (enc_q + flow + KL + disc) on real audio", "[gan][full]") {
  using namespace voxmutatio;

  auto nsff0 = load_bin("../tests/fixtures/vits_ref_nsff0.bin");
  auto m_p = load_bin("../tests/fixtures/vits_ref_m_p.bin");
  auto logs_p = load_bin("../tests/fixtures/vits_ref_logs_p.bin");
  REQUIRE(nsff0.ok); REQUIRE(m_p.ok); REQUIRE(logs_p.ok);
  int Tfull = nsff0.shape[0];
  REQUIRE(m_p.shape[0] == 192);

  std::string g_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
  std::string d_path = "../models/pretrained_v2/pretrained_v2/f0D40k.safetensors";

  synthesizer::SynthesizerConfig scfg;
  scfg.model_path = g_path; scfg.version = ModelVersion::kV2;
  scfg.sample_rate = 40000; scfg.spk_embed_dim = 109;
  synthesizer::Synthesizer synth; REQUIRE(synth.init(scfg));
  auto har_full = synth.debug_har(nsff0.data.data(), Tfull);

  auto a40 = io::read_audio("../tests/fixtures/speech_librispeech.wav", 40000);
  REQUIRE(a40.has_value());

  const int Tseg = 64, Lseg = Tseg * 400;
  REQUIRE(static_cast<int>(a40->data.size()) >= Lseg);
  std::vector<float> target(a40->data.begin(), a40->data.begin() + Lseg);
  std::vector<float> har_seg(har_full.begin(), har_full.begin() + Lseg);

  int Tspec = 0;
  auto spec = training::compute_spec(target.data(), Lseg, 2048, 400, Tspec);
  REQUIRE(Tspec == Tseg);

  // Prior stats (enc_p output) sliced to the segment.
  std::vector<float> mp_seg(192 * Tseg), lsp_seg(192 * Tseg);
  for (int c = 0; c < 192; ++c)
    for (int t = 0; t < Tseg; ++t) {
      mp_seg[c * Tseg + t] = m_p.data[c * Tfull + t];
      lsp_seg[c * Tseg + t] = logs_p.data[c * Tfull + t];
    }

  training::MelSpecConfig mcfg;
  mcfg.n_fft = 1024; mcfg.hop = 256; mcfg.n_mels = 80; mcfg.sample_rate = 40000;

  training::GANTrainer gan;
  REQUIRE(gan.init(g_path, d_path, 0, mcfg, 1e-4f, 1e-4f));

  float first_mel = 0.0f, last_mel = 0.0f;
  const int steps = 3;
  for (int it = 0; it < steps; ++it) {
    auto ls = gan.train_step_full(spec, har_seg, target, mp_seg, lsp_seg, Tseg, Lseg);
    if (it == 0) first_mel = ls.mel;
    last_mel = ls.mel;
    std::printf("[gan-full] step %d  D=%.3f G=%.3f (mel=%.4f kl=%.4f fm=%.4f adv=%.4f)\n",
                it + 1, ls.d, ls.g, ls.mel, ls.kl, ls.fm, ls.adv);
    REQUIRE(std::isfinite(ls.d));
    REQUIRE(std::isfinite(ls.g));
    REQUIRE(std::isfinite(ls.mel));
    REQUIRE(std::isfinite(ls.kl));
    REQUIRE(std::isfinite(ls.fm));
  }
  std::printf("[gan-full] mel %.4f -> %.4f\n", first_mel, last_mel);
  // Full posterior-path GAN must stay finite and stable (no divergence).
  CHECK(std::isfinite(last_mel));
  CHECK(last_mel < first_mel * 3.0f + 1.0f);
}
