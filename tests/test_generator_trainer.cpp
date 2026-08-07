// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Validates the trainable autograd decoder against the proven inference
// decoder: given the same latent z and harmonic source, the autograd
// generator must reproduce the reference waveform (spec 002 correctness gate).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "voxmutatio/autograd/tensor.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/synthesizer/synthesizer.h"
#include "voxmutatio/training/generator_trainer.h"
#include "voxmutatio/training/mel_loss.h"

using voxmutatio::autograd::Tensor;

namespace {

struct Ref { std::vector<int> shape; std::vector<float> data; bool ok = false; };
Ref load(const std::string& p) {
  Ref r; std::ifstream f(p, std::ios::binary); if (!f.is_open()) return r;
  int32_t nd; f.read(reinterpret_cast<char*>(&nd), 4); if (nd <= 0 || nd > 8) return r;
  int64_t tot = 1;
  for (int i = 0; i < nd; ++i) { int32_t s; f.read(reinterpret_cast<char*>(&s), 4); r.shape.push_back(s); tot *= s; }
  r.data.resize(tot); f.read(reinterpret_cast<char*>(r.data.data()), tot * 4);
  r.ok = f.good() || f.eof(); return r;
}
double corr(const std::vector<float>& a, const std::vector<float>& b) {
  size_t n = std::min(a.size(), b.size()); double ma = 0, mb = 0;
  for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n;
  double c = 0, va = 0, vb = 0;
  for (size_t i = 0; i < n; ++i) { double da = a[i]-ma, db = b[i]-mb; c += da*db; va += da*da; vb += db*db; }
  return c / (std::sqrt(va * vb) + 1e-12);
}

}  // namespace

TEST_CASE("Autograd decoder matches inference decoder", "[training][decoder][alignment]") {
  using namespace voxmutatio;

  auto z = load("../tests/fixtures/vits_ref_z.bin");        // [192,584]
  auto nsff0 = load("../tests/fixtures/vits_ref_nsff0.bin"); // [584]
  auto ref_audio = load("../tests/fixtures/vits_ref_audio.bin"); // [233600]
  REQUIRE(z.ok); REQUIRE(nsff0.ok); REQUIRE(ref_audio.ok);
  int T = z.shape[1];  // 584
  REQUIRE(z.shape[0] == 192);

  std::string g_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";

  // Harmonic source from the (proven) synthesizer sine generator.
  synthesizer::SynthesizerConfig scfg;
  scfg.model_path = g_path;
  scfg.version = ModelVersion::kV2;
  scfg.sample_rate = 40000;
  scfg.spk_embed_dim = 109;
  synthesizer::Synthesizer synth;
  REQUIRE(synth.init(scfg));
  auto har = synth.debug_har(nsff0.data.data(), T);
  REQUIRE(static_cast<int>(har.size()) == T * 400);

  // Trainable autograd decoder.
  training::GeneratorTrainer gt;
  REQUIRE(gt.init(g_path, 0));

  auto z_const = Tensor::from_host(z.data, {192, T}, false);
  auto har_const = Tensor::from_host(har, {1, T * 400}, false);
  auto out = gt.decode(z_const, har_const, T);
  auto cpp = out.to_host();

  std::cout << "autograd decoder: " << cpp.size() << " (ref " << ref_audio.data.size() << ")" << std::endl;
  REQUIRE(cpp.size() == ref_audio.data.size());

  double c = corr(cpp, ref_audio.data);
  double rms = 0.0;
  for (size_t i = 0; i < cpp.size(); ++i) { double d = cpp[i] - ref_audio.data[i]; rms += d * d; }
  rms = std::sqrt(rms / cpp.size());
  std::cout << "correlation: " << c << ", RMS: " << rms << std::endl;

  // The autograd decoder must reproduce the inference decoder output.
  CHECK(c > 0.999);
}

TEST_CASE("Decoder fine-tune reduces mel loss (real audio)", "[training][decoder][finetune]") {
  using namespace voxmutatio;
  namespace ag = voxmutatio::autograd;

  auto z = load("../tests/fixtures/vits_ref_z.bin");
  auto nsff0 = load("../tests/fixtures/vits_ref_nsff0.bin");
  REQUIRE(z.ok); REQUIRE(nsff0.ok);
  int Tfull = z.shape[1];

  std::string g_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";

  synthesizer::SynthesizerConfig scfg;
  scfg.model_path = g_path; scfg.version = ModelVersion::kV2;
  scfg.sample_rate = 40000; scfg.spk_embed_dim = 109;
  synthesizer::Synthesizer synth; REQUIRE(synth.init(scfg));
  auto har_full = synth.debug_har(nsff0.data.data(), Tfull);

  // Target voice: real speech resampled 16k -> 40k.
  auto a16 = io::read_audio("../tests/fixtures/speech_librispeech.wav", 16000);
  REQUIRE(a16.has_value());
  auto t40 = io::resample_linear(a16->data.data(), (int)a16->data.size(), 16000, 40000);

  // Train on a short segment for speed.
  const int Tseg = 32, Lseg = Tseg * 400;  // 12800 samples @ 40k
  REQUIRE((int)t40.size() >= Lseg);
  std::vector<float> z_seg(192 * Tseg), har_seg(har_full.begin(), har_full.begin() + Lseg);
  for (int c = 0; c < 192; ++c)
    for (int t = 0; t < Tseg; ++t) z_seg[c * Tseg + t] = z.data[c * Tfull + t];
  std::vector<float> target_seg(t40.begin(), t40.begin() + Lseg);

  training::MelSpecConfig mcfg;
  mcfg.n_fft = 1024; mcfg.hop = 256; mcfg.n_mels = 80; mcfg.sample_rate = 40000;
  training::MelLoss mel(mcfg);
  int Tm = 0;
  auto tgt_host = mel.target_log_mel(target_seg.data(), Lseg, Tm);
  auto tgt = Tensor::from_host(tgt_host, {Tm, mcfg.n_mels}, false);

  training::GeneratorTrainer gt; REQUIRE(gt.init(g_path, 0));
  ag::AdamW opt(gt.params(), /*lr=*/2e-4f);

  float first = 0.0f, last = 0.0f;
  const int steps = 20;
  for (int it = 0; it < steps; ++it) {
    auto zc = Tensor::from_host(z_seg, {192, Tseg}, false);
    auto hc = Tensor::from_host(har_seg, {1, Lseg}, false);
    auto out = gt.decode(zc, hc, Tseg);       // [1, Lseg]
    auto gm = mel.log_mel(out, Lseg);
    auto loss = mel.l1(gm, tgt, Tm);
    float lv = loss.to_host()[0];
    if (it == 0) first = lv;
    last = lv;
    ag::backward(loss);
    opt.step();
  }
  std::printf("[decoder-finetune] mel loss %.4f -> %.4f over %d steps\n", first, last, steps);

  // Fine-tuning must reduce the reconstruction loss toward the target voice.
  CHECK(last < first);
  CHECK(std::isfinite(last));
}
