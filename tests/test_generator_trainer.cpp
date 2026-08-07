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
#include "voxmutatio/synthesizer/synthesizer.h"
#include "voxmutatio/training/generator_trainer.h"

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
