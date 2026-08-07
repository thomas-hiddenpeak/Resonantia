// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Trainable NSF-HiFiGAN decoder expressed on the autograd engine (spec 002).
// Loads pretrained dec.* weights as parameters; decode() builds a differentiable
// graph from a (frozen) latent z and harmonic source har.

#pragma once

#include <array>
#include <string>
#include <vector>

#include "voxmutatio/autograd/tensor.h"

namespace voxmutatio::training {

class GeneratorTrainer {
 public:
  /// Load pretrained generator (40k v2). speaker_id folds emb_g/cond into bias.
  bool init(const std::string& g_model_path, int speaker_id = 0);

  /// Differentiable decoder: z[192,T], har[1, T*upp] -> audio[1, T*upp].
  autograd::Tensor decode(const autograd::Tensor& z, const autograd::Tensor& har,
                          int T);

  /// Trainable parameters (for the optimizer).
  std::vector<autograd::Tensor>& params() { return params_; }

  [[nodiscard]] int upp() const { return 400; }

 private:
  autograd::Tensor conv_pre_w_, conv_pre_b_;
  std::array<autograd::Tensor, 4> ups_w_, ups_b_;
  std::array<autograd::Tensor, 4> noise_w_, noise_b_;
  std::array<std::array<autograd::Tensor, 3>, 12> c1w_, c1b_, c2w_, c2b_;
  autograd::Tensor conv_post_w_;
  std::vector<autograd::Tensor> params_;

  autograd::Tensor resblock(const autograd::Tensor& xn, int idx, int C, int L,
                            int kernel, const int* dil);
};

}  // namespace voxmutatio::training
