// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// MultiPeriodDiscriminator (RVC v2): 1 DiscriminatorS (1D, grouped convs) +
// 8 DiscriminatorP (2D period-reshaped convs, periods 2,3,5,7,11,17,23,37).
// Loads pretrained f0D*.safetensors; forward returns per-sub-D score + the
// intermediate feature maps used for feature-matching loss.

#pragma once

#include <string>
#include <vector>

#include "voxmutatio/autograd/tensor.h"

namespace voxmutatio::training {

struct SubDiscResult {
  std::vector<autograd::Tensor> fmaps;  // per-layer feature maps (post leaky)
  autograd::Tensor score;               // final map (flattened for LSGAN)
};

class Discriminator {
 public:
  bool init(const std::string& d_model_path);

  /// Run all sub-discriminators on audio [1, L]. Returns one result per sub-D.
  std::vector<SubDiscResult> forward(const autograd::Tensor& audio, int L);

  std::vector<autograd::Tensor>& params() { return params_; }

 private:
  struct Conv1dW { autograd::Tensor w, b; int cin, cout, k, stride, pad, groups; };
  struct Conv2dW { autograd::Tensor w, b; int cin, cout, kh, kw, sh, sw, ph, pw; };

  std::vector<Conv1dW> s_convs_;
  autograd::Tensor s_post_w_, s_post_b_;

  struct PDisc { int period; std::vector<Conv2dW> convs; autograd::Tensor post_w, post_b; };
  std::vector<PDisc> p_;

  std::vector<autograd::Tensor> params_;
};

}  // namespace voxmutatio::training
