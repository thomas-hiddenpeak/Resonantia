// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// VITS PosteriorEncoder (enc_q) on the autograd engine: linear spectrogram ->
// pre conv -> WaveNet (16 gated dilated conv layers, speaker-conditioned) ->
// proj -> (m_q, logs_q); z = m_q + exp(logs_q) * eps. Loads pretrained enc_q.*.

#pragma once

#include <string>
#include <vector>

#include "voxmutatio/autograd/tensor.h"

namespace voxmutatio::io { class SafetensorsLoader; }

namespace voxmutatio::training {

/// Linear-magnitude spectrogram matching VITS (reflect pad (n_fft-hop)/2,
/// Hann window). Returns [n_freq, T] (freq-major), sets out_T = L/hop.
std::vector<float> compute_spec(const float* audio, int L, int n_fft, int hop,
                                int& out_T);

/// A single WaveNet layer's weights (weight_norm reconstructed, cond folded).
struct WNLayer {
  autograd::Tensor in_w, in_b;      // Conv1d(hidden, 2*hidden, k) + cond bias
  autograd::Tensor rs_w, rs_b;      // res_skip Conv1d(hidden, 2*hidden|hidden, 1)
  int rs_ch = 0;
};

/// Reusable WaveNet (used by enc_q and flow). Loads enc.* under a prefix.
class WaveNet {
 public:
  void load(const io::SafetensorsLoader& L, const std::string& prefix,
            int hidden, int n_layers, int kernel, int speaker_id,
            std::vector<autograd::Tensor>& params_out);
  /// x [hidden, T] -> [hidden, T].
  autograd::Tensor forward(const autograd::Tensor& x, int T) const;

 private:
  int hidden_ = 0, kernel_ = 0;
  std::vector<WNLayer> layers_;
};

class PosteriorEncoder {
 public:
  bool init(const std::string& g_model_path, int speaker_id = 0);

  /// spec [n_spec, T] constant -> stats. Fills m_q, logs_q [inter, T].
  /// z = m_q + exp(logs_q)*eps (eps=0 => mean). Returns z [inter, T].
  autograd::Tensor forward(const autograd::Tensor& spec, int T, bool sample,
                           autograd::Tensor& m_q, autograd::Tensor& logs_q);

  std::vector<autograd::Tensor>& params() { return params_; }

 private:
  static constexpr int kSpec = 1025;
  static constexpr int kHidden = 192;
  static constexpr int kInter = 192;
  autograd::Tensor pre_w_, pre_b_;
  WaveNet wn_;
  autograd::Tensor proj_w_, proj_b_;
  std::vector<autograd::Tensor> params_;
};

}  // namespace voxmutatio::training
