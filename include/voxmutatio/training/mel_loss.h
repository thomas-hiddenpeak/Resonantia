// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Differentiable log-mel spectrogram + L1 loss for training (spec 002).

#pragma once

#include <vector>

#include "voxmutatio/autograd/tensor.h"

namespace voxmutatio::training {

struct MelSpecConfig {
  int n_fft = 1024;
  int hop = 256;
  int n_mels = 80;
  int sample_rate = 40000;
  float fmin = 0.0f;
  float fmax = 0.0f;  // 0 => sample_rate/2
};

/// Differentiable log-mel spectrogram. The Hann window is folded into the
/// (constant) DFT matrices, so log_mel() is a pure autograd graph over audio.
class MelLoss {
 public:
  explicit MelLoss(const MelSpecConfig& cfg);

  /// Number of frames for a signal of length L (no center padding).
  [[nodiscard]] int num_frames(int L) const;

  /// Differentiable log-mel of audio [1, L] -> [T, n_mels].
  [[nodiscard]] autograd::Tensor log_mel(const autograd::Tensor& audio, int L) const;

  /// Constant target log-mel (host) from raw audio; sets out_T.
  [[nodiscard]] std::vector<float> target_log_mel(const float* audio, int L,
                                                  int& out_T) const;

  /// L1 mel loss between a differentiable log-mel and a constant target.
  [[nodiscard]] autograd::Tensor l1(const autograd::Tensor& gen_logmel,
                                    const autograd::Tensor& target_const,
                                    int T) const;

  [[nodiscard]] const MelSpecConfig& config() const { return cfg_; }

 private:
  MelSpecConfig cfg_;
  int n_freq_;
  autograd::Tensor cosw_;         // [n_fft, n_freq] windowed DFT cos
  autograd::Tensor sinw_;         // [n_fft, n_freq] windowed DFT -sin
  autograd::Tensor mel_basis_t_;  // [n_freq, n_mels]
};

}  // namespace voxmutatio::training
