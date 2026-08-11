// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// GPU STFT / iSTFT (cuFFT) — shared front-end for source-separation models and
// a GPU replacement for host spectrogram compute. Hann window, center padding,
// overlap-add reconstruction (spec 004).

#pragma once

#include <vector>

namespace voxmutatio::separation {

class Stft {
 public:
  // pad_override < 0 uses center default (n_fft/2 when center, else 0); otherwise
  // pads by exactly pad_override on each side (still reflect). VITS spec uses
  // pad_override = (n_fft - hop) / 2.
  Stft(int n_fft, int hop, int win_length = 0, bool center = true, int pad_override = -1);

  [[nodiscard]] int n_freq() const { return n_fft_ / 2 + 1; }
  [[nodiscard]] int num_frames(int L) const;

  /// Complex STFT of audio[L] (host in) -> re, im each [n_freq * T] freq-major; sets T.
  /// Invalid or CUDA-int-unrepresentable public dimensions throw
  /// std::invalid_argument before reading/allocating; execution failures throw
  /// std::runtime_error.
  void forward(const float* audio, int L, std::vector<float>& re,
               std::vector<float>& im, int& T) const;

  /// iSTFT of re, im [n_freq * T] (freq-major) -> audio[L_out] (host).
  /// Invalid, incompatible, or CUDA-int-unrepresentable dimensions throw
  /// std::invalid_argument before allocation; execution failures throw
  /// std::runtime_error.
  [[nodiscard]] std::vector<float> inverse(const std::vector<float>& re,
                                           const std::vector<float>& im,
                                           int T, int L_out) const;

  /// Magnitude spectrogram [n_freq * T] (freq-major), sqrt(re^2+im^2); sets T.
  [[nodiscard]] std::vector<float> magnitude(const float* audio, int L, int& T) const;

 private:
  int n_fft_, hop_, win_, pad_;
  bool center_;
  std::vector<float> window_;  // Hann (host)
};

}  // namespace voxmutatio::separation
