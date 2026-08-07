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
  Stft(int n_fft, int hop, int win_length = 0, bool center = true);

  [[nodiscard]] int n_freq() const { return n_fft_ / 2 + 1; }
  [[nodiscard]] int num_frames(int L) const;

  /// Complex STFT of audio[L] (host in) -> re, im each [n_freq * T] freq-major; sets T.
  void forward(const float* audio, int L, std::vector<float>& re,
               std::vector<float>& im, int& T) const;

  /// iSTFT of re, im [n_freq * T] (freq-major) -> audio[L_out] (host).
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
