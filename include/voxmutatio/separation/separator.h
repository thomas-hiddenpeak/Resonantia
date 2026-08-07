// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Vocal separator (Open-Unmix umxhq) — pure C++/CUDA forward-only runner.
// STFT -> per-frame FC/BatchNorm -> 3-layer bidirectional LSTM -> mask ->
// mixture-phase iSTFT. Loads models/separation/umxhq_vocals.safetensors
// (converted offline by tools/convert_separation_weights.py). Spec 004 S2.
// Zero Python at runtime.

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace voxmutatio::separation {

class Separator {
 public:
  /// Loads umxhq vocals weights from a safetensors file (F32).
  explicit Separator(const std::string& weights_path);
  ~Separator();

  Separator(const Separator&) = delete;
  Separator& operator=(const Separator&) = delete;

  [[nodiscard]] bool valid() const;
  [[nodiscard]] int n_fft() const;
  [[nodiscard]] int hop() const;
  [[nodiscard]] int sample_rate() const;    // 44100
  [[nodiscard]] int nb_output_bins() const;  // 2049

  /// Neural mask model on mixture magnitude, layout [nb_channels=2, nb_bins=2049,
  /// nb_frames] (channel-major). Returns estimated vocal magnitude, same layout.
  [[nodiscard]] std::vector<float> run_model(const std::vector<float>& mix_mag,
                                             int nb_frames) const;

  /// Full separation of a stereo signal (planar [2, Lch]) -> vocal [2, Lch].
  [[nodiscard]] std::vector<float> separate_stereo(const std::vector<float>& stereo,
                                                   int Lch) const;

  /// Convenience: mono audio at sr -> mono vocal at sr (resamples to 44100 and
  /// back; duplicates to stereo for the model, downmixes the result).
  [[nodiscard]] std::vector<float> separate_mono(const float* audio, int L, int sr) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace voxmutatio::separation
