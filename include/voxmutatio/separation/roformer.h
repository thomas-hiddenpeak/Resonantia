// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// MelBand-RoFormer (de-reverb / de-harmony) — pure C++/CUDA forward runner.
// STFT -> mel band-split -> depth x [time transformer, freq transformer]
// (RMSNorm + RoPE gated attention + FFN) -> per-band mask estimator ->
// scatter-average mask -> complex modulate -> iSTFT. Loads weights converted
// offline by tools/convert_roformer_weights.py. Spec 004 S6. Zero Python runtime.

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace voxmutatio::separation {

class Roformer {
 public:
  /// dir contains dereverb_roformer.safetensors + roformer_*.i64 band maps.
  explicit Roformer(const std::string& dir, const std::string& model = "dereverb_roformer");
  ~Roformer();
  Roformer(const Roformer&) = delete;
  Roformer& operator=(const Roformer&) = delete;

  [[nodiscard]] bool valid() const;
  [[nodiscard]] int n_fft() const;
  [[nodiscard]] int hop() const;
  [[nodiscard]] int sample_rate() const;

  /// Full separation of a stereo signal (planar [2, L]) -> stem [2, L].
  [[nodiscard]] std::vector<float> separate_stereo(const std::vector<float>& stereo, int L) const;
  /// Mono at sr -> mono stem at sr (resample to 44100, stereo, run, downmix).
  [[nodiscard]] std::vector<float> separate_mono(const float* audio, int L, int sr) const;

  // --- staged alignment hooks (return host tensors) ---
  std::vector<int> debug_offsets() const;  // per-band [off0..off59, din0..din59]
  std::vector<float> debug_bandsplit_in(const std::vector<float>& stereo, int L, int& T) const; // [T,7916]
  std::vector<float> debug_bandsplit(const std::vector<float>& stereo, int L, int& T) const; // [T,60,256]
  std::vector<float> debug_blocks(const std::vector<float>& stereo, int L, int& T) const;    // [T,60,256]
  std::vector<float> debug_blocks_from(const std::vector<float>& bandsplit, int T) const;    // [T,60,256]->[T,60,256]
  std::vector<float> debug_mask(const std::vector<float>& stereo, int L, int& T) const;      // [T,7916]

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace voxmutatio::separation
