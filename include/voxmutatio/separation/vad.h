// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Silero VAD v5 (voice activity detection) — pure C++/CUDA forward runner, for
// speech-accurate slicing. STFT-conv -> 4 conv encoder (ReLU) -> LSTM cell ->
// classifier -> per-chunk speech probability. Reimplemented from the OFFICIAL
// snakers4/silero-vad (MIT); weights converted by tools/convert_vad_weights.py.

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace voxmutatio::separation {

class Vad {
 public:
  explicit Vad(const std::string& dir, const std::string& model = "silero_vad");
  ~Vad();
  Vad(const Vad&) = delete;
  Vad& operator=(const Vad&) = delete;

  [[nodiscard]] bool valid() const;
  [[nodiscard]] int sample_rate() const;  // 16000

  /// Speech probability per 512-sample chunk for 16 kHz mono audio[n].
  [[nodiscard]] std::vector<float> probs(const float* audio16k, int n) const;

  /// Speech segments [start,end) (in samples at the input sr) with hysteresis.
  /// Resamples to 16k internally. min_speech_ms / min_silence_ms / pad_ms shape
  /// the segmentation the way get_speech_timestamps does.
  [[nodiscard]] std::vector<std::pair<int, int>> segments(
      const float* audio, int n, int sr, float thresh = 0.5f,
      int min_speech_ms = 250, int min_silence_ms = 100, int pad_ms = 30) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace voxmutatio::separation
