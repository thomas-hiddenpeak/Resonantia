#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "voxmutatio/io/safetensors.h"

namespace voxmutatio::f0 {

/// FCPE (Fast Context-based Pitch Estimation) configuration
struct FcpeConfig {
    std::string model_path;         // path to FCPE safetensors weights
    float threshold = 0.006f;       // finite confidence threshold in [0, 1]
};

/// F0 extractor using fully-convolutional CFNaiveMelPE. This API does not
/// provide a real-time latency guarantee.
class FcpeExtractor {
public:
    /// Load and validate every required model tensor. A failed call leaves the
    /// extractor uninitialized, including after a successful earlier init().
    bool init(const FcpeConfig& config);

    /// Extract F0 contour from finite 16kHz mono PCM. Returns an empty vector
    /// for an uninitialized extractor or invalid input.
    /// F0 values are in Hz (0.0 = unvoiced).
    std::vector<float> infer(const float* audio, int num_samples);

    /// Log-mel spectrogram [num_frames, 128] (exposed for alignment testing).
    /// On failure, returns an empty vector and sets out_frames to zero.
    std::vector<float> compute_mel(const float* audio, int num_samples,
                                   int& out_frames);

    /// Apply a pitch shift in [-48, 48] semitones. Returns empty for a
    /// non-finite contour, an unsupported shift, or a non-finite result.
    [[nodiscard]] static std::vector<float> pitch_shift(
        const std::vector<float>& f0, int semitones);

private:
    void reset() noexcept;

    float threshold_ = 0.006f;
    bool loaded_ = false;
    io::SafetensorsLoader loader_;
    std::vector<float> mel_basis_;   // [128 * 513] librosa slaney, fmin=0
    std::vector<float> w_out_;       // [360 * 512] materialised weight-norm
    std::vector<float> b_out_;       // [360]
    std::vector<float> cent_table_;  // [360]
};

}  // namespace voxmutatio::f0
