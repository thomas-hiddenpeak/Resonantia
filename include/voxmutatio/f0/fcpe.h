#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "voxmutatio/io/safetensors.h"

namespace voxmutatio::f0 {

/// FCPE (Fast Context-based Pitch Estimation) configuration
struct FcpeConfig {
    std::string model_path;         // path to FCPE safetensors weights
    float threshold = 0.006f;       // confidence threshold for V/UV decision
};

/// F0 extractor using FCPE (fully-convolutional CFNaiveMelPE, real-time)
class FcpeExtractor {
public:
    /// Initialize and load model weights
    bool init(const FcpeConfig& config);

    /// Extract F0 contour from 16kHz mono PCM
    /// Returns F0 values in Hz (0.0 = unvoiced)
    std::vector<float> infer(const float* audio, int num_samples);

    /// Log-mel spectrogram [num_frames, 128] (exposed for alignment testing).
    std::vector<float> compute_mel(const float* audio, int num_samples,
                                   int& out_frames);

    /// Apply pitch shift (semitones) to F0 contour
    [[nodiscard]] static std::vector<float> pitch_shift(
        const std::vector<float>& f0, int semitones);

private:
    float threshold_ = 0.006f;
    bool loaded_ = false;
    io::SafetensorsLoader loader_;
    std::vector<float> mel_basis_;   // [128 * 513] librosa slaney, fmin=0
    std::vector<float> w_out_;       // [360 * 512] materialised weight-norm
    std::vector<float> b_out_;       // [360]
    std::vector<float> cent_table_;  // [360]
};

}  // namespace voxmutatio::f0
