#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace voxmutatio::f0 {

/// RMVPE (Robust Model-based Vocal Pitch Estimation) configuration
struct RmvpeConfig {
    std::string model_path;         // path to RMVPE safetensors weights
    float threshold = 0.03f;        // confidence threshold for V/UV decision
};

/// F0 extractor using RMVPE Conformer encoder (offline/high-quality)
class RmvpeExtractor {
public:
    /// Initialize and load model weights
    bool init(const RmvpeConfig& config);

    /// Extract F0 contour from 16kHz mono PCM
    /// Returns F0 values in Hz (0.0 = unvoiced)
    std::vector<float> infer(const float* audio, int num_samples);

    /// Apply pitch shift (semitones) to F0 contour
    [[nodiscard]] static std::vector<float> pitch_shift(
        const std::vector<float>& f0, int semitones);

private:
    float threshold_ = 0.03f;
    // Model weights and GPU buffers (implementation details in .cu)
};

}  // namespace voxmutatio::f0
