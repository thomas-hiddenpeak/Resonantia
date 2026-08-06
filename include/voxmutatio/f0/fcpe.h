#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace voxmutatio::f0 {

/// FCPE (Fast Context-based Pitch Estimation) configuration
struct FcpeConfig {
    std::string model_path;         // path to FCPE safetensors weights
    float threshold = 0.03f;        // confidence threshold for V/UV decision
};

/// F0 extractor using FCPE (real-time/low-latency)
class FcpeExtractor {
public:
    /// Initialize and load model weights
    bool init(const FcpeConfig& config);

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
