#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace voxmutatio::content {

/// WavLM-Base+ encoder configuration
struct WavlmConfig {
    std::string model_path;         // path to safetensors weights
    int output_dim = 768;           // 768 (wavlm-base+)
    int num_layers = 12;            // encoder layers to use
    bool half_precision = false;
};

/// WavLM content feature extractor: 16kHz PCM → content vectors
class WavlmEncoder {
public:
    /// Initialize and load model weights
    bool init(const WavlmConfig& config);

    /// Extract features from audio (16kHz mono PCM)
    /// Returns [1, T, output_dim] row-major tensor
    std::vector<float> extract(const float* audio, int num_samples);

    /// Get output dimension
    [[nodiscard]] int output_dim() const noexcept { return output_dim_; }

private:
    int output_dim_ = 768;
    bool half_precision_ = false;
    // Model weights and GPU buffers (implementation details in .cu)
};

}  // namespace voxmutatio::content
