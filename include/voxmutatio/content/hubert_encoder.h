#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace voxmutatio::content {

/// HuBERT encoder configuration (legacy RVC v1/v2 compatibility)
struct HubertConfig {
    std::string model_path;         // path to safetensors weights
    int output_dim = 256;           // 256 (v1) or 768 (v2)
    int num_layers = 12;            // encoder layers to use
    bool use_final_proj = false;    // v1 uses final_proj after layer 9
    bool half_precision = false;
};

/// HuBERT feature extractor: 16kHz PCM → content vectors
class HubertEncoder {
public:
    /// Initialize and load model weights
    bool init(const HubertConfig& config);

    /// Extract features from audio (16kHz mono PCM)
    /// Returns [1, T, output_dim] row-major tensor
    std::vector<float> extract(const float* audio, int num_samples);

    /// Get output dimension
    [[nodiscard]] int output_dim() const noexcept { return output_dim_; }

private:
    int output_dim_ = 256;
    bool half_precision_ = false;
    // Model weights and GPU buffers (implementation details in .cu)
};

}  // namespace voxmutatio::content
