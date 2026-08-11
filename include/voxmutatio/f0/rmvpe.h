#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace voxmutatio::f0 {

/// RMVPE (Robust Model-based Vocal Pitch Estimation) configuration
struct RmvpeConfig {
    std::string model_path;         // path to RMVPE safetensors weights
    float threshold = 0.03f;        // finite confidence threshold in [0, 1]
};

/// F0 extractor using the RMVPE DeepUnet + bidirectional-GRU architecture.
class RmvpeExtractor {
public:
    /// Copies share the immutable, validated checkpoint mapping while keeping
    /// their own threshold and initialization handle.
    RmvpeExtractor() = default;
    RmvpeExtractor(const RmvpeExtractor&) = default;
    RmvpeExtractor& operator=(const RmvpeExtractor&) = default;
    RmvpeExtractor(RmvpeExtractor&&) noexcept = default;
    RmvpeExtractor& operator=(RmvpeExtractor&&) noexcept = default;

    /// Load and validate every required model tensor. A failed call leaves the
    /// extractor uninitialized, including after a successful earlier init().
    bool init(const RmvpeConfig& config);

    /// Extract F0 contour from finite 16kHz mono PCM. Returns an empty vector
    /// for an uninitialized extractor or invalid input.
    /// F0 values are in Hz (0.0 = unvoiced).
    std::vector<float> infer(const float* audio, int num_samples);

    /// Apply a pitch shift in [-48, 48] semitones. Returns empty for a
    /// non-finite contour, an unsupported shift, or a non-finite result.
    [[nodiscard]] static std::vector<float> pitch_shift(
        const std::vector<float>& f0, int semitones);

    /// Extract the raw salience map [T, 360] (for alignment testing). On
    /// failure, returns an empty vector and sets out_frames to zero.
    std::vector<float> infer_salience(const float* audio, int num_samples,
                                      int& out_frames);

private:
    struct ModelState;

    void reset() noexcept;

    float threshold_ = 0.03f;
    std::shared_ptr<const ModelState> state_;
};

}  // namespace voxmutatio::f0
