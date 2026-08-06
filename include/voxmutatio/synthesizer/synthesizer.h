#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "voxmutatio/core/types.h"

namespace voxmutatio::synthesizer {

/// VITS synthesizer configuration (auto-detected from checkpoint)
struct SynthesizerConfig {
    std::string model_path;           // path to safetensors weights

    // Architecture parameters (loaded from checkpoint config)
    int spec_channels = 1025;         // FFT/2 + 1
    int segment_size = 3200;          // training segment size
    int inter_channels = 192;
    int hidden_channels = 192;
    int filter_channels = 768;
    int n_heads = 2;
    int n_layers = 6;
    int kernel_size = 3;
    int p_dropout = 0;

    // HiFiGAN vocoder parameters
    std::string resblock = "1";       // "1" or "2"
    std::vector<int> resblock_kernel_sizes = {3, 7, 11};
    std::vector<std::vector<int>> resblock_dilation_sizes = {
        {1, 3, 5}, {1, 3, 5}, {1, 3, 5}
    };
    std::vector<int> upsample_rates = {5, 4, 4, 2, 2};
    int upsample_initial_channel = 512;
    std::vector<int> upsample_kernel_sizes = {11, 8, 8, 4, 4};

    // Speaker embedding
    int spk_embed_dim = 256;          // number of speakers
    int gin_channels = 256;

    // Sample rate
    int sample_rate = 40'000;         // "40k" = 40000

    // Model variant
    ModelVersion version = ModelVersion::kV1;
    bool has_f0 = true;
    bool half_precision = false;
};

/// VITS voice synthesizer with NSF (Harmonic + Noisy) vocoder
class Synthesizer {
public:
    /// Initialize and load model weights
    bool init(const SynthesizerConfig& config);

    /// Infer (synthesize audio from content features)
    /// For models with F0:
    ///   features: [1, T, 256/768], pitch: [1, T], pitchf: [1, T]
    /// For models without F0:
    ///   features: [1, T, 256/768], pitch/pitchf = nullptr
    /// speaker_id: speaker embedding index
    /// Returns synthesized audio at model's native sample rate
    AudioBuffer infer(const float* features, int frames,
                      const float* pitch, const float* pitchf,
                      int speaker_id);

    /// Streaming infer with skip/return windows (for real-time mode)
    AudioBuffer infer_stream(const float* features, int frames,
                             const float* pitch, const float* pitchf,
                             int speaker_id,
                             int skip_head, int return_length);

    /// Get model's native output sample rate
    [[nodiscard]] int sample_rate() const noexcept { return sample_rate_; }

    /// Get model version
    [[nodiscard]] ModelVersion version() const noexcept { return version_; }

    /// Get number of speakers
    [[nodiscard]] int num_speakers() const noexcept { return num_speakers_; }

    /// Check if model uses F0 conditioning
    [[nodiscard]] bool has_f0() const noexcept { return has_f0_; }

    /// Test hook: run TextEncoder only, return m_p [inter_channels, T].
    std::vector<float> debug_text_encoder(const float* features, int frames,
                                          const int* pitch_coarse);

    /// Test hook: run TextEncoder + Flow, return z [inter_channels, T].
    std::vector<float> debug_flow(const float* features, int frames,
                                  const int* pitch_coarse, int speaker_id);

    /// Test hook: sine source har [T*upp].
    std::vector<float> debug_har(const float* f0, int frames);

    /// Test hook: conv_pre + cond output [512, T].
    std::vector<float> debug_convpre(const float* z, int frames, int speaker_id);

    /// Test hook: generator intermediates. which: 0=ups0, 1=noiseconv0, 2=stage0.
    std::vector<float> debug_gen_stage0(const float* z, const float* f0,
                                        int frames, int speaker_id, int which);

private:
    SynthesizerConfig config_;
    int sample_rate_ = 40'000;
    ModelVersion version_ = ModelVersion::kV1;
    bool has_f0_ = true;
    int num_speakers_ = 1;
};

}  // namespace voxmutatio::synthesizer
