#pragma once

#include <string>
#include <vector>

#include "voxmutatio/core/types.h"
#include "voxmutatio/content/hubert_encoder.h"
#include "voxmutatio/f0/rmvpe.h"
#include "voxmutatio/index/cuda_flat_index.h"
#include "voxmutatio/synthesizer/synthesizer.h"

namespace voxmutatio::pipeline {

/// End-to-end voice conversion pipeline
///
/// Orchestrates: Audio I/O → Resample → Content → F0 → Index → Synthesizer → Resample → Output
class VoiceConversionPipeline {
public:
    /// Initialize all sub-modules from config
    bool init(const VCConfig& config);

    /// Convert a single audio file
    /// Loads input from file, runs full pipeline, writes output to file
    VCResult convert_file(const std::string& input_path,
                          const std::string& output_path,
                          int speaker_id = 0);

    /// Convert raw audio buffer
    VCResult convert_buffer(const AudioBuffer& input,
                            int speaker_id = 0);

    /// Update per-request conversion params without reloading models.
    void set_runtime_params(int f0_up_key, double formant_shift,
                            double index_rate, double rms_mix_rate, double protect,
                            int filter_radius = 3) {
        config_.f0_up_key = f0_up_key;
        config_.formant_shift = formant_shift;
        config_.index_rate = index_rate;
        config_.rms_mix_rate = rms_mix_rate;
        config_.protect = protect;
        config_.filter_radius = filter_radius;
    }

    /// Get config
    [[nodiscard]] const VCConfig& config() const noexcept { return config_; }

    /// Get model version (auto-detected)
    [[nodiscard]] ModelVersion version() const;

    /// Get number of speakers
    [[nodiscard]] int num_speakers() const;

private:
    VCConfig config_;

    // Sub-modules (initialized on demand)
    content::HubertEncoder hubert_encoder_;
    content::HubertConfig hubert_cfg_;

    f0::RmvpeExtractor f0_extractor_;
    f0::RmvpeConfig f0_cfg_;

    index::CudaFlatIndex feature_index_;

    synthesizer::Synthesizer synth_;
    synthesizer::SynthesizerConfig synth_cfg_;

    // Runtime state
    bool initialized_ = false;
    bool index_loaded_ = false;
};

}  // namespace voxmutatio::pipeline
