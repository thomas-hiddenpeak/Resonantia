#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "voxmutatio/core/types.h"
#include "voxmutatio/content/hubert_encoder.h"
#include "voxmutatio/content/wavlm_encoder.h"
#include "voxmutatio/f0/rmvpe.h"
#include "voxmutatio/f0/fcpe.h"
#include "voxmutatio/index/cuda_flat_index.h"
#include "voxmutatio/synthesizer/synthesizer.h"

namespace voxmutatio::pipeline {

/// End-to-end voice conversion pipeline
///
/// Orchestrates: Audio I/O → Resample → Content → F0 → Index → Synthesizer → Resample → Output
/// A pipeline instance is not thread-safe. Callers must externally serialize
/// init(), set_runtime_params(), and all conversion calls on the same instance.
class VoiceConversionPipeline {
public:
    /// Initialize all sub-modules from config.
    ///
    /// Validation failures and recoverable model/runtime exceptions return
    /// false and leave the whole pipeline uninitialized, including when
    /// replacing a previous configuration. A host allocation failure may
    /// propagate when even the reset/result state cannot be constructed.
    bool init(const VCConfig& config);

    /// Convert a single audio file
    /// Loads input from file, runs full pipeline, writes output to file
    /// Recoverable model, I/O, CUDA, and runtime exceptions are returned as
    /// success=false with a non-empty diagnostic when a VCResult can be
    /// constructed. A host allocation failure may propagate.
    VCResult convert_file(const std::string& input_path,
                          const std::string& output_path,
                          int speaker_id = 0);

    /// Convert a seekable WAV with a duration-independent process working set.
    ///
    /// The converted 40 kHz PCM16 WAV is delivered through an atomic file
    /// replacement. On success the returned AudioBuffer deliberately owns no
    /// samples; its sample_rate records the model rate. This synchronous
    /// offline API does not change the owning behavior of convert_file() or
    /// convert_buffer(), and does not provide a realtime/session contract.
    /// The bounded path currently accepts qualified HuBERT profiles; a WavLM
    /// configuration fails before payload I/O until a matching WavLM-trained
    /// synthesizer asset has passed the bounded numerical gates. Legacy APIs
    /// retain their existing configuration behavior.
    /// Recoverable model, I/O, CUDA, and runtime exceptions are returned as
    /// success=false with a non-empty diagnostic when a VCResult can be
    /// constructed; a host allocation failure may propagate. Any failure
    /// before the atomic rename commit keeps an existing target unchanged.
    VCResult convert_file_bounded(const std::string& input_path,
                                  const std::string& output_path,
                                  int speaker_id = 0);

    /// Convert raw audio buffer.
    /// Recoverable model, CUDA, and runtime exceptions are returned as
    /// success=false with a non-empty diagnostic when a VCResult can be
    /// constructed. A host allocation failure may propagate.
    VCResult convert_buffer(const AudioBuffer& input,
                            int speaker_id = 0);

    /// Update per-request conversion params without reloading models.
    /// Returns false and leaves the previous values unchanged when any value is
    /// unsupported, non-finite, or outside its documented range. Once the
    /// pipeline is initialized, a non-zero index rate also requires a loaded,
    /// feature-width-compatible index.
    bool set_runtime_params(int f0_up_key, double formant_shift,
                            double index_rate, double rms_mix_rate, double protect,
                            int filter_radius = 3) noexcept;

    /// Get config
    [[nodiscard]] const VCConfig& config() const noexcept { return config_; }

    /// Get the caller-selected model version validated against the checkpoint.
    [[nodiscard]] ModelVersion version() const;

    /// Get number of speakers
    [[nodiscard]] int num_speakers() const;

    /// Get the content feature width derived from the configured RVC version.
    [[nodiscard]] int feature_dim() const noexcept { return feature_dim_; }

private:
    /// Initialize a fresh staging object; public init() commits it atomically.
    bool init_in_place(const VCConfig& config);

    /// Extract F0 with the configured method (RMVPE default, FCPE if selected+loaded).
    std::vector<float> extract_f0(const float* audio, int num_samples);

    /// Extract content features with the configured encoder (HuBERT default,
    /// WavLM if selected+loaded). Returns [T, feature_dim_] row-major.
    std::vector<float> extract_content(const float* audio, int num_samples);

    /// Content → synth for one 16k mono segment using a precomputed F0/UV slice
    /// (both at 100Hz / hop=160). Runs HuBERT → index → 2x-interp → protect →
    /// coarse-pitch → VITS. Timing is accumulated into `timing`.
    AudioBuffer synth_segment(const float* seg16k, int segn,
                              const float* f0_slice, const std::uint8_t* uv_slice,
                              int f0_len, int speaker_id, VCResult& timing,
                              std::uint64_t sampling_frame_offset = 0);

    /// Long-audio path: high-pass + reflect-pad the whole clip, compute F0 once
    /// (global pitch consistency), split at silence minima, convert each chunk via
    /// synth_segment, trim the context pad, and stitch. HuBERT/Index/VITS are
    /// chunked; whole-clip host preprocessing and F0 working memory are not bounded.
    VCResult convert_long(std::vector<float>& audio16k, int speaker_id);

    VCConfig config_;

    // Sub-modules (initialized on demand)
    content::HubertEncoder hubert_encoder_;
    content::HubertConfig hubert_cfg_;

    content::WavlmEncoder wavlm_encoder_;
    content::WavlmConfig wavlm_cfg_;
    bool wavlm_loaded_ = false;

    f0::RmvpeExtractor f0_extractor_;
    f0::RmvpeConfig f0_cfg_;

    f0::FcpeExtractor fcpe_extractor_;
    f0::FcpeConfig fcpe_cfg_;
    bool fcpe_loaded_ = false;

    index::CudaFlatIndex feature_index_;

    synthesizer::Synthesizer synth_;
    synthesizer::SynthesizerConfig synth_cfg_;

    // Runtime state
    bool initialized_ = false;
    bool index_loaded_ = false;
    int feature_dim_ = 256;
};

}  // namespace voxmutatio::pipeline
