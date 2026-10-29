#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include "voxmutatio/core/pitch_edit.h"

namespace voxmutatio {

// ============================================================================
// Audio types
// ============================================================================

/// Audio sample format
enum class SampleFormat : std::uint8_t {
    kFloat32,  // float32 PCM, range [-1.0, 1.0]
    kInt16,    // int16 PCM
};

/// Audio buffer: single-channel PCM samples
struct AudioBuffer {
    std::vector<float> data;          // always float32 internally
    int sample_rate = 16'000;
    SampleFormat source_format = SampleFormat::kFloat32;

    [[nodiscard]] std::size_t num_samples() const noexcept {
        return data.size();
    }
    [[nodiscard]] double duration_s() const noexcept {
        return sample_rate > 0
            ? static_cast<double>(data.size()) / static_cast<double>(sample_rate)
            : 0.0;
    }
    [[nodiscard]] const float* ptr() const noexcept { return data.data(); }
    [[nodiscard]] float* ptr() noexcept { return data.data(); }
};

// ============================================================================
// Model metadata
// ============================================================================

/// RVC model version
enum class ModelVersion : std::uint8_t {
    kV1,  // 256-D HuBERT, SynthesizerTrnMs256NSFsid
    kV2,  // 768-D HuBERT, SynthesizerTrnMs768NSFsid
};

/// F0 extraction method
enum class F0Method : std::uint8_t {
    kRmvpe,  // RMVPE DeepUnet + bidirectional GRU (default)
    kFcpe,   // Optional Fast Context-based extractor; no end-to-end latency guarantee
    kPm,     // Reserved; current pipeline rejects this unsupported method
};

/// Reproducible sampled synthesis, or the legacy zero-noise numerical control.
enum class SynthesisMode : std::uint8_t { kSampled, kMean };

/// Content (SSL) encoder selection
enum class ContentEncoder : std::uint8_t {
    kHubert,  // HuBERT/ContentVec (default; stock RVC v1/v2 models)
    kWavlm,   // WavLM-Base+ (requires a WavLM-trained synthesizer)
};


// ============================================================================
// Configuration
// ============================================================================

/// Voice conversion configuration
struct VCConfig {
    // Optional manual correction; bounded file conversion only, <=600s input.
    std::vector<PitchEdit> pitch_edits;
    // Model paths
    std::string hubert_model_path;       // required when content_encoder=kHubert
    std::string synthesizer_model_path;  // VITS synthesizer safetensors
    std::string index_path;              // required when index_rate > 0
    std::string rmvpe_model_path;        // required when f0_method=kRmvpe
    std::string fcpe_model_path;         // required when f0_method=kFcpe
    std::string wavlm_model_path;        // required when content_encoder=kWavlm

    // Inference parameters
    int f0_up_key = 0;                   // pitch shift in semitones
    double formant_shift = 0.0;          // reserved; current pipeline requires 0
    double index_rate = 0.0;             // index retrieval blend [0.0, 1.0]
    F0Method f0_method = F0Method::kRmvpe;  // F0 extractor: kRmvpe (default) or kFcpe
    ContentEncoder content_encoder = ContentEncoder::kHubert;  // content SSL encoder
    double rms_mix_rate = 0.5;           // source/target RMS energy blend [0.0, 1.0]
    double protect = 0.5;                // unvoiced protection strength [0.0, 1.0]
    int filter_radius = 0;               // F0 median-filter kernel (0=off; odd >=3 smooths)
    double edge_pad_sec = 0.5;           // reflect-pad seconds each end (RVC x_pad; anti edge-artifact)
    bool apply_highpass = true;          // RVC 48Hz Butterworth high-pass on 16k input
    SynthesisMode synthesis_mode = SynthesisMode::kSampled;
    std::uint64_t synthesis_seed = 0;   // fixed draws per call; no persistent RNG state
    bool interp_unvoiced = false;        // preserve unvoiced F0; true explicitly enables legacy gap filling

    // Long-audio handling: clips longer than max_chunk_sec are split at silence and
    // converted per-chunk, then stitched. HuBERT/Index/VITS run per chunk, while F0
    // and host preprocessing still consume the whole clip and grow with its duration.
    double max_chunk_sec = 30.0;         // bounded to [1,30] seconds by the pipeline
    double chunk_pad_sec = 2.0;          // context padding each side of a chunk (trimmed after synth)
    double chunk_query_sec = 1.0;        // search radius to snap each cut to the quietest point

    // I/O parameters
    int target_sample_rate = 0;          // 0 or the model's native sample rate
    std::string output_format = "wav";   // WAV is the supported output format

    // Runtime
    std::string device = "cuda";         // CUDA is the supported backend
    bool use_half_precision = false;     // reserved; current pipeline requires false
    int gpu_device = 0;

    // Checkpoint contract selected by the caller and validated at initialization.
    ModelVersion version = ModelVersion::kV1;
    bool has_f0 = true;                  // only F0-conditioned checkpoints are supported
    int num_speakers = 1;                // compatibility hint; checkpoint is authoritative
    int model_sample_rate = 40'000;      // only 40 kHz checkpoints are currently supported
};

// ============================================================================
// Results
// ============================================================================

/// Result of a voice conversion operation
struct VCResult {
    AudioBuffer audio;                   // converted audio
    double hubert_ms = 0.0;             // HuBERT feature extraction time
    double f0_ms = 0.0;                 // F0 extraction time
    double index_ms = 0.0;              // index retrieval time
    double synth_ms = 0.0;              // synthesizer time
    double total_ms = 0.0;              // total pipeline time
    bool success = false;
    // Only bounded conversion with an explicit checkpoint can pause. A pause
    // publishes no output; success stays false and committed blocks survive.
    bool paused = false;
    std::string error_message;
};

// ============================================================================
// Error handling
// ============================================================================

/// Error codes
enum class ErrorCode : std::uint8_t {
    kSuccess,
    kInvalidInput,
    kModelLoadFailed,
    kDeviceError,
    kInferenceError,
    kIoError,
    kNotFound,
};

[[nodiscard]] std::string_view error_code_string(ErrorCode code) noexcept;

}  // namespace voxmutatio
