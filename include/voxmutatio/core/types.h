#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

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
    kRmvpe,  // Conformer-based (default)
    kFcpe,   // Fast Context-based (real-time)
    kPm,     // Parselmouth (CPU fallback)
};

// ============================================================================
// Configuration
// ============================================================================

/// Voice conversion configuration
struct VCConfig {
    // Model paths
    std::string hubert_model_path;       // HuBERT safetensors directory
    std::string synthesizer_model_path;  // VITS synthesizer safetensors
    std::string index_path;              // CUDA Flat Index file (optional)
    std::string rmvpe_model_path;        // RMVPE safetensors (optional)

    // Inference parameters
    int f0_up_key = 0;                   // pitch shift in semitones
    double formant_shift = 0.0;          // formant shift in semitones
    double index_rate = 0.0;             // index retrieval blend [0.0, 1.0]
    double rms_mix_rate = 0.5;           // source/target RMS energy blend [0.0, 1.0]
    double protect = 0.5;                // unvoiced protection strength [0.0, 1.0]
    int filter_radius = 0;               // F0 median-filter kernel (0=off; odd >=3 smooths)

    // I/O parameters
    int target_sample_rate = 0;          // 0 = keep model's native sample rate
    std::string output_format = "wav";   // wav, flac

    // Runtime
    std::string device = "cuda";         // cuda, cpu
    bool use_half_precision = false;     // FP16 inference
    int gpu_device = 0;

    // Model metadata (auto-detected from checkpoint)
    ModelVersion version = ModelVersion::kV1;
    bool has_f0 = true;                  // whether the model uses F0 conditioning
    int num_speakers = 1;
    int model_sample_rate = 40'000;      // native model output sample rate
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
