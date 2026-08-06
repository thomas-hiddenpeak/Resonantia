# Data Model: Voice Conversion Engine

**Date**: 2026-08-07

## Core Types

### AudioBuffer
```cpp
struct AudioBuffer {
    std::vector<float> data;          // float32 PCM, range [-1.0, 1.0]
    int sample_rate = 16'000;
    SampleFormat source_format = SampleFormat::kFloat32;
};
```
- **Invariants**: `data.size() > 0`, `sample_rate > 0`
- **Memory**: Contiguous row-major float32 array

### VCConfig
```cpp
struct VCConfig {
    // Model paths
    std::string hubert_model_path;
    std::string synthesizer_model_path;
    std::string index_path;
    std::string rmvpe_model_path;

    // Inference parameters
    int f0_up_key = 0;                // semitones [-12, +12]
    double formant_shift = 0.0;       // semitones
    double index_rate = 0.0;          // [0.0, 1.0]
    double rms_mix_rate = 0.5;        // [0.0, 1.0]
    double protect = 0.5;             // [0.0, 1.0]

    // Runtime
    std::string device = "cuda";
    bool use_half_precision = false;
    int gpu_device = 0;

    // Auto-detected
    ModelVersion version = ModelVersion::kV1;
    bool has_f0 = true;
    int num_speakers = 1;
    int model_sample_rate = 40'000;
};
```

### VCResult
```cpp
struct VCResult {
    AudioBuffer audio;
    double hubert_ms = 0.0;
    double f0_ms = 0.0;
    double index_ms = 0.0;
    double synth_ms = 0.0;
    double total_ms = 0.0;
    bool success = false;
    std::string error_message;
};
```

## Model Metadata

### ModelVersion
```cpp
enum class ModelVersion : std::uint8_t {
    kV1,  // 256-D HuBERT, SynthesizerTrnMs256NSFsid
    kV2,  // 768-D HuBERT, SynthesizerTrnMs768NSFsid
};
```

### F0Method
```cpp
enum class F0Method : std::uint8_t {
    kRmvpe,  // Conformer-based (offline)
    kFcpe,   // Fast Context-based (real-time)
    kPm,     // Parselmouth (CPU fallback)
};
```

## safetensors Tensor Format

### Tensor Metadata
```cpp
struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;  // bytes per dimension
    std::size_t data_offset;       // byte offset in mmap
    std::size_t data_nbytes;
};
```

### File Layout
```
[safetensors file]
├─ [8 bytes] uint64_t header_size (little-endian)
├─ [header_size bytes] JSON header
│  └─ { "tensor_name": {"dtype": "F32", "shape": [128, 768], "data_offsets": [0, 98304]}, ... }
└─ [data bytes] contiguous tensor data
```

## Pipeline Data Flow

```
Input WAV (任意采样率)
    │
    ▼
AudioBuffer (16kHz, mono, float32)  ← resample_linear()
    │
    ▼
std::vector<float> [T, 256/768]     ← HubertEncoder::extract()
    │
    ▼
std::vector<float> [T] Hz           ← RmvpeExtractor::infer()
    │
    ▼
std::pair<distances, indices>       ← CudaFlatIndex::search()
    │
    ▼
std::vector<float> [T, 256/768]    ← blend_features()
    │
    ▼
AudioBuffer (40kHz, mono, float32)  ← Synthesizer::infer()
    │
    ▼
Output WAV (40kHz or target_sr)
```

## Memory Layout Conventions

| Tensor | Shape | Layout | Precision |
|--------|-------|--------|-----------|
| Audio Input | [N] | Row-major | FP32 |
| Content Features | [T, D] | Row-major | FP32 |
| F0 Contour | [T] | Row-major | FP32 |
| Pitch (stretched) | [T * hop] | Row-major | FP32 |
| Speaker Embedding | [S, D] | Row-major | FP32 |
| Model Weights | Varies | Row-major | FP32/FP16 |

**Note**: All tensors use row-major (C-style) layout for consistency with PyTorch defaults.
