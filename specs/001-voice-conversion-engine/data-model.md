# Data Model: Voice Conversion Engine

**Date**: 2026-08-09
**Status**: As-built CUDA FP32 / WAV contract

## 1. Core Audio Types

```cpp
enum class SampleFormat : std::uint8_t {
    kFloat32,
    kInt16,
};

struct AudioBuffer {
    std::vector<float> data;       // mono float32 PCM in memory
    int sample_rate = 16'000;
    SampleFormat source_format = SampleFormat::kFloat32;
};
```

- `AudioBuffer` is a contiguous one-dimensional mono buffer. A multichannel WAV is downmixed
  by the file loader before entering the pipeline.
- Public conversion requires a positive sample rate, non-empty bounded sample count and finite
  float values.
- `source_format` records the source WAV representation; it does not change the internal
  float32 computation contract.
- The file API is WAV-only. No FLAC/MP3 decoder is implied by this type.

## 2. Model Selection Types

```cpp
enum class ModelVersion : std::uint8_t {
    kV1,  // 256-D HuBERT final projection
    kV2,  // 768-D HuBERT transformer representation
};

enum class F0Method : std::uint8_t {
    kRmvpe,  // default
    kFcpe,   // optional native CUDA implementation
    kPm,     // reserved; pipeline rejects it
};

enum class ContentEncoder : std::uint8_t {
    kHubert, // default
    kWavlm,  // optional; v2 and matching synthesizer required
};
```

枚举值可被源码表达不代表均已支持。特别是 `kPm` 是兼容/未来预留值，当前
pipeline 必须拒绝。

## 3. VCConfig

```cpp
struct VCConfig {
    std::string hubert_model_path;
    std::string synthesizer_model_path;
    std::string index_path;
    std::string rmvpe_model_path;
    std::string fcpe_model_path;
    std::string wavlm_model_path;

    int f0_up_key = 0;
    double formant_shift = 0.0;
    double index_rate = 0.0;
    F0Method f0_method = F0Method::kRmvpe;
    ContentEncoder content_encoder = ContentEncoder::kHubert;
    double rms_mix_rate = 0.5;
    double protect = 0.5;
    int filter_radius = 0;
    double edge_pad_sec = 0.5;
    bool apply_highpass = true;
    bool interp_unvoiced = true;

    double max_chunk_sec = 30.0;
    double chunk_pad_sec = 2.0;
    double chunk_query_sec = 1.0;

    int target_sample_rate = 0;
    std::string output_format = "wav";

    std::string device = "cuda";
    bool use_half_precision = false;
    int gpu_device = 0;

    ModelVersion version = ModelVersion::kV1;
    bool has_f0 = true;
    int num_speakers = 1;
    int model_sample_rate = 40'000;
};
```

### Initialization invariants

| Field | Accepted contract |
|---|---|
| `device` | exactly `"cuda"` |
| `use_half_precision` | exactly `false` |
| `has_f0` | exactly `true` |
| `output_format` | exactly `"wav"` |
| `target_sample_rate` | `0` or exactly `model_sample_rate` |
| `formant_shift` | finite and exactly `0.0` |
| `f0_up_key` | [-48, 48] |
| `index_rate`, `rms_mix_rate`, `protect` | finite [0, 1] |
| `filter_radius` | 0, or odd [3, 31] |
| `edge_pad_sec` | finite [0, 2] |
| `max_chunk_sec` | finite [1, 30] |
| `chunk_pad_sec` | finite [0, 5] |
| `chunk_query_sec` | finite [0, 5] and no more than half `max_chunk_sec` |
| `model_sample_rate` | exactly `40000` for the currently supported synthesizer architecture |

Encoder/model coupling:

- `kHubert + kV1` produces 256D; `kHubert + kV2` produces 768D.
- `kWavlm` is accepted only for v2 after validating a WavLM checkpoint and still requires a
  synthesizer trained for those features.
- `kFcpe` requires `fcpe_model_path`; default `kRmvpe` requires a valid RMVPE checkpoint.
- A non-empty `index_path` must load successfully and its feature dimension must match.
- `index_rate > 0` requires a non-empty compatible `index_path`; lookup failure is a
  conversion failure rather than an implicit no-index fallback.

`target_sample_rate` is a compatibility field, not an independent resampling request.

## 4. Result Type

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

Timing fields are per-call observations, not service-level guarantees. Callers must inspect
`success`; on failure, `error_message` describes the boundary that failed.

## 5. Pipeline Data Flow

```text
Input WAV (supported PCM/float encoding, any accepted sample rate/channels)
    -> mono AudioBuffer<float>
    -> 16 kHz resample + optional high-pass/edge padding
    -> HuBERT [T,256] (v1) or [T,768] (v2)
       OR optional WavLM [T,768] for a matching v2 model
    -> optional Flat Index blend [T,D]
    -> RMVPE(default) or FCPE [Tf] Hz/voicing
    -> interpolate/protect/coarse pitch
    -> F0 VITS synthesizer
    -> mono float32 at model_sample_rate
    -> WAV
```

长音频在 16 kHz 前端上进行全局 F0 处理，并按安静点分块；每块带上下文推理，
裁剪后拼接。HuBERT、Index 与 VITS 受单块大小约束；F0 仍对整段 padded clip 一次执行，
输入、全段前处理和完整输出也保留在 host memory，因此 host 与 F0 CPU/GPU working set
都随时长增长。该状态模型不定义流式协议。

## 6. Tensor and Weight Conventions

| Data | Shape | Layout | Runtime precision |
|---|---|---|---|
| audio | `[N]` | contiguous | FP32 |
| content features | `[T,D]` | row-major | FP32 |
| F0 | `[T]` | contiguous | FP32 |
| speaker embedding | model-specific | checked against checkpoint | FP32 |
| weights | model-specific | safetensors metadata + byte-range checked | FP32 |

运行时不支持 FP16。加载器必须在访问前验证 dtype、shape、乘积、offset、byte range
和需要参与 norm 的有限性，不能仅依据 tensor 名称信任 checkpoint。

## 7. State and Failure Semantics

- `VoiceConversionPipeline::init` 使用 staged initialization；失败会使对象回到未初始化
  状态，不保留旧模型或 mmap。
- `set_runtime_params` 只有在所有参数同时合法时才提交，否则保留旧值。
- buffer、CUDA allocation 和 cuBLAS handle 由 RAII 管理；异常路径不得泄漏前序资源。
- 文件入口在目标同目录创建唯一 staging 文件，完整写入、`fsync` 和 close 成功后以
  rename 原子替换目标；失败清理 staging 并保留旧目标。调用方仍必须以
  `VCResult::success` 判断交付，不能仅以目标文件存在作为成功依据。
