# Approved Semantic Data Model: Realtime Voice Conversion

**Date**: 2026-08-10
**Status**: Approved semantic contract; names and layouts are not current public API
**Governing spec**: [spec.md](spec.md)
**Architecture decision**:
[ADR-0002](../../docs/adr/0002-persistent-streaming-runtime-and-local-audio-boundary.md)

本文描述已批准但尚未实现的语义，不承诺最终 C++ 拼写。在实现和门禁完成前，不得在
CLI、public header 或状态文档中把这些类型描述为当前能力。

## 1. Qualified profile

首版不把任意 `VCConfig` 组合带入实时路径，而是只接受一个显式 profile：

```cpp
enum class RealtimeProfile : std::uint8_t {
    kOrinV2HubertFcpe40kIndexed,
};
```

`kOrinV2HubertFcpe40kIndexed` 的已批准不变量：

| Field | Required value |
|---|---|
| platform/backend | Jetson Orin Linux / CUDA FP32 |
| model version | `ModelVersion::kV2` |
| content encoder | `ContentEncoder::kHubert`，768D |
| F0 | `F0Method::kFcpe` |
| synthesizer | F0 VITS v2，native 40 kHz |
| index | 有效 Flat Index、`index_rate > 0`；具体 rate、rows 上限和执行 backend 由 R0 数据固化 |
| speaker | checkpoint range 内固定值或 chunk-boundary snapshot |
| unsupported fields | CPU、FP16、PM、WavLM、RMVPE、formant shift、non-F0 均拒绝 |

现有本地 selected weights 的文件大小合计约 541 MiB（HuBERT 378,298,232 bytes、FCPE
43,339,596 bytes、VITS v2 G 145,897,024 bytes）。这只是文件 payload 量，不是显存或
进程峰值保证；Index、workspace、allocator、CUDA context 和音频环必须单独测量。R0 已有
standalone CPU exact reference 与 persistent-GPU exact retrieval 的
synthetic/available-feature 诊断，
但仍没有代表性目标声线 Index 的 full-pipeline realtime latency/memory/quality baseline；这些
数据不能选择产品 backend、支持上限或 SLO。

no-index 不是上述 qualified profile 的等价变体，只是显式诊断/兼容模式，或用户在启动时
明确授权的 overload 降级目标。其运行结果不得计入 Index-enabled profile 的质量或 SLO 通过。

## 2. Realtime configuration

概念配置如下：

```cpp
struct RealtimeConfig {
    RealtimeProfile profile = RealtimeProfile::kOrinV2HubertFcpe40kIndexed;
    VCConfig conversion;

    std::string index_path;  // required by the qualified profile
    RetrievalMode retrieval_mode = RetrievalMode::kIndexRequired;
    IndexFallbackPolicy index_fallback_policy = IndexFallbackPolicy::kDisabled;

    std::string capture_device = "default";
    std::string playback_device = "default";

    int capture_sample_rate = 48'000;
    int playback_sample_rate = 48'000;
    int channels = 1;
    SampleFormat device_format = SampleFormat::kInt16;

    int device_period_ms = 10;
    int emit_hop_ms = 0;    // 20 ms aligned; resolved by the qualified profile
    int window_ms = 0;       // resolved by the qualified profile
    int lookahead_ms = 0;    // resolved by the qualified profile
    int crossfade_ms = 0;    // resolved by the qualified profile

    std::size_t input_ring_hops = 0;   // non-zero fixed profile capacity
    std::size_t output_ring_hops = 0;  // non-zero fixed profile capacity
};
```

概念语义（最终枚举拼写由接口实现时确定）：

- `RetrievalMode::kIndexRequired` 是默认正常质量模式：`index_path` 必须有效且
  `conversion.index_rate > 0`。
- `RetrievalMode::kNoIndexDiagnostic` 必须由用户显式选择：不得打开 Index，且
  `conversion.index_rate == 0`；它不是 qualified/competitive profile。
- `IndexFallbackPolicy::kDisabled` 禁止运行中旁路 Index。Index 错误或超预算仍按 error/
  overload 契约处理。
- 只有启动配置显式选择的 `kAllowStickyNoIndexOnSustainedOverload` 才授权在 R0 固化的触发
  条件后，于完整 chunk boundary 一次性降级。降级后在本 session 内不自动恢复 Index；
  `reset/reinitialize` 才能重新进入 Index-enabled 状态。

本文不选择具体 `index_rate`、Index rows 上限、deadline 次数或检索算法。这些值必须由 R0
使用真实目标声线 Index 的硬数据决定并回写契约。

上述数值中的 48 kHz/10 ms 是首版 ALSA 请求的 proposed defaults；startup 必须记录实际
协商值，并在不属于批准集合时失败。不得依赖未报告的 ALSA plug conversion。

### Initialization invariants

- 所有 rate、period、hop、window、lookahead、crossfade 和 capacity 均为正、有限、可检查；
  byte/sample/frame 乘法不得溢出。
- `emit_hop_ms` 是 20 ms 的正整数倍；内部 F0/VITS frame 仍以 10 ms 对齐。
- `window_ms >= emit_hop_ms + lookahead_ms`，emit 区间非空。
- crossfade 不超过 emitted audio 的一半；input/output ring 必须至少容纳 profile 的完整
  algorithmic context，且有固定最大值。
- capture/playback 是 mono。若硬件只能提供不同 channel/layout，adapter 显式转换并报告，
  或启动失败；不能把多声道 interleaved data 当 mono。
- `conversion` 必须满足 qualified profile 和 Spec 001 的共同不变量；保留字段的非默认值
  不得静默忽略。
- Index-required 时必须在打开 ALSA 前验证 header、hash、rows、dim、`k`、`index_rate`、
  byte ranges 与 profile compatibility；缺失、损坏或超过 R0 上限时启动失败，不得静默转为
  no-index。no-index diagnostic 时反向拒绝非零 rate 或含混的 Index 配置。

resolved window/lookahead/crossfade 是 profile contract，而不是首版自由调节的用户 flag。
R3 benchmark 选定后将具体值写回本文件和 CLI help，避免在用户面暴露未经验证的组合。

## 3. Audio chunk and preallocated storage

概念上的 chunk metadata：

```cpp
struct AudioChunkView {
    std::span<const float> samples;  // mono normalized PCM, borrowed for this call only
    int sample_rate = 0;
    std::uint64_t sequence = 0;
    std::int64_t capture_time_ns = 0;  // monotonic clock domain
    bool concealed = false;
};
```

- `samples` 不得被 session 保留到调用返回后；capture ring slot 是实际 owner。
- 每个 ring 在 init 时一次性分配固定数量、固定最大样本数的 slots；稳态只移动 slot index 和
  metadata，不增长 `std::vector`。
- sequence 单调递增。drop-oldest 会产生可观测 gap；禁止重编号来隐藏丢块。
- timestamp 必须来自 monotonic clock domain，并记录与 ALSA hardware timestamp 的换算。
- concealment slot 与真实转换 slot 有明确 flag，不能计入模型成功样本。

## 4. Runtime and session ownership

### PersistentInferenceRuntime

批准的 runtime 语义是 move-only、不可复制、首版不共享的 session-owned 对象：

```text
PersistentInferenceRuntime
├── validated model metadata
├── selected device-resident HuBERT weights
├── selected device-resident FCPE weights
├── selected device-resident VITS weights
├── validated session-owned Flat Index mapping/execution state
├── one owned CUDA stream
├── stream-bound cuBLAS/cuFFT handles
├── profile-sized device workspaces
└── CUDA timing events
```

- host safetensors mapping 只在 staged validation/upload 期间存在；commit 后不依赖隐藏 static
  path cache。需要保留的 mmap 必须有显式 owner 和计入内存报告。
- 只有所有 weights、handles、workspace 和 warm-up 成功后才提交 runtime。
- Index backend 可以是 host mmap、批处理并行或 device-resident execution，但必须由 R0 的
  latency/memory/quality evidence 选择；无论选择什么，都保持 exact weighted k-NN reference
  语义，并显式计入 mapping、workspace 与 resident memory。
- 一个 runtime 首版只服务一个 session/worker；不预建多 session sharing 或锁体系。
- weight buffers immutable；workspace 和 stream mutable，因而 session 非线程安全。

### StreamingSession

```text
StreamingSession
├── PersistentInferenceRuntime
├── input/output SPSC rings
├── input and output resampler phase/history
├── high-pass/filter state
├── HuBERT sliding context and emitted-frame cursor
├── FCPE mel/F0/UV/filter/protect context
├── retrieval mode, Index query state and sticky fallback generation
├── VITS overlap tail and NSF harmonic phase
├── current immutable parameter snapshot
├── lifecycle/error state
└── monotonic counters and latency histograms
```

ALSA handles、capture/playback threads 和 signal handling 不属于 `StreamingSession`；它们由
`vc_realtime` app adapter 拥有。模型层不得 include ALSA headers。

## 5. Lifecycle state machine

```text
Uninitialized --init+warmup--> Ready --start--> Running --request_stop--> Draining
      ^                           |                 |                         |
      |                           | reset           | reset                   | queues empty
      |                           v                 v                         v
      +------------------------- Ready <--------- Ready                    Stopped

Any staged/active fatal failure -----------------------------------------> Failed
Failed --reset/reinitialize only-----------------------------------------> Uninitialized/Ready
```

语义：

- `init` transactional；失败回到 `Uninitialized`，不得保留旧 runtime 或部分 device weights。
- `start` 只允许从 `Ready`；重复 start 稳定失败。
- `request_stop` 不在 signal handler 中同步释放资源，只设置无锁 stop request。
- `drain` 停止接收新 input，输出已接受且可完成的 data，然后进入 `Stopped`。
- `reset` 清空全部 DSP/model context、rings、phase 和 current error；是否保留 immutable weights
  由最终 API 决定，但不能保留部分 mutable state。
- `Failed` 后 `push`/`pull` 不返回普通成功；调用方必须读取 fatal error 并 reset/reinitialize。

## 6. Processing and result types

概念状态码：

```cpp
enum class StreamStatus : std::uint8_t {
    kAccepted,
    kNeedMoreInput,
    kOutputAvailable,
    kInputDropped,
    kOutputConcealed,
    kDrained,
    kInvalidState,
    kFatal,
};

enum class StreamError : std::uint8_t {
    kNone,
    kInvalidConfig,
    kModelValidation,
    kIndexValidation,
    kIndexDeadline,
    kDeviceNegotiation,
    kCudaInference,
    kDeadlineMiss,
    kOverload,
    kDeviceRemoved,
};
```

`deadline miss` 本身可以是可恢复统计事件；只有达到 profile 的连续/累计上限或伴随 CUDA/
device failure 时才进入 `Fatal`。Index 超预算默认仍走 deadline/overload 契约；只有启动时
预授权 fallback 才能触发粘滞 no-index transition。调用方不得仅依据 output samples 非空
判断成功。

运行参数使用完整 immutable snapshot：

```cpp
struct RealtimeParameters {
    int speaker_id = 0;
    int f0_up_key = 0;
    double protect = 0.5;
    std::uint64_t generation = 0;
};
```

更新先校验 speaker/pitch/protect 全部字段，在下一个 emit boundary 一次性替换；失败保留旧
generation 和所有值。

Index path、`index_rate`、retrieval mode 与 fallback authorization 不属于普通运行参数；首版
不允许任意热切换。唯一运行中变化是预授权 policy 触发的 Index-enabled → no-index 粘滞
降级，并产生独立 generation/event。不得自动 no-index → Index-enabled 或更换 Index 文件。

## 7. Queue and overload model

首版采用两个固定 SPSC rings：

- capture producer → GPU worker consumer；
- GPU worker producer → playback consumer。

input full 时，worker/capture boundary 丢弃最旧的**完整 hop**再写入最新 hop，增加
`input_hops_dropped` 并保留 sequence gap。不得等待 GPU 造成 capture thread 阻塞，也不得只
覆盖部分 hop。

output empty 时，playback 产生固定长度、带短 crossfade 的 silence/concealment slot，增加
`output_hops_concealed` 和 `playback_underruns`。concealment 不是模型输出，不得计入质量通过。

nominal acceptance run 要求这些计数均为 0；overload test 才要求计数与注入精确一致。

Index 超预算不改变队列基本政策：未授权 fallback 时继续按 drop-oldest 保持输入有界，并在
超过批准的 recovery/fatal 上限后失败；已授权时可在完整 chunk boundary 粘滞降级一次，
若仍落后则同样 drop-oldest。任何路径都不得阻塞 capture、静默旁路 Index 或用 no-index
样本充当 qualified output。

## 8. Metrics snapshot

```cpp
struct RealtimeMetrics {
    std::uint64_t input_hops_received = 0;
    std::uint64_t input_hops_dropped = 0;
    std::uint64_t output_hops_produced = 0;
    std::uint64_t output_hops_concealed = 0;
    std::uint64_t capture_overruns = 0;
    std::uint64_t playback_underruns = 0;
    std::uint64_t alsa_xruns = 0;
    std::uint64_t alsa_recoveries = 0;
    std::uint64_t deadline_misses = 0;
    std::uint64_t index_queries = 0;
    std::uint64_t index_enabled_hops = 0;
    std::uint64_t index_bypassed_hops = 0;
    std::uint64_t index_deadline_misses = 0;
    std::uint64_t index_fallback_transitions = 0;

    RetrievalMode retrieval_mode;
    IndexFallbackReason index_fallback_reason;
    std::uint64_t retrieval_generation = 0;

    LatencySummary queue_wait;
    LatencySummary hubert;
    LatencySummary f0;
    LatencySummary index;
    LatencySummary synth;
    LatencySummary gpu_total;
    LatencySummary capture_to_playback;
};
```

`LatencySummary` 至少有 count、p50、p95、p99、max，单位固定为 milliseconds；不得混用
CUDA event time、CPU queue time 和 ALSA device delay。snapshot 是一致读，不要求 audio
threads格式化 histogram 或日志。run metadata 还必须包含 Index SHA-256、rows、dim、`k`、
`index_rate` 和 backend 标识；fallback transition 另有结构化时间/sequence/reason，不能只
通过最终 counter 猜测。

## 9. Latency definitions

- **input age**：capture timestamp 到完整 inference window ready 的时间。
- **queue wait**：window ready 到 GPU work 实际开始。
- **compute**：该 emit hop 对应的 persistent model execution wall time；另记录各 CUDA stage。
- **index**：从本 hop 的 query batch ready 到 exact weighted retrieval 输出 ready；host、GPU
  或混合 backend 都使用同一 wall boundary，并另记录可用的 CUDA event 子阶段。
- **algorithmic latency**：window 所需 lookahead + emit/crossfade policy 的固定延迟。
- **output queue/device delay**：GPU output ready 到 ALSA 预计播放的时间。
- **capture-to-playback**：以上各项合成并由 ALSA timestamp/delay 校准；必须包含 lookahead、
  queue 和 device buffers，不得只报告 kernel time。

warm-up/model load 不计入 steady-state SLO，但单独报告 startup time。RT-DEC-003 批准 nominal
10-minute Index-enabled run p95 `<= 250 ms`、p99 `<= 350 ms`；compute p99 必须小于或等于
该 chunk 的 emitted audio duration，且 0 Index fallback。no-index diagnostic/fallback latency
单独报告，不能用于替代 qualified SLO。

## 10. Memory accounting

报告至少分开：

- selected safetensors mappings/file-backed resident pages；
- Flat Index mapping、resident pages、可选 device copy 与 retrieval workspace；
- device weights；
- maximum device workspace/activations；
- CUDA context/library allocator；
- input/output/overlap host rings；
- process RSS/VmHWM 与 Jetson system-memory delta。

不能把文件大小之和当作显存，也不能在 Jetson unified memory 上把 RSS 与 CUDA allocation
简单相加后宣称总量。profile 必须给出每类最大 buffer 的尺寸公式；10-minute run 在 warm-up
后不得出现随 chunk count 单调增长的 retained allocation。
