# Feature Specification: Realtime Voice Conversion

**Feature**: `006-realtime-voice-conversion`
**Created**: 2026-08-10
**Status**: Approved; R0 evidence is partially implemented; no realtime product path exists
**Architecture decision**:
[ADR-0002](../../docs/adr/0002-persistent-streaming-runtime-and-local-audio-boundary.md)
(`Accepted` on 2026-08-10)

## 1. Scope and authority

本规格定义在 Jetson Orin Linux 上新增本机、单会话、低延迟声音转换的已批准目标：

```text
ALSA capture -> bounded input ring -> stateful streaming inference
             -> bounded output ring -> ALSA playback

real human WAV replay -----------> the same streaming inference -> benchmark/fixture output
```

当前仓库只有 Spec 001 定义的离线 WAV 转换。`VoiceConversionPipeline::convert_buffer`
处理完整 buffer，`Synthesizer::infer_stream` 也只是完整合成后的窗口裁剪；二者都不是本规格
所需的流式 session。本文中的接口、CLI、延迟目标和模块均是**提案**，不得描述为当前能力。

[Spec 001](../001-voice-conversion-engine/spec.md) 继续管理离线 WAV、模型兼容和既有数值
门槛。本规格不得降低其 HuBERT、F0、VITS 或端到端门禁，也不得改变 `vc_convert`、
`vc_batch`、`voxmutatio` facade、权重/Index/WAV 格式或现有失败语义。

## 2. Maintainer decisions

维护者于 2026-08-10 批准推进本规格。RT-DEC-001、003、004 采用推荐项；RT-DEC-002
根据维护者对竞争力的要求修改为“Index-enabled 正常质量路径 + 显式 no-index fallback”。
这些是已批准的产品/架构边界，不代表实现或性能已经完成验证。

| ID | 决策范围 | 已批准选择 | 被拒绝或延后方案 |
|---|---|---|---|
| **RT-DEC-001 Product boundary** | 首版设备/传输边界 | 本机 ALSA capture/playback + deterministic WAV replay | WebSocket/browser streaming 延后并另立 ADR；不在现有 HTTP 子进程 route 中嵌入音频流 |
| **RT-DEC-002 Qualified profile** | 首个可宣称支持的模型/检索组合 | F0 VITS v2、HuBERT、FCPE、40 kHz；有效 Flat Index 是正常质量路径，`index_rate > 0`；显式 no-index 只作诊断/兼容或经启动配置预授权的有界 overload fallback | WavLM、RMVPE、v1 和多 profile 在分别通过延迟/质量/内存门禁后再加入；索引实现策略须由 R0 基线决定 |
| **RT-DEC-003 Latency SLO** | nominal 10-minute run 的 capture-to-playback 延迟 | p95 `<= 250 ms`、p99 `<= 350 ms`，且 chunk compute p99 不超过该 chunk 产生的音频时长 | 500 ms relaxed 档或未定义“低延迟”；没有可复现门限时不得称为 realtime |
| **RT-DEC-004 Overload policy** | 输入快于推理时的有界行为 | 丢弃最旧的完整 input hop，保持 newest-first；输出 silence/crossfade，并累计 drop/xrun 计数 | 阻塞 capture 导致无界设备延迟，或静默覆盖/伪造连续成功；“首次过载立即退出”仅作为严格诊断模式 |

Index-enabled 是 RT-DEC-003 的主验收路径。R0 已用 standalone CPU exact reference 与
persistent-GPU exact probe 测量诊断性规模/延迟/内存可行性，但没有代表性目标声线
model+Index pair、质量 A/B 或完整 pipeline/stream SLO，因此尚未资格化 backend 或支持上限。
R0 仍须固定 target index hash、rows、dim、`k`、`index_rate` 和 query batch，并测量质量、
延迟与内存。不得用 synthetic/no-index run 替代 Index-enabled SLO 或质量验收。

## 3. Approved target contract

本表是已批准但尚未实现的目标；只有对应实现与验收证据齐全后，才能转为已支持能力。

| 维度 | 首版已批准目标 | 明确不支持/延后 |
|---|---|---|
| 平台 | Jetson Orin Linux、CUDA FP32、单 GPU | CPU、FP16、非 Linux 音频后端 |
| 会话 | 单进程、单 realtime session、单 speaker | 多会话、GPU 共享调度、多租户 |
| 音频边界 | 本机 ALSA PCM；mono；启动时报告实际设备格式/采样率 | 网络音频、浏览器麦克风、隐式格式降级 |
| 模型 profile | v2 HuBERT + FCPE + F0 VITS，模型原生 40 kHz | WavLM、RMVPE、v1、non-F0 |
| 检索 | 正常质量路径要求已验证 Flat Index 与 `index_rate > 0`；显式 no-index fallback 可用 | 缺失/损坏/超上限 Index 的静默忽略，或无授权的运行中自动降级 |
| 参数 | speaker、pitch；protect 仅在跨块语义验证后启用 | formant shift、运行中切换模型/F0/encoder |
| I/O 持久化 | 默认不录制，不把麦克风或输出写盘 | 隐式保存、上传或远程传输 |
| Web | 无 | 修改 `vc_serve` 为 WebSocket/常驻模型服务 |

目标新增独立 CLI `vc_realtime`，至少接收模型路径、Index 路径/检索模式、capture/playback
ALSA device、speaker 和 pitch。它必须在打开实时采集前完成模型/Index 校验、device
allocation 和 warm-up；初始化、
设备协商或 warm-up 失败时返回非零，不得先开始采集再以长时间静音掩盖加载。

no-index 有两种且都必须显式：启动时由用户选择；或用户在启动时预先授权“持续 deadline
miss 时粘滞降级到 no-index”。后者只在完整 chunk boundary 提交，同一 session 不自动恢复
Index，且必须在状态、结构化事件与计数器中可见。未授权时不得为了掩盖过载静默关闭
Index；继续按 RT-DEC-004 drop-oldest，并在超过恢复上限时诚实失败。

确定性 WAV replay 使用同一 streaming session，而不是调用离线 `convert_buffer`。benchmark
入口可以是独立工具或测试 runner，但其输入、profile、warm-up、时钟和输出 schema 必须稳定。

## 4. User stories

### Story 1 — 本机实时监听（P0）

用户选择本机 capture/playback device、已验证的模型和 speaker，启动 `vc_realtime`。完成
warm-up 后，服务持续把麦克风音频转换后送到 playback device；Ctrl-C 触发有界 drain 和
清理。正常退出打印结构化统计，fatal model/device error 返回非零。

### Story 2 — 可复现的流式回放（P0）

开发者将具有明确来源的真实人声 WAV 按固定 ALSA/stream hop replay 到同一 session，获取
输出和逐 chunk 时间戳。相同模型、配置、输入与构建在允许的数值容差内得到可比较结果；
报告包含硬件、功耗模式、clocks、温度、模型哈希、warm-up 和 latency 分位数。

### Story 3 — 有界过载与错误诚实性（P0）

当 inference worker 落后时，输入队列不增长。系统按批准策略丢弃完整旧 hop，记录 drop；
playback 缺数据时输出显式 concealment 并记录 underrun。持续 CUDA/device failure 使 session
进入 `Failed`，之后不再返回可被误认为成功转换的音频。

### Story 4 — 会话内参数更新与重置（P1）

允许的 speaker/pitch/protect 更新只在 chunk boundary 以完整快照提交；任何非法组合都保留
旧快照。`reset` 清除 resampler、filter、F0/context、NSF phase、overlap 和 queue state；
`drain` 只输出已接受数据，不接收新输入。

## 5. Functional requirements

- **RT-FR-001 Stateful session**：提供独立 realtime session，不把离线
  `convert_buffer` 或局部 `infer_stream` helper 当作实时产品入口。
- **RT-FR-002 Transactional initialization**：先验证模型/profile/设备和所有固定上限，
  staged 分配权重、workspace、stream/handles 与 ring，warm-up 成功后才进入 `Ready`；失败
  释放全部 staged 资源并保持未初始化。
- **RT-FR-003 Persistent CUDA runtime**：稳态不得重新 mmap/解析模型、逐算子上传权重或
  逐算子 `cudaMalloc/cudaFree`；selected weights、CUDA stream、library handles 和 bounded
  workspace 由 session/runtime 显式 RAII 持有。Index mapping/accelerator state 也必须有
  session owner 和经 profile 上限约束；qualified target/full-pipeline evidence 完成前不预设
  host 或 device 产品 backend。
- **RT-FR-004 Bounded queues**：capture、inference 和 playback 缓冲均有固定容量；所有
  size/byte 算术在分配前检查，10-minute nominal 与 overload run 均不得出现无界增长。
- **RT-FR-005 Real-time thread boundary**：capture/playback 线程不得执行模型推理、动态
  分配、阻塞互斥、文件/网络访问或格式化日志；GPU worker 不在 ALSA critical path 上运行。
- **RT-FR-006 Frame alignment**：stream window/hop 必须与 HuBERT 20 ms stride、F0/VITS
  10 ms frame 对齐；任何重采样 phase 和余数跨 chunk 保留，不得逐块重新起算。
- **RT-FR-007 Explicit context**：profile 显式定义 lookbehind、lookahead、emit window 和
  crossfade。HuBERT 与既有 VITS/RMVPE 路径不得被误称为 causal；首版用有限滑窗、中心裁剪
  和 overlap/crossfade 保持 stock model 兼容。
- **RT-FR-008 Cross-chunk state**：filter history、F0/UV context、median/protect context、
  RMS state、NSF harmonic phase、resampler phase、overlap tail 和 sequence/timestamps 都有
  单一 session owner，并由 `reset` 一次性清除。
- **RT-FR-009 Qualified profile**：首版只接受 RT-DEC-002 批准的 profile。正常质量模式
  必须校验 compatible Flat Index、hash/rows/dim/`k`/`index_rate` 与 profile 上限；显式
  no-index 模式必须报告降级状态。不兼容 encoder/synthesizer、非 40 kHz 模型、non-F0、
  CPU、FP16、损坏/超限 Index 和未支持参数在启动前失败。
- **RT-FR-010 ALSA adapter**：设备名、direction、实际 format/rate/channels、period/buffer
  必须校验并记录；不允许 ALSA 在未报告的情况下静默采用不同契约。xrun recovery 有次数
  上限，超过上限进入 `Failed`。
- **RT-FR-011 Backpressure**：overload 严格采用 RT-DEC-004；每次 drop、underrun、overrun、
  xrun/recovery 均进入单调计数器，不能只写临时日志。只有启动时显式授权的 fallback policy
  才能在持续 deadline miss 后于 chunk boundary 粘滞切到 no-index；触发阈值由 R0 基线固化，
  切换不能代替 queue/drop 上界，也不能自动来回振荡。
- **RT-FR-012 Error semantics**：区分 invalid config、model validation、device negotiation、
  CUDA inference、deadline miss、overload 和 device removal；fatal error 后输出状态稳定且不可
  继续，调用方必须显式 reset/reinitialize。
- **RT-FR-013 Runtime updates**：支持的请求参数以 immutable snapshot 在 chunk boundary
  原子切换；不完整或非法更新不得部分生效。
- **RT-FR-014 Observability**：使用 monotonic `steady_clock`、CUDA events 和 ALSA timestamp/
  delay，分别记录 capture age、queue wait、各模型 stage、GPU total、output queue、device
  delay 与 capture-to-playback estimate。Index 还须记录 stage latency、query 数、当前
  retrieval mode、enabled/bypassed hops、fallback transition/reason/generation，以及配置的
  index hash/rows/dim/`k`/`index_rate`。
- **RT-FR-015 Latency and stability**：以 RT-DEC-003 门控 10-minute nominal run；主验收
  必须在 Index-enabled 正常质量模式运行，且 0 drop、0 xrun、0 fallback、0 non-finite、
  0 fatal error，compute p99 不超过 emitted duration。no-index 结果单独报告，不得替代主验收。
- **RT-FR-016 Numerical and continuity safety**：逐 stage 继续满足 Spec 001/Constitution 的
  数值门槛；Index-enabled 检索须与离线 exact weighted k-NN reference 对齐。完整 stream 输出
  有限、时长稳定且通过批准的频谱/边界连续性指标。R0 还必须用真实目标声线建立
  Index-enabled 与 no-index 的盲听/客观 A/B 基线并固化门槛；门槛未定或未通过时不得宣称
  Index 带来竞争性质量。不能用 shape、音频文件存在或“听起来正常”替代断言。
- **RT-FR-017 Memory evidence**：selected profile 只保留一份 device weights 和固定最大
  workspace；报告 peak process/system memory 与 CUDA allocations。10-minute steady state 在
  warm-up 后不得持续增长，所有临时资源在 stop/failure 后释放。
- **RT-FR-018 Offline compatibility**：离线 public headers/API、CLI、WAV/Index/safetensors、
  model-rate 输出和 CTest 名称/门槛保持兼容；若共享 runtime 导致离线数值或错误行为变化，
  必须回滚该阶段。

## 6. Explicit non-goals

- WebSocket、WebRTC、browser AudioWorklet、远程公网音频和 `vc_serve` 常驻模型。
- 多用户、多会话、动态 GPU 调度或跨进程 model server。
- CPU、FP16、TensorRT、ONNX Runtime 或新的大型运行时。
- WavLM、RMVPE、v1、non-F0 或运行中切换模型；除获预授权的粘滞 no-index fallback 外，
  不支持运行中切换检索模式或 Index 文件。
- 训练、声源分离、降噪/去混响或音频设备发现 UI。
- 无损等同于离线整段波形；滑窗/crossfade 的差异必须由独立质量指标约束，不能伪称 bit-exact。
- 默认录音、持久化、上传或遥测。

## 7. Verification and assets

### Default asset-free contract

默认测试使用 fake executor/PCM seam 和底层确定性信号，仅验证配置、ring、生命周期、
rollback、分块 DSP、backpressure、统计、错误和清理。合成信号不能证明模型质量或 realtime
产品完成。

最低覆盖包括：

- invalid/overflow profile、queue capacity、device format 和 timestamp；
- ring wrap/full/empty/drop-oldest、sequence gap、reset/drain/fatal state；
- stateful resampler/IIR 分块结果与整段参考一致；
- fake slow executor 下的 deadline miss、drop、concealment 和 bounded memory；
- model/device init 中途失败的事务回滚和重复 init/stop；
- legacy `Synthesizer::infer_stream` 的既有 crop/validation characterization。

当前 `test_realtime_benchmark_report` 只门控内部报告字段/模式一致性；尚无 session producer、
稳定 serialization 或完整 outcome 字段 population。
`test_vits_stream_characterization` 只冻结 legacy full-infer crop、40 kHz 下每 frame 400 samples、
非法窗口、重复调用与 NSF phase-reset 事实；二者都不实现 session 或生成产品 SLO 报告。

2026-08-10 的 current-tree 完整回归为 29/29 CTest 通过（10 个 contract、19 个
asset-gated integration、0 失败，real 331.28 秒）。它证明上述两个 R0 门禁加入后既有测试集
没有回退；普通 CTest 全绿不能替代 realtime session、真实目标 Index、质量或 10-minute SLO
验收。

### Real-asset integration and benchmark

模型/E2E 必须使用 provenance-tracked 真实人声、真实 HuBERT/FCPE/VITS weights、目标声线
Flat Index 和固定 reference。性能测试使用独立 `benchmark` label/runner；未执行、缺资产
或普通机器上的 skip 不能计作通过。

基准至少执行 5 s warm-up 和 10 min nominal replay，并记录：

- current commit/tree、Release flags、CUDA/compiler、Orin model、power mode 与 clocks；
- 模型/音频/Index SHA-256、Index rows/dim/`k`/`index_rate`、profile/window/hop/lookahead、
  ALSA period/buffer；
- p50/p95/p99/max stage（含 Index）、compute 与 capture-to-playback latency；
- RTF、drop/underrun/overrun/xrun/recovery、Index bypass/fallback、峰值/稳态内存、温度和降频；
- 输出时长、有限值、逐 stage 对齐和完整 stream continuity 结果。

当前 R0 资产证据见 [machine-readable manifest](r0-assets.json) 与
[research.md](research.md)：现有真实语音、HuBERT、FCPE、VITS v2 文件的 path/bytes/SHA-256
已核验，音频可从固定 dataset revision/utterance 精确重放，HuBERT/VITS 的原始 checkpoint
也已与固定官方 revision 对上。FCPE 已从精确 `torchfcpe==0.0.4` wheel/bundled checkpoint
逐字节复现，并固定 checkpoint/package hash 与 MIT license。当前仍没有同一份有授权 corpus
产生的代表性目标声线 v2 model+Flat Index pair；manifest 因此保持
`qualification_status: blocked`。这两项 blocker 修复前，本地资产可用于研究和既有离线测试，
但不足以完成 Index-enabled realtime profile。

## 8. Success criteria

- **RT-SC-001**：RT-DEC-001..004 已于 2026-08-10 获维护者批准，ADR-0002 为 `Accepted`；
  该治理前提已满足，但不构成任何实现或性能通过证据。
- **RT-SC-002**：所有 realtime default contract 实际执行并通过，0 skip；现有离线 contract
  与 asset integration 数量、名称和数值门槛不回退。
- **RT-SC-003**：真实资产 stream integration 满足 Spec 001 的 module 与 Index exact-retrieval
  alignment；完整输出 spectral correlation 建议 `> 0.98`，最终 boundary metric 以及
  Index-enabled 相对 no-index 的质量门槛须在 R0 baseline 固化。
- **RT-SC-004**：Index-enabled 正常质量模式的 10-minute nominal run 达到 p95
  `<= 250 ms`、p99 `<= 350 ms`，compute p99 不超过 emitted duration，并且
  0 drop/xrun/fallback/fatal/non-finite。no-index run 只作为独立 fallback 证据。
- **RT-SC-005**：受控 overload run 的 queue 始终有界，只按批准策略丢弃完整旧 hop；计数、
  sequence gap 和 concealment 与注入次数一致。
- **RT-SC-006**：连续 start/stop/reset/failure 与 10-minute run 无持续内存增长；峰值、稳态、
  selected weights 和 workspace 上界有可复现记录。
- **RT-SC-007**：ALSA loopback 或批准的物理设备 run 覆盖协商、timestamp、xrun recovery、
  device removal、Ctrl-C drain 和非零 fatal exit。
- **RT-SC-008**：未实现的 profile、WebSocket 和参数稳定失败；损坏/超限 Index 或未授权
  fallback 不被静默接受，CLI/help/metrics 清楚区分 Index-enabled 与 no-index 状态。

只完成 streaming helper、单次短 buffer、组件计时或生成音频都不能把本规格标为 Complete。
