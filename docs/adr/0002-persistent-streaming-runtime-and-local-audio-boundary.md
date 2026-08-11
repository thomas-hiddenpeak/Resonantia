# ADR-0002: 持久流式推理运行时与本机音频边界

**Status**: Accepted
**Date**: 2026-08-10
**Accepted**: 2026-08-10
**Owners**: Resonantia maintainers
**Governing specs**: [Spec 006](../../specs/006-realtime-voice-conversion/spec.md)、
[Spec 001](../../specs/001-voice-conversion-engine/spec.md)、
[Spec 003](../../specs/003-webui/spec.md)、[ADR-0001](0001-modular-runtime-and-orchestration-boundaries.md)

维护者于 2026-08-10 批准推进该架构；RT-DEC-001、003、004 采用推荐项，RT-DEC-002
按维护者“索引不可因竞争力被放弃”的反馈修订后接受。`Accepted` 授权按 R0–R5 分阶段实施，
不表示 realtime runtime、CLI、Index SLO 或产品质量已经实现或通过。

## Context and decision drivers

当前产品链是离线 WAV conversion 和 HTTP → CLI subprocess 编排。公共 Pipeline 没有流式
session；`Synthesizer::infer_stream` 只在完整 `infer` 后裁剪窗口。离线前后处理包含
zero-phase filter、reflect padding、future-dependent F0 interpolation、全段 RMS envelope，
HuBERT/RMVPE/VITS 也没有完整的跨块状态契约。

执行层同样不适合先接音频设备：HuBERT/WavLM/RMVPE 在 inference 中重新 mmap 模型；大量
content/F0/VITS 算子逐次分配 device memory、上传 host weights/activations、同步拷回 host；
没有 session-owned stream、handles、events 或有界 workspace。Synthesizer 的 mmap loader 使用
process-global static path cache，生命周期和模型数量上界不由 Pipeline 表达。ADR 接受时的
Flat Index 产品实现是 host exact scan；它保留 exact weighted retrieval 语义，但没有代表性
目标声线 rows/query batch 的 full-pipeline realtime latency、memory 或质量 A/B baseline。
随后完成的 standalone 诊断证据及其非产品边界见下方 2026-08-10 R0 progress note。

一次诊断性 Release run 在 Orin NX/MAXN、未锁 clocks 的环境中处理 5.855 s 真实语音：
Pipeline total 16.507 s、wall 16.775 s、RTF 约 2.82；HuBERT 1.219 s、RMVPE 2.142 s、Synth
13.145 s。数值 correlation 0.999716，但当前路径不能持续 realtime。完整方法和限制见
[Spec 006 research](../../specs/006-realtime-voice-conversion/research.md)。

决策必须同时满足：

- 不降低 Spec 001 的模型兼容、数值、错误和持久格式契约；
- C++20/CUDA FP32、safetensors-only、无 Python runtime；
- 模型、CUDA/ALSA、线程、ring 和 temporary state 的显式 RAII owner；
- 有界内存、可测量 latency/jitter/drop/xrun，而不是 helper 名称或平均 kernel time；
- 保持 ADR-0001 的 compatibility facade 和 app/runtime 依赖方向；
- 不把 WebSocket、多会话或新大型 runtime 偷渡进首版。

## Accepted maintainer decisions

| ID | Accepted decision | Alternative/consequence |
|---|---|---|
| **RT-DEC-001** | 首版为本机 ALSA capture/playback，并提供同 session 的 deterministic WAV replay | WebSocket-first 会改变 Spec 003 的 protocol、auth、deployment、GPU session 与 browser boundary，需要独立 ADR |
| **RT-DEC-002** | 首个 qualified profile 固定为 v2/HuBERT/FCPE/40 kHz，正常质量路径要求有效 Flat Index 与 `index_rate > 0`；no-index 只作显式诊断/兼容或启动时预授权的粘滞 overload fallback | WavLM/RMVPE/v1/多 profile 延后；Index backend、rate 和 rows 上限由 R0 硬数据选择，不能通过静默 no-index 换取达标 |
| **RT-DEC-003** | nominal 10-minute capture-to-playback p95 `<= 250 ms`、p99 `<= 350 ms`，compute p99 不超过 emitted duration | 选择 500 ms relaxed SLO 必须在产品契约中直写；未定义门限不能称 realtime |
| **RT-DEC-004** | overload 时 bounded drop-oldest whole-hop + silence/crossfade concealment + monotonic counters | blocking capture 会累积陈旧延迟；fail-fast 可作为 diagnostic mode，但不作为推荐交互默认 |

## Decision

采用“persistent model/retrieval state + single-session mutable stream state + host audio
adapter”的结构。Index-enabled 是 qualified/competitive contract 的主路径，不是可为性能
静默删除的装饰阶段。

### 1. Build and module boundary

- 在 ADR-0001 的 12 个 internal OBJECT components 之后新增
  `resonantia_streaming_obj`，仍聚合进单一公共 `voxmutatio` static facade；不发布第二套公共
  library 或改变现有 target 名。
- `streaming` 只依赖 core/I/O/content/F0/synthesizer 及明确共享的 execution internals，不能
  依赖 app-support、ALSA、Web、training 或 separation。
- 新增 command-scoped `resonantia_vc_realtime_app_support` 和薄 `vc_realtime` main。ALSA
  headers/library 只进入该 app target；不从公共模型 header 泄漏。
- `vc_serve`、`resonantia_web_support` 和 Spec 003 routes 不变。WebSocket 或常驻 model service
  需要后续 Spec/ADR。

允许依赖方向：

```text
core/io/cuda_ops -> content/f0/synthesizer execution
                               |
                               v
                           streaming
                               |
                               v
                    vc_realtime app-support -> ALSA

offline pipeline ----------------^  (shares validated execution, not stream queues)
Web/training/separation ---------X  (not realtime dependencies)
```

### 2. Persistent inference runtime

- 选择的 HuBERT、FCPE、VITS v2 weights 和 Flat Index 在 staged init 中完成 safetensors/
  Index header、dtype/shape/byte-range/finite/model/profile invariant 校验；模型 weights 再一次性
  上传到 session-owned device buffers，Index mapping/execution state 也由 session 显式拥有。
- commit 后稳态不重新 mmap/解析模型，不逐算子 H2D weight copy，不逐算子
  `cudaMalloc/cudaFree`。需要保留的 host mapping 必须显式拥有并计入 memory report；不得依赖
  global static path cache。
- runtime 持有一个 CUDA stream、与该 stream 绑定的 cuBLAS/cuFFT handles、events 和按批准
  profile/window 计算的最大 workspace。初始化失败释放全部 staged state。
- Index 执行可以保留 host mmap exact scan、采用批处理/并行优化或 device-resident retrieval；
  ADR 不预选算法。R0 必须先记录 Index hash、rows、dim、`k`、`index_rate`、query batch、
  latency、memory 与质量，再固化支持上限。任何优化均与离线 exact weighted k-NN 对拍。
- 首版一个 runtime 只服务一个 session/worker。权重 immutable，workspace/stream mutable；
  不为尚未批准的 multi-session sharing 预建 locks/cache/backend abstraction。
- offline Pipeline 可以分阶段复用该 execution runtime，但其 public API、model validation、
  errors、timing fields 和数值输出必须由既有门禁保护。

### 3. Streaming state and window policy

- session 显式拥有 input/output resampler phase、filter history、HuBERT context/emitted cursor、
  FCPE mel/F0/UV/protect context、Index query/mode/fallback generation、VITS overlap tail/NSF
  phase、parameter snapshot 和统计。
- stock HuBERT/VITS 不被重写为假想 causal model。首版使用有限 sliding window、显式
  lookbehind/lookahead、center emit crop 和 bounded crossfade；R3 只在 10/20 ms 对齐 matrix
  中选择一个 qualified profile。
- window/hop/lookahead/crossfade 在批准 profile 中固定，不作为首版任意 CLI knobs。它们的
  algorithmic latency 必须计入 capture-to-playback SLO。
- 既有 `Synthesizer::infer_stream` 在 R0 characterization 后保留兼容 crop 行为或走显式
  deprecation；新 session 不用其名称代替跨块 state。

### 4. Threading, queues and backpressure

- capture producer、single GPU worker、playback consumer 通过 init-time preallocated SPSC
  rings 通信；queue/slot/sample/byte 上限在启动前检查。
- capture/playback critical threads 不做模型推理、动态分配、blocking mutex、文件/网络 I/O
  或格式化日志。signal handler 只设置 stop request。
- input ring full 时丢弃最旧完整 hop，保留 sequence gap 并增加 drop counter。output ring
  empty 时产生固定长度 silence/crossfade concealment 并增加 underrun/conceal counter。
- Index failure 或 stage 超预算默认不旁路检索，按 error/deadline/overload 契约处理。只有用户
  在启动配置中明确授权时，达到 R0 固化触发条件后才可在完整 chunk boundary 一次性、粘滞
  降级到 no-index；同一 session 不自动恢复，transition reason/sequence/generation/bypassed
  hops 必须可观测。no-index 仍不能替代 drop-oldest 的 queue 上界。
- nominal SLO run 不允许 drop/xrun；overload test 才验证上述政策。持续 CUDA、device removal
  或超过批准 recovery limit 的 xrun 使 session 进入 `Failed`，之后不返回普通成功音频。
  Index-enabled nominal run 还要求 0 fallback。

### 5. Lifecycle and error boundary

session 状态为 `Uninitialized -> Ready -> Running -> Draining -> Stopped`，任何 fatal path
进入 `Failed`。init/warm-up 是 transactional；reset 清除所有 mutable DSP/model/ring/phase
state；drain 停止接收新 input 后只完成已接受数据。

公开结果区分 invalid config、model validation、Index validation/deadline/fallback、device
negotiation、CUDA inference、deadline miss、overload 和 device removal。调用方必须根据
状态/error 判断，不得按 output 非空推断成功。缺失、损坏或超出 R0 上限的 Index 在正常
质量模式启动失败，不能静默转为 no-index。

### 6. Metrics and acceptance evidence

- CPU/queue 使用 `steady_clock`；GPU 使用 CUDA events；设备使用 ALSA hardware timestamp/
  delay。不得把三种时钟混为一个未定义 timing。
- 报告 queue wait、HuBERT、F0、Index、Synth、GPU total、algorithmic lookahead、output
  queue/device delay、capture-to-playback 的 p50/p95/p99/max。
- 同时报告 RTF、drop/underrun/overrun/xrun/recovery、Index mode/query/deadline/bypass/fallback、
  selected model/audio/Index hashes、Index rows/dim/`k`/`index_rate`、process/system/CUDA memory、
  power mode、clocks、温度和降频。
- 模型/E2E 数值只能用 provenance-tracked 真实模型和真实人声；performance runner 单独标为
  `benchmark`，缺资产或未运行不得计作通过。
- 主 SLO 与质量验收使用 Index-enabled mode 且 0 fallback；no-index diagnostic/fallback 单列。
  R0 使用真实目标声线做 Index-enabled/no-index A/B，在看到数据前不凭空设质量优势门槛。

### 7. Privacy and deployment

- 首版只访问用户显式选择的本机 ALSA devices；默认不录音、不持久化、不上传、不联网。
- CLI 进程继承当前用户设备权限，不提升权限，也不隐式运行 `jetson_clocks`/`nvpmodel`。
- 正式 benchmark 可以记录或由操作者显式设置 power/clocks，但这不是 runtime side effect。

## Alternatives considered

### Loop over `convert_buffer` or existing `infer_stream`

存在 whole-buffer future dependencies、模型/weight 重复工作和 NSF phase reset，无法表达有界
lookahead/backpressure，拒绝。

### Start with ALSA/WebSocket before execution refactor

当前 RTF 约 2.82，设备/协议只会增加 queue 和 jitter；先接 I/O 会产生持续陈旧音频而不是
real-time，拒绝。persistent runtime 与 deterministic replay 必须先完成。

### WebSocket inside `vc_serve`

现有 Web 是 request/response + CLI subprocess。加入长连接会改变 auth、rate/resource limits、
GPU mutex、worker lifetime、TLS/origin 与 multi-client semantics，延后并另立 ADR。

### New causal content/F0/vocoder models

可能降低 lookahead，但会改变权重、训练、模型兼容和数值 reference，超出首版 stock RVC
兼容范围。先采用 bounded sliding window；若 SLO 无法达到，以 evidence 触发新 Spec/ADR，
不能静默换模型。

### Retain host-vector/per-op CUDA path for streaming

实现简单但 allocation、weight transfer 和同步次数随算子增加，无法给出稳定 deadline 或
memory bound，拒绝。旧 executor 只作为 staged rollback，不能作为完成态。

### Remove Index from the first profile

关闭 current host exact search 可以缩小 latency 风险，但会删除 RVC 的检索音色路径并可能削弱
产品竞争力，维护者明确拒绝。正确做法是在 R0 测量真实 Index envelope，选择有界实现，并
保留显式、可观测的 no-index 诊断/overload fallback；不能用 fallback 结果宣称 qualified。

### Expose every window and backend knob

会形成未经验证的组合矩阵和 inert options，违反稳定接口与 no-speculation 原则。首版只暴露
qualified profile；diagnostic matrix 留在 benchmark，不成为用户契约。

## Consequences

正面影响：模型、Index 和 workspace 生命周期显式；内存/latency 可界定；stream state 与
ALSA/Web 分离；离线和实时可以共享已验证 execution 而不共享队列；Index 质量路径得以保留，
overload/failure/fallback 可测试且诚实。

成本与风险：这是对当前 host-vector CUDA execution 的实质重构，触及 content、F0、VITS 和
Pipeline 数值路径；device weights + workspace 在 Jetson unified memory 上需要严谨 accounting；
stock bidirectional/non-causal models 需要滑窗重复计算，host exact Index 还随 rows/query
batch 扩展，可能无法达到批准 SLO；ALSA 设备和 physical routing 仍需要硬件门禁。

如果 R3 在保持 FP32/stock model/数值门槛时无法达到 SLO，正确结论是 ADR/profile 未满足，
而不是放宽数值、隐藏 drop、只报告平均 kernel time或把 500 ms 未批准行为称为 realtime。

## Migration and rollback

### R0 — Approved decisions and frozen evidence

四项 maintainer decisions 已记录。legacy `infer_stream` characterization、构造期报告
validator、R0 asset manifest/FCPE provenance、离线 Pipeline Index fail-closed 修复和
standalone exact-retrieval probe 已完成；继续补齐 target model+matching Index/licensed corpus、
Index-enabled/no-index 质量 A/B、target/full-pipeline latency/memory 和 session tests。不新增
realtime runtime。

**Rollback**：删除 proposal/baseline artifacts；不触碰用户模型、音频或 run data。

### R1 — Persistent component execution

逐组件建立 device weights、session-owned Index、stream-bound handles 和 bounded workspace，
先由 offline callers 对拍。Index backend 根据 R0 evidence 选择并与 exact reference 对齐；
旧 executor 暂时作为 internal rollback。

**Rollback trigger**：任何现有数值门槛/错误语义失败、内存上界无法说明、steady allocation/
weight upload 仍存在。切回 R0 executor/facade，不改权重格式。

### R2 — Deterministic stateful session

新增 streaming component、preallocated rings、state machine、fake executor/retrieval、
stateful DSP、显式 no-index/fallback state 和 replay harness；不接 ALSA。

**Rollback trigger**：queue 无界、critical path allocation/blocking、reset/rollback 不完整或
existing target/CTest 减少。移除 streaming objects，保留最后通过的 offline runtime。

### R3 — Model window/state integration

连接 HuBERT/FCPE/Index/VITS persistent execution，选择有限 window/lookahead/crop/crossfade，
保留 NSF/resampler/filter/F0 state，运行真实资产 numerical/continuity/latency/quality matrix。

**Rollback trigger**：Spec 001 对齐回退、boundary artifact 超门槛、nominal replay drop，或
compute p99 无法接近 emitted duration。切回 R2；不得修改 reference/tolerance。

### R4 — ALSA app boundary

新增薄 `vc_realtime`、command-scoped app-support、Index-required/no-index/fallback startup
contract、capture/GPU/playback owners、device negotiation、xrun recovery 和 Ctrl-C drain。
ALSA 不进入 public model runtime。

**Rollback trigger**：设备格式静默变化、fatal 返回成功、线程无法有界停止、默认录音/网络
side effect 或离线 facade 变化。禁用/删除新 app target。

### R5 — Acceptance and cleanup

完成 Index-enabled 10-minute SLO（0 fallback）、独立 no-index/overload/device/CUDA failure、
质量、内存/thermal evidence、全部离线/实时门禁；再同步 DESIGN/PROJECT_STATE/TRACEABILITY/
README 和 ADR outcome。

**Rollback trigger**：任一 mandatory asset 未执行、skip 被计为 pass、SLO/质量/内存不满足或
existing tests 回退。保持 Spec/feature 未完成，关闭 realtime target；用户数据无需迁移。

每阶段应独立可回滚并保留 exact commands/results。dirty worktree 或无法创建阶段 commit 时，
必须像 ADR-0001 一样保存独立 build/evidence，而不能通过 reset 丢弃维护者改动。

## R0 progress note (2026-08-10)

R0 evidence 当前为**部分完成**，不改变本 ADR 的 accepted target architecture，也不把产品
路径写成已实现：

- `test_vits_stream_characterization` 已冻结 legacy full-infer crop、40 kHz 下每 frame 400
  samples、非法窗口、repeated-call bit-exact 与每次调用 NSF phase reset；这证明现有 helper
  没有连续 session state。
- mode-aware `BenchmarkReport` construction-time validator 已有 contract 门禁，但尚无 session
  producer、稳定 serialization 或完整 outcome 字段 population，不能生成 10-minute 产品报告。
- 离线 Spec 001 Pipeline 对 `index_rate > 0` 的 Index 缺失、feature-width mismatch 和 lookup
  failure 已 fail-closed；这保持旧产品契约，不是 realtime backend。
- machine-readable asset manifest 已核验 present 资产。FCPE 的 exact
  `torchfcpe==0.0.4` package/checkpoint provenance、locked tool project、byte-identical replay
  与 same-filesystem staged + process-local rollback 事务式发布已完成；这是逐文件
  `os.replace` 与进程内多文件回滚，不是 crash-consistent 的多文件原子事务。
- standalone exact-Index probe 已对 synthetic/available-feature 输入记录 CPU exact reference /
  persistent-GPU latency、
  persistent bytes 与 exact reference 对拍。它未接 CMake/CTest 或产品 Pipeline，明确是
  diagnostic probe，不是产品 backend、qualified rows/rate/batch 上限或 realtime SLO。
- `/tmp/resonantia-final.udNc8h` 的 current-tree 完整 CTest 为 29/29 passed（10 contract +
  19 integration、0 failures，real 331.28 秒）。该回归证明 R0 scaffold 加入后既有 suite 未
  回退，不是 realtime acceptance。

qualification 只剩两个显式 blocker：provenance-tracked 的目标声线 v2/40 kHz model，以及
与同一份有授权 corpus 匹配的 Flat Index。它们完成前不能进行代表性 target quality A/B、
full-pipeline Index SLO 或支持上限判定。persistent runtime、streaming session、`vc_realtime`
和 ALSA adapter 仍不存在，因此 R0 与 realtime 产品均不得标记 Complete。

## Verification and SDD impact

- R0/R1：现有 default contract、HuBERT/FCPE/Index/VITS/E2E/long-audio assets、archive/API/CLI
  compatibility 和 current Index baseline。
- R2：asset-free ring/state/DSP/error/rollback tests，fake slow executor/retrieval、未授权降级
  拒绝、粘滞 fallback 和 sanitizer/long-run host tests（平台可用时）。
- R3：真实人声/model/Index per-stage alignment、Index-enabled/no-index quality A/B、stream
  continuity、absolute length/frame accounting、repeated-call resource tests。
- R4：CLI process contract、fake ALSA seam、loopback/physical device negotiation/xrun/removal。
- R5：Orin Release 5 s warm-up + 10 min nominal benchmark、controlled overload、memory/thermal/
  clocks report，以及全部 current-tree CTest。

本 ADR 已获批并同步 Spec 006 的 decision table。当前只可把它写成 accepted target
architecture；实现完成并有门禁后，才把 runtime、CLI、Index SLO 或质量写成 current/as-built
事实。
