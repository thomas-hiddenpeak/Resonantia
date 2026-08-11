# Implementation Plan: Realtime Voice Conversion

**Feature**: `006-realtime-voice-conversion`
**Status**: Approved; R0 evidence is partially implemented; realtime product path remains unimplemented
**Updated**: 2026-08-10
**Governing spec**: [spec.md](spec.md)
**Architecture decision**:
[ADR-0002](../../docs/adr/0002-persistent-streaming-runtime-and-local-audio-boundary.md)

## 1. Preconditions

维护者已于 2026-08-10 批准 RT-DEC-001..004，ADR-0002 已为 `Accepted`。实施从 R0 开始；
开始 R1 前仍必须同时满足：

1. R0 保存 current-tree 离线数值、错误、内存和性能基线；
2. R0 单独保存 standalone CPU exact reference 与 persistent-GPU prototype 的诊断性
   rows/dim/`k`/query-batch latency、内存与 exact-retrieval evidence；该 feasibility 已测，但
   当前仍没有代表性目标声线 Index/full-pipeline baseline，不能据此资格化 backend 或上限；
3. [R0 asset manifest](r0-assets.json) 记录真实音频/模型/目标声线 Index 的来源、revision、
   许可证、SHA-256 和生成命令；当前 present 文件完整性与 FCPE exact package/checkpoint
   provenance 已核验，但目标声线 model+Index pair 仍使 qualification gate 保持 blocked；
4. 为即将改变的行为先建立能失败的 focused test 或可复现 characterization。

已批准组合是：本机 ALSA；v2/HuBERT/FCPE/40 kHz，Index-enabled 为正常质量路径并保留
显式 no-index fallback；p95 `<= 250 ms`、p99 `<= 350 ms`；bounded drop-oldest。不得用
no-index benchmark 代替 Index-enabled 主验收，也不得在未授权时静默关闭 Index。

## 2. Compatibility envelope

本计划新增 realtime path，不重定义 Spec 001：

- `voxmutatio` 继续是公共 static facade；现有 target、公共 header、CLI 和安装面保持可用。
- `vc_convert`/`vc_batch` 的 argv、默认值、退出码、日志依赖字段和 WAV 提交语义不变。
- WAV、safetensors、Flat Index、F0 v1/v2 兼容和 CUDA FP32 支持矩阵不变；realtime 的
  Index 路径须保持离线 exact weighted k-NN 语义，任何加速先对拍。
- 既有 `Synthesizer::infer_stream` characterization 已冻结当前 crop/错误/phase-reset 行为；后续
  决定保留或弃用时不得静默改变这些兼容事实。
- 共享 persistent runtime 的任何阶段都必须先证明离线逐 stage/E2E 数值不回退。

## 3. R0–R5 migration

### R0 — Approval, characterization and baseline

**Goal**：冻结当前事实，补齐批准与资产证据；不新增 realtime runtime。characterization 中
发现的离线 Spec 001 Index 静默旁路偏差按原规格 fail-closed 修正，不代表流式能力实现。

- [x] 记录维护者对 RT-DEC-001..004 的明确决定并接受 ADR-0002（2026-08-10）。
- [x] 固化 `infer_stream` 的完整 infer + crop 行为、40 kHz 下 400 samples/frame、非法窗口、
  repeated-call bit-exact 与每次调用 NSF phase reset 现状；公共注释明确它不是 realtime session。
- [ ] 用真实 LibriSpeech/HuBERT/RMVPE/FCPE/VITS 保存离线逐 stage 与 E2E 输出、timing、
  RTF、峰值内存和模型/fixture SHA-256。
- [ ] 取得由同一份有授权 corpus 产生、彼此匹配的目标声线 v2/40 kHz model+Flat Index，固定
  model/config/speaker/hash、Index rows/dim/`k`/`index_rate`/query batch 与 source/license；
  分别记录 current host exact search 和 Index-enabled 离线 pipeline 的
  p50/p95/p99/max、内存、质量。当前 tree 与 [R0 manifest](r0-assets.json) 都明确记录该
  target pair 为 `missing`；generic pretrained `f0G40k` 不替代目标模型，也没有可用于 realtime
  资格判断的 Index baseline。
- [x] 建立 standalone R0 exact-retrieval probe；它对拍 top-k distance、归一化 weight、纯
  retrieved 与 blended output，并记录 standalone CPU-reference/GPU
  synthetic/available-feature latency 与显存。
  probe 明确 `diagnostic_only=true`、未接 CMake/CTest，数据不得称为 qualified target 上限。
- [x] 固定 FCPE 的 official package release、bundled checkpoint hash 与许可证，并证明当前
  `models/fcpe/model.safetensors` 可由 `torchfcpe==0.0.4` 精确输入逐字节复现；export 脚本在
  加载前校验 package version/checkpoint SHA-256 并强制 weights-only。
- [ ] 使用真实目标声线/真实语音建立 Index-enabled 与 no-index 的盲听/客观 A/B 基线；
  在看到数据前不虚构“竞争性质量”阈值，R0 review 后将批准门槛写回 Spec。
- [x] 建立内部 `BenchmarkReport` 构造期 validator，覆盖环境、profile、warm-up、
  p50/p95/p99/max、各 stage（含 Index）、retrieval mode/fallback、drop/xrun、内存、温度和
  降频字段，并按 Index-enabled/no-index/sticky fallback 模式拒绝矛盾证据。
- [ ] 为完整 realtime runner 建立稳定序列化与完整 outcome 字段 population；当前 validator
  没有 producer，standalone Index probe 使用独立低层 JSON schema，不能冒充 10-minute
  session report。
- [x] 修复 `tests/fixtures/README.md` 与 `models/README.md` 的实际路径漂移，并建立
  [machine-readable R0 manifest](r0-assets.json)；2026-08-10 已验证所有 `present` 项的
  path/bytes/SHA-256；FCPE provenance 已补全，目标声线 model/Index/corpus 缺失仍为显式
  blocker，没有伪造补全。
- [x] 2026-08-10 R0 快照完整回归 29/29 CTest 通过（10 contract + 19 asset-gated
  integration、0 失败，real 331.28 秒）；当前全局回归见 `../PROJECT_STATE.md`。该快照只
  证明当时既有 suite 未退化，不是 realtime 产品/SLO 验收。
- [ ] 用 fake executor/PCM seam 先写 session lifecycle、bounded ring、overflow、reset/drain 和
  init rollback 的 failing tests。

**Gate**：R0 evidence 可复现，且没有把当前 RTF、standalone Index 诊断性能或尚未测量的
target/full-pipeline Index quality 写成达标结论。Index-enabled 主路径的支持上限和 fallback
trigger 必须由 representative target evidence 固化。

**Rollback**：只撤销新增 proposal/baseline artifacts；不接触模型、fixture 或用户 run data。

### R1 — Persistent CUDA execution runtime

**Goal**：先消除离线执行中的逐算子权重传输和临时资源抖动，同时保持输出契约。

- [ ] 在现有 content/F0/synthesizer 组件内部建立显式、move-only 的 device weight owners。
- [ ] session/runtime 持有单一 CUDA stream、cuBLAS/cuFFT handles 和 profile-bounded workspace。
- [ ] init 只选择 HuBERT、FCPE、VITS v2 与一个已验证 Flat Index；不同时加载
  RMVPE/WavLM/多个 Index。显式 no-index 模式不创建 Index state。
- [ ] 权重 staging、shape/dtype/finite/byte-range 校验和 device upload 全部成功后原子提交。
- [ ] 移除 realtime 执行对 process-global synthesizer loader cache 的依赖；生命周期和内存
  必须由对象表达。
- [ ] 让 tensors 在相邻算子间保持 device-resident，只在输入/最终输出/明确观测点跨边界。
- [ ] 基于 R0 数据选择有界的 Index 执行策略（保留 mmap exact scan、批处理/并行优化或
  device-resident retrieval）；无论 backend，先与现有 exact weighted k-NN reference 对拍，
  并为 rows/dim/`k`/workspace 设置启动时上限。
- [ ] 为每个 kernel launch、async copy、event 和 stream handoff 建立错误可见点。

**Gate**：离线 HuBERT、FCPE、VITS、Index retrieval 和 E2E 真实资产门槛不回退；重复 call
内存稳定；focused benchmark 证明没有稳态 model reload、per-op weight H2D 或 per-op
allocation，并记录而非猜测 Index stage latency/内存。

**Rollback**：保留旧 host-vector executor behind internal selection，回到 R0 facade；不得修改
权重格式或覆盖用户模型。

### R2 — Stateful stream core and scheduler

**Goal**：在无 ALSA 依赖的 deterministic replay 中建立完整 session 语义。

- [ ] 新增 proposed `streaming` internal component 和 public session contract；不让它反向依赖
  app/ALSA/Web。
- [ ] 建立 preallocated SPSC input/output rings、sequence/timestamp、fixed-capacity overlap 和
  immutable runtime-parameter snapshots。
- [ ] 实现 state machine：Uninitialized → Ready → Running → Draining → Stopped；fatal → Failed。
- [ ] 实现 stateful input/output resampling、causal filter state、reset/drain/stop。
- [ ] fake executor 覆盖 nominal、slow worker、drop-oldest、underrun concealment、fatal error 和
  bounded memory。
- [ ] fake retrieval seam 覆盖 Index-required、explicit no-index、未授权不得降级、预授权的
  sustained-deadline 粘滞 fallback、transition reason/generation/counters，以及 fallback 后仍
  按 drop-oldest 保持 queue 有界。
- [ ] 使用 `steady_clock` 定义 queue/worker latency；不以 `high_resolution_clock` 作为 SLO
  时间源。

**Gate**：所有 asset-free contract 通过；受控 overload 中 queue 永不越界，drop/sequence
gap/Index fallback 计数精确；audio critical path 无 allocation、blocking mutex、file/network/
log I/O。no-index 不得由实现自行选择。

**Rollback**：从 `voxmutatio` facade 和工具图移除新增 streaming objects；R1 offline runtime
仍可独立保留或按其数值门禁回滚。

### R3 — Model windowing, state and continuity

**Goal**：把 persistent HuBERT/FCPE/Index/VITS profile 接到 session，并验证跨块质量。

- [ ] 在固定 matrix 上研究 `hop × context × lookahead × crossfade`，只从 10/20 ms 对齐值选择。
- [ ] HuBERT 使用有限滑窗和 emit-center crop；不声称 causal/KV cache。
- [ ] FCPE 使用显式 receptive-field context 和跨块 F0/UV/filter/protect state。
- [ ] 对每个 emit window 执行已批准的 Index retrieval，并把 query batching/context crop 与
  离线 exact reference 对齐；Index 不得因窗口边界重复/遗漏 query frame。
- [ ] VITS 保留 NSF harmonic phase，或使用经数值/连续性批准的 crop/crossfade；不得每块把
  phase reset 伪装为连续输出。
- [ ] 保持 source/output sample accounting，不允许累计 frame jitter 或隐式 time stretch。
- [ ] 逐 stage 对齐使用真实模型/真实人声；完整 stream 使用频谱、时长和 boundary metric。
- [ ] 在同一输入、模型、窗口和目标声线下分别运行 Index-enabled/no-index A/B；质量报告
  显式标注 mode，fallback hop 不计入 Index-enabled quality/SLO 样本。
- [ ] 记录不同候选的 p50/p95/p99、RTF、内存和质量，选择满足 Spec 的最小窗口。

**Gate**：真实资产 stream integration 通过；逐 stage（含 Index exact retrieval）仍满足
Constitution/Spec 001 门槛；完整 stream 无 NaN/Inf，建议 spectral correlation `> 0.98`，
boundary 和 Index 质量阈值由 R0 记录固化；Index-enabled nominal replay 0 drop/fallback。

**Rollback**：切回 R2 fake executor；不得以放宽 001 tolerance、修改 reference 或删除
failure test 使候选通过。

### R4 — ALSA adapter and thin CLI

**Goal**：在 app boundary 交付本机 capture/playback，不让 ALSA 泄漏进模型/runtime API。

- [ ] 新增 command-scoped `vc_realtime` app-support 与薄 main；ALSA 仅为该 target 的依赖。
- [ ] startup 严格解析 models/Index/retrieval mode/fallback authorization/device/speaker/pitch，
  完成 device negotiation、runtime init 和 warm-up 后才进入 capture。
- [ ] capture、GPU worker、playback 各有明确 owner 和 shutdown ordering；Ctrl-C 只请求停止，
  不在 signal handler 释放 CUDA/ALSA 资源。
- [ ] 报告实际 PCM format/rate/channels/period/buffer；不静默采用不同格式。
- [ ] 覆盖 xrun recovery、recovery limit、device removal、playback underrun 和 fatal exit。
- [ ] CLI help 只显示已支持 profile/参数，并清楚区分默认 Index-required、显式 no-index 和
  可选的预授权 overload fallback；不显示 WebSocket/RMVPE/WavLM future knobs。

**Gate**：真实 CLI process contract、ALSA fake seam 和批准的 loopback/physical integration
通过；默认不录音、不写用户音频；fatal failure 非零退出且停止设备线程。

**Rollback**：删除/禁用新增 executable 和 ALSA app-support target；公共离线 facade 不变。

### R5 — Acceptance, hardening and SDD synchronization

**Goal**：只有完整证据满足 Spec 时才宣称 realtime。

- [ ] 在固定 Orin NX/Release/Index-enabled profile 与固定 Index 上完成 warm-up + 10-minute
  nominal replay；另跑 no-index fallback，不混合结果。
- [ ] 验证 p95/p99 SLO、compute deadline、0 nominal drop/xrun/fatal/non-finite。
- [ ] 验证 Index-enabled nominal 为 0 fallback，并报告 Index stage latency 与 rows/dim/`k`/
  `index_rate`；no-index 结果不得替代主验收。
- [ ] 运行 controlled overload、device removal、CUDA failure、reset/restart 和 repeated
  start/stop；验证有界恢复与清理。
- [ ] 记录 process/system/CUDA peak 与 steady memory、温度、power mode、clocks 和降频。
- [ ] 运行全部现有 default contract 与 asset integration；任何 skip、未注册或缺资产单列。
- [ ] 同步 Spec 001/003 cross-reference、DESIGN、PROJECT_STATE、TRACEABILITY、README、CLI
  文档和 ADR outcome；本步骤不提前把 proposal 写成 current fact。

**Gate**：RT-SC-001..008 全有实现位置和实际执行证据，才可调整 Spec 状态。局部 helper、
短时 smoke 或平均 latency 不满足完成条件。

**Rollback**：保留离线产品路径和 R0 evidence；关闭 realtime target/feature，不覆盖模型、
音频或 run data。若 shared runtime 影响离线数值/错误语义，回到最后一个通过全部离线门禁的
stage。

## 4. Verification order

每阶段遵循 red → green → refactor：

1. 最小 asset-free target/focused CTest；
2. 受影响的 existing contract suite；
3. 对应 real-model/real-speech/real-Index integration；
4. 专用 benchmark/hardware integration；
5. 最后才运行完整 current-tree suite 与 SDD checks。

性能 benchmark 不能替代数值门禁，普通 CTest 通过也不能替代 Orin/ALSA SLO。所有报告
必须列出命令、环境、资产、pass/fail/skip 和未验证项。

## 5. Dependency and deployment discipline

- ALSA 是首版唯一拟新增系统依赖，并只链接 `vc_realtime` app boundary；core library 在
  不构建 realtime CLI 时仍应可配置。
- 不引入 Python、PyTorch、ONNX Runtime、TensorRT、PortAudio、WebSocket 或浏览器运行时。
- 不请求 root 权限作为正常运行契约。锁 clocks 若用于正式基准，应作为记录明确的实验步骤，
  不是 runtime 隐式副作用。
- WebSocket 或常驻多会话 model service 需要新的 Spec/ADR 决策，不能在 R4 顺手加入。
