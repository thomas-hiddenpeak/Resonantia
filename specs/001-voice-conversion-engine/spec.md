# Feature Specification: Voice Conversion Engine

**Feature Branch**: `001-voice-conversion-engine`
**Created**: 2026-08-05
**Status**: Offline baseline and bounded HuBERT file path implemented; specification verification partially complete
**Scope date**: 2026-08-12

## 1. Scope

本规格定义当前已经实现的离线声音转换主链：

```text
WAV -> mono float32 -> 16 kHz -> HuBERT -> optional Flat Index
    -> RMVPE (or optional FCPE) -> F0 VITS v1/v2 -> 40 kHz model-rate WAV
```

默认内容编码器是 HuBERT。v1 synthesizer 使用 256D 特征，v2 使用 768D 特征。
WavLM 是已实现的可选编码器，但只能搭配在相同特征空间训练的 synthesizer，不能把
它当作 stock RVC 模型的无条件替代。

Web 编排见 Spec 003，训练见 Spec 002，声源分离见 Spec 004；它们不属于本规格的
离线推理完成条件。端到端实时变声由独立的
[Spec 006](../006-realtime-voice-conversion/spec.md) 与
[ADR-0002](../../docs/adr/0002-persistent-streaming-runtime-and-local-audio-boundary.md) 管理；
两者已于 2026-08-10 获批，但尚未实现 realtime 产品路径，也不改变本规格的离线完成条件。

## 2. Supported Contract

| 维度 | 当前支持 | 不支持 / 必须拒绝 |
|---|---|---|
| 输入、输出 | WAV | FLAC、MP3 和其他编码格式 |
| 后端 | CUDA | CPU |
| 计算精度 | FP32 | FP16 |
| 内容编码 | HuBERT（默认）；WavLM（可选且需匹配模型） | 任意不匹配的 encoder/synthesizer 组合 |
| RVC 模型 | F0 v1 / 256D；F0 v2 / 768D | non-F0 |
| F0 | RMVPE（默认）；FCPE（可选） | PM |
| 输出采样率 | 当前兼容 synthesizer 的原生 40 kHz | 32/48 kHz 等其他模型率；独立 target sample rate |
| 运行参数 | pitch、index rate、RMS mix、protect、F0 filter | formant shift |

`target_sample_rate=0` 表示使用当前兼容模型的原生 40 kHz；显式值只有等于
`model_sample_rate=40000` 才合法。这不是一个额外的输出重采样功能。

## 3. User Scenarios

### Story 1 — 单文件离线转换（P1）

用户提供 WAV、HuBERT、RMVPE、F0 synthesizer 和可选 Flat Index，CLI 输出模型
原生 40 kHz 的 WAV。writer 先在目标同目录完整写入唯一 staging 文件，完成
`fsync`/close 后以 rename 原子替换；任何校验、写入或提交失败都保留原目标。失败时返回
非零状态或结构化错误，不得把部分输出宣称为成功。

### Story 2 — v1 / v2 模型适配（P1）

v1 模型只接受 256D 内容特征，v2 模型只接受 768D 内容特征。版本、维度、speaker
范围、F0 指针和长度在进入 CUDA 计算前校验。

### Story 3 — 长音频（P1）

HuBERT `vc_convert`/`vc_batch` 使用 `convert_file_bounded`：seekable WAV 先形成稳定 16 kHz
spool，quiet planner、windowed F0、HuBERT/Index/VITS 与全局 RMS/atomic writer 都以有界
working set 执行，临时磁盘仍为 `O(duration)`。WavLM 尚无匹配的 WavLM-trained VITS 资格
资产，因此 CLI/batch 明确保留 legacy `convert_file`。公共 `convert_buffer` 和旧
`convert_file` 继续返回完整 owning audio，并保留全段 F0/reference 行为；这两个 legacy API
的 host/F0 working set 仍随时长增长。任一路径都不是低延迟 realtime/session 接口。

### Story 4 — 可选检索与参数更新（P2）

Flat Index 按 `index_rate` 混合检索特征；运行时允许更新已经实现的 pitch、
index rate、RMS mix 与 protect。任何保留但未实现的参数都应明确失败。

## 4. Functional Requirements

- **FR-001 WAV I/O**：读取 PCM/float WAV、下混至 mono float32，并写出合法 WAV。
- **FR-002 Resampling**：内容/F0 前端使用 16 kHz；当前输出保持兼容模型的原生 40 kHz。
- **FR-003 HuBERT default**：提供 v1 256D final projection 与 v2 768D layer feature。
- **FR-004 Optional WavLM**：仅在显式选择且提供匹配模型时启用。
- **FR-005 F0**：RMVPE 为默认；FCPE 可选；PM 必须返回不支持。
- **FR-006 Index**：校验 Flat Index 文件头、维度、范围和 `k`，加载为私有、已 unlink 的
  filesystem-backed snapshot，再以只读 mmap 执行既有精确加权 k-NN。snapshot 优先
  `FICLONE`，不支持时以 1 MiB bounded buffer 复制并核对；fallback 需要约一份 Index 的
  临时 filesystem 容量，失败必须 fail-closed。非零 `index_rate` 要求显式提供并成功加载与
  内容特征同宽的 Index；缺失、错维或检索失败不得静默按 no-index 返回成功。
- **FR-007 Synthesizer**：支持 F0 VITS v1/v2 的 TextEncoder、Flow 和
  NSF/HiFi-GAN Generator，拒绝 non-F0。初始化必须校验每个运行时已用 tensor 的
  F32 dtype、精确 shape/nbytes、float alignment 和有限值，并拒绝退化或不可有限重建的
  weight-norm。VITS checkpoint 必须不超过 256 MiB，并从单一只读 fd 复制到实例共享的
  read-only anonymous snapshot；源 fd 复制前后身份必须一致。模型路径在 init 时固定为
  绝对路径，checkpoint 路径身份在公开调用前改变时仍必须 fail-closed；其他模型消费者仍
  使用普通 file-backed safetensors mapping，不能把该 VITS 专用资源语义泛化到全部模型。
- **FR-008 Pipeline**：编排内容、F0、Index、protect、RMS mix、分块与拼接；提供 additive
  bounded file API，并保留 legacy owning buffer/file API。可恢复的模型、I/O、CUDA 与内部
  runtime failure 以失败结果返回；持续 host allocation failure 在无法构造结果时可传播。
- **FR-009 Numerical safety**：公开入口校验空指针、尺寸、乘法溢出和非有限数据，合成器
  的 infer/debug 入口不得返回含 NaN/Inf 的可观察输出；RMVPE/VITS 的 CUDA 临时 buffer 与其他
  CUDA/cuBLAS 临时资源必须异常安全。私有 test-only seam 对代表性的 allocation、H2D/D2H
  copy、pre-launch 与 observable synchronization failure 做故障注入；该范围不等于每个
  cuBLAS/cuFFT/CUDA 调用或真实系统 OOM 都已注入。
- **FR-010 Output**：成功时以同目录 staging + atomic rename 产生当前支持的 40 kHz
  model-rate WAV；失败
  返回错误、保留旧目标且不得伪造其他格式或采样率。
- **FR-011 CLI**：`vc_convert` 暴露当前支持参数并对未知/非法选项失败；数值 flags 使用
  全量消费、有限值、溢出与范围检查。`test_cli_contract` 以真实进程门控非法语法和退出码；
  `test_cli_bounded_callers` 以真实 HuBERT/RMVPE/v2/speech 资产门控 `vc_convert` 与单文件
  `vc_batch` 的成功 WAV 和 bounded caller 路由。HuBERT 走 bounded，WavLM 走 legacy。
- **FR-012 Auxiliary CLIs**：`vc_batch` 与 `vc_probe` 是构建目标；`vc_batch` 有严格解析和
  单文件真实 bounded caller 门禁，`vc_probe` 仍是无行为验收的骨架，整体属于部分实现。

## 5. Explicit Non-goals

- CPU 推理、FP16 推理和 PM F0。
- FLAC/MP3 解码或编码。
- non-F0 synthesizer。
- formant shift。
- 32/48 kHz 等其他 synthesizer 架构，以及与模型原生采样率不同的独立输出采样率。
- 麦克风、WebSocket 或端到端实时服务；已批准的本机实时能力由 Spec 006 单独管理，仍不是
  Spec 001 的完成条件。
- 在本规格内实现训练、WebUI 或声源分离。

## 6. Verification and Acceptance

默认无模型资产构建必须执行 contract 测试；模型与 fixture 齐全时，启用
`VOXMUTATIO_ENABLE_MODEL_TESTS=ON` 执行 integration。当前 CMake 在资产齐全配置下注册
**39 个 distinct tests**：16 个带 `contract` label、24 个带 `integration` label；
`test_bounded_file_wavlm_preflight` 同时带两个 label，因此 16+24 不能作为 distinct 总数。
2026-08-12 的 fresh Release/SM87 build `/tmp/resonantia-nonrt-final.a8O1Ay` 已实际执行
39/39（0 失败，`Total Test time (real) = 1297.94 sec`）。其中
`test_cli_bounded_callers` 为 75.89 秒、`test_long_audio_chunking` 为 438.00 秒、
`test_bounded_file_resources` 为 326.92 秒、`test_bounded_file_fcpe_v1` 为
249.34 秒。此前 30/30（11 contract、19 integration、356.41 秒）仅是加固前历史基线。

已记录的 focused evidence 包括最终 bounded 实现的 RMVPE/FCPE F0 profile、RMVPE/v2 与
FCPE/v1 的真实语音 bounded-vs-legacy 输出，以及 Orin/CC 8.7/Release 的 17.565/35.130 秒
资源对照：peak RSS
`1342624`/`1354844 KiB`，device growth `892334080`/`851812352` bytes，RMVPE 单次 F0
硬上界 `256000` samples。它只有两个时长点，且尚无独立冻结的 click/transient 指标；须与
结构性 hard bounds 一起解释，不能外推为任意时长实测保证；也不能把这两个 E2E pairing
写成四种 F0/version Cartesian combinations 都有独立 E2E。Spec 006 R0 测试也不扩大
Spec 001 产品范围，因此这些 focused 结果不足以把整个 001 规格标为完成。

验收重点：

- HuBERT v1/v2 输出维度正确，模型布局与 tensor 字节数在推理前复验。
- RMVPE/FCPE 初始化完整校验运行时使用的 tensor；失败重初始化不得保留旧状态，公开
  输入在 CUDA 前拒绝空指针、非法长度和非有限值。VITS 另需覆盖完整已用 tensor manifest、
  weight-norm 非退化/有限重建、256 MiB anonymous snapshot/绝对路径与文件身份变化、各公开
  入口的 frame 算术边界和非有限输出拒绝。
  RMVPE、FCPE、WavLM 与 VITS 只在对应资产实际存在并执行对齐测试后报告模型数值结果。
- pipeline 对 CPU、FP16、PM、non-F0、formant shift、非 WAV 和独立
  `target_sample_rate` 返回失败。
- bounded 长音频输出有限、非空、长度符合测试门槛，HuBERT/Index/VITS 按块执行且临时产物
  退出后清理；legacy buffer/file 与 WavLM caller 仍不承诺 duration-independent working set。
- 错误路径不泄漏 CUDA/cuBLAS 资源，边界尺寸不触发整数除零或有符号溢出。

## 7. Success Criteria

- **SC-001**：当前所有 16 个 contract-labelled tests 实际执行并通过；label overlap 必须按
  distinct CTest 口径报告。当前树已以 39 distinct / 16 contract-labelled /
  24 integration-labelled / 1 overlap 口径完成全量。
- **SC-002**：资产齐全时所有被注册的模型集成测试实际执行并通过；缺资产时清楚列出
  未注册项。
- **SC-003**：v1 256D 与 v2 768D 的版本/特征契约均有直接测试。
- **SC-004**：有效 WAV 转换产生有限、非空的 40 kHz model-rate WAV；失败由返回状态判断，
  调用方不得依据目标文件存在推断成功。
- **SC-005**：所有明确不支持组合稳定失败，而不是静默降级。
- **SC-006（部分满足）**：`vc_convert`/`vc_batch` 的严格参数语法与非法退出码由默认 CLI
  进程 CTest 门控；两者的真实模型 HuBERT bounded 成功路径已注册资产门禁；WAV writer 的
  成功替换与失败保留由默认 I/O 门禁覆盖，且它们已在当前 39/39 全量中通过。
  `vc_probe` 产品范围仍未决定。

本规格不以某个固定延迟、吞吐倍数或显存数字作为已达成结论；性能基准必须另附硬件、
模型、输入长度、编译配置和测量方法。

## 8. Implemented Migration — 有界离线文件转换

> 本节是 [ADR-0004](../../docs/adr/0004-bounded-offline-file-conversion.md) 的迁移契约与
> current implementation status。R1–R4 已实现；R5 已把 HuBERT CLI/batch caller 切到
> bounded API，WavLM 在匹配资产资格化前保留 legacy。旧 buffer/file API 仍在兼容窗口内，
> 当前 Release/SM87 最终全量 CTest 已 39/39 通过。本迁移只属于 Spec 001，
> 不修改 Spec 006 realtime 范围。

### 8.1 Compatibility boundary

- **VC-MIG-001 Buffer compatibility**：`convert_buffer(const AudioBuffer&)` 保持现有签名、
  owning 输入/完整 `VCResult::audio`、全段 F0 reference、错误与长音频拼接行为。由于输入与
  输出本身均为整段 vector，该 API 明确不承诺与时长无关的 host memory。
- **VC-MIG-002 Legacy file compatibility**：现有 `convert_file` 在至少一个明确迁移窗口内保持
  完整返回音频和数值行为；本迁移不得默认清空其 `VCResult::audio`，也不得猜测固定秒数后
  静默拒绝。CLI/batch 完成迁移和兼容门禁后才可标记 deprecated；以后增加上限、改返回或
  删除都需要单独批准。
- **VC-MIG-003 Additive bounded file API**：已新增同步离线
  `convert_file_bounded(input_path, output_path, speaker_id)`。成功以原子提交的 40 kHz WAV
  为 delivery，返回 success/error/timing，`VCResult::audio.data` 为空且 sample rate 为模型率；
  失败清理全部 staging/spool 并保留旧目标。它不创建 callback/session/ALSA/realtime 契约。

### 8.2 Resource and numerical contract

- **VC-MIG-004 Bounded working set**：新 file API 必须以 seekable 分段 WAV reader、一次一个
  core 的 quiet-cut planner、有限 context F0/crop、逐块 HuBERT/Index/VITS 及 incremental
  atomic writer 执行。单次 decoded/16 kHz/F0/synth buffer 和 CPU/GPU scratch 的上限只能
  依赖已验证的模型/Index、`max_chunk_sec`、`chunk_query_sec`、`chunk_pad_sec` 与固定 DSP
  halo，不得依赖总文件时长；不得在内存保留全部 input、energy/bounds、F0/UV 或 float output。
  为保持全局 resampling/high-pass/RMS 与文件事务而使用的 RAII 临时磁盘可为
  `O(duration)`，但必须有唯一所有者、checked byte range 和所有失败路径清理。
- **VC-MIG-005 Windowed F0 migration**：RMVPE/FCPE 从整段调用迁移为带双侧 context 的有限
  窗口；各 window 先生成 raw F0/UV，按全局 160-sample lattice crop 后写 spool，再以有界
  全局 pass 完成跨任意长 unvoiced run 的 interpolation、median filter 与 pitch shift。RMVPE
  BiGRU 和 FCPE GroupNorm 都没有可证明逐样本等价的有限 halo，因此该 additive API 明确不
  承诺与旧全段调用逐样本相同；profile context 必须分别用真实资产资格化。F0 门禁必须按
  VC-MIG-005A 的冻结定义测量，并比较 bounded file output 与旧 long-buffer reference：40 kHz、
  有限、非空、现有长度范围，以及 phase-invariant magnitude-spectrogram correlation
  `>0.98`；测量前不得宣称门槛成立，也不得用 synthetic audio 宣称模型/E2E 迁移正确。
- **VC-MIG-005A Frozen F0 profile and metrics**：首轮 raw F0 core 固定为 `128000` 个 16 kHz
  samples（8 秒）；RMVPE 单侧 context 固定为 `64000` samples（4 秒），FCPE 单侧 context
  固定为 `32000` samples（2 秒）。crop 后结果与 whole-clip reference 必须位于同一全局
  160-sample lattice。RMVPE/FCPE 分别资格化时，双方均 voiced 帧的 mean absolute F0 error
  `<0.5 Hz`、p95 absolute F0 error `<0.5 Hz`，全帧 UV agreement `>0.95`，内部 window
  边界双方均 voiced 帧的 max absolute F0 error `<0.5 Hz`；global max 只记录诊断，不作门禁。
  该 profile/阈值最初来自 2026-08-12 本地真实语音与真实模型资产探索，随后已在最终 R2–R4
  实现上重跑；它仍不构成 WavLM 或任意时长的证明。
- **VC-MIG-006 Resource evidence**：测试必须记录每次 F0 的最大输入和推导上限，并在隔离
  进程中用相同模型/config 比较至少两个不同时长的真实 WAV，记录 peak RSS/device memory、
  硬件、命令和允许的 allocator/driver slack。只有证明峰值不随总时长增长，才能把 bounded
  file path 标为实现；单个短 fixture、空返回 vector 或仅在返回前 `clear()` 均不满足。当前
  17.565/35.130 秒 gate 的 RSS 增量为 `12220 KiB`（允许 `65536 KiB`），device growth 长样本
  未高于短样本；这是两个时长点的证据，不是任意时长经验外推。

### 8.3 Rollout and rollback

1. R0 RED 先要求 additive API、空 owning result、完整原子 WAV、旧 buffer reference 和资源
   观测；该红灯已由 R1–R4 实现转绿。
2. R1 private bounded WAV reader/spool/writer 与 R2 quiet planner 已实现，旧 buffer 算法未被
   替换。
3. R3/R4 的最终真实 RMVPE/FCPE 数值、短/长资源和失败事务取得 focused 证据后，R5 将
   `vc_convert`/`vc_batch` 的 HuBERT caller 切到新入口；WavLM 仍走 legacy，Web 继续通过 CLI。
4. 任一旧 reference 回退、峰值随时长增长、Index 被静默旁路、临时文件泄漏、旧目标损坏、
   CTest 注册减少或错误变成功，均回滚当前阶段到旧 `convert_file` caller；不修改或删除用户
   输入、模型与既有目标。本次同步 DESIGN、TRACEABILITY、PROJECT_STATE 与 CLI/API 文档；
   旧 file API 的 deprecation 窗口仍由维护者另行决定。

### 8.4 Approved execution invariants

- 首个 pass 必须在 fd 身份/size 前后复验下，把 source WAV decode/downmix/resample 成稳定
  16 kHz raw-float spool；后续 pass 不得重新读取可变 source path。
- 递归 high-pass 必须以 double 全局 forward/reverse spool 实现；有限 halo window 不能替代
  全段 IIR state。synth raw float 先 spool，完整 output length/envelope 已知后再做全局 RMS 与
  PCM commit。
- 对 source `N>=2` 和 checked 16 kHz 配置 sample 数 `C/Q/P`：`Qeff=max(Q,160)`，
  `Peff=min(max(P,160),N-1)`，`Ceff=min(max(C,160),640000-(2Qeff+2Peff+160))`。core hard
  bound 为 `Ceff+2Qeff`，content/Index/VITS segment hard bound 为
  `Ceff+2Qeff+2Peff+160`。F0 使用独立 raw core：单次输入必须满足
  `input_samples <= 128000 + 2F`；RMVPE/FCPE 的
  `F` 分别为 64000/32000，并保持全局 160 lattice。所有换算须 checked。
- HuBERT bounded profiles 保持原 encoder；WavLM bounded 调用在 payload/staging 前
  fail-fast，CLI/batch 则显式选择 legacy API，任何路径都不能静默换 encoder。
- raw voice/spool 临时文件使用 mode `0600`；incremental WAV staging 保持现有 mode `0666` +
  caller umask 语义。exact sample count、header backpatch 与文件 `fsync`/close 必须在 rename
  前成功；rename 是不可回滚的 runtime commit point，之后父目录 `fsync` 仅 best-effort，不能
  把已提交结果改报失败。契约保证运行时原子替换，不宣称完整 directory crash durability。
