# Feature Specification: Voice Conversion Engine

**Feature Branch**: `001-voice-conversion-engine`
**Created**: 2026-08-05
**Status**: Library/pipeline baseline implemented; specification verification partially complete
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

超出单段上限的 WAV 在安静位置分块，带上下文转换后裁剪并拼接。HuBERT、Index 与 VITS
按块执行，但为了全局音高一致性，当前实现仍对整段 padded clip 一次计算 F0；file/buffer
路径也保留整段输入、前处理与输出。因此 host 内存和 F0 的 CPU/GPU working set 都随时长
增长；这不构成任意长度恒定内存或低延迟流式接口。

### Story 4 — 可选检索与参数更新（P2）

Flat Index 按 `index_rate` 混合检索特征；运行时允许更新已经实现的 pitch、
index rate、RMS mix 与 protect。任何保留但未实现的参数都应明确失败。

## 4. Functional Requirements

- **FR-001 WAV I/O**：读取 PCM/float WAV、下混至 mono float32，并写出合法 WAV。
- **FR-002 Resampling**：内容/F0 前端使用 16 kHz；当前输出保持兼容模型的原生 40 kHz。
- **FR-003 HuBERT default**：提供 v1 256D final projection 与 v2 768D layer feature。
- **FR-004 Optional WavLM**：仅在显式选择且提供匹配模型时启用。
- **FR-005 F0**：RMVPE 为默认；FCPE 可选；PM 必须返回不支持。
- **FR-006 Index**：内存映射 Flat Index，校验文件头、维度、范围和 `k`，执行精确
  加权 k-NN 混合。非零 `index_rate` 要求显式提供并成功加载与内容特征同宽的 Index；
  缺失、错维或检索失败不得静默按 no-index 返回成功。
- **FR-007 Synthesizer**：支持 F0 VITS v1/v2 的 TextEncoder、Flow 和
  NSF/HiFi-GAN Generator，拒绝 non-F0。初始化必须校验每个运行时已用 tensor 的
  F32 dtype、精确 shape/nbytes、float alignment 和有限值，并拒绝退化或不可有限重建的
  weight-norm；已验证 mmap 由实例持有，模型路径在 init 时固定为绝对路径，checkpoint
  文件身份在公开调用前改变时必须 fail-closed。
- **FR-008 Pipeline**：编排内容、F0、Index、protect、RMS mix、分块与拼接。
- **FR-009 Numerical safety**：公开入口校验空指针、尺寸、乘法溢出和非有限数据，合成器
  的 infer/debug 入口不得返回含 NaN/Inf 的可观察输出；RMVPE/VITS 的 CUDA 临时 buffer 与其他
  CUDA/cuBLAS 临时资源必须异常安全。
- **FR-010 Output**：成功时以同目录 staging + atomic rename 产生当前支持的 40 kHz
  model-rate WAV；失败
  返回错误、保留旧目标且不得伪造其他格式或采样率。
- **FR-011 CLI**：`vc_convert` 暴露当前支持参数并对未知/非法选项失败；数值 flags 使用
  全量消费、有限值、溢出与范围检查。`test_cli_contract` 以真实进程门控非法语法和退出码；
  真实模型的 CLI 成功 WAV 路径仍缺独立资产门禁。
- **FR-012 Auxiliary CLIs**：`vc_batch` 与 `vc_probe` 是构建目标；当前缺 CLI E2E 门禁，
  且 `vc_probe` 仍是骨架，属于部分实现。

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
`VOXMUTATIO_ENABLE_MODEL_TESTS=ON` 执行 integration。2026-08-12 current tip 的完整资产
环境验证为 30/30 CTest 通过（11 个 contract + 19 个 integration、0 失败，
`Total Test time (real) = 356.41 sec`），其中覆盖 HuBERT、RMVPE、FCPE、WavLM、Index、
VITS、pipeline、端到端 library/pipeline 转换和长音频；`test_long_audio_chunking` 为
223.61 秒。默认 `test_pipeline_smoke` 覆盖 VITS sparse-checkpoint、weight-norm、文件替换、
CWD 独立性、输入/输出 finite 与各公开入口 frame-size fail-closed 边界；RMVPE/VITS 算子临时
分配由 RAII 管理。该结果仍不包含真实模型 CLI 成功 WAV 或 `vc_probe` 行为验收；Spec 006
R0 测试也不扩大 Spec 001 产品范围，因此不足以把整个 001 规格标为完成。

验收重点：

- HuBERT v1/v2 输出维度正确，模型布局与 tensor 字节数在推理前复验。
- RMVPE/FCPE 初始化完整校验运行时使用的 tensor；失败重初始化不得保留旧状态，公开
  输入在 CUDA 前拒绝空指针、非法长度和非有限值。VITS 另需覆盖完整已用 tensor manifest、
  weight-norm 非退化/有限重建、实例 mmap/绝对路径与文件身份变化、各公开入口的 frame
  算术边界和非有限输出拒绝。
  RMVPE、FCPE、WavLM 与 VITS 只在对应资产实际存在并执行对齐测试后报告模型数值结果。
- pipeline 对 CPU、FP16、PM、non-F0、formant shift、非 WAV 和独立
  `target_sample_rate` 返回失败。
- 长音频输出有限、非空、长度符合测试门槛，HuBERT/Index/VITS 按块执行且临时产物退出后
  清理；当前整段 host memory 与 F0 working set 仍随时长增长。
- 错误路径不泄漏 CUDA/cuBLAS 资源，边界尺寸不触发整数除零或有符号溢出。

## 7. Success Criteria

- **SC-001**：当前 11 个默认 contract 测试全部通过。
- **SC-002**：资产齐全时所有被注册的模型集成测试实际执行并通过；缺资产时清楚列出
  未注册项。
- **SC-003**：v1 256D 与 v2 768D 的版本/特征契约均有直接测试。
- **SC-004**：有效 WAV 转换产生有限、非空的 40 kHz model-rate WAV；失败由返回状态判断，
  调用方不得依据目标文件存在推断成功。
- **SC-005**：所有明确不支持组合稳定失败，而不是静默降级。
- **SC-006（部分满足）**：`vc_convert`/`vc_batch` 的严格参数语法与非法退出码已由
  CLI 进程 CTest 门控，WAV writer 的成功替换与失败保留由默认 I/O 门禁覆盖；仍需真实
  模型的 CLI 成功转换门禁，并决定 `vc_probe` 的产品范围。

本规格不以某个固定延迟、吞吐倍数或显存数字作为已达成结论；性能基准必须另附硬件、
模型、输入长度、编译配置和测量方法。
