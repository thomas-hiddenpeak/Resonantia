# Implementation Plan: Voice Conversion Engine

**Branch**: `001-voice-conversion-engine`
**Status**: Library/pipeline implemented; CLI acceptance incomplete
**Updated**: 2026-08-09

## Current Delivery

当前交付是 Jetson/SM87 面向的 C++20/CUDA20 离线推理链。运行时模型为
safetensors，内部音频为 mono float32，模型计算为 CUDA FP32：

```text
WAV -> 16 kHz front-end -> HuBERT(default) -> optional Flat Index
    -> RMVPE(default) or FCPE -> F0 VITS v1/v2 -> 40 kHz model-rate WAV
```

## Implemented Phases

### P0 — Build, device and contracts

- [x] CMake C++20/CUDA20，默认 SM87，可自动发现 CUDA compiler。
- [x] CUDA device 初始化、RAII buffer 和错误传播。
- [x] 无本地模型资产的 contract 层。

### P1 — Core I/O and weights

- [x] WAV 读取/写入、mono 下混与线性重采样。
- [x] safetensors 解析与布局/字节数校验；模型消费者按各自 manifest 校验必需权重的
  dtype、shape、数值与模型不变量。
- [x] 明确拒绝非 WAV 音频格式。

### P2 — Content and F0

- [x] HuBERT 默认路径：v1 256D、v2 768D。
- [x] WavLM 可选实现与资产门控对齐；要求匹配训练的 synthesizer。
- [x] RMVPE 默认实现。
- [x] FCPE 可选实现与资产门控对齐。
- [x] PM 保留枚举但在 pipeline 明确拒绝。

### P3 — Index and synthesizer

- [x] 内存映射 Flat Index 与精确加权 k-NN。
- [x] F0 VITS v1/v2：TextEncoder、Flow、NSF/HiFi-GAN Generator。
- [x] 版本、特征维度、speaker、pitch、F0 指针和尺寸边界校验。
- [x] non-F0 明确拒绝。

### P4 — Pipeline and CLI

- [x] HuBERT/RMVPE/Index/VITS 编排。
- [x] protect、RMS mix、pitch、F0 filter。
- [x] 安静点分块、上下文 padding、拼接和输出长度边界。
- [x] `vc_convert` 与当前支持的 40 kHz model-rate WAV 输出；失败返回非零状态。
- [x] WAV 目标文件同目录 staged write + `fsync`/close + atomic rename；失败保留旧目标并
  清理 staging。
- [x] CPU、FP16、formant shift、非 WAV、独立 target sample rate 的失败契约。

### P5 — Verification

- [x] 默认 contract：最近一次验证 11/11，包含 CLI/HTTP 进程、legacy Web API 与 realtime 报告
  validator；后者不扩大 Spec 001 产品范围。
- [x] 本地资产齐全时：最近一次完整验证 30/30 CTest（11 contract + 19 integration、
  0 fail，real 335.05 秒）。
- [x] HuBERT、RMVPE、FCPE、WavLM、VITS、端到端和长音频均由资产门控的
  integration 测试覆盖。
- [x] `vc_convert`/`vc_batch` 非法参数语法与退出码的真实进程 CTest。
- [x] 严格数值解析，拒绝 malformed、非有限、溢出、空白、十六进制浮点、尾随字符和
  越界值。
- [ ] 真实模型 `vc_convert` 成功 WAV 的 CLI 资产门禁，以及 `vc_probe` 的产品验收；
  当前 `vc_probe` 仍是骨架。

因此 30/30 是当前库/模型与已注册 CLI/HTTP/R0 scaffold 契约证据，不是整个 001 规格
Complete 的依据。

## Current Constraints

- 只支持 CUDA FP32 和 WAV I/O。
- HuBERT 是默认内容编码器；WavLM 不是 stock RVC 模型的透明替代。
- 当前只支持 40 kHz synthesizer；输出保持该模型率，不提供 32/48 kHz 架构或独立重采样。
- 长音频只把 HuBERT/Index/VITS 约束到单块；F0 仍整段执行，file/buffer pipeline 的 host
  内存与 F0 CPU/GPU working set 都随时长增长。
- `infer_stream` 等局部 helper 不等于完整实时产品链。
- 模型/fixture 不入 Git；CMake 必须在注册 integration 前检查资产。

## Superseded Historical Decisions

以下内容曾出现在早期计划中，但已被当前实现或规格拆分取代，不得再当作现状：

| 历史决策 | 当前结论 |
|---|---|
| WavLM 作为主路径、HuBERT 作为 legacy | **已取代**：HuBERT 是默认；WavLM 仅可选且需匹配模型 |
| FLAC 输入属于基线 | **已取代**：当前音频 API 仅支持 WAV |
| CPU fallback、FP16 或 PM 可自动降级 | **已取代**：这些配置明确失败 |
| `target_sample_rate` 可独立控制输出 | **已取代**：只能为 0 或等于当前支持的 40 kHz 模型率 |
| 固定秒数、实时倍数或显存数字可作完成证据 | **已取代**：没有带完整环境元数据的可复用基准 |
| 在 001 内交付训练与 WebUI | **已取代**：训练、Web 和分离分别由 002、003、004 管理 |

## Next Work

端到端实时变声已拆分到 Spec 006 和 ADR-0002：决策已批准、R0 证据部分完成，但产品
runtime/session/ALSA 入口仍未实现，且 qualified target model+matching Index/corpus 仍阻塞。
不得从离线分块或 standalone Index probe 推导“已实时”。
