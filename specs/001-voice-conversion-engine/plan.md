# Implementation Plan: Voice Conversion Engine

**Branch**: `001-voice-conversion-engine` | **Date**: 2026-08-07
**Status**: 推理部分 Complete(P0–P5 已完成并数值对齐);训练与 WebUI(原 P6)已拆分为独立 spec 002/003。

> 事实基线见 [specs/PROJECT_STATE.md](../PROJECT_STATE.md)。本 plan 仅覆盖 001 的推理引擎范围。

## Summary

构建纯 C++20/CUDA 的声音转换引擎，完整实现 RVC (Retrieval-based Voice Conversion) 的推理和训练管线。核心模块包括：内容编码器 (WavLM/HuBERT)、F0 提取器 (RMVPE/FCPE)、CUDA 特征检索、VITS 合成器、端到端 Pipeline、CLI 工具和 WebUI。

## Technical Context

**Language/Version**: C++20 / CUDA 20
**Primary Dependencies**: cuBLAS, CUTLASS (submodule), CUDA Runtime
**Storage**: safetensors (zero-copy mmap)
**Testing**: CTest + 数值对齐探针
**Target Platform**: NVIDIA Jetson AGX Orin (SM87)
**Project Type**: 独立 C++ 库 + CLI 工具 + WebUI 服务
**Performance Goals**: 端到端 < 5s (5s 音频, Jetson AGX Orin)

## Constitution Check
*GATE: 宪法 v1.4.0(11 条原则)。*

| Principle | Compliance | Notes |
|-----------|------------|-------|
| I. Library-First | ✅ | 每个模块独立编译单元 |
| II. CLI Interface | ✅ | vc_convert, vc_batch, vc_probe, build_index |
| III. Test-First | ✅ | 数值对齐门控;10/10 测试通过 |
| IV. safetensors-Only | ✅ | 零拷贝 mmap,无 pickle |
| V. Zero-Dependency Purity | ✅ | 手写 CUDA Index,无 FAISS/ONNX |
| VI. Numerical Alignment | ✅ | HuBERT 7e-7 / F0 1.5e-5Hz / 音频 0.9997 |
| VII. Simplicity | ✅ | 不照搬 RVC-WebUI |
| VIII. Anti-Abstraction | ✅ | 直接使用 cuBLAS/CUDA |
| IX. Integration-First | ✅ | test_e2e_conversion 为最高优先级 |
| X. Real Audio Only | ✅ | 全部用 LibriSpeech 真实语音对齐 |
| XI. Python Isolation | ✅ | Python 仅在 tools/(uv);运行时零依赖 |

## Project Structure

### Documentation (this feature)
```
specs/001-voice-conversion-engine/
├── spec.md               # 功能规格
├── plan.md               # 实施计划 (本文档)
├── research.md           # 技术调研
└── data-model.md         # 数据模型
```

### Source Code (repository root)
```
include/voxmutatio/
├── core/                 # P0: 核心类型、设备管理
├── io/                   # P0: 音频 I/O、safetensors 加载
├── content/              # P1: WavLM/HuBERT 编码器
├── f0/                   # P2: RMVPE/FCPE F0 提取
├── index/                # P3: CUDA Flat Index
├── synthesizer/          # P4: VITS 合成器
├── pipeline/             # P5: 端到端编排
├── training/             # P6: 微调训练
└── webui/                # P6: WebUI 服务

src/                      # 对应实现文件 (.cpp, .cu, .cuh)
tools/                    # CLI 工具
tests/                    # 测试套件
```

## Implementation Phases

> 状态图例:✅ 完成并验证 · ⚠️ 骨架/桩 · ➡️ 拆分到独立 spec

### Phase 0: Infrastructure (P0) — ✅ 完成
- [x] `core/types.cpp` - 错误码字符串化
- [x] `core/device.cpp` - CUDA 设备管理
- [x] `io/safetensors.cpp` - safetensors 零拷贝加载(健壮 tokenizer,213/741/560 张量)
- [x] `io/audio_io.cpp` - WAV 读写 + 重采样
- [x] 编译验证:`cmake --build build`

### Phase 1: Content Encoder (P1) — ✅ HuBERT 完成 / ⚠️ WavLM 桩
- [x] `content/hubert_encoder.cpp` - HuBERT (ContentVec) CNN 特征提取器 + 12 层 Transformer
- [ ] `content/wavlm_encoder` - ⚠️ 文件存在但未接入/未验证(规划中)
- [x] `test_hubert_inference.cpp` - 数值对齐(RMS 7.1e-07 < 1e-4)

### Phase 2: F0 Extraction (P2) — ✅ RMVPE 完成 / ⚠️ FCPE 未实现
- [x] `f0/rmvpe.cpp` + `rmvpe_ops.cu` - DeepUnet + BiGRU CUDA 实现
- [ ] `f0/fcpe` - ⚠️ 未实现(规划中)
- [x] `test_rmvpe_alignment.cpp` - F0 对齐(1.5e-05 Hz < 0.5 Hz)

### Phase 3: Feature Index (P3) — ✅ 完成
- [x] `index/cuda_flat_index.cpp` - 加权 k-NN 检索 + build_index 工具
- [x] `test_index_retrieval.cpp` - 检索一致性(cosine 1.0)

### Phase 4: Synthesizer (P4) — ✅ 完成
- [x] TextEncoder(相对位置注意力) + Flow(WaveNet 耦合) + GeneratorNSF
- [x] `test_vits_alignment.cpp` - 音频对齐(相关性 0.9997 > 0.999)

### Phase 5: Pipeline (P5) — ✅ 完成
- [x] `pipeline/pipeline.cpp` - 端到端编排(含 index_rate 混合)
- [x] `test_e2e_conversion.cpp` - 集成测试(相关性 0.9997)
- [x] `tools/vc_convert.cpp` - CLI 工具可用

### Phase 6: 训练 + WebUI — ➡️ 拆分为独立 spec
- ➡️ `training/` 微调训练 → **spec 002-training**(当前仅 TODO 桩)
- ➡️ `webui/` WebUI 服务 → **spec 003-webui**(前端存在但无 server 可执行文件)
- [ ] `tools/vc_batch.cpp` - ⚠️ 骨架,待随对齐更新验证

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| 双内容编码器 (WavLM + HuBERT) | WavLM 鲁棒性强，HuBERT 生态兼容 | 只做 HuBERT 会丢失 WavLM 的噪音鲁棒性 |
| 双 F0 提取器 (RMVPE + FCPE) | 离线/实时双场景需求 | 只做 RMVPE 无法满足实时 < 2ms 延迟 |
| 手写 CUDA Index | 宪法原则 V (零依赖) | FAISS 引入大型外部依赖，违反纯度原则 |
