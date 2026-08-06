# Implementation Plan: Voice Conversion Engine

**Branch**: `001-voice-conversion-engine` | **Date**: 2026-08-07

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
*GATE: Must pass before Phase 0 research.*

| Principle | Compliance | Notes |
|-----------|------------|-------|
| I. Library-First | ✅ | 每个模块独立编译单元 |
| II. CLI Interface | ✅ | vc_convert, vc_batch, vc_probe |
| III. Test-First | ✅ | 测试骨架先行，数值对齐门控 |
| IV. safetensors-Only | ✅ | 零拷贝 mmap，无 pickle |
| V. Zero-Dependency Purity | ✅ | 手写 CUDA Index，无 FAISS/ONNX |
| VI. Numerical Alignment | ✅ | L2<1e-4, F0<0.5Hz, SRCC>0.999 |
| VII. Simplicity | ✅ | WebUI 简洁现代，不照搬 RVC-WebUI |
| VIII. Anti-Abstraction | ✅ | 直接使用 cuBLAS/CUDA，不包装 |
| IX. Integration-First | ✅ | test_pipeline.cpp 为最高优先级 |

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

### Phase 0: Infrastructure (P0)
- [ ] `core/types.cpp` - 错误码字符串化
- [ ] `core/device.cpp` - CUDA 设备管理
- [ ] `io/safetensors.cpp` - safetensors 零拷贝加载
- [ ] `io/audio_io.cpp` - WAV 读写 + 重采样
- [ ] 编译验证：`cmake --build build`

### Phase 1: Content Encoder (P1)
- [ ] `content/hubert_encoder.cu` - HuBERT CUDA 实现
- [ ] `content/wavlm_encoder.cu` - WavLM CUDA 实现
- [ ] `test_hubert.cpp` - 数值对齐测试 (L2 < 1e-4)

### Phase 2: F0 Extraction (P2)
- [ ] `f0/rmvpe.cu` - RMVPE Conformer CUDA 实现
- [ ] `f0/fcpe.cu` - FCPE 轻量 CUDA 实现
- [ ] `test_f0.cpp` - F0 对齐测试 (< 0.5 Hz)

### Phase 3: Feature Index (P3)
- [ ] `index/cuda_flat_index.cu` - 手写 CUDA Flat Index
- [ ] Index 检索测试 (Top-K 一致性 100%)

### Phase 4: Synthesizer (P4)
- [ ] `synthesizer/text_encoder.cu` - Transformer encoder
- [ ] `synthesizer/flow.cu` - Residual Coupling Flow
- [ ] `synthesizer/generator_nsf.cu` - HiFiGAN + NSF
- [ ] `test_synthesizer.cpp` - 音频对齐测试 (SRCC > 0.999)

### Phase 5: Pipeline (P5)
- [ ] `pipeline/pipeline.cpp` - 端到端编排
- [ ] `test_pipeline.cpp` - 集成测试
- [ ] `tools/vc_convert.cpp` - CLI 工具完善

### Phase 6: Polish (P6)
- [ ] `training/` - 微调训练基础设施
- [ ] `webui/` - WebUI 服务
- [ ] `tools/vc_batch.cpp` - 批量转换

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| 双内容编码器 (WavLM + HuBERT) | WavLM 鲁棒性强，HuBERT 生态兼容 | 只做 HuBERT 会丢失 WavLM 的噪音鲁棒性 |
| 双 F0 提取器 (RMVPE + FCPE) | 离线/实时双场景需求 | 只做 RMVPE 无法满足实时 < 2ms 延迟 |
| 手写 CUDA Index | 宪法原则 V (零依赖) | FAISS 引入大型外部依赖，违反纯度原则 |
