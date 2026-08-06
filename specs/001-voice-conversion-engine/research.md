# Technical Research: Voice Conversion Engine

**Date**: 2026-08-07
**Status**: Complete

## 1. 内容编码器选型

### 候选方案
| 模型 | 维度 | 优势 | 劣势 |
|------|------|------|------|
| **WavLM-Base+** | 768D | 对噪音/混响鲁棒性强，SOTA | 计算量略大 |
| **HuBERT-Base** | 256D/768D | RVC 经典选择，生态完善 | 对干净音频依赖强 |

### 决策
**双轨制**：WavLM-Base+ 作为主要编码器，HuBERT-Base 作为 legacy 兼容。

### 实现路径
- 两者均为标准 Transformer Encoder 架构
- 复用 Qwen3x-Orin 项目已优化的 Attention/GEMM/LayerNorm CUDA Kernels
- 只需调整：hidden_dim (768), num_heads (12), activation (GELU), positional_encoding

## 2. F0 提取器选型

### 候选方案
| 模型 | 延迟 | 精度 | 适用场景 |
|------|------|------|----------|
| **RMVPE** | ~50ms | 高 | 离线/高质量 |
| **FCPE** | ~2ms | 中高 | 实时/低延迟 |

### 决策
**双轨制**：RMVPE 用于离线转换，FCPE 用于实时流式推理。

### 实现路径
- **RMVPE**: Conformer encoder (Conv + Transformer)，导出为 ONNX → TensorRT
- **FCPE**: 轻量 1D Conv + Pooling，手写 CUDA Kernel

## 3. 特征检索方案

### 候选方案
| 方案 | 依赖 | 性能 | 代码量 |
|------|------|------|--------|
| **FAISS C++ API** | 外部库 | 优秀 | 少 |
| **手写 CUDA Flat Index** | 零依赖 | 优秀 | ~200 行 |

### 决策
**手写 CUDA Flat Index**（宪法原则 V: Zero-Dependency Purity）

### 实现路径
- Shared Memory + Warp Shuffle 指令
- L2 Distance / Inner Product 计算
- GPU 暴力 Top-K 检索（10k 特征 < 1ms）

## 4. 声源分离方案

### 候选方案
| 模型 | 架构 | CUDA 移植难度 |
|------|------|---------------|
| **HTDemucs v4** | Hybrid Transformer | 高（需 ONNX→TensorRT） |
| **MDX-Net** | 2D Conv + U-Net | 低（手写 CUDA） |

### 决策
**MDX-Net 优先**（手写 CUDA 难度低于 Transformer）

## 5. 权重格式

### 决策
**safetensors 唯一格式**（宪法原则 IV）

### 理由
- 零拷贝 mmap 加载
- 无 pickle 安全风险
- 跨语言兼容（Python 转换 → C++ 加载）

### 转换工具
```python
# tools/convert_to_safetensors.py
import torch
import safetensors

state_dict = torch.load("model.pth", map_location="cpu")
safetensors.torch.save_file(state_dict, "model.safetensors")
```

## 6. 数值对齐策略

### 验证方法
| 模块 | 对比指标 | 阈值 |
|------|----------|------|
| HuBERT/WavLM | L2 误差 | < 1e-4 |
| F0 提取 | 绝对误差 | < 0.5 Hz |
| 特征检索 | Top-K 一致性 | 100% |
| 合成音频 | SRCC | > 0.999 |

### 工具
- `vc_probe.cpp`: 加载 Python 导出的参考张量，与 C++ 输出对比

## 7. 性能基准 (Jetson AGX Orin)

| 操作 | 目标延迟 | 备注 |
|------|----------|------|
| HuBERT 特征提取 (5s) | < 500ms | 16kHz → ~9 frames |
| RMVPE F0 提取 (5s) | < 250ms | Conformer encoder |
| CUDA Flat Index (10k) | < 1ms | Top-K 检索 |
| VITS 合成 (5s) | < 2s | TextEncoder + Flow + HiFiGAN |
| 端到端 (5s 音频) | < 5s | 完整 Pipeline |

## Open Questions
- [ ] VITS Flow 层的 Normalizing Flow 是否需要自定义 CUDA Kernel？
- [ ] HiFiGAN 的 ConvTranspose1D 是否使用 cuDNN 或手写？
- [ ] WebUI 前端技术栈选型（Vanilla JS vs 框架）？
