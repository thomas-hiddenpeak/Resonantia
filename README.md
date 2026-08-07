# Resonantia — VoxMutatio

> *Vox mutatur, mens manet.* (声音变化，意识不变。)

**[English](./docs/en/README.md)**

---

Resonantia 是 [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 的纯 C++/CUDA 重构实现，提供零 Python 依赖的声音转换引擎。

## 状态：可用 ✅

端到端语音转换已实现并验证，与 Python RVC 数值对齐（真实 LibriSpeech 语音）：

| 模块 | 指标 | 结果 |
|------|------|------|
| HuBERT 内容编码器 | RMS 误差 | 7.1e-07 |
| RMVPE F0 提取器 | 平均误差 | 1.5e-05 Hz |
| VITS 合成器 | 波形相关性 | 0.9997 |
| **端到端转换** | **波形相关性** | **0.9997** |

全部 13 个测试通过。`vc_convert` 可转换任意真实音频，`vc_train` 可从真实录音微调目标声线（见下方「实际使用工作流」）。

## 设计目标

- **纯 C++20/CUDA**：推理和训练链路完全脱离 Python/PyTorch 运行时
- **GPU 优先**：所有计算密集型算子在 GPU 上执行
- **safetensors 唯一权重格式**：零拷贝 mmap 加载，无 pickle 安全风险
- **数值对齐**：与 Python 原始实现逐样本对齐验证

## 架构

```
Resonantia/
├── CMakeLists.txt
├── include/
│   └── voxmutatio/
│       ├── core/             # 核心类型、配置、错误码
│       ├── separation/       # 声源分离 (HTDemucs/MDX-Net)
│       ├── content/          # 内容编码器 (WavLM + HuBERT)
│       ├── f0/               # F0 基频提取器 (RMVPE + FCPE)
│       ├── index/            # CUDA 特征检索 (手写 Flat Index)
│       ├── synthesizer/      # VITS 合成器 (NSF vocoder)
│       ├── training/         # 微调训练基础设施
│       ├── pipeline/         # 端到端推理管线编排
│       ├── io/               # 音频 I/O、safetensors 加载
│       └── webui/            # 嵌入式 WebUI 服务
├── src/
│   ├── core/                 # 核心实现
│   ├── separation/           # 声源分离 CUDA 实现
│   ├── content/              # WavLM/HuBERT CUDA 实现
│   │   ├── wavlm_encoder.{h,cu}
│   │   └── hubert_encoder.{h,cu}
│   ├── f0/                   # F0 提取 CUDA 实现
│   │   ├── rmvpe.{h,cu}
│   │   └── fcpe.{h,cu}
│   ├── index/                # CUDA 特征检索实现
│   │   └── cuda_flat_index.{h,cu}
│   ├── synthesizer/          # VITS 合成器 CUDA 实现
│   │   ├── text_encoder.{h,cu}
│   │   ├── flow.{h,cu}
│   │   ├── generator_nsf.{h,cu}
│   │   └── source_hn_nsf.{h,cu}
│   ├── training/             # 训练实现
│   ├── pipeline/             # 管线编排
│   │   └── pipeline.{h,cpp}
│   └── io/                   # 音频 I/O、权重加载
│       ├── audio_io.{h,cpp}
│       └── safetensors.{h,cpp}
├── tools/                    # 命令行工具
│   ├── vc_convert.cpp        # 离线单次转换工具
│   ├── vc_batch.cpp          # 批量转换工具
│   └── vc_probe.cpp          # 模型检查和数值验证探针
├── tests/                    # 测试
│   ├── test_hubert.cpp       # HuBERT 特征对齐测试
│   ├── test_f0.cpp           # F0 提取对齐测试
│   ├── test_synthesizer.cpp  # 合成器数值测试
│   └── test_pipeline.cpp     # 端到端管线测试
├── webui/                    # WebUI 前端资源
│   ├── index.html            # 主页面
│   ├── app.js                # 前端逻辑
│   └── styles.css            # 样式表
├── docs/                     # 设计文档
│   ├── DESIGN.md             # 架构设计总览
│   └── NUMERICAL_ALIGNMENT.md # 数值对齐策略
└── third_party/              # 第三方依赖
    └── cutlass/              # CUTLASS (submodule)
```

> **实现状态**：`content/hubert`、`f0/rmvpe`、`index`、`synthesizer`、`pipeline`、`training`（解码器微调）、`io` 已实现并数值验证；`separation/`、`content/wavlm`、`f0/fcpe`、`webui`（后端服务）、`third_party/cutlass` 为规划中，当前未接入运行时。

## 推理管线

```
输入音频 (任意采样率)
    │
    ▼
┌─────────────┐
│ 声源分离     │  ← HTDemucs/MDX-Net (可选)
│ (Optional)  │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ 重采样到16kHz │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  WavLM/     │  ← CUDA encoder (ContentVec, 256D/768D)
│  HuBERT     │
│  特征提取    │
└─────┬───────┘
      │
      ▼
┌─────────────┐     ┌─────────────┐
│   F0 提取    │────▶│  protect    │  ← 非声门脉冲保护
│ (RMVPE/FCPE)│     │  混合       │
└─────┬───────┘     └─────────────┘
      │
      ▼
┌─────────────┐
│  CUDA 索引   │  ← 手写 Flat Index (零 FAISS 依赖)
│  检索增强    │
└─────┬───────┘
      │
      ▼
┌──────────────────┐
│   VITS 合成器     │
│  ┌──────────────┐│
│  │ TextEncoder   ││  ← Transformer encoder
│  │  (Flow)      ││  ← Residual Coupling Flow
│  └──────┬───────┘│
│         │        │
│  ┌──────▼───────┐│
│  │ GeneratorNSF  ││  ← HiFiGAN + NSF 声源
│  │ (HiFiGAN+NSF) ││
│  └──────┬───────┘│
└─────────┼────────┘
          │
          ▼
┌─────────────┐
│ 重采样到目标  │  ← 40kHz / 48kHz
│ 采样率       │
└─────┬───────┘
      │
      ▼
  输出音频 (WAV/FLAC)
```

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 实际使用工作流

从真实录音训练目标声线，再用它转换任意语音/歌声。**运行时零 Python**——全部由
C++/CUDA CLI 完成（`tools/` 下的 Python 仅用于一次性下载与离线对拍）。

### 1. 准备模型

```bash
cd tools && uv run download_models.py && cd ..
```

需要 `models/` 下：`hubert_base/model.safetensors`（内容编码器）、`rmvpe.safetensors`
（F0 提取）、`pretrained_v2/pretrained_v2/f0G40k.safetensors`（40k 预训练 G）。

### 2. 训练目标声线

```bash
# 原始录音（一个 WAV 或一个目录）-> 切片 -> 微调 -> 检索索引
scripts/train_voice.sh --raw my_voice/ --work runs/alice --steps 300
```

产出 `runs/alice/model.safetensors`（微调模型）与 `runs/alice/model.index`（检索索引）。
流程：`vc_preprocess`（去静音 + 定长切片）→ `vc_train`（autograd 解码器 mel-L1 微调）
→ `build_index`。

### 3. 转换

```bash
scripts/convert_voice.sh \
  --model runs/alice/model.safetensors \
  --index runs/alice/model.index \
  --input song.wav --output song_alice.wav \
  --index-rate 0.5 --pitch 0
```

### 直接调用 CLI（脚本等价展开）

```bash
build/vc_preprocess --input my_voice/ --output-dir runs/alice/clips \
  --sr 40000 --seg-sec 3.0 --trim
build/vc_train --hubert models/hubert_base/model.safetensors \
  --rmvpe models/rmvpe.safetensors \
  --pretrained models/pretrained_v2/pretrained_v2/f0G40k.safetensors \
  --target runs/alice/clips --out runs/alice/model.safetensors --steps 300
build/build_index --hubert models/hubert_base/model.safetensors \
  --input-dir runs/alice/clips --output runs/alice/model.index
build/vc_convert --hubert models/hubert_base/model.safetensors \
  --rmvpe models/rmvpe.safetensors --model runs/alice/model.safetensors \
  --index runs/alice/model.index --index-rate 0.5 \
  --input song.wav --output song_alice.wav --version v2 --speakers 109 --sr 40000
```

> 训练当前为**解码器微调**（NSF-HiFiGAN 声码器适配目标音色），从预训练权重出发、
> 纯 C++/CUDA 自研 autograd 完成；导出模型与推理运行时逐比特兼容（往返 corr 1.0）。
> 完整 GAN（后验编码器 + 判别器）为后续增强。

## 参考项目

| 项目 | 角色 |
|------|------|
| [RVC-Project/RVC-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | 原始 Python 实现（重构目标） |
| [Qwen3x-Orin](https://github.com/thomas-hiddenpeak/Qwen3x-Orin) | C++/CUDA 引擎风格参考 |
| [Orator](https://github.com/thomas-hiddenpeak/Orator) | C++20 规范和测试风格参考 |
| [qwen35-thor](https://github.com/thomas-hiddenpeak/qwen35-thor) | SM110a 架构参考 |

## 许可证

MIT License
