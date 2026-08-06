# Resonantia 架构设计

## 1. 目标

将 [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 的推理和训练管线从 Python/PyTorch 完全迁移到纯 C++20/CUDA，消除对 Python 运行时的依赖，同时保持数值输出与原始实现的一致性。

### 1.1 使用场景
1. **独立 CLI 工具**：离线批量声音转换
2. **WebUI 模式**：嵌入式 HTTP 服务 + 现代前端界面
3. **库模式**：作为独立 C++ 库被其他项目集成

## 2. 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | **纯 C++20 / CUDA** — 零 Python 依赖 |
| 构建 | CMake 3.24+, nvcc (SM87), C++20 + CUDA 12 |
| 风格 | Google C++ Style + 宪法治理 (static_assert/constexpr) |
| GEMM | cuBLAS + CUTLASS (submodule) |
| 权重格式 | **safetensors 唯一格式**（零拷贝 mmap 加载） |
| 精度 | FP32 为主，FP16 可选（与 RVC `is_half` 配置对齐） |
| 音频 I/O | WAV/FLAC 自研读写 + 重采样 |
| 特征检索 | **手写 CUDA Flat Index**（零 FAISS 依赖） |
| WebUI | httplib (嵌入式 HTTP) + 纯前端 (HTML/JS/CSS) |
| 测试 | CTest + 自定义数值对齐探针 |

## 3. 宪法治理特性

参考 Orator/Qwen3x-Orin 项目，使用编译时验证确保代码正确性：

```cpp
// 编译时维度验证
static_assert(kHubertHiddenDim == 768U);
static_assert(kVitsSegmentSize == 3200U);

// 编译时 tile 配置验证
[[nodiscard]] constexpr bool tile_config_valid() const noexcept {
    return thread_m_ >= 8U && thread_m_ <= 64U;
}
static_assert(tile_config(1U).valid());
static_assert(!tile_config(33U).valid());

// 编译时内存布局验证
static_assert(sizeof(AudioBuffer) == sizeof(std::vector<float>) + sizeof(int));
```

## 4. 模块分解

### 4.1 `core/` — 核心类型与配置

```
include/voxmutatio/core/
├── types.h           # AudioBuffer, VCConfig, ErrorCode 等基础类型
├── device.h          # CUDA 设备探测和管理
├── allocator.h       # GPU 内存分配器 (pinned + device)
└── log.h             # 日志设施
```

**职责**：项目内所有模块共享的基础设施。不依赖任何具体模型。

### 4.2 `separation/` — 声源分离 (Source Separation)

```
include/voxmutatio/separation/
├── separator.h       # SourceSeparator 接口
└── mdx_config.h      # MDX-Net 配置

src/separation/
├── mdx_net.{h,cu}    # MDX-Net U-Net CUDA 实现
└── mdx_common.cuh    # 共享算子 (2D Conv, U-Net skip connections)
```

**目标**：从混合音频中剥离纯净干声（Dry Vocal）。

**实现方案**：
- **MDX-Net**（推荐）：2D 卷积 + U-Net 结构，手写 CUDA Kernel 难度低于 Transformer
- **HTDemucs**（备选）：ONNX → TensorRT 路径

### 4.3 `content/` — 内容编码器 (Content Encoder)

```
include/voxmutatio/content/
├── wavlm_encoder.h   # WavLM-Base+ 接口
└── hubert_encoder.h  # HuBERT 接口 (legacy 兼容)

src/content/
├── wavlm_encoder.{h,cu}    # WavLM CUDA 实现
├── hubert_encoder.{h,cu}   # HuBERT CUDA 实现
└── content_common.cuh      # 共享 Transformer 算子
```

**目标**：剥离说话人音色，提取纯粹的"语言学/发音内容"。

**实现方案**：
- **WavLM-Base+**（主要）：对带噪音/混响的音频鲁棒性更强
- **HuBERT-Base**（兼容）：RVC v1/v2 经典选择

**关键实现点**：
- WavLM/HuBERT 本质上是标准的 **Transformer Encoder**
- 复用 Qwen3x-Orin 项目中已优化的 Attention、GEMM、LayerNorm CUDA Kernels
- 只需调整维度、激活函数 (GELU) 和位置编码

### 4.4 `f0/` — F0 基频提取 (Pitch Extraction)

```
include/voxmutatio/f0/
├── f0_extractor.h    # F0Extractor 统一接口
├── rmvpe.h           # RMVPE 具体实现
└── fcpe.h            # FCPE 具体实现

src/f0/
├── rmvpe.{h,cu}      # RMVPE Conformer encoder CUDA 实现
├── fcpe.{h,cu}       # FCPE 轻量级 CUDA 实现
└── f0_common.cuh     # 共享算子 (Mel spectrogram, 1D Conv)
```

**目标**：精准捕捉源音频的音高曲线（F0）。

**双轨制实现**：
- **RMVPE**（离线/高质量）：Bi-LSTM + CNN，抗干扰能力最强
- **FCPE**（实时/低延迟）：1D 卷积 + 池化，延迟 < 2ms

### 4.5 `index/` — CUDA 特征检索 (Feature Retrieval)

```
include/voxmutatio/index/
└── cuda_flat_index.h # CUDA Flat Index 接口

src/index/
└── cuda_flat_index.{h,cu}  # 手写 CUDA Flat Index 实现
```

**目标**：通过向量相似度搜索，强制生成器使用目标音色。

**关键实现点**：
- **手写 CUDA Flat Index**（~200 行代码，零 FAISS 依赖）
- 利用 Shared Memory 和 Warp Shuffle 指令
- 计算 L2 Distance 或 Inner Product
- 对于数万条特征的库，GPU 暴力 Top-K 检索 < 1ms

### 4.6 `synthesizer/` — VITS 合成器

```
include/voxmutatio/synthesizer/
├── synthesizer.h       # Synthesizer 接口 (VITS v1/v2)
├── text_encoder.h      # TextEncoder (Transformer)
├── flow.h              # ResidualCouplingBlock
└── generator_nsf.h     # GeneratorNSF (HiFiGAN + NSF)

src/synthesizer/
├── synthesizer.{h,cpp}       # 主编排逻辑
├── text_encoder.{h,cu}       # Transformer encoder CUDA
├── flow.{h,cu}               # Residual Coupling Flow CUDA
├── generator_nsf.{h,cu}      # HiFiGAN upsampler + NSF 注入 CUDA
├── source_hn_nsf.{h,cu}      # SineGen + SourceModuleHnNSF CUDA
└── synth_common.cuh          # 共享算子 (Conv1d, ConvTranspose1d, GLU 等)
```

**关键实现点**：
- **TextEncoder**：Transformer encoder，输入 [1, T, 256/768] + F0 → 输出 [1, inter_channels, T]
- **ResidualCouplingBlock**：5 个残差耦合层，每个含 ConvNeXt Block
- **GeneratorNSF**：HiFiGAN 变体，含多阶段上采样 + NSF 谐波注入
- **SourceModuleHnNSF**：从 F0 生成谐波正弦波 + 噪声激励

### 4.7 `training/` — 微调训练基础设施

```
include/voxmutatio/training/
├── trainer.h           # 训练器接口
├── optimizer.h         # AdamW 优化器
├── loss.h              # 损失函数 (Mel/Feature/Generator/Discriminator)
└── data_loader.h       # 数据加载器

src/training/
├── trainer.{h,cpp}     # 训练循环实现
├── optimizer.{h,cu}    # AdamW CUDA 实现
├── loss.{h,cu}         # 损失函数 CUDA 实现
└── data_loader.{h,cpp} # 数据加载和预处理
```

**目标**：支持基于预训练权重的 Fine-tuning（不包含从零预训练）。

### 4.8 `pipeline/` — 端到端管线

```
include/voxmutatio/pipeline/
└── pipeline.h          # VoiceConversionPipeline 接口

src/pipeline/
└── pipeline.{h,cpp}    # 管线编排实现
```

**关键实现点**：
- 编排 Separation → Content → F0 → Index → Synthesizer 的完整链路
- 音频分块处理（长音频切分为 ~5s 块）
- RMS 能量混合（源音频和目标音频的音量均衡）
- 重采样到目标采样率

### 4.9 `io/` — 音频 I/O 与权重加载

```
include/voxmutatio/io/
├── audio_io.h          # 音频文件读写
└── safetensors.h       # Safetensors 权重加载

src/io/
├── audio_io.{h,cpp}    # WAV/FLAC 读写 + 重采样
└── safetensors.{h,cpp} # Safetensors 解析（零拷贝 mmap）
```

### 4.10 `webui/` — 嵌入式 WebUI 服务

```
include/voxmutatio/webui/
└── server.h            # HTTP 服务器接口

src/webui/
├── server.{h,cpp}      # httplib 服务器实现
└── routes.{h,cpp}      # API 路由处理

webui/
├── index.html          # 主页面
├── app.js              # 前端逻辑
└── styles.css          # 样式表
```

**设计原则**：
- 简洁现代，新手友好
- 不照搬 RVC-WebUI 的复杂界面
- 清晰的引导、合理的默认值、实时反馈

## 5. 权重格式转换策略

RVC 的原始权重是 PyTorch `.pth` 格式（pickle 序列化）。需要一个一次性转换工具：

```
tools/convert_to_safetensors.py
  └── 输入：RVC .pth 模型文件
  └── 输出：safetensors 格式（按模块分片）
```

该工具是 Python 脚本（仅运行一次），不属于 C++ 运行时。转换后的 safetensors 文件可被 C++ 零拷贝 mmap 加载。

## 6. 开发阶段

| 阶段 | 内容 | 交付物 |
|------|------|--------|
| **P0** | 项目骨架 + 权重加载 + 音频 I/O | 可编译、可加载模型 |
| **P1** | WavLM/HuBERT encoder CUDA 实现 | 特征提取数值对齐 |
| **P2** | RMVPE/FCPE F0 提取 CUDA 实现 | F0 序列数值对齐 |
| **P3** | CUDA Flat Index 实现 | 特征检索数值对齐 |
| **P4** | VITS 合成器 CUDA 实现 | 合成音频数值对齐 |
| **P5** | 完整管线 + CLI 工具 | 端到端转换可用 |
| **P6** | WebUI + 训练模块 | 完整产品可用 |
| **P7** | 实时流式推理支持 | 低延迟流式 API |

## 7. 数值对齐验证

每个模块完成后，与 Python 原始实现进行逐样本对齐：
- **WavLM/HuBERT**：相同输入 PCM → 比较输出特征向量 (L2 < 1e-4)
- **F0**：相同输入 PCM → 比较 F0 序列 (绝对误差 < 0.5 Hz)
- **Index**：相同特征 → 比较检索结果 (Top-K 一致)
- **Synthesizer**：相同特征 + F0 → 比较输出 PCM (SRCC > 0.999)

验证工具：`tools/vc_probe.cpp`，支持加载 Python 导出的中间张量作为参考值。
