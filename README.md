# Resonantia — VoxMutatio

> *Vox mutatur, mens manet.*（声音变化，意识不变。）

Resonantia 是面向 Jetson Orin（SM87）的 RVC 风格 C++20/CUDA 声音转换项目。
当前成熟入口是源码树内的 CLI；运行时不依赖 Python/PyTorch，Python 只用于
`tools/` 中的模型下载、转换和参考数据生成。

## 能力与成熟度

| 能力 | 当前状态 | 关键边界 |
|---|---|---|
| 离线声音转换 | **已实现** | CUDA FP32、WAV、F0 RVC v1/v2、当前仅兼容 40 kHz synthesizer |
| Flat Index 检索 | **已实现** | host mmap 精确检索；非零 `index_rate` 缺失、错维或检索失败时 fail-closed |
| 训练 | **实验性** | `vc_train` 提供 decoder/GAN 工程路径；无 resume、生产稳定性或音色质量保证 |
| 声源分离与清理 | **实验性、资产门控** | UMX、MelBand-RoFormer、Silero VAD；权重与 reference 不随 Git 分发 |
| Web 编排 | **已实现、验收部分** | `vc_serve` 调用 CLI；默认监听 `0.0.0.0` 且默认无认证；非实时、无 TLS |
| 端到端实时变声 | **未实现** | Spec 006 已批准、R0 证据部分完成；尚无 persistent runtime、session、ALSA CLI |

离线主链是：

```text
WAV -> mono float32 -> 16 kHz -> HuBERT -> optional Flat Index
    -> RMVPE (or optional FCPE) -> F0 VITS v1/v2 40 kHz -> WAV
```

WavLM 只有在 synthesizer 以相同特征空间训练时才兼容。CPU、FP16、PM F0、
non-F0、formant shift、FLAC/MP3，以及独立目标采样率转换均不支持并应明确失败。
完整的当前状态与最近验证记录见
[`specs/PROJECT_STATE.md`](specs/PROJECT_STATE.md)。

## 最短可复现路径

### 1. 构建

要求 CMake 3.24+、C++20 编译器、CUDA Toolkit 和支持目标架构的 CUDA GPU：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DVOXMUTATIO_CUDA_ARCHITECTURES=87
cmake --build build --parallel 2
```

产品和 CI 基线是 Jetson Orin / SM87；在其他 CUDA GPU 上开发时必须显式设置
对应的 `VOXMUTATIO_CUDA_ARCHITECTURES`。

### 2. 准备资产

模型不进入 Git。按 [`models/README.md`](models/README.md) 的固定 revision、锁定工具环境
和哈希流程准备至少以下 safetensors：

- `models/hubert_base/model.safetensors`
- `models/rmvpe.safetensors`
- 一个兼容的 F0 v1/v2、40 kHz synthesizer

`models/pretrained_v2/pretrained_v2/f0G40k.safetensors` 是架构和数值验证用的通用预训练
generator，**不是目标声线模型**。实际目标声线转换应使用同一份有权使用的语料得到的
`model.safetensors` 与版本匹配的 Flat `model.index`（v1 为 256D、v2 为 768D；仓库训练
路径当前使用 v2/768D）；仓库当前没有可发布的目标声线资产对。

### 3. 转换 WAV

直接运行当前 CLI：

```bash
build/vc_convert \
  --hubert models/hubert_base/model.safetensors \
  --rmvpe models/rmvpe.safetensors \
  --model models/pretrained_v2/pretrained_v2/f0G40k.safetensors \
  --input source.wav --output converted.wav \
  --version v2 --sr 40000
```

使用训练产物和匹配 Index：

```bash
scripts/convert_voice.sh \
  --model runs/alice/model.safetensors \
  --index runs/alice/model.index \
  --input source.wav --output converted.wav \
  --index-rate 0.5 --pitch 0
```

以进程退出码和库的 `VCResult::success` 判断成功，不能只看旧输出文件是否存在。
CLI 参数、默认值和错误行为见 [`docs/CLI.md`](docs/CLI.md) 与各命令的 `--help`。

## 测试

默认 contract 不依赖被忽略的模型和数值 fixture：

```bash
ctest --test-dir build --output-on-failure --label-regex contract
```

模型、对齐、训练和端到端测试只有在声明资产存在时才注册：

```bash
cmake -S . -B build-model-tests \
  -DCMAKE_BUILD_TYPE=Release \
  -DVOXMUTATIO_CUDA_ARCHITECTURES=87 \
  -DVOXMUTATIO_ENABLE_MODEL_TESTS=ON
cmake --build build-model-tests --parallel 2
ctest --test-dir build-model-tests --output-on-failure
```

缺资产、未注册或 skip 都不等于通过。当前测试数量、环境和完整结果只维护在
[`specs/PROJECT_STATE.md`](specs/PROJECT_STATE.md)，避免 README 复制易过期数字。

## 实验性训练

```bash
scripts/train_voice.sh \
  --raw my_voice_wav/ \
  --work runs/alice \
  --steps 300
```

该脚本编排 `vc_preprocess -> vc_train -> build_index`。训练路径用于工程验证，不承诺
大数据集稳定性、断点续训、生产资源上限、声音相似度或可商用权利。保留原始录音与
预训练模型，不要把实验输出当作唯一可恢复副本。

## WebUI

唯一受支持的 Web 服务入口是 `vc_serve`：

```bash
build/vc_serve --repo "$PWD" --webroot webui \
  --host 0.0.0.0 --port 8080
```

默认配置会暴露给所有可达 IPv4 网络且 API 不认证。只在受信网络使用；面向不可信网络
至少启用 bearer token，并由反向代理提供 TLS、访问控制和审计：

```bash
build/vc_serve --repo "$PWD" --webroot webui \
  --host 0.0.0.0 --port 8080 \
  --auth-token 'replace-with-a-secret'
```

`--auth-token` 只保护 `/api/`，静态 WebUI 仍可读取。完整路由、资源上限和部署 non-goals
见 [`specs/003-webui/spec.md`](specs/003-webui/spec.md)。该服务是离线 CLI 编排层，
不是 WebSocket 或实时音频服务器。

## 作为库使用

当前仓库提供 `voxmutatio` CMake static facade 供源码树内目标和 `add_subdirectory` 集成；
尚无 `install()`、导出 package config 或稳定的预编译 SDK 发行契约。公共头位于
`include/voxmutatio/`，但嵌入方应准备随源码重编译，并以对应 Spec 和 contract tests
判断接口成熟度。

## 项目导航

- [`models/README.md`](models/README.md)：模型来源、转换、哈希与目标声线资产边界
- [`docs/CLI.md`](docs/CLI.md)：CLI 工作流、常用参数和失败语义
- [`specs/README.md`](specs/README.md)：SDD 权威层级、规格索引和维护规则
- [`specs/PROJECT_STATE.md`](specs/PROJECT_STATE.md)：当前能力与最近验证快照
- [`specs/TRACEABILITY.md`](specs/TRACEABILITY.md)：需求、实现和测试追踪
- [`docs/DESIGN.md`](docs/DESIGN.md)：当前架构与实现边界
- [`docs/REFACTORING_GUIDE.md`](docs/REFACTORING_GUIDE.md)：重构、迁移和回滚门禁
- [`AGENTS.md`](AGENTS.md)：Codex 的 SDD 桥接入口

修改 SDD 或桥接入口后运行：

```bash
bash scripts/check_sdd.sh
```

## 参考项目

- [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [Qwen3x-Orin](https://github.com/thomas-hiddenpeak/Qwen3x-Orin)
- [Orator](https://github.com/thomas-hiddenpeak/Orator)

## 许可证

MIT License
