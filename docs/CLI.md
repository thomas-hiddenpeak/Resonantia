# Resonantia CLI contract

本文补充根 README 的常用工作流。命令自身的 `--help` 是当前参数名称和默认值的直接
实现证据；如果本文与可执行文件不一致，应按 SDD drift 修正，而不是猜测隐藏行为。

## 构建目标

| 命令 | 用途 | 成熟度 |
|---|---|---|
| `vc_convert` | 单个 WAV 离线转换 | 已实现 |
| `vc_batch` | 目录内 WAV 批量转换 | 已实现 |
| `vc_preprocess` | 切片、VAD、分离和清理 | 实验性；模型资产门控 |
| `vc_train` | decoder/GAN 微调 | 实验性 |
| `build_index` | 从 WAV/HuBERT 特征构建 Flat Index | 已实现的独立工具 |
| `vc_serve` | 静态 WebUI 与 CLI 编排 HTTP 服务 | 已实现；默认开放、非实时 |
| `vc_probe` | 开发探针 | 骨架，不是稳定产品入口 |

仓库没有安装包布局；以下命令默认从配置的 build directory 运行。

## `vc_convert`

始终必需：

```text
--model <safetensors-file>
--input <wav>
--output <wav>
```

另外必须按所选执行路径提供对应资产：

- `--content-encoder hubert`（默认）要求 `--hubert`；选择 `wavlm` 要求 `--wavlm`。
- `--f0-method rmvpe`（默认）要求 `--rmvpe`；选择 `fcpe` 要求 `--fcpe`。
- `--index-rate > 0` 要求 `--index`。

这些条件在参数解析阶段检查；缺失时不会进入 CUDA、文件扫描或模型初始化。

当前模型契约：

- `--version v1|v2` 选择必须与 checkpoint 匹配的 256D/768D 架构；它不是自动探测。
- `--sr 40000` 声明模型采样率；当前 synthesizer 只支持 40 kHz。
- `--speakers` 是历史兼容 hint；实际 speaker 数从 checkpoint 的 `emb_g.weight` 派生。
- `--speaker` 选择实际 speaker id，并在推理前按派生数量检查。
- 所选内容编码器、F0 方法与 Index 的参数依赖见上；资产存在性、格式和维度随后由
  pipeline 初始化检查，任何失败都不会静默降级。
- `--filter-radius` 只接受 `0` 或 3..31 的奇数。
- `--pitch` 接受 -48..48；`--index-rate`、`--rms-mix`、`--protect` 接受有限的 0..1。

CPU、FP16、PM、non-F0、formant shift、非 WAV 和独立输出重采样都会失败。

## Batch、训练与预处理

`vc_batch` 复用 `vc_convert` 的条件模型契约；它始终要求 `--model`、`--input-dir` 和
`--output-dir`，并支持 `--recursive`。单个文件失败会进入批次失败计数；调用方仍应
检查最终退出码。

实验训练的推荐编排是：

```text
vc_preprocess -> vc_train -> build_index
```

便捷脚本是 `scripts/train_voice.sh`。这条路径没有 checkpoint resume 或生产音质保证。
分离模型名称必须存在于 registry 且与请求 task 匹配；未知、缺失默认或错任务模型会在
加载权重前失败。

## 错误和输出

- 没有帮助请求时，`vc_convert`、`vc_batch`、`vc_train` 与 `vc_preprocess` 对未知
  option、缺值、尾随字符、NaN/Inf、整数或派生尺寸溢出返回非零。
- 为保持历史兼容，`vc_convert` 与 `vc_batch` 会预扫完整 argv；任意位置出现 `--help`
  或 `-h` 都会短路其余解析并成功显示帮助，即使同一次调用还包含未知 option。
  其他命令的帮助与解析顺序以各自 `--help` 和退出码为准。
- WAV 输出先在目标同目录写 staging 文件，完成 `fsync`/close 后 rename；失败保留旧目标。
- CLI 输出文字主要用于人类诊断，不是稳定的机器协议；自动化以退出码和最终产物校验为准。
- Web 编排使用 argv 直接启动这些命令，不通过 shell；CLI 的非零退出必须传播为 HTTP 错误。

相关规范：[`Spec 001`](../specs/001-voice-conversion-engine/spec.md)、
[`Spec 002`](../specs/002-voice-training/spec.md)、
[`Spec 003`](../specs/003-webui/spec.md)、
[`Spec 004`](../specs/004-source-separation/spec.md)。
