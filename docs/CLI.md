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
| `build_index` | 从 WAV/HuBERT 特征构建 Flat Index | 已实现；确定排序、原子发布 |
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

### 文件转换路径与资源边界

`vc_convert`/`vc_batch` 通过同一条 caller policy 选择 file API：

- HuBERT v1/v2（RMVPE 或 FCPE）使用 `convert_file_bounded`。成功交付是原子提交的 40 kHz
  PCM16 WAV；内部 `VCResult::audio.data` 为空，CLI 不依赖 owning output vector。
- WavLM 在仓库取得匹配的 WavLM-trained VITS 并完成 bounded 数值门禁前，继续使用旧
  `convert_file`。它不会静默换成 HuBERT，但 legacy 路径也不承诺与时长无关的 working set。

bounded 路径使用稳定 16 kHz spool、全局 high-pass/RMS pass、quiet cores 和独立 F0 windows；
process working set 由配置、模型、Index 和固定 halo 约束，临时磁盘仍为 `O(duration)`。
17.565/35.130 秒真实语音 resource gate 是两个时长点，不是任意时长实测保证；当前也没有
独立冻结的 click/transient 指标。精确 `Ceff/Qeff/Peff` 与 `128000+2F` hard bounds 见
[`DESIGN.md`](DESIGN.md#pipeline)。这些是同步离线语义，不是 realtime/session。

## Batch、训练与预处理

`vc_batch` 复用 `vc_convert` 的条件模型契约；它始终要求 `--model`、`--input-dir` 和
`--output-dir`，并支持 `--recursive`。单个文件失败会进入批次失败计数；调用方仍应
检查最终退出码。程序会输出所选的 `bounded HuBERT` 或 `legacy WavLM` file path；该文本
用于诊断，自动化仍应以退出码和最终 WAV 校验为准。

批量输出保持 `output-dir/<stem>.wav` 的平铺布局。若不同输入映射到同一输出，例如
递归目录的 `a/voice.wav` 和 `b/voice.wav`，整个批次在模型加载及任何输出写入前失败，
并列出冲突路径。请先重命名冲突输入或分批指定输出目录；不会自动选择一个文件覆盖。

实验训练的推荐编排是：

```text
vc_preprocess -> vc_train -> build_index
```

便捷脚本是 `scripts/train_voice.sh`。这条路径没有 checkpoint resume 或生产音质保证。
decoder 模式导出训练后的 decoder；`--gan` 模式同时导出训练后的 decoder 与 Flow，
合并为一个原子发布的推理 safetensors。它不保存 posterior/D/optimizer 恢复状态。
分离模型名称必须存在于 registry 且与请求 task 匹配；未知、缺失默认或错任务模型会在
加载权重前失败。

`build_index` 只接受 HuBERT、输入目录和输出路径。它先按确定的 WAV 路径顺序提取 v2/768D
features，再在目标同目录创建唯一 staging；完整 header/payload 写入、file `fsync` 和 checked
close 后 rename，父目录 fsync best-effort。rename 前失败保留旧目标并清理 staging。
runtime 加载 Index 时会建立 private、已 unlink 的 filesystem-backed snapshot：优先
`FICLONE`，不支持时复制并需要约一份 Index 的临时 filesystem 容量；空间/权限失败使
Index-required pipeline fail-closed。reflink fallback 当前没有被故障注入门禁强制触发。

## 错误和输出

- 没有帮助请求时，`vc_convert`、`vc_batch`、`vc_train` 与 `vc_preprocess` 对未知
  option、缺值、尾随字符、NaN/Inf、整数或派生尺寸溢出返回非零。
- 为保持历史兼容，`vc_convert` 与 `vc_batch` 会预扫完整 argv；任意位置出现 `--help`
  或 `-h` 都会短路其余解析并成功显示帮助，即使同一次调用还包含未知 option。
  其他命令的帮助与解析顺序以各自 `--help` 和退出码为准。
- WAV 输出先在目标同目录写 staging 文件，完成 `fsync`/close 后 rename；失败保留旧目标。
- VITS checkpoint 单文件上限为 256 MiB，并复制到 read-only anonymous snapshot；这是
  VITS 专用 host-residency 取舍，其他 safetensors consumer 仍可使用 file-backed mapping。
- CLI 输出文字主要用于人类诊断，不是稳定的机器协议；自动化以退出码和最终产物校验为准。
- Web 编排使用 argv 直接启动这些命令，不通过 shell；CLI 的非零退出必须传播为 HTTP 错误。

相关规范：[`Spec 001`](../specs/001-voice-conversion-engine/spec.md)、
[`Spec 002`](../specs/002-voice-training/spec.md)、
[`Spec 003`](../specs/003-webui/spec.md)、
[`Spec 004`](../specs/004-source-separation/spec.md)。
