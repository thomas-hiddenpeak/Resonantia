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
| `vc_serve` | 静态 WebUI 与 CLI 编排 HTTP 服务 | 已实现；默认开放、非实时；`--ffmpeg /absolute/path`默认`/usr/bin/ffmpeg`供Web多格式上传解码 |
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

`--rms-mix 0`完全跟随原音的1秒RMS包络，1保留模型音量、不做音量匹配；默认0.5。
2026-10-26修正居中补零与插值时间坐标，两条文件路径及buffer API同步；mix<1的
旧音频复现需要旧CLI，不改权重或Index。短窗增强动态未通过听审，当前没有相关选项。

2026-10-24起，转换默认不插值填满无声音高帧，避免把辅音和停顿变成持续有声激励。
这同时适用于`vc_batch`和Web调用的新`vc_convert`；`--filter-radius`的局部平滑规则保留。
C++库调用方可显式设`VCConfig::interp_unvoiced=true`复现旧模式；模型与Index无需重建。

合成默认使用`--synthesis-mode sampled`，恢复预训练模型的prior和NSF高斯采样；
`--synthesis-mode mean`显式复现两处噪声均关闭的旧模式。
`--synthesis-seed`接受0到18446744073709551615，默认0；相同输入、配置和seed可复现。
以上同样适用于`vc_batch`；Web转换继承sampled/0默认值，界面不另设采样控件。
旧模型仍可加载，但新模式下需要重新听审，不能据数值对齐宣称音质合格。

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

便捷脚本是 `scripts/train_voice.sh`。旧默认路径没有 checkpoint resume 或生产音质保证；
新增长训练显式选择`joint-full`，见下文。
decoder 模式导出训练后的 decoder；`--gan` 模式同时导出训练后的 decoder 与 Flow，
合并为一个原子发布的推理 safetensors。它不保存 posterior/D/optimizer 恢复状态。
旧decoder/GAN的`vc_train --synthesis-mode sampled|mean`控制前处理中的prior/NSF采样（默认sampled），
使用现有`--seed`（0到4294967295，默认0）作为合成及数据采样种子。decoder的latent和两种训练模式的
NSF source仍在每个clip前处理时缓存，不是每个优化step重新抽样。GAN的posterior采样
规则不由此选项改写。推理模型文件不记录这些运行参数，复现实验应另行保存CLI命令。
分离模型名称必须存在于 registry 且与请求 task 匹配；未知、缺失默认或错任务模型会在
加载权重前失败。

`vc_train --gan --gan-profile joint`选择有界联合微调，要求G、D及冻结前端模型。
默认profile为`legacy`；joint默认50步、`--seg 32 --epochs 0 --lr 0.0001 --lr-decay 1`、
sampled及seed0。当前joint只允许1..50步、seg32、epochs0、sampled、lr-decay1；
显式学习率须有限且在(0,1]，不支持resume。先用`vc_preprocess`准备40k、0.32..6秒WAV片段；
超长、过短或错采样率会在CUDA初始化前拒绝，不会自动裁短。每步重新提取当前clip的
前端特征并刷新posterior/NSF噪声，更新enc_p/enc_q/Flow/decoder及选定speaker embedding。
推理导出写入enc_p/Flow/decoder/选定embedding，保留其他speaker行和原enc_q权重。
这是显式实验选项；通过数值和短训练门禁不代表目标声线已获人工音质接受。

`vc_train --gan --gan-profile joint-full --epochs 200 --cache-dir <cache> --checkpoint-dir <state>`
显式启用完整联合训练；仍需上述HuBERT/RMVPE/G/D/target/out参数。默认200轮、LR=1e-4、
decay=0.999875、crop32、sampled、seed0，单GPU FP32 batch1；每轮每个clip恰好访问一次。
最多4096个40k、0.32..6秒WAV；共同有效特征须至少32帧。禁止`--steps`，轮次1..2000；
全部调度LR须保持有限正值。缓存按音频、前端配方和权重SHA-256命名，含phone/F0/pitch/
目标audio/spec与payload校验；单文件上限8MiB，只读取当前clip，训练prior及噪声不缓存。
数据、模型、配方或缓存内容不一致时拒绝恢复，不忽略损坏缓存。

新训练要求checkpoint目录为空（内部锁除外）。第0步、每5轮、最终轮和协作停止时保存
完整`state_step_XXXXXXXXXX.safetensors`，原子提交后保留最新两份；50/100/150/200轮另存
`candidate_epoch_XXXXXXXXXX.safetensors`。同目录锁避免并发写入，阶段耗时及epoch/step/LR
打印到日志。`--resume <state_file>`须保留相同原始模型、数据、轮次及其他配方参数；
恢复随机遍历/裁剪、优化器和训练随机状态。取消仍退出2且不替换最终`--out`，恢复点保留。
异常退出从最后成功保存点恢复，尚未保存的更新需要重算；不承诺每次更新都已落盘。
缓存准备期取消尚无训练更新，可直接用固定素材重新启动；推理model不能用于resume。
这些产物须独立听审，不代表质量已通过。当前真实整作业验证结果见PROJECT_STATE。

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

### 立体声清理与人工局部修音（2026-10-29）

`vc_preprocess --preserve-stereo --no-slice --sr 44100` 保留左右声道，支持已有
`--separate/--deharmony/--dereverb/...`。禁止同时trim/VAD，输入最长600秒/128MiB，
输出双声道float32 WAV；默认CLI单声道行为不变。

`vc_convert --pitch-edits 'smooth:17.36:17.48;shift:23.05:23.11:12'` 演示人工修音格式：
输入音频的秒数，平滑保留端点，偏移单位半音。最多16个有序不接触区间，每段0.02–2秒，
输入最长600秒；仅bounded文件转换支持。平滑跨无声区、越界/非有限数值均失败。
示例时间只适用于已听审的28秒尾段，不应直接套用其他音频。默认无修音。

Web使用说明、正式声线与验证命令见[Web发布记录](WEBUI_RELEASE_20261029.md)。
