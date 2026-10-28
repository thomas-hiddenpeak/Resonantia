# Resonantia — VoxMutatio

> *Vox mutatur, mens manet.*（声音变化，意识不变。）

Resonantia 是面向 Jetson Orin（SM87）的 RVC 风格 C++20/CUDA 声音转换项目。
当前成熟入口是源码树内的 CLI；运行时不依赖 Python/PyTorch，Python 只用于
`tools/` 中的模型下载、转换和参考数据生成。

## 能力与成熟度

| 能力 | 当前状态 | 关键边界 |
|---|---|---|
| 离线声音转换 | **已实现** | HuBERT CLI/batch 使用有界 file path；WavLM 与 legacy library APIs 仍为 owning/unbounded；CUDA FP32、WAV、40 kHz |
| Flat Index 检索 | **已实现** | private unlinked filesystem snapshot 上 host mmap exact retrieval；非零 `index_rate` fail-closed |
| 训练 | **实验性** | decoder/GAN及联合训练；`joint-full`新增完整状态/缓存/轮次恢复，验证状态见PROJECT_STATE；生产稳定性和音色仍需验收 |
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
`vc_convert`/`vc_batch` 对 HuBERT v1/v2（RMVPE/FCPE）使用 bounded spool/window/atomic writer；
WavLM 在匹配的 WavLM-trained VITS 完成资格化前保留 legacy file path，不会静默换 encoder。
bounded working set 不随总时长保存全段音频，但临时磁盘仍为 `O(duration)`；旧
`convert_buffer`/`convert_file` 继续返回完整 owning audio，不获得这一资源承诺。
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
转换默认恢复prior/NSF采样，固定seed 0可复现；CLI可用
`--synthesis-mode mean`复现旧的零噪声对照，或用`--synthesis-seed`指定抽样。
这项数值修正仍需各声线独立听审，不构成目标音色质量保证。
使用 Index 时，runtime 会建立 private、已 unlink 的 filesystem-backed snapshot；不支持
`FICLONE` 时需要约一份 Index 的临时 filesystem 容量，空间或权限不足会明确失败而不是
退化为 no-index。

## 测试

每次 `main` push、PR 和手动 workflow 都在 GitHub 托管的 `ubuntu-24.04` 上运行仓库检查
和以下独立 Linux host 子集；不需要 CUDA、模型或设备 runner。Catch2 使用与完整构建相同的
系统包优先/固定3.4.0归档及SHA-256策略：

```bash
cmake -S cmake/host-tests -B build-host-ci -DCMAKE_BUILD_TYPE=Release
cmake --build build-host-ci --parallel 2
ctest --test-dir build-host-ci --output-on-failure --parallel 1 --no-tests=error
```

该入口有11项host门禁：作业初始化、A/B/C所有权、结果消费、训练取消、素材提交中取消、
最终发布中拒绝取消、多格式上传与发布/翻唱错误契约、取消令牌/完整迭代协议、bounded音频I/O、旧Web API兼容和CLI异常入口。
它使用替身和临时文件，没有真实模型或训练，也不代表完整设备套件通过。
原`test_training_cancel`的纯协议断言已原样移至`test_training_cancel_protocol`；真实CLI预取消
与参数解析仍由原测试保留。它和`test_cli_contract`需要实际CUDA链接的CLI，留在完整构建路径。
具体差异和边界见[验收计划§9](specs/003-webui/acceptance-plan.md#9-托管host覆盖完整性与最终发布窗口)。

设备作业仅在 `main` 上手动勾选 `run_orinnx`，并且仓库变量 `ORINNX_RUNNER_READY`
为字符串 `true` 时才可调度。它固定要求全部标签
`[self-hosted, linux, ARM64, resonantia-orinnx-dedicated]`，没有用户提供标签或通用兜底。
专属标签是待配置约定，并不表示 runner 已注册或身份已核实。维护者必须先另行核实唯一匹配
runner 的 OrinNX 身份并授权，才能设置就绪标记；本次修改没有设置变量、注册 runner 或手动触发。
没有已授权的匹配 runner 就不执行设备测试，状态记为**未运行**。设备开始后另查 hostname 和
device-tree 型号作为第二层检查，不能代替调度前的身份/标签核实。

设备上获授权的无模型 contract 入口仍保留以下精确排除：

```bash
ctest --test-dir build --output-on-failure --label-regex contract \
  --exclude-regex '^test_web_transaction_failure$' --parallel 1 --no-tests=error
```

CI 的 host 子集不包含 `test_web_transaction_failure`，手动设备入口继续使用上述排除项；
CI 不启用模型资产测试或 GPU 训练。本地八项修复的完整验证范围与结果见
[`specs/PROJECT_STATE.md`](specs/PROJECT_STATE.md)，不把 CI 未执行项计为通过。
底层 CUDA 单元检查不等于真实模型、训练或 GPU 取消验收。根构建中 Node.js 18+ 可用时
另注册 UI 认证与取消测试；它只用于测试，缺失时两项明确不注册，不增加运行时依赖。

模型、对齐、训练和端到端测试只有在声明资产存在时才注册：

```bash
cmake -S . -B build-model-tests \
  -DCMAKE_BUILD_TYPE=Release \
  -DVOXMUTATIO_CUDA_ARCHITECTURES=87 \
  -DVOXMUTATIO_ENABLE_MODEL_TESTS=ON
cmake --build build-model-tests --parallel 2
ctest --test-dir build-model-tests --output-on-failure
```

需要包含真实 120 秒 Web body deadline 门禁时，再配置
`-DVOXMUTATIO_ENABLE_SLOW_WEB_TESTS=ON`；普通配置默认关闭此慢测试。

缺资产、未注册或 skip 都不等于通过。当前测试数量、环境和完整结果只维护在
[`specs/PROJECT_STATE.md`](specs/PROJECT_STATE.md)，避免 README 复制易过期数字。

## 实验性训练

```bash
scripts/train_voice.sh \
  --raw my_voice_wav/ \
  --work runs/alice \
  --steps 300
```

该脚本编排 `vc_preprocess -> vc_train -> build_index`，保留旧默认路径且不提供resume。
新`--gan-profile joint-full`按轮次运行，支持磁盘特征缓存和完整状态恢复；参数及边界见
[CLI说明](docs/CLI.md)，真实验证和当前限制见[项目状态](specs/PROJECT_STATE.md)。
训练产物仍需音质与声音相似度验收；不承诺大数据集稳定性、生产资源上限或可商用权利。保留原始录音与
预训练模型，不要把实验输出当作唯一可恢复副本。

## WebUI

唯一受支持的 Web 服务入口是 `vc_serve`：

上传支持 WAV、MP3、M4A（AAC/ALAC）、AAC、FLAC、OGG（Vorbis/Opus）、Opus、WMA、
AIFF/AIF，训练可多文件上传。新格式由服务器自动转为 PCM16/44.1 kHz WAV（转换上传保留立体声，训练默认mono），
结果仍下载为 WAV。WAV 原样保留，模型 CLI 的 WAV 接口不变。

2026-10-29已接入听审通过的正式 **Tom Cybersinger**（原CyberTom正式版），本机旧声线已清理：
完整歌曲按“分离人声→去和声→转换”处理，
可合回原伴奏；训练页提供200轮完整训练、断点恢复、候选试听与确认发布。另有默认关闭的
短区间人工音高校正。使用方法、质量边界和本次验证见[Web交付记录](docs/WEBUI_RELEASE_20261029.md)。

工作室界面按“制作翻唱 / 训练声线 / 声线库”分开日常任务：上传后确认输入类型与声线，
查看处理概览再生成，结果可与原声同位置对照；训练启动前核对计划，候选听审与发布在
声线库中完成。交互依据、无障碍/手机验收与自审记录见
[工作室设计与验收](docs/WEBUI_STUDIO_20261029.md)。

多格式需要带完整音频解码器的 FFmpeg，以及 coreutils `timeout` 和 util-linux `prlimit`。
默认使用 `/usr/bin/ffmpeg`，可用 `--ffmpeg /absolute/path` 指定。缺依赖时 WAV 仍可用，
其他格式返回503。本机 NVIDIA 的裁剪版 FFmpeg 缺少 MP3/AAC/WMA，已改用 Ubuntu
`7:6.1.1-3ubuntu5`；其他机器也应核验实际 codec，不能只检查可执行文件存在。
新格式解码上限为64 MiB/文件、128 MiB/批（约12.7/25.4分钟单声道音频）；长录音请分段
并选择合适的素材集合，每次上传替换当前素材。整次HTTP body上限128 MiB；解码超过120秒或损坏会明确报错并保留旧素材。
只有上传准备完成后才启用预处理、训练或转换。浏览器无法估算部分格式时长不影响上传。

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
