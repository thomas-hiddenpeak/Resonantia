# Implementation Plan: Voice Conversion Engine

**Branch**: `001-voice-conversion-engine`
**Status**: Offline pipeline, bounded HuBERT callers, and full-suite verification complete; specification still partial because `vc_probe` has no product acceptance
**Updated**: 2026-08-12

## Current Delivery

当前交付是 Jetson/SM87 面向的 C++20/CUDA20 离线推理链。运行时模型为
safetensors，内部音频为 mono float32，模型计算为 CUDA FP32：

```text
WAV -> 16 kHz front-end -> HuBERT(default) -> optional Flat Index
    -> RMVPE(default) or FCPE -> F0 VITS v1/v2 -> 40 kHz model-rate WAV
```

## Implemented Phases

### P0 — Build, device and contracts

- [x] CMake C++20/CUDA20，默认 SM87，可自动发现 CUDA compiler。
- [x] CUDA device 初始化、RAII buffer 和错误传播。
- [x] 无本地模型资产的 contract 层。

### P1 — Core I/O and weights

- [x] WAV 读取/写入、mono 下混与线性重采样。
- [x] safetensors 解析与布局/字节数校验；模型消费者按各自 manifest 校验必需权重的
  dtype、shape、数值与模型不变量。
- [x] 明确拒绝非 WAV 音频格式。

### P2 — Content and F0

- [x] HuBERT 默认路径：v1 256D、v2 768D。
- [x] WavLM 可选实现与资产门控对齐；要求匹配训练的 synthesizer。
- [x] RMVPE 默认实现；算子临时 device allocation 使用 RAII buffer。
- [x] FCPE 可选实现与资产门控对齐。
- [x] PM 保留枚举但在 pipeline 明确拒绝。

### P3 — Index and synthesizer

- [x] Flat Index 加载为私有、已 unlink 的 filesystem-backed snapshot，并保持 mmap exact
  weighted k-NN 热路径；优先 `FICLONE`，copy fallback 需要 `O(Index bytes)` 临时容量。
- [x] F0 VITS v1/v2：TextEncoder、Flow、NSF/HiFi-GAN Generator。
- [x] 版本、特征维度、speaker、pitch、F0 指针和尺寸边界校验。
- [x] 完整已用 tensor 的 F32/shape/nbytes/alignment/finite 与 weight-norm 不变量校验；VITS
  checkpoint 以 256 MiB hard cap 复制到 read-only anonymous snapshot，并保持绝对模型路径/
  源身份 fail-closed；其他 safetensors 消费者仍使用 file-backed mapping。
- [x] VITS 算子临时 device allocation 使用 RAII buffer。
- [x] non-F0 明确拒绝。

### P4 — Pipeline and CLI

- [x] HuBERT/RMVPE/Index/VITS 编排。
- [x] protect、RMS mix、pitch、F0 filter。
- [x] 安静点分块、上下文 padding、拼接和输出长度边界。
- [x] `vc_convert` 与当前支持的 40 kHz model-rate WAV 输出；失败返回非零状态。
- [x] HuBERT `vc_convert`/`vc_batch` 经共享 policy 使用 `convert_file_bounded`；WavLM 在
  matching WavLM-trained VITS 资格资产出现前保持 legacy `convert_file`。
- [x] WAV 目标文件同目录 staged write + `fsync`/close + atomic rename；失败保留旧目标并
  清理 staging。
- [x] CPU、FP16、formant shift、非 WAV、独立 target sample rate 的失败契约。

### P5 — Verification

- [x] 当前 CMake 资产齐全配置注册 39 个 distinct tests（16 contract-labelled、24
  integration-labelled；WavLM preflight 同时属于两类）；fresh Release/SM87 当前工作树
  已 39/39 通过（0 失败，1297.94 秒）。
- [x] 加固前 30/30（11 contract + 19 integration、356.41 秒）保留为历史基线，不替代
  当前变更证据。
- [x] HuBERT、RMVPE、FCPE、WavLM、VITS、端到端和长音频均由资产门控的
  integration 测试覆盖。
- [x] `vc_convert`/`vc_batch` 非法参数语法与退出码的真实进程 CTest。
- [x] 严格数值解析，拒绝 malformed、非有限、溢出、空白、十六进制浮点、尾随字符和
  越界值。
- [x] 真实模型 `test_cli_bounded_callers` 已注册，锁定 `vc_convert` 与单文件 `vc_batch` 的
  HuBERT bounded 路由和成功 40 kHz WAV。
- [ ] `vc_probe` 的产品验收；当前仍是骨架。

注册和 focused 通过本身不等于 39/39；当前树已另行完成全量，但 `vc_probe` 等规格
缺口仍使 001 保持 specification verification partially complete。

## Current Constraints

- 只支持 CUDA FP32 和 WAV I/O。
- HuBERT 是默认内容编码器；WavLM 不是 stock RVC 模型的透明替代。
- 当前只支持 40 kHz synthesizer；输出保持该模型率，不提供 32/48 kHz 架构或独立重采样。
- HuBERT CLI/batch 的 bounded file path 具有结构性 window/core hard bounds，临时磁盘仍为
  `O(duration)`；legacy `convert_buffer`/`convert_file` 与 WavLM caller 仍保留整段 owning
  audio/F0 行为，不承诺 duration-independent working set。
- `infer_stream` 等局部 helper 不等于完整实时产品链。
- 模型/fixture 不入 Git；CMake 必须在注册 integration 前检查资产。

## Superseded Historical Decisions

以下内容曾出现在早期计划中，但已被当前实现或规格拆分取代，不得再当作现状：

| 历史决策 | 当前结论 |
|---|---|
| WavLM 作为主路径、HuBERT 作为 legacy | **已取代**：HuBERT 是默认；WavLM 仅可选且需匹配模型 |
| FLAC 输入属于基线 | **已取代**：当前音频 API 仅支持 WAV |
| CPU fallback、FP16 或 PM 可自动降级 | **已取代**：这些配置明确失败 |
| `target_sample_rate` 可独立控制输出 | **已取代**：只能为 0 或等于当前支持的 40 kHz 模型率 |
| 固定秒数、实时倍数或显存数字可作完成证据 | **已取代**：没有带完整环境元数据的可复用基准 |
| 在 001 内交付训练与 WebUI | **已取代**：训练、Web 和分离分别由 002、003、004 管理 |

## Separate realtime scope

端到端实时变声已拆分到 Spec 006 和 ADR-0002：决策已批准、R0 证据部分完成，但产品
runtime/session/ALSA 入口仍未实现，且 qualified target model+matching Index/corpus 仍阻塞。
不得从离线分块或 standalone Index probe 推导“已实时”；当前非实时收口不实施该产品路径。

## Approved P6 — Bounded offline file conversion

**ADR-0004 accepted 2026-08-12; P6-R1–R5 are implemented.** 本阶段只处理 Spec 001
seekable WAV 离线转换，不复用或修改 Spec 006 realtime runtime/session。R5 只切换
qualified HuBERT callers；WavLM 保持 legacy。

- [x] **P6-R0 governance/RED**：记录 current data flow、约 18 秒 current test 的单点 RSS
  诊断，写 Proposed ADR 与本迁移章节；加入要求 additive bounded file API 的 RED。该勾选只
  表示提案/红灯已建立，不表示产品实现。
- [x] **P6-R1 private I/O seam**：在 io 组件加入带 fd identity/size 复验的 bounded seekable
  WAV reader、mode-0600 checked raw-sample RAII spool，以及保持 mode-0666+umask 的
  incremental same-directory atomic WAV writer；覆盖 truncate/overflow、同输入输出、header
  backpatch、rename 前 file fsync、runtime commit point、best-effort directory fsync、失败保留
  和临时清理。保持现有 `read_audio`/`write_audio` 行为，测试 target 不链接 CUDA。
- [x] **P6-R2 bounded planner**：先把 decode/downmix/resample 固定为稳定 16 kHz spool；以
  全局 sample index 和 checked arithmetic 一次规划/执行一个 quiet core。递归 high-pass 使用
  double 全局 forward/reverse spool；synth raw float 与 RMS envelope 使用顺序 spool，禁止全段
  audio、energy、bounds 或 output vector。强制 effective core `Ceff+2Qeff`、synth
  `Ceff+2Qeff+2Peff+160` 上限，其中 Q/P 至少 160、P 再 clamp 到 `N-1`。
- [x] **P6-R3 bounded F0**：raw F0 core 冻结为 `128000` 个 16 kHz samples（8 秒），RMVPE/
  FCPE 单侧 context 分别冻结为 `64000`（4 秒）/`32000`（2 秒）samples；按全局 160 lattice
  crop raw F0/UV 并写 spool 后，再做有界全局 interp/median/pitch pass。最终实现须分别重跑：
  both-voiced mean absolute F0 error `<0.5 Hz`、p95 `<0.5 Hz`、全帧 UV agreement `>0.95`、
  内部 window 边界 both-voiced max `<0.5 Hz`；global max 仅诊断。冻结值来自 2026-08-12
  本地真实语音/真实模型资产探索，并已在最终实现重跑。F0 硬上界独立为 `128000+2F`，
  不是 content segment bound 再加 context。
- [x] **P6-R4 additive pipeline API**：实现 `convert_file_bounded`，成功返回 timing/status 但
  不返回 owning audio；保持 `convert_buffer` 和旧 `convert_file` 签名、完整结果与 reference。
- [x] **P6-R5 caller migration（门禁后切换）**：真实输出与旧 long-buffer reference 保持
  40 kHz、长度范围、finite 及 phase-invariant spectrogram correlation `>0.98`；17.565/35.130
  秒隔离进程的 RSS 为 `1342624`/`1354844 KiB`、device growth 为
  `892334080`/`851812352` bytes。HuBERT `vc_convert`/`vc_batch` 已切换，Web 继续经 CLI；
  WavLM 保留 legacy。两点资源证据须与代码 hard bounds 一起解释，且没有独立 click 阈值。
- [x] **P6-R6 compatibility/SDD**：本次已同步 DESIGN、TRACEABILITY、PROJECT_STATE、API/CLI
  文档并通过 32-file SDD/link gate；旧 `convert_file` 未 deprecated，是否增加 legacy hard cap 另行
  批准。当前 39 项 asset-free/model integration 已在 fresh Release/SM87 合入门禁中全部通过。

实施面包括公共 additive API、private bounded I/O/pipeline seam、共享 caller policy、DSP/
resource/model/CLI 门禁。公共配置、旧 API 和 WAV 格式未改变；以后改变 legacy hard cap、返回
所有权或支持矩阵仍须回到 ADR/Spec 评审。
