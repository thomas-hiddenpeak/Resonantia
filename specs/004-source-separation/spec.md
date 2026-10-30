# Feature Specification: Source Separation and Vocal Cleanup

**Feature**: `004-source-separation`
**Updated**: 2026-10-31
**Status**: **Model runners complete / Experimental / Workflow acceptance partial / Asset gated**

## 1. Scope

本模块为训练素材和手动转换提供可选的音频前处理：

- vocal separation：歌曲混音 → vocal stem；
- de-harmony：削弱伴唱/和声，保留 lead vocal；
- de-reverb、de-echo、de-noise；
- Silero VAD smart slicing；
- 长音频 chunked overlap-add。

它不是 `vc_convert` 默认主链的一部分，只有 CLI/Web 显式选择时才运行。runner 已实现
不等于权重已随仓库发布，也不等于所有输入上已经达到生产质量。

## 2. As-built Architecture

```text
WAV
  -> CUDA STFT / spectrogram front-end
  -> selected task runner
       Open-Unmix (UMX) vocal separation
       OR config-driven MelBand-RoFormer task model
  -> mask / estimated source
  -> CUDA iSTFT
  -> chunked overlap-add for long audio
  -> WAV

optional slicing:
  waveform -> Silero VAD -> speech segments
```

实现位于 `src/separation/` 与 `include/voxmutatio/separation/`：

- CUDA STFT/iSTFT（cuFFT、Hann window、center padding、overlap-add）。
- Open-Unmix `umxhq` vocal runner。
- config-driven MelBand-RoFormer（band split、RoPE/gated attention、mask head）。
- task/model registry 与 drop-in `<name>.json` metadata。
- Silero VAD v5 native C++/CUDA runner。
- long-audio chunking 与 overlap-add。

## 3. Model and Asset Contract

模型权重、converted fixture 和 Python reference dump 不进 Git。运行时模型使用 FP32
safetensors；一次性权重转换/参考生成脚本可放在 `tools/`，运行时不得依赖 Python。

典型本地资产包括：

- `models/separation/umxhq_vocals.safetensors`；
- 各任务的 MelBand-RoFormer safetensors 和 registry JSON；
- `models/vad/silero_vad.safetensors`；
- `tests/fixtures/separation/` 与 `tests/fixtures/vad/` 下的参考数据。

CMake 仅在每个测试声明的全部资产存在时注册该 integration test。缺资产时“不注册”
是环境限制，不是功能通过或失败；发布者必须另行说明权重来源、许可证和获取方式。

## 4. User Scenarios

### Story 1 — CLI vocal cleanup（P1，已实现）

`vc_preprocess` 接受 `--separate`、`--deharmony`、`--dereverb`、
`--deecho`、`--denoise`、`--vad`，按请求处理 WAV 并输出 WAV。

### Story 2 — Per-task model selection（P1，已实现）

每个 separation stage 可通过对应 `--*-model` 选择已安装模型。registry 校验任务与
模型匹配，未知、缺失默认或错任务模型在加载权重前失败；真实 CLI 进程门禁覆盖这些
拒绝路径。

### Story 3 — Web staged preprocessing（P2，已实现并有默认门禁）

Spec 003 的 `/api/sep-models` 提供模型发现，`/api/step` 对 material work set
应用单个 stage。结果先写入唯一 staging 目录，再通过 backup/rename 发布；提交失败恢复
或保留旧 work backup，并由真实 HTTP 故障场景门禁验证 live work 不丢失。

### Story 4 — Production cleanup quality（P1，未验证）

在代表性音乐、说话、混响、伴唱与噪声数据集上评估 stem leakage、伪影、语音可懂度
与下游训练收益。当前 fixture 对齐不构成这一产品质量证明。

## 5. Functional Requirements

- **FR-001 STFT/iSTFT**：参数和输入尺寸前置校验，重建误差受测试门控。
- **FR-002 UMX runner**：加载声明的 FP32 safetensors，执行 vocal mask/source 推理。
- **FR-003 RoFormer runner**：配置驱动的 band split、transformer、mask 与 iSTFT。
- **FR-004 Task coverage**：separate、de-harmony、de-reverb、de-echo、de-noise 可由
  相应权重驱动。
- **FR-005 Long audio**：按模型配置 chunking，并使用 overlap-add 归一化拼接。
- **FR-006 VAD**：Silero VAD 可选，供 smart slicing 使用。
- **FR-007 Registry**：枚举已安装模型并验证 model/task 关系；配置与 safetensors
  必须匹配。`vc_preprocess` 对未注册、缺失默认或错任务名称立即失败。
- **FR-008 CLI/Web**：`vc_preprocess` 和 `vc_serve` 只暴露已安装且通过边界校验的
  stage/model；Web step 只有在 staged 结果完整且可提交时才替换 live work set，任何
  提交失败都必须保留或恢复旧 work set。
- **FR-009 Numerical safety**：拒绝无效 FFT、band、attention、chunk 和 tensor
  尺寸；拒绝非有限中间/输出。`vc_preprocess` 数值 flags 采用严格解析，并在 seconds×sr
  转换前校验派生样本数范围；任意恶意模型 config 的资源上界仍属部分门禁。
- **FR-010 Runtime purity**：C++/CUDA 运行时不加载 PyTorch/pickle，不调用 Python。

## 6. Verification Gates

启用 `VOXMUTATIO_ENABLE_MODEL_TESTS=ON` 后，当前 CMake 按资产分别注册：

- `test_separation_stft`；
- `test_separation_spec`；
- `test_separation_umx`；
- `test_separation_roformer`；
- `test_separation_vad`。

2026-08-12 加固前所有本地资产齐全的完整项目验证为 30/30 CTest（11 contract +
19 integration、0 fail，`Total Test time (real) = 356.41 sec`），通过项包含上述五个测试以及
默认 CLI/HTTP/legacy Web contract。当前 CMake 已扩展为 39 个 distinct tests（16
contract-labelled、24 integration-labelled、WavLM preflight label overlap）；fresh Release/SM87
当前工作树已 39/39 通过（0 失败，1297.94 秒）。旧 30/30 只是加固前历史。
这些结果仅是对应资产集和环境的实现证据，不应外推为
任意权重/任意录音的质量保证。
默认 `test_cli_contract` 覆盖 `vc_preprocess` 数值、缺值、registry/default/task mismatch，
`test_web_contract` 覆盖真实 `/api/step` busy、失败回滚和成功清理。`/api/sep-models` 的
完整多模型响应与五任务真实权重执行仍不是默认门禁，因此完整工作流验收仍是部分状态。
UMX、RoFormer 与 VAD 的资产门禁还构造错误 shape 或非有限 checkpoint，并从公开入口注入
NaN/Inf 音频，验证 loader 在上传前拒绝权重且运行入口不返回貌似有效的输出。

模型 integration 必须至少验证：

- 关键中间 tensor 与离线参考的布局、形状、有限性和误差门；
- STFT/iSTFT reconstruction；
- UMX 和 RoFormer 的端到端输出；
- 长音频 chunked 与 whole-path 的一致性门；
- VAD probability/segment 边界；
- 缺失、错误或任务不匹配的模型配置稳定失败。
- `vc_preprocess` 数值 flags 严格拒绝 malformed、非有限值、溢出和越界范围。

## 7. Explicit Limitations

- 权重与 fixtures 不随源码仓库提供，测试和运行都可能因本地资产缺失而不可用。
- 模块仍按实验性能力管理；未建立跨数据集的生产音质、鲁棒性和吞吐保证。
- 不承诺实时处理；已有 chunking/scratch reuse 只说明实现方式。
- 不在 Spec 002 中自动证明“清理后训练效果更好”。
- 不支持把任意 PyTorch checkpoint 直接交给 C++ 运行时。

## 8. Historical Decisions

- **已取代**：“MDX conv U-Net 是 first target / current runner”。实际首先交付并对齐的
  runner 是 Open-Unmix，随后实现 MelBand-RoFormer。
- **已取代**：把孤立的加速倍数写为模块完成证据。性能结论必须附硬件、模型、输入、
  warm-up、编译配置和测量方法。
- **已取代**：把 runner 完成称为生产 ready。当前正确状态是“实现完成、实验性、
  模型资产门控”。

## 9. Completion and Promotion Criteria

代码实现项已完成；从“实验性”升级前仍需：

- [ ] 发布可审计的模型清单、来源、哈希、许可证和下载流程。
- [ ] 建立覆盖不同内容/语言/音乐类型的真实数据质量基准。
- [ ] 证明 cleanup 对下游训练的收益，并记录失败样例。
- [ ] 建立带环境元数据的时延、吞吐和显存基准。
- [ ] 明确服务部署的资源配额、并发和故障恢复策略。
- [x] 将 Web step 的 live work replacement 改为可回滚提交，并增加真实 HTTP 故障测试。
- [x] 让 `vc_preprocess --*-model` 对未知、缺失默认或 task 不匹配的模型失败，并覆盖 CLI 门禁。
- [x] 严格解析并约束 `sr/seg/hop/RMS/trim/VAD` 等预处理数值参数。

## 2026-10-29 立体声 Web 清理契约

新增 `vc_preprocess --preserve-stereo --no-slice --sr 44100`：在 RoFormer 各阶段间保留
左右声道，mono 输入复制成双声道；禁止 trim/VAD/其他采样率，以保持歌曲时间轴。
输出为双声道 float32 WAV，内部处理最长 600 秒、一般输入文件不超过 128 MiB。
2026-10-31修正阶段间重读上限：仅44.1k双声道float32允许≤256 MiB，容纳10分钟
中间产物（211,680,044 bytes）；其他格式保留128 MiB，超界或非有限样本失败。
格式、帧数和文件上限均在载入payload前检查；不改变已接受输入的样本处理。
部署需同时替换`vc_preprocess`与Web；旧CLI块缓存仍校验可执行文件摘要，不绕过版本门禁。
旧 CLI 默认单声道路径不变。Web 清理显式使用此路径；模型推理前才混合成单声道。
该路径源于 CyberTom《需要人陪》真实试听：去和声明显缓解尾段异常音高，但不保证
清除所有错音，也不自动替代人工局部音高校正。

### 2026-10-29 长任务进度迁移（实施中）

依据[ADR-0010](../../docs/adr/0010-durable-web-task-lifecycle.md)，Roformer分离入口增加可选
同步领域ProgressObserver，按实际完成的overlap-add块计数，短输入为一块。空观察器不改变
分块、边界窗与归一化；异常遵守原入口异常契约。vc_preprocess显式`--progress`输出任务阶段/
文件及块计数。单纯进度不等于恢复。

Roformer两个分离入口另增可选、借用的`io::BlockCheckpoint*`；调用者绑定输入、模型/配置、
实现身份。完成窗口保留完整模型输出；继续时按原顺序重放窗口、首尾边界权重及overlap-add，
重建累加器而不重算已完成的GPU窗口。完整块保存后才能抛出`io::OperationPaused`；普通错误
和损坏块沿runtime_error契约，不返回半段音频。短输入也是一块。默认空指针不写checkpoint。

`vc_preprocess --checkpoint-dir <dir> [--resume] [--pause-file <path>]`将输入文件序列、
各启用清理模型的权重/json/i64、VAD权重（如启用）、数值配方及二进制摘要绑定到同一CLI会话；
每个输入/清理阶段独立保存Roformer窗口。模型加载前校验身份，退出0完成、2安全暂停、1错误。
恢复重建清理波形、切片/输出文件，不能把低成本重建说成已保存了整个阶段输出；VAD/切片本身
没有细粒度checkpoint。最多4096个输入文件；每个块store的32MiB/4096块/2GiB限制及失败语义
同Spec001。Web整任务磁盘预算与阶段产物完成标记另行验收，不能以子store界限替代。
