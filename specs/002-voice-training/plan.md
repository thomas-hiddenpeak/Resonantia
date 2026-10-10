# Implementation Plan: Voice Training

**Branch**: `002-voice-training`
**Updated**: 2026-08-12
**Status**: Experimental / partially complete

## Delivery Strategy

复用已对齐的推理前端和 synthesizer 权重，在 C++/CUDA autograd 上先交付可验证的
decoder fine-tune，再扩展到 posterior/flow/discriminator 的完整 GAN step。当前代码
已达到“开发者可运行实验”，尚未达到“可恢复、可规模化、质量有保证”的训练产品。

## As-built Phases

### A0 — Autograd core: implemented

- [x] RAII device tensor storage and tape-based backward graph。
- [x] add/mul/sum/matmul 及训练所需的 reshape/slice/concat/embedding 等基元。
- [x] linear、conv、transpose-conv、normalization、activation、softmax 等反向。
- [x] 有限差分/contract 测试覆盖关键算子与非法尺寸。

### A1 — Optimizer and differentiable mel: implemented

- [x] AdamW、bias correction、梯度裁剪和 non-finite 防护。
- [x] differentiable frame/STFT/magnitude/mel/log/L1。
- [x] real-audio mel integration test。

### A2 — Trainable generator path: implemented, experimental

- [x] `GeneratorTrainer` 加载 v2/40k G 的 decoder 参数。
- [x] decoder forward/backward 和 mel-L1 更新。
- [x] 初始化失败回滚，不保留旧 checkpoint。
- [x] 原子导出 inference-ready G，测试导出后推理往返。

### A3 — Full GAN components: implemented, experimental

- [x] PosteriorEncoder `enc_q`。
- [x] Flow 训练图。
- [x] 一个 scale discriminator + 八个 period discriminators。
- [x] mel、KL、feature-matching 与 LSGAN losses。
- [x] G/D 交替优化、梯度裁剪和 epoch LR decay。
- [x] 真实 WAV 的有限 loss 与完整 step 集成测试。

### A4 — CLI and data baseline: implemented, limited

- [x] `vc_train` 读取 WAV 文件/目录。
- [x] HuBERT + RMVPE + v2/40k 前端，随机固定长度片段。
- [x] decoder 与 `--gan` 两种实验模式。
- [x] step/loss 文本进度和最终 G 导出。
- [x] 独立 `build_index` CLI；输入 WAV 按确定路径顺序处理，最终 Index 以同目录唯一 staging、
  `fsync`/close + rename 原子发布，失败保留旧目标。真实 HuBERT/语音 contract 已在
  fresh Release/SM87 当前工作树 39/39 全量中通过。
- [ ] 完整数据 catalog/cache/版本化与可重入预处理事务。
- [ ] 稳定的机器可读事件协议。
- [x] 严格解析 CLI 数值参数，并由真实进程门禁拒绝 malformed、非有限、溢出、空白、
  十六进制浮点、尾随字符和越界值。

### A5 — Checkpoint and resume: not implemented

- [ ] 保存 G、D 和 optimizer moments。
- [ ] 保存 epoch、step、LR scheduler、RNG 与数据遍历位置。
- [ ] 原子 checkpoint manifest 与版本兼容。
- [ ] kill/restart 后 next-step 等价测试。

`FineTuneTrainer::save_checkpoint/load_checkpoint` 当前明确返回 false，不能用它作为
A5 的占位完成证据。

### A6 — Quality and productization: not verified

- [ ] 多规模真实数据集的长时间稳定性和资源基准。
- [ ] held-out 音频的 intelligibility、pitch preservation、artifact 指标。
- [ ] speaker embedding 与人工盲听的声线相似度协议。
- [ ] 失败恢复、磁盘配额、并发隔离、产物保留和许可证说明。
- [ ] 确认 Web 训练编排只暴露已经具备的实验能力。

## Supported vs Unsupported

| Area | Current |
|---|---|
| Runtime | C++/CUDA FP32 |
| Training mode | v2/40k fine-tune; decoder-only or experimental GAN |
| Export | inference G safetensors |
| Resume | unsupported |
| `FineTuneTrainer` | unsupported legacy facade |
| FP16 / CPU / distributed / multi-GPU | unsupported |
| From-scratch pretraining | out of scope |
| Production voice similarity | unverified |

## Superseded Plan Claims

- **已取代**：“FR-001..012 全部完成”。当前 resume、产品数据管线和
  `FineTuneTrainer` 没有完成。
- **已取代**：“SC-005 声线学习有效已通过”。现有训练 loss/推理往返不能证明生产
  声线相似度。
- **已取代**：“端到端验证完成即可称训练完成”。当前证据只支持实验性组件与小规模
  真实音频 step。
- **已取代**：“checkpoint 指最终导出 G”。最终推理导出不包含 optimizer 与恢复状态，
  不能称为断点续训。

## Next Milestone

下一里程碑优先完成 A5 的可验证 resume，再建立 A6 的质量协议；在这两项完成前，
规格状态保持“实验性 / 部分完成”。

## Authorized orchestration increment: complete-iteration cancellation

- [x] Optional job channel and distinct completed/error/cancelled CLI statuses.
- [x] Poll between initialization/clips and complete decoder iterations or GAN D/G pairs; gate final export.
- [x] Web owns token lifetime, natural reap and staging cleanup before releasing the job.
- [x] Host contracts for parameters, protocol, fake complete steps, actual pre-CUDA cancellation and Web result consumption.
- [ ] Independent review and real GPU cancellation/resource/latency acceptance; no real training run in this increment.

Unchanged A5/A6 milestones and unsupported resume still apply. See [DESIGN](../../docs/DESIGN.md).
