# Implementation Plan: Voice Training

**Branch**: `002-voice-training`
**Updated**: 2026-10-25
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
- [x] sqrt/log 裁剪梯度、AdamW 非有限 beta/LR、MelLoss 配置/shape 与奇数 padding 频谱边界门禁。

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
- [x] GAN 推理导出同时保存训练后的 decoder + Flow；零步、训练后往返和失败保留回归。

### A4 — CLI and data baseline: implemented, limited

- [x] `vc_train` 读取 WAV 文件/目录。
- [x] HuBERT + RMVPE + v2/40k 前端，随机固定长度片段。
- [x] decoder 与 `--gan` 两种实验模式。
- [x] 显式`--gan-profile joint`和Web `mode=joint`：单clip重算、严格输入边界、最多50步、取消与导出。
- [x] step/loss 文本进度和最终 G 导出。
- [x] 独立 `build_index` CLI；输入 WAV 按确定路径顺序处理，最终 Index 以同目录唯一 staging、
  `fsync`/close + rename 原子发布，失败保留旧目标。真实 HuBERT/语音 contract 已在
  2026-08-12 历史 fresh Release/SM87 39/39 全量中通过；本轮记录见 [PROJECT_STATE](../PROJECT_STATE.md)。
- [ ] 完整数据 catalog/cache/版本化与可重入预处理事务。
- [ ] 稳定的机器可读事件协议。
- [x] 严格解析 CLI 数值参数，并由真实进程门禁拒绝 malformed、非有限、溢出、空白、
  十六进制浮点、尾随字符和越界值。

### A5 — Checkpoint and resume: joint-full implemented, fault coverage partial

- [x] joint-full保存G、D和两个AdamW的moments/计数。
- [x] 保存epoch、step、LR scheduler、RNG与数据遍历/裁剪位置。
- [x] 原子checkpoint manifest、身份绑定与版本不兼容拒绝。
- [x] 同一checkpoint的新进程next-step等价、协作取消/恢复及失败保留门禁。
- [ ] 强杀/掉电及写入中崩溃的实际故障验证；当前不承诺未保存更新的持久化。

当前完整66项回归及严格模块阈值见PROJECT_STATE；正式200轮训练进行中。

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
| Training mode | v2/40k fine-tune；decoder-only、旧GAN、有界joint及轮次joint-full |
| Export | inference G safetensors；旧GAN为decoder+Flow，joint另含enc_p/选定embedding；不含续训状态 |
| Resume | joint-full完整checkpoint支持；旧profile不支持 |
| `FineTuneTrainer` | unsupported legacy facade |
| FP16 / CPU / distributed / multi-GPU | unsupported |
| From-scratch pretraining | out of scope |
| Production voice similarity | unverified |

## Superseded Plan Claims

- **已取代**：“FR-001..012 全部完成”。joint-full恢复已有证据，通用产品数据管线和
  `FineTuneTrainer`仍没有完成。
- **已取代**：“SC-005 声线学习有效已通过”。现有训练 loss/推理往返不能证明生产
  声线相似度。
- **已取代**：“端到端验证完成即可称训练完成”。当前证据只支持实验性组件与小规模
  真实音频 step。
- **已取代**：“checkpoint 指最终导出 G”。最终推理导出不包含 optimizer 与恢复状态，
  不能称为断点续训。

## Next Milestone

2026-10-26维护者已授权推进[CyberTom完整训练计划](full-training-20261026.md)：
补齐A5完整恢复及固定前端缓存、轮次/LR与实际监测后完成200轮，集中听审后交付。
以下“不启动长训练”为前一轮资格化的历史边界；新实施受[ADR-0009](../../docs/adr/0009-resumable-joint-training.md)约束。

原计划优先A5 resume。2026-10-25按CyberTom听审反馈，先完成短样本质量诊断；
[ADR-0008](../../docs/adr/0008-joint-voice-training-alignment.md)提出联合微调及训练对齐，
维护者已于2026-10-25批准，库内严格单步及600帧/32帧crop资源复验已通过，
完整回归结果见PROJECT_STATE。CLI/Web显式接线已完成，通过独立Web实例完成47步，
加前置1步与真实取消2步共50步；具体CyberTom短样本已完成ASR和听审，保守动态版已接入Web，
当时不启动长训练、A5尚未实现；当前已由ADR-0009增量推进，A6仍未验证，不能因执行完成标记音质通过。

## Authorized orchestration increment: complete-iteration cancellation

- [x] Optional job channel and distinct completed/error/cancelled CLI statuses.
- [x] Poll between initialization/clips and complete decoder iterations or GAN D/G pairs; gate final export.
- [x] Web owns token lifetime, natural reap and staging cleanup before releasing the job.
- [x] Host contracts for parameters, protocol, fake complete steps, actual pre-CUDA cancellation and Web result consumption.
- [ ] Independent review and real GPU cancellation/resource/latency acceptance; no real training run in this increment.

该历史取消增量未实现A5；当前joint-full的A5证据见上文，A6仍未验证。
旧训练profile的取消不保存恢复状态。See [DESIGN](../../docs/DESIGN.md).
