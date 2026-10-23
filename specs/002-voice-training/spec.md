# Feature Specification: Voice Training

**Feature Branch**: `002-voice-training`
**Created**: 2026-08-07
**Updated**: 2026-10-24
**Status**: **Experimental / Partially Complete**
**Architecture**: C++/CUDA autograd, 40 kHz v2 fine-tuning path

## 1. Product Boundary

项目已经有可执行的训练实验路径，但尚未达到“用户提交数据即可稳定训练出生产声线”
的产品状态。

当前可用入口是：

- `vc_train` decoder-only mel fine-tune。
- `vc_train --gan` posterior/flow/decoder + discriminator 的 GAN fine-tune。
- `GeneratorTrainer`、`GANTrainer` 及其底层 autograd/AdamW 组件。
- 独立 `build_index` CLI。

旧公共 `FineTuneTrainer` 仅为源码兼容保留。其 `init/train/save_checkpoint/
load_checkpoint` 明确返回失败，**不支持训练**；不得把它计入完成能力。

## 2. Current Workflow

```text
WAV target clips
  -> HuBERT + RMVPE + v2/40k synthesizer front-end
  -> fixed-length training segments
  -> decoder mel-L1 fine-tune
     OR full GAN step:
        PosteriorEncoder + Flow + Decoder
        one scale discriminator + eight period discriminators
        mel + KL + feature-matching + LSGAN losses
  -> inference-ready G safetensors (atomic export)

optional, separate:
  extracted features -> build_index -> .index
```

当前导出只保证代码和测试覆盖的推理权重往返。它不等于已经证明新目标声线的主观质量、
相似度或大数据集稳定性。

完整 GAN 导出必须同时写入训练后的 decoder 与 Flow；训练时折叠进偏置的说话人条件
在导出时扣除，weight norm 按推理布局重建。两个模块合并后只做一次原子发布。
posterior encoder、判别器、optimizer/RNG 不属于推理导出契约；该文件仍不能用于等价续训。

## 3. User Scenarios

### Story 1 — 开发者运行 decoder 微调实验（P1，已实现）

开发者向 `vc_train` 提供 HuBERT、RMVPE、预训练 v2/40k G、WAV 文件或目录、训练
步数和输出路径。程序执行真实优化并原子写出 inference-ready safetensors。

### Story 2 — 开发者运行完整 GAN step（P1，已实现但实验性）

在额外提供 D checkpoint 后，`--gan` 路径执行 enc_q、flow、decoder、判别器和
mel/KL/feature-matching/adversarial 损失。训练 loss 必须有限，错误初始化必须清空
旧 checkpoint 状态。

### Story 3 — 断点续训（P1，未实现）

保存/恢复 G、D、optimizer、epoch/step 和 RNG 状态，并验证恢复后的下一步等价。
当前 `vc_train` 只有最终 G 导出，不具备这一契约。

### Story 4 — 生产声线质量（P1，未验证）

在代表性数据集上验证音色相似度、可懂度、音高保持、伪影和人工听感。现有单元/集成
测试不能替代这一验证。

## 4. Requirements and Status

| ID | Requirement | Status |
|---|---|---|
| FR-001 | CUDA autograd tape、反向图与基础/神经网络算子 | 已实现并有 contract/梯度测试；覆盖度持续维护 |
| FR-002 | AdamW、梯度裁剪、非有限梯度/损失防护 | 已实现 |
| FR-003 | differentiable STFT/mel L1 | 已实现 |
| FR-004 | NSF/HiFi-GAN decoder 的可训练前向 | 已实现 |
| FR-005 | PosteriorEncoder、Flow、判别器训练图 | 已实现 |
| FR-006 | KL、feature matching、LSGAN、mel 的 G/D 交替 step | 已实现，实验性 |
| FR-007 | WAV 数据发现、前端特征/F0、训练片段采样 | `vc_train` 内实现基线路径；非完整数据产品 |
| FR-008 | 结构化进度 | 仅 CLI 文本 step/loss；稳定机器可读协议未定义 |
| FR-009 | inference G safetensors 导出 | decoder-only 导出 decoder；GAN 同时导出 decoder + Flow；临时文件 + rename 一次原子提交 |
| FR-010 | 独立 Flat Index 构建 | `build_index` 按确定路径顺序构建，并以同目录 staging + fsync/close + rename 原子发布；仍未与 G 导出形成同一训练事务 |
| FR-011 | 持久化 checkpoint 与断点续训 | **未实现** |
| FR-012 | 旧 `FineTuneTrainer` 门面 | **不支持，必须失败** |
| FR-013 | 分布式、FP16、多 GPU、从零预训练 | **不支持** |
| FR-014 | 生产级音质/声线相似度保证 | **未验证** |
| FR-015 | 可选作业令牌、完整迭代协作取消和独立退出码 | CLI/Web编排已实现；host验证，真实GPU取消未验收 |

## 5. Safety and Correctness Gates

- 训练入口对 speaker、steps/epochs、segment、learning rate 和数据路径做范围/存在性
  校验；数值参数使用严格全量解析并拒绝非有限值、溢出、尾随字符和越界值，真实
  `vc_train` 进程由 `test_cli_contract` 门控。
- checkpoint tensor 必须验证 dtype、shape、byte range、有限值和 weight norm。
- `GeneratorTrainer`、`GANTrainer` 初始化采用失败回滚；失败后旧权重不可继续使用。
- loss 非有限时不得更新或导出伪成功模型。
- `sqrt(max(x,0))` 在 `x<=0`、`log(max(x,1e-12))` 在 `x<=1e-12` 的反向取零，
  包括裁剪端点；其余区域使用对应数学导数，梯度门禁须覆盖裁剪区间。
- AdamW 的 beta 必须有限且位于 `[0,1)`；学习率构造/更新都必须有限且大于零，
  无效更新抛出 `invalid_argument` 并保留旧学习率。
- MelLoss 要求 `n_fft>1`、正 hop/mels/sample_rate、有限的
  `0<=fmin<effective_fmax<=Nyquist`（`fmax=0` 表示 Nyquist），并在分配/索引前检查
  matrix/frame 尺寸。负长度、空目标、短于一帧的运算和不匹配的 loss shape 必须拒绝。
- host/GPU 线性频谱使用实际 pad=`floor((n_fft-hop)/2)`，
  T=`1+(L+2*pad-n_fft)/hop`；奇数 padding 差也必须一致。无完整帧或不可表示的尺寸拒绝，
  不得以 `L/hop` 强行多读一帧。
- 最终 G 导出使用同目录临时文件、精确 little-endian 写入、flush/close 后 rename；
  失败清理临时产物。
- `build_index` 对输入 WAV 路径排序，最终 `.index` 使用唯一同目录 staging、`0666` + caller
  umask、完整写入与 `fsync`/close 后 rename；rename 前失败保留旧目标并清理 staging。
  `test_build_index_contract` 以真实 HuBERT/语音资产门控确定性字节、旧 inode 保留和失败清理。
- 功能集成使用真实 WAV；小型合成张量只用于算子梯度/边界测试。

## 6. Verified Evidence

2026-10-24 新增默认 `test_training_contract`（频谱奇数 padding、MelLoss 配置/shape），
补充 `test_autograd` 裁剪梯度及 AdamW 非有限边界，扩展 `test_gan` 完整训练后 Flow
变化、零步条件还原、导出后 inverse/decoder 一致性和失败保留旧文件。
本轮命令、失败基线及全量结果见 [PROJECT_STATE](../PROJECT_STATE.md)；以下为历史基线。

加固前本地资产齐全的 2026-08-12 30/30 CTest（11 contract + 19 integration、0 fail，
`Total Test time (real) = 356.41 sec`）中包含：

- autograd contract；
- real-audio mel loss；
- generator trainer 初始化回滚、训练、导出与推理往返；
- discriminator、posterior/flow 和完整 GAN step；
- GAN 初始化回滚与未初始化失败契约。

当时 CMake 资产齐全配置已扩展为 39 个 distinct tests（16 contract-labelled、24
integration-labelled，WavLM preflight label 重叠），并新增 `test_build_index_contract`；
该历史树的 fresh Release/SM87 全量已 39/39 通过（0 失败，1297.94 秒）。
旧 30/30 只保留为加固前历史，注册本身也不替代本次实际执行。

这些证据说明训练代码会执行真实优化且关键数值/资源边界受测试，不证明：

- 长时间或大数据集训练稳定；
- checkpoint resume；
- 导出模型在未知数据上的泛化；
- 声线 embedding 相似度提升或生产音质。

## 7. Completion Criteria

要把状态从“实验性 / 部分完成”升级为产品能力，至少还需：

- [ ] 实现并测试 G/D/optimizer/epoch/step/RNG 的 checkpoint resume。
- [ ] 定义并完成可复现的真实数据集训练稳定性门。
- [ ] 完成导出 G + Index 的一致性/事务边界。
- [ ] 建立客观指标与盲听协议，证明目标声线相似度和音质。
- [ ] 定义失败恢复、磁盘配额、产物保留和版本兼容策略。
- [ ] 决定旧 `FineTuneTrainer` 的正式实现或移除计划。
- [x] 使用严格数值解析并以真实进程测试 malformed、非有限、溢出、尾随字符和越界参数。

## 9. Cooperative cancellation (FR-015)

后续用户授权增加`--cancel-file <path>`。0/1一字节令牌在初始化/clip边界、完整GAN D/G对或decoder
整步之间及导出前检查；退出0=完成、1=错误、2=取消且未开始导出。已开始导出完成，错误优先。
无参数不访问令牌且保留旧训练参数、采样、数值更新顺序。取消丢弃进度，不属于FR-011 resume。
协议、Web收尾、host证据与未验证GPU边界见[DESIGN](../../docs/DESIGN.md#完整迭代协作取消2026-10-10-后续授权tr-fr-015--web-fr-011)。

普通异常退出边界：`tools/vc_train.cpp`捕获options/parse/run期间的标准/未知C++异常，
普通栈展开后返回1；异常诊断失败也保持1，不覆盖正常0/取消2。`test_vc_train_entry`
以生产main+host替身验证此契约，不宣称覆盖noexcept析构异常、崩溃或致命GPU故障的清理。
