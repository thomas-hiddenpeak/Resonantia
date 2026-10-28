# Feature Specification: Voice Training

**Feature Branch**: `002-voice-training`
**Created**: 2026-08-07
**Updated**: 2026-10-26（完整训练/恢复门禁通过，正式200轮任务启动；具体CyberTom短样本已交付，通用生产质量未验证）
**Status**: **Experimental / Partially Complete**
**Architecture**: C++/CUDA autograd, 40 kHz v2 fine-tuning path

## 1. Product Boundary

项目已经有可执行的训练实验路径，但尚未达到“用户提交数据即可稳定训练出生产声线”
的产品状态。

当前可用入口是：

- `vc_train` decoder-only mel fine-tune。
- `vc_train --gan` posterior/flow/decoder + discriminator 的 GAN fine-tune。
- `vc_train --gan --gan-profile joint`显式有界联合微调（包含可训练prior与speaker embedding）。
- `vc_train --gan --gan-profile joint-full`显式轮次训练、冻结前端缓存与完整checkpoint恢复。
- `GeneratorTrainer`、`GANTrainer`、`JointTrainer`及其底层 autograd/AdamW 组件。
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

旧完整 GAN 导出必须同时写入训练后的 decoder 与 Flow；训练时折叠进偏置的说话人条件
在导出时扣除，weight norm 按推理布局重建。两个模块合并后只做一次原子发布。
显式joint使用训练图中的speaker条件，另导出更新后的enc_p及选定embedding；布局保持兼容。
posterior encoder、判别器、optimizer/RNG 不属于推理导出契约；该文件仍不能用于等价续训。

## 3. User Scenarios

### Story 1 — 开发者运行 decoder 微调实验（P1，已实现）

开发者向 `vc_train` 提供 HuBERT、RMVPE、预训练 v2/40k G、WAV 文件或目录、训练
步数和输出路径。程序执行真实优化并原子写出 inference-ready safetensors。

### Story 2 — 开发者运行完整 GAN step（P1，已实现但实验性）

在额外提供 D checkpoint 后，`--gan` 路径执行 enc_q、flow、decoder、判别器和
mel/KL/feature-matching/adversarial 损失。训练 loss 必须有限，错误初始化必须清空
旧 checkpoint 状态。

### Story 3 — 断点续训（P1，已实现并有真实资产门禁）

保存/恢复 G、D、optimizer、epoch/step 和 RNG 状态，并验证恢复后的下一步等价。
2026-10-26库内及显式joint-full CLI/Web已实现完整G/D、optimizer、RNG、epoch、
数据身份与裁剪/排列状态的保存/恢复；同一checkpoint的连续/新进程下一步、真实取消/恢复
及数据变化拒绝门禁已通过，精确命令与数值统计边界见PROJECT_STATE。
不承诺bitwise重放；异常退出回到最近成功保存的恢复点，不承诺每次更新的崩溃持久化。
其他训练profile仍只有推理G导出，不能用于等价恢复。

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
| FR-009 | inference G safetensors 导出 | decoder-only 导出decoder；旧GAN为decoder+Flow，joint另含enc_p/选定embedding；临时文件+rename一次原子提交 |
| FR-010 | 独立 Flat Index 构建 | `build_index` 按确定路径顺序构建，并以同目录 staging + fsync/close + rename 原子发布；仍未与 G 导出形成同一训练事务 |
| FR-011 | 持久化 checkpoint 与断点续训 | **已实现·真实资产门禁通过**：joint-full完整状态、同一状态下一步及取消/恢复；长期运行验证进行中，统计边界见PROJECT_STATE |
| FR-012 | 旧 `FineTuneTrainer` 门面 | **不支持，必须失败** |
| FR-013 | 分布式、FP16、多 GPU、从零预训练 | **不支持** |
| FR-014 | 生产级音质/声线相似度保证 | **未验证** |
| FR-015 | 可选作业令牌、完整迭代协作取消和独立退出码 | CLI/Web编排及host验证；joint真实2步取消退出2、旧文件保留，尚无通用时延保证 |

## 5. Safety and Correctness Gates

生成器输出激活的负半轴系数在训练和推理中均使用参考值`0.01`，前面的残差/上采样层
保持`0.1`。独立真实语音参考除了相关性还检查相对L2`<1e-3`，避免训练和推理共享同一
错误却通过往返测试。兼容性与旧权重重新验收边界见
[Spec 001 §2.2](../001-voice-conversion-engine/spec.md#22-生成器输出激活对齐修正2026-10-24)。

2026-10-24 性能修复移除 GAN 前处理重复 prior/inverse-Flow 与未使用的 decoder-only
trainer，复用固定 host segments 和有上限的 dead-graph device buffers。梯度范数采用
分块归约，必须保持 NaN/Inf 拒绝、裁剪门限和新图零梯度语义；`test_autograd` 包含
超过 `2^24` 个单位梯度的精度回归与图生命周期复用测试。CLI 新增 feature/GAN timing
文本，仍不构成 FR-008 的稳定机器协议。验证记录见 [PROJECT_STATE](../PROJECT_STATE.md)。

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

### RMVPE前端修正与训练兼容（2026-10-24）

训练使用的RMVPE须遵循[Spec 001 §2.3](../001-voice-conversion-engine/spec.md#23-rmvpe-前端与独立参考修正2026-10-24)的HTK前端契约。
`vc_train`每次从WAV重新提取F0，没有跨次持久F0缓存；新试验须使用修正后的CLI与
原始预训练G，不能从旧Slaney标签微调模型继续训练并宣称已修复。旧模型/素材保留，
Index仅含HuBERT特征，本次不涉及Index格式或重建要求。人工听审仍为独立验收条件。

### 采样合成修正与训练兼容（2026-10-24）

训练前处理使用[Spec 001 §2.4](../001-voice-conversion-engine/spec.md#24-采样合成恢复与兼容2026-10-24)
定义的sampled默认模式，`--synthesis-mode mean`可恢复原对照，现有`--seed`驱动采样。
decoder-only缓存的prior/逆Flow latent及旧两种训练模式缓存的NSF source均采用新配置；
encoder prior stats、GAN posterior及优化器算法不因此改变。仍按clip前处理一次并缓存，
旧路径不宣称实现上游每步刷新噪声的数据流程；显式joint见下述迁移。
相同采样输入下的可训练decoder须匹配独立
上游参考（相关性>0.999、相对L2<1e-3），旧mean参考继续验证。
模型布局不变；旧微调权重保留且需重新听审，推理与训练CLI整组备份部署。

## Approved migration: bounded joint fine-tuning

[ADR-0008](../../docs/adr/0008-joint-voice-training-alignment.md)提出在现有路径之外增加
显式选择的联合微调：可训练enc_p和选定speaker embedding，完整有界clip先编码再裁
decoder片段，对齐参考训练损失与随机输入，原子导出全部参与推理的更新参数。
维护者于2026-10-25批准按ADR实施，严格先完成单步与内存验证，再最多50步短训练。
库内`JointTrainer`及RVC Mel路径通过严格单步验证，CLI/Web已显式接入新profile；
独立Web实例完成限定短训练；具体短样本听审通过后，以`CyberTom_20261026`接入内网Web。
原CyberTom保留，通用生产质量尚未验收。

迁移分库内独立对齐、0/1步与资源验证、CLI/Web接线、最多50步小验证四阶段。
保留旧API/default/profile和原始权重，实验输出到新目录；失败回到旧路径，不覆盖正式
CyberTom。现有safetensors布局保持不变，仍不实现断点续训。完整规则、回滚条件和
门禁见ADR；逐步验证结果见PROJECT_STATE，不能将批准或单步执行计入完整FR或音质通过。

2026-10-25后续实现已修正weight norm、Softmax、卷积/说话人条件的FP32累加顺序及
NSF host的FMA合并，加入独立算子、真实六层逐值比较和workspace越界回归。
该修改未改变原始参考文件、梯度阈值、优化器epsilon或训练配方；当前库内单步严格
比较与6秒输入资源复验已通过。CLI/Web接线及最多50步资格化的执行证据见PROJECT_STATE，
未获人工音质接受前不得发布CyberTom。最新完整验证见PROJECT_STATE。

`vc_train --gan --gan-profile joint`显式选择联合路径；缺省`legacy`保留原行为。
joint默认50步、32帧decoder crop、学习率`1e-4`、decay=1、epochs=0、sampled、seed=0；
当前仅接受1..50步、32帧crop、epochs=0、sampled及decay=1，显式学习率须有限且在(0,1]。
输入须为预切片40k WAV，每段0.32..6秒；所有文件元信息在CUDA初始化前校验，超长拒绝，
不静默截断。每步仅物化当前片段，按HuBERT/RMVPE/波形共有有效帧对齐后计算频谱，
不缓存全数据集的音频、频谱或prior。每步刷新posterior/NSF随机输入。
Web的`mode=joint`使用同一路径及取消/暂存发布事务，限制一致；
`GET /api/voices`的`training_profiles`声明可用选项，旧服务不声明时UI隐藏joint。

## 6. Verified Evidence

### 2026-10-26授权的完整训练迁移（实施中）

维护者已明确同意补齐恢复/缓存/长训练能力，并要求按200轮计划推进；此前50步限制仅为
旧joint资格化profile契约，不再限制本次已授权的新增长训练能力。
新架构与迁移/回滚见[ADR-0009](../../docs/adr/0009-resumable-joint-training.md)，
阶段、素材与验收见[完整训练计划](full-training-20261026.md)。新增`joint-full`显式入口已接线：
缓存只含冻结前端/目标特征，8MiB/clip上限；1..4096 clips、每轮逐片访问、batch1、
默认200轮及LR1e-4/decay0.999875、严格crop32。完整恢复绑定输入及模型/配方身份，
取消时保留完成D/G更新后的状态，失败不得更新最终输出。旧joint/legacy不接受resume。
Web固定素材与状态持久保留，完整候选须听审后发布；真实验证状态见PROJECT_STATE，
不能将接线或训练结束当作正式音质验收。

2026-10-24 CyberTom使用212个真实普通话片段（630.02355秒）经Web的GAN 2800步
预设完整训练并导出，但中文留出音频及《需要人陪》短片段均被人工听审否决为明显电音，
英文多个短样本也未达到ASR辅助可懂度门限。训练执行完成不构成FR-014验收通过。
实验条件、失败结果、F0修正的独立验证和证据路径见
[PROJECT_STATE](../PROJECT_STATE.md#2026-10-24-cybertom-重新训练与听审)。

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

上述历史证据说明训练代码会执行真实优化且关键数值/资源边界受测试，不证明：

- 长时间或大数据集训练稳定；
- checkpoint resume（joint-full的新增恢复证据见2026-10-26增量和PROJECT_STATE）；
- 导出模型在未知数据上的泛化；
- 声线 embedding 相似度提升或生产音质。

## 7. Completion Criteria

要把状态从“实验性 / 部分完成”升级为产品能力，至少还需：

- [x] joint-full实现并测试G/D/optimizer/epoch/step/RNG的checkpoint resume；真实强杀/掉电故障不在已验证范围。
- [ ] 定义并完成可复现的真实数据集训练稳定性门。
- [ ] 完成导出 G + Index 的一致性/事务边界。
- [ ] 建立客观指标与盲听协议，证明目标声线相似度和音质。
- [ ] 定义失败恢复、磁盘配额、产物保留和版本兼容策略。
- [ ] 决定旧 `FineTuneTrainer` 的正式实现或移除计划。
- [x] 使用严格数值解析并以真实进程测试 malformed、非有限、溢出、尾随字符和越界参数。

## 9. Cooperative cancellation (FR-015)

后续用户授权增加`--cancel-file <path>`。0/1一字节令牌在初始化/clip边界、完整GAN D/G对或decoder
整步之间及导出前检查；退出0=完成、1=错误、2=取消且未开始导出。已开始导出完成，错误优先。
无参数不访问令牌且保留旧训练参数、采样、数值更新顺序。旧profile取消丢弃进度；
joint-full扩展在完整更新边界保存FR-011恢复状态，原profile语义不变。
协议、Web收尾、host证据与未验证GPU边界见[DESIGN](../../docs/DESIGN.md#完整迭代协作取消2026-10-10-后续授权tr-fr-015--web-fr-011)。

普通异常退出边界：`tools/vc_train.cpp`捕获options/parse/run期间的标准/未知C++异常，
普通栈展开后返回1；异常诊断失败也保持1，不覆盖正常0/取消2。`test_vc_train_entry`
以生产main+host替身验证此契约，不宣称覆盖noexcept析构异常、崩溃或致命GPU故障的清理。
