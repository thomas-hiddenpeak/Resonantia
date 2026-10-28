# ADR-0008: 有界联合微调与独立训练对齐

**Status**: Accepted
**Date**: 2026-10-25
**Approval**: 维护者于2026-10-25明确同意按本方案修复；先完成单步和内存验证，
再做最多50步短训练，暂不跑长训练；生成音频送聊天听审。批准不等于实现或音质通过。
维护者在获知严格单步门禁失败后再次明确“保持原门槛，继续定位”，未授权例外短训练。
**Owners**: Resonantia maintainers
**Governing specs**: [Spec 002](../../specs/002-voice-training/spec.md)、
[Spec 003](../../specs/003-webui/spec.md)

## Context and decision drivers

用户要求保留声音厚度、自然度并学到Tom，同时明确暂不跑长训练。
最近的10.68秒参考转换被听审为抖动/颗粒感减轻，但声音变薄、发空且不像Tom。
该样本用60秒素材、50步decoder-only Pilot；239个decoder tensor改变，其余321个
包括enc_p/Flow/emb_g均与预训练相同。相同输入下Pilot输出与基础模型相关性0.77881，
表明训练确实影响了输出，不能解释为“模型未加载”或“训练完全无效”。

当前CLI的GAN路径也将enc_p prior预计算成常量；优化器只覆盖decoder、enc_q、Flow，
speaker条件折叠进可训练层偏置。冻结emb_g并不意味着完全不能学习目标声音，
但这与本地上游RVC的联合训练不同；训练上下文、mel配置及随机激励生命周期也不同。
现有同唱句前向对齐、213个内容权重身份核对通过，不能替代训练梯度和音质验收。
证据见[PROJECT_STATE](../../specs/PROJECT_STATE.md)。

本提案是可检验的训练改进路径，**不是已经证实的电音/音色损失唯一根因或音质保证**。
它改变训练核心数据流、可训练参数所有权与导出范围，需要维护者批准后实施。

## Decision

1. 在现有C++20/CUDA训练模块增加显式选择的联合微调路径；保留decoder-only和旧GAN
   API及默认值。CLI拟增加仅对`--gan`合法的`--gan-profile joint`；缺省保持既有行为，
   未实现前CLI仍拒绝该选项。通过门禁后Web才增加对应显式选项，不自动切换既有任务。
2. G optimizer覆盖enc_p、enc_q、Flow、decoder和选定speaker embedding；HuBERT/RMVPE
   保持冻结。说话人条件在训练图中显式计算，共享同一个可训练embedding，不再折叠成
   不带embedding梯度的常量。非选定speaker行不得因weight decay意外改变。
3. 一次step输入完整有界clip的phone `[T,768]`、coarse pitch `[T]`、F0 `[T]`、
   spectrum `[1025,T]`、40k waveform `[400*T]`、speaker和decoder crop范围。
   先在完整clip运行enc_p/enc_q/Flow，再裁decoder latent及对应F0、目标mel/audio。
   KL按完整有效帧计算；不能将56帧posterior/Flow局部上下文界套用于全局attention prior。
4. 首个实现batch=1，`T<=600`（6秒），超限在step前报错，不能静默截断。
   host只物化当前clip及模型必要数据，不缓存整个训练集的spec/prior；可以重算被淘汰的
   前端特征，优先保持可说明的资源上界。不得宣称这已经通过Orin内存资格化。
5. 训练配方固定到本地上游已记录版本：40k、FFT2048/hop400/125 mel、对应reflect
   padding、频谱power floor `2e-7`、log clamp `2e-6`；先对齐STFT/mel/梯度再训练。
   使用同一step的posterior和NSF随机输入完成D/G整对更新，每步更新随机输入；诊断
   模式允许显式注入同一随机数组以做数值对拍。新路径默认decoder crop32帧，显式crop
   必须与参考相同。AdamW epsilon、betas、损失系数、梯度处理和LR规则均纳入配方记录。
6. additive trainer接口保持现有init返回bool、非法step抛异常的习惯；host输入为同步
   借用，返回后可释放。模型/optimizer/device tensor由trainer拥有，单实例不可并发step。
   shape/长度/有限值/数值溢出、crop范围、模型兼容及OOM必须显式失败，init事务式回滚。
7. 推理G导出扩展为同一原子事务写入enc_p、Flow、decoder和更新的speaker embedding；
   保持现有560 tensor布局与safetensors格式，其他tensor按兼容契约保留。
   enc_q、D和optimizer仍不构成可恢复checkpoint；此提案不实现resume，不伪装等价续训。

## Alternatives considered

- 只增加现有50步Pilot的训练时长：不能验证上述训练差异，且不符合用户暂缓长训练的要求。
- 继续只调整Index/NSF噪声：未触及目标声线学习；可以保留为有明确假设的独立对照。
- 接入Python/PyTorch生产训练：违反当前runtime purity；仅允许用于tools内独立参考。
- 直接替换现有GAN默认：难以复现已有实验且放大未验证风险，采用显式选择与旧路径并存。

## Consequences

需要新增可训练prior、说话人条件梯度路径，并扩展导出和跨模块门禁；代码、资源和
验证成本显著高于改几个训练参数。完整clip图比40帧图更占内存，必须先测量再决定能否
进入短实验；不得通过静默降精度、裁短上下文或CPU降级隐藏失败。
联合训练仍可能无法改善音质；人耳未通过时不发布新CyberTom，也不扩大训练规模。

## Migration and rollback

1. 维护者批准提案后，先补失败基线和离线独立参考，再实现库内联合路径。
2. 完成0步加载/导出、确定性1步梯度/更新对齐、真实短clip资源门禁和旧路径回归。
3. 门禁通过后才增加CLI/Web显式选择及输入、取消、事务发布测试；旧服务和默认声线
   保持原样。新产物必须使用新的实验目录，保留旧模型、Index、原始素材与部署CLI。
4. 本轮资格化最多使用已有60秒素材、50个训练step，先做1步，再做小规模完整迭代。
   采用Tom自身干声的保真对照和留出人声转换，分开判断自然度/声音厚度与Tom相似度。
   ASR只辅助可懂度。**批准本提案不等于授权长训练、整曲转换或替换正式CyberTom。**
5. 任一梯度/导出/资源/错误或旧路径回归失败，停止新路径推进；切回旧profile/CLI即可。
   不删除失败证据，不用旧实验模型继续训练来掩盖新配方的效果。

## Verification and SDD impact

实施状态（2026-10-25）：原严格单步及6秒输入资源门禁已通过；CLI/Web显式joint已接入，
独立Web实例完成47步，连同前置单步与取消验证共50步。2026-10-26维护者接受两份短音频的
自然度和厚度，并确认Tom音色改善；不构成任意音频泛化保证。部署及精确命令见PROJECT_STATE；
以下原始验收阈值保持不变。

- 实际前向与独立参考：内容RMS及latent相对L2 `<1e-4`；音频相关性`>0.999`
  且相对L2 `<1e-3`。逐模块梯度相对L2 `<1e-3`，参考梯度L2 `<1e-8`时改用
  绝对RMS `<1e-6`；AdamW一步后参数/状态分别对拍，参数绝对RMS `<1e-6`，
  非零moments相对L2 `<1e-3`。先冻结独立参考再实施，失败必须定位，不能事后
  放宽门槛或用形状、“非零梯度”、仅最终参数接近代替梯度/状态对齐。
- 真实模型/语音测试检查enc_p、embedding、posterior、Flow、decoder参数归属和梯度，
  非选定speaker行不变；覆盖首/中/尾crop、短clip和600帧上限。
- 0步与1步后的推理导出往返必须包含全部新增可训练推理参数；覆盖失败保留旧文件、
  非有限输入/梯度、初始化回滚、OOM/取消边界和CLI/Web未支持组合拒绝。
- 单step记录host/device峰值与耗时，证明内存随clip上限而非训练集总长度增长；
  在Orin通过资源检查前不得进行50步实验或对外宣称可用。
- 先运行相关target与CTest，再运行完整已配置测试集；逐项记录真实资产、失败、skip。
- 同步TR-FR-003/004/005/006/007/009/014、Spec 002 migration/plan、DESIGN、CLI和Web
  契约。听审不通过时FR-014仍为未满足；实施前所有新能力保持Proposed/未实现。
