# ADR-0009: 可恢复的联合长训练与固定前端缓存

**Status**: Proposed
**Date**: 2026-10-26
**Owners**: Resonantia maintainers
**Governing specs**: [Spec 002](../../specs/002-voice-training/spec.md)、[Spec 003](../../specs/003-webui/spec.md)
**Implementation authorization**: 维护者先明确同意能力补齐计划，随后要求“按计划制定完整的目标推进吧”。
该授权涵盖完整状态保存/恢复、冻结前端缓存、轮次/LR调度、训练过程实测及200轮训练。
本记录在代码实施前建立，保留Proposed作为正式ADR评审状态；不重复索取已给出的实施授权。

## Context and decision drivers

当前joint短训练数值门禁及具体CyberTom短样本通过，但只有最终推理G导出，最多50步，
每步重复提取冻结HuBERT/RMVPE与频谱。推理导出不包含enc_q、D、optimizer或RNG，不能续训。
维护者已将下一目标扩展为完整训练：包含粤语及《劲歌金曲》的全部素材按质量筛选，
验证去混响，200轮训练后集中听审；真实性能在训练中记录，不先预测耗时。

## Decision

1. 保留现有legacy和50步joint入口；新增显式长训练选择，不静默修改旧默认或取消契约。
   单GPU、FP32、40k、T<=600、crop32及原严格数值门槛保持。
2. JointTrainer新增完整状态保存/恢复、学习率与步数访问。恢复包含训练G（含enc_q）、D、
   全部AdamW一/二阶矩、步数/LR、trainer RNG、原始enc_q和其他speaker行的推理保留值。
   应用层另提供受限run state：数据身份、配方、epoch、下一片段位置、shuffle/crop RNG。
   正常状态必须在完整D/G对之后保存；推理G文件绝不能被误接受为恢复文件。
3. 恢复格式版本1采用单个safetensors文件，参数/矩阵为F32。受限文本状态逐字节编码为
   F32精确整数[0,255]，保留现有loader仅接受F32的契约，不为恢复文件放宽推理dtype边界。
   先验证manifest/shape/dtype/长度/有限值与版本，再在暂存trainer中恢复并提交；失败
   保留原可用trainer与调用者状态。写入以同目录唯一临时文件、flush/fsync、rename提交；
   写入失败保留旧文件。逐tensor读取/写入，禁止形成整个训练状态的多份host副本。
4. 固定HuBERT、RMVPE、目标频谱可按素材和前端身份持久缓存；只物化当前clip。
   缓存不包含可训练prior/posterior/Flow结果，也不冻结每步随机激励。损坏、配置变化、
   素材变化必须失效或拒绝；缓存通过与直接提取结果及下一步数值一致性验证。
5. 长训练按完整数据遍历定义epoch，初始LR=1e-4，每轮衰减0.999875，总目标200轮。
   每5轮保存完整恢复点并滚动保留最近2个成功版本；50/100/150/200轮另存推理候选。
   恢复不重放已保存状态包含的更新、不重置LR，不向Web发布半成品。
   协作取消保存当前完整步；异常退出只能从最后成功保存点恢复，未保存的更新需要重算。
6. 记录每轮/阶段的真实耗时、loss、已处理片段/更新数、缓存命中、GPU和内存；监测不改变
   优化顺序。不将GPU忙碌率解释为吞吐量，也不声称已比参考实现更快。
7. batch=1仍是当前已验证数值契约。更大batch须独立正确性与资源验证后才能采用；
   不将梯度累积伪称为并行GPU batch，不通过改变精度或放宽门槛绕过验证。

## Alternatives considered

- 直接解除50步限制：无法保留中途进度，重复前端开销，拒绝。
- 用推理G继续：丢失enc_q/D/optimizer/RNG，不能等价恢复，拒绝。
- 全数据GPU缓存：本批约22.9分钟原始素材，使用有界按clip读取与磁盘缓存。
- Python生产训练：违反运行时约束；Python仅用于tools中的独立参考和离线分析。

## Migration and rollback

先建立AdamW恢复失败/等价单步测试，再接JointTrainer完整状态与真实模型恢复测试；
之后接显式CLI长训练、缓存、schedule与取消，最后Web编排。旧profile/已发布模型保留。
正式产物使用新目录CyberTom_Full_20261026；原始素材只读，数据/缓存/恢复点/候选分开存放。
每个阶段独立验证；任一原数值或事务门槛失败则停止该阶段推进，保留失败记录及已提交状态。
未知版本、不同模型/配方/数据集不允许静默续训；提供显式失败与重开新run的路径。
最终模型仅在集中客观验证和维护者听审接受后接入Web，保留CyberTom_20261026短训练版。

## Verification and SDD impact

- TR-FR-002/007/008/009/011/014/015与WEB-FR-005/006/010/011受影响。
- optimizer保存/恢复下一步，非法尺寸/NaN/负二阶矩/错误步数拒绝且不改变旧状态。
- 真实G/D/语音：连续训练与保存、销毁、恢复后的下一步参数/矩/RNG输出比较；
  同时检查推理保留值、缺文件/损坏/未知版本、原子失败、恢复初始化回滚。
- CLI数据改变/缓存损坏/调度边界、取消后恢复、失败不发布；Web进度与GPU串行。
- 保留原joint独立数值门槛，运行相关target/CTest再完整已配置套件，不把skip算通过。
- 更新Spec 002/003、plan、DESIGN、TRACEABILITY、PROJECT_STATE及CLI说明，执行check_sdd。
