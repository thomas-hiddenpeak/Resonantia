# Feature Specification: Voice Training (学习声线与音高)

**Feature Branch**: `002-voice-training`
**Created**: 2026-08-07
**Status**: Approved — 架构 **A(纯 C++/CUDA autograd)** · 40k · fine-tune only

## Decision (2026-08-07)

用户确认 **方案 A:纯 C++/CUDA 训练(自研 autograd)**。

- 训练是本项目**不可分离**的一部分,与推理同等一等公民。
- 训练产物必须是符合本项目与参考项目风格的 **safetensors 权重 + 索引**
  (可被 C++ 推理运行时直接加载)。
- 采样率**先只做 40k**(与已对齐的 f0G40k 一致),后期再完善 32k/48k。
- **只做 fine-tune**(基于预训练模型微调),不做从零预训练。

宪法 XI(Python 仅在 tools/)在此语境下的解读:**训练与推理运行时都用纯
C++/CUDA**;Python 仅用于离线数据准备的一次性辅助(如无必要则不引入)。

## 目标

给定目标说话人/歌手的音频数据集,通过纯 C++/CUDA 的 GAN 微调,产出
`G_<name>.safetensors` + `<name>.index`,供推理"模仿与演唱"该目标声线。

---

## 1. 训练管线概览(对齐 RVC)

```
目标音频目录
  → 预处理(切片/重采样 40k+16k/响度归一)
  → 特征提取(HuBERT 内容 + RMVPE F0,复用已实现 C++ 前向)
  → 频谱(用于 PosteriorEncoder enc_q 与 mel 损失)
  → GAN 微调:
       Generator(enc_p + enc_q + flow + dec)  ← 需反向
       Discriminator(MultiPeriod + MultiScale) ← 需反向
       损失 = mel L1 + KL + feature-matching + adversarial(LSGAN)
       优化器 = AdamW(G) + AdamW(D)
  → 检查点(safetensors)+ 断点续训
  → 导出仅推理所需的 G → safetensors
  → 构建检索索引(复用已实现 C++ build_index)
```

**关键新增(训练专用,推理未实现)**:
- 反向自动微分引擎(autograd)+ AdamW 优化器。
- PosteriorEncoder(enc_q,WaveNet)前向 + 反向。
- 判别器(MPD/MSD)前向 + 反向。
- 训练损失(mel/KL/fm/adv)前向 + 反向。
- 生成器所有算子的**反向**(推理已有前向)。

---

## 2. User Scenarios

### Story 1 — 训练目标声线 (P1)
```bash
vc_train --pretrained-g models/pretrained_v2/f0G40k.safetensors \
         --pretrained-d models/pretrained_v2/f0D40k.safetensors \
         --hubert models/hubert_base/model.safetensors \
         --rmvpe models/rmvpe.safetensors \
         --input-dir data/target_voice --out models/mymodel --epochs 100
# 产出 models/mymodel/G.safetensors + model.index
vc_convert --model models/mymodel/G.safetensors --index models/mymodel/model.index ...
```

### Story 2 — 断点续训 (P2)
从最近检查点恢复,loss 连续。

### Story 3 — 进度可观测 (P2)
输出结构化 epoch/step/loss,供 WebUI(spec 003)消费。

---

## 3. Functional Requirements

- **FR-001 Autograd**:反向自动微分引擎(Tensor + 反向图 + 拓扑反传),
  device 存储用 RAII(CudaBuffer);每个算子有梯度检查(有限差分)门控。
- **FR-002 Optimizer**:AdamW,凸问题上验证收敛。
- **FR-003 训练算子反向**:linear、conv1d(dilation/stride/groups)、
  conv_transpose1d、layer_norm、gelu/leaky_relu/tanh、attention、
  fused tanh·sigmoid;每个都有梯度检查。
- **FR-004 PosteriorEncoder(enc_q)**:前向 + 反向(训练用后验编码器)。
- **FR-005 判别器**:MultiPeriodDiscriminator + MultiScaleDiscriminator 前向 + 反向。
- **FR-006 损失**:mel L1、KL、feature-matching、adversarial(LSGAN),前向 + 反向。
- **FR-007 数据管线**:切片、重采样(40k+16k)、响度归一、特征/F0/频谱缓存、
  filelist、batching(变长 → 截断/填充到训练片段)。
- **FR-008 训练循环**:GAN 交替优化 G/D;梯度裁剪;学习率调度。
- **FR-009 检查点**:safetensors 保存/加载 G+D+optimizer state;断点续训。
- **FR-010 导出**:导出仅推理所需 G → safetensors(与 f0G40k 同构,可被 spec 001 加载)。
- **FR-011 索引**:训练集特征 → `.index`(复用 C++ build_index)。
- **FR-012 CLI**:`vc_train`。

---

## 4. Success Criteria

- **SC-001 Autograd 正确**:所有算子反向相对有限差分误差 < 1e-3(梯度检查)。
- **SC-002 优化器正确**:AdamW 在凸问题上收敛到已知最优。
- **SC-003 端到端可训**:~10 分钟目标音频,完成 N epoch 微调无崩溃/NaN。
- **SC-004 产物可用**:导出 safetensors 被 C++ 推理运行时加载并转换出有效音频(无 NaN/Inf)。
- **SC-005 声线学习有效**:转换输出音色更接近目标(目标说话人 embedding 余弦相似度提升)。
- **SC-006 断点续训**:从检查点恢复后 loss 连续。
- **SC-007 索引可用**:产出 `.index` 被 `vc_convert --index` 正常检索。

---

## 5. 质量门(NON-NEGOTIABLE)

- **梯度检查是训练的"数值对齐"**:每个反向算子必须有有限差分梯度检查测试
  (类比推理阶段的逐样本对齐)。无梯度检查的反向算子不得合入。
- **真实音频**:功能测试用真实目标音频(宪法 X);合成信号仅用于 autograd 单元梯度检查。
- **可复现参考**:关键损失/子图可与 PyTorch(tools/ 离线)对拍一次以确立正确性,
  但训练运行时不依赖 Python。

---

## Completion Checklist

- [ ] FR-001..012 实现
- [ ] SC-001..007 验证(真实目标音频)
- [ ] 所有反向算子梯度检查通过
- [ ] 导出产物经 C++ 推理运行时端到端验证
- [ ] plan.md / tasks 建立并推进
