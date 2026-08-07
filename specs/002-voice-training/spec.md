# Feature Specification: Voice Training (学习声线与音高)

**Feature Branch**: `002-voice-training`
**Created**: 2026-08-07
**Status**: Draft — **需要架构决策(见 §1)后才能进入 plan**

## 目标

实现产品的另一半:**学习声线与音高**。给定目标说话人/歌手的一段音频数据集,
产出一个可被 C++ 推理运行时(spec 001)直接加载的 safetensors 模型 + 检索索引,
使得后续推理能"模仿与演唱"该目标声线。

对应 RVC 训练管线(`train/`):数据预处理 → 特征/F0 提取 → GAN 微调
(VITS 生成器 + 多周期判别器)→ 检查点 → 构建检索索引。

---

## 1. 架构根本决策(NON-NEGOTIABLE 前置)

RVC 训练是**完整的对抗训练(GAN)**:VITS 生成器与多周期/多尺度判别器交替优化,
损失包含 mel L1、KL 散度、特征匹配、对抗(LSGAN)。这需要**每个算子的反向传播 +
优化器(AdamW)**。当前推理实现只有前向。

三条可行路线:

### 方案 A:纯 C++/CUDA 训练(自研 autograd)
- 为 VITS + 判别器所有算子实现反向 + AdamW + GAN 循环。
- ✅ 完全符合"推理和训练都脱离 Python"的原始愿景。
- ❌ 工程量巨大(需完整反向传播框架),远超全部推理工作之和;风险高、周期长。

### 方案 B:tools/ 离线微调(uv + PyTorch),产出 safetensors(推荐)
- 训练作为**离线、一次性**过程放在 `tools/`(uv 管理),复用 RVC 训练逻辑,
  产出 safetensors 权重 + 索引,由**纯 C++ 推理运行时消费**。
- ✅ 符合宪法 XI(Python 仅在 tools/;运行时零 Python 依赖),与 Orator
  "Python 仅作 tools/ 下离线工具"的约定一致。
- ✅ 可快速交付可用的"学习声线"能力,让产品闭环。
- ⚠️ 训练本身不是纯 C++(但运行时仍是)。

### 方案 C:纯 C++ 推理侧轻量自适应
- 不做完整 GAN:仅在 C++ 侧优化 speaker embedding / 少量层(需要局部反向)。
- ✅ 比 A 小;❌ 效果弱于完整微调,仍需部分 autograd。

**推荐:方案 B。** 理由:训练是离线产物生产,宪法明确允许 tools/ 使用 Python;
可最快让产品闭环("学习声线"真正可用),而运行时零 Python 依赖的核心承诺不变。
方案 A 可作为远期 spec 单独立项。

> **本 spec 在决策确定前不进入 plan/实现。** 下方需求以方案 B 为默认草拟,
> 若选 A/C 将大幅改写。

---

## 2. User Scenarios

### Story 1 — 训练目标声线 (P1)
用户提供目标说话人的音频(一个目录),运行训练,得到 `G_<name>.safetensors`
与 `<name>.index`,可直接用于 `vc_convert --model G_<name>.safetensors --index <name>.index`。

**Independent Test**:
```bash
# 方案 B
tools/ (uv) train --input-dir data/target_voice --pretrained f0G40k --out models/mymodel
build/vc_convert --model models/mymodel/G.safetensors --index models/mymodel/model.index ...
```

### Story 2 — 断点续训 (P2)
训练中断后可从最近检查点恢复。

### Story 3 — 进度可观测 (P2)
训练过程输出 epoch/step/loss,可被 WebUI(spec 003)展示。

---

## 3. Functional Requirements(方案 B 默认)

- **FR-001** 数据预处理:切片(静音切分)、重采样到 40k/16k、响度归一。
- **FR-002** 特征提取:HuBERT 内容特征 + RMVPE F0(可复用 C++ `build_index` 的特征路径或 tools 提取)。
- **FR-003** 数据集构建:filelist(audio|feature|f0|speaker)。
- **FR-004** 微调训练:加载预训练 f0G40k,GAN 微调,损失 = mel + KL + fm + adv。
- **FR-005** 检查点:定期保存 G/D;导出**仅推理所需**的 G → safetensors(去除判别器/优化器状态)。
- **FR-006** 索引构建:训练集特征 → `.index`(可直接调用已实现的 C++ `build_index`)。
- **FR-007** 恢复训练:从检查点续训。
- **FR-008** 进度输出:结构化 epoch/step/loss(供 WebUI 消费)。

---

## 4. Success Criteria

- **SC-001** 端到端可训:给定 ~10 分钟目标音频,完成 N epoch 微调,无崩溃。
- **SC-002** 产物可用:导出的 safetensors 能被 C++ 推理运行时加载并转换出有效音频(无 NaN/Inf)。
- **SC-003** 声线学习有效:转换输出的音色更接近目标(主观 + 目标说话人 embedding 余弦相似度提升)。
- **SC-004** 索引可用:产出的 `.index` 能被 `vc_convert --index` 正常检索。
- **SC-005** 断点续训:从检查点恢复后 loss 连续。

---

## 5. 待你决策的开放问题

1. **训练架构 A / B / C?**(决定本 spec 走向;推荐 B)
2. 目标采样率:先只做 40k(与已对齐的 f0G40k 一致)?还是也要 32k/48k?
3. 训练产物是否需要同时导出 D(判别器)以便继续训练,还是只保留可推理的 G?
4. 是否需要"从零预训练"(pretrain),还是只做"微调"(fine-tune,推荐)?

---

## Completion Checklist(方案 B,待决策后细化)

- [ ] 架构决策已确认
- [ ] FR-001..008 实现
- [ ] SC-001..005 验证(真实目标音频)
- [ ] 产物经 C++ 推理运行时端到端验证
- [ ] plan.md / tasks 建立
