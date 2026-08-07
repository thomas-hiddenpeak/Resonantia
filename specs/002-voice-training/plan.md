# Implementation Plan: Voice Training (002)

**Branch**: `002-voice-training` | **Date**: 2026-08-07
**Architecture**: 纯 C++/CUDA 自研 autograd(方案 A) · 40k · fine-tune only

> 事实基线见 [specs/PROJECT_STATE.md](../PROJECT_STATE.md)。质量门:每个反向算子必须有有限差分梯度检查(训练版"数值对齐")。

## 策略

复用推理阶段已实现且对齐的**前向**;新增**反向**与训练专用组件。核心是先建一个
**最小 tape-based 反向自动微分引擎**,device 存储用 RAII(顺带偿还审计 P0 的
裸 cudaMalloc 债务),再在其上搭生成器/判别器/损失,最后 GAN 循环。

## Constitution Check (v1.4.0)

| Principle | Compliance | Notes |
|-----------|------------|-------|
| III. Test-First | ✅ | 梯度检查门控每个反向算子 |
| V. Zero-Dependency Purity | ✅ | 纯 C++/CUDA autograd,无第三方训练框架 |
| VI. Numerical Alignment | ✅ | 梯度检查 + 关键子图与 PyTorch 对拍一次 |
| X. Real Audio Only | ✅ | 功能测试用真实目标音频 |
| XI. Python Isolation | ✅ | 训练运行时纯 C++;Python 仅离线一次性对拍 |

## Phases

### Phase A0: Autograd 核心 — ✅ 完成
- [x] `core/cuda_buffer.h` — RAII device 缓冲(替代裸 cudaMalloc/free)
- [x] `autograd/tensor.{h}` + `autograd.cu` — Tensor + 反向图 + 拓扑反传 `backward(loss)`
- [x] 基础算子:add、mul、sum、matmul(前向 + 反向)
- [x] `test_autograd`:有限差分梯度检查(< 4e-6)
- **门控 SC-001(部分)✅**

### Phase A1: NN 算子反向 — ✅ 完成
- [x] linear(bias)、conv1d(dilation/stride/groups)、conv_transpose1d
- [x] layer_norm、gelu、relu、leaky_relu、tanh、sigmoid、softmax
- [x] 每个算子梯度检查通过(max rel err < 4e-4)
- [ ] 相对位置注意力 → 由基元(linear+matmul+softmax+reshape)在 A3 组合
- **门控 SC-001(基元全部)✅**

### Phase A2: 优化器 — ✅ 完成
- [x] AdamW(decoupled weight decay、bias correction、per-param m/v RAII)
- [x] 凸问题收敛测试(max|x-target| 1.2e-7)
- **门控 SC-002✅**

> **可微分算子集 + mel 损失地基已完成(A0–A2 + mel):**
> - 额外基元:transpose2d/concat_rows/slice_rows/embedding/scale/sqrt/log/abs/frame(均梯度检查)
> - 可微 mel 损失(`training/mel_loss`):frame→DFT matmul→幅度→Slaney mel→log→L1
> - 已证明:真实语音 mel 优化 loss 5.19→2.45(autograd+AdamW 端到端)
> **剩余为“模型组装”:在已验证基元上搭生成器/训练循环。**

### Phase A3: 训练专用前向/反向组件 — 进行中(模型组装)
- [ ] 生成器(dec NSF-HiFiGAN)forward-on-autograd(加载预训练权重为参数)
- [ ] PosteriorEncoder(enc_q):spec + WaveNet + 反向
- [ ] MultiPeriodDiscriminator + MultiScaleDiscriminator:前向 + 反向
- [ ] 梯度检查

### Phase A4: 损失 — mel 已完成
- [x] mel L1(可微 STFT + Slaney mel + log,已验证驱动优化)
- [ ] KL 散度(m_p/logs_p/m_q/logs_q)
- [ ] feature-matching、adversarial(LSGAN)
- [ ] 与 PyTorch 对拍一次

### Phase A5: 数据管线
- [ ] 预处理:静音切片、重采样(40k + 16k)、响度归一
- [ ] 特征/F0/频谱提取与缓存(复用 C++ HuBERT/RMVPE 前向)
- [ ] filelist + batching(训练片段截断/填充)

### Phase A6: 训练循环
- [ ] GAN 交替优化 G/D;梯度裁剪;lr 调度
- [ ] 结构化进度输出(epoch/step/loss)
- **门控 SC-003**

### Phase A7: 检查点与导出
- [ ] safetensors 保存/加载 G+D+optimizer;断点续训
- [ ] 导出仅推理 G(与 f0G40k 同构)
- [ ] `tools/build_index` 产出 `.index`
- **门控 SC-006、SC-007**

### Phase A8: 端到端验证
- [ ] `vc_train` CLI
- [ ] 真实目标音频 fine-tune → 导出 → C++ 推理运行时转换验证
- **门控 SC-004、SC-005**

## 风险与约束

| 风险 | 缓解 |
|------|------|
| Autograd 工程量大 | 最小 tape 引擎,只实现所需算子;梯度检查逐个门控 |
| 反向数值不稳 | 有限差分梯度检查 + 关键子图 PyTorch 对拍 |
| 显存/性能 | RAII 显存池;训练片段定长;先正确后优化 |
| GAN 不收敛 | 从预训练权重 fine-tune(非从零);对齐 RVC 超参 |

## 复杂度记录

| 选择 | 原因 | 被拒替代 |
|------|------|---------|
| 自研 tape autograd | 全生成器需反向,手写逐算子不可维护 | 无框架逐函数反向(不可维护) |
| device RAII(CudaBuffer) | 训练大量临时张量;偿还审计债务 | 裸 cudaMalloc(泄漏风险、无异常安全) |
