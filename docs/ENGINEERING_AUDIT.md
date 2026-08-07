# Resonantia 工程审计报告

**日期**: 2026-08-07
**审计范围**: 项目架构、SDD 文档、实际推进、代码质量、WebUI/UX、风格一致性
**基线**: 三大模型数值对齐完成、端到端可用、10/10 测试通过之后的首次全面审计
**参考**: 功能参考 [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI);风格参考 [Qwen3x-Orin](https://github.com/thomas-hiddenpeak/Qwen3x-Orin)、[Orator](https://github.com/thomas-hiddenpeak/Orator)

---

## 0. 结论摘要

核心推理链是坚实的 A 级资产(三模型 + e2e 逐样本对齐)。但距离产品化(**训练 + 推理 = 学习声线音高 + 模仿演唱**)最大的两块短板是:**训练完全缺失(0%)** 与 **WebUI 不可运行且 UX 不达标**。文档治理与 CUDA 内存管理是次级但明确的技术债。

| 维度 | 评分 | 一句话结论 |
|------|------|-----------|
| 推理正确性 | A | 三模型 + e2e 数值对齐,真实音频验证,10/10 测试 |
| 项目架构 | C+ | 设计文档超前于现实;训练缺失;有孤儿模块 |
| SDD 文档 | D | 单一巨型 spec;plan.md 陈旧;缺 STATE/ROADMAP/CHANGELOG |
| 代码质量 | C | 65 处裸 cudaMalloc(违反 RAII);内核启动无错误检查 |
| 性能 | D | 合成 ~8s(应 <1s);瓶颈=每算子 malloc/free |
| WebUI/UX | D | 前端无后端;全是技术名词;无简单/高级模式 |
| 风格一致性 | C | 方法名 snake_case 与参考仓库 PascalCase 相悖 |

---

## 1. 推理正确性(A) — 已验证的资产

真实 LibriSpeech 语音,与 Python 逐样本对齐:

| 阶段 | 指标 | 结果 | 阈值 |
|------|------|------|------|
| HuBERT v2 (layer12) | RMS | 7.09e-07 | < 1e-4 (SC-001) |
| RMVPE F0 | 平均误差 | 1.52e-05 Hz | < 0.5 Hz (SC-002) |
| VITS 音频 | 相关性 | 0.9997 | > 0.999 (SC-003) |
| 端到端 | 相关性 | 0.9997 | > 0.99 |

方法论(为每个 Python 组件建可复现参考 + C++ 逐阶段对齐)已被证明有效,应固化为项目工程规则。

---

## 2. 架构审计(C+) — 设计与现实脱节

`docs/DESIGN.md` 与 `specs/001/plan.md` 描述 10 大模块,但实际:

| 设计声称 | 实际状态 |
|---------|---------|
| `separation/` 声源分离(HTDemucs/MDX) | 未实现,src 下无此目录 |
| 双内容编码器(WavLM + HuBERT) | wavlm_encoder.cpp 存在但从未接入/验证,只用 HuBERT |
| 双 F0(RMVPE + FCPE) | 只有 RMVPE 对齐,FCPE 未实现 |
| 训练(微调) | `src/training/fine_tuning.cpp` 是 TODO 桩,无真实训练 |

**风险**:`plan.md` 的 Constitution Check 表仍是 9 条旧原则(现 11 条),所有 Phase 复选框未勾选,与"10/10 测试通过"直接矛盾,会误导任何接手者(包括 AI agent)。

---

## 3. SDD 文档审计(D) — 最薄弱

1. **单一巨型 spec**:`001` 覆盖整个引擎。参考仓库拆成聚焦的多 spec + `PROJECT_STATE.md` 汇总。**训练必须是独立 spec**。
2. **plan.md 陈旧**:阶段全部未勾选,与现实矛盾。
3. **缺关键文档**:参考仓库有 ROADMAP/ARCHITECTURE/PROJECT_STATE/PERFORMANCE_BASELINE/CHANGELOG/CONTRIBUTING/SECURITY/AGENTS/NOTICE。Resonantia 只有 DESIGN.md。
4. **无署名(NOTICE)**:改编自 RVC + HuBERT/RMVPE/transformers 架构,应显式署名。

---

## 4. 代码质量审计(C)

**P0 — 裸 cudaMalloc 反模式(65 处)**
`src/synthesizer/vits_ops.cu`、`src/f0/rmvpe_ops.cu`、`src/content/cuda/kernels.cu` 中每个算子都 `cudaMalloc → memcpy → 计算 → memcpy → cudaFree`。违反 Orator"性能路径禁止裸 cudaMalloc,必须 RAII",也是合成器 ~8s 的根因。

**P0 — 内核启动无错误检查**
`kernel<<<>>>()` 之后普遍无 `cudaGetLastError()` / 同步检查。违反"每个 CUDA 调用必须检查错误"。

**P1 — 方法命名分歧**
`.clang-format` 定义函数=snake_case;但 Qwen3x-Orin/Orator 都用 Google 风格 PascalCase 方法。需拍板。

**P1 — 性能全 host 化**
注意力、GRU、mel 滤波在 host 循环跑(为先对齐)。正确性已达成,现为明确优化目标。

---

## 5. WebUI/UX 审计(D) — 与产品要求差距最大

1. **前端无后端**:`webui/` 有前端,`src/webui/server.cpp` 存在,但 CMakeLists 无任何 server 可执行文件。前端 `fetch('/api')` 无处可连,**WebUI 不可运行**。
2. **全是技术名词**:当前选项为"说话人 ID / 音高偏移(半音)/ 特征检索率 / RMS 混合率 / 清音保护",且是数字输入框。与产品要求相反。
3. **要求**:简单模式用滑块 + 效果描述(如"音调:低沉↔尖细"、"音色相似度:更像我↔更像目标");高级模式才暴露真实参数。
4. **形态**:采用 Orator 的无框架、无构建步骤 ES module SPA + 内置 HTTP 静态服务器。

---

## 6. 路线图(优先级)

拆成独立 spec,遵循 SDD:

**P0 — 文档真相校准(消除误导源,已优先前置)**
- 落盘本审计;更新 plan.md/DESIGN.md 反映现实;新增 PROJECT_STATE.md;NOTICE 署名。

**P1 — 产品闭环**
- spec 002:训练/微调(数据预处理、损失、优化器、检查点)——"学习声线音高"的落地。
- spec 003:WebUI(新增 `vc_serve`,简单/高级双模式,简单模式效果化滑块)。

**P2 — 代码硬化(不改行为)**
- `CudaBuffer` RAII + 每模块持久化显存池,消除 65 处裸 malloc;注意力/GRU 迁回 GPU;目标合成 <1s。
- 内核启动后加 `cudaGetLastError()`。
- 决策方法命名。

**P3 — 收敛设计**
- DESIGN.md 与现实对齐:separation/WavLM/FCPE 标"规划中";明确孤儿模块。

---

## 7. 优先级调整说明(2026-08-07)

原设计顺序为 P0=产品闭环、P2=文档。经评估,**陈旧 SDD 文档是主动误导源**,若在文档矛盾状态下推进训练 spec,判断会被污染。故将**文档真相校准提前为最高优先级**(先清理地基再施工),再推进训练与 WebUI。此调整已被采纳并执行。
