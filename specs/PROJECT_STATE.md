# Resonantia 项目状态 (PROJECT_STATE)

> **代码是最高事实来源。** 本文件汇总各能力的真实状态;若与其他文档冲突,以代码与本文件为准。
> 最后更新: 2026-08-07

## 项目最终目标

**训练 + 推理 = 学习声线与音高 + 模仿与演唱。**
一个可训练目标声线、并通过面向普通用户的 WebUI 进行歌声/语音转换的纯 C++/CUDA 引擎(RVC 的零 Python 运行时重构)。

---

## 能力状态总表

| 能力 | 状态 | 证据 |
|------|------|------|
| 音频 I/O (WAV 读写 + 重采样) | ✅ 可用 | `test_audio_io` |
| safetensors 零拷贝加载 | ✅ 可用 | `test_model_load`,213/741/560 张量 |
| HuBERT (ContentVec) 内容编码 | ✅ 对齐 | `test_hubert_inference` RMS 7.1e-07 |
| RMVPE F0 提取 | ✅ 对齐 | `test_rmvpe_alignment` 1.5e-05 Hz |
| CUDA Flat Index (加权 k-NN 检索) | ✅ 可用 | `test_index_retrieval` cosine 1.0 |
| VITS 合成 (TextEncoder+Flow+NSF-HiFiGAN) | ✅ 对齐 | `test_vits_alignment` 音频 corr 0.9997 |
| 端到端推理 Pipeline | ✅ 可用 | `test_e2e_conversion` corr 0.9997 |
| CLI: vc_convert / vc_train / vc_preprocess / build_index / vc_serve | ✅ 可用 | 转换/微调(含 --gan)/切片/建索/HTTP 服务,真实音频端到端验证 |
| 端到端使用工作流 (scripts/) | ✅ 可用 | `train_voice.sh` + `convert_voice.sh`:切片→微调→建索→转换全链路跑通 |
| **训练 / 微调(解码器 + 完整 GAN)** | ✅ 可用 | 解码器 fine-tune(mel -70%,导出往返 corr 1.0)+ 完整 VITS GAN(enc_q/flow/KL/MPD-V2/adv/fm),真实音频损失有限收敛 |
| **WebUI (可运行)** | ✅ 可用 | `vc_serve`(纯 C++ HTTP,静态 + `/api/convert`),curl 验证前端可用 |
| WavLM 内容编码 | ⚠️ 桩 | 文件存在,未接入/未验证 |
| FCPE 实时 F0 | ⚠️ 桩 | 未实现 |
| 声源分离 (separation/) | ❌ 未实现 | 仅设计文档提及 |
| CLI: vc_batch / vc_probe | ⚠️ 骨架 | 存在但未随对齐更新验证 |

---

## Spec 状态

| Spec | 主题 | 状态 |
|------|------|------|
| 001-voice-conversion-engine | 推理引擎(内容/F0/检索/合成/pipeline) | Complete(推理部分) |
| 002-voice-training | 训练/微调:纯 C++/CUDA autograd(方案 A) · 40k | Complete(解码器 + 完整 GAN)— A0-A2 autograd/AdamW/可微 mel;解码器对齐 0.9997、fine-tune -70%、导出往返 corr 1.0;完整 VITS GAN:exp/conv2d/flip_rows 梯度检查、enc_q 重建 corr 0.956、flow 逆 9e-7、MPD-V2 判别器、KL/fm/adv,`vc_train --gan` 端到端 |
| 003-webui | 面向普通用户的 WebUI:`vc_serve` 后端 + 前端 | Complete(基础)— 纯 C++ HTTP 服务静态前端 + /api/convert,curl 验证 |

> 注:`001` 的 spec.md 已标 Complete 并附验证清单;其 `plan.md` 覆盖范围原本包含训练/WebUI(P6),但这两项实际未完成,已在下方"已知偏差"记录,并将拆分为独立 spec 002/003。

---

## 已知偏差(文档 vs 现实)

1. `plan.md` 的 Phase 复选框历史上全部未勾选,与实际完成状态不符 —— 本轮已校准。
2. `plan.md` Constitution Check 表停留在 9 条原则;宪法现为 11 条(新增 X. Real Audio Only,并重排),版本 v1.4.0。
3. `DESIGN.md` 将 separation/WavLM/FCPE 描述为架构组成,实际未实现 —— 已加"实现状态"标注。
4. `001/plan.md` 的 P6(训练 + WebUI)未落地 —— 拆为 spec 002/003。

---

## 下一步(优先级,2026-08-07 调整后)

- **P0 文档真相校准**(本轮进行中):审计落盘、PROJECT_STATE、plan/DESIGN 校准、NOTICE。
- **P1 产品闭环**:spec 002 训练;spec 003 WebUI(简单/高级双模式 + `vc_serve`)。
- **P2 代码硬化**:CudaBuffer RAII + 显存池(消除 65 处裸 malloc);内核错误检查;方法命名决策。
- **P3 收敛设计**:DESIGN 与现实完全对齐;处理孤儿模块。

---

## 工程规则(源自对齐实践与风格参考)

- 正确性先于优化:每个优化内核必须有简单参考实现 + 数值对比测试。
- 仅用真实音频做功能测试(宪法 X);合成信号仅用于底层工具 smoke test。
- Python 仅存在于 `tools/`(uv 管理);运行时零 Python 依赖。
- 性能声明必须附:设备、功耗模式、时钟、CUDA/JetPack 版本、显存占用。
