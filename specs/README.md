# Resonantia SDD 导航与证据链

本目录保存产品意图、验收条件和历史实施决策。它不是运行时能力清单；一个
spec 写着 `Complete`，不等于当前提交已经构建并执行了对应测试。当前需求到代码的
映射见 [TRACEABILITY.md](./TRACEABILITY.md)，不改外部行为的工程调整见
[REFACTORING_GUIDE.md](../docs/REFACTORING_GUIDE.md)。

## 两条证据链

规范权威与实现事实回答不同问题，不能合并成“代码优先”或“文档优先”的单一列表。

### 规范权威链：应该是什么

1. **用户批准的变更**：本次明确批准的目标、范围和取舍。
2. **项目宪章**：`.specify/memory/constitution.md` 的治理与不可协商工程原则。
3. **批准的 feature spec**：对应 `spec.md` 的用户行为、FR、验收场景和成功标准。
4. **批准的架构设计**：[docs/DESIGN.md](../docs/DESIGN.md) 对模块边界、依赖和部署
   契约的解释。架构性变更还需要宪章要求的
   [ADR](../docs/adr/README.md) 与迁移章节。

较低层规范不得静默覆盖较高层规范。spec 顶部的历史 `Complete` 状态也必须满足宪章的
完成条件，不能仅靠勾选或旧日志成立。

### 实现事实链：现在是什么

1. **当前可观察表面**：`include/voxmutatio/` 公共 API 和默认值、`tools/*.cpp` 的 CLI
   参数与 `--help`、HTTP 路由以及 WAV/safetensors/Index 等格式定义。
2. **当前实现**：`src/` 与 `tools/` 的实际接受、拒绝、资源和错误行为。声明了字段但
   实现明确拒绝时，当前状态仍是“不支持”。
3. **构建与实际验证**：根 `CMakeLists.txt`、`cmake/ResonantiaTests.cmake`、CTest
   实际注册结果、CI 配置，以及从当前树记录的构建/测试命令与结果。测试文件存在不等于
   注册，注册不等于执行，skip 或缺资产不等于通过。

两条链不一致时，结论是 **drift（规范—实现偏差）**：同时记录规范要求和已实现行为，
在 `TRACEABILITY.md` 建立待修复项，并在授权范围内修实现或发起规范变更。代码中的 bug
不能自动改写规范；规范声明也不能替代运行证据。

[README.md](../README.md)、[PROJECT_STATE.md](./PROJECT_STATE.md) 与本追踪表都是维护性
摘要，应由上述两条链派生，不能覆盖它们。各 feature 的 `plan.md`、`research.md`、
日期化实验记录和 [docs/ENGINEERING_AUDIT.md](../docs/ENGINEERING_AUDIT.md) 属于历史材料；
工程审计尤其是 **2026-08-07 的历史快照**，只能解释问题来源，不能证明当前能力、测试
数量或性能。

不要通过删除失败测试、放宽阈值、把可选资产测试称为默认 CI，或把当前 bug 写进 spec
来“解决” drift。

## 状态词汇

追踪文档只使用以下状态：

- **已实现·默认门禁**：实现存在，且有不依赖忽略资产的已注册 CTest。
- **已实现·可选门禁**：实现存在，模型/参考资产齐全时有已注册 CTest；默认 CI 不证明它。
- **部分**：只有部分范围、工具入口或测试证据，不能宣称完整需求满足。
- **不支持**：当前契约明确拒绝或没有实现。
- **未知**：缺少足够的当前代码或执行证据；不得推断为通过。

“本地已通过”必须另外记录运行日期、配置、测试名和资产；它不是永久状态词。

## 按任务阅读

| 任务 | 先读 | 再核对 |
|---|---|---|
| 修改 VC 配置、推理或 CLI | [001 spec](./001-voice-conversion-engine/spec.md)、[追踪表](./TRACEABILITY.md#vc-推理与转换) | `core/types.h`、`pipeline/`、相关工具、pipeline/e2e CTest |
| 规划或实现实时变声 | [006 realtime spec](./006-realtime-voice-conversion/spec.md)、[ADR-0002](../docs/adr/0002-persistent-streaming-runtime-and-local-audio-boundary.md) | 001 离线契约、pipeline/synth `infer_stream` 事实、Index exact reference/基线、stream benchmark/ALSA 门禁；ADR 已 Accepted，从 R0 证据开始实施 |
| 修改 autograd、微调或 GAN | [002 spec](./002-voice-training/spec.md)、[训练追踪](./TRACEABILITY.md#训练) | `autograd/`、`training/`、`vc_train`、autograd/training CTest |
| 修改 STFT、分离或 VAD | [004 spec](./004-source-separation/spec.md)、[分离追踪](./TRACEABILITY.md#声源分离与预处理) | `separation/`、`vc_preprocess`、可选 separation CTest 与资产清单 |
| 修改 Web/API/上传 | [003 WebUI spec](./003-webui/spec.md)、[Web 追踪](./TRACEABILITY.md#web-服务) | `tools/vc_serve.cpp`、`tools/web/`、`webui/`、真实 HTTP/CLI 进程契约 |
| 修改 WAV、safetensors、Index 或输入边界 | [I/O 与安全追踪](./TRACEABILITY.md#io输入边界与安全) | 公共头、`io/`、`index/`、[ADR-0004](../docs/adr/0004-bounded-offline-file-conversion.md)、[ADR-0005](../docs/adr/0005-immutable-flat-index-snapshot.md)、默认 contract CTest |
| 修改构建、依赖或 CI | [构建追踪](./TRACEABILITY.md#构建与验证) | `CMakeLists.txt`、`cmake/ResonantiaTests.cmake`、CI workflow |
| 纯重构 | [重构章程](../docs/REFACTORING_GUIDE.md) | 受影响需求行、依赖方向、迁移/回滚与全部相关门禁；改变架构边界时先写 [ADR](../docs/adr/README.md) |
| 汇报项目状态 | [追踪表](./TRACEABILITY.md) | 当前支持矩阵和一份可复现测试运行记录；不要引用历史审计作现状 |

## Spec 索引与使用边界

| 文档 | 规范范围 | 使用说明 |
|---|---|---|
| [001 Voice Conversion](./001-voice-conversion-engine/spec.md) | 内容、F0、检索、合成、pipeline、bounded/legacy 文件转换与工具 | 功能目标有效；顶部完成状态和内嵌历史数值需由当前追踪证据复核 |
| [001 data model](./001-voice-conversion-engine/data-model.md) | 核心数据结构意图 | 与 `core/types.h` 不一致时记录 drift，不由代码自动覆盖模型意图 |
| [002 Voice Training](./002-voice-training/spec.md) | C++/CUDA 微调与 GAN | 断点续训、生产稳定性和音色效果等未验证项不得由已有训练 smoke 推断 |
| [003 WebUI](./003-webui/spec.md) | Web/API、上传、训练与转换编排 | As-built baseline；默认开放、可选 token、WAV-only、非实时且服务本身无 TLS |
| [004 Source Separation](./004-source-separation/spec.md) | STFT、UMX、RoFormer、VAD、预处理接入 | 大模型与 reference 测试属于可选资产门禁 |
| [006 Realtime Voice Conversion](./006-realtime-voice-conversion/spec.md) | 持久流式 runtime、本机 ALSA、Index-enabled 质量路径、时延/背压/连续性 | `Approved` / R0 evidence 部分完成 / 产品未实现；已有 legacy characterization、报告 validator、manifest 与 standalone exact-Index 诊断 probe，后者不是产品 backend/上限；目标 model+matching Index/licensed corpus 仍阻塞 qualification |
| [PROJECT_STATE](./PROJECT_STATE.md) | 维护性状态摘要 | 仅作导航；冲突时回到规范权威链和实现事实链并记录 drift |

新增或拆分 spec 时，应同时分配稳定需求 ID、在 `TRACEABILITY.md` 建行、声明资产和
测试层级，并写清不在范围内的能力。计划和研究可以保留被替代方案，但必须标明日期和
历史状态。

编号 `005` 已 retired，按编号不复用规则不得重建；实时提案因此使用 `006`。治理原因与
校验约束记录在 [006 research](./006-realtime-voice-conversion/research.md#1-evidence-boundary)。
