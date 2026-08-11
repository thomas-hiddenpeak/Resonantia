# Resonantia Architecture Decision Records

ADR 记录会长期影响模块边界、依赖、持久格式、运行时、部署或兼容性的架构决策。
普通缺陷修复、行为保持型重构、局部性能优化和实现笔记不创建 ADR；它们按
[`../REFACTORING_GUIDE.md`](../REFACTORING_GUIDE.md) 和相关 Spec 管理。

## 文件与状态

- 文件名：`NNNN-short-decision-title.md`，编号只增不复用。
- 状态：`Proposed`、`Accepted`、`Rejected` 或 `Superseded by ADR-NNNN`。
- `Accepted` 需要维护者批准；Codex 或实现者不能自行把提案标为已批准。
- 决策改变时新增 ADR 并把旧 ADR 标为 superseded，不重写旧决策的历史背景。

## 当前记录

- [`ADR-0001`](0001-modular-runtime-and-orchestration-boundaries.md)：模块化运行时与应用
  编排边界（`Accepted`，2026-08-10；R0–R5 已实施，原完成记录为 27 项 CTest；同日
  Spec 006 R0 历史快照扩展为 29/29，当前结果见项目状态）。
- [`ADR-0002`](0002-persistent-streaming-runtime-and-local-audio-boundary.md)：持久流式
  推理运行时与本机音频边界（`Accepted`，2026-08-10；本机 ALSA、Index-enabled
  qualified profile、250/350 ms SLO 与 bounded drop-oldest 已批准；R0 evidence 已部分完成，
  但 target model + matching Index/licensed corpus 仍阻塞 qualification，当前仍无 realtime
  产品路径）。
- [`ADR-0003`](0003-retire-legacy-embedded-webserver.md)：退役不可工作的旧嵌入式
  `WebServer` 公共表面（`Accepted`，2026-08-11；`vc_serve` 为唯一受支持 Web 入口，旧声明
  保留一个兼容窗口并 deprecated，`start()` 改为带迁移指引的 fail-fast）。

## 最小模板

```markdown
# ADR-NNNN: Decision title

**Status**: Proposed
**Date**: YYYY-MM-DD
**Owners**: ...
**Governing specs**: ...

## Context and decision drivers

当前约束、问题、证据和必须保持的外部行为。

## Decision

选择的边界、依赖方向、所有权、错误和兼容契约。

## Alternatives considered

候选方案、拒绝理由和仍然未知的条件。

## Consequences

正面影响、成本、风险和新增运维/测试责任。

## Migration and rollback

分阶段切换、兼容窗口、数据/权重处理、回滚条件和恢复步骤。

## Verification and SDD impact

受影响的 FR/追踪行、characterization test、数值/资源门禁，以及需要同步的
Spec、DESIGN、README、CLI/API 文档。
```

## 使用流程

1. 在实现前以 `Proposed` 提交，链接受影响的 Spec 和
   [`../../specs/TRACEABILITY.md`](../../specs/TRACEABILITY.md) 行。
2. 明确外部行为是否变化；变化时先完成需求与迁移评审。
3. 获批后再按 ADR 的阶段切换代码，每一步保留可复现验证和回滚点。
4. 完成迁移后同步追踪矩阵和当前设计；ADR 保留决策原因，不承担项目状态汇总。
