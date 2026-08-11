# ADR-0003: 退役旧嵌入式 `WebServer` 公共表面

**Status**: Accepted
**Date**: 2026-08-11
**Accepted**: 2026-08-11（维护者先批准“立即弃用并 fail-fast，保留一个迁移周期后删除”，
随后指示“好的，请开动”）
**Owners**: Resonantia maintainers
**Governing specs**: [Spec 003](../../specs/003-webui/spec.md)、
[ADR-0001](0001-modular-runtime-and-orchestration-boundaries.md)、
[重构章程](../REFACTORING_GUIDE.md)

## Context and decision drivers

仓库目前存在两个名称相近但能力完全不同的 Web 表面：

- `tools/vc_serve.cpp` 与内部 `tools/web/` 实现真实 HTTP socket、静态资源和 CLI 编排，
  是 Spec 003 已验证的产品入口；
- `include/voxmutatio/webui/server.h` 与 `src/webui/server.cpp` 暴露一个所谓嵌入式
  `WebServer`，但 `start()` 只进入忙循环，没有监听 socket，默认路由返回伪造成功。

旧公共类还持有不完整类型 `std::unique_ptr<Impl>`，却没有在实现文件中定义析构函数，导致
最小外部调用方仅实例化该类就无法编译。继续补写第二套服务器会复制 Spec 003 的鉴权、
资源上限、事务、进程和安全边界；直接删除又会无迁移窗口地破坏已包含该头文件的源码。

本项目当前不提供已安装 SDK/package config 或稳定动态 ABI，因此本决策保护的是当前源码和
静态链接消费路径，不宣称新的跨版本二进制 ABI 保证。

## Decision

1. `vc_serve` 是唯一受支持的 Web 服务入口。它的 CLI、HTTP、默认开放策略、路由和运行行为
   不因本 ADR 改变。
2. 保留旧 `WebServer`、相关配置/请求/响应类型和 `register_vc_api` 的声明及静态库符号至少
   一个文档化兼容窗口，并在公共头中标记旧服务器入口为 deprecated。
3. 为 `WebServer` 提供 out-of-line 析构及相应特殊成员定义，使包含公共头、实例化、移动和
   析构保持可编译、可链接。
4. `init()`、`register_route()`、`stop()` 和 `is_running()` 保持现有签名；`init()` 仍只保存
   配置以维持源码行为。`start()` 不再打印“启动”或进入忙循环，而是立即抛出带有
   `vc_serve` 迁移指引的 `std::logic_error`，且对象保持非运行状态。
5. 不把 `tools/web/` 暴露成公共嵌入式 API，也不让旧类适配或启动 `vc_serve` 子进程。若未来
   出现真实嵌入式服务器需求，必须单独定义生命周期、线程、认证、资源和错误契约。
6. 旧表面只能在后续明确宣布的不兼容版本中删除；删除前要搜索外部/仓库调用方并再次提供
   迁移通知。当前没有既定发布日期，不能把本次弃用误写成已经删除。

## Alternatives considered

- **完成旧 `WebServer` 的 socket 实现**：会形成第二套 HTTP transport、路由和安全边界，
  与 ADR-0001 的单一 Web 编排实现冲突，拒绝。
- **让旧类包装内部 `tools/web/`**：内部层依赖工具目录、CLI 布局和 POSIX 进程适配器，提升为
  公共 API 会冻结尚未设计的嵌入式生命周期与部署契约，拒绝。
- **本次直接删除头文件和对象代码**：最简单但没有兼容窗口，拒绝。
- **保留当前忙循环并只修析构**：仍会把不可工作的占位实现伪装成服务器，并持续消耗 CPU，
  拒绝。

## Consequences

调用方可以继续包含头文件并构造旧对象，但编译期会收到迁移警告，真正启动时得到快速、
确定且可诊断的失败。项目只需维护一套 HTTP 行为和安全边界，也不会再由占位实现伪造
`/api/convert` 成功。

代价是 `start()` 的错误行为从无限忙循环变为异常，这是刻意的用户可观察修复；尚在调用
旧类的代码必须迁移为启动/管理 `vc_serve` 并使用 Spec 003 HTTP 契约。deprecated 属性可能
使把警告视为错误的消费者需要先局部抑制警告，直到迁移完成。

## Migration and rollback

1. 先把本迁移契约写入 Spec 003，并添加能包含公共头、实例化/析构、调用 `init()` 以及断言
   `start()` fail-fast 的无资产契约测试。
2. 在测试记录旧实现的编译失败后，加入 out-of-line 特殊成员和 `logic_error` 实现；不修改
   `vc_serve` 或 `tools/web/`。
3. 在兼容窗口内保留旧声明和符号；仓库内及外部调用方迁移到 `vc_serve` 进程/HTTP 边界。
4. 后续删除必须作为明确的不兼容变更，更新 Spec、设计、README 和追踪矩阵，并确认调用方
   搜索结果。

若公共头最小消费者仍不能编译/链接、现有 `vc_serve` 契约回退或默认 CTest 注册意外减少，
则停止迁移并恢复旧声明/符号形状；不得通过恢复忙循环来掩盖兼容问题，也不得改动用户数据。

## Verification and SDD impact

- 新增默认、无资产的 legacy public API contract，验证构造/析构、移动、`init()` 兼容和
  `start()` 的异常类型、迁移信息及非运行状态。
- 持续运行既有 `test_web_contract`，证明真实 `vc_serve` HTTP 行为未改变。
- 运行 asset-free contract suite、`scripts/check_sdd.sh`、`git diff --check`。
- 同步 Spec 003、`docs/DESIGN.md`、`specs/TRACEABILITY.md`、`specs/PROJECT_STATE.md` 与
  README；后四项由整体审计修复阶段统一校准。
