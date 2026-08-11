# ADR-0001: 模块化运行时与应用编排边界

**Status**: Accepted
**Date**: 2026-08-09
**Accepted**: 2026-08-10（维护者指示“按照计划进行”）
**Implementation completed**: 2026-08-10
**Owners**: Resonantia maintainers
**Governing specs**: [Spec 001](../../specs/001-voice-conversion-engine/spec.md)、
[Spec 002](../../specs/002-voice-training/spec.md)、
[Spec 003](../../specs/003-webui/spec.md)、
[Spec 004](../../specs/004-source-separation/spec.md)、
[重构章程](../REFACTORING_GUIDE.md)

R0 可复现证据见 [重构基线](0001-baseline.md)。

## Context and decision drivers

当前 `voxmutatio` 由一个递归 source glob 构成，core、I/O、CUDA 算子、推理、训练、分离和
Web stub 被打进同一 static library；所有 CLI 又链接这个完整目标。目录大体表达了领域，
但构建图不能阻止反向依赖，也不能让局部改动只编译和测试实际消费者。

应用层的边界更集中：`tools/vc_serve.cpp` 同时负责 HTTP 解析、鉴权、资源上限、路由、文件
事务、GPU 作业状态、线程和子进程命令；各 CLI 也各自承担参数解析、文件发现和业务编排。
本轮已经建立三个可复用的风险基线：原子 WAV 发布、真实 CLI 进程契约、真实 HTTP socket
契约。它们使后续拆分可以先证明行为保持，而不是依靠文件搬迁后的人工判断。

必须保持的外部行为包括：公共头与 `voxmutatio` CMake target、CLI 名/flags/退出码、HTTP
路由/状态码/默认开放策略、WAV/safetensors/Index 格式、CUDA FP32 支持矩阵、数值门槛、
CTest 名/label/asset gate，以及现有 staging/rollback 和 GPU 串行语义。

## Decision

采用“兼容 facade + 目录对齐内部组件 + 可注入应用适配器”的渐进式结构：

1. 用显式 source list 取代递归 glob。先按现有目录建立内部 OBJECT target：`core`、`io`、
   `cuda_ops`、`content`、`f0`、`index`、`synthesizer`、`separation`、`autograd`、`training`、
   `pipeline` 和 legacy WebUI public surface。`voxmutatio` 继续作为 static compatibility
   facade 聚合这 12 个对象，保持目标名与单一 archive 交付；本 ADR 不新增公共库 API。
2. 内部依赖只允许遵循
   [`REFACTORING_GUIDE.md`](../REFACTORING_GUIDE.md#2-允许的依赖方向) 的方向。组件 public
   include、compile definition 与 CUDA library 依赖在各自 target 上声明，不再由全局
   `include_directories` 或最终可执行文件偶然提供。
3. 建立仅供 `tools/` 使用的 host-side app-support 层；共享严格 value parser 保持 host-only，
   四个主要命令分别由独立 `resonantia_vc_*_app_support` target 承载 Options/parser/runner，
   避免一个命令的修改触发其余 runner 重编。它们不进入公共 `include/voxmutatio`。
4. 将 Web 编排按四项职责拆分：HTTP transport/resource limits/auth、route/workflow service、
   filesystem/process adapters，以及 service-owned GPU gate/job state。实现映射为
   `http_transport.*`、`web_service.*`、`workflow_support.*` 三组模块；`vc_serve` 的 `main`
   只解析启动配置并装配这些对象，禁止新增匿名全局状态。
5. CLI 逐个迁移为“纯参数到 config 的解析 + 可测试 runner + 薄 main”。迁移期间保留原
   argv、输出路径、退出码和日志中的脚本依赖字段；不借重构新增 flags 或扩大输入范围。
6. public model/pipeline 类型继续负责模型级校验与 CUDA 所有权；app layer 只做语法、路径、
   事务与编排校验，不复制 tensor shape 或数值算法。

## Alternatives considered

- **一次性重写目录、API、构建和 Web**：回归面跨越数值、资源和持久数据，无法给出小步
  回滚点，拒绝。
- **只把大文件拆成更多 `.cpp`，保持单一隐式构建图**：能改善可读性但不能约束依赖或资源
  所有权，不能解决主要驱动因素，拒绝。
- **立即拆成多个进程/服务**：会改变部署、认证、故障和延迟契约，超出行为保持型重构，
  拒绝；若未来需要，另立 feature spec/ADR。
- **立即发布多个公共库**：会过早冻结 ABI/安装布局。先保留 `voxmutatio` facade，等内部
  边界稳定且发现真实外部消费者后再单独决策。

## Consequences

正面影响：显式 target 与源码清单让反向依赖更容易审计；Web/CLI 错误和事务可在无模型资产时测试；GPU、线程、
文件与子进程所有权会从全局约定变成对象契约；组件迁移可以逐步编译和回滚。

成本与风险：当前所有组件仍可见同一公共 include tree，构建图本身不会机械阻止所有反向
include/call；需要 code review 或后续 dependency lint 才能强制该方向。OBJECT target 的
CUDA/C++ usage requirements 需要精确维护；拆分过程中可能
暴露之前由全局 include/link 顺序掩盖的依赖；compatibility facade 会在一段时间内保留
重复的“内部组件 + 外部聚合”概念。不得用删测试、放宽数值容差或改变错误码来消除这些
迁移问题。

## Implementation outcome

R0–R5 已于 2026-08-10 完成：运行时使用 12 个显式 OBJECT 组件，公共
`voxmutatio` 仍交付单一 static facade；四个 CLI runner 各自使用独立内部 support target；
Web transport、workflow adapter 与 service 已拆分，job state、GPU gate 和后台 worker 生命周期
归 `WebService` 实例所有。旧递归 source glob、匿名全局 Web job state、内联 CLI runner 与
旧 parser 转发路径均已移除。

本 ADR 完成时的 current-tree 构建在 `/tmp/resonantia-final.udNc8h` 记录为 CTest 27/27 通过
（9 个 contract、18 个 asset-gated integration、0 失败、0 skip，333.39 秒）。后续加入
2026-08-10 加入 Spec 006 R0 门禁后的同目录历史快照为 29/29（10 contract + 19
integration、0 失败，real 331.28 秒）；该后续记录不改写原 27 项完成基线。当前结果见
`specs/PROJECT_STATE.md`。SDD contract 与
`git diff --check` 结果见最终项目状态。最终审查还修复了 R0 已存在的规范漂移：畸形双符号
数值现在被严格拒绝、全部输入不可读的 `vc_preprocess` 返回非零、`vc_batch --help` 的
filter-radius 默认值与公共配置一致。这些是显式契约修复，不计作结构重构的行为保持结论。

本轮在包含维护者既有修改的 dirty worktree 中按阶段构建和验证，未执行 Git 写操作，也未创建
ADR 所建议的阶段性 commits；回滚证据由 R0 基线、独立构建目录与逐阶段门禁保留。后续提交
或分支整理仍须由维护者另行授权。

## Migration and rollback

本 ADR 已由维护者批准，按以下顺序实施：

1. **R0 冻结基线**：保存 current-tree fresh configure/build、9 项默认 contract、18 项资产
   integration 的注册与执行记录；记录模型/fixture 清单和工作树基准。
2. **R1 显式构建图**：先把 glob 改成显式 source list，再引入内部 OBJECT targets；不移动
   文件、不改调用方。`voxmutatio` archive 的符号和所有现有 target 必须仍可链接。
3. **R2 叶子组件**：按 core → io/CUDA ops → content/f0/index/synthesizer/separation/
   autograd 的顺序声明所有权和依赖，每一步运行对应 contract/数值门禁。
4. **R3 上层组件**：迁移 pipeline 与 training；保持初始化事务、模型 rollback、长音频和
   导出格式，运行相关真实模型门禁。
5. **R4 应用 seam**：提取 `app_support`、CLI config/runner，再拆 Web transport、router、
   workflow 与 adapters。先让旧 main 调用新 seam，对拍后才删除旧内联路径。
6. **R5 切换与清理**：搜索旧 source glob、全局 job state、重复 parser/transaction helper
   和旧命令构造；完整门禁通过后单独删除兼容代码并同步 SDD。

每阶段独立提交。任一阶段出现 CTest 注册减少、CLI/HTTP 契约变化、数值门槛失败、资源上界
回退或旧格式无法加载时，回滚该阶段到上一个仍通过的 facade；不迁移或覆盖用户模型、音频
与 run 目录，因此不需要数据回滚。

## Verification and SDD impact

- CMake：fresh out-of-tree default/model configure，审计 target/link graph 与 `ctest -N`；
  9 个 contract 和资产齐全时 18 个 integration 不得减少或静默 skip。
- Host contracts：`test_cli_contract`、`test_web_contract`、`test_audio_io` 必须持续门控严格
  解析、404/405/auth/busy、事务回滚和原子产物。
- 数值与模型：按 [追踪矩阵](../../specs/TRACEABILITY.md) 执行 HuBERT、F0、VITS、pipeline、
  training 和 separation 的真实资产门禁；不修改 reference 或容差来适配结构迁移。
- 每阶段同步 `docs/DESIGN.md`、`specs/TRACEABILITY.md` 与受影响 plan；如果发现必须改变
  public API、CLI、HTTP、格式或部署边界，暂停本 ADR，先走功能规格和兼容迁移评审。
