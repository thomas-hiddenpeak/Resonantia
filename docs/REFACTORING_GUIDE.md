# Resonantia 行为保持型重构章程

本章程适用于“改善结构但不改变外部行为”的变更。需求导航与证据层级见
[specs/README.md](../specs/README.md)，逐项门禁见
[specs/TRACEABILITY.md](../specs/TRACEABILITY.md)。若变更需要修改用户可观察行为，先更新
或新增 spec；不能继续以“重构”名义绕过需求评审。

## 1. 目标与边界

允许的目标包括：缩短依赖链、消除重复、收拢资源所有权、引入内部 seam、改善命名和
可测试性、把隐式错误状态改成统一的内部处理。重构成功的定义是外部契约和数值门禁不变，
而不是文件数量减少或抽象层增加。

默认不得改变：

- 公共头中的类型、函数签名、默认值、返回形状和所有权语义；
- CLI 名称、flags、默认值、退出码以及脚本依赖的输出文件布局；
- safetensors、WAV、Index 和 sidecar 文件格式；
- HTTP 路由、方法、鉴权范围、状态码、响应结构和资源上限；
- 模型支持矩阵、拒绝的配置、数值容差和长音频拼接语义；
- CMake options、target 名、CTest 名/label/asset gate；
- Python 只用于离线工具、运行时 C++/CUDA 的边界。

如果其中任何一项必须改变，变更类型是迁移或功能修改：需要 spec、兼容策略、弃用窗口和
回滚说明。涉及架构决策时，先按 [`adr/README.md`](adr/README.md) 创建 `Proposed` ADR，
经维护者批准后再实施。

## 2. 允许的依赖方向

目标方向如下，箭头表示“可以依赖”：

```text
tools / Web orchestration -> public library APIs
pipeline                  -> content + f0 + index + synthesizer + io + core
training                  -> autograd + synthesizer + separation::Stft + io + core
content/f0/index/
synthesizer/separation    -> io + core + CUDA libraries
autograd                  -> core + CUDA libraries
io                        -> core / standard library
core                      -> CUDA runtime / standard library
```

约束：

- `core`、`io` 和算子层不得反向包含 pipeline、training、Web 或 tools。
- pipeline 不拥有训练策略；Web 不直接复制模型算法，而是调用稳定库或 CLI 契约。
- 模型模块不读取 Web session、CLI 全局变量或产品目录布局。
- 共享代码只有在两个已存在调用方具有相同契约时才下沉；不要为假想复用建通用框架。
- 避免新的静态全局可变状态。GPU context、scratch、job state 必须有明确并发与生命周期契约。
- 发现现有反向依赖时，先加 characterization test 和 seam，再逐步反转；不要大爆炸搬迁。

## 3. 分阶段流程

### 阶段 0：冻结基线

1. 在 `TRACEABILITY.md` 标出受影响需求、默认/可选门禁和已知缺口。
2. 记录干净构建命令、CTest 注册清单、所用本地资产和提交 ID。
3. 对无自动证据的 CLI、HTTP、文件迁移或错误路径，先补 characterization test。
4. 保存数值参考时记录 dtype、shape、采样率和容差；不新增无环境的性能数字。

### 阶段 1：定义 seam 与所有权

- 写清输入校验发生在哪一层、哪个对象拥有 GPU/文件/socket/线程资源、失败如何传播。
- 新接口先保持内部可见；除非 spec 要求，不扩大 public surface。
- 对初始化采用 staging/commit 或 whole-object reset。`init=false` 后 forward/train 必须不可用，
  且对象不得混用旧参数与新参数。

### 阶段 2：叶子优先迁移

- 先移动无上层依赖的 core/io/算子代码，再迁移模型，最后切 pipeline/tools/Web 调用方。
- 每一步保持可编译、可回滚；旧路径与新路径短期并存时，只允许一个明确选择点。
- CUDA 优化先保留简单参考路径或数值 fixture，再替换实现。不得在同一步同时改算法、布局、
  精度和资源策略。

### 阶段 3：切换与删除旧路径

- 所有调用方切到新 seam 后运行受影响的默认和资产门禁。
- 搜索旧符号、旧 flags、旧文件名和未注册测试，确认没有隐式消费者。
- 删除兼容层前确认没有脚本、Web 子进程命令或导出格式依赖；删除本身单独成提交更易回滚。

### 阶段 4：同步 SDD

- 更新当前支持矩阵、追踪行、实现位置和测试层级。
- plan/research 可保留历史设计，但标明被替代日期；不要重写历史审计来伪造连续性。
- 若门禁仍依赖本地资产，状态只能是“已实现·可选门禁”或“部分”。

## 4. 迁移与回滚

- 优先使用可逆的小提交：新增 seam → 双路径对拍 → 切调用方 → 删除旧路径。
- 文件格式或持久目录迁移必须先写到 staging，校验完成后原子替换；失败恢复旧对象，保留无法
  自动恢复的备份并报告路径。
- 模型权重转换保持源文件只读，输出到新路径；不要覆盖用户唯一副本。
- CLI/API 迁移若不能原地兼容，应提供显式版本或弃用窗口，不静默改变字段含义。
- 回滚条件至少包括：数值容差失败、资源上界回退、错误状态变成成功、旧模型/格式无法加载、
  CTest 注册数量意外减少。
- 回滚不使用删除用户数据、硬重置工作树或依赖未解析环境变量的命令。

## 5. 资源契约

### GPU 与主机内存

- device allocation 由 RAII 对象拥有；分配、拷贝和 kernel/library 错误必须停止当前操作，
  不允许记录日志后继续使用空指针。
- 临时 scratch 必须有上界和释放点。长音频/attention 使用有界 chunk；不能靠“通常输入较短”
  假设安全。
- autograd closure 只捕获 backward 必需状态；大 workspace 应在 backward 重算或由有界池复用。
- 优化器会持有参数强引用；模型回滚必须同时销毁相关 optimizer，避免 GPU Tensor 暗中常驻。
- destructor 不抛异常；无法在析构中恢复的释放错误只记录。显式操作返回失败或抛出带上下文异常。

### 文件、socket 与线程

- 解析前先验证整数溢出、shape/offset/range、Content-Length、part/file 数量和路径边界。
- 上传使用唯一临时目录；发布到 live tree 前完成写入与校验，替换失败必须回滚。
- 静态文件路径 canonicalize 后仍须证明位于 webroot；文件名不进入 shell 或路径前先经过白名单。
- socket 读写处理短读/短写、EINTR、deadline 和连接上限。后台线程异常不得越过线程入口。
- GPU 工作串行策略是当前 Web 行为的一部分；改变并发模型前必须建立资源与竞态测试。

## 6. 错误契约

- **库边界**：无效参数在分配或索引前拒绝。现有 API 使用 `false`、空结果、`VCResult` 或异常
  的方式不同；重构应保持调用方可观察语义，统一方式属于单独的 API 迁移。
- **初始化**：失败后对象处于明确未初始化状态；后续 forward/train 返回既定失败或抛
  `logic_error`，绝不能运行半加载图。
- **CLI**：解析、模型加载、I/O 或运行失败返回非零；不得生成看似成功的最终产物。临时产物
  与最终产物分离。
- **HTTP**：客户端输入错误、鉴权失败、资源超限、busy 和内部失败保持可区分状态；异常在
  connection/thread 边界转成响应并清理 reservation。
- 错误消息不得包含 token、上传内容或不必要的本机秘密路径。

## 7. 测试门禁

每个重构至少运行受影响集合；“编译成功”不是行为保持证据。

| 改动区域 | 必需门禁 |
|---|---|
| core/io/index/content kernels/autograd | 默认 `contract` CTest 中对应目标 |
| pipeline/config/synthesizer | pipeline contract + 相关 v1/v2、VITS、e2e 资产门禁；缺资产时明确报告未执行 |
| long audio | `test_long_audio_chunking`，同时证明 reference 未进入 chunk path |
| training | `test_autograd`；decoder 改动加 `test_generator_trainer`；posterior/flow/discriminator/GAN 加 `test_gan` |
| separation | STFT/spec 门禁；涉及模型时执行对应 UMX/RoFormer/VAD 资产门禁 |
| Web | 在已有 HTTP CTest 前，至少需要可复现 socket/curl characterization；修改鉴权、路径、limit 或 rollback 时应先补自动测试 |
| CMake/CI | fresh out-of-tree configure/build、`ctest -N` 注册审计、默认 contract；模型模式检查缺资产不注册原因 |

数值测试只能在相同输入、shape、dtype 和约定下比较。不得为了让重构通过而无说明放宽阈值、
改 reference、缩短覆盖范围或把失败用例从 CTest 注销。测试被隔离或跳过时，追踪状态必须降级。

## 8. SDD 同步规则

1. 每个行为变更先更新或同提交更新 spec；纯重构至少更新受影响追踪行的实现/测试链接。
2. 新需求分配稳定 ID；拆分/替代时保留旧 ID 和迁移指向，不复用编号。
3. 公共默认值、拒绝范围、CLI/API、格式、CMake option/target/test 名改变时，同步 README、DESIGN
   与 `TRACEABILITY.md`。
4. 只有默认 CI 注册且实际运行的无资产测试可称默认门禁；模型/fixture 用例始终注明可选条件。
5. `docs/ENGINEERING_AUDIT.md` 保持历史快照。plan/research 可以记录历史选择，但不承担当前状态。
6. 合入前检查所有“完成、支持、通过、性能”措辞是否有当前证据。未知或部分必须原样暴露。

## 9. 合入检查清单

- [ ] 外部行为边界已列出，变更确属重构；否则已有 spec/迁移计划。
- [ ] 依赖方向未反转，没有新增上层到叶子层的隐式全局耦合。
- [ ] 初始化、资源释放、并发与错误状态有明确契约。
- [ ] 迁移步骤可逐提交回滚，没有覆盖用户唯一数据。
- [ ] 默认 contract 和受影响可选资产门禁已执行或明确报告缺资产。
- [ ] `ctest -N` 没有意外减少注册测试；skip/未注册未被称为通过。
- [ ] README、DESIGN、spec 与 TRACEABILITY 同步，历史文档仍标为历史。
- [ ] 没有新增未经环境、输入与命令验证的性能或音质声明。
