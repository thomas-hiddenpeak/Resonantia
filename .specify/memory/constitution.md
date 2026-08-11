# Resonantia Constitution

> *Vox mutatur, mens manet.* (声音变化，意识不变。)

本宪章定义项目不可被普通实现便利性覆盖的工程与治理约束。功能 Spec 定义
“应当实现什么”，当前代码、公共接口、构建配置与实际测试结果证明“现在实现了
什么”；两者冲突时必须记录差距，不得把文档声明当作运行证据。

## Core Principles

### I. Module Boundaries

`core`、`content`、`f0`、`index`、`synthesizer`、`training`、`separation` 等领域
必须保持清晰的公共接口、私有实现和单向依赖。每个模块必须能够被独立测试和替换，
但不要求每个源码目录都对应独立的 CMake target；target 划分由构建和链接需求决定。

### II. Stable User Interfaces

用户可见工作流必须通过稳定的库接口暴露，并在适用时提供 CLI 烟雾入口。公共配置、
错误语义和已支持的输入必须显式验证；不得静默接受未实现的选项或改变既有契约。

### III. Test-First (NON-NEGOTIABLE)

新增行为和缺陷修复必须先建立能够失败的测试或可复现证据，再实现最小修复。CUDA
数值代码必须有参考结果、边界测试和与风险相称的集成验证。若客观上无法先写自动化
测试，变更记录必须解释原因并提供等价的可复现验证。

### IV. safetensors-Only Runtime

运行时权重格式锁定为 safetensors，并在 mmap 后验证 dtype、shape、byte range、有限值
及模型不变量。运行时禁止加载 pickle 或 PyTorch `.pth`；这类格式只允许由 `tools/`
中的一次性转换工具处理。大型权重和索引不得无必要地完整复制到主机或设备内存。

### V. Runtime Purity

推理与训练运行时使用 C++20/CUDA，不依赖 Python/PyTorch、FAISS、ONNX Runtime 或
TensorRT。允许使用项目明确采用的 CUDA 基础库和小型、可审计的系统依赖；引入新的
大型运行时必须由架构决策记录说明动机、成本和迁移方案。

### VI. Numerical Alignment

与参考实现等价的模块必须逐样本验证，默认门限为：

- 内容特征：L2 或规范化误差 `< 1e-4`
- F0 序列：绝对误差 `< 0.5 Hz`
- 合成音频：相关性 `> 0.999`

若某个 Spec 定义了更适合该模型的指标或更严格门限，以该 Spec 为准。门限、数据集、
命令、硬件和结果必须一同记录；不能用只检查 shape 的测试代替数值对齐。

### VII. Simplicity and No Speculation

优先完成已验证需求，避免为假设中的后端、模型或扩展点预建框架。第三次出现且语义
稳定的重复通常才值得抽取公共组件；安全性或正确性所需的公共边界不受此经验规则
限制。Web 界面保持聚焦，不复制无关的历史功能。

### VIII. Explicit Ownership and RAII

反对投机性抽象不等于反对资源抽象。文件、mmap、主机/设备内存、CUDA stream/event、
cuBLAS/cuFFT handle、锁和临时文件必须具有明确所有者，并优先使用 RAII 保证所有退出
路径释放资源。公共包装应保持薄、可审计，表达所有权、生命周期、同步和错误边界；
禁止以“直接调用原语”为由重复裸资源管理或牺牲异常安全。

### IX. Layered Integration Testing

端到端 Pipeline 是跨模块行为的最高级门禁，但不能替代快速的单元、契约和数值测试。
测试分层为：默认可运行的无资产契约测试，以及需要真实模型/fixture 的集成与对齐测试。
缺少资产、跳过或仅编译成功都不得报告为功能通过。

### X. Representative Test Data (NON-NEGOTIABLE)

模型行为、数值对齐和端到端语音功能必须使用具有明确来源的真实模型和真实人声音频。
合成数据只允许用于底层确定性单元测试、smoke test、尺寸/边界/错误路径、DSP 基元和
梯度检查；不得用正弦波、白噪声等替代真实语音来宣称模型或 E2E 功能正确。

### XI. Python Isolation (NON-NEGOTIABLE)

`tools/` 是唯一允许项目 Python 代码的目录，且必须通过 `uv`、`pyproject.toml` 和
`.python-version` 管理环境。`src/`、`include/`、`tests/` 与运行时构建不得引入 Python
或 PyTorch 依赖，CMake 不得通过 `find_package(Python)` 或脚本执行形成隐式运行依赖。

## Engineering Standards

- 语言基线是 C++20 与项目工具链支持的 CUDA C++；公共头文件不得泄露不必要的实现
  细节或形成循环依赖。
- 文件和函数使用 `snake_case`，类型使用 `PascalCase`，成员使用尾随下划线，常量和
  枚举值使用 `kPascalCase`。
- 在公共边界验证空指针、尺寸、整数转换、有限值、dtype/shape/byte range 和兼容性。
  初始化应分阶段完成并原子提交；失败后对象必须保持可析构、可重试的定义状态。
- GPU/CPU 内存和工作区必须有可说明的上界。检查 CUDA API、kernel launch 和异步同步
  错误，禁止依赖未定义生命周期或未经验证的 host/device 指针。
- `[[nodiscard]]`、`noexcept`、`constexpr` 与 `static_assert` 在能准确表达契约时使用，
  不得为了形式门禁而写出错误的异常或返回值承诺。

## Evidence and Status

- 实现状态只能由当前提交中的代码、公共 API、CMake/CTest 注册，以及在当前树上实际
  执行的构建、测试、基准或推理记录支持。
- 状态记录必须注明验证日期、命令、环境/硬件、资产、通过/失败/跳过数量和未验证项。
  Skip、缺失资产、历史日志和仅打印指标均不等于通过。
- `specs/PROJECT_STATE.md` 和 traceability 文档必须随能力变化更新。历史审计、计划和
  research 是来源记录，不得替代对当前代码与测试的检查。
- `Complete` 要求所有 FR、Acceptance Scenarios、Success Criteria 和 Edge Cases 均有
  可追溯实现及证据；性能或对齐指标未测量时必须保持未完成，而非按推断勾选。

## Governance

### Spec lifecycle

1. 新的用户可见能力或独立模块需要功能 Spec；优先扩展已有 Spec，避免重复契约。
2. 普通重构、性能优化、Bug 修复和测试增强，在不改变用户行为或核心架构时不得新建
   功能 Spec；应更新相关测试、设计、追踪表和变更记录。
3. 改变模块边界、持久化格式、核心数据流、运行时依赖或部署/兼容契约的架构变更，
   必须在实现前提供 ADR，并在相关 Spec 或设计文档中加入迁移章节、回滚策略和追踪项。
4. 新 Spec 编号前必须核对已有 Spec 的状态和范围。允许在彼此独立、依赖已声明时并行
   推进；禁止为了编号顺序虚构完成状态。
5. Spec 只有在需求、验收场景、成功指标、边界、集成运行和必需真实资产均有可复现
   证据后才能标记 `Complete`。

### Authority and conflict handling

- 所有 Core Principles 都是合并门禁；标注 `NON-NEGOTIABLE` 的原则不得由功能 Spec
  豁免。
- 本宪章约束工程与治理；批准的功能 Spec 约束产品行为；`docs/DESIGN.md` 解释当前架构。
  较低层文档不得静默覆盖较高层契约。
- 对“当前事实”的判断以当前公共接口、实现、构建配置和实际测试证据为准。它们若不
  满足宪章或 Spec，结论是存在待修复偏差，而不是自动修改规范或宣称已经合规。
- 改动必须保护用户已有的未提交变更，并按任务范围同步代码、接口、测试、CMake、
  Spec、设计和状态记录。

### Amendments

- 宪章修改需要明确理由、维护者审核、版本升级和受影响文档的迁移检查。
- 版本采用治理 SemVer：MAJOR 表示不兼容的治理变化，MINOR 表示新增或实质扩展原则，
  PATCH 表示不改变要求的澄清。
- 每次合并前必须检查适用原则与验证证据；无法执行的门禁必须明确记录，不能隐式略过。

**Version**: 1.4.0 | **Ratified**: 2026-08-07 | **Last Amended**: 2026-08-09
