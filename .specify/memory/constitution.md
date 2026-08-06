# Resonantia Constitution

> *Vox mutatur, mens manet.* (声音变化，意识不变。)

## Core Principles

### I. Library-First
每个功能模块从独立库开始。`src/` 下的每个子目录（`core/`, `content/`, `f0/`, `synthesizer/` 等）都是独立的编译单元，通过清晰的 public header 暴露接口。

### II. CLI Interface
每个核心功能通过 CLI 工具暴露。`vc_convert`, `vc_batch`, `vc_probe` 等工具是功能的"烟雾测试"入口。

### III. Test-First (NON-NEGOTIABLE)
每个模块在实现前必须有对应的测试骨架。数值对齐测试（与 Python 原始实现对比）是核心质量门控。

### IV. safetensors-Only
权重格式锁定 safetensors。零拷贝 mmap 加载，无 pickle 安全风险。PyTorch `.pth` 格式仅通过一次性 Python 转换工具支持。

### V. Zero-Dependency Purity
推理链路完全脱离 Python/PyTorch 运行时。禁止引入 FAISS、ONNX Runtime 等大型第三方推理库。手写 CUDA Kernel 优先于链接外部库。

### VI. Numerical Alignment
每个模块的输出必须与 Python 原始实现逐样本对齐：
- 特征向量: L2 < 1e-4
- F0 序列: 绝对误差 < 0.5 Hz
- 合成音频: SRCC > 0.999

### VII. Simplicity
简单优先，YAGNI (You Ain't Gonna Need It)。不预先实现不确定的功能。WebUI 设计简洁现代，不照搬 RVC-WebUI 的复杂界面。

### VIII. Anti-Abstraction
直接使用 cuBLAS/CUDA 原语，不包装抽象层。避免"过早抽象"——只有在第三个重复模式出现时才提取公共代码。

### IX. Integration-First Testing
优先端到端集成测试（完整 Pipeline），而非孤立的单元测试。`test_pipeline.cpp` 是最高优先级的测试。

### X. Real Audio Only (NON-NEGOTIABLE)
**所有测试必须使用真实音频数据，禁止使用合成数据（正弦波、白噪声等）进行功能验证。**
- ✅ 必须使用真实人声录音进行测试
- ❌ 禁止使用正弦波、多音信号、白噪声等合成音频替代真实人声
- ❌ 合成数据仅可用于底层工具函数的 smoke test（如 resample 尺寸检查）
- 原因：语音转换的特征分布（HuBERT 特征、F0 轮廓、能量包络）与合成信号完全不同
  - 正弦波的 HuBERT 特征无意义
  - 合成信号无法测试 F0 提取器的真实表现
  - 无法验证 index search 在真实特征空间中的效果

### XI. Python Isolation (NON-NEGOTIABLE)
**`tools/` 是唯一允许使用 Python 的目录**。所有 Python 脚本必须通过 `uv` 管理虚拟环境，确保系统环境干净：
- 权重转换工具 (`tools/convert_to_safetensors.py`) — 一次性使用
- 数据预处理工具 — 训练前准备
- **禁止**在 `src/`, `tests/`, `include/`, `CMakeLists.txt` 中引入任何 Python 依赖
- **禁止**在 CMake 中使用 `find_package(Python)` 或执行 Python 脚本
- `tools/` 目录下必须包含 `pyproject.toml` 和 `.python-version` 定义 uv 环境

## Coding Standards

### Language
- **C++20** / **CUDA 20**
- **Google C++ Style Guide** 为基础
- **宪法治理**: `static_assert` / `constexpr` 编译时验证

### Naming
| Element | Convention | Example |
|---------|-----------|---------|
| Files | `snake_case` | `hubert_encoder.h` |
| Namespaces | `snake_case` 单行嵌套 | `namespace voxmutatio::content {` |
| Types | `PascalCase` | `HubertEncoder` |
| Functions | `snake_case` | `extract_features()` |
| Members | `trailing_` | `output_dim_` |
| Constants | `kPascalCase` | `kHubertHiddenDim` |
| Enums | `enum class` + `kPascalCase` | `ModelVersion::kV1` |

### Quality Gates
- `[[nodiscard]]` on all value-returning functions
- `noexcept` on non-throwing functions
- `constexpr` for compile-time computation
- `static_assert` for invariant validation

## Governance

### Spec Creation Rules (NON-NEGOTIABLE)

**Specs 是项目的核心契约，不得随意创建。**

1. **Sequential Completion**: 已创建的 Spec 必须标记为完成后，才能创建下一个 Spec。
   - `specs/###-feature-name/spec.md` 的 Status 必须从 `Draft` → `Complete`
   - 每个 FR (Functional Requirement) 必须有对应的实现和测试

2. **Necessity Audit**: 计划创建新 Spec 前，必须审计：
   - [ ] 当前 Spec 是否已完成？
   - [ ] 计划工作是否已被现有 Spec 覆盖？
   - [ ] 是否属于升级/改善（而非新增功能）？

3. **No Spec for Refactoring/Performance**: 在不改变核心意图的情况下，
   **不得**因为以下原因新增 Spec：
   - ❌ 性能提升（kernel optimization → commit message 说明即可）
   - ❌ 代码重构（refactor → commit message 说明即可）
   - ❌ Bug 修复（fix → commit message 说明即可）
   - ❌ 测试覆盖改进（test → commit message 说明即可）

4. **Valid Reasons for New Spec**:
   - ✅ 新增用户可见功能（如：新增 WebUI）
   - ✅ 新增模块（如：新增 WavLM 编码器）
   - ✅ 改变核心架构（如：从 FP32 迁移到 FP16-only）

5. **Spec Upgrade vs. New Spec**:
   - 改善/扩展现有 Spec 的功能 → 更新现有 `spec.md`，version +1
   - 完全独立的新功能 → 创建新 Spec `specs/###-new-feature/`

6. **Spec Completion Criteria (NON-NEGOTIABLE)**:
   
   Spec 标记为 `Complete` 必须满足以下**所有**条件：
   
   - [ ] **所有 FR (Functional Requirements)** 已有对应的实现代码
   - [ ] **所有 Acceptance Scenarios** 已有对应的测试用例
   - [ ] **所有测试用例均已通过** (`ctest` 或等价工具)
   - [ ] **所有 Success Criteria** 已验证达标（数值对齐/性能指标）
   - [ ] **Edge Cases** 已处理并有测试覆盖
   - [ ] **集成测试通过** (`test_pipeline.cpp` 端到端验证)
   - [ ] **实际运行验证**: 至少一次完整的端到端推理已成功执行
   - [ ] **模型权重**: 所有必需的 safetensors 权重已加载并验证
   
   **禁止行为**：
   - ❌ 仅有骨架代码就标记 Complete
   - ❌ 测试未运行就标记 Complete
   - ❌ 测试有失败就标记 Complete
   - ❌ 数值对齐未验证就标记 Complete
   - ❌ **没有实际运行过推理就标记 Complete**
   - ❌ **没有加载过真实模型权重就标记 Complete**
   - ❌ **仅靠单元测试通过就标记 Complete**（必须包含集成测试）
   
   **Completion Checklist 模板**：
   ```
   ## Completion Checklist
   - [x] FR-001: Implementation + test passed
   - [x] FR-002: Implementation + test passed
   - ...
   - [x] SC-001: L2 error < 1e-4 verified
   - [x] SC-002: F0 error < 0.5 Hz verified
   - ...
   - [x] Edge cases: all handled
   - [x] Integration test: passed
   - [x] Actual inference: executed successfully with real models
   ```

### Authority

- **Authority**: Principles I-VI 和 X 是强制门控。违反原则的代码不得合并。
- **Spec Violation**: 未经审计创建 Spec 视为 CRITICAL 违规。

### Amendments

- 修改需要 PR + 理由 + 维护者审批 + 版本升级
- Versioning policy (SemVer for governance):
  - MAJOR = 不兼容的治理变更
  - MINOR = 新原则/章节
  - PATCH = 澄清和非语义修改
- Compliance review: 每个 PR 必须验证合规性

**Version**: 1.3.0 | **Ratified**: 2026-08-07 | **Last Amended**: 2026-08-07
