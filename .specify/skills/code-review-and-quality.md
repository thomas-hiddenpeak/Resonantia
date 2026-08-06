<!-- 
Resonantia Custom Skill: Code Review and Quality
Adapted from addyosmani/agent-skills (MIT License)
Customized for C++20/CUDA + Google Style + Constitutional Governance
-->

# Code Review and Quality (Resonantia Edition)

## Overview

Multi-dimensional code review with quality gates for C++20/CUDA projects. Every change gets reviewed before merge — no exceptions. Review covers **six axes**: correctness, readability, architecture, CUDA safety, numerical alignment, and constitutional compliance.

**Constitution Principle III**: Test-First is NON-NEGOTIABLE. No merge without tests.

## The Six-Axis Review

### 1. Correctness

Does the code do what it claims to do?

- Does it match the spec (`specs/001-voice-conversion-engine/spec.md`)?
- Are edge cases handled (null pointers, empty buffers, zero-length audio)?
- Are CUDA errors checked (`cudaGetLastError()`, `cudaPeekAtLastError()`)?
- Does it pass all CTest tests?
- Are there race conditions in asynchronous CUDA streams?

### 2. Readability & Google Style

Can another engineer understand this code?

- **Naming**: `snake_case` functions, `PascalCase` types, `kPascalCase` constants, `trailing_` members
- **Namespace**: Single-line nested `namespace voxmutatio::content {`
- **No "clever" CUDA tricks**: Clear kernel launches over obscure shared memory hacks
- **Comments**: Explain *why*, not *what*
- **File size**: < 1000 lines per file

### 3. Architecture

Does the change fit the system's design?

- Does it follow the module decomposition in `docs/DESIGN.md`?
- Are public headers (`include/`) clean from implementation details (`src/`)?
- No circular dependencies between modules
- **Constitution Principle VIII**: Anti-Abstraction — prefer direct cuBLAS/CUDA over wrapper layers

### 4. CUDA Safety

Does the code avoid common CUDA pitfalls?

- [ ] Kernel launch bounds checked (`grid_size`, `block_size`)
- [ ] Shared memory usage within SM limits (`cudaDeviceGetAttribute`)
- [ ] No bank conflicts in shared memory access patterns
- [ ] Memory coalescing for global memory accesses
- [ ] `cudaDeviceSynchronize()` before host reads
- [ ] Error checking after every `cudaMalloc`/`cudaMemcpy`/kernel launch
- [ ] No device memory leaks (`cudaMemGetInfo` before/after)

### 5. Numerical Alignment

Does the output match Python reference?

- [ ] L2 error < 1e-4 for feature extractors
- [ ] F0 error < 0.5 Hz for pitch extractors
- [ ] SRCC > 0.999 for synthesizer output
- [ ] Test fixtures committed in `tests/fixtures/`

### 6. Constitutional Compliance

Does the change violate any Constitution principles?

| Principle | Check |
|-----------|-------|
| I. Library-First | Is the module independently compilable? |
| II. CLI Interface | Is the feature exposed via CLI tool? |
| III. Test-First | Are tests written before/during implementation? |
| IV. safetensors-Only | No `.pth` loading in runtime code? |
| V. Zero-Dependency | No FAISS/ONNX Runtime introduced? |
| VI. Numerical Alignment | Alignment tests pass? |
| VII. Simplicity | YAGNI — no over-engineering? |
| VIII. Anti-Abstraction | Direct CUDA/cuBLAS usage? |
| IX. Integration-First | `test_pipeline.cpp` passes? |
| X. Python Isolation | Python only in `tools/` with uv? |

### 7. Spec Governance (CRITICAL)

Does the change comply with Spec Creation Rules?

- [ ] If new `specs/###-feature/` created: was necessity audit performed?
- [ ] If new `specs/###-feature/` created: is previous Spec marked `Complete`?
- [ ] If performance/refactor/fix: is there a new Spec? (should be **NO**)
- [ ] If feature improvement: is existing `spec.md` updated instead of new Spec?

**Red Flag**: New Spec created for performance optimization, refactoring, or bug fix.

## Review Process

### Step 1: Understand the Context
```
- What spec requirement does this implement?
- What is the expected behavior change?
- Which Constitution principles are relevant?
```

### Step 2: Review the Tests First
```
- Do tests exist for the change?
- Do they test numerical alignment (not just shape)?
- Are edge cases covered (short audio, long audio, stereo, different sample rates)?
- Does test_pipeline.cpp still pass?
```

### Step 3: Review CUDA Code
For each `.cu`/`.cuh` file:
```
1. Kernel launch configuration correct?
2. Shared memory usage safe?
3. Memory access patterns coalesced?
4. Error handling complete?
5. Precision mode (FP32/FP16) handled?
```

### Step 4: Categorize Findings

| Prefix | Meaning | Action |
|--------|---------|--------|
| *(none)* | Required | Must fix before merge |
| **Critical:** | Blocks merge | CUDA bug, numerical misalignment, Constitution violation |
| **Nit:** | Optional | Style preference |
| **FYI** | Informational | No action needed |

## Change Sizing

```
~100 lines changed   → Good. Reviewable in one sitting.
~300 lines changed   → Acceptable for single logical change.
~1000 lines changed  → Too large. Split it.
```

**Separate refactoring from feature work.** A kernel optimization and a new feature are two changes.

## Commit Message Convention

```
<type>: <short description>

<optional body explaining why, referencing spec/constitution>
```

**Types**:
- `feat` — New feature (new module, new kernel)
- `fix` — Bug fix (numerical alignment fix, CUDA bug)
- `refactor` — Kernel optimization, code restructuring
- `test` — Adding/updating tests, new fixtures
- `docs` — Documentation, spec updates
- `chore` — Build config, dependencies

**Examples**:
```
feat: add HuBERT encoder CUDA kernel (specs/001/spec.md FR-003)
fix: numerical alignment for F0 extraction (L2 2e-4 → 8e-5)
test: add test_pipeline integration test (Constitution IX)
```

## Red Flags

- CUDA kernel without error checking
- Numerical alignment test missing or skipped
- Constitution violation (e.g., FAISS dependency introduced)
- Public header exposes implementation details (`src/` types)
- No `[[nodiscard]]` on value-returning functions
- No `noexcept` on non-throwing functions
- Missing `static_assert` for compile-time invariants
- Large uncommitted changes (> 1000 lines)
- `test_pipeline.cpp` broken

## Verification Checklist

- [ ] All 6 axes reviewed
- [ ] CTest suite passes: `ctest --output-on-failure`
- [ ] Numerical alignment thresholds met
- [ ] Constitution compliance verified
- [ ] No CUDA memory leaks
- [ ] Both FP32 and FP16 tested (if applicable)
- [ ] Commit messages follow convention
- [ ] Change size < 300 lines (or justified)

## See Also

- `.specify/memory/constitution.md` — All 9 principles
- `.specify/skills/test-driven-development.md` — TDD workflow
- `.specify/skills/git-workflow-and-versioning.md` — Commit conventions
- `specs/001-voice-conversion-engine/spec.md` — Functional requirements
