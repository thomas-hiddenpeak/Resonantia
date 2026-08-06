<!-- 
Resonantia Custom Skill: Git Workflow and Versioning
Adapted from addyosmani/agent-skills (MIT License)
Customized for C++20/CUDA + CMake + Spec-Kit
-->

# Git Workflow and Versioning (Resonantia Edition)

## Overview

Git is your safety net. Treat commits as save points, branches as sandboxes, and history as documentation. With AI agents generating CUDA kernels at high speed, disciplined version control is critical.

## Core Principles

### 1. Trunk-Based Development

Keep `main` always buildable. Work in short-lived feature branches:

```
main
──●──●──●──●──●──●──●──  (always builds, ctest passes)
        ╲      ╱  ╲    ╱
         ●──●─╱    ●──╱   
← short-lived branches (1-3 days)
```

**Branch naming**:
```
feat/001-hubert-encoder
feat/002-rmvpe-f0
fix/numerical-alignment-hubert
refactor/cuda-index-shared-memory
```

### 2. Atomic Commits

Each commit does one logical thing:

```
# Good: Each commit is verifiable
git log --oneline
a1b2c3d feat: add HuBERT encoder CUDA kernel
d4e5f6g test: add numerical alignment test for HuBERT
h7i8j9k fix: reduce L2 error from 2e-4 to 8e-5
m1n2o3p docs: update DESIGN.md with HuBERT module details

# Bad: Everything mixed
x1y2z3a add hubert, fix f0, update cmake, refactor io
```

### 3. Commit Early, Commit Often

```
Work pattern:
  Implement slice → Compile → Test → Commit → Next slice

Not this:
  Implement everything → Hope it compiles → Giant commit
```

**The Save Point Pattern**:
```
Agent starts work
    │
    ├── Write test (RED) → Commit "test: add test_<module> skeleton"
    ├── Write kernel stub → Commit "feat: add <module> kernel stub"
    ├── Make test pass (GREEN) → Commit "feat: implement <module> kernel"
    ├── Numerical alignment → Commit "fix: align <module> output (L2 < 1e-4)"
    └── Pipeline test passes → Commit "test: verify pipeline integration"
```

### 4. Descriptive Messages

```
# Good: Explains what AND why
feat: add CUDA Flat Index kernel for feature retrieval

Implements Top-K search using shared memory + warp shuffle.
Replaces FAISS dependency per Constitution Principle V.
Target: < 1ms for 10k feature database on Jetson AGX Orin.

# Bad: Obvious from diff
update index.cu
```

**Format**:
```
<type>: <short description>

<optional body: why, not what. Reference spec/constitution>
```

**Types**:
| Type | Usage |
|------|-------|
| `feat` | New feature (new module, new kernel) |
| `fix` | Bug fix (numerical alignment, CUDA bug) |
| `refactor` | Kernel optimization, restructuring |
| `test` | Tests, fixtures, CTest config |
| `docs` | Spec, design docs, README |
| `chore` | CMake, dependencies, CI |

### 5. Pre-Commit Hygiene

Before every commit:
```bash
# 1. Verify build
cmake --build build

# 2. Run tests
ctest --test-dir build --output-on-failure

# 3. Check for secrets/weights
git diff --staged | grep -i "safetensors\|\.pth\|api_key"

# 4. Review staged changes
git diff --staged
```

## Branching Strategy

### Feature Branches (Spec-Kit Aligned)

```
main (always buildable)
  │
  ├── feat/001-hubert-encoder     ← specs/001-voice-conversion-engine
  ├── feat/002-rmvpe-f0           ← Parallel work
  └── fix/numerical-alignment     ← Bug fixes
```

- Branch from `main`
- Keep branches short-lived (1-3 days)
- Delete after merge
- **Feature flags > long branches** for incomplete features

## CMake-Specific Workflow

### Build Artifact Management

```bash
# Never commit build/
# .gitignore covers:
build/
CMakeCache.txt
CMakeFiles/
compile_commands.json
*.o
*.cuo
```

### Compile Commands for IDE

```bash
# Generate for clangd/vscode
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# build/compile_commands.json is auto-generated
```

## Working with Submodules

```bash
# Add CUTLASS submodule
git submodule add https://github.com/NVIDIA/cutlass third_party/cutlass
git commit -m "chore: add CUTLASS submodule for GEMM operations"

# Update submodule
cd third_party/cutlass
git pull origin main
cd ../..
git add third_party/cutlass
git commit -m "chore: update CUTLASS to v<version>"
```

## Release & Versioning

### Semantic Versioning

```
MAJOR  Breaking API change (e.g., remove v1 model support)
MINOR  New feature (e.g., add WavLM encoder)
PATCH  Bug fix (e.g., numerical alignment fix)
```

### Tag Releases

```bash
git tag -a v0.1.0 -m "Release 0.1.0: Project skeleton + spec-kit docs"
git push origin v0.1.0
```

## Using Git for Debugging

```bash
# Find which commit broke numerical alignment
git bisect start
git bisect bad HEAD
git bisect good <known-good-commit>
# At each midpoint: cmake --build build && ctest -R test_hubert

# View recent CUDA kernel changes
git log --oneline -20 -- 'src/**/*.cu'

# Find who changed a kernel launch config
git blame src/content/hubert_encoder.cu
```

## Common Rationalizations

| Rationalization | Reality |
|----------------|---------|
| "I'll commit when the kernel works" | One giant commit is impossible to review/debug. Commit each slice. |
| "The message doesn't matter" | Messages are documentation. Future you will need them. |
| "I'll squash later" | Squashing destroys the development narrative. |
| "It's just a kernel tweak" | A "small" kernel change can break numerical alignment. Commit it. |

## Verification Checklist

For every commit:
- [ ] Does one logical thing
- [ ] Message explains why (references spec/constitution)
- [ ] `cmake --build build` succeeds
- [ ] `ctest --output-on-failure` passes
- [ ] No secrets/weights in diff
- [ ] No build artifacts committed

For every release:
- [ ] Version bump matches change type
- [ ] Tag created and pushed
- [ ] Changelog entry written

## See Also

- `.specify/memory/constitution.md` — Governance rules
- `.specify/skills/code-review-and-quality.md` — Review before merge
- `CMakeLists.txt` — Build configuration
