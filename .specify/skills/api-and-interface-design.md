<!-- 
Resonantia Custom Skill: API and Interface Design
Adapted from addyosmani/agent-skills (MIT License)
Customized for C++20 Module Boundaries + CUDA Kernels
-->

# API and Interface Design (Resonantia Edition)

## Overview

Design stable, well-documented C++ interfaces that are hard to misuse. Good interfaces make the right thing easy and the wrong thing hard. This applies to module public headers, CUDA kernel launch APIs, and configuration structs.

**Constitution Principle VIII**: Anti-Abstraction. Prefer simple, direct interfaces over clever abstractions.

## When to Use

- Designing new module public headers (`include/voxmutatio/<module>/`)
- Defining CUDA kernel launch signatures
- Creating configuration structs
- Changing existing public APIs

## Core Principles

### 1. Contract First (Header First)

Define the public header before implementing. The header is the contract — implementation follows.

```cpp
// include/voxmutatio/content/hubert_encoder.h (contract)
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace voxmutatio::content {

struct HubertConfig {
    std::string model_path;
    int output_dim = 256;
    bool half_precision = false;
};

class HubertEncoder {
public:
    bool init(const HubertConfig& config);
    std::vector<float> extract(const float* audio, int num_samples);
    [[nodiscard]] int output_dim() const noexcept { return output_dim_; }

private:
    int output_dim_ = 256;
};

}  // namespace voxmutatio::content
```

**Implementation** (`.cu` file) is private to the module.

### 2. Consistent Error Semantics

Pick one error strategy and use it everywhere:

```cpp
// Pattern 1: bool return for init/load operations
bool init(const HubertConfig& config);  // false = failure

// Pattern 2: std::optional for fallible queries
std::optional<std::string> Device::init(...);  // nullopt = success, string = error

// Pattern 3: ErrorCode enum for detailed errors
enum class ErrorCode : std::uint8_t {
    kSuccess,
    kInvalidInput,
    kModelLoadFailed,
    // ...
};
```

**Don't mix patterns.** If `init()` returns `bool`, don't have `load()` return `std::optional`.

### 3. Validate at Boundaries

Trust internal code. Validate at public API boundaries:

```cpp
// Public API (validate)
std::vector<float> HubertEncoder::extract(const float* audio, int num_samples) {
    if (!audio || num_samples <= 0) {
        throw std::invalid_argument("audio must be non-null and non-empty");
    }
    // ... implementation ...
}

// Internal helper (trust)
__device__ float hubert_activation(float x) {
    return x * (1.0f + std::log(std::exp(x)));  // GELU approximation
    // No validation needed — called from kernel with validated input
}
```

### 4. Prefer Addition Over Modification

Extend interfaces without breaking existing consumers:

```cpp
// Good: Add optional config fields
struct HubertConfig {
    std::string model_path;
    int output_dim = 256;
    bool half_precision = false;
    int num_layers = 12;  // Added later, has default
};

// Bad: Remove fields or change types
struct HubertConfig {
    std::string model_path;
    // int output_dim = 256;  // Removed — breaks existing code
};
```

### 5. Predictable Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Files | `snake_case` | `hubert_encoder.h` |
| Namespaces | `snake_case` | `voxmutatio::content` |
| Types | `PascalCase` | `HubertEncoder` |
| Functions | `snake_case` | `extract()` |
| Members | `trailing_` | `output_dim_` |
| Constants | `kPascalCase` | `kHubertHiddenDim` |
| Enums | `kPascalCase` | `ModelVersion::kV1` |

## CUDA Kernel API Patterns

### Kernel Launch Wrapper

Hide launch configuration from callers:

```cpp
// Public API (simple)
std::vector<float> extract(const float* audio, int num_samples);

// Internal wrapper (handles launch config)
__global__ void hubert_forward_kernel(
    const float* audio, float* output,
    int num_samples, int output_dim,
    const float* weights  // captured in weights_
);

// Implementation
std::vector<float> HubertEncoder::extract(const float* audio, int num_samples) {
    int block_size = 256;
    int grid_size = (num_samples + block_size - 1) / block_size;
    hubert_forward_kernel<<<grid_size, block_size>>>(...);
    cudaDeviceSynchronize();
    // ...
}
```

### Configuration Structs

Use structs for kernel parameters > 3 arguments:

```cpp
// Good: Config struct
struct KernelConfig {
    int block_size = 256;
    int shared_memory_bytes = 48 * 1024;
    bool use_half_precision = false;
};

void launch_kernel(const KernelConfig& cfg, ...);

// Bad: Too many parameters
void launch_kernel(int block_size, int shared_mem, bool half_prec, int foo, float bar, ...);
```

## Header Organization

### Public Headers (`include/`)

```
include/voxmutatio/content/
├── hubert_encoder.h      # Public API only
└── wavlm_encoder.h       # Public API only
```

**Rules**:
- No `#include` of `src/` internal headers
- No implementation details (GPU pointers, kernel configs)
- Only what external callers need

### Internal Headers (`src/`)

```
src/content/
├── hubert_encoder.cu     # Implementation
├── hubert_encoder.h      # Internal forward declarations
└── hubert_common.cuh     # Shared CUDA utilities (private)
```

**Rules**:
- `.cuh` files are private to the module
- Can include anything
- Not exposed in `include/`

## Hyrum's Law for C++ APIs

> With a sufficient number of users of an API, all observable behaviors will be depended on.

**Implications**:
- Every public function signature is a commitment
- Error message text is observable (don't change casually)
- Return value ordering is observable
- Default parameter values are observable

**Mitigation**:
- Mark deprecated APIs with `[[deprecated("use X instead")]]`
- Keep backward-compatible wrappers during deprecation period
- Document breaking changes in commit messages

## Common Rationalizations

| Rationalization | Reality |
|----------------|---------|
| "We'll design the API later" | The header IS the design. Write it first. |
| "Internal APIs don't need contracts" | Internal consumers are still consumers. |
| "CUDA kernels are implementation details" | Kernel launch signatures are APIs. Design them well. |
| "We can always refactor the header" | Header changes propagate to all consumers. |

## Verification Checklist

- [ ] Public header compiles independently
- [ ] No `src/` headers leaked to `include/`
- [ ] Naming conventions followed (Google C++ Style)
- [ ] Error semantics consistent with existing APIs
- [ ] New fields have default values (backward compatible)
- [ ] `[[nodiscard]]` on value-returning functions
- [ ] `noexcept` on non-throwing functions
- [ ] `static_assert` for compile-time invariants

## See Also

- `.specify/memory/constitution.md` — Principle VIII (Anti-Abstraction)
- `.clang-format` — Auto-formatting rules
- `docs/DESIGN.md` — Module decomposition
