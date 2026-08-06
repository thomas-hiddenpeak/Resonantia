<!-- 
Resonantia Custom Skill: Test-Driven Development
Adapted from addyosmani/agent-skills (MIT License)
Customized for C++20/CUDA + CTest + Numerical Alignment
-->

# Test-Driven Development (Resonantia Edition)

## Overview

Write a failing test before writing the code that makes it pass. For CUDA kernel development, this means:
1. CPU reference implementation test (RED)
2. CUDA kernel implementation (GREEN)
3. Numerical alignment verification (REFACTOR)

**Constitution Principle III**: Test-First is NON-NEGOTIABLE.

## When to Use

- Implementing any new CUDA kernel
- Adding a new module (content/, f0/, synthesizer/)
- Fixing numerical alignment bugs
- Modifying existing inference logic

## The C++/CUDA TDD Cycle

```
    RED                    GREEN              REFACTOR
 Write CTest that     Write CUDA kernel    Verify numerical
 fails against        that makes it        alignment vs Python
 reference impl  ──→  passes basic test  ──→  reference (L2 < 1e-4)
      │                    │                    │
      ▼                    ▼                    ▼
   Test FAILS          Test PASSES        Max diff < threshold
```

### Step 1: RED — Write a Failing Test

Create test skeleton in `tests/test_<module>.cpp`:

```cpp
// tests/test_hubert.cpp
#include <cassert>
#include <cmath>
#include <vector>
#include "voxmutatio/content/hubert_encoder.h"

void test_hubert_numerical_alignment() {
    // Load reference output from Python (exported as binary)
    std::vector<float> reference = load_reference("fixtures/hubert_output.bin");

    // Run C++ implementation
    voxmutatio::content::HubertEncoder encoder;
    encoder.init(cfg);
    auto output = encoder.extract(audio.data(), audio.size());

    // This FAILS because encoder is not implemented yet
    float max_diff = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(output[i] - reference[i]));
    }
    assert(max_diff < 1e-4f);  // Constitution Principle VI
}
```

Run with CTest:
```bash
cmake --build build --target test_hubert
ctest -R test_hubert --output-on-failure
```

### Step 2: GREEN — Make It Pass (Basic Correctness)

Implement the minimum CUDA kernel to pass shape/basic tests:

```cpp
// src/content/hubert_encoder.cu
__global__ void hubert_forward_kernel(...) {
    // Minimum implementation
}
```

Verify:
```bash
ctest -R test_hubert  # Should pass basic shape test
```

### Step 3: REFACTOR — Numerical Alignment

Tune the kernel until numerical alignment passes:

```cpp
// Verify against Python reference
float l2_norm = 0.0f;
for (size_t i = 0; i < output.size(); ++i) {
    float diff = output[i] - reference[i];
    l2_norm += diff * diff;
}
l2_norm = std::sqrt(l2_norm / static_cast<double>(output.size()));
assert(l2_norm < 1e-4);  // ✅ Constitution Principle VI
```

## Numerical Alignment Thresholds

| Module | Metric | Threshold | Constitution Ref |
|--------|--------|-----------|-----------------|
| HuBERT/WavLM | L2 error | < 1e-4 | VI |
| F0 (RMVPE/FCPE) | Absolute error | < 0.5 Hz | VI |
| Feature Index | Top-K match | 100% | VI |
| Synthesizer | SRCC | > 0.999 | VI |

## CUDA-Specific Testing Patterns

### Device Memory Leak Detection
```cpp
size_t free_before, total_before, free_after, total_after;
cudaMemGetInfo(&free_before, &total_before);
// ... kernel execution ...
cudaMemGetInfo(&free_after, &total_after);
assert(free_before == free_after);  // No leaks
```

### Kernel Launch Configuration Tests
```cpp
// Test that kernel works for various block/grid sizes
for (int block_size : {32, 64, 128, 256}) {
    int grid_size = (num_elements + block_size - 1) / block_size;
    kernel<<<grid_size, block_size>>>(...);
    cudaDeviceSynchronize();
    verify_output(output);
}
```

### FP16 vs FP32 Precision Tests
```cpp
// Verify half_precision mode stays within tolerance
auto output_fp32 = encoder_fp32.extract(audio);
auto output_fp16 = encoder_fp16.extract(audio);
float max_diff = max_element_wise_diff(output_fp32, output_fp16);
assert(max_diff < 1e-2f);  // FP16 tolerance is looser
```

## Test Fixture Management

```
tests/
├── fixtures/
│   ├── hubert_input.wav           # Test audio input
│   ├── hubert_output.bin          # Python reference output
│   ├── rmvpe_f0_reference.npy     # F0 reference
│   └── synthesizer_output.wav     # Synthesis reference
├── test_helpers.cpp               # Common test utilities
├── test_helpers.h
├── test_hubert.cpp
├── test_f0.cpp
├── test_synthesizer.cpp
└── test_pipeline.cpp              # Integration test (highest priority)
```

**Constitution Principle IX**: Integration-First Testing. `test_pipeline.cpp` is the highest priority test.

## Test Anti-Patterns for CUDA

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Testing only with small inputs | Misses memory coalescing issues | Test with 1s, 5s, 30s audio |
| Ignoring cudaGetLastError() | Silent kernel failures | Check after every kernel launch |
| No synchronization before verify | Race conditions | `cudaDeviceSynchronize()` |
| Testing only FP32 | Misses FP16 precision bugs | Test both precisions |
| No memory leak checks | Device OOM in production | `cudaMemGetInfo()` before/after |

## Verification Checklist

- [ ] Test fails before implementation (RED)
- [ ] Test passes with basic implementation (GREEN)
- [ ] Numerical alignment passes (L2 < 1e-4 or module-specific threshold)
- [ ] Full CTest suite passes: `ctest --output-on-failure`
- [ ] No CUDA memory leaks detected
- [ ] Both FP32 and FP16 modes tested (if applicable)
- [ ] Integration test (`test_pipeline`) passes

## See Also

- `.specify/memory/constitution.md` — Principle III (Test-First), VI (Numerical Alignment), IX (Integration-First)
- `specs/001-voice-conversion-engine/spec.md` — Success Criteria SC-001 to SC-003
