<!--
Resonantia project skill. Adapted from addyosmani/agent-skills (MIT License).
-->

# Test-Driven Development

Use this guide when adding behavior, fixing defects, changing kernels, or modifying
CTest coverage. Start from the governing contract and select the lowest test layer
that can reproduce the behavior without weakening the evidence.

## Red, green, refactor

1. **Red:** add a focused test that fails for the intended reason, or preserve a
   minimal reproducer when a pre-fix run cannot safely be repeated.
2. **Green:** implement only the behavior needed to satisfy the contract. Verify the
   test actually executes the changed path.
3. **Refactor:** simplify ownership and implementation while keeping the focused test
   green, then run the relevant broader layer.
4. Record the command, environment, assets, pass/fail/skip counts, and numerical
   result. A test that only prints a metric must also assert its acceptance threshold.

## Two test layers

### Contract tests: default and asset-free

Run these without external model or audio assets. Cover:

- public input validation and supported/unsupported options;
- dtype, shape, byte-range, overflow, finite-value, and size boundaries;
- initialization rollback, repeated initialization, cleanup, and error propagation;
- deterministic DSP primitives, CUDA kernels, gradients, and small numerical helpers;
- CMake/CTest registration and stable repository-relative asset/output paths.

Synthetic arrays, impulses, sine waves, noise, and generated tensors are allowed in
this layer only when they test a low-level deterministic property, smoke path,
boundary, failure path, or gradient. They do not validate speech-model quality.

### Integration and numerical tests: real assets

Use documented real model weights, real human speech, and provenance-tracked reference
outputs for content encoders, F0 extractors, index behavior, synthesis, separation,
training integration, and end-to-end pipelines.

- Apply the active Spec's metric and threshold; record the measured value.
- Validate model/config compatibility and the complete cross-module path.
- Keep asset-dependent tests clearly gated. Missing weights, audio, fixtures, or
  hardware must produce an explicit skip/unverified result and must never count as a
  pass or satisfy a completion checklist.

## CUDA test requirements

- Check the return value of CUDA calls and the launch error after each kernel.
- Synchronize the relevant stream before host verification so asynchronous failures
  are observable; avoid relying on unrelated global synchronization.
- Exercise empty, minimal, non-aligned, representative, and maximum supported sizes.
- Test repeated calls and cleanup when the change owns device resources or scratch.
- Check numerical output with finite-value assertions and a stable reference, not
  only shape or successful launch.

FP32/CUDA is the current baseline. Do not add FP16-success requirements unless an
approved Spec introduces FP16 support; test explicit rejection when unsupported
precision is part of a public boundary.

## Execution

Build and run the smallest target first, for example:

```bash
cmake --build <build-dir> --target <test-target>
ctest --test-dir <build-dir> -R '<focused-regex>' --output-on-failure
```

Then run the broader asset-free contract suite when practical. Run the documented
asset-enabled integration suite for model, numerical, or pipeline claims. Do not hide
failures with permissive shell constructs, and do not describe unrun tests as passing.

## Completion checklist

- The test failed for the expected pre-fix reason or the reproducer is documented.
- Success, invalid input, boundary, and relevant failure/cleanup paths are asserted.
- CUDA asynchronous failures are observable at the correct boundary.
- Numerical tests assert the governing threshold.
- Contract and required real-asset integration layers passed, or gaps are explicit.
- CTest runs correctly from the configured build directory without source-tree
  working-directory assumptions.

See `.specify/memory/constitution.md`, `tests/fixtures/README.md`, and the relevant
feature Spec for asset provenance and numerical criteria.
