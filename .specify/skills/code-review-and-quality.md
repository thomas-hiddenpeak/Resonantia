<!--
Resonantia project skill. Adapted from addyosmani/agent-skills (MIT License).
-->

# Code Review and Quality

Use this guide for implementation review and repository audits. Review the current
tree and executable evidence; do not use historical audit text as proof of current
behavior. Do not edit code when the request is review-only.

## Review order

1. Establish scope with `git status`, the relevant diff, public headers, callers,
   CMake targets, and governing Spec requirements. Preserve unrelated dirty changes.
2. Inspect tests before trusting implementation claims. Record which commands ran,
   which assets were present, and every failure or skip.
3. Trace changed data across module boundaries: type, shape, units, ownership,
   lifetime, synchronization, error propagation, and compatibility.
4. Review failure paths and cleanup with the same attention as the successful path.
5. Report actionable findings first, with severity, exact file/line, trigger, impact,
   and a reproducible basis. Separate unverified risk from demonstrated defects.

## Required review surfaces

### Boundaries and arithmetic

- Validate null/empty inputs, enums, dtype/shape/byte ranges, finite values, tensor
  compatibility, path handling, and file replacement/truncation.
- Check addition, multiplication, frame-count calculations, signed/unsigned
  conversion, and every narrowing cast before allocation or kernel launch.
- Reject unreasonable lengths and dimensions before padding, resampling, or workspace
  sizing. Ensure loops make progress and output-size calculations have an upper bound.

### Ownership and failure safety

- Require RAII or an equally reliable single owner for files, mmap, allocations,
  CUDA resources, locks, and temporary outputs.
- Check partial initialization, repeated initialization, exceptions, early returns,
  and destructor behavior. Initialization must stage new state and commit only after
  all validation succeeds.
- Check borrowed pointer lifetimes after remap, move, asynchronous launch, and object
  destruction. Shared scratch must be operation-local or correctly serialized.

### CUDA correctness

- Check API return codes, launch bounds, dynamic shared-memory limits, grid arithmetic,
  and a launch error check after every kernel.
- Check asynchronous errors at the synchronization or handoff boundary that makes the
  result observable. Do not require global `cudaDeviceSynchronize()` when correct
  stream/event synchronization preserves the contract.
- Verify stream ownership, event lifetime, host-read synchronization, and library
  handle/stream association. Look for use-after-free and cross-stream races.

### Memory and numerical behavior

- Derive peak host and device memory from input/model dimensions. Flag unbounded
  `N x M` intermediates, whole-file copies, retained scratch, and duplicate weights.
- Check accumulation precision, division by zero, NaN/Inf handling, normalization,
  index bounds, padding limits, and degenerate audio/model inputs.
- Apply the active Spec's numerical metric. Shape-only checks and printed metrics are
  not alignment evidence.

### Contracts and integration

- Compare public API, implementation, CLI/help, Web behavior, CMake registration,
  tests, and SDD claims. Report cross-module disagreement explicitly.
- Treat current FP32/CUDA behavior as the supported baseline unless the active Spec
  says otherwise. Unsupported modes need rejection tests, not speculative support.
- Require real assets for model and E2E claims. A missing fixture or skipped test is
  an evidence gap, not success.

## Severity and handoff

- **P0**: data loss, security exposure, memory corruption, or broadly unusable build.
- **P1**: likely crash, wrong output, contract break, race, leak, or severe regression.
- **P2**: bounded correctness, robustness, maintainability, or evidence defect.
- **P3/Nit**: optional cleanup with no demonstrated correctness impact.

If no actionable findings remain, say so and list residual untested risks. Include
the exact build/CTest commands and results; never generalize a focused pass into a
full-suite claim.

See `.specify/memory/constitution.md`, `.specify/skills/test-driven-development.md`,
`docs/DESIGN.md`, and the task's relevant Spec.
