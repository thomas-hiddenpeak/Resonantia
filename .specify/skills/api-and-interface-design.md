<!--
Resonantia project skill. Adapted from addyosmani/agent-skills (MIT License).
-->

# API and Interface Design

Use this guide when changing public headers, configuration structs, CLI options, or
cross-module call boundaries. Read the Constitution, the relevant Spec, and the
affected public header and callers before proposing an interface.

## Workflow

1. State the observable contract before editing: accepted inputs, output shape and
   units, ownership, lifetime, error behavior, initialization state, and concurrency.
2. Inspect every current producer and consumer. Preserve behavior that callers can
   observe unless the governing Spec and migration plan explicitly change it.
3. Keep the public surface minimal. Put kernels, launch geometry, device storage,
   weight layout, and scratch allocation in private implementation details.
4. Add or update contract tests before implementation. Test invalid inputs and
   failure states as well as the happy path.
5. Update CMake wiring and active SDD traceability when the public contract changes.

## Contract rules

- Use the existing module's error convention consistently. Do not mix exceptions,
  booleans, sentinels, and optional error strings within one operation family without
  a documented migration.
- Distinguish invalid caller input from initialization, I/O, model-validation, and
  CUDA execution failures. Never return plausible output after a failed operation.
- Make initialization transactional: validate and allocate into staged state, then
  commit it atomically. On failure, leave the object destructible and in a documented
  uninitialized or previous-valid state.
- Validate public sizes before narrowing to `int` or computing byte counts. Check
  nullability, empty input, overflow, dtype, shape, finite values, compatible model
  variants, and supported enum values at the boundary.
- Treat defaults, result ordering, units, error categories, and accepted file formats
  as API. Do not silently broaden a field to advertise unsupported behavior.

## Ownership and CUDA boundaries

- Prefer owning RAII types for files, mmap regions, host/device buffers, CUDA
  streams/events, library handles, locks, and temporary files.
- Mark borrowed buffers and views clearly. Document the required lifetime, memory
  space, mutability, alignment, and stream association; never retain a borrowed view
  beyond its contract.
- Define whether a CUDA-facing call is synchronous or asynchronous. For asynchronous
  calls, state who owns the stream, when inputs may be reused, and where failures
  become observable.
- Keep scratch memory instance-owned or operation-owned and bounded. If an object is
  not thread-safe, make that part of the contract and enforce it internally where
  shared mutable state exists.

## Compatibility

- Prefer additive changes with validated defaults only when the default preserves
  existing behavior.
- Preserve RVC v1/v2 tensor dimensions and model compatibility checks. Do not make
  WavLM interchangeable with HuBERT, or FCPE interchangeable with RMVPE, without the
  required model/config compatibility validation.
- Reject unsupported precision, formats, sample-rate modes, and algorithms explicitly
  rather than exposing inert configuration fields.
- For a breaking API or architecture change, require the governing migration section,
  ADR, updated callers, and regression tests in the same change.

## Verification

- Build every affected library, CLI, and test target.
- Run focused contract tests, then the broader asset-free CTest suite when practical.
- Run real-model integration tests for changes that affect model compatibility or
  numerical output; report unavailable assets as unverified, not passed.
- Check public headers for private implementation leakage and circular dependencies.

See `.specify/memory/constitution.md`, `docs/DESIGN.md`, and the task's relevant Spec.
