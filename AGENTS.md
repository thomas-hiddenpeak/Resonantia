# Resonantia Agent Guide

This file is the Codex entry point for repository governance. Keep it concise; do not
copy the full SDD documents here.

## Required context

Before changing or reviewing the repository:

1. Read `.specify/memory/constitution.md`.
2. Read `specs/README.md`, then `specs/PROJECT_STATE.md`.
3. Read the `spec.md` (and, when relevant, `plan.md`, `data-model.md`, or
   `research.md`) for every feature touched by the task.
4. Read the affected sections of `docs/DESIGN.md`.
5. For a refactor, also read `docs/REFACTORING_GUIDE.md` and
   `specs/TRACEABILITY.md` before editing.

Read only the task-relevant portions after the required top-level context. If a
required document is absent, stale, or self-contradictory, report that gap; do not
invent its contents.

## Task-specific skills

Load the relevant instruction file rather than duplicating it here:

- Public headers, configuration, or API/interface work:
  `.specify/skills/api-and-interface-design.md`
- Implementation or code audit/review:
  `.specify/skills/code-review-and-quality.md`
- Test creation, test changes, or verification:
  `.specify/skills/test-driven-development.md`
- Git operations only:
  `.specify/skills/git-workflow-and-versioning.md`

When several categories apply, read all applicable files. Do not load the Git skill
for work that does not perform Git operations.

## Authority and evidence

- For claims about what the current tree actually builds or does, the strongest
  evidence is the current public API and implementation, build configuration, and
  test results produced from the current tree. Never claim a test passed without a
  recorded command and result, and never count a skip or missing asset as a pass.
- For normative intent, authority descends from the Constitution, to the relevant
  approved feature spec and acceptance criteria, to `docs/DESIGN.md`.
- `specs/PROJECT_STATE.md`, `specs/TRACEABILITY.md`, plans, research notes, and the
  README are maintained summaries. Reconcile them with primary evidence rather than
  using them to override it.
- `docs/ENGINEERING_AUDIT.md` is a historical snapshot. It may explain provenance,
  but it must not substitute for inspection of the current code, interfaces, CMake,
  or CTest results.
- If normative documents and executable evidence disagree, preserve both facts:
  describe the implemented behavior, identify the unmet contract, and update the
  authoritative documentation or implementation within the task's authorized scope.

## Change discipline

- Stay within the requested scope. Preserve all pre-existing user changes and
  untracked files; never discard or overwrite a dirty worktree to simplify a task.
- Keep public contracts, implementation, CMake wiring, tests, and traceability in
  sync. Do not silently broaden accepted inputs or report unsupported behavior as
  implemented.
- Refactors and performance work do not get new feature specs. Architecture changes
  require an explicit migration section in the governing spec or design document
  and a `Proposed` ADR following `docs/adr/README.md` before implementation;
  implementation starts only after maintainer approval.
- Maintain C++20/CUDA runtime purity, explicit ownership and RAII, bounded memory,
  checked numeric and size boundaries, and transactional initialization.

## Verification gates

- Build the affected targets and run the smallest relevant CTest set first, then the
  broader asset-free contract suite when practical.
- Model, numerical-alignment, and end-to-end claims require the documented real model
  and real-audio assets. Synthetic inputs are acceptable only for low-level
  deterministic unit, smoke, boundary, and gradient tests.
- Run integration tests for cross-module or pipeline changes. Validate public error
  behavior and failure paths, not only the happy path.
- Report exact commands, failures, skipped tests, unavailable assets, and unverified
  requirements in the handoff. Do not mark a spec or task complete on partial
  evidence.
- After changing SDD or this bridge, run `scripts/check_sdd.sh` and fix every broken
  required path or active-document link before handing off.
