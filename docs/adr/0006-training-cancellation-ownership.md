# ADR-0006: Training cancellation at owned safe points

**Status**: Proposed
**Date**: 2026-10-10
**Owners**: Resonantia maintainer
**Governing spec**: [Spec 003](../../specs/003-webui/spec.md), WEB-FR-011

## Authorization and implementation scope

The user explicitly approved training cancellation on 2026-10-10 at 12:41:53 UTC,
following the 10:30 UTC proposal: queued cancellation is immediate, running work
stops at safe points, no forced process termination, and an already started material
commit finishes. The current delegated instruction explicitly authorizes implementation
of job identity/ownership and the distinction between a cancel request and actual stop.
This record captures that authorized bounded increment before implementation. Its
formal ADR review status remains Proposed; it does not authorize broader cancellation,
training timeouts, new process signals, or changes to numerical training behavior.

## Context and decision drivers

Web training currently blocks on the shared GPU mutex, then synchronously runs
preprocess, vc_train and build_index. Neither the CLI nor ProcessRunner exposes a
cooperative stop interface. A flag alone cannot stop a live CLI or a queued mutex wait.
The existing shared running bit also cannot identify a stale request's job.

## Decision

- Each admitted job receives a service-instance-prefixed monotonic job identifier.
  Cancel is training-only and must name the exact job_id; a stale id cannot cancel,
  update, or release the next generation, including a step job.
- POST /api/train/cancel accepts multipart with one job_id field. Status and training
  acceptance responses add job_id; status adds kind, cancel_requested, cancelled and
  an explicit state. Existing running/done/error/stage fields remain.
- A cancel request returns promptly and is observable immediately. It does not set
  done or clear running. The request owner before worker handoff, and then only the
  matching worker, publish the terminal state after owned temporaries, GPU lock and
  synchronous child-process handling have finished.
- GPU waiting uses try_lock plus a condition-variable wait with a short fallback
  interval. Cancellation wakes a queued worker without waiting for the GPU holder.
  GPU mutex ownership remains with the executing operation; the cancel endpoint never
  unlocks it and cannot admit a new job while the old owner could still perform work.
- Running cancellation uses checks before/after successful CLI stages. The current
  CLI is allowed to exit naturally and be reaped by the existing runner. Pending
  status says it is waiting for that stage; there is no per-step vc_train stop hook,
  no bounded stop latency for a live CLI, no signals and no forced termination.
- An input-material commit already past its serialized commit boundary completes.
  Cancellation then prevents later work and preserves those committed materials.
  A final trained-artifact publication is a no-cancel boundary: a late cancel returns
  an explicit too-late conflict and publication finishes as done/error. Cancellation
  never claims a completed publication was undone. Real CLI or commit failures remain
  errors rather than being hidden by a concurrent cancel request.
- UI cancellation is tied to the displayed job_id, distinguishes requested/stopped/
  completed/error, and never labels a cancelled job as a generated voice.

## Alternatives considered

Directly clearing running from the endpoint risks the next job and is rejected.
Blocking on the GPU mutex cannot satisfy queued cancellation. Sending TERM/KILL,
interposing pthread or waitpid, and adding a guessed training timeout are outside
approval. A training-loop cooperative stop hook is a later, separately specified
extension; this increment does not modify training kernels or model output.

## Migration and rollback

Additive HTTP fields and one endpoint; existing clients that never cancel preserve
normal behavior. Update the bundled UI together with the endpoint. Preserve input
material commits and old published voices on cancelled staging work. No persisted
job database, model/data migration, checkpoint/resume or restart recovery is added.
Server-instance identity invalidates old browser requests after restart. Rollback is
reverting this bounded source increment after owned work has ended, retaining the
fixed source v8/evidence v9 snapshots and all user data; no automatic live deployment.

## Verification and SDD impact

Test first against the fixed baseline using existing ProcessRunner doubles, socketpair
HTTP responses and tiny owned fixtures. Verify queued cancellation while GPU remains
held, each stage boundary, duplicate/stale ids, busy-until-cleanup, normal retry,
completion/cancel races, committed-material retention and no residual work. Add actual
HTTP method/auth/cancel-input contract coverage and check bundled UI syntax/logic.
Run serial default contracts; preserve the optional 120s body evidence without an
unnecessary rerun. No real models, downloaded ownership probes, OS fault generation
or numerical-quality claim. Record any commit-interleaving case not deterministically
exercised by the supported test interfaces. Sync Spec 003, DESIGN and TRACEABILITY.

## Later explicitly authorized iteration extension (2026-10-10)

Following the read-only feasibility review, the user separately authorized CLI
orchestration and Web integration, retaining whole iterations and no forced kill.
The protocol/lifetime design was recorded in the increment evidence before code.
The original stage-only decision above is extended as follows; formal review remains
Proposed, and no new numerical trainer API, CUDA changes or process manager is added.

Use one job-owned 0600 regular byte file: source creates exclusively, reader opens
once, 0=continue/1=request. Source request and close serialize; close prevents late
writers from reaching a replacement job. CLI --cancel-file is optional and strict;
no option performs no token I/O. Exit 0=complete, 1=error, 2=cancelled without export.
Poll after completed init/clip stages, before each indivisible decoder step or GAN
D/G pair, and before export. An already begun export completes. Web exports only to
staging and discards late-cancelled results at the next checkpoint.

The Web lease exists before registration, rechecks cancellation while publishing the
source to the matching generation, then closes/unbinds after normal runner return,
before staging retires and the final owner releases resources. A requested exit2 is
recognized only for vc_train with no process/channel error. Write/close failure is
latched as final error; the endpoint does not claim a stopped process. Closed old
sources are inert. The ProcessRunner and generic success predicate remain unchanged.

Compatibility: Web and vc_train must be updated together; old vc_train rejects the
new option and Web reports error without publication. Direct callers without the
option keep their arguments/algorithm. No active service is deployed by this change.
Rollback reverts this source increment after owned work ends, preserving v9, the
checked commit-test snapshot, data and model files. There is no checkpoint/resume.

Host tests prove orchestration only, including a naturally exiting tiny helper and
real CLI pre-CUDA cancellation. GPU teardown/latency and low-level channel I/O failure
injection remain unverified. Existing cudaFree/TempDir cleanup reporting and runner
exception guard behavior are unchanged; no absolute stop bound is promised.
