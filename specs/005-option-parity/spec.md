# Spec 005 — Option Parity, Selectable SOTA Models & Realtime

**Status**: Draft (planning) · **Date**: 2026-08-09
**Constitution**: pure C++/CUDA, zero Python at runtime; Python only in `tools/` (uv)
for one-time weight conversion + offline numerical alignment; Real Audio Only.

## Why
Audit (2026-08-09) against RVC found: the project keeps RVC's **pipeline** (content /
F0 / index / VITS / protect / rms-envelope / filter) and even exceeds it in some
inference details, but it does **not** keep RVC's **option flexibility**, and several
mandatory items in specs 001 remain unmet:

- Only **RMVPE** F0 actually works — `src/f0/fcpe.cpp::infer()` is a **stub** (returns
  zero F0). Spec 001 **FR-004** (RMVPE + FCPE) is unmet.
- **WavLM** content encoder is a stub/unwired. Spec 001 **FR-003** (WavLM + HuBERT)
  is unmet.
- Each separation stage hard-codes **one** model (`tools/vc_preprocess.cpp`); there is
  **no user choice** among SOTA variants.
- **Long-audio chunked inference** (RVC `opt_ts` silence-cut + crossfade) is missing;
  spec 001 edge case ">5min 需分块" is unmet — the VITS pipeline processes whole.
- **Batch inference** (`vc_batch`) is a skeleton — spec 001 **FR-010** partly unmet.
- **Formant shift** and **output resample-SR** exist in `VCConfig` but are not exposed
  (and formant has no DSP behind it).
- **Realtime/streaming** conversion is a stated P0 "must-do before human testing" with
  no spec and no implementation.

This spec consolidates the remaining work so a human evaluation sees credible RVC
parity + selectable SOTA options + the WebUI to match.

## Scope & Requirements (mandatory)

### F0
- **FR-501**: Implement a **real FCPE** F0 extractor (pure C++/CUDA, aligned vs a
  one-time Python reference), replacing the stub. Retire the "returns zero" path.
- **FR-502**: Make the F0 method **user-selectable** at inference (`--f0-method
  {rmvpe,fcpe}`) and at training feature extraction; surface it in the WebUI (F0
  dropdown; drop the hard-coded `presets.json` `f0method`).

### Selectable SOTA separation models
- **FR-503**: Allow **multiple registered models per task** and let the user pick one
  per stage. Backend: `vc_preprocess --sep-model <name>` per stage (or `--<stage>-model`);
  `vc_serve` `/api/step` accepts a `model` field. Registry
  (`tools/convert_roformer_weights.py`) registers ≥1 SOTA alternative per task where a
  credible checkpoint exists (e.g. anvuew dereverb aggressive / less_aggressive / mono).
- **FR-504**: WebUI cascade shows a **model picker** per stage (defaulting to the
  current SOTA), populated from a `/api/sep-models` endpoint.

### Long audio & batch
- **FR-505**: **Chunked conversion** for long inputs — silence-aware cut points
  (RVC `opt_ts` equivalent) + reflect-pad + crossfade, so >5min audio converts without
  OOM/quality loss. Wire into `pipeline.cpp` (behind a length threshold; identity for
  short clips so `test_e2e_conversion` corr>0.99 is unaffected).
- **FR-506**: Real **batch inference** — `vc_batch --input-dir --output-dir [--recursive]`
  converts a folder; optional WebUI batch entry.

### Content encoder
- **FR-507**: Wire **WavLM** as a selectable content encoder (verify alignment) OR, if
  not pursued, explicitly de-scope FR-003 in spec 001 (record the decision). No stubs
  left masquerading as features.

### Timbre / output
- **FR-508**: Expose **formant shift** (with a real formant-warp DSP) and
  **output resample-SR** (`--formant`, `--resample-sr`) in CLI + WebUI, or de-scope
  formant explicitly if deferred.

### Realtime (P0 gate item)
- **FR-509**: **Streaming conversion**: mic capture (WebAudio/AudioWorklet) → `vc_serve`
  WebSocket → stateful low-latency conversion (reuse `synthesizer::infer_stream` skip/
  return windows + streaming F0/content) → playback. Target latency on par with RVC
  (~90–170 ms). Ship a minimal WebUI realtime panel.

## Phases (priority order)
- [ ] P0-A **FR-505** long-audio chunked conversion (highest human-test risk).
- [ ] P0-B **FR-503/FR-504** selectable separation models (directly answers the audit).
- [ ] P1-A **FR-501/FR-502** real FCPE + F0-method selection.
- [ ] P1-B **FR-506** batch inference.
- [ ] P2-A **FR-508** formant + resample-SR exposure (or explicit de-scope).
- [ ] P2-B **FR-507** WavLM (or explicit de-scope).
- [ ] P2-C **FR-509** realtime streaming (largest; gate item per PROJECT_STATE).

## Gates
Real audio only for functional tests. Every new model/extractor numerically aligned vs
a one-time Python reference (offline). `test_e2e_conversion` corr>0.99 must remain
(chunking/formant identity on the reference path). Zero Python at runtime. No feature is
"selectable" in the UI unless a **real** implementation stands behind it.

## Non-goals / explicit decisions
- pm / harvest / crepe F0 back-ends are **not** planned (RMVPE + a real FCPE cover
  offline + realtime); recorded so FR-004 is satisfied by RMVPE + FCPE only.
