# Spec 004 — Source Separation (vocal / de-reverb / de-harmony)

**Status**: In Progress (foundation) · **Date**: 2026-08-07
**Constitution**: pure C++/CUDA, **zero Python at runtime** (Principle V/XI). Python
only in `tools/` (uv) for one-time weight conversion + offline numerical alignment.

## Why
Training a target voice needs a **clean single-speaker vocal**. Real user recordings
are often full songs (vocal + accompaniment + reverb + backing harmony). So audio
front-end preprocessing is a **first-class project feature**, not a test tool:

- **Vocal separation**: full song → vocal stem + instrumental.
- **De-reverb**: remove room reverb / echo tails from a vocal.
- **De-harmony (karaoke)**: remove backing/harmony vocals → lead only.

The WebUI "录音内容 × 含混响" selection maps to a preprocessing chain:
- 纯人声 + 干净 → (slice) → train
- 纯人声 + 含混响 → de-reverb → slice → train
- 带伴奏歌曲 → separate → de-harmony → de-reverb → slice → train

## Model choice (pragmatic first target)
The SOTA family is spectrogram Transformers/U-Nets over STFT:
- **MDX-Net / UVR conv U-Net** (magnitude-spectrogram U-Net → mask) — simplest, reuses
  our conv kernels + a GPU STFT. **First target.**
- **MelBand-RoFormer / BS-RoFormer** (band-split + rotary attention) — current SOTA,
  heavier (band-split, rotary pos emb). Second target; same STFT/iSTFT front-end.
- De-reverb + karaoke are the **same architecture, different weights** — once the
  runner exists, new tasks = new safetensors.

All are magnitude-mask or complex-mask models on STFT; **de-reverb/karaoke RoFormers
share the runner**. So the module is: STFT → model → mask → apply → iSTFT.

## Architecture (`src/separation/`, `include/voxmutatio/separation/`)
- `stft.{h,cu}` — GPU STFT/iSTFT via cuFFT (complex). Hann window, hop, center pad,
  overlap-add reconstruction. **Foundation (also lets us move `compute_spec` to GPU).**
- `separator.{h,cpp}` — `Separator` class: load a task model (safetensors), run
  chunked overlap-add inference on a waveform, return stems. Tasks: `kSeparate`,
  `kDeReverb`, `kDeHarmony`.
- Model runner (conv U-Net first; RoFormer later) on the autograd-free inference path
  (forward only, cuBLAS + conv kernels).
- Chunked processing (RoFormer default ~11 s @ 44.1k, overlap 4) for long audio + VRAM.

## Weights
`tools/convert_separation_weights.py` (uv, offline) converts a chosen open checkpoint
(MSST/UVR MDX or MelBand-RoFormer) → F32 safetensors under `models/separation/`.

## CLI + WebUI integration
- `vc_preprocess --separate --dereverb --deharmony` (chain flags) call the Separator
  before slicing.
- `vc_serve` maps 录音内容/含混响 → the chain; training preprocessing runs it.

## Phases
- [x] S0: spec + module skeleton + GPU STFT/iSTFT foundation (this round).
- [ ] S1: STFT round-trip numerical test (reconstruct == input within tol).
- [ ] S2: conv U-Net (MDX-style) forward runner + weight conversion + alignment vs ref.
- [ ] S3: chunked overlap-add long-audio inference; vocal/instrumental stems.
- [ ] S4: de-reverb + de-harmony weights on the same runner.
- [ ] S5: wire into vc_preprocess + WebUI chain; real-audio validation.
- [ ] S6 (SOTA): MelBand-RoFormer runner (band-split + rotary attention).

## Gates
Real audio only for functional tests. STFT round-trip < 1e-4. Each model runner
numerically aligned vs a one-time Python reference (offline). Zero Python at runtime.
