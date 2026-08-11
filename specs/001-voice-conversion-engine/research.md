# Research Record: Voice Conversion Engine

**Status**: Historical research record; superseded decisions are marked explicitly
**Updated**: 2026-08-09

## Current Conclusions

### Runtime architecture

- C++20/CUDA20, safetensors weights, no Python/PyTorch dependency at runtime.
- CUDA FP32 is the only supported execution mode.
- WAV is the only supported input/output file format.
- Offline pipeline output remains at the synthesizer model's native sample rate.

### Content encoder

**Current decision**: HuBERT/ContentVec is the default RVC encoder.

- v1 synthesizer consumes the 256D final projection.
- v2 synthesizer consumes the 768D transformer representation.
- WavLM-Base+ has an independent implementation and alignment test, but is optional and requires
  a synthesizer trained for WavLM features.

**Superseded**: “WavLM is primary and HuBERT is legacy.” This is not the current product
contract and would break compatibility with stock HuBERT-based RVC models.

### F0

**Current decision**: RMVPE is default; FCPE is an implemented option; PM is unsupported.

RMVPE and FCPE run through native C++/CUDA implementations loaded from safetensors. They are
validated only when the corresponding local model and reference fixture are available.

**Superseded**:

- ONNX Runtime/TensorRT as the RMVPE runtime path.
- CPU PM fallback.
- Unqualified “FCPE is real-time” or fixed-millisecond claims. No reproducible project-wide
  benchmark currently supports those statements.

### Feature retrieval

**Current decision**: a private, unlinked filesystem-backed Flat Index snapshot is memory-mapped
for exact weighted k-NN retrieval. This avoids a FAISS runtime dependency, keeps the exact hot
path on host memory and isolates active bytes from source truncate/overwrite. `FICLONE` is preferred;
copy fallback needs index-sized temporary filesystem capacity. File header, dimension, query range
and `k` remain contract boundaries.

### Synthesizer

**Current decision**: F0 VITS v1/v2 inference is implemented in CUDA FP32. non-F0 models are
rejected. Numerical alignment is established by asset-gated integration tests, not by the
mere presence of loading code.

### Source separation

Separation is managed by Spec 004. The as-built implementation includes Open-Unmix,
MelBand-RoFormer task models and Silero VAD.

**Superseded**: the early “MDX conv U-Net first” selection. It remains useful background
research but is not the runner that was delivered first and must not be described as current.

## Data and Weight Decisions

- Runtime weights use safetensors; pickle/`.pth` loading stays in offline Python conversion
  tools only.
- Model weights and numerical fixture dumps are local assets and are excluded from Git.
- CMake registers an integration test only after checking every declared asset.
- FP16 tensor variants may exist in source checkpoints or metadata, but the current inference
  runtime contract is FP32 only.

## Numerical Validation Strategy

1. Keep small contract tests independent of local model assets.
2. For each converted model, generate a one-time trusted reference in `tools/`.
3. Register the C++ alignment test only when model, input and all reference tensors exist.
4. Compare intermediate stages as well as final audio so layout/normalization errors are
   localized.
5. Reject non-finite values, invalid dimensions and overflow before device work.

Current design thresholds include HuBERT RMS error, RMVPE F0 error and VITS correlation gates,
but a passing default contract suite alone does not prove those asset-gated tests ran.

## Performance Research Boundary

Earlier notes contained isolated latency and speedup numbers. They are **superseded as project
status claims** because hardware state, clocks, model, input duration, warm-up, compiler flags
and measurement method were not recorded consistently. Future benchmark reports must include
all of that metadata and separate component latency from end-to-end latency.

The bounded offline migration now has a two-point 17.565/35.130-second Orin resource gate plus
structural core/F0 bounds. It is evidence for the qualified HuBERT profiles, not a cross-machine
benchmark, an arbitrary-duration empirical proof, or a realtime result; WavLM remains on the
legacy file path pending a matching trained synthesizer.

## Open Research

- End-to-end real-time state and chunk boundary design.
- Reproducible latency, throughput and memory benchmark harness.
- WavLM-trained synthesizer compatibility matrix.
- Objective and human evaluation protocols for trained voice similarity and separation quality.
