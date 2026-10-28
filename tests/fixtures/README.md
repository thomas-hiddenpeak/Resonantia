# Test Fixtures

`tests/fixtures/` 保存不进入 Git 的真实音频和数值参考资产。文件存在只说明本机具备资产，
不等于对应测试已经注册、执行或通过；模型/E2E 结论仍以 asset-gated CTest 的实际结果为准。

## Realtime R0 manifest

Spec 006 首版 profile 使用的真实音频和模型由
[`r0-assets.json`](../../specs/006-realtime-voice-conversion/r0-assets.json) 记录。该 manifest
逐项保存 repo-relative path、bytes、SHA-256、source revision、license evidence 和 generation
provenance。当前所有 `present` 文件的完整性可机器核验：

```bash
jq -r '.assets[] | select(.status == "present") | "\(.sha256)  \(.path)"' \
  specs/006-realtime-voice-conversion/r0-assets.json |
  sha256sum --check --strict
```

完整资格门当前应失败，因为尚无同一份有授权目标语料产生的代表性目标声线
model+Index pair 与 corpus manifest：

```bash
jq -e '.qualification_status == "ready" and
  (.qualification_blockers | length == 0)' \
  specs/006-realtime-voice-conversion/r0-assets.json
```

不要删除 blocker 或填入虚构路径来让该命令成功。

## Real human speech

三条 WAV 均来自 Hugging Face
[`hf-internal-testing/librispeech_asr_dummy`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/5be91486e11a2d616f4ec5db8d3fd248585ac07a)
revision `5be91486e11a2d616f4ec5db8d3fd248585ac07a` 的 clean validation parquet。2026-08-10
从固定 revision 重新读取对应 row 并用当前 `tools/uv.lock` 环境写 WAV，所得 bytes/SHA-256
与本地文件逐字节一致。上游
[`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr)
卡片标注 CC BY 4.0；dummy mirror 自身没有重复 license metadata，此差异在 manifest 中保留。

| Path | Dataset row / utterance | Frames @ 16 kHz | Duration | SHA-256 |
|---|---|---:|---:|---|
| `speech_00.wav` | 0 / `1272-128104-0000` | 93,680 | 5.855 s | `799f78ed4beb4de7ceae3a809262d4ce242394342ccd1d58cef7d49dbc2def46` |
| `speech_01.wav` | 1 / `1272-128104-0001` | 77,040 | 4.815 s | `473b1bb573e8f7cdcd639c6e318036236190107b66bfe6892286121a92d464c1` |
| `speech_02.wav` | 2 / `1272-128104-0002` | 199,760 | 12.485 s | `5f4d8f7331acffc3d2efbe0f764b9eb3447911dc17f9843e38e1083eb1685b9c` |
| `speech_librispeech.wav` | stable alias of row 0 | 93,680 | 5.855 s | `799f78ed4beb4de7ceae3a809262d4ce242394342ccd1d58cef7d49dbc2def46` |

`tools/download_test_audio.py` 是生成入口，并已固定到 manifest 中的 immutable revision。
重新生成后仍须按 manifest 核对 row metadata、文件大小和 SHA-256，不能只凭脚本退出成功
宣称资产相同。

## Numerical reference inventory

Web codec集成通过`RESONANTIA_ENABLE_AUDIO_CODEC_TESTS=ON`启用：
`test_web_audio_codecs`从上述`speech_00.wav`取首1秒，用当前FFmpeg在唯一临时目录
编码WAV/MP3/AAC/ALAC/FLAC/Vorbis/Opus/WMA/AIFF，再经Web上传边界解码为WAV。
检查1±0.15秒时长（允许有损codec帧padding）及非空、非静音完整PCM，完成后删除临时
音频；没有模型调用，不构成声线质量或模型数值对齐证据。缺FFmpeg配置失败、缺speech
fixture明确不注册，均不能计通过。默认host的`test_web_audio_upload`使用合成PCM和
mock runner，只验证格式边界、失败事务和资源限额，不依赖FFmpeg或真实资产。

当前目录中的实际 reference 是带 shape header 的 `.bin` 或 WAV，不是旧文档曾列出的 `.npy`：

| Area | Current files | Generator / evidence boundary |
|---|---|---|
| HuBERT | `hubert_ref_*.bin` | `tools/gen_hubert_reference.py`; 使用真实 row-0 WAV 和本地 HuBERT；生成环境历史版本未单独保存 |
| RMVPE | `rmvpe_htk_{basis,mel,salience,f0}.bin` | `tools/gen_rmvpe_htk_reference.py`直接调用上游；`rmvpe_htk_manifest.json`记录模型/源代码/音频/产物哈希 |
| E2E | `e2e_htk_{audio,nsff0,pitch}.bin` | 同上，正确HTK前端与未经微调的基础G；不含Index、高通、padding、随机噪声或RMS混合 |
| Historical RMVPE | `rmvpe_ref_*.bin` | 旧C++与Python共同错用Slaney频率刻度，已撤销其官方音高对齐证据资格；仅保留历史控制输入 |
| FCPE | `fcpe_ref_mel.bin`, `fcpe_ref_f0.bin` | `tools/export_fcpe.py`; `torchfcpe==0.0.4` bundled checkpoint 来源/hash/MIT license 与本地 safetensors byte-identical replay 已固定；reference 仍是既有数值资产 |
| WavLM | `wavlm_ref_*.bin` | `tools/export_wavlm.py`; 不属于首个 realtime profile |
| VITS v2 | `vits_ref_*.bin`, `vits_ref_output.wav` | `tools/gen_vits_reference.py`; deterministic noise-zero reference，依赖本地 v2 generator |
| Separation | `separation/*` | conversion/reference scripts；模型、任务和许可证需按 Spec 004 单独治理 |
| VAD | `vad/*` | `tools/convert_vad_weights.py`; 不属于 realtime profile |
| Local comparison output | `chunk_out.wav`, `whole_out.wav` | 生成命令和来源未保存，不能作为规范 reference |

上述数值 reference 仍由现有可选 integration tests 使用；此处不把缺失的历史 package revision
或许可证猜成已验证。为新的 realtime quality/continuity 门槛生成 reference 时，必须新建带
输入、模型、Index、工具 revision、命令和输出 hash 的记录，不能覆盖这些历史文件后沿用旧结论。

## Sampled VITS reference (2026-10-24)

`uv run --project tools --no-sync python -u tools/gen_vits_sampled_reference.py`
loads the independent upstream RVC network and the original v2/40k G. It uses the
real `vits_ref_phone.bin` plus corrected `e2e_htk_{pitch,nsff0}.bin`, and supplies
explicit Gaussian draws: prior scale 0.66666, NSF voiced std 0.003/unvoiced 0.1/3.
Seed 0 uses an independently implemented SplitMix64/Box-Muller recipe, not a claim
of reproducing PyTorch's version-specific RNG sequence. `vits_sampled_manifest.json`
records input, weight, upstream and output hashes and library versions.

`vits_sampled_{prior_noise,nsf_noise,z,har,audio}.bin` and
`vits_sampled_reference.wav` preserve draws and intermediate/output values.
The VITS and trainable-decoder tests require correlation >0.999 and relative L2
<1e-3; E2E checks the production default against the sampled reference. Historical
`vits_ref_*` and `e2e_htk_audio.bin` are preserved and explicitly tested in mean
mode. These are numerical gates, not naturalness or target-voice acceptance.

## RMVPE HTK correction (2026-10-24)

Active RMVPE/E2E CTests require the new names; they never silently fall back to the
historical Slaney files. The old `tools/gen_rmvpe_reference.py` entry point delegates
to the independent generator, avoiding another locally copied network/frontend.
Standalone VITS tests retain their unchanged fixed-input fixtures, including their
historical F0 values: these test decoder arithmetic, not pitch accuracy.

Reproduce with `uv run --project tools --no-sync python tools/gen_rmvpe_htk_reference.py`.
Requires the local upstream `tools/rvc_ref`, real row-0 audio, HuBERT layer-12 fixture,
RMVPE and base v2/40k G safetensors. The optional historical comparison additionally
uses `rmvpe_ref_f0.bin` when present. All model loads are strict; no pickle is loaded.
Generation uses CPU FP32, 2 threads, torch `2.13.0+cu130`, librosa `0.11.0`.

- Upstream: [RVC RMVPE](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer/rmvpe.py),
  local source SHA-256 `074171aec63e4b68f04d8691c62af50a1d99b9bc6dc56c4d6749fad4ba64853a`.
- RMVPE weights SHA-256 `820e94253031f7709724372dc7990576a444c486645120fbd93b01954d1ec704`.
- Base G SHA-256 `8792f5dc13bcbf8a5bf0a0890823ec0d5e5cb1102374a3f3a30bf59cf680a2da`.
- Input SHA-256 is the immutable row-0 hash above; output shapes are mel `[128,586]`,
  salience `[586,360]`, F0 `[586]`, E2E audio `[233600]` at 40 kHz (5.84 seconds).
- Filterbank uses **HTK frequency spacing and Slaney area normalization**. Seven
  independent coefficients across the spectrum are frozen in asset-free `test_f0`.
  Real-model tests compare salience, every voicing decision and voiced F0 (max/mean
  below 0.5 Hz). The previous bounded-window thresholds remain unchanged.
- Historical wrong-scale and upstream correct-scale F0 differ by median 558.94 cents
  on 358 jointly voiced frames. This does not establish human naturalness or timbre.

## Joint training one-step reference (2026-10-25)

`joint_training/` contains the immutable independent CPU FP32 reference for
[ADR-0008](../../docs/adr/0008-joint-voice-training-alignment.md). Hashes, model/source
identity and upstream source hashes are in `joint_training_manifest.json`.
It uses frames 200–327 of the existing real `singing-phrases/source_phrase.wav`
diagnostic, complete 128-frame encoders, decoder crop 48–79, fixed Gaussian inputs,
original weight-norm g/v parameters, selected speaker row 0, and one alternating
AdamW D/G update. This 0.32-second decoder crop is a numerical fixture, not a
listening sample. No production model or training dataset is modified.

Generate into a new directory with:

```sh
uv run --project tools --no-sync python tools/gen_joint_training_reference.py <new-directory>
```

Requires the pinned local `tools/rvc_ref`, v2/40k G and D, and the provenance-checked
`runs/CyberTom_Pilot_HTK/experiment/{singing-phrases,content-stages}` assets. Copy the
generated files into `tests/fixtures/joint_training/` only after checking provenance;
do not substitute synthetic speech or overwrite the frozen reference after a failure.
`--resource-fixture` only prepares a real 600-frame input; `--replay-cpp` is explicitly
a diagnostic isolation experiment and **must never replace the acceptance reference**.

`test_joint_numerical` checks every latent stage, audio correlation/error, module
gradients, AdamW moments/parameters and inference export. It is asset-gated and
currently records an unresolved strict gradient/update alignment failure. Upstream
FP32/FP64 discriminator branch sensitivity is recorded separately; it does not waive
the original thresholds. `test_joint_model_contract` requires only the real G/D and
checks malformed shape rejection with the same byte count and failed-init rollback.

`joint_training_kernels/` is a separate diagnostic fixture set. It fixes the real
inputs and incoming gradients of the decoder's four transposed convolutions and
two residual convolutions (including dilation 5), then compares their input, raw
weight-norm g/v and bias gradients. It is generated by
`tools/probe_joint_decoder_gradient.py --kernel-fixtures <new-directory>` using the
recorded C++ intermediate trace; hashes are in `joint_training_kernels_manifest.json`.
`test_joint_kernel_gradients` checks these six layers at the original 0.001 gradient
threshold. This can isolate primitive backward arithmetic from activation-branch
sensitivity, but does not replace the frozen independent whole-step reference,
cover every network branch, or establish voice quality.

### Strict-gate failure attribution

The maintainer explicitly retained the original gates after the one-step failure.
The following offline tools make diagnostic interventions; none changes the frozen
golden files, authorizes short training, or establishes a passing whole-step gate:

- `tools/probe_joint_activations.cpp` privately recompiles the trainer with a
  compile-time observer. It records 77 decoder activation outputs at the recorded
  C++ latent and incoming audio gradient, with zero optimizer updates. The normal
  library compiles these observer calls away.
- `tools/probe_joint_branch_sensitivity.py` retains native upstream forward values
  but replays recorded C++ activation signs in backward. This isolates derivative
  branch changes from convolution-backward arithmetic.
- `tools/probe_joint_graph_attribution.py` extends that intervention to full G
  backward with recorded audio and updated D fixed. It takes no optimizer step.
- `tools/probe_joint_discriminator_branches.py` checks all nine D branches at the
  same real/generated audio, with and without recorded backward masks, and no
  optimizer step.
- `tools/probe_joint_optimizer_isolation.py` feeds recorded C++ gradients to
  independent CPU AdamW. Agreement isolates optimizer arithmetic; it does not
  imply that the original independently computed gradients or updates agree.

Run from the repository root using the retained experiment inputs and a new output
path, for example:

```sh
c++ -O3 -std=c++20 -Iinclude -Isrc -I/usr/local/cuda-13.2/targets/sbsa-linux/include tools/probe_joint_activations.cpp /tmp/resonantia-performance-20261024.uemhTG/libvoxmutatio.a -L/usr/local/cuda-13.2/lib64 -Wl,-rpath,/usr/local/cuda-13.2/lib64 -lcudart -lcublas -lcufft -lcublasLt -lculibos -lpthread -ldl -lrt -o /tmp/resonantia-joint-activations-20261025
/tmp/resonantia-joint-activations-20261025 models/pretrained_v2/pretrained_v2/f0G40k.safetensors runs/CyberTom_Joint_20261025/experiment/reference runs/CyberTom_Joint_20261025/experiment/cpp-step-v5 <new-activation-directory>
uv run --project tools --no-sync python tools/probe_joint_branch_sensitivity.py <reference> <cpp-step-v5> <activation-directory> <new-report.json>
uv run --project tools --no-sync python tools/probe_joint_graph_attribution.py <reference> <cpp-step-v5> <activation-directory> <new-report.json>
uv run --project tools --no-sync python tools/probe_joint_discriminator_branches.py <reference> <cpp-step-v5> <new-report.json>
uv run --project tools --no-sync python tools/probe_joint_optimizer_isolation.py <reference> <cpp-step-v5> <new-report.json>
```

Evidence is retained under
`runs/CyberTom_Joint_20261025/experiment/strict-followup/`; numerical results and
remaining acceptance failures are recorded in `specs/PROJECT_STATE.md`.
`tools/probe_cuda_allocations.cpp` and `tools/probe_host_allocations.cpp` are Linux
LD_PRELOAD observers, not runtime dependencies. They measure explicit CUDA buffers
and glibc totals at large allocations respectively, not total device memory or RSS.
`tools/probe_process_memory.py` samples Linux mappings in an isolated child with a
time/memory guard; sampling can miss transient peaks. Original CTest resource
thresholds and uninstrumented results remain the acceptance evidence.

### FP32 operator regressions

`test_autograd` also contains small, asset-free cancellation regressions for weight
norm, Softmax backward, grouped-convolution bias and transposed-convolution channel
accumulation. Hexadecimal expected values were evaluated independently with the
same CPU/aarch64 torch build as the frozen joint fixture. They include vector tails,
overlapping transposed-convolution taps and 512 input channels. The columns-workspace
overflow case must throw before allocating the product of otherwise small tensors.
These are operator tests, not speech-quality evidence or replacement joint goldens.

Relevant upstream arithmetic is pinned to torch commit
`cf30153c4c131c8164ee7798e5022d810682e2cb`:
[weight norm](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/WeightNormKernel.cpp),
[Softmax](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/SoftMaxKernel.cpp),
and [transposed convolution](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/NaiveConvolutionTranspose2d.cpp).
Red/green commands and remaining whole-step failures are retained under
`runs/CyberTom_Joint_20261025/experiment/numerical-repair/` and summarized in
`specs/PROJECT_STATE.md`. The original 23-file joint fixture remains unchanged.

The subsequent `forward-repair/` evidence fixes forward accumulation for the
frozen aarch64 backend choices: oneDNN/ACL channels-last above 20480 input
elements, NVPL 0.3.0 small convolutions with 640-element reduction blocks, and
the four-lane speaker-conditioning dot product. Synthetic operator constants
cover both dispatch paths, non-aligned periods and block boundaries; all six
real decoder fixtures now assert elementwise-exact forward output as well as
their original backward tolerances. No reference masks enter production.
The new conditioning test initially used canonical 2D `linear` rather than
RVC's `Conv1d` layout; those initial failures are not defect evidence. A control
binary with the specialized reduction disabled fails all seven corrected
independent `Conv1d` constants. Original joint fixtures were never changed.

`test_joint_training` also checks separate FP32 NSF phase/amplitude rounding
against upstream SineGen constants, including voiced/unvoiced transitions and
invalid dimensions/non-finite values. The 2e-8 absolute operator tolerance
covers scalar sinf versus SLEEF rounding; it does not replace or relax the
original whole-step gradient/update gate. The latter now passes after fixing
host FMA contraction. Frozen manifest wording about a known failure describes
the original generation baseline; current verification is in PROJECT_STATE.

The CLI's private `joint_spectrum` helper is checked against the original real
waveform/spectrum fixture by `test_joint_numerical` at relative L2 < 1e-4.
`test_joint_training` covers its floor, finite-input and 32..600-frame boundaries.
No fixture regeneration accompanies the CLI/Web wiring. The Node
`test_web_joint_ui` covers capability gating against old/new servers and stale
HTML, mode constraints and preservation of edits during capability refresh.

## RMS envelope reference (2026-10-26)

`tools/gen_rms_operator_reference.py <new.json>` extracts the pinned upstream
`change_rms` function AST and records librosa/torch versions and source hash.
The default 1000 ms case is unmodified upstream code. Low-level constants in
`test_bounded_pipeline_dsp` use piecewise finite arrays at 16k/40k to verify
centered zero padding, fixed-window normalization, half-pixel interpolation,
mix endpoints and PCM quantization across I/O blocks. They failed 22 assertions
before the fix. Other generated window sizes are offline diagnostics, not a
supported runtime API. Original joint-training goldens are unchanged.

Real-audio checks use the two recorded <=15-second source/raw-model pairs and
`tools/probe_rms_envelope.cpp` to export the exact C++ filtered source and RMS
outputs, then `tools/probe_joint_dynamics.py` calls that same independent
upstream function. Evidence is under the current joint-entry/dynamics-20261026
directory; numerical alignment does not establish accepted dynamic/timbre quality.

## Storage policy

WAV、`.bin` 和大型模型均被仓库 `.gitignore` 排除，普通 checkout 不会自动获得它们。不要
把缺资产导致的 CTest 未注册或 skip 报告为通过，也不要把真实语音以外的正弦/噪声 fixture
用于模型质量或 E2E 声音结论。
