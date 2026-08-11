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

当前目录中的实际 reference 是带 shape header 的 `.bin` 或 WAV，不是旧文档曾列出的 `.npy`：

| Area | Current files | Generator / evidence boundary |
|---|---|---|
| HuBERT | `hubert_ref_*.bin` | `tools/gen_hubert_reference.py`; 使用真实 row-0 WAV 和本地 HuBERT；生成环境历史版本未单独保存 |
| RMVPE | `rmvpe_ref_*.bin` | `tools/gen_rmvpe_reference.py`; 来源与环境仍需独立 provenance manifest |
| FCPE | `fcpe_ref_mel.bin`, `fcpe_ref_f0.bin` | `tools/export_fcpe.py`; `torchfcpe==0.0.4` bundled checkpoint 来源/hash/MIT license 与本地 safetensors byte-identical replay 已固定；reference 仍是既有数值资产 |
| WavLM | `wavlm_ref_*.bin` | `tools/export_wavlm.py`; 不属于首个 realtime profile |
| VITS v2 | `vits_ref_*.bin`, `vits_ref_output.wav` | `tools/gen_vits_reference.py`; deterministic noise-zero reference，依赖本地 v2 generator |
| Separation | `separation/*` | conversion/reference scripts；模型、任务和许可证需按 Spec 004 单独治理 |
| VAD | `vad/*` | `tools/convert_vad_weights.py`; 不属于 realtime profile |
| Local comparison output | `chunk_out.wav`, `whole_out.wav` | 生成命令和来源未保存，不能作为规范 reference |

上述数值 reference 仍由现有可选 integration tests 使用；此处不把缺失的历史 package revision
或许可证猜成已验证。为新的 realtime quality/continuity 门槛生成 reference 时，必须新建带
输入、模型、Index、工具 revision、命令和输出 hash 的记录，不能覆盖这些历史文件后沿用旧结论。

## Storage policy

WAV、`.bin` 和大型模型均被仓库 `.gitignore` 排除，普通 checkout 不会自动获得它们。不要
把缺资产导致的 CTest 未注册或 skip 报告为通过，也不要把真实语音以外的正弦/噪声 fixture
用于模型质量或 E2E 声音结论。
