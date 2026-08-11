# Resonantia Model Assets

`models/` 保存本机模型和一次性转换输入，大型文件不进入 Git。C++/CUDA runtime 只加载
safetensors；`.pt`/`.pth` 是 `tools/` 下离线转换的输入，不是运行时依赖。

## Realtime R0 qualified profile

Spec 006 的 machine-readable 权威清单是
[`r0-assets.json`](../specs/006-realtime-voice-conversion/r0-assets.json)。以下状态来自
2026-08-10 的实际文件、官方 Hugging Face headers/page 和当前生成脚本审计：

| Role | Runtime path | Bytes | Source / revision | License evidence | R0 state |
|---|---|---:|---|---|---|
| HuBERT v2 | `hubert_base/model.safetensors` | 378,298,232 | `lj1995/VoiceConversionWebUI` @ `e6d0c1a17da07c33557852f9dfa2bd44cc75737d`, converted from `hubert_base/pytorch_model.bin` | distribution repo MIT; upstream `facebook/hubert-base-ls960` Apache-2.0 | present; hash verified |
| FCPE | `fcpe/model.safetensors` | 43,339,596 | `torchfcpe==0.0.4`, bundled `fcpe_c_v001.pt` | exact PyPI wheel metadata and bundled MIT license | present; byte-identical replay verified |
| F0 VITS v2 40 kHz architecture/reference generator | `pretrained_v2/pretrained_v2/f0G40k.safetensors` | 145,897,024 | `lj1995/VoiceConversionWebUI` @ `e6d0c1a17da07c33557852f9dfa2bd44cc75737d`, converted from `pretrained_v2/f0G40k.pth` | repository metadata: MIT | present; hash verified; **not a target voice** |
| Representative target-speaker v2/40 kHz generator | — | — | no licensed target-speaker model/corpus selected | missing | **missing and R0-blocking** |
| Representative target-speaker Flat Index | — | — | no target-speaker corpus/revision selected | missing | **missing and R0-blocking** |

完整 SHA-256、原始 checkpoint hashes、official URLs、脚本 hashes 和未验证说明都在 manifest，
不要从表中省略的信息反向推断许可证或来源。

## Integrity and qualification checks

从仓库根目录验证所有当前存在的 R0 文件：

```bash
jq -r '.assets[] | select(.status == "present") | "\(.sha256)  \(.path)"' \
  specs/006-realtime-voice-conversion/r0-assets.json |
  sha256sum --check --strict
```

资格检查当前必须非零退出；只有补齐同一份有授权语料产生的真实目标声线 model+Index pair
及 corpus manifest 后，经过 review 才能将 manifest 的 `qualification_status` 改为 `ready`：

```bash
jq -e '.qualification_status == "ready" and
  (.qualification_blockers | length == 0)' \
  specs/006-realtime-voice-conversion/r0-assets.json
```

## Acquisition and conversion facts

`tools/download_models.py` 将 RVC 模型仓库固定在 revision
`e6d0c1a17da07c33557852f9dfa2bd44cc75737d`，用 `torch.load(weights_only=True)` 读取一次性
checkpoint，展开 state dict、去除 `module.` 前缀、转为 float32 后写 safetensors。manifest
审计的既有 HuBERT 转换输入是历史 nested layout 下的
`hubert_base/hubert_base/pytorch_model.bin`；v2 转换输入仍是
`pretrained_v2/pretrained_v2/f0G40k.pth`。manifest 已验证这些既有输入与官方 remote
bytes/hash 相同。

```bash
uv run --project tools --locked python tools/download_models.py
```

该命令只在目标 safetensors 不存在时转换；它不是“覆盖后自动验证”的命令。转换后仍须运行
manifest hash check。历史生成时的 Python package versions 未另存，审计未为节省记录完整性而
猜测它们。

`download_hubert()` 当前 clean-driver 路径使用 `MODELS_DIR` 作为 Hugging Face local root，
并直接把 downloader 返回的 `hubert_base/pytorch_model.bin` 作为转换输入；mocked
clean-driver contract 已证明新下载不会再产生重复的 `hubert_base/hubert_base` 层级。上述
legacy nested 文件仍用于本轮 provenance 审计，不能据此反推当前 driver 仍会生成 nested
路径。大型模型输出本轮没有重新生成，既有 output hash 仍以 manifest 为准。

`tools/export_fcpe.py` 明确要求 `torchfcpe==0.0.4`，在加载前检查 bundled
`fcpe_c_v001.pt` SHA-256，并强制 PyTorch restricted weights-only loader。2026-08-10 从
PyPI 精确 wheel（40,222,012 bytes，SHA-256
`f042c463d850d76c6f4899a0b84f0b694bb560adf05f4de951097a756d17472d`）提取 checkpoint
（43,363,260 bytes，SHA-256
`b9aeaeb673436eeda50ceafd632aa681aa63417e52eae4207503d180c9b10015`），构造模型、加载 state
dict 并写 safetensors；结果与本地 `fcpe/model.safetensors` 逐字节相同。该 wheel 的 METADATA
与随包 LICENSE 均声明 MIT。`torchfcpe==0.0.4` 已固定在 `tools/pyproject.toml` 与
`tools/uv.lock`；复现命令为：

```bash
uv run --project tools --locked python tools/export_fcpe.py
```

导出器在仓库根目录所在文件系统建立 staging；每个文件以 `os.replace` 发布，进程内任一后续
步骤失败时回滚已发布项并保留旧资产。这是 same-filesystem staged + rollback 的事务式发布，
不是 crash-consistent 的多文件原子事务。`torchfcpe` 仍只是锁定的一次性工具依赖，不进入
C++ runtime。

## Other local assets

当前还存在以下非首版 realtime-profile 资产：

- `rmvpe.safetensors`、`wavlm/model.safetensors`；
- `pretrained/pretrained/f0G40k.safetensors` 与 v1/v2 discriminator safetensors；
- `separation/*.safetensors`、对应 JSON/sidecar；
- `vad/silero_vad.safetensors` 与 JSON。

它们由 Spec 001/002/004 的可选 integration 使用，不受 realtime R0 manifest 完整覆盖。除非各自
README/manifest 已保存 source revision、license、input/output hashes 和 generation command，
不得把本目录中文件存在解释成来源合规、数值已验证或可发布。

## Target model and Index requirement

仓库当前没有 provenance-tracked 的目标声线 v2/40 kHz model，也没有真实 `.index` 文件。
Web contract 中创建的 `model.index` 是临时假产物，只验证事务与 argv，不是检索质量资产。
R0 必须选择有明确授权的目标声线语料，记录每条 utterance 及哈希，训练或取得兼容的目标
generator，并以 v2-compatible HuBERT 特征生成匹配 Flat Index；随后保存 model/index
path/bytes/SHA-256、speaker/config、Index rows/dim/`k`/`index_rate` 和完整命令。在此之前
Index-enabled latency/quality gate 保持 blocked。
