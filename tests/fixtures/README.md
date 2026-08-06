# Test Fixtures

This directory contains reference data for numerical alignment tests.

## Expected Files

| 文件 | 说明 | 来源 |
|------|------|------|
| `hubert_256.npy` | HuBERT v1 参考输出 (256-dim) | Python RVC |
| `hubert_768.npy` | HuBERT v2 参考输出 (768-dim) | Python RVC |
| `wavlm_768.npy` | WavLM-Base+ 参考输出 (768-dim) | Python RVC |
| `rmvpe_f0.npy` | RMVPE F0 参考输出 | Python RVC |
| `vits_audio.wav` | VITS 合成参考音频 | Python RVC |

## Generation Script

Use `tools/generate_fixtures.py` to generate reference data from Python RVC:

```bash
cd tools
uv run python generate_fixtures.py --input tests/fixtures/test_input.wav --output tests/fixtures/
```

## Git LFS

Large fixture files should use Git LFS:

```bash
git lfs track "*.npy"
git lfs track "*.wav"
```
