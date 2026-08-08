#!/usr/bin/env python3
"""One-off (uv): convert official Silero VAD v5 (MIT) -> F32 safetensors + config +
reference probs, for the pure-C++/CUDA runner (tools/-only, zero Python at runtime).

Reimplemented from the OFFICIAL snakers4/silero-vad (MIT); NOT derived from Orator.

Run:
  uv run --with silero-vad --with torch --with numpy --with soundfile --with librosa \
      convert_vad_weights.py
"""
import json
import os

import numpy as np
import soundfile as sf
import librosa
import torch
from silero_vad import load_silero_vad
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAD_DIR = os.path.join(ROOT, "models", "vad")
FIX = os.path.join(ROOT, "tests", "fixtures", "vad")
CHUNK, CONTEXT, NFFT, HOP = 512, 64, 256, 128


def main():
    os.makedirs(VAD_DIR, exist_ok=True)
    os.makedirs(FIX, exist_ok=True)
    inner = load_silero_vad(onnx=False)._model
    sd = inner.state_dict()
    keys = ["stft.forward_basis_buffer",
            "encoder.0.reparam_conv.weight", "encoder.0.reparam_conv.bias",
            "encoder.1.reparam_conv.weight", "encoder.1.reparam_conv.bias",
            "encoder.2.reparam_conv.weight", "encoder.2.reparam_conv.bias",
            "encoder.3.reparam_conv.weight", "encoder.3.reparam_conv.bias",
            "decoder.rnn.weight_ih", "decoder.rnn.weight_hh",
            "decoder.rnn.bias_ih", "decoder.rnn.bias_hh",
            "decoder.decoder.2.weight", "decoder.decoder.2.bias"]
    out = {k: sd[k].float().contiguous().clone() for k in keys}
    save_file(out, os.path.join(VAD_DIR, "silero_vad.safetensors"))
    json.dump({"sample_rate": 16000, "chunk": CHUNK, "context": CONTEXT,
               "n_fft": NFFT, "hop": HOP}, open(os.path.join(VAD_DIR, "silero_vad.json"), "w"), indent=2)
    print("saved", len(out), "VAD tensors")

    # Reference: real 16k audio -> per-chunk speech probs (context+state carried).
    y, sr = sf.read(os.path.join(ROOT, "tests/fixtures/speech_librispeech.wav"))
    if y.ndim > 1:
        y = y[:, 0]
    x = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else y.astype(np.float32)
    n = (len(x) // CHUNK) * CHUNK
    x = x[:n]
    ctx = torch.zeros(1, CONTEXT); state = torch.zeros(0); probs = []
    with torch.no_grad():
        for i in range(0, n, CHUNK):
            x1 = torch.cat([ctx, torch.from_numpy(x[i:i + CHUNK]).unsqueeze(0)], 1)
            p, state = inner(x1, state)
            probs.append(float(p.item()))
            ctx = x1[:, -CONTEXT:]
    x.astype(np.float32).tofile(os.path.join(FIX, "vad_input.bin"))
    np.asarray(probs, np.float32).tofile(os.path.join(FIX, "vad_probs.bin"))
    json.dump({"samples": int(n), "chunks": len(probs)}, open(os.path.join(FIX, "vad_ref.json"), "w"))
    print("reference: %d samples, %d chunks, max %.3f mean %.3f" %
          (n, len(probs), max(probs), float(np.mean(probs))))


if __name__ == "__main__":
    main()
