#!/usr/bin/env python3
"""Export torchfcpe's bundled FCPE model for the C++/CUDA port.

Runs the reference (torchfcpe) to:
  1. print the exact architecture (mel front-end config, conformer dims/depth,
     decoder cent-table) so we can mirror it in C++;
  2. dump the model weights to models/fcpe/model.safetensors;
  3. generate a reference F0 (+ intermediate mel) on the real speech fixture
     for numerical alignment (Constitution VI: F0 < 0.5 Hz).

Run:
  uv run --with torchfcpe --with soundfile --with safetensors --with numpy \
         --with torch tools/export_fcpe.py
"""

import os
import json
import struct
import numpy as np
import torch
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "fcpe")
FIXTURE_WAV = os.path.join(FIXTURES_DIR, "speech_librispeech.wav")


def save_bin1d(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32).ravel())
    with open(path, "wb") as f:
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", arr.size))
        f.write(arr.tobytes())


def describe(obj, prefix=""):
    for name, mod in obj.named_children():
        p = f"{prefix}{name}: {mod.__class__.__name__}"
        extra = ""
        for attr in ("in_features", "out_features", "num_heads", "hidden_size",
                     "embed_dim", "kernel_size", "num_layers", "dim", "n_layers"):
            if hasattr(mod, attr):
                extra += f" {attr}={getattr(mod, attr)}"
        print(p + extra)
        describe(mod, prefix + "  ")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    from torchfcpe import spawn_bundled_infer_model

    infer = spawn_bundled_infer_model("cpu")
    model = infer.model
    wav2mel = infer.wav2mel

    print("=== FCPE infer_model attrs ===")
    print("model class:", model.__class__.__name__)
    print("out_dims:", int(getattr(model, "out_dims", -1)))
    ct = model.cent_table.detach().float().cpu().numpy()
    print("cent_table shape:", ct.shape, "min", ct.min(), "max", ct.max(),
          "first5", ct[:5], "last5", ct[-5:])

    print("\n=== wav2mel config ===")
    for attr in ("sampling_rate", "sr", "n_mels", "num_mels", "n_fft", "win_size",
                 "hop_size", "hop_length", "fmin", "fmax", "clip_val", "mel_type"):
        if hasattr(wav2mel, attr):
            print(f"  wav2mel.{attr} = {getattr(wav2mel, attr)}")
    if hasattr(wav2mel, "__dict__"):
        for k, v in wav2mel.__dict__.items():
            if not k.startswith("_") and isinstance(v, (int, float, str, bool)):
                print(f"  wav2mel[{k}] = {v}")

    print("\n=== model architecture ===")
    describe(model)

    print("\n=== state_dict (name: shape) ===")
    sd = model.state_dict()
    for k, v in sd.items():
        print(f"  {k}: {tuple(v.shape)}")

    # Dump weights to safetensors.
    try:
        from safetensors.torch import save_file
        cpu_sd = {k: v.contiguous().cpu() for k, v in sd.items()}
        save_file(cpu_sd, os.path.join(MODELS_DIR, "model.safetensors"))
        print("\nwrote", os.path.join(MODELS_DIR, "model.safetensors"))
    except Exception as e:  # noqa: BLE001
        print("safetensors dump failed:", e)

    # Save the cent table for the C++ decoder.
    save_bin1d(os.path.join(MODELS_DIR, "cent_table.bin"), ct)

    # Reference F0 + mel on the real fixture.
    wav, sr = sf.read(FIXTURE_WAV, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav).unsqueeze(0)
    with torch.no_grad():
        mel = wav2mel(wav_t, sr)
        print("\nmel shape:", tuple(mel.shape))
        f0 = infer.infer(wav_t, sr=sr, decoder_mode="local_argmax", threshold=0.006)
    f0np = f0.detach().float().cpu().numpy().ravel()
    print("f0 frames:", f0np.size, "voiced:", int((f0np > 0).sum()),
          "min", float(f0np[f0np > 0].min()) if (f0np > 0).any() else 0,
          "max", float(f0np.max()))
    save_bin1d(os.path.join(FIXTURES_DIR, "fcpe_ref_f0.bin"), f0np)
    save_bin1d(os.path.join(FIXTURES_DIR, "fcpe_ref_mel.bin"),
               mel.detach().float().cpu().numpy())
    meta = {"sr": int(sr), "mel_shape": list(mel.shape), "f0_frames": int(f0np.size),
            "out_dims": int(getattr(model, "out_dims", -1))}
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote reference F0/mel to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
