#!/usr/bin/env python3
"""Export torchfcpe's bundled FCPE model for the C++/CUDA port.

Runs the reference (torchfcpe) to:
  1. print the exact architecture (mel front-end config, conformer dims/depth,
     decoder cent-table) so we can mirror it in C++;
  2. dump the model weights to models/fcpe/model.safetensors;
  3. generate a reference F0 (+ intermediate mel) on the real speech fixture
     for numerical alignment (Constitution VI: F0 < 0.5 Hz).

Run:
  uv run --project tools --locked python tools/export_fcpe.py
"""

import hashlib
import importlib.metadata
import os
import json
import struct
import tempfile
import numpy as np
import torch
import soundfile as sf

# The audited bundled checkpoint contains only tensors/config primitives. Force
# PyTorch's restricted loader even though torchfcpe 0.0.4 does not pass the
# weights_only argument itself.
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "fcpe")
FIXTURE_WAV = os.path.join(FIXTURES_DIR, "speech_librispeech.wav")
TORCHFCPE_VERSION = "0.0.4"
TORCHFCPE_BUNDLED_CHECKPOINT_SHA256 = (
    "b9aeaeb673436eeda50ceafd632aa681aa63417e52eae4207503d180c9b10015"
)
FCPE_SAFETENSORS_SHA256 = (
    "6590a66a231e10ff3d52f8d90a6fb3df603ce76ccb149c93ae6f77ae62ec7167"
)


def verify_torchfcpe_provenance():
    """Fail before import/model loading unless the audited package asset matches."""
    distribution = importlib.metadata.distribution("torchfcpe")
    if distribution.version != TORCHFCPE_VERSION:
        raise RuntimeError(
            f"torchfcpe {TORCHFCPE_VERSION} is required; got {distribution.version}"
        )
    checkpoint = distribution.locate_file("torchfcpe/assets/fcpe_c_v001.pt")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != TORCHFCPE_BUNDLED_CHECKPOINT_SHA256:
        raise RuntimeError("torchfcpe bundled checkpoint SHA-256 mismatch")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_outputs(staged_outputs):
    """Replace all generated files, rolling back every prior replacement on error."""
    with tempfile.TemporaryDirectory(prefix=".fcpe-publish-", dir=PROJECT_ROOT) as txn:
        moved = []
        installed = []
        try:
            for index, (staged, target) in enumerate(staged_outputs):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                if os.path.lexists(target):
                    if os.path.islink(target) or not os.path.isfile(target):
                        raise RuntimeError(f"refusing to replace non-regular output: {target}")
                    backup = os.path.join(txn, f"backup-{index}")
                    os.replace(target, backup)
                    moved.append((target, backup))
                os.replace(staged, target)
                installed.append(target)
        except Exception:
            for target in reversed(installed):
                if os.path.lexists(target):
                    os.unlink(target)
            for target, backup in reversed(moved):
                if os.path.lexists(backup):
                    os.replace(backup, target)
            raise


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
    verify_torchfcpe_provenance()
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

    with tempfile.TemporaryDirectory(prefix=".fcpe-export-", dir=PROJECT_ROOT) as stage:
        staged_models = os.path.join(stage, "models")
        staged_fixtures = os.path.join(stage, "fixtures")
        os.makedirs(staged_models)
        os.makedirs(staged_fixtures)

        # Dump weights to safetensors. Any failure propagates and leaves every
        # previously published asset untouched.
        from safetensors.torch import save_file
        cpu_sd = {k: v.contiguous().cpu() for k, v in sd.items()}
        staged_model = os.path.join(staged_models, "model.safetensors")
        save_file(cpu_sd, staged_model)
        if sha256_file(staged_model) != FCPE_SAFETENSORS_SHA256:
            raise RuntimeError("generated FCPE safetensors SHA-256 mismatch")

        # Save the cent table for the C++ decoder.
        staged_cent = os.path.join(staged_models, "cent_table.bin")
        save_bin1d(staged_cent, ct)

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
        staged_f0 = os.path.join(staged_fixtures, "fcpe_ref_f0.bin")
        staged_mel = os.path.join(staged_fixtures, "fcpe_ref_mel.bin")
        save_bin1d(staged_f0, f0np)
        save_bin1d(staged_mel, mel.detach().float().cpu().numpy())
        meta = {"sr": int(sr), "mel_shape": list(mel.shape),
                "f0_frames": int(f0np.size),
                "out_dims": int(getattr(model, "out_dims", -1))}
        staged_meta = os.path.join(staged_models, "meta.json")
        with open(staged_meta, "w", encoding="utf-8") as output:
            json.dump(meta, output, indent=2)

        publish_outputs([
            (staged_model, os.path.join(MODELS_DIR, "model.safetensors")),
            (staged_cent, os.path.join(MODELS_DIR, "cent_table.bin")),
            (staged_meta, os.path.join(MODELS_DIR, "meta.json")),
            (staged_f0, os.path.join(FIXTURES_DIR, "fcpe_ref_f0.bin")),
            (staged_mel, os.path.join(FIXTURES_DIR, "fcpe_ref_mel.bin")),
        ])
    print("\nwrote", os.path.join(MODELS_DIR, "model.safetensors"))
    print("wrote reference F0/mel to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
