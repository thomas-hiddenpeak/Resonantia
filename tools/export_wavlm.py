#!/usr/bin/env python3
"""Export microsoft/wavlm-base-plus for the C++/CUDA content-encoder port.

WavLM-Base+ shares the wav2vec2 backbone with our HuBERT encoder; the one new
piece is the *gated relative position bias* in self-attention. This script:
  1. prints the exact config + attention source so we can mirror it in C++;
  2. dumps weights to models/wavlm/model.safetensors;
  3. exports reference hidden states (raw audio, no normalization — matching the
     C++ path) on the real speech fixture for numerical alignment.

Run:
  uv run --with transformers --with torch --with soundfile --with safetensors \
         --with numpy tools/export_wavlm.py
"""

import os
import json
import struct
import inspect
import numpy as np
import torch
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "wavlm")
FIXTURE_WAV = os.path.join(FIXTURES_DIR, "speech_librispeech.wav")
MODEL_ID = "microsoft/wavlm-base-plus"


def save_bin(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    with open(path, "wb") as f:
        f.write(struct.pack("<i", arr.ndim))
        for s in arr.shape:
            f.write(struct.pack("<i", int(s)))
        f.write(arr.tobytes())


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    from transformers import WavLMModel

    model = WavLMModel.from_pretrained(MODEL_ID)
    model.eval()
    cfg = model.config

    print("=== WavLM config ===")
    for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_buckets", "max_bucket_distance", "conv_dim", "conv_kernel",
              "conv_stride", "conv_bias", "feat_extract_norm",
              "do_stable_layer_norm", "num_conv_pos_embeddings",
              "num_conv_pos_embedding_groups", "intermediate_size",
              "hidden_act", "layer_norm_eps"):
        print(f"  {k} = {getattr(cfg, k, None)}")

    print("\n=== WavLMAttention.forward ===")
    try:
        attn = model.encoder.layers[0].attention
        print(inspect.getsource(type(attn).forward))
    except Exception as e:  # noqa: BLE001
        print("  (source unavailable)", e)

    print("\n=== attention-related params (layer 0) ===")
    for name, p in model.encoder.layers[0].named_parameters():
        if any(t in name for t in ("rel_attn", "gru_rel", "grep")):
            print(f"  {name}: {tuple(p.shape)}")

    print("\n=== state_dict (first 40 + attention keys) ===")
    sd = model.state_dict()
    keys = list(sd.keys())
    for k in keys[:40]:
        print(f"  {k}: {tuple(sd[k].shape)}")
    print("  ... attention keys ...")
    for k in keys:
        if "layers.0." in k and ("attention" in k or "rel" in k):
            print(f"  {k}: {tuple(sd[k].shape)}")

    # Dump weights.
    try:
        from safetensors.torch import save_file
        save_file({k: v.contiguous().cpu() for k, v in sd.items()},
                  os.path.join(MODELS_DIR, "model.safetensors"))
        print("\nwrote", os.path.join(MODELS_DIR, "model.safetensors"))
    except Exception as e:  # noqa: BLE001
        print("safetensors dump failed:", e)

    # Reference hidden states on the real fixture (raw audio, no normalization).
    audio, sr = sf.read(FIXTURE_WAV, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    iv = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        out = model(iv, output_hidden_states=True)
    hs = out.hidden_states  # tuple: [0]=pos-conv out, [i]=after layer i
    last = hs[len(hs) - 1][0].cpu().numpy()  # [T, 768]
    print("\nlast hidden:", last.shape, "mean", float(last.mean()), "std", float(last.std()))
    save_bin(os.path.join(FIXTURES_DIR, "wavlm_ref_last.bin"), last)
    save_bin(os.path.join(FIXTURES_DIR, "wavlm_ref_l9.bin"), hs[9][0].cpu().numpy())
    meta = {"sr": int(sr), "T": int(last.shape[0]), "dim": int(last.shape[1]),
            "num_layers": int(cfg.num_hidden_layers),
            "num_heads": int(cfg.num_attention_heads),
            "num_buckets": int(cfg.num_buckets),
            "max_bucket_distance": int(cfg.max_bucket_distance)}
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote reference hidden states to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
