#!/usr/bin/env python3
"""Generate HuBERT reference outputs for C++ numerical alignment.

Loads the same safetensors weights used by C++, runs inference with transformers,
and saves intermediate + final outputs as binary files for comparison.

RVC HuBERT usage:
  - v1: output_layer=9, then final_proj (768 -> 256)
  - v2: output_layer=12, no projection (768-dim)
"""

import os
import struct
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

HUBERT_DIR = os.path.join(MODELS_DIR, "hubert_base", "hubert_base")


def save_tensor_binary(path, arr):
    """Save a numpy array as binary: [ndim][shape...][float32 data]."""
    arr = np.ascontiguousarray(arr.astype(np.float32))
    with open(path, "wb") as f:
        f.write(struct.pack("<i", arr.ndim))
        for s in arr.shape:
            f.write(struct.pack("<i", s))
        f.write(arr.tobytes())
    print(f"    Saved {path}: shape={list(arr.shape)}")


def generate_hubert_reference(audio_path, output_prefix, output_layer=12,
                              use_final_proj=False):
    """Run HuBERT and save reference outputs."""
    from transformers import HubertModel

    print(f"\nProcessing: {audio_path}")

    # Load audio (16kHz mono)
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    assert sr == 16000, f"Expected 16kHz, got {sr}"
    print(f"  Audio: {len(audio)} samples @ {sr} Hz")

    # Load HuBERT model (transformers format matches our safetensors)
    print(f"  Loading HuBERT from {HUBERT_DIR}...")
    model = HubertModel.from_pretrained(HUBERT_DIR, torch_dtype=torch.float32)
    model = model.float()
    model.eval()

    # RVC normalizes input: layer_norm over the raw waveform is NOT applied;
    # instead feats are used directly. transformers applies feature projection internally.
    input_values = torch.from_numpy(audio).unsqueeze(0)  # [1, T]

    with torch.no_grad():
        outputs = model(
            input_values,
            output_hidden_states=True,
            return_dict=True,
        )

    # hidden_states[0] = input embeddings (after feature projection + pos conv + layer norm)
    # hidden_states[i] = after transformer layer i
    hidden_states = outputs.hidden_states
    print(f"  Num hidden states: {len(hidden_states)}")

    # Extract the requested layer output
    layer_output = hidden_states[output_layer][0].cpu().numpy()  # [T, 768]
    print(f"  Layer {output_layer} output: {layer_output.shape}")

    # Save layer output (768-dim, v2)
    save_tensor_binary(f"{output_prefix}_layer{output_layer}.bin", layer_output)

    # Apply final projection for v1 (768 -> 256)
    if use_final_proj:
        from safetensors.torch import load_file
        sf_path = os.path.join(MODELS_DIR, "hubert_base", "model.safetensors")
        weights = load_file(sf_path)
        final_w = weights["final_proj.weight"]  # [256, 768]
        final_b = weights["final_proj.bias"]    # [256]
        layer_t = torch.from_numpy(layer_output)
        proj = F.linear(layer_t, final_w, final_b).numpy()  # [T, 256]
        print(f"  Final proj output: {proj.shape}")
        save_tensor_binary(f"{output_prefix}_final256.bin", proj)

    # Also save the feature extractor output (before transformer) for debugging
    # Run feature extractor separately
    with torch.no_grad():
        extract_features = model.feature_extractor(input_values)  # [1, 512, T']
        extract_features = extract_features.transpose(1, 2)       # [1, T', 512]
        print(f"  Feature extractor output: {extract_features.shape}")
        save_tensor_binary(f"{output_prefix}_featextract.bin",
                           extract_features[0].cpu().numpy())

        # Feature projection output
        proj_out = model.feature_projection(extract_features)
        proj_features = proj_out[0] if isinstance(proj_out, tuple) else proj_out
        print(f"  Feature projection output: {proj_features.shape}")
        save_tensor_binary(f"{output_prefix}_featproj.bin",
                           proj_features[0].cpu().numpy())

    return layer_output


if __name__ == "__main__":
    audio_path = os.path.join(FIXTURES_DIR, "speech_librispeech.wav")
    prefix = os.path.join(FIXTURES_DIR, "hubert_ref")

    print("=" * 60)
    print("Generating HuBERT reference outputs")
    print("=" * 60)

    # v2: layer 12, 768-dim
    generate_hubert_reference(audio_path, prefix, output_layer=12,
                              use_final_proj=False)

    # v1: layer 9, final_proj to 256
    generate_hubert_reference(audio_path, prefix + "_v1", output_layer=9,
                              use_final_proj=True)

    print("\nDone! Reference files:")
    for f in sorted(os.listdir(FIXTURES_DIR)):
        if f.startswith("hubert_ref") and f.endswith(".bin"):
            fp = os.path.join(FIXTURES_DIR, f)
            print(f"  {f}: {os.path.getsize(fp)/1024:.1f} KB")
