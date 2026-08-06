#!/usr/bin/env python3
"""Download RVC model weights and convert to safetensors format."""

import os
import sys
import json
import torch
from huggingface_hub import hf_hub_download

# Project root models directory (not tools/models)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

REPO_ID = "lj1995/VoiceConversionWebUI"

def download_file(filename, local_dir=None):
    """Download a file from HuggingFace."""
    print(f"  Downloading: {filename}")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=local_dir or MODELS_DIR
    )
    size = os.path.getsize(path) / 1024 / 1024
    print(f"    {size:.1f} MB -> {path}")
    return path

def flatten_dict(d, prefix=""):
    """Flatten nested OrderedDict to flat dict with dotted keys."""
    from collections import OrderedDict
    items = []
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, OrderedDict) or (isinstance(v, dict) and any(isinstance(val, torch.Tensor) for val in v.values())):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def convert_state_dict(pt_path, sf_path):
    """Convert PyTorch state dict to safetensors."""
    from safetensors.torch import save_file
    from collections import OrderedDict
    
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    
    # Extract state_dict if wrapped
    if hasattr(obj, 'state_dict'):
        state_dict = obj.state_dict()
    elif isinstance(obj, (dict, OrderedDict)):
        state_dict = obj
    else:
        print(f"    WARNING: Unknown format for {pt_path}, skipping conversion")
        return None
    
    # Flatten nested dicts (VITS models have nested OrderedDict)
    if any(isinstance(v, (dict, OrderedDict)) for v in state_dict.values()):
        state_dict = flatten_dict(state_dict)
    
    # Clean keys (remove 'module.' prefix) and convert to float32
    clean_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "")
        if isinstance(v, torch.Tensor):
            clean_dict[clean_k] = v.float()
        else:
            # Skip non-tensor values (e.g., strings, ints)
            print(f"    SKIP non-tensor key: {k} (type: {type(v).__name__})")
    
    save_file(clean_dict, sf_path)
    size = os.path.getsize(sf_path) / 1024 / 1024
    print(f"    Converted: {sf_path} ({size:.1f} MB, {len(clean_dict)} tensors)")
    
    # Save metadata
    meta = {
        "shapes": {k: list(v.shape) for k, v in clean_dict.items()},
        "dtypes": {k: str(v.dtype) for k, v in clean_dict.items()}
    }
    with open(sf_path + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    
    return clean_dict

def download_hubert():
    """Download HuBERT model (transformers format)."""
    print("\n[1/4] HuBERT Base...")
    hubert_dir = os.path.join(MODELS_DIR, "hubert_base")
    os.makedirs(hubert_dir, exist_ok=True)
    
    # Download transformers-format model (not fairseq)
    for f in ["pytorch_model.bin", "config.json", "preprocessor_config.json"]:
        download_file(f"hubert_base/{f}", hubert_dir)
    
    # Convert to safetensors
    pt_path = os.path.join(hubert_dir, "pytorch_model.bin")
    sf_path = os.path.join(hubert_dir, "model.safetensors")
    if os.path.exists(pt_path) and not os.path.exists(sf_path):
        convert_state_dict(pt_path, sf_path)

def download_rmvpe():
    """Download RMVPE model."""
    print("\n[2/4] RMVPE...")
    pt_path = download_file("rmvpe.pt")
    
    # Try to convert (may fail due to custom classes)
    sf_path = pt_path.replace(".pt", ".safetensors")
    if not os.path.exists(sf_path):
        try:
            convert_state_dict(pt_path, sf_path)
        except Exception as e:
            print(f"    RMVPE conversion failed: {e}")
            print(f"    Keeping .pt format for now")

def download_pretrained_v2():
    """Download RVC v2 pretrained VITS weights."""
    print("\n[3/4] RVC v2 Pretrained (f0G40k + f0D40k)...")
    pretrained_dir = os.path.join(MODELS_DIR, "pretrained_v2")
    os.makedirs(pretrained_dir, exist_ok=True)
    
    for f in ["f0G40k.pth", "f0D40k.pth"]:
        pt_path = download_file(f"pretrained_v2/{f}", pretrained_dir)
        sf_path = pt_path.replace(".pth", ".safetensors")
        if not os.path.exists(sf_path):
            try:
                convert_state_dict(pt_path, sf_path)
            except Exception as e:
                print(f"    {f} conversion failed: {e}")

def download_pretrained_base():
    """Download base pretrained weights."""
    print("\n[4/4] Pretrained base models...")
    pretrained_dir = os.path.join(MODELS_DIR, "pretrained")
    os.makedirs(pretrained_dir, exist_ok=True)
    
    for f in ["f0G40k.pth", "f0D40k.pth"]:
        pt_path = download_file(f"pretrained/{f}", pretrained_dir)
        sf_path = pt_path.replace(".pth", ".safetensors")
        if not os.path.exists(sf_path):
            try:
                convert_state_dict(pt_path, sf_path)
            except Exception as e:
                print(f"    {f} conversion failed: {e}")

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Downloading models to: {MODELS_DIR}")
    
    download_hubert()
    download_rmvpe()
    download_pretrained_v2()
    download_pretrained_base()
    
    print("\n" + "="*60)
    print("Download complete! Models:")
    print("="*60)
    for root, dirs, files in os.walk(MODELS_DIR):
        for f in sorted(files):
            path = os.path.join(root, f)
            rel = os.path.relpath(path, MODELS_DIR)
            size = os.path.getsize(path) / 1024 / 1024
            print(f"  {rel} ({size:.1f} MB)")
