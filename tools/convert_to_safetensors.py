#!/usr/bin/env python3
"""
Convert PyTorch .pth / .pt checkpoints to safetensors format.

Usage:
    python convert_to_safetensors.py <input.pth> <output.safetensors>

This is a one-time Python utility. The C++ engine loads safetensors directly
via zero-copy mmap (no Python runtime needed at inference time).
"""
import argparse
import sys
from pathlib import Path

import torch
import safetensors


def convert_checkpoint(input_path: str, output_path: str):
    """Convert a PyTorch checkpoint to safetensors format."""
    print(f"Loading {input_path}...")
    state_dict = torch.load(input_path, map_location="cpu")

    # Handle wrapped state dicts (e.g., {"model": {...}, "sd": {...}})
    if isinstance(state_dict, dict):
        # Flatten nested dicts
        tensors = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        tensors[f"{key}.{k}"] = v
    else:
        tensors = state_dict

    # Convert to CPU float32
    tensors = {k: v.cpu().float() for k, v in tensors.items()}

    print(f"Saving {len(tensors)} tensors to {output_path}...")
    safetensors.torch.save_file(tensors, output_path)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch to safetensors")
    parser.add_argument("input", help="Input .pth or .pt file")
    parser.add_argument("output", help="Output .safetensors file")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    convert_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
