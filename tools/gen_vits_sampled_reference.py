#!/usr/bin/env python3
"""Offline upstream VITS reference with explicit, reproducible Gaussian inputs.

The random input recipe is specified independently of PyTorch's version-dependent
RNG. The upstream network computes sampling, NSF excitation, Flow and decoding.
Historical mean-mode fixtures are preserved.
"""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from gen_vits_reference import CONFIG, load_bin, save_bin
from infer.module.models import SynthesizerTrnMs768NSFsid

MASK = (1 << 64) - 1
PRIOR = 0x243F6A8885A308D3
NSF = 0x13198A2E03707344


def mix(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)


def gaussian(seed, stream, count):
    # Box-Muller pairs, open (0,1) midpoints from the top 32 hash bits.
    key = mix(seed ^ stream)
    pairs = np.arange((count + 1) // 2, dtype=np.uint64)
    u = np.array([((mix((key + 2 * int(i)) & MASK) >> 32) + 0.5) / 2**32 for i in pairs])
    v = np.array([((mix((key + 2 * int(i) + 1) & MASK) >> 32) + 0.5) / 2**32 for i in pairs])
    radius, angle = np.sqrt(-2 * np.log(u)), 2 * np.pi * v
    return np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=1).reshape(-1)[:count].astype(np.float32)


def main():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests/fixtures"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    weights = root / "models/pretrained_v2/pretrained_v2/f0G40k.safetensors"
    phone = torch.from_numpy(load_bin(fixture / "vits_ref_phone.bin").copy()).unsqueeze(0)
    f0 = torch.from_numpy(load_bin(fixture / "e2e_htk_nsff0.bin").copy()).reshape(1, -1)
    pitch = torch.from_numpy(load_bin(fixture / "e2e_htk_pitch.bin").copy()).long().reshape(1, -1)
    frames = phone.shape[1]
    assert frames == 584 and f0.shape == pitch.shape == (1, frames)
    prior = gaussian(0, PRIOR, frames * 192).reshape(frames, 192).T.copy()
    nsf = gaussian(0, NSF, frames * 400).reshape(1, frames * 400, 1)
    net = SynthesizerTrnMs768NSFsid(**CONFIG).float().eval()
    net.load_state_dict({k.removeprefix("model."): v for k, v in load_file(str(weights)).items()}, strict=True)
    net.remove_weight_norm()
    report = {"seed": 0, "prior_scale": 0.66666, "nsf_voiced_std": 0.003,
              "nsf_unvoiced_std": 0.1 / 3, "torch": torch.__version__, "numpy": np.__version__,
              "source": "speech_librispeech.wav / vits_ref_phone.bin / e2e_htk_nsff0.bin",
              "rng": "SplitMix64 counter hash; 32-bit open uniform midpoints; Box-Muller pairs; frame-major prior index",
              "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "upstream_sha256": hashlib.sha256((root / "tools/rvc_ref/infer/module/models.py").read_bytes()).hexdigest(),
              "inputs": {name: hashlib.sha256((fixture / name).read_bytes()).hexdigest()
                         for name in ("speech_librispeech.wav", "vits_ref_phone.bin",
                                      "e2e_htk_nsff0.bin", "e2e_htk_pitch.bin")},
              "outputs": {}}
    def nsf_draw(value, *args, **kwargs):
        assert tuple(value.shape) == tuple(nsf.shape)
        return torch.from_numpy(nsf.copy()).to(value)
    with torch.inference_mode():
        g = net.emb_g(torch.tensor([0])).unsqueeze(-1)
        mean, logs, mask = net.enc_p(phone, pitch, torch.tensor([frames]))
        zp = (mean + torch.exp(logs) * torch.from_numpy(prior).unsqueeze(0) * 0.66666) * mask
        z = net.flow(zp, mask, g=g, reverse=True)
        with patch("torch.randn_like", nsf_draw):
            har = net.dec.m_source(f0, 400)[0][0, :, 0].numpy()
            audio = net.dec(z * mask, f0, g=g)[0, 0].numpy()
        for name, data in (("prior_noise", prior), ("nsf_noise", nsf[0,:,0]),
                           ("har", har), ("z", z[0].numpy()), ("audio", audio)):
            assert np.isfinite(data).all()
            target = fixture / f"vits_sampled_{name}.bin"
            save_bin(target, data)
            report["outputs"][target.name] = {"shape": list(data.shape), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        sf.write(fixture / "vits_sampled_reference.wav", audio, 40000, subtype="FLOAT")
    (fixture / "vits_sampled_manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
