#!/usr/bin/env python3
"""Independent real-audio RMVPE/E2E fixtures from the upstream RVC implementation.

Uses the read-only tools/rvc_ref checkout and strictly loaded safetensors. Keeps
historical rmvpe_ref/vits_ref assets intact. Python is an offline validation tool.
Run: uv run --project tools --no-sync python tools/gen_rmvpe_htk_reference.py
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from gen_vits_reference import CONFIG, f0_to_coarse, load_bin, save_bin
from infer.module.models import SynthesizerTrnMs768NSFsid
from infer.rmvpe import E2E, MelSpectrogram, RMVPE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "tests/fixtures")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fixture = root / "tests/fixtures"
    source = fixture / "speech_librispeech.wav"
    weights = root / "models/rmvpe.safetensors"
    generator = root / "models/pretrained_v2/pretrained_v2/f0G40k.safetensors"
    upstream = root / "tools/rvc_ref/infer/rmvpe.py"
    if digest(source) != "799f78ed4beb4de7ceae3a809262d4ce242394342ccd1d58cef7d49dbc2def46":
        raise ValueError("unexpected real speech fixture")
    samples, rate = sf.read(source, dtype="float32")
    if rate != 16000 or samples.shape != (93680,):
        raise ValueError("expected mono 16 kHz LibriSpeech row 0")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    report = {
        "command": sys.argv, "torch": torch.__version__, "librosa": librosa.__version__,
        "source": str(source), "source_sha256": digest(source),
        "weights_sha256": digest(weights), "generator_sha256": digest(generator),
        "upstream": "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer/rmvpe.py",
        "upstream_file_sha256": digest(upstream),
        "upstream_synthesizer_sha256": digest(root / "tools/rvc_ref/infer/module/models.py"),
        "frontend": {"scale": "HTK", "normalization": "Slaney area", "n_fft": 1024,
                     "hop": 160, "mel_bands": 128, "fmin": 30, "fmax": 8000,
                     "center": True, "padding": "reflect", "clamp": 1e-5},
        "device": "cpu", "dtype": "float32", "threshold": 0.03,
        "outputs": {},
    }
    with torch.inference_mode():
        model = E2E(4, 1, (2, 2)).float().eval()
        model.load_state_dict(load_file(str(weights)), strict=True)
        frontend = MelSpectrogram(False, 128, 16000, 1024, 160, None, 30, 8000)
        mel = frontend(torch.from_numpy(samples).unsqueeze(0))
        frames = mel.shape[-1]
        salience = model(F.pad(mel, (0, (-frames) % 32)))[0, :frames].numpy()
        # Only use the official decoder methods; avoid its pickle-loading constructor.
        decoder = object.__new__(RMVPE)
        decoder.cents_mapping = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))
        f0 = decoder.decode(salience, thred=0.03).astype(np.float32)
        for name, array in (("mel", mel[0].numpy()), ("salience", salience), ("f0", f0),
                            ("basis", frontend.mel_basis.numpy())):
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite reference: {name}")
            path = out / f"rmvpe_htk_{name}.bin"
            save_bin(path, array)
            report["outputs"][path.name] = {"shape": list(array.shape), "sha256": digest(path)}
        old_path = fixture / "rmvpe_ref_f0.bin"
        if old_path.exists():
            old = load_bin(old_path).reshape(-1)
            if old.shape != f0.shape:
                raise ValueError("historical F0 comparison has a different shape")
            joint = (f0 > 0) & (old > 0)
            report["historical_slaney_comparison"] = {
                "joint_voiced": int(joint.sum()), "voicing_agreement": float(np.mean((f0 > 0) == (old > 0))),
                "median_abs_cents": (float(np.median(np.abs(1200 * np.log2(f0[joint] / old[joint]))))
                                     if joint.any() else None),
                "htk_voiced": int((f0 > 0).sum()), "slaney_voiced": int((old > 0).sum()),
            }
        del model, frontend, decoder
        net = SynthesizerTrnMs768NSFsid(**CONFIG).float().eval()
        net.load_state_dict({k.removeprefix("model."): v for k, v in load_file(str(generator)).items()}, strict=True)
        net.remove_weight_norm()
        features_path = fixture / "hubert_ref_layer12.bin"
        report["hubert_features_sha256"] = digest(features_path)
        phone = torch.from_numpy(load_bin(features_path).copy()).unsqueeze(0)
        phone = F.interpolate(phone.transpose(1, 2), scale_factor=2, mode="nearest").transpose(1, 2)
        count = min(phone.shape[1], len(f0))
        phone = phone[:, :count]
        nsff0 = torch.from_numpy(f0[:count].copy()).unsqueeze(0)
        pitch = torch.from_numpy(f0_to_coarse(f0[:count].copy())).unsqueeze(0)
        g = net.emb_g(torch.tensor([0])).unsqueeze(-1)
        mean, _, mask = net.enc_p(phone, pitch, torch.tensor([count]))
        latent = net.flow(mean * mask, mask, g=g, reverse=True)
        with patch("torch.randn_like", lambda x, *a, **kw: torch.zeros_like(x)):
            audio = net.dec(latent * mask, nsff0, g=g)[0, 0].numpy()
        if audio.shape != (count * 400,) or not np.isfinite(audio).all():
            raise ValueError("invalid reference waveform")
        for name, array in (("audio", audio), ("nsff0", f0[:count]), ("pitch", pitch[0].numpy())):
            path = out / f"e2e_htk_{name}.bin"
            save_bin(path, array)
            report["outputs"][path.name] = {"shape": list(array.shape), "sha256": digest(path)}
        sf.write(out / "e2e_htk_reference.wav", audio, 40000, subtype="FLOAT")
        report["e2e"] = {"frames": count, "seconds": len(audio) / 40000,
                         "nsf_noise": False, "prior_noise": False, "speaker": 0,
                         "highpass": False, "edge_pad": 0, "index": 0, "rms_mix": 1,
                         "limitation": "Numerical alignment, not CyberTom or human quality acceptance"}
    (out / "rmvpe_htk_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report.get("historical_slaney_comparison", {})), flush=True)


if __name__ == "__main__":
    main()
