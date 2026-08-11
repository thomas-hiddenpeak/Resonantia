#!/usr/bin/env python3
"""One-off (uv): convert MelBand-RoFormer (de-reverb) -> F32 safetensors + band map
+ staged alignment reference, using the version-matched MSST model code.

Constitution: Python only here in tools/ for offline conversion + numerical-alignment
reference. The C++/CUDA runtime loads the safetensors directly (spec 004 S6).

Run:
  uv run --with torch --with numpy --with librosa --with einops --with beartype \
      --with rotary-embedding-torch --with safetensors --with pyyaml \
      --with huggingface-hub convert_roformer_weights.py
"""
import json
import hashlib
import os
import sys
import urllib.request

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roformer_ref")
MSST_REVISION = "e247dfe4abc1f17c69dff719207fe045dc04413a"
MSST_FILES = {
    "mel_band_roformer.py": "3b4a57ab268933900172e05fb76dd9e8acc1eb770c745d583a4e42b94b79ec15",
    "attend.py": "0459d799ade55541df2994b0becf7aec12214491360c5a06e346f6d615eaed15",
}


def _bootstrap_msst_code():
    """Fetch and verify the exact reviewed MSST source revision."""
    base = ("https://raw.githubusercontent.com/ZFTurbo/"
            f"Music-Source-Separation-Training/{MSST_REVISION}/models/bs_roformer")
    pkg = os.path.join(REF_DIR, "models", "bs_roformer")
    os.makedirs(pkg, exist_ok=True)
    for d in (os.path.join(REF_DIR, "models"), pkg):
        open(os.path.join(d, "__init__.py"), "a").close()
    for fn, expected_sha256 in MSST_FILES.items():
        dst = os.path.join(pkg, fn)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(f"{base}/{fn}", dst)
        with open(dst, "rb") as source:
            actual_sha256 = hashlib.sha256(source.read()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"refusing to import {dst}: SHA-256 {actual_sha256} does not match "
                f"the pinned source {expected_sha256}"
            )


_bootstrap_msst_code()
sys.path.insert(0, REF_DIR)
from models.bs_roformer.mel_band_roformer import MelBandRoformer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models", "separation")
FIX = os.path.join(ROOT, "tests", "fixtures", "separation")

# name -> (out_name, hf_repo, ckpt_path, yaml_path). All are MelBand-RoFormer
# checkpoints loadable by the version-matched MSST model code.
REGISTRY = {
    "dereverb": ("dereverb_roformer", "anvuew/dereverb_mel_band_roformer",
                 "archive only/8_256_6/deverb_mel_band_roformer_8_256dim_6depth.ckpt",
                 "archive only/8_256_6/deverb_mel_band_roformer_8_256dim_6depth.yaml"),
    "vocals": ("vocal_roformer", "becruily/mel-band-roformer-vocals",
               "mel_band_roformer_vocals_becruily.ckpt", "config_vocals_becruily.yaml"),
    "denoise": ("denoise_roformer", "poiqazwsx/melband-roformer-denoise",
                "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
                "model_mel_band_roformer_denoise.yaml"),
    # Removes echo AND reverb; outputs the dry stem (num_stems=1). CC-BY-NC-SA-4.0.
    "deecho": ("deecho_roformer", "Sucial/Dereverb-Echo_Mel_Band_Roformer",
               "dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt",
               "config_dereverb_echo_mbr_v2.yaml"),
    # num_stems=2 (Vocals/Instrumental); output stem 0 (lead vocal) for de-harmony.
    "karaoke": ("karaoke_roformer", "becruily/mel-band-roformer-karaoke",
                "mel_band_roformer_karaoke_becruily.ckpt", "config_karaoke_becruily.yaml", 0),
}
REVISIONS = {
    "dereverb": "cef05ad2b5b3145ea5c149d3ad5d1f8439b34d06",
    "vocals": "af457f56e56eb23fa8322929eef6a63b455e5858",
    "denoise": "4e39bc34a36dda8e73254cd8f5d44f15de2bd7b9",
    "deecho": "9c83bf27196213e6107cf0de5ff2d06d56a46876",
    "karaoke": "0c149975cfaa261c7d87baf54330a9da85bcf888",
}
# Registry key -> (pipeline stage/task, human label). Written into <name>.json so
# the C++ SepModelRegistry (spec 004 S7) can offer the model as a per-stage choice.
TASK_INFO = {
    "vocals": ("separate", "人声分离 · MelBand-RoFormer"),
    "karaoke": ("deharmony", "去和声 · Karaoke RoFormer"),
    "dereverb": ("dereverb", "去混响 · MelBand-RoFormer"),
    "deecho": ("deecho", "去回声 · MelBand-RoFormer"),
    "denoise": ("denoise", "去噪 · MelBand-RoFormer"),
}
VALID = {
    "dim", "depth", "stereo", "num_stems", "time_transformer_depth",
    "freq_transformer_depth", "linear_transformer_depth", "num_bands", "dim_head",
    "heads", "attn_dropout", "ff_dropout", "flash_attn", "dim_freqs_in",
    "sample_rate", "stft_n_fft", "stft_hop_length", "stft_win_length",
    "stft_normalized", "mask_estimator_depth",
}


def dump(path, arr):
    np.asarray(arr, dtype=np.float32).ravel().tofile(path)


def main():
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(FIX, exist_ok=True)
    key = sys.argv[1] if len(sys.argv) > 1 else "dereverb"
    if key not in REGISTRY:
        raise SystemExit(f"unknown model '{key}'; known: {list(REGISTRY)}")
    entry = REGISTRY[key]
    revision = REVISIONS[key]
    name, REPO, CKPT, YAML = entry[:4]
    target_stem = entry[4] if len(entry) > 4 else 0  # which stem to output (multi-stem)
    is_ref = key == "dereverb"  # only the anvuew dereverb model backs the alignment test
    cfg = yaml.safe_load(open(hf_hub_download(REPO, YAML, revision=revision)))
    mcfg = {k: v for k, v in cfg["model"].items() if k in VALID}
    mcfg.update(flash_attn=False, attn_dropout=0.0, ff_dropout=0.0)
    model = MelBandRoformer(**mcfg).eval()
    sd = torch.load(hf_hub_download(REPO, CKPT, revision=revision),
                    map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    print(f"[{name}] loaded cleanly ({len(sd)} tensors); missing buffers: {len(missing)}")

    # Weights -> F32 safetensors (clone to break shared rotary buffers).
    out_sd = {k: v.float().contiguous().clone() for k, v in model.state_dict().items()
              if v.is_floating_point()}
    save_file(out_sd, os.path.join(MODELS, f"{name}.safetensors"))

    # Band map (registered buffers, not persistent) needed by the C++ band split.
    fi = model.freq_indices.cpu().numpy().astype(np.int64)
    nbpf = model.num_bands_per_freq.cpu().numpy().astype(np.int64)
    dim_inputs = [2 * int(f) * (2 if mcfg["stereo"] else 1)
                  for f in model.num_freqs_per_band.tolist()]
    hp = {
        "n_fft": mcfg["stft_n_fft"], "hop": mcfg["stft_hop_length"],
        "win": mcfg["stft_win_length"], "sample_rate": mcfg["sample_rate"],
        "dim": mcfg["dim"], "depth": mcfg["depth"], "num_bands": mcfg["num_bands"],
        "dim_head": mcfg["dim_head"], "heads": mcfg["heads"], "stereo": mcfg["stereo"],
        "nb_output_bins": mcfg["dim_freqs_in"], "mask_estimator_depth": mcfg["mask_estimator_depth"],
        "freq_indices_len": int(fi.size), "dim_inputs": dim_inputs,
        "task": TASK_INFO.get(key, ("", ""))[0],
        "label": TASK_INFO.get(key, ("", name))[1] or name,
    }
    json.dump(hp, open(os.path.join(MODELS, f"{name}.json"), "w"), indent=2)
    fi.tofile(os.path.join(MODELS, f"{name}_freq_indices.i64"))
    nbpf.tofile(os.path.join(MODELS, f"{name}_num_bands_per_freq.i64"))
    np.asarray(dim_inputs, dtype=np.int64).tofile(os.path.join(MODELS, f"{name}_dim_inputs.i64"))
    # Runtime config for the C++ runner: [nfft,hop,dim,depth,num_bands,heads,dim_head,mask_depth,ff_mult,sr,chunk_size,target_stem]
    ff_mult = int(cfg["model"].get("mlp_expansion_factor", 4))
    # The CUDA runner uses full O(T^2) time attention. Four seconds keeps the
    # score tensor bounded on commodity GPUs; the runtime also clamps legacy
    # sidecars that were exported with substantially larger chunks.
    chunk_size = min(
        int(cfg.get("audio", {}).get("chunk_size", mcfg["sample_rate"] * 4)),
        int(mcfg["sample_rate"] * 4),
    )
    cfg_i64 = np.asarray([mcfg["stft_n_fft"], mcfg["stft_hop_length"], mcfg["dim"],
                          mcfg["depth"], mcfg["num_bands"], mcfg["heads"], mcfg["dim_head"],
                          mcfg["mask_estimator_depth"], ff_mult, mcfg["sample_rate"],
                          chunk_size, target_stem], dtype=np.int64)
    cfg_i64.tofile(os.path.join(MODELS, f"{name}_config.i64"))
    print(f"band map: freq_indices={fi.size}, bands={len(dim_inputs)}, "
          f"sum dim_inputs={sum(dim_inputs)}")
    if not is_ref:
        print(f"[{name}] converted (no staged reference; only 'dereverb' backs the test).")
        return

    # Staged reference on real audio (2 s, resampled 44100, stereo).
    import librosa
    y, _ = librosa.load(os.path.join(ROOT, "tests/fixtures/speech_librispeech.wav"),
                        sr=mcfg["sample_rate"], mono=True)
    y = y[: mcfg["sample_rate"] * 2]
    stereo = np.stack([y, y * 0.9], 0)
    audio = torch.from_numpy(stereo).float().unsqueeze(0)  # (1,2,T)

    caps = {}
    h = []
    h.append(model.band_split.register_forward_hook(
        lambda m, i, o: caps.__setitem__("band_split", o.detach())))
    h.append(model.band_split.register_forward_pre_hook(
        lambda m, i: caps.__setitem__("band_split_in", i[0].detach())))
    for i in range(mcfg["depth"]):
        ft = model.layers[i][1]  # freq transformer, output == x after block i
        h.append(ft.register_forward_hook(
            lambda m, inp, o, i=i: caps.__setitem__(f"block{i}", o.detach())))
    h.append(model.mask_estimators[0].register_forward_hook(
        lambda m, i, o: caps.__setitem__("mask", o.detach())))
    with torch.no_grad():
        recon = model(audio)  # (1,2,T)
    for x in h:
        x.remove()

    # torch STFT of channel 0 (precision reference for our cuFFT Stft).
    win = torch.hann_window(mcfg["stft_win_length"])
    s0 = torch.stft(torch.from_numpy(stereo[0]).float(), n_fft=mcfg["stft_n_fft"],
                    hop_length=mcfg["stft_hop_length"], win_length=mcfg["stft_win_length"],
                    window=win, center=True, normalized=False, return_complex=True)
    dump(os.path.join(FIX, "rof_stft_re.bin"), s0.real.numpy())
    dump(os.path.join(FIX, "rof_stft_im.bin"), s0.imag.numpy())

    dump(os.path.join(FIX, "rof_input_wave.bin"), stereo)
    dump(os.path.join(FIX, "rof_band_split_in.bin"), caps["band_split_in"].numpy())
    dump(os.path.join(FIX, "rof_band_split.bin"), caps["band_split"].numpy())
    dump(os.path.join(FIX, "rof_block_final.bin"), caps[f"block{mcfg['depth']-1}"].numpy())
    dump(os.path.join(FIX, "rof_mask.bin"), caps["mask"].numpy())
    dump(os.path.join(FIX, "rof_output_wave.bin"), recon.squeeze(0).numpy())
    ref = {"T": int(stereo.shape[1]),
           "band_split_shape": list(caps["band_split"].shape),
           "block_final_shape": list(caps[f"block{mcfg['depth']-1}"].shape),
           "mask_shape": list(caps["mask"].shape),
           "recon_shape": list(recon.shape)}
    json.dump(ref, open(os.path.join(FIX, "rof_ref.json"), "w"), indent=2)
    print("reference:", ref)
    print("recon range [%.4f, %.4f]" % (float(recon.min()), float(recon.max())))


if __name__ == "__main__":
    main()
