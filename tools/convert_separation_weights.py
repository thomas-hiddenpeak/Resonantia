#!/usr/bin/env python3
"""One-off (uv): convert Open-Unmix umxhq vocals -> F32 safetensors + reference.

Constitution: Python lives ONLY here in tools/ for offline weight conversion and
numerical-alignment reference generation. The C++/CUDA runtime loads the
safetensors directly (spec 004 S2, zero Python at runtime).

Run:
  uv run --with openunmix --with torch --with numpy --with librosa \
      convert_separation_weights.py

Outputs:
  models/separation/umxhq_vocals.safetensors   (F32 weights)
  models/separation/umxhq_vocals.json          (hyperparameters)
  tests/fixtures/separation/umx_*.bin          (staged alignment reference)
  tests/fixtures/separation/umx_ref.json       (reference shapes)
"""
import json
import os

import numpy as np
import torch
from openunmix import umxhq
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models", "separation")
FIX = os.path.join(ROOT, "tests", "fixtures", "separation")
N_FFT, HOP, SR = 4096, 1024, 44100


def dump_bin(path: str, arr: np.ndarray) -> None:
    arr.astype(np.float32).ravel().tofile(path)


def main() -> None:
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(FIX, exist_ok=True)

    sep = umxhq(targets=["vocals"], device="cpu", pretrained=True)
    model = sep.target_models["vocals"].eval()

    # ---- 1. Weights -> F32 safetensors (drop int64 num_batches_tracked) ----
    sd = {k: v.float().contiguous() for k, v in model.state_dict().items()
          if v.dtype == torch.float32}
    save_file(sd, os.path.join(MODELS, "umxhq_vocals.safetensors"))
    hp = {
        "n_fft": N_FFT, "hop": HOP, "sample_rate": SR,
        "nb_channels": 2, "nb_bins": int(model.nb_bins),
        "nb_output_bins": int(model.nb_output_bins),
        "hidden_size": int(model.hidden_size), "nb_layers": 3,
        "bidirectional": True, "bn_eps": 1e-5,
    }
    with open(os.path.join(MODELS, "umxhq_vocals.json"), "w") as f:
        json.dump(hp, f, indent=2)
    print("saved weights + hyperparams for", len(sd), "tensors")

    # ---- 2. Reference on real audio (resample 44.1k, stereo) ----
    import librosa
    wav_path = os.path.join(ROOT, "tests", "fixtures", "speech_librispeech.wav")
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    y = y[: SR * 2]  # 2 s keeps the C++ alignment test quick
    stereo = np.stack([y, y * 0.9], axis=0)  # distinct channels (0.9 gain on R)
    audio = torch.from_numpy(stereo).float().unsqueeze(0)  # (1, 2, T)

    win = torch.hann_window(N_FFT)
    # torch.stft on (channels, T): returns (channels, freq, frames) complex.
    spec = torch.stft(torch.from_numpy(stereo).float(), n_fft=N_FFT, hop_length=HOP,
                      window=win, center=True, return_complex=True)  # (2, 2049, F)
    mix_mag = spec.abs()  # (2, 2049, F)
    nb_frames = mix_mag.shape[-1]

    # Model input layout: (nb_samples=1, nb_channels=2, nb_bins=2049, nb_frames).
    x_in = mix_mag.unsqueeze(0)  # (1, 2, 2049, F)
    with torch.no_grad():
        est = model(x_in)  # (1, 2, 2049, F) estimated vocal magnitude
    est_mag = est.squeeze(0)  # (2, 2049, F)

    # ---- 3. Reconstruct vocal via mixture-phase reuse (what C++ will do) ----
    phase = spec / (mix_mag + 1e-10)  # unit-magnitude complex phase
    voc_spec = est_mag * phase  # (2, 2049, F) complex
    voc = torch.istft(voc_spec, n_fft=N_FFT, hop_length=HOP, window=win,
                      center=True, length=stereo.shape[1])  # (2, T)

    dump_bin(os.path.join(FIX, "umx_input_wave.bin"), stereo)          # (2, T)
    dump_bin(os.path.join(FIX, "umx_mix_mag.bin"), mix_mag.numpy())    # (2, 2049, F)
    dump_bin(os.path.join(FIX, "umx_model_out.bin"), est_mag.numpy())  # (2, 2049, F)
    dump_bin(os.path.join(FIX, "umx_vocal_wave.bin"), voc.numpy())     # (2, T)
    ref = {
        "T": int(stereo.shape[1]), "nb_frames": int(nb_frames),
        "nb_bins": 2049, "nb_channels": 2, "n_fft": N_FFT, "hop": HOP,
    }
    with open(os.path.join(FIX, "umx_ref.json"), "w") as f:
        json.dump(ref, f, indent=2)
    print("reference: T=%d frames=%d" % (ref["T"], ref["nb_frames"]))
    print("est_mag range [%.4f, %.4f], voc range [%.4f, %.4f]" %
          (est_mag.min(), est_mag.max(), voc.min(), voc.max()))


if __name__ == "__main__":
    main()
