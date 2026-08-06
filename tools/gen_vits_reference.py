#!/usr/bin/env python3
"""Generate VITS (SynthesizerTrnMs768NSFsid) reference outputs for C++ alignment.

Deterministic (noise=0) inference. Saves intermediates (m_p, logs_p, z_p, z)
and the final waveform for stage-by-stage numerical alignment.
"""

import os
import sys
import struct
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RVC_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvc_ref")

sys.path.insert(0, RVC_REF)

from infer.module.models import SynthesizerTrnMs768NSFsid  # noqa: E402

# 40k v2 config
CONFIG = dict(
    spec_channels=1025,
    segment_size=32,
    inter_channels=192,
    hidden_channels=192,
    filter_channels=768,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    p_dropout=0,
    resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[10, 10, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 4, 4],
    spk_embed_dim=109,
    gin_channels=256,
    sr=40000,
    is_half=False,
)

F0_MEL_MIN = 1127 * np.log(1 + 50 / 700)
F0_MEL_MAX = 1127 * np.log(1 + 1100 / 700)


def save_bin(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr).astype(np.float32))
    with open(path, "wb") as f:
        f.write(struct.pack("<i", arr.ndim))
        for s in arr.shape:
            f.write(struct.pack("<i", s))
        f.write(arr.tobytes())
    print(f"    Saved {path}: {list(arr.shape)}")


def f0_to_coarse(f0):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    mask = f0_mel > 0
    f0_mel[mask] = (f0_mel[mask] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return np.rint(f0_mel).astype(np.int64)


def load_bin(path):
    with open(path, "rb") as f:
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = [struct.unpack("<i", f.read(4))[0] for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)


if __name__ == "__main__":
    from safetensors.torch import load_file

    torch.manual_seed(0)

    # Make inference fully deterministic: disable all Gaussian noise
    # (flow noise_scale=0 handled manually; SineGen additive noise -> 0).
    _orig_randn_like = torch.randn_like
    torch.randn_like = lambda x, *a, **k: torch.zeros_like(x)
    _orig_randn = torch.randn
    torch.randn = lambda *a, **k: torch.zeros(*a, **k)

    print("Building SynthesizerTrnMs768NSFsid (40k v2)...")
    net_g = SynthesizerTrnMs768NSFsid(**CONFIG)
    state = load_file(os.path.join(MODELS_DIR, "pretrained_v2",
                                   "pretrained_v2", "f0G40k.safetensors"))
    # Strip "model." prefix
    clean = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = net_g.load_state_dict(clean, strict=False)
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print("   e.g. missing:", missing[:5])
    net_g.eval().float()
    net_g.remove_weight_norm()

    # Load HuBERT features (v2, 768) and F0 references
    print("Loading HuBERT features and F0...")
    feats = load_bin(os.path.join(FIXTURES_DIR, "hubert_ref_layer12.bin"))  # [T, 768]
    f0 = load_bin(os.path.join(FIXTURES_DIR, "rmvpe_ref_f0.bin")).reshape(-1)  # [Tf]
    print(f"  HuBERT: {feats.shape}, F0: {f0.shape}")

    # RVC: interpolate features 2x (50Hz -> 100Hz to match F0)
    feats_t = torch.from_numpy(feats).unsqueeze(0)  # [1, T, 768]
    feats_up = F.interpolate(feats_t.permute(0, 2, 1), scale_factor=2,
                             mode="nearest").permute(0, 2, 1)  # [1, 2T, 768]
    print(f"  Interpolated features: {feats_up.shape}")

    # Align lengths
    T = min(feats_up.shape[1], len(f0))
    feats_up = feats_up[:, :T, :]
    f0 = f0[:T]
    pitch_coarse = f0_to_coarse(f0.copy())

    phone = feats_up.float()                                   # [1, T, 768]
    phone_lengths = torch.LongTensor([T])
    pitch = torch.LongTensor(pitch_coarse).unsqueeze(0)        # [1, T]
    nsff0 = torch.from_numpy(f0.astype(np.float32)).unsqueeze(0)  # [1, T]
    sid = torch.LongTensor([0])

    save_bin(os.path.join(FIXTURES_DIR, "vits_ref_phone.bin"), phone[0].numpy())
    save_bin(os.path.join(FIXTURES_DIR, "vits_ref_pitch.bin"),
             pitch_coarse.astype(np.float32).reshape(-1, 1))
    save_bin(os.path.join(FIXTURES_DIR, "vits_ref_nsff0.bin"),
             f0.astype(np.float32).reshape(-1, 1))

    print("Running deterministic inference (noise=0)...")
    with torch.no_grad():
        g = net_g.emb_g(sid).unsqueeze(-1)  # [1, gin, 1]
        m_p, logs_p, x_mask = net_g.enc_p(phone, pitch, phone_lengths)
        save_bin(os.path.join(FIXTURES_DIR, "vits_ref_m_p.bin"), m_p[0].numpy())
        save_bin(os.path.join(FIXTURES_DIR, "vits_ref_logs_p.bin"), logs_p[0].numpy())

        # Deterministic: z_p = m_p * x_mask  (noise scale = 0)
        z_p = m_p * x_mask
        z = net_g.flow(z_p, x_mask, g=g, reverse=True)
        save_bin(os.path.join(FIXTURES_DIR, "vits_ref_z.bin"), z[0].numpy())

        o = net_g.dec(z * x_mask, nsff0, g=g)
        audio = o[0, 0].numpy()
        print(f"  Output audio: {audio.shape} ({len(audio)/40000:.2f}s @ 40kHz)")
        save_bin(os.path.join(FIXTURES_DIR, "vits_ref_audio.bin"), audio.reshape(-1, 1))
        sf.write(os.path.join(FIXTURES_DIR, "vits_ref_output.wav"), audio, 40000)

        # --- Generator intermediates for debugging ---
        dec = net_g.dec
        har, _, _ = dec.m_source(nsff0, dec.upp)  # [1, T*upp, 1]
        har_t = har.transpose(1, 2)  # [1, 1, T*upp]
        save_bin(os.path.join(FIXTURES_DIR, "vits_ref_har.bin"),
                 har_t[0, 0].numpy().reshape(-1, 1))
        xg = dec.conv_pre(z * x_mask)
        xg = xg + dec.cond(g)
        save_bin(os.path.join(FIXTURES_DIR, "vits_ref_convpre.bin"), xg[0].numpy())

        # Manual first upsample stage
        import torch.nn.functional as _F
        x1 = _F.leaky_relu(xg, dec.lrelu_slope)
        x1 = dec.ups[0](x1)
        save_bin(os.path.join(FIXTURES_DIR, "vits_ref_ups0.bin"), x1[0].detach().numpy())
        x1s = dec.noise_convs[0](har_t)
        save_bin(os.path.join(FIXTURES_DIR, "vits_ref_noiseconv0.bin"), x1s[0].detach().numpy())
        x1 = x1 + x1s
        xs = None
        for j in range(3):
            rb = dec.resblocks[j](x1)
            xs = rb if xs is None else xs + rb
        x1 = xs / 3
        save_bin(os.path.join(FIXTURES_DIR, "vits_ref_stage0.bin"), x1[0].detach().numpy())

    print("\nDone! VITS reference files:")
    for f in sorted(os.listdir(FIXTURES_DIR)):
        if f.startswith("vits_ref"):
            fp = os.path.join(FIXTURES_DIR, f)
            print(f"  {f}: {os.path.getsize(fp)/1024:.1f} KB")
