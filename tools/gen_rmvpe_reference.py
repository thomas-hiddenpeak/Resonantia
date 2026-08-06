#!/usr/bin/env python3
"""RMVPE reference model (PyTorch) for generating F0 ground truth.

Canonical RMVPE E2E architecture (DeepUnet + BiGRU) matching the safetensors
weights. Used to produce reference mel/salience/F0 for C++ alignment.
"""

import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from librosa.filters import mel as librosa_mel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

N_MELS = 128
N_CLASS = 360


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        return self.conv(x) + x


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks,
                 out_channels=16, momentum=0.01):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for _ in range(n_encoders):
            self.layers.append(
                ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum))
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            t, x = self.layers[i](x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList(
            [ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)])
        for _ in range(n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride,
                               (1, 1), output_padding=out_padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList(
            [ConvBlockRes(out_channels * 2, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4,
                 in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size,
                               n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel,
            inter_layers, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers,
                               kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers,
                          batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5,
                 inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers,
                             in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)  # [B, 1, T, 128]
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)  # [B, T, 3*128]
        x = self.fc(x)  # [B, T, 360]
        return x


class MelSpectrogram(nn.Module):
    def __init__(self, n_mel, sr=16000, n_fft=1024, win=1024, hop=160,
                 fmin=30, fmax=8000, clamp=1e-5):
        super().__init__()
        mel_basis = librosa_mel(sr=sr, n_fft=n_fft, n_mels=n_mel,
                                fmin=fmin, fmax=fmax)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.n_fft = n_fft
        self.win = win
        self.hop = hop
        self.clamp = clamp
        self.register_buffer("window", torch.hann_window(win))

    def forward(self, audio):
        fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop,
                         win_length=self.win, window=self.window,
                         center=True, return_complex=True)
        magnitude = torch.abs(fft)  # [B, F, T]
        mel = torch.matmul(self.mel_basis, magnitude)  # [B, n_mel, T]
        log_mel = torch.log(torch.clamp(mel, min=self.clamp))
        return log_mel


def save_tensor_binary(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    with open(path, "wb") as f:
        f.write(struct.pack("<i", arr.ndim))
        for s in arr.shape:
            f.write(struct.pack("<i", s))
        f.write(arr.tobytes())
    print(f"    Saved {path}: shape={list(arr.shape)}")


# Cents mapping for F0 decode
CENTS_MAPPING = 20 * np.arange(N_CLASS) + 1997.3794084376191


def decode_f0(salience, thred=0.03):
    """Decode salience [T, 360] -> F0 [T] via local weighted average."""
    center = np.argmax(salience, axis=1)  # [T]
    salience = np.pad(salience, ((0, 0), (4, 4)))
    center += 4
    todo_salience = []
    todo_cents_mapping = []
    starts = center - 4
    ends = center + 5
    for idx in range(salience.shape[0]):
        todo_salience.append(salience[idx, starts[idx]:ends[idx]])
        todo_cents_mapping.append(CENTS_MAPPING[starts[idx] - 4:ends[idx] - 4])
    todo_salience = np.array(todo_salience)          # [T, 9]
    todo_cents_mapping = np.array(todo_cents_mapping)  # [T, 9]
    product_sum = np.sum(todo_salience * todo_cents_mapping, axis=1)
    weight_sum = np.sum(todo_salience, axis=1)
    devided = product_sum / weight_sum
    maxx = np.max(salience, axis=1)
    devided[maxx <= thred] = 0
    return devided  # cents


def cents_to_f0(cents):
    f0 = 10 * (2 ** (cents / 1200))
    f0[cents == 0] = 0
    return f0


if __name__ == "__main__":
    from safetensors.torch import load_file

    audio_path = os.path.join(FIXTURES_DIR, "speech_librispeech.wav")
    prefix = os.path.join(FIXTURES_DIR, "rmvpe_ref")

    print("Building RMVPE E2E model...")
    model = E2E(4, 1, (2, 2))
    state = load_file(os.path.join(MODELS_DIR, "rmvpe.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print("  Sample missing:", missing[:5])
    if unexpected:
        print("  Sample unexpected:", unexpected[:5])
    model.eval()

    # Load audio
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    print(f"  Audio: {audio.shape[1]} samples @ {sr} Hz")

    mel_extractor = MelSpectrogram(N_MELS)
    with torch.no_grad():
        mel = mel_extractor(audio)  # [1, 128, T]
        print(f"  Mel: {mel.shape}")
        save_tensor_binary(f"{prefix}_mel.bin", mel[0].numpy())

        # Pad mel time to multiple of 32 (RMVPE requirement)
        T = mel.shape[-1]
        n_pad = 32 * ((T - 1) // 32 + 1) - T
        mel_padded = F.pad(mel, (0, n_pad), mode="constant") if n_pad > 0 else mel
        print(f"  Mel padded: {mel_padded.shape}")

        salience = model(mel_padded)  # [1, T', 360]
        salience = salience[0, :T].numpy()  # trim to original T
        print(f"  Salience: {salience.shape}")
        save_tensor_binary(f"{prefix}_salience.bin", salience)

        cents = decode_f0(salience)
        f0 = cents_to_f0(cents)
        print(f"  F0: {f0.shape}, non-zero frames: {(f0 > 0).sum()}")
        print(f"  F0 range: {f0[f0>0].min():.1f} - {f0[f0>0].max():.1f} Hz" if (f0>0).any() else "  all zero")
        save_tensor_binary(f"{prefix}_f0.bin", f0.reshape(-1, 1))

    print("\nDone! RMVPE reference files:")
    for f in sorted(os.listdir(FIXTURES_DIR)):
        if f.startswith("rmvpe_ref"):
            fp = os.path.join(FIXTURES_DIR, f)
            print(f"  {f}: {os.path.getsize(fp)/1024:.1f} KB")
