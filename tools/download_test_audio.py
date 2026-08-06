#!/usr/bin/env python3
"""Download real speech audio for testing (LibriSpeech, public domain / CC-BY-4.0)."""

import os
import io
import numpy as np
import soundfile as sf
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")

LIBRISPEECH_PARQUET = (
    "https://huggingface.co/datasets/hf-internal-testing/"
    "librispeech_asr_dummy/resolve/main/clean/validation-00000-of-00001.parquet"
)


def download_real_speech():
    """Extract real speech samples from LibriSpeech parquet."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    print("Downloading LibriSpeech parquet...")
    df = pd.read_parquet(LIBRISPEECH_PARQUET)
    print(f"  {len(df)} samples available")

    saved = []
    for i in range(min(3, len(df))):
        audio_data = df.iloc[i]["audio"]
        audio_bytes = audio_data["bytes"]
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        text = df.iloc[i]["text"]

        out_path = os.path.join(FIXTURES_DIR, f"speech_{i:02d}.wav")
        sf.write(out_path, audio, sr)
        print(f"  [{i}] {out_path}: {len(audio)} samples @ {sr} Hz "
              f"({len(audio)/sr:.2f}s)")
        print(f"      \"{text[:60]}...\"")
        saved.append((out_path, audio, sr, text))

    main_path = os.path.join(FIXTURES_DIR, "speech_librispeech.wav")
    _, audio, sr, _ = saved[0]
    sf.write(main_path, audio, sr)
    print(f"\nMain test file: {main_path}")

    return saved


if __name__ == "__main__":
    print("Downloading real speech audio for testing...")
    download_real_speech()

    print("\nReal audio fixtures:")
    for f in sorted(os.listdir(FIXTURES_DIR)):
        if f.endswith((".wav", ".flac")):
            fp = os.path.join(FIXTURES_DIR, f)
            info = sf.info(fp)
            print(f"  {f}: {info.frames} frames @ {info.samplerate} Hz "
                  f"({info.duration:.2f}s)")
