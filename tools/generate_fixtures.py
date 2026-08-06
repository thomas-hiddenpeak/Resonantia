#!/usr/bin/env python3
"""Generate test audio and Python reference outputs for numerical alignment."""

import os
import numpy as np
import soundfile as sf
from scipy import signal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")
FIXTURES_DIR = os.path.join(TESTS_DIR, "fixtures")

def generate_test_audio():
    """Generate test WAV files with known content."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    
    sr = 44100  # Sample rate
    
    # Test 1: Pure sine wave (440 Hz)
    t = np.arange(sr * 2) / sr  # 2 seconds
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio = audio.astype(np.float32)
    
    path = os.path.join(FIXTURES_DIR, "test_sine_440hz.wav")
    sf.write(path, audio, sr)
    print(f"Generated: {path} ({len(audio)} samples, {len(audio)/sr:.2f}s)")
    
    # Test 2: Multi-tone (speech-like)
    tones = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.1 * np.sin(2 * np.pi * 800 * t)
    )
    audio2 = tones.astype(np.float32)
    
    path2 = os.path.join(FIXTURES_DIR, "test_multitone.wav")
    sf.write(path2, audio2, sr)
    print(f"Generated: {path2} ({len(audio2)} samples, {len(audio2)/sr:.2f}s)")
    
    return [path, path2]

def generate_hubert_reference(audio_path):
    """Generate HuBERT feature reference using transformers."""
    try:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        
        print("\nLoading HuBERT model...")
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Process audio
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        features = model(**inputs).last_hidden_state
        
        # Get feature sequence (average over feature dimension for comparison)
        features_np = features[0].cpu().numpy().astype(np.float32)
        
        ref_path = os.path.join(FIXTURES_DIR, "hubert_reference.npz")
        np.savez(ref_path, 
                 features=features_np,
                 shape=features_np.shape,
                 dtype=str(features_np.dtype))
        print(f"HuBERT reference: {ref_path}")
        print(f"  Shape: {features_np.shape}")
        
        return ref_path
    except ImportError:
        print("  SKIP: transformers not installed")
        return None

if __name__ == "__main__":
    print("Generating test audio files...")
    audio_files = generate_test_audio()
    
    print("\nGenerating Python reference outputs...")
    for af in audio_files:
        generate_hubert_reference(af)
    
    print("\nDone! Fixtures in tests/fixtures/")
    for f in sorted(os.listdir(FIXTURES_DIR)):
        size = os.path.getsize(os.path.join(FIXTURES_DIR, f))
        print(f"  {f} ({size / 1024:.1f} KB)")
