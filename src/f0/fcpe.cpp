// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/f0/fcpe.h"

#include <cmath>
#include <algorithm>
#include <vector>

namespace voxmutatio::f0 {

bool FcpeExtractor::init(const FcpeConfig& config) {
    threshold_ = config.threshold;
    
    // TODO: Load FCPE safetensors weights and allocate GPU buffers
    // For now, stub implementation
    return true;
}

std::vector<float> FcpeExtractor::infer(const float* audio,
                                         int num_samples) {
    // Frame parameters
    const int hop_length = 160;      // 10ms at 16kHz
    const int window_size = 512;     // Smaller FFT for real-time
    
    // Calculate number of frames
    int num_frames = (num_samples - window_size) / hop_length + 1;
    
    std::vector<float> f0(num_frames, 0.0f);
    
    // TODO: Full FCPE inference with:
    // 1. Lightweight STFT
    // 2. Context-based pitch estimation
    // 3. Confidence scoring
    
    // Stub: return zero F0
    return f0;
}

std::vector<float> FcpeExtractor::pitch_shift(const std::vector<float>& f0,
                                               int semitones) {
    std::vector<float> result(f0.size());
    
    // Pitch shift formula: f_new = f_old * 2^(semitones/12)
    float multiplier = std::pow(2.0f, semitones / 12.0f);
    
    for (size_t i = 0; i < f0.size(); ++i) {
        if (f0[i] > 0.0f) {  // Only shift voiced frames
            result[i] = f0[i] * multiplier;
        }
    }
    
    return result;
}

}  // namespace voxmutatio::f0
