// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/synthesizer/synthesizer.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace voxmutatio::synthesizer {

bool Synthesizer::init(const SynthesizerConfig& config) {
    sample_rate_ = config.sample_rate;
    version_ = config.version;
    has_f0_ = config.has_f0;
    num_speakers_ = config.gin_channels;  // Speaker embedding dimension
    
    // TODO: Load safetensors weights and allocate GPU buffers
    // For now, stub implementation
    return true;
}

AudioBuffer Synthesizer::infer(const float* features, int frames,
                                const float* pitch, const float* pitchf,
                                int speaker_id) {
    AudioBuffer output;
    
    // Output samples = frames * hop_length (typically 256)
    const int hop_length = 256;
    int num_samples = frames * hop_length;
    
    output.data.resize(num_samples, 0.0f);
    output.sample_rate = sample_rate_;
    output.source_format = SampleFormat::kFloat32;
    
    // TODO: Full VITS synthesis with:
    // 1. TextEncoder forward
    // 2. Flow decoder (with F0 conditioning if has_f0_)
    // 3. HiFiGAN NSF vocoder
    
    // Stub: return zero audio
    return output;
}

AudioBuffer Synthesizer::infer_stream(const float* features, int frames,
                                       const float* pitch, const float* pitchf,
                                       int speaker_id,
                                       int skip_head, int return_length) {
    AudioBuffer full_output = infer(features, frames, pitch, pitchf, speaker_id);
    
    AudioBuffer output;
    
    // Extract window from full output
    if (skip_head >= 0 && return_length > 0) {
        int start = std::min(skip_head, static_cast<int>(full_output.data.size()));
        int end = std::min(start + return_length, 
                          static_cast<int>(full_output.data.size()));
        
        if (end > start) {
            output.data.assign(full_output.data.begin() + start,
                              full_output.data.begin() + end);
        }
    }
    
    output.sample_rate = full_output.sample_rate;
    output.source_format = full_output.source_format;
    
    return output;
}

}  // namespace voxmutatio::synthesizer
