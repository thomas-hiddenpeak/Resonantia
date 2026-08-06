// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/f0/rmvpe.h"
#include "voxmutatio/content/cuda/kernels.h"
#include "voxmutatio/io/safetensors.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace voxmutatio::f0 {

namespace {

struct RmvpeModel {
    // Conformer encoder layers
    float* conv1_weight = nullptr;
    float* conv1_bias = nullptr;
    
    float* sub_sampling_weight = nullptr;
    float* sub_sampling_bias = nullptr;
    
    std::vector<float*> encoder_weights;
    std::vector<float*> encoder_biases;
    
    float* head_weight = nullptr;
    float* head_bias = nullptr;
    
    int dim = 256;
    int num_bins = 360;  // F0 range: 50-1100 Hz
};

RmvpeModel load_rmvpe_model(const std::string& model_path) {
    RmvpeModel model;
    
    io::SafetensorsLoader loader;
    if (!loader.load(model_path)) {
        return model;
    }
    
    auto load_tensor = [&loader](const std::string& name, float** ptr) {
        const io::Tensor* t = loader.get_tensor(name);
        if (!t) return false;
        
        int num_elements = 1;
        for (int64_t s : t->shape) num_elements *= s;
        
        *ptr = new float[num_elements];
        std::memcpy(*ptr, loader.data(name), num_elements * sizeof(float));
        return true;
    };
    
    load_tensor("conv1.weight", &model.conv1_weight);
    load_tensor("conv1.bias", &model.conv1_bias);
    load_tensor("sub_sampling.weight", &model.sub_sampling_weight);
    load_tensor("sub_sampling.bias", &model.sub_sampling_bias);
    load_tensor("head.weight", &model.head_weight);
    load_tensor("head.bias", &model.head_bias);
    
    return model;
}

// Compute spectrogram features for RMVPE
std::vector<float> compute_spectrogram(const float* audio, int num_samples,
                                        int fft_size = 1024, int hop_length = 160) {
    int num_frames = (num_samples - fft_size) / hop_length + 1;
    int num_freqs = fft_size / 2 + 1;
    
    // Apply Hann window
    std::vector<float> window(fft_size);
    for (int i = 0; i < fft_size; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (fft_size - 1)));
    }
    
    // Windowed frames
    std::vector<float> windowed(num_frames * fft_size);
    for (int f = 0; f < num_frames; ++f) {
        for (int i = 0; i < fft_size; ++i) {
            windowed[f * fft_size + i] = audio[f * hop_length + i] * window[i];
        }
    }
    
    // Simple DFT magnitude (could use cuFFT for GPU)
    std::vector<float> spec(num_frames * num_freqs);
    for (int f = 0; f < num_frames; ++f) {
        for (int k = 0; k < num_freqs; ++k) {
            float real = 0.0f, imag = 0.0f;
            for (int n = 0; n < fft_size; ++n) {
                float angle = -2.0f * M_PI * k * n / fft_size;
                real += windowed[f * fft_size + n] * std::cos(angle);
                imag += windowed[f * fft_size + n] * std::sin(angle);
            }
            spec[f * num_freqs + k] = std::sqrt(real * real + imag * imag);
        }
    }
    
    return spec;
}

std::vector<float> rmvpe_forward(const RmvpeModel& model,
                                  const float* spec, int num_frames) {
    int num_bins = 360;
    
    // Conformer encoder forward (simplified)
    std::vector<float> features(num_frames * model.dim);
    
    // Sub-sampling
    content::cuda::matmul(spec, num_frames, 513,
                 model.sub_sampling_weight, model.dim,
                 features.data());
    
    // Add bias
    for (int i = 0; i < num_frames; ++i) {
        for (int d = 0; d < model.dim; ++d) {
            features[i * model.dim + d] += model.sub_sampling_bias[d];
        }
    }
    
    // Head: predict F0 probabilities
    std::vector<float> logits(num_frames * num_bins);
    content::cuda::matmul(features.data(), num_frames, model.dim,
                 model.head_weight, num_bins,
                 logits.data());
    
    // Softmax and argmax to get F0
    std::vector<float> f0(num_frames, 0.0f);
    
    for (int f = 0; f < num_frames; ++f) {
        // Find max probability bin
        int max_bin = 0;
        float max_prob = -1e30f;
        
        for (int b = 0; b < num_bins; ++b) {
            if (logits[f * num_bins + b] > max_prob) {
                max_prob = logits[f * num_bins + b];
                max_bin = b;
            }
        }
        
        // Convert bin to F0 (Hz)
        // F0 = 10 * (2^(b/12)) where b is bin index
        if (max_bin > 0) {
            f0[f] = 10.0f * std::pow(2.0f, max_bin / 12.0f);
        }
    }
    
    return f0;
}

}  // namespace

bool RmvpeExtractor::init(const RmvpeConfig& config) {
    threshold_ = config.threshold;
    
    return true;
}

std::vector<float> RmvpeExtractor::infer(const float* audio,
                                          int num_samples) {
    if (num_samples == 0) {
        return {};
    }
    
    RmvpeModel model = load_rmvpe_model("");
    
    // Compute spectrogram
    auto spec = compute_spectrogram(audio, num_samples);
    int num_frames = static_cast<int>(spec.size()) / 513;
    
    // Forward pass
    auto f0 = rmvpe_forward(model, spec.data(), num_frames);
    
    return f0;
}

std::vector<float> RmvpeExtractor::pitch_shift(const std::vector<float>& f0,
                                                int semitones) {
    std::vector<float> result(f0.size());
    
    float multiplier = std::pow(2.0f, semitones / 12.0f);
    
    for (size_t i = 0; i < f0.size(); ++i) {
        if (f0[i] > 0.0f) {
            result[i] = f0[i] * multiplier;
        }
    }
    
    return result;
}

}  // namespace voxmutatio::f0
