// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/synthesizer/synthesizer.h"
#include "voxmutatio/content/cuda/kernels.h"
#include "voxmutatio/io/safetensors.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace voxmutatio::synthesizer {

namespace {

struct VitsModel {
    // Text encoder
    float* text_embedding_weight = nullptr;
    float* text_embedding_bias = nullptr;
    
    // Flow decoder
    std::vector<float*> flow_weights;
    std::vector<float*> flow_biases;
    
    // HiFiGAN vocoder
    float* upsamp1_weight = nullptr;
    float* upsamp1_bias = nullptr;
    float* resblock1_weight = nullptr;
    float* resblock1_bias = nullptr;
    float* out_conv_weight = nullptr;
    float* out_conv_bias = nullptr;
    
    // Speaker embedding
    float* spk_embed = nullptr;
    
    int spec_channels = 1025;
    int hidden_channels = 192;
    int output_dim = 80;  // Mel spectrogram dim
};

VitsModel load_vits_model(const std::string& model_path, ModelVersion version) {
    VitsModel model;
    
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
    
    load_tensor("text_encoder.embedding.weight", &model.text_embedding_weight);
    load_tensor("text_encoder.embedding.bias", &model.text_embedding_bias);
    
    float* flow_w = nullptr;
    load_tensor("flow.decoder.weight", &flow_w);
    if (flow_w) model.flow_weights.push_back(flow_w);
    
    load_tensor("upsample.0.weight", &model.upsamp1_weight);
    load_tensor("upsample.0.bias", &model.upsamp1_bias);
    load_tensor("resblocks.0.conv1.weight", &model.resblock1_weight);
    load_tensor("resblocks.0.conv1.bias", &model.resblock1_bias);
    load_tensor("out_conv.0.weight", &model.out_conv_weight);
    load_tensor("out_conv.0.bias", &model.out_conv_bias);
    load_tensor("spk_embed", &model.spk_embed);
    
    return model;
}

// HiFiGAN vocoder forward pass
std::vector<float> hifigan_forward(const VitsModel& model,
                                    const float* mel, int num_frames) {
    int hop_length = 256;
    int num_samples = num_frames * hop_length;
    
    // Multi-scale upsampling
    std::vector<float> audio(num_samples, 0.0f);
    
    // Simplified: direct mel-to-audio conversion
    // Full implementation would use transposed convolutions
    
    return audio;
}

// VITS flow decoder forward pass
std::vector<float> vits_flow_forward(const VitsModel& model,
                                      const float* features, int frames,
                                      const float* pitch, const float* pitchf,
                                      const float* spk_embed) {
    int dim = model.hidden_channels;
    
    // Text encoder
    std::vector<float> encoded(frames * dim);
    content::cuda::matmul(features, frames, 256,
                          model.text_embedding_weight, dim,
                          encoded.data());
    
    // Add speaker embedding
    for (int i = 0; i < frames; ++i) {
        for (int d = 0; d < dim; ++d) {
            encoded[i * dim + d] += spk_embed[d];
        }
    }
    
    // Flow decoder (simplified)
    std::vector<float> mel(frames * model.output_dim);
    content::cuda::matmul(encoded.data(), frames, dim,
                          model.flow_weights.empty() ? nullptr : model.flow_weights[0],
                          model.output_dim,
                          mel.data());
    
    return mel;
}

}  // namespace

bool Synthesizer::init(const SynthesizerConfig& config) {
    sample_rate_ = config.sample_rate;
    version_ = config.version;
    has_f0_ = config.has_f0;
    num_speakers_ = config.gin_channels;
    
    return true;
}

AudioBuffer Synthesizer::infer(const float* features, int frames,
                                const float* pitch, const float* pitchf,
                                int speaker_id) {
    AudioBuffer output;
    
    const int hop_length = 256;
    int num_samples = frames * hop_length;
    
    output.data.resize(num_samples, 0.0f);
    output.sample_rate = sample_rate_;
    output.source_format = SampleFormat::kFloat32;
    
    VitsModel model = load_vits_model("", version_);
    
    // Get speaker embedding
    const float* spk_embed = model.spk_embed ? 
                             model.spk_embed + speaker_id * 256 : nullptr;
    
    // Flow decoder
    auto mel = vits_flow_forward(model, features, frames,
                                  pitch, pitchf, spk_embed);
    
    // HiFiGAN vocoder
    auto audio = hifigan_forward(model, mel.data(), frames);
    
    if (!audio.empty()) {
        int copy_len = std::min(static_cast<int>(audio.size()), num_samples);
        std::memcpy(output.data.data(), audio.data(), 
                   copy_len * sizeof(float));
    }
    
    return output;
}

AudioBuffer Synthesizer::infer_stream(const float* features, int frames,
                                       const float* pitch, const float* pitchf,
                                       int speaker_id,
                                       int skip_head, int return_length) {
    AudioBuffer full_output = infer(features, frames, pitch, pitchf, speaker_id);
    
    AudioBuffer output;
    
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
