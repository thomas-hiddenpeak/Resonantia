// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/content/wavlm_encoder.h"
#include "voxmutatio/content/cuda/kernels.h"
#include "voxmutatio/io/safetensors.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace voxmutatio::content {

namespace {

struct WavlmLayer {
    float* q_weight = nullptr;
    float* q_bias = nullptr;
    float* k_weight = nullptr;
    float* k_bias = nullptr;
    float* v_weight = nullptr;
    float* v_bias = nullptr;
    float* out_weight = nullptr;
    float* out_bias = nullptr;
    
    float* ffn1_weight = nullptr;
    float* ffn1_bias = nullptr;
    float* ffn2_weight = nullptr;
    float* ffn2_bias = nullptr;
    
    float* ln1_weight = nullptr;
    float* ln1_bias = nullptr;
    float* ln2_weight = nullptr;
    float* ln2_bias = nullptr;
};

struct WavlmModel {
    float* fbank_weight = nullptr;
    float* fbank_bias = nullptr;
    float* position_embed = nullptr;
    
    std::vector<WavlmLayer> layers;
    
    int dim = 768;
    int num_layers = 12;
    int output_dim = 768;
};

WavlmModel load_wavlm_model(const std::string& model_path) {
    WavlmModel model;
    
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
    
    load_tensor("feature_projection.weight", &model.fbank_weight);
    load_tensor("feature_projection.bias", &model.fbank_bias);
    load_tensor("encoder.pos_emb_embed", &model.position_embed);
    
    model.layers.resize(model.num_layers);
    for (int i = 0; i < model.num_layers; ++i) {
        std::string prefix = "encoder.layers." + std::to_string(i) + ".";
        
        WavlmLayer& layer = model.layers[i];
        
        load_tensor(prefix + "self_attn.q_proj.weight", &layer.q_weight);
        load_tensor(prefix + "self_attn.q_proj.bias", &layer.q_bias);
        load_tensor(prefix + "self_attn.k_proj.weight", &layer.k_weight);
        load_tensor(prefix + "self_attn.k_proj.bias", &layer.k_bias);
        load_tensor(prefix + "self_attn.v_proj.weight", &layer.v_weight);
        load_tensor(prefix + "self_attn.v_proj.bias", &layer.v_bias);
        load_tensor(prefix + "self_attn.out_proj.weight", &layer.out_weight);
        load_tensor(prefix + "self_attn.out_proj.bias", &layer.out_bias);
        
        load_tensor(prefix + "fc1.weight", &layer.ffn1_weight);
        load_tensor(prefix + "fc1.bias", &layer.ffn1_bias);
        load_tensor(prefix + "fc2.weight", &layer.ffn2_weight);
        load_tensor(prefix + "fc2.bias", &layer.ffn2_bias);
        
        load_tensor(prefix + "layer_norm0.weight", &layer.ln1_weight);
        load_tensor(prefix + "layer_norm0.bias", &layer.ln1_bias);
        load_tensor(prefix + "layer_norm1.weight", &layer.ln2_weight);
        load_tensor(prefix + "layer_norm1.bias", &layer.ln2_bias);
    }
    
    return model;
}

std::vector<float> wavlm_forward(const WavlmModel& model,
                                  const float* fbank, int num_frames) {
    int dim = model.dim;
    int fbank_dim = 80;
    
    std::vector<float> features(num_frames * dim);
    cuda::matmul(fbank, num_frames, fbank_dim,
                 model.fbank_weight, dim,
                 features.data());
    
    for (int i = 0; i < num_frames; ++i) {
        for (int d = 0; d < dim; ++d) {
            features[i * dim + d] += model.fbank_bias[d];
        }
    }
    
    for (const auto& layer : model.layers) {
        std::vector<float> norm1 = cuda::layer_norm(features.data(), num_frames, dim);
        
        for (int i = 0; i < num_frames; ++i) {
            for (int d = 0; d < dim; ++d) {
                norm1[i * dim + d] = norm1[i * dim + d] * layer.ln1_weight[d] +
                                     layer.ln1_bias[d];
            }
        }
        
        std::vector<float> attn_out = cuda::multihead_attention(
            norm1.data(), norm1.data(), norm1.data(),
            num_frames, 1, dim);
        
        for (int i = 0; i < num_frames * dim; ++i) {
            features[i] += attn_out[i];
        }
        
        std::vector<float> norm2 = cuda::layer_norm(features.data(), num_frames, dim);
        
        for (int i = 0; i < num_frames; ++i) {
            for (int d = 0; d < dim; ++d) {
                norm2[i * dim + d] = norm2[i * dim + d] * layer.ln2_weight[d] +
                                     layer.ln2_bias[d];
            }
        }
        
        std::vector<float> ffn_out(num_frames * dim);
        cuda::matmul(norm2.data(), num_frames, dim,
                     layer.ffn2_weight, dim,
                     ffn_out.data());
        
        for (int i = 0; i < num_frames * dim; ++i) {
            features[i] += ffn_out[i];
        }
    }
    
    return features;
}

}  // namespace

bool WavlmEncoder::init(const WavlmConfig& config) {
    output_dim_ = config.output_dim;
    half_precision_ = config.half_precision;
    
    return true;
}

std::vector<float> WavlmEncoder::extract(const float* audio,
                                          int num_samples) {
    if (num_samples == 0) {
        return {};
    }
    
    WavlmModel model = load_wavlm_model("");
    
    auto fbank = cuda::compute_fbank(audio, num_samples, 80, 1600, 320, 16000);
    int num_frames = static_cast<int>(fbank.size()) / 80;
    
    auto features = wavlm_forward(model, fbank.data(), num_frames);
    
    return features;
}

}  // namespace voxmutatio::content
