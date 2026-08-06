// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/content/hubert_encoder.h"
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

// Transformer layer weights
struct TransformerLayer {
    // Self-attention
    float* q_weight = nullptr;   // [dim, dim]
    float* q_bias = nullptr;     // [dim]
    float* k_weight = nullptr;   // [dim, dim]
    float* k_bias = nullptr;     // [dim]
    float* v_weight = nullptr;   // [dim, dim]
    float* v_bias = nullptr;     // [dim]
    float* out_weight = nullptr; // [dim, dim]
    float* out_bias = nullptr;   // [dim]
    
    // FFN
    float* ffn1_weight = nullptr; // [dim, dim*4]
    float* ffn1_bias = nullptr;   // [dim*4]
    float* ffn2_weight = nullptr; // [dim*4, dim]
    float* ffn2_bias = nullptr;   // [dim]
    
    // Layer norm
    float* ln1_weight = nullptr; // [dim]
    float* ln1_bias = nullptr;   // [dim]
    float* ln2_weight = nullptr; // [dim]
    float* ln2_bias = nullptr;   // [dim]
};

struct HubertModel {
    // Feature projection
    float* fbank_weight = nullptr; // [80, dim]
    float* fbank_bias = nullptr;   // [dim]
    
    // Position embedding
    float* position_embed = nullptr; // [5026, dim]
    
    // Transformer layers
    std::vector<TransformerLayer> layers;
    
    // Final projection (v1)
    float* final_proj_weight = nullptr;
    float* final_proj_bias = nullptr;
    
    int dim = 768;
    int num_layers = 12;
    int output_dim = 256;
    bool use_final_proj = false;
};

// Load model from safetensors
HubertModel load_hubert_model(const std::string& model_path, int output_dim,
                               bool use_final_proj) {
    HubertModel model;
    model.output_dim = output_dim;
    model.dim = use_final_proj ? 768 : output_dim;
    model.use_final_proj = use_final_proj;
    
    io::SafetensorsLoader loader;
    if (!loader.load(model_path)) {
        return model;
    }
    
    // Helper to load tensor as float array
    auto load_tensor = [&loader](const std::string& name, float** ptr) {
        const io::Tensor* t = loader.get_tensor(name);
        if (!t) return false;
        
        int num_elements = 1;
        for (int64_t s : t->shape) num_elements *= s;
        
        *ptr = new float[num_elements];
        std::memcpy(*ptr, loader.data(name), num_elements * sizeof(float));
        return true;
    };
    
    // Load feature projection
    load_tensor("feature_projection.weight", &model.fbank_weight);
    load_tensor("feature_projection.bias", &model.fbank_bias);
    
    // Load position embedding
    load_tensor("embed_positions.weight", &model.position_embed);
    
    // Load transformer layers
    model.layers.resize(model.num_layers);
    for (int i = 0; i < model.num_layers; ++i) {
        std::string prefix = "encoder.layers." + std::to_string(i) + ".";
        
        TransformerLayer& layer = model.layers[i];
        
        // Self-attention
        load_tensor(prefix + "self_attn.q_proj.weight", &layer.q_weight);
        load_tensor(prefix + "self_attn.q_proj.bias", &layer.q_bias);
        load_tensor(prefix + "self_attn.k_proj.weight", &layer.k_weight);
        load_tensor(prefix + "self_attn.k_proj.bias", &layer.k_bias);
        load_tensor(prefix + "self_attn.v_proj.weight", &layer.v_weight);
        load_tensor(prefix + "self_attn.v_proj.bias", &layer.v_bias);
        load_tensor(prefix + "self_attn.out_proj.weight", &layer.out_weight);
        load_tensor(prefix + "self_attn.out_proj.bias", &layer.out_bias);
        
        // FFN
        load_tensor(prefix + "fc1.weight", &layer.ffn1_weight);
        load_tensor(prefix + "fc1.bias", &layer.ffn1_bias);
        load_tensor(prefix + "fc2.weight", &layer.ffn2_weight);
        load_tensor(prefix + "fc2.bias", &layer.ffn2_bias);
        
        // Layer norm
        load_tensor(prefix + "layer_norm0.weight", &layer.ln1_weight);
        load_tensor(prefix + "layer_norm0.bias", &layer.ln1_bias);
        load_tensor(prefix + "layer_norm1.weight", &layer.ln2_weight);
        load_tensor(prefix + "layer_norm1.bias", &layer.ln2_bias);
    }
    
    // Load final projection (v1)
    if (use_final_proj) {
        load_tensor("projector.0.weight", &model.final_proj_weight);
        load_tensor("projector.0.bias", &model.final_proj_bias);
    }
    
    return model;
}

// Transformer encoder forward pass
std::vector<float> hubert_forward(const HubertModel& model,
                                   const float* fbank, int num_frames) {
    int dim = model.dim;
    int fbank_dim = 80;
    
    // Feature projection: fbank @ fbank_weight + fbank_bias
    std::vector<float> features(num_frames * dim);
    cuda::matmul(fbank, num_frames, fbank_dim,
                 model.fbank_weight, dim,
                 features.data());
    
    // Add bias
    for (int i = 0; i < num_frames; ++i) {
        for (int d = 0; d < dim; ++d) {
            features[i * dim + d] += model.fbank_bias[d];
        }
    }
    
    // Add position embedding
    for (int i = 0; i < num_frames && i < 5026; ++i) {
        for (int d = 0; d < dim; ++d) {
            features[i * dim + d] += model.position_embed[i * dim + d];
        }
    }
    
    // Transformer layers
    for (const auto& layer : model.layers) {
        // Layer norm 1
        std::vector<float> norm1 = cuda::layer_norm(features.data(), num_frames, dim);
        
        // Scale and shift
        for (int i = 0; i < num_frames; ++i) {
            for (int d = 0; d < dim; ++d) {
                norm1[i * dim + d] = norm1[i * dim + d] * layer.ln1_weight[d] +
                                     layer.ln1_bias[d];
            }
        }
        
        // Self-attention (simplified single-head for now)
        std::vector<float> attn_out = cuda::multihead_attention(
            norm1.data(), norm1.data(), norm1.data(),
            num_frames, 1, dim);
        
        // Residual connection
        for (int i = 0; i < num_frames * dim; ++i) {
            features[i] += attn_out[i];
        }
        
        // Layer norm 2
        std::vector<float> norm2 = cuda::layer_norm(features.data(), num_frames, dim);
        
        // Scale and shift
        for (int i = 0; i < num_frames; ++i) {
            for (int d = 0; d < dim; ++d) {
                norm2[i * dim + d] = norm2[i * dim + d] * layer.ln2_weight[d] +
                                     layer.ln2_bias[d];
            }
        }
        
        // FFN
        std::vector<float> ffn_out(num_frames * dim);
        cuda::matmul(norm2.data(), num_frames, dim,
                     layer.ffn2_weight, dim,
                     ffn_out.data());
        
        // Residual connection
        for (int i = 0; i < num_frames * dim; ++i) {
            features[i] += ffn_out[i];
        }
    }
    
    // Final projection (v1)
    if (model.use_final_proj) {
        std::vector<float> output(num_frames * model.output_dim);
        cuda::matmul(features.data(), num_frames, dim,
                     model.final_proj_weight, model.output_dim,
                     output.data());
        return output;
    }
    
    return features;
}

}  // namespace

bool HubertEncoder::init(const HubertConfig& config) {
    output_dim_ = config.output_dim;
    half_precision_ = config.half_precision;
    
    // Model will be loaded on first extract() call
    return true;
}

std::vector<float> HubertEncoder::extract(const float* audio,
                                           int num_samples) {
    if (num_samples == 0) {
        return {};
    }
    
    // Load model
    HubertModel model = load_hubert_model(
        "", output_dim_, output_dim_ == 256);  // v1 uses final_proj
    
    // Compute fbank features
    auto fbank = cuda::compute_fbank(audio, num_samples, 80, 1600, 320, 16000);
    int num_frames = static_cast<int>(fbank.size()) / 80;
    
    // Forward pass
    auto features = hubert_forward(model, fbank.data(), num_frames);
    
    return features;
}

}  // namespace voxmutatio::content
