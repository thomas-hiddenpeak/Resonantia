// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// HuBERT (ContentVec) encoder — numerically aligned with HuggingFace
// transformers HubertModel. Processes raw 16kHz waveform through a 7-layer
// CNN feature extractor, feature projection, positional conv embedding, and
// 12 post-norm Transformer layers.

#include "voxmutatio/content/hubert_encoder.h"
#include "voxmutatio/content/cuda/kernels.h"
#include "voxmutatio/io/safetensors.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace voxmutatio::content {

namespace {

namespace k = voxmutatio::content::cuda;

// HuBERT base architecture constants
constexpr int kNumConvLayers = 7;
constexpr int kConvDim = 512;
constexpr int kHidden = 768;
constexpr int kFFN = 3072;
constexpr int kNumLayers = 12;
constexpr int kNumHeads = 12;
constexpr int kPosConvKernel = 128;
constexpr int kPosConvGroups = 16;

const int kConvKernels[kNumConvLayers] = {10, 3, 3, 3, 3, 2, 2};
const int kConvStrides[kNumConvLayers] = {5, 2, 2, 2, 2, 2, 2};
const int kConvInCh[kNumConvLayers] = {1, 512, 512, 512, 512, 512, 512};

// Zero-copy view into the mmap'd safetensors data.
struct Weights {
    io::SafetensorsLoader loader;
    bool ok = false;

    bool load(const std::string& path) {
        ok = loader.load(path);
        return ok;
    }

    const float* get(const std::string& name) const {
        const uint8_t* p = loader.data(name);
        if (!p) {
            fprintf(stderr, "HuBERT: missing tensor '%s'\n", name.c_str());
        }
        return reinterpret_cast<const float*>(p);
    }

    bool has(const std::string& name) const {
        return loader.get_tensor(name) != nullptr;
    }
};

// Reconstruct pos_conv weight from weight_norm parametrization.
// original0 (g): [1, 1, K], original1 (v): [out_ch, in_per_group, K]
// weight[o,i,k] = g[k] * v[o,i,k] / ||v[:,:,k]||   (norm over dims 0,1)
std::vector<float> reconstruct_pos_conv_weight(const Weights& w) {
    const float* g = w.get("encoder.pos_conv_embed.conv.parametrizations.weight.original0");
    const float* v = w.get("encoder.pos_conv_embed.conv.parametrizations.weight.original1");
    if (!g || !v) return {};

    const int out_ch = kHidden;               // 768
    const int in_per_group = kHidden / kPosConvGroups;  // 48
    const int K = kPosConvKernel;             // 128

    // Compute per-k norm over (out_ch, in_per_group)
    std::vector<float> norm(K, 0.0f);
    for (int o = 0; o < out_ch; ++o) {
        for (int i = 0; i < in_per_group; ++i) {
            for (int kk = 0; kk < K; ++kk) {
                float val = v[(o * in_per_group + i) * K + kk];
                norm[kk] += val * val;
            }
        }
    }
    for (int kk = 0; kk < K; ++kk) norm[kk] = std::sqrt(norm[kk]);

    std::vector<float> weight(out_ch * in_per_group * K);
    for (int o = 0; o < out_ch; ++o) {
        for (int i = 0; i < in_per_group; ++i) {
            for (int kk = 0; kk < K; ++kk) {
                int idx = (o * in_per_group + i) * K + kk;
                weight[idx] = g[kk] * v[idx] / norm[kk];
            }
        }
    }
    return weight;
}

// Transpose [rows, cols] -> [cols, rows]
std::vector<float> transpose(const std::vector<float>& in, int rows, int cols) {
    std::vector<float> out(rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            out[c * rows + r] = in[r * cols + c];
    return out;
}

// Feature extractor: raw waveform [num_samples] -> [T, 512] (row-major)
std::vector<float> feature_extractor(const Weights& w, const float* audio,
                                     int num_samples) {
    // Channels-first working buffer: start [1, num_samples]
    std::vector<float> x(audio, audio + num_samples);
    int in_ch = 1;
    int in_len = num_samples;

    for (int layer = 0; layer < kNumConvLayers; ++layer) {
        std::string prefix = "feature_extractor.conv_layers." + std::to_string(layer) + ".";
        const float* conv_w = w.get(prefix + "conv.weight");
        int out_ch = kConvDim;
        int kernel = kConvKernels[layer];
        int stride = kConvStrides[layer];

        // Conv (no bias, conv_bias=false)
        auto conv_out = k::conv1d_strided(x.data(), in_ch, in_len,
                                          conv_w, nullptr, out_ch, kernel, stride);
        int out_len = (in_len - kernel) / stride + 1;

        // Layer 0 has group norm
        if (layer == 0) {
            const float* gn_w = w.get(prefix + "layer_norm.weight");
            const float* gn_b = w.get(prefix + "layer_norm.bias");
            std::vector<float> normed(out_ch * out_len);
            k::group_norm(conv_out.data(), normed.data(), out_ch, out_len,
                          out_ch, gn_w, gn_b);  // num_groups == channels
            conv_out = std::move(normed);
        }

        // GELU (exact)
        std::vector<float> activated(conv_out.size());
        k::gelu_exact(conv_out.data(), activated.data(),
                      static_cast<int>(conv_out.size()));

        x = std::move(activated);
        in_ch = out_ch;
        in_len = out_len;
    }

    // x is [512, T] channels-first -> transpose to [T, 512]
    return transpose(x, in_ch, in_len);  // [T, 512]
}

// Feature projection: [T,512] -> layer_norm -> linear -> [T,768]
std::vector<float> feature_projection(const Weights& w,
                                      const std::vector<float>& feats, int T) {
    const float* ln_w = w.get("feature_projection.layer_norm.weight");
    const float* ln_b = w.get("feature_projection.layer_norm.bias");
    auto normed = k::layer_norm_affine(feats.data(), T, kConvDim, ln_w, ln_b);

    const float* proj_w = w.get("feature_projection.projection.weight");  // [768,512]
    const float* proj_b = w.get("feature_projection.projection.bias");    // [768]
    return k::linear(normed.data(), T, kConvDim, proj_w, proj_b, kHidden);  // [T,768]
}

// Positional conv embedding + residual + encoder layer norm
std::vector<float> apply_pos_conv(const Weights& w,
                                  const std::vector<float>& hidden, int T,
                                  const std::vector<float>& pos_weight) {
    // Transpose [T,768] -> [768,T] channels-first
    auto x_cf = transpose(hidden, T, kHidden);

    const float* pos_b = w.get("encoder.pos_conv_embed.conv.bias");
    // Conv1d(768,768,k=128,pad=64,groups=16), out_len = T + 2*64 - 128 + 1 = T+1
    auto conv_out = k::conv1d_grouped(x_cf.data(), kHidden, T,
                                      pos_weight.data(), pos_b,
                                      kHidden, kPosConvKernel, 1, 64, kPosConvGroups);
    int conv_len = T + 2 * 64 - kPosConvKernel + 1;  // = T + 1

    // SamePadLayer: kernel even -> remove last time step
    int pad_remove = (kPosConvKernel % 2 == 0) ? 1 : 0;
    int pos_len = conv_len - pad_remove;  // = T

    // GELU on [768, pos_len]
    std::vector<float> pos_act(kHidden * pos_len);
    // Only process the retained columns
    {
        std::vector<float> trimmed(kHidden * pos_len);
        for (int c = 0; c < kHidden; ++c)
            for (int t = 0; t < pos_len; ++t)
                trimmed[c * pos_len + t] = conv_out[c * conv_len + t];
        k::gelu_exact(trimmed.data(), pos_act.data(),
                      static_cast<int>(trimmed.size()));
    }

    // Transpose pos_act [768, T] -> [T, 768] and add to hidden
    auto pos_tf = transpose(pos_act, kHidden, pos_len);  // [T,768]
    std::vector<float> summed(T * kHidden);
    for (int i = 0; i < T * kHidden; ++i) summed[i] = hidden[i] + pos_tf[i];

    // encoder.layer_norm
    const float* ln_w = w.get("encoder.layer_norm.weight");
    const float* ln_b = w.get("encoder.layer_norm.bias");
    return k::layer_norm_affine(summed.data(), T, kHidden, ln_w, ln_b);
}

// One transformer encoder layer (post-norm)
std::vector<float> encoder_layer(const Weights& w, int layer_idx,
                                 std::vector<float> hidden, int T) {
    std::string p = "encoder.layers." + std::to_string(layer_idx) + ".";

    // --- Self-attention ---
    const float* qw = w.get(p + "attention.q_proj.weight");
    const float* qb = w.get(p + "attention.q_proj.bias");
    const float* kw = w.get(p + "attention.k_proj.weight");
    const float* kb = w.get(p + "attention.k_proj.bias");
    const float* vw = w.get(p + "attention.v_proj.weight");
    const float* vb = w.get(p + "attention.v_proj.bias");
    const float* ow = w.get(p + "attention.out_proj.weight");
    const float* ob = w.get(p + "attention.out_proj.bias");

    auto q = k::linear(hidden.data(), T, kHidden, qw, qb, kHidden);
    auto kk = k::linear(hidden.data(), T, kHidden, kw, kb, kHidden);
    auto vv = k::linear(hidden.data(), T, kHidden, vw, vb, kHidden);

    auto attn = k::multihead_attention_split(q.data(), kk.data(), vv.data(),
                                             T, kHidden, kNumHeads);
    auto attn_out = k::linear(attn.data(), T, kHidden, ow, ob, kHidden);

    // Residual + layer norm
    for (int i = 0; i < T * kHidden; ++i) hidden[i] += attn_out[i];
    const float* ln1_w = w.get(p + "layer_norm.weight");
    const float* ln1_b = w.get(p + "layer_norm.bias");
    hidden = k::layer_norm_affine(hidden.data(), T, kHidden, ln1_w, ln1_b);

    // --- Feed forward ---
    const float* iw = w.get(p + "feed_forward.intermediate_dense.weight");
    const float* ib = w.get(p + "feed_forward.intermediate_dense.bias");
    const float* dw = w.get(p + "feed_forward.output_dense.weight");
    const float* db = w.get(p + "feed_forward.output_dense.bias");

    auto inter = k::linear(hidden.data(), T, kHidden, iw, ib, kFFN);  // [T,3072]
    std::vector<float> inter_act(inter.size());
    k::gelu_exact(inter.data(), inter_act.data(), static_cast<int>(inter.size()));
    auto ff_out = k::linear(inter_act.data(), T, kFFN, dw, db, kHidden);  // [T,768]

    // Residual + final layer norm
    for (int i = 0; i < T * kHidden; ++i) hidden[i] += ff_out[i];
    const float* ln2_w = w.get(p + "final_layer_norm.weight");
    const float* ln2_b = w.get(p + "final_layer_norm.bias");
    hidden = k::layer_norm_affine(hidden.data(), T, kHidden, ln2_w, ln2_b);

    return hidden;
}

}  // namespace

bool HubertEncoder::init(const HubertConfig& config) {
    output_dim_ = config.output_dim;
    half_precision_ = config.half_precision;
    config_ = config;
    return true;
}

std::vector<float> HubertEncoder::extract(const float* audio, int num_samples) {
    if (num_samples == 0) return {};

    Weights w;
    if (!w.load(config_.model_path)) {
        fprintf(stderr, "HuBERT: failed to load model '%s'\n",
                config_.model_path.c_str());
        return {};
    }

    // 1. Feature extractor: [num_samples] -> [T, 512]
    auto feats = feature_extractor(w, audio, num_samples);
    int T = static_cast<int>(feats.size()) / kConvDim;
    if (T == 0) return {};

    // 2. Feature projection: [T,512] -> [T,768]
    auto hidden = feature_projection(w, feats, T);

    // 3. Positional conv embedding + encoder layer norm
    auto pos_weight = reconstruct_pos_conv_weight(w);
    hidden = apply_pos_conv(w, hidden, T, pos_weight);

    // 4. Transformer layers. RVC uses layer 9 (v1) or 12 (v2) output.
    int output_layer = config_.use_final_proj ? 9 : 12;
    for (int i = 0; i < output_layer; ++i) {
        hidden = encoder_layer(w, i, std::move(hidden), T);
    }

    // 5. Final projection for v1 (768 -> 256)
    if (config_.use_final_proj) {
        const float* fw = w.get("final_proj.weight");  // [256,768]
        const float* fb = w.has("final_proj.bias") ? w.get("final_proj.bias") : nullptr;
        return k::linear(hidden.data(), T, kHidden, fw, fb, output_dim_);  // [T,256]
    }

    return hidden;  // [T, 768]
}

}  // namespace voxmutatio::content
