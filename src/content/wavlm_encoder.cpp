// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// WavLM-Base+ content encoder — numerically aligned with HuggingFace
// microsoft/wavlm-base-plus. Shares the wav2vec2 front-end with our HuBERT
// encoder (7-layer conv extractor, feature projection, positional conv) and
// adds WavLM's gated relative position bias in self-attention.

#include "voxmutatio/content/wavlm_encoder.h"
#include "voxmutatio/content/cuda/kernels.h"
#include "voxmutatio/io/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace voxmutatio::content {

namespace {

namespace k = voxmutatio::content::cuda;

constexpr int kHidden = 768;
constexpr int kFFN = 3072;
constexpr int kNumHeads = 12;
constexpr int kHeadDim = kHidden / kNumHeads;  // 64
constexpr int kConvDim = 512;
constexpr int kNumConvLayers = 7;
constexpr int kPosConvKernel = 128;
constexpr int kPosConvGroups = 16;
constexpr int kNumBuckets = 320;     // config.num_buckets
constexpr int kMaxDistance = 800;    // config.max_bucket_distance
const int kConvKernels[kNumConvLayers] = {10, 3, 3, 3, 3, 2, 2};
const int kConvStrides[kNumConvLayers] = {5, 2, 2, 2, 2, 2, 2};

struct Weights {
    io::SafetensorsLoader loader;
    bool load(const std::string& path) { return loader.load(path); }
    const float* get(const std::string& name) const {
        const uint8_t* p = loader.data(name);
        if (!p) std::fprintf(stderr, "WavLM: missing tensor '%s'\n", name.c_str());
        return reinterpret_cast<const float*>(p);
    }
    bool has(const std::string& name) const { return loader.get_tensor(name) != nullptr; }
};

// Transpose [rows, cols] -> [cols, rows]
std::vector<float> transpose(const std::vector<float>& in, int rows, int cols) {
    std::vector<float> out(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) out[static_cast<size_t>(c) * rows + r] = in[static_cast<size_t>(r) * cols + c];
    return out;
}

// pos_conv weight_norm: weight[o,i,k] = g[k] * v[o,i,k] / ||v[:,:,k]|| (norm over dims 0,1).
std::vector<float> reconstruct_pos_conv_weight(const Weights& w) {
    const float* g = w.get("encoder.pos_conv_embed.conv.parametrizations.weight.original0");
    const float* v = w.get("encoder.pos_conv_embed.conv.parametrizations.weight.original1");
    if (!g || !v) return {};
    const int out_ch = kHidden, in_per_group = kHidden / kPosConvGroups, K = kPosConvKernel;
    std::vector<float> norm(K, 0.0f);
    for (int o = 0; o < out_ch; ++o)
        for (int i = 0; i < in_per_group; ++i)
            for (int kk = 0; kk < K; ++kk) {
                float val = v[(o * in_per_group + i) * K + kk];
                norm[kk] += val * val;
            }
    for (int kk = 0; kk < K; ++kk) norm[kk] = std::sqrt(norm[kk]);
    std::vector<float> weight(static_cast<size_t>(out_ch) * in_per_group * K);
    for (int o = 0; o < out_ch; ++o)
        for (int i = 0; i < in_per_group; ++i)
            for (int kk = 0; kk < K; ++kk) {
                int idx = (o * in_per_group + i) * K + kk;
                weight[idx] = g[kk] * v[idx] / norm[kk];
            }
    return weight;
}

// Feature extractor: raw waveform [num_samples] -> [T, 512].
std::vector<float> feature_extractor(const Weights& w, const float* audio, int num_samples) {
    std::vector<float> x(audio, audio + num_samples);
    int in_ch = 1, in_len = num_samples;
    for (int layer = 0; layer < kNumConvLayers; ++layer) {
        std::string prefix = "feature_extractor.conv_layers." + std::to_string(layer) + ".";
        const float* conv_w = w.get(prefix + "conv.weight");
        int out_ch = kConvDim, kernel = kConvKernels[layer], stride = kConvStrides[layer];
        auto conv_out = k::conv1d_strided(x.data(), in_ch, in_len, conv_w, nullptr, out_ch, kernel, stride);
        int out_len = (in_len - kernel) / stride + 1;
        if (layer == 0) {  // group norm on layer 0 (feat_extract_norm="group")
            std::vector<float> normed(static_cast<size_t>(out_ch) * out_len);
            k::group_norm(conv_out.data(), normed.data(), out_ch, out_len, out_ch,
                          w.get(prefix + "layer_norm.weight"), w.get(prefix + "layer_norm.bias"));
            conv_out = std::move(normed);
        }
        std::vector<float> activated(conv_out.size());
        k::gelu_exact(conv_out.data(), activated.data(), static_cast<int>(conv_out.size()));
        x = std::move(activated);
        in_ch = out_ch;
        in_len = out_len;
    }
    return transpose(x, in_ch, in_len);  // [T,512]
}

// Feature projection: [T,512] -> layer_norm -> linear -> [T,768].
std::vector<float> feature_projection(const Weights& w, const std::vector<float>& feats, int T) {
    auto normed = k::layer_norm_affine(feats.data(), T, kConvDim,
                    w.get("feature_projection.layer_norm.weight"),
                    w.get("feature_projection.layer_norm.bias"));
    return k::linear(normed.data(), T, kConvDim, w.get("feature_projection.projection.weight"),
                     w.get("feature_projection.projection.bias"), kHidden);
}

// Positional conv embedding + residual + encoder layer norm.
std::vector<float> apply_pos_conv(const Weights& w, const std::vector<float>& hidden, int T,
                                  const std::vector<float>& pos_weight) {
    auto x_cf = transpose(hidden, T, kHidden);
    auto conv_out = k::conv1d_grouped(x_cf.data(), kHidden, T, pos_weight.data(),
                    w.get("encoder.pos_conv_embed.conv.bias"),
                    kHidden, kPosConvKernel, 1, 64, kPosConvGroups);
    int conv_len = T + 2 * 64 - kPosConvKernel + 1;  // T+1
    int pos_len = conv_len - 1;                       // SamePad: even kernel drops last col
    std::vector<float> trimmed(static_cast<size_t>(kHidden) * pos_len);
    for (int c = 0; c < kHidden; ++c)
        for (int t = 0; t < pos_len; ++t) trimmed[static_cast<size_t>(c) * pos_len + t] = conv_out[static_cast<size_t>(c) * conv_len + t];
    std::vector<float> pos_act(trimmed.size());
    k::gelu_exact(trimmed.data(), pos_act.data(), static_cast<int>(trimmed.size()));
    auto pos_tf = transpose(pos_act, kHidden, pos_len);
    std::vector<float> summed(static_cast<size_t>(T) * kHidden);
    for (size_t i = 0; i < summed.size(); ++i) summed[i] = hidden[i] + pos_tf[i];
    return k::layer_norm_affine(summed.data(), T, kHidden,
                                w.get("encoder.layer_norm.weight"), w.get("encoder.layer_norm.bias"));
}

// WavLM bidirectional relative-position bucket (config num_buckets=320, max_distance=800).
int relative_position_bucket(int rel) {
    const int num_buckets = kNumBuckets / 2;   // 160
    const int max_exact = num_buckets / 2;     // 80
    int bucket = (rel > 0) ? num_buckets : 0;
    int n = std::abs(rel);
    if (n < max_exact) {
        bucket += n;
    } else {
        double large = std::log(static_cast<double>(n) / max_exact) /
                       std::log(static_cast<double>(kMaxDistance) / max_exact) * (num_buckets - max_exact);
        int big = max_exact + static_cast<int>(large);  // trunc toward zero
        bucket += std::min(big, num_buckets - 1);
    }
    return bucket;
}

// Position bias [num_heads, T, T] from layer 0's rel_attn_embed (computed once, carried).
std::vector<float> compute_position_bias(const float* rel_embed, int T) {
    std::vector<float> pos(static_cast<size_t>(kNumHeads) * T * T);
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < T; ++j) {
            int b = relative_position_bucket(j - i);  // memory - context
            for (int h = 0; h < kNumHeads; ++h)
                pos[(static_cast<size_t>(h) * T + i) * T + j] = rel_embed[b * kNumHeads + h];
        }
    return pos;
}

// One WavLM encoder layer (post-norm) with gated relative position bias.
std::vector<float> wavlm_encoder_layer(const Weights& w, int layer_idx,
                                       std::vector<float> hidden, int T,
                                       const std::vector<float>& position_bias) {
    std::string p = "encoder.layers." + std::to_string(layer_idx) + ".";

    auto q = k::linear(hidden.data(), T, kHidden, w.get(p + "attention.q_proj.weight"),
                       w.get(p + "attention.q_proj.bias"), kHidden);
    auto kk = k::linear(hidden.data(), T, kHidden, w.get(p + "attention.k_proj.weight"),
                        w.get(p + "attention.k_proj.bias"), kHidden);
    auto vv = k::linear(hidden.data(), T, kHidden, w.get(p + "attention.v_proj.weight"),
                        w.get(p + "attention.v_proj.bias"), kHidden);

    // Per-head, per-query gate from this layer's hidden states.
    const float* grw = w.get(p + "attention.gru_rel_pos_linear.weight");  // [8,64]
    const float* grb = w.get(p + "attention.gru_rel_pos_linear.bias");    // [8]
    const float* gconst = w.get(p + "attention.gru_rel_pos_const");       // [1,12,1,1]
    std::vector<float> gate(static_cast<size_t>(kNumHeads) * T);          // [H,T]
    for (int h = 0; h < kNumHeads; ++h)
        for (int t = 0; t < T; ++t) {
            const float* hd = &hidden[static_cast<size_t>(t) * kHidden + h * kHeadDim];  // head slice
            float proj[8];
            for (int o = 0; o < 8; ++o) {
                double acc = grb[o];
                for (int d = 0; d < kHeadDim; ++d) acc += static_cast<double>(grw[o * kHeadDim + d]) * hd[d];
                proj[o] = static_cast<float>(acc);
            }
            float s0 = proj[0] + proj[1] + proj[2] + proj[3];
            float s1 = proj[4] + proj[5] + proj[6] + proj[7];
            float ga = 1.0f / (1.0f + std::exp(-s0));
            float gb = 1.0f / (1.0f + std::exp(-s1));
            gate[static_cast<size_t>(h) * T + t] = ga * (gb * gconst[h] - 1.0f) + 2.0f;
        }

    // Attention per head: softmax((q·k)/sqrt(dh) + gate*pos_bias) · v.
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    std::vector<float> ctx(static_cast<size_t>(T) * kHidden);
    std::vector<float> scores(T);
    for (int h = 0; h < kNumHeads; ++h) {
        int off = h * kHeadDim;
        for (int i = 0; i < T; ++i) {
            const float* qi = &q[static_cast<size_t>(i) * kHidden + off];
            float gi = gate[static_cast<size_t>(h) * T + i];
            const float* pbi = &position_bias[(static_cast<size_t>(h) * T + i) * T];
            float mx = -1e30f;
            for (int j = 0; j < T; ++j) {
                const float* kj = &kk[static_cast<size_t>(j) * kHidden + off];
                double dot = 0.0;
                for (int d = 0; d < kHeadDim; ++d) dot += static_cast<double>(qi[d]) * kj[d];
                float s = static_cast<float>(dot) * scale + gi * pbi[j];
                scores[j] = s;
                if (s > mx) mx = s;
            }
            double sum = 0.0;
            for (int j = 0; j < T; ++j) { scores[j] = std::exp(scores[j] - mx); sum += scores[j]; }
            float inv = static_cast<float>(1.0 / (sum + 1e-20));
            float* co = &ctx[static_cast<size_t>(i) * kHidden + off];
            for (int d = 0; d < kHeadDim; ++d) co[d] = 0.0f;
            for (int j = 0; j < T; ++j) {
                float wgt = scores[j] * inv;
                const float* vj = &vv[static_cast<size_t>(j) * kHidden + off];
                for (int d = 0; d < kHeadDim; ++d) co[d] += wgt * vj[d];
            }
        }
    }

    auto attn_out = k::linear(ctx.data(), T, kHidden, w.get(p + "attention.out_proj.weight"),
                              w.get(p + "attention.out_proj.bias"), kHidden);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += attn_out[i];
    hidden = k::layer_norm_affine(hidden.data(), T, kHidden,
                w.get(p + "layer_norm.weight"), w.get(p + "layer_norm.bias"));

    // Feed forward.
    auto inter = k::linear(hidden.data(), T, kHidden,
                w.get(p + "feed_forward.intermediate_dense.weight"),
                w.get(p + "feed_forward.intermediate_dense.bias"), kFFN);
    std::vector<float> inter_act(inter.size());
    k::gelu_exact(inter.data(), inter_act.data(), static_cast<int>(inter.size()));
    auto ff_out = k::linear(inter_act.data(), T, kFFN,
                w.get(p + "feed_forward.output_dense.weight"),
                w.get(p + "feed_forward.output_dense.bias"), kHidden);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += ff_out[i];
    hidden = k::layer_norm_affine(hidden.data(), T, kHidden,
                w.get(p + "final_layer_norm.weight"), w.get(p + "final_layer_norm.bias"));
    return hidden;
}

}  // namespace

bool WavlmEncoder::init(const WavlmConfig& config) {
    output_dim_ = config.output_dim;
    half_precision_ = config.half_precision;
    config_ = config;
    return true;
}

std::vector<float> WavlmEncoder::extract(const float* audio, int num_samples) {
    if (num_samples == 0) return {};
    Weights w;
    if (!w.load(config_.model_path)) {
        std::fprintf(stderr, "WavLM: failed to load model '%s'\n", config_.model_path.c_str());
        return {};
    }

    auto feats = feature_extractor(w, audio, num_samples);  // [T,512]
    int T = static_cast<int>(feats.size()) / kConvDim;
    if (T == 0) return {};

    auto hidden = feature_projection(w, feats, T);          // [T,768]
    auto pos_weight = reconstruct_pos_conv_weight(w);
    hidden = apply_pos_conv(w, hidden, T, pos_weight);      // [T,768]

    // Position bias is computed once (layer 0's rel_attn_embed) and reused by all layers.
    const float* rel0 = w.get("encoder.layers.0.attention.rel_attn_embed.weight");  // [320,12]
    auto position_bias = compute_position_bias(rel0, T);

    int layers = std::min(config_.num_layers, 12);
    for (int i = 0; i < layers; ++i)
        hidden = wavlm_encoder_layer(w, i, std::move(hidden), T, position_bias);

    return hidden;  // [T, 768]
}

}  // namespace voxmutatio::content

