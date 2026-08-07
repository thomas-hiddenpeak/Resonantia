// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// VITS synthesizer (SynthesizerTrnMs768NSFsid) — numerically aligned with RVC.
// TextEncoder (relative-position attention) + ResidualCoupling Flow +
// NSF-HiFiGAN generator (SineGen source + MRF upsampling).

#include "voxmutatio/synthesizer/synthesizer.h"
#include "voxmutatio/synthesizer/vits_ops.h"
#include "voxmutatio/content/cuda/kernels.h"
#include "voxmutatio/io/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace voxmutatio::synthesizer {

namespace {

namespace op = voxmutatio::synthesizer::ops;
namespace k = voxmutatio::content::cuda;

constexpr int kHidden = 192;
constexpr int kInter = 192;
constexpr int kNHeads = 2;
constexpr int kNLayers = 6;
constexpr int kFilter = 768;
constexpr int kWindow = 10;
constexpr int kGin = 256;
constexpr float kLRelu = 0.1f;

struct Weights {
    std::shared_ptr<io::SafetensorsLoader> loader;
    std::string prefix = "model.";
    // Cache the mmap'd, header-parsed loader per path: repeated w.load() calls in
    // debug_*/infer reuse it instead of re-mmapping+re-parsing 145MB each time.
    bool load(const std::string& p) {
        static std::mutex m;
        static std::unordered_map<std::string, std::shared_ptr<io::SafetensorsLoader>> cache;
        std::lock_guard<std::mutex> lk(m);
        auto it = cache.find(p);
        if (it == cache.end()) {
            auto l = std::make_shared<io::SafetensorsLoader>();
            if (!l->load(p)) return false;
            it = cache.emplace(p, std::move(l)).first;
        }
        loader = it->second;
        return true;
    }
    const float* get(const std::string& n) const {
        const uint8_t* p = loader->data(prefix + n);
        if (!p) fprintf(stderr, "VITS: missing '%s'\n", (prefix + n).c_str());
        return reinterpret_cast<const float*>(p);
    }
    const io::Tensor* meta(const std::string& n) const {
        return loader->get_tensor(prefix + n);
    }
    bool has(const std::string& n) const { return meta(n) != nullptr; }
};

// Reconstruct weight_norm weight: weight[o,inner] = g[o] * v[o,inner] / ||v[o]||.
std::vector<float> reconstruct_wn(const Weights& w, const std::string& name,
                                  int out_dim, int inner) {
    const float* g = w.get(name + ".weight_g");
    const float* v = w.get(name + ".weight_v");
    std::vector<float> out(out_dim * inner);
    for (int o = 0; o < out_dim; ++o) {
        double n2 = 0.0;
        for (int i = 0; i < inner; ++i) { float x = v[o * inner + i]; n2 += (double)x * x; }
        float inv = (float)(1.0 / std::sqrt(n2));
        float gg = g[o];
        for (int i = 0; i < inner; ++i) out[o * inner + i] = gg * v[o * inner + i] * inv;
    }
    return out;
}

// Conv1d kernel=1 on [Cin,T] -> [Cout,T] via cuda::linear (transpose layout).
std::vector<float> conv1x1(const std::vector<float>& x, int Cin, int T,
                           const float* weight, const float* bias, int Cout) {
    // x is [Cin, T]; transpose to [T, Cin]
    std::vector<float> xt(T * Cin);
    for (int c = 0; c < Cin; ++c)
        for (int t = 0; t < T; ++t) xt[t * Cin + c] = x[c * T + t];
    // linear: [T,Cin] @ weight[Cout,Cin]^T + bias -> [T,Cout]
    auto y = k::linear(xt.data(), T, Cin, weight, bias, Cout);
    // transpose back to [Cout, T]
    std::vector<float> out(Cout * T);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < Cout; ++c) out[c * T + t] = y[t * Cout + c];
    return out;
}

// LayerNorm over channels (x [C,T], normalize each column over C).
void layer_norm_ct(std::vector<float>& x, int C, int T,
                   const float* gamma, const float* beta, float eps = 1e-5f) {
    for (int t = 0; t < T; ++t) {
        double mean = 0.0;
        for (int c = 0; c < C; ++c) mean += x[c * T + t];
        mean /= C;
        double var = 0.0;
        for (int c = 0; c < C; ++c) { double d = x[c * T + t] - mean; var += d * d; }
        var /= C;
        float inv = (float)(1.0 / std::sqrt(var + eps));
        for (int c = 0; c < C; ++c)
            x[c * T + t] = (float)((x[c * T + t] - mean) * inv) * gamma[c] + beta[c];
    }
}

// ---- Relative-position attention helpers (heads_share=True, n_heads_rel=1) ----

// Get relative embeddings for length T. emb: [21, dk]. Returns [2T-1, dk].
std::vector<float> get_rel_emb(const float* emb, int T, int dk) {
    int window = kWindow;
    int max_rel = 2 * window + 1;  // 21
    int pad_length = std::max(T - (window + 1), 0);
    int slice_start = std::max((window + 1) - T, 0);
    int out_len = 2 * T - 1;
    std::vector<float> out(out_len * dk, 0.0f);
    // padded length = max_rel + 2*pad_length; padded[j] = emb[j-pad_length] if in range
    for (int m = 0; m < out_len; ++m) {
        int j = slice_start + m;               // index into padded
        int src = j - pad_length;              // index into emb (0..20)
        if (src >= 0 && src < max_rel)
            std::memcpy(&out[m * dk], &emb[src * dk], dk * sizeof(float));
    }
    return out;
}

// x: [T, 2T-1] -> [T, T]
std::vector<float> rel_to_abs(const std::vector<float>& x, int T) {
    int cols = 2 * T - 1;
    // pad last dim by (0,1): [T, 2T]
    std::vector<float> xp(T * (cols + 1), 0.0f);
    for (int i = 0; i < T; ++i)
        std::memcpy(&xp[i * (cols + 1)], &x[i * cols], cols * sizeof(float));
    // flat [T*2T], pad (0, T-1) -> [T*2T + T-1]
    int flat = T * (cols + 1);
    std::vector<float> xf(flat + (T - 1), 0.0f);
    std::memcpy(xf.data(), xp.data(), flat * sizeof(float));
    // reshape [T+1, 2T-1], slice [:T, T-1:]
    std::vector<float> out(T * T);
    int rc = 2 * T - 1;
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < T; ++j)
            out[i * T + j] = xf[i * rc + (T - 1) + j];
    return out;
}

// x: [T, T] -> [T, 2T-1]
std::vector<float> abs_to_rel(const std::vector<float>& x, int T) {
    // pad last dim (0, T-1): [T, 2T-1]
    int cols = 2 * T - 1;
    std::vector<float> xp(T * cols, 0.0f);
    for (int i = 0; i < T; ++i)
        std::memcpy(&xp[i * cols], &x[i * T], T * sizeof(float));
    // flat [T*(2T-1)], pad (T, 0) -> [T*(2T-1)+T]
    int flat = T * cols;
    std::vector<float> xf(flat + T, 0.0f);
    std::memcpy(&xf[T], xp.data(), flat * sizeof(float));
    // reshape [T, 2T], slice [:, 1:]
    std::vector<float> out(T * cols);
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < cols; ++j)
            out[i * cols + j] = xf[i * (2 * T) + 1 + j];
    return out;
}

// Multi-head relative attention. x: [C=192, T]. Returns [C, T].
std::vector<float> rel_attention(const Weights& w, const std::string& p,
                                 const std::vector<float>& x, int T) {
    int C = kHidden, H = kNHeads, dk = C / H;  // 96
    float scale = 1.0f / std::sqrt((float)dk);

    auto q = conv1x1(x, C, T, w.get(p + "conv_q.weight"), w.get(p + "conv_q.bias"), C);
    auto kk = conv1x1(x, C, T, w.get(p + "conv_k.weight"), w.get(p + "conv_k.bias"), C);
    auto vv = conv1x1(x, C, T, w.get(p + "conv_v.weight"), w.get(p + "conv_v.bias"), C);
    // q,kk,vv are [C, T]. Per head h: channels [h*dk, (h+1)*dk).

    const float* emb_k = w.get(p + "emb_rel_k");  // [1,21,96] -> [21,96]
    const float* emb_v = w.get(p + "emb_rel_v");
    auto rel_k = get_rel_emb(emb_k, T, dk);  // [2T-1, dk]
    auto rel_v = get_rel_emb(emb_v, T, dk);

    std::vector<float> out(C * T, 0.0f);

    for (int h = 0; h < H; ++h) {
        int off = h * dk;
        // scores[i,j] = sum_d q[off+d, i]*kk[off+d, j] * scale
        std::vector<float> scores(T * T);
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < T; ++j) {
                float s = 0.0f;
                for (int d = 0; d < dk; ++d)
                    s += q[(off + d) * T + i] * kk[(off + d) * T + j];
                scores[i * T + j] = s * scale;
            }
        }
        // rel_logits[i,m] = sum_d (q[i,d]*scale) * rel_k[m,d]  (m in 0..2T-2)
        int rl_cols = 2 * T - 1;
        std::vector<float> rel_logits(T * rl_cols);
        for (int i = 0; i < T; ++i)
            for (int m = 0; m < rl_cols; ++m) {
                float s = 0.0f;
                for (int d = 0; d < dk; ++d)
                    s += (q[(off + d) * T + i] * scale) * rel_k[m * dk + d];
                rel_logits[i * rl_cols + m] = s;
            }
        auto scores_local = rel_to_abs(rel_logits, T);  // [T,T]
        for (int i = 0; i < T * T; ++i) scores[i] += scores_local[i];

        // softmax over j
        for (int i = 0; i < T; ++i) {
            float* row = &scores[i * T];
            float mx = row[0];
            for (int j = 1; j < T; ++j) mx = std::max(mx, row[j]);
            float sum = 0.0f;
            for (int j = 0; j < T; ++j) { row[j] = std::exp(row[j] - mx); sum += row[j]; }
            for (int j = 0; j < T; ++j) row[j] /= sum;
        }

        // out_h[i,d] = sum_j p[i,j]*vv[off+d, j]
        std::vector<float> out_h(T * dk);
        for (int i = 0; i < T; ++i)
            for (int d = 0; d < dk; ++d) {
                float s = 0.0f;
                for (int j = 0; j < T; ++j) s += scores[i * T + j] * vv[(off + d) * T + j];
                out_h[i * dk + d] = s;
            }
        // relative values: rel_weights = abs_to_rel(p) [T, 2T-1]; out += rel_weights @ rel_v
        auto rel_weights = abs_to_rel(scores, T);  // [T, 2T-1]
        for (int i = 0; i < T; ++i)
            for (int d = 0; d < dk; ++d) {
                float s = 0.0f;
                for (int m = 0; m < rl_cols; ++m)
                    s += rel_weights[i * rl_cols + m] * rel_v[m * dk + d];
                out_h[i * dk + d] += s;
            }
        // scatter to out [C,T]
        for (int i = 0; i < T; ++i)
            for (int d = 0; d < dk; ++d)
                out[(off + d) * T + i] = out_h[i * dk + d];
    }

    // conv_o
    return conv1x1(out, C, T, w.get(p + "conv_o.weight"), w.get(p + "conv_o.bias"), C);
}

// FFN: conv1d(k=3,pad1) -> relu -> conv1d(k=3,pad1). x [C,T] -> [C,T]
std::vector<float> ffn(const Weights& w, const std::string& p,
                       const std::vector<float>& x, int T) {
    const float* w1 = w.get(p + "conv_1.weight");  // [filter, C, 3]
    const float* b1 = w.get(p + "conv_1.bias");
    auto h = op::conv1d(x.data(), kHidden, T, w1, b1, kFilter, 3, 1, 1, 1, 1);
    // relu
    for (auto& v : h) v = std::max(0.0f, v);
    const float* w2 = w.get(p + "conv_2.weight");  // [C, filter, 3]
    const float* b2 = w.get(p + "conv_2.bias");
    return op::conv1d(h.data(), kFilter, T, w2, b2, kHidden, 3, 1, 1, 1, 1);
}

// TextEncoder -> m_p [inter, T]
std::vector<float> text_encoder(const Weights& w, const float* phone, int T,
                                const int* pitch, std::vector<float>* logs_out) {
    // emb_phone: Linear(768,192). phone [T,768] -> [T,192]
    auto xtc = k::linear(phone, T, 768, w.get("enc_p.emb_phone.weight"),
                         w.get("enc_p.emb_phone.bias"), kHidden);  // [T,192]
    // emb_pitch: Embedding(256,192). Add.
    const float* emb_pitch = w.get("enc_p.emb_pitch.weight");  // [256,192]
    for (int t = 0; t < T; ++t) {
        int idx = pitch[t];
        for (int c = 0; c < kHidden; ++c)
            xtc[t * kHidden + c] += emb_pitch[idx * kHidden + c];
    }
    // x *= sqrt(192); leaky_relu
    float sq = std::sqrt((float)kHidden);
    for (auto& v : xtc) { v *= sq; if (v < 0) v *= kLRelu; }

    // transpose to [C,T]
    std::vector<float> x(kHidden * T);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < kHidden; ++c) x[c * T + t] = xtc[t * kHidden + c];

    // 6 encoder layers
    for (int l = 0; l < kNLayers; ++l) {
        std::string ap = "enc_p.encoder.attn_layers." + std::to_string(l) + ".";
        std::string n1 = "enc_p.encoder.norm_layers_1." + std::to_string(l) + ".";
        std::string fp = "enc_p.encoder.ffn_layers." + std::to_string(l) + ".";
        std::string n2 = "enc_p.encoder.norm_layers_2." + std::to_string(l) + ".";

        auto y = rel_attention(w, ap, x, T);
        for (int i = 0; i < kHidden * T; ++i) x[i] += y[i];
        layer_norm_ct(x, kHidden, T, w.get(n1 + "gamma"), w.get(n1 + "beta"));

        auto f = ffn(w, fp, x, T);
        for (int i = 0; i < kHidden * T; ++i) x[i] += f[i];
        layer_norm_ct(x, kHidden, T, w.get(n2 + "gamma"), w.get(n2 + "beta"));
    }

    // proj: Conv1d(192, 384, 1) -> [384, T]; split m, logs
    auto stats = conv1x1(x, kHidden, T, w.get("enc_p.proj.weight"),
                         w.get("enc_p.proj.bias"), 2 * kInter);
    std::vector<float> m(kInter * T);
    std::memcpy(m.data(), stats.data(), kInter * T * sizeof(float));
    if (logs_out) {
        logs_out->resize(kInter * T);
        std::memcpy(logs_out->data(), stats.data() + kInter * T, kInter * T * sizeof(float));
    }
    return m;
}

// WaveNet (WN) for flow. x [hidden, T], g [gin,1]. Returns [hidden, T].
std::vector<float> wn_forward(const Weights& w, const std::string& p,
                              std::vector<float> x, int T, const float* g) {
    const int H = kHidden, nLayers = 3, kernel = 5;
    // cond: Conv1d(gin, 2H*nLayers, 1). g is [gin,1] -> broadcast to [2H*nLayers, T]
    auto cond_w = reconstruct_wn(w, p + "cond_layer", 2 * H * nLayers, kGin);
    const float* cond_b = w.get(p + "cond_layer.bias");
    // g_cond[c] = bias[c] + sum_gi cond_w[c,gi]*g[gi]
    std::vector<float> g_cond(2 * H * nLayers);
    for (int c = 0; c < 2 * H * nLayers; ++c) {
        float s = cond_b[c];
        for (int gi = 0; gi < kGin; ++gi) s += cond_w[c * kGin + gi] * g[gi];
        g_cond[c] = s;
    }

    std::vector<float> output(H * T, 0.0f);
    for (int i = 0; i < nLayers; ++i) {
        std::string il = p + "in_layers." + std::to_string(i);
        int dilation = 1;  // dilation_rate=1
        int pad = (kernel * dilation - dilation) / 2;  // 2
        auto in_w = reconstruct_wn(w, il, 2 * H, H * kernel);
        const float* in_b = w.get(il + ".bias");
        auto x_in = op::conv1d(x.data(), H, T, in_w.data(), in_b, 2 * H, kernel, 1, pad, dilation, 1);
        // fused add tanh sigmoid multiply with g_l
        int coff = i * 2 * H;
        std::vector<float> acts(H * T);
        for (int c = 0; c < H; ++c)
            for (int t = 0; t < T; ++t) {
                float a = x_in[c * T + t] + g_cond[coff + c];
                float b = x_in[(H + c) * T + t] + g_cond[coff + H + c];
                acts[c * T + t] = std::tanh(a) * (1.0f / (1.0f + std::exp(-b)));
            }
        // res_skip_layer
        std::string rl = p + "res_skip_layers." + std::to_string(i);
        int rs_ch = (i < nLayers - 1) ? 2 * H : H;
        auto rs_w = reconstruct_wn(w, rl, rs_ch, H);
        const float* rs_b = w.get(rl + ".bias");
        auto rs = conv1x1(acts, H, T, rs_w.data(), rs_b, rs_ch);
        if (i < nLayers - 1) {
            for (int c = 0; c < H; ++c)
                for (int t = 0; t < T; ++t) x[c * T + t] += rs[c * T + t];
            for (int c = 0; c < H; ++c)
                for (int t = 0; t < T; ++t) output[c * T + t] += rs[(H + c) * T + t];
        } else {
            for (int i2 = 0; i2 < H * T; ++i2) output[i2] += rs[i2];
        }
    }
    return output;
}

// Flow reverse: z_p [inter, T] -> z [inter, T]
std::vector<float> flow_reverse(const Weights& w, std::vector<float> x, int T,
                                const float* g) {
    const int half = kInter / 2;  // 96
    // 4 coupling layers, each followed by flip. Reverse order.
    for (int f = 3; f >= 0; --f) {
        // Flip (reverse channel order)
        std::vector<float> flipped(kInter * T);
        for (int c = 0; c < kInter; ++c)
            std::memcpy(&flipped[c * T], &x[(kInter - 1 - c) * T], T * sizeof(float));
        x = std::move(flipped);

        // ResidualCouplingLayer reverse (mean_only)
        std::string p = "flow.flows." + std::to_string(f * 2) + ".";
        // split x0 (first half), x1 (second half)
        std::vector<float> x0(half * T), x1(half * T);
        std::memcpy(x0.data(), x.data(), half * T * sizeof(float));
        std::memcpy(x1.data(), x.data() + half * T, half * T * sizeof(float));

        // h = pre(x0): Conv1d(half, hidden, 1)
        auto h = conv1x1(x0, half, T, w.get(p + "pre.weight"), w.get(p + "pre.bias"), kHidden);
        // h = enc(h, g)  [WN]
        h = wn_forward(w, p + "enc.", std::move(h), T, g);
        // stats = post(h): Conv1d(hidden, half, 1) (mean_only)
        auto m = conv1x1(h, kHidden, T, w.get(p + "post.weight"), w.get(p + "post.bias"), half);
        // x1 = (x1 - m) * exp(-logs); logs=0 -> x1 = x1 - m
        for (int i = 0; i < half * T; ++i) x1[i] -= m[i];
        // x = cat(x0, x1)
        std::memcpy(x.data(), x0.data(), half * T * sizeof(float));
        std::memcpy(x.data() + half * T, x1.data(), half * T * sizeof(float));
    }
    return x;
}

// SineGen: f0 [T] -> har_source [T*upp]. harmonic_num=0, deterministic (noise=0).
std::vector<float> sine_source(const float* f0, int T, int upp, int sr,
                               const Weights& w) {
    // rad[t,i] = f0[t]/sr * (i+1), i in 0..upp-1
    // rad2[t] = fmod(rad[t, upp-1] + 0.5, 1.0) - 0.5
    // rad_acc[t] = cumsum_t(rad2).fmod(1.0)
    // rad[t,i] += pad(rad_acc, shift down 1)  (rad_acc[t-1], 0 for t=0)
    // sine[t*upp + i] = sin(2pi * rad[t,i])   (dim=1, harmonic 1, rand_ini=0)
    std::vector<float> rad2(T);
    for (int t = 0; t < T; ++t) {
        float last = f0[t] / sr * upp;  // rad[t, upp-1] = f0/sr*upp
        rad2[t] = std::fmod(last + 0.5f, 1.0f) - 0.5f;
    }
    std::vector<float> rad_acc(T);
    float acc = 0.0f;
    for (int t = 0; t < T; ++t) {
        acc += rad2[t];
        rad_acc[t] = std::fmod(acc, 1.0f);
    }
    // sine amp 0.1
    std::vector<float> sine(T * upp);
    for (int t = 0; t < T; ++t) {
        float shift = (t > 0) ? rad_acc[t - 1] : 0.0f;
        for (int i = 0; i < upp; ++i) {
            float rad = f0[t] / sr * (i + 1) + shift;
            sine[t * upp + i] = std::sin(2.0f * (float)M_PI * rad) * 0.1f;
        }
    }
    // uv: f0>0 per frame, interpolated (nearest) to upp
    // sine_waves = sine * uv + noise(=0)
    for (int t = 0; t < T; ++t) {
        float uv = (f0[t] > 0) ? 1.0f : 0.0f;
        for (int i = 0; i < upp; ++i) sine[t * upp + i] *= uv;
    }
    // SourceModule: l_linear(1,1) + tanh
    float lw = *w.get("dec.m_source.l_linear.weight");
    float lb = *w.get("dec.m_source.l_linear.bias");
    for (auto& s : sine) s = std::tanh(s * lw + lb);
    return sine;  // [T*upp]
}

// ResBlock1: 3 pairs (leaky_relu -> conv dilated -> leaky_relu -> conv) + residual
std::vector<float> resblock1(const Weights& w, const std::string& p,
                             std::vector<float> x, int C, int T, int kernel,
                             const std::vector<int>& dil) {
    for (int j = 0; j < 3; ++j) {
        auto xt = x;
        op::leaky_relu_inplace(xt.data(), C * T, kLRelu);
        std::string c1 = p + "convs1." + std::to_string(j);
        auto w1 = reconstruct_wn(w, c1, C, C * kernel);
        int pad1 = dil[j] * (kernel - 1) / 2;
        xt = op::conv1d(xt.data(), C, T, w1.data(), w.get(c1 + ".bias"), C, kernel, 1, pad1, dil[j], 1);
        op::leaky_relu_inplace(xt.data(), C * T, kLRelu);
        std::string c2 = p + "convs2." + std::to_string(j);
        auto w2 = reconstruct_wn(w, c2, C, C * kernel);
        int pad2 = (kernel - 1) / 2;
        xt = op::conv1d(xt.data(), C, T, w2.data(), w.get(c2 + ".bias"), C, kernel, 1, pad2, 1, 1);
        for (int i = 0; i < C * T; ++i) x[i] += xt[i];
    }
    return x;
}

// GeneratorNSF: z [inter, T], f0 [T] -> audio [T*400]
std::vector<float> generator(const Weights& w, std::vector<float> x, int T,
                             const float* f0, const float* g, int sr) {
    const int upsample_rates[4] = {10, 10, 2, 2};
    const int upsample_kernels[4] = {16, 16, 4, 4};
    const int resblock_k[3] = {3, 7, 11};
    const std::vector<std::vector<int>> resblock_d = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    int upp = 400;

    // Sine source [T*upp]
    auto har = sine_source(f0, T, upp, sr, w);  // [T*upp], channels-first [1, T*upp]

    // conv_pre: Conv1d(192, 512, 7, pad 3)
    auto xc = op::conv1d(x.data(), kInter, T, w.get("dec.conv_pre.weight"),
                         w.get("dec.conv_pre.bias"), 512, 7, 1, 3, 1, 1);
    int C = 512, L = T;
    // + cond(g): Conv1d(gin,512,1). g[gin] -> add per channel
    {
        const float* cw = w.get("dec.cond.weight");  // [512,256,1]
        const float* cb = w.get("dec.cond.bias");
        for (int c = 0; c < 512; ++c) {
            float s = cb[c];
            for (int gi = 0; gi < kGin; ++gi) s += cw[c * kGin + gi] * g[gi];
            for (int t = 0; t < L; ++t) xc[c * L + t] += s;
        }
    }

    for (int i = 0; i < 4; ++i) {
        // leaky_relu
        op::leaky_relu_inplace(xc.data(), C * L, kLRelu);
        // ups[i]: ConvTranspose1d(C, C/2, k, stride, pad=(k-u)/2), weight_norm
        int Cout = C / 2;
        int u = upsample_rates[i], kk = upsample_kernels[i], pad = (kk - u) / 2;
        auto up_w = reconstruct_wn(w, "dec.ups." + std::to_string(i), C, Cout * kk);
        // weight layout [C, Cout, kk]
        auto up = op::conv_transpose1d(xc.data(), C, L, up_w.data(),
                                       w.get("dec.ups." + std::to_string(i) + ".bias"),
                                       Cout, kk, u, pad);
        int Lout = (L - 1) * u - 2 * pad + kk;

        // noise_convs[i](har): downsample source
        std::vector<float> xsrc;
        if (i + 1 < 4) {
            int stride_f0 = 1;
            for (int j = i + 1; j < 4; ++j) stride_f0 *= upsample_rates[j];
            int nk = stride_f0 * 2, npad = stride_f0 / 2;
            xsrc = op::conv1d(har.data(), 1, (int)har.size(),
                              w.get("dec.noise_convs." + std::to_string(i) + ".weight"),
                              w.get("dec.noise_convs." + std::to_string(i) + ".bias"),
                              Cout, nk, stride_f0, npad, 1, 1);
        } else {
            xsrc = op::conv1d(har.data(), 1, (int)har.size(),
                              w.get("dec.noise_convs." + std::to_string(i) + ".weight"),
                              w.get("dec.noise_convs." + std::to_string(i) + ".bias"),
                              Cout, 1, 1, 0, 1, 1);
        }
        // x = up + xsrc (align lengths)
        int use_len = std::min(Lout, (int)xsrc.size() / Cout);
        std::vector<float> xn(Cout * use_len);
        for (int c = 0; c < Cout; ++c)
            for (int t = 0; t < use_len; ++t)
                xn[c * use_len + t] = up[c * Lout + t] + xsrc[c * (xsrc.size() / Cout) + t];

        // MRF: sum of 3 resblocks / 3
        std::vector<float> acc(Cout * use_len, 0.0f);
        for (int jr = 0; jr < 3; ++jr) {
            std::string rp = "dec.resblocks." + std::to_string(i * 3 + jr) + ".";
            auto rb = resblock1(w, rp, xn, Cout, use_len, resblock_k[jr], resblock_d[jr]);
            for (int idx = 0; idx < Cout * use_len; ++idx) acc[idx] += rb[idx];
        }
        for (auto& v : acc) v /= 3.0f;

        xc = std::move(acc);
        C = Cout;
        L = use_len;
    }

    // final leaky_relu + conv_post (no bias) + tanh
    op::leaky_relu_inplace(xc.data(), C * L, kLRelu);
    auto out = op::conv1d(xc.data(), C, L, w.get("dec.conv_post.weight"), nullptr, 1, 7, 1, 3, 1, 1);
    op::tanh_inplace(out.data(), (int)out.size());
    return out;  // [L] audio
}

}  // namespace

bool Synthesizer::init(const SynthesizerConfig& config) {
    config_ = config;
    sample_rate_ = config.sample_rate;
    version_ = config.version;
    has_f0_ = config.has_f0;
    num_speakers_ = config.spk_embed_dim;
    return true;
}

std::vector<float> Synthesizer::debug_text_encoder(const float* features, int frames,
                                                   const int* pitch_coarse) {
    Weights w;
    if (!w.load(config_.model_path)) return {};
    return text_encoder(w, features, frames, pitch_coarse, nullptr);
}

std::vector<float> Synthesizer::debug_flow(const float* features, int frames,
                                           const int* pitch_coarse, int speaker_id) {
    Weights w;
    if (!w.load(config_.model_path)) return {};
    auto m_p = text_encoder(w, features, frames, pitch_coarse, nullptr);
    const float* emb_g = w.get("emb_g.weight");  // [num_spk, gin]
    std::vector<float> g(kGin);
    for (int i = 0; i < kGin; ++i) g[i] = emb_g[speaker_id * kGin + i];
    return flow_reverse(w, std::move(m_p), frames, g.data());
}

std::vector<float> Synthesizer::debug_har(const float* f0, int frames) {
    Weights w;
    if (!w.load(config_.model_path)) return {};
    return sine_source(f0, frames, 400, sample_rate_, w);
}

std::vector<float> Synthesizer::debug_convpre(const float* z, int frames, int speaker_id) {
    Weights w;
    if (!w.load(config_.model_path)) return {};
    const float* emb_g = w.get("emb_g.weight");
    std::vector<float> g(kGin);
    for (int i = 0; i < kGin; ++i) g[i] = emb_g[speaker_id * kGin + i];
    std::vector<float> x(z, z + kInter * frames);
    auto xc = op::conv1d(x.data(), kInter, frames, w.get("dec.conv_pre.weight"),
                         w.get("dec.conv_pre.bias"), 512, 7, 1, 3, 1, 1);
    const float* cw = w.get("dec.cond.weight");
    const float* cb = w.get("dec.cond.bias");
    for (int c = 0; c < 512; ++c) {
        float s = cb[c];
        for (int gi = 0; gi < kGin; ++gi) s += cw[c * kGin + gi] * g[gi];
        for (int t = 0; t < frames; ++t) xc[c * frames + t] += s;
    }
    return xc;
}

std::vector<float> Synthesizer::debug_gen_stage0(const float* z, const float* f0,
                                                 int frames, int speaker_id, int which) {
    Weights w;
    if (!w.load(config_.model_path)) return {};
    const float* emb_g = w.get("emb_g.weight");
    std::vector<float> g(kGin);
    for (int i = 0; i < kGin; ++i) g[i] = emb_g[speaker_id * kGin + i];

    auto har = sine_source(f0, frames, 400, sample_rate_, w);
    std::vector<float> x(z, z + kInter * frames);
    auto xc = op::conv1d(x.data(), kInter, frames, w.get("dec.conv_pre.weight"),
                         w.get("dec.conv_pre.bias"), 512, 7, 1, 3, 1, 1);
    const float* cw = w.get("dec.cond.weight");
    const float* cb = w.get("dec.cond.bias");
    for (int c = 0; c < 512; ++c) {
        float s = cb[c];
        for (int gi = 0; gi < kGin; ++gi) s += cw[c * kGin + gi] * g[gi];
        for (int t = 0; t < frames; ++t) xc[c * frames + t] += s;
    }
    int C = 512, L = frames;
    op::leaky_relu_inplace(xc.data(), C * L, kLRelu);
    int Cout = 256, u = 10, kk = 16, pad = 3;
    auto up_w = reconstruct_wn(w, "dec.ups.0", C, Cout * kk);
    auto up = op::conv_transpose1d(xc.data(), C, L, up_w.data(),
                                   w.get("dec.ups.0.bias"), Cout, kk, u, pad);
    int Lout = (L - 1) * u - 2 * pad + kk;
    if (which == 0) return up;  // [256, Lout]

    int stride_f0 = 10 * 2 * 2;  // 40
    auto xsrc = op::conv1d(har.data(), 1, (int)har.size(),
                           w.get("dec.noise_convs.0.weight"), w.get("dec.noise_convs.0.bias"),
                           Cout, stride_f0 * 2, stride_f0, stride_f0 / 2, 1, 1);
    if (which == 1) return xsrc;

    int use_len = std::min(Lout, (int)xsrc.size() / Cout);
    std::vector<float> xn(Cout * use_len);
    for (int c = 0; c < Cout; ++c)
        for (int t = 0; t < use_len; ++t)
            xn[c * use_len + t] = up[c * Lout + t] + xsrc[c * (xsrc.size() / Cout) + t];
    const std::vector<std::vector<int>> rd = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    const int rk[3] = {3, 7, 11};
    std::vector<float> acc(Cout * use_len, 0.0f);
    for (int jr = 0; jr < 3; ++jr) {
        auto rb = resblock1(w, "dec.resblocks." + std::to_string(jr) + ".", xn, Cout, use_len, rk[jr], rd[jr]);
        for (int i = 0; i < Cout * use_len; ++i) acc[i] += rb[i];
    }
    for (auto& v : acc) v /= 3.0f;
    return acc;
}

std::vector<float> Synthesizer::debug_decode(const float* z, const float* f0,
                                             int frames, int speaker_id) {
    Weights w;
    if (!w.load(config_.model_path)) return {};
    const float* emb_g = w.get("emb_g.weight");
    std::vector<float> g(kGin);
    for (int i = 0; i < kGin; ++i) g[i] = emb_g[speaker_id * kGin + i];
    std::vector<float> x(z, z + kInter * frames);
    return generator(w, std::move(x), frames, f0, g.data(), sample_rate_);
}

std::vector<float> Synthesizer::debug_encp(const float* features, int frames,
                                           const int* pitch_coarse, bool want_logs) {
    Weights w;
    if (!w.load(config_.model_path)) return {};
    std::vector<float> logs;
    auto m_p = text_encoder(w, features, frames, pitch_coarse, want_logs ? &logs : nullptr);
    if (want_logs) m_p.insert(m_p.end(), logs.begin(), logs.end());  // [m_p ; logs_p]
    return m_p;
}

std::vector<float> Synthesizer::debug_flow_reverse(const float* x, int frames, int speaker_id) {
    Weights w;
    if (!w.load(config_.model_path)) return {};
    const float* emb_g = w.get("emb_g.weight");
    std::vector<float> g(kGin);
    for (int i = 0; i < kGin; ++i) g[i] = emb_g[speaker_id * kGin + i];
    std::vector<float> in(x, x + kInter * frames);
    return flow_reverse(w, std::move(in), frames, g.data());
}

AudioBuffer Synthesizer::infer(const float* features, int frames,
                               const float* pitch, const float* pitchf,
                               int speaker_id) {
    AudioBuffer result;
    result.sample_rate = sample_rate_;

    Weights w;
    if (!w.load(config_.model_path)) return result;

    // pitch here is coarse (int stored as float); pitchf is f0 in Hz.
    std::vector<int> pitch_coarse(frames);
    for (int t = 0; t < frames; ++t) pitch_coarse[t] = (int)std::lround(pitch[t]);

    // Speaker embedding
    const float* emb_g = w.get("emb_g.weight");
    std::vector<float> g(kGin);
    for (int i = 0; i < kGin; ++i) g[i] = emb_g[speaker_id * kGin + i];

    // enc_p -> m_p  (deterministic: z_p = m_p)
    auto m_p = text_encoder(w, features, frames, pitch_coarse.data(), nullptr);
    // flow reverse -> z
    auto z = flow_reverse(w, m_p, frames, g.data());
    // generator -> audio
    auto audio = generator(w, std::move(z), frames, pitchf, g.data(), sample_rate_);

    result.data = std::move(audio);
    return result;
}

AudioBuffer Synthesizer::infer_stream(const float* features, int frames,
                                      const float* pitch, const float* pitchf,
                                      int speaker_id, int skip_head, int return_length) {
    return infer(features, frames, pitch, pitchf, speaker_id);
}

}  // namespace voxmutatio::synthesizer
