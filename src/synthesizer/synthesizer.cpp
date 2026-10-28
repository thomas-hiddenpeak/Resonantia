// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// VITS synthesizer (SynthesizerTrnMs768NSFsid) — numerically aligned with RVC.
// TextEncoder (relative-position attention) + ResidualCoupling Flow +
// NSF-HiFiGAN generator (SineGen source + MRF upsampling).

#include "voxmutatio/synthesizer/synthesizer.h"
#include "sampling.h"
#include "relative_attention.h"
#include "voxmutatio/synthesizer/vits_ops.h"
#include "voxmutatio/content/cuda/kernels.h"
#include "voxmutatio/io/safetensors.h"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
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
    std::shared_ptr<const io::SafetensorsLoader> loader;
    std::string prefix = "model.";
    bool load(const std::string& p) {
        auto staged = std::make_shared<io::SafetensorsLoader>();
        if (!staged->load_snapshot(p, Synthesizer::kMaxCheckpointBytes)) return false;
        loader = std::move(staged);
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

using CheckpointIdentity = io::SafetensorsSourceIdentity;

bool read_checkpoint_identity(const std::string& path, CheckpointIdentity& identity) {
    struct stat status {};
    if (stat(path.c_str(), &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_size <= 0) {
        return false;
    }
    identity.device = static_cast<std::uintmax_t>(status.st_dev);
    identity.inode = static_cast<std::uintmax_t>(status.st_ino);
    identity.mode = static_cast<std::uintmax_t>(status.st_mode);
    identity.size = static_cast<std::uintmax_t>(status.st_size);
    identity.modified_seconds = static_cast<std::int64_t>(status.st_mtim.tv_sec);
    identity.modified_nanoseconds = static_cast<std::int64_t>(status.st_mtim.tv_nsec);
    identity.changed_seconds = static_cast<std::int64_t>(status.st_ctim.tv_sec);
    identity.changed_nanoseconds = static_cast<std::int64_t>(status.st_ctim.tv_nsec);
    return true;
}

bool checkpoint_is_current(const std::string& path,
                           const CheckpointIdentity& expected) {
    CheckpointIdentity current;
    return read_checkpoint_identity(path, current) && current == expected;
}

bool resolve_checkpoint_path(const std::string& configured,
                             std::string& resolved) {
    std::error_code error;
    const std::filesystem::path absolute =
        std::filesystem::absolute(configured, error);
    if (error || absolute.empty()) return false;
    resolved = absolute.lexically_normal().string();
    return !resolved.empty();
}

bool expect_finite_tensor(const Weights& w, const std::string& name,
                          std::initializer_list<int64_t> shape) {
    const io::Tensor* tensor = w.meta(name);
    if (!tensor) {
        fprintf(stderr, "VITS: missing required tensor 'model.%s'\n", name.c_str());
        return false;
    }
    if (tensor->shape != std::vector<int64_t>(shape)) {
        fprintf(stderr, "VITS: tensor 'model.%s' has incompatible shape\n", name.c_str());
        return false;
    }
    std::size_t elements = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0 || static_cast<std::uint64_t>(dim) >
                            std::numeric_limits<std::size_t>::max()) {
            fprintf(stderr, "VITS: tensor 'model.%s' has an invalid dimension\n",
                    name.c_str());
            return false;
        }
        const std::size_t size = static_cast<std::size_t>(dim);
        if (elements > std::numeric_limits<std::size_t>::max() / size) {
            fprintf(stderr, "VITS: tensor 'model.%s' size overflows\n", name.c_str());
            return false;
        }
        elements *= size;
    }
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        tensor->dtype != "F32" ||
        tensor->data_nbytes != elements * sizeof(float)) {
        fprintf(stderr, "VITS: tensor 'model.%s' is not F32 with the expected size\n",
                name.c_str());
        return false;
    }
    const std::uint8_t* bytes = w.loader->data(w.prefix + name);
    if (!bytes || reinterpret_cast<std::uintptr_t>(bytes) % alignof(float) != 0) {
        fprintf(stderr, "VITS: tensor 'model.%s' has unreadable or unaligned data\n",
                name.c_str());
        return false;
    }
    const auto* values = reinterpret_cast<const float*>(bytes);
    for (std::size_t i = 0; i < elements; ++i) {
        if (!std::isfinite(values[i])) {
            fprintf(stderr, "VITS: tensor 'model.%s' contains a non-finite value\n",
                    name.c_str());
            return false;
        }
    }
    return true;
}

bool expect_weight_norm(const Weights& w, const std::string& name,
                        int out_dim,
                        std::initializer_list<int64_t> value_shape) {
    if (out_dim <= 0 || value_shape.size() == 0 ||
        *value_shape.begin() != out_dim ||
        !expect_finite_tensor(w, name + ".weight_g", {out_dim, 1, 1}) ||
        !expect_finite_tensor(w, name + ".weight_v", value_shape)) {
        return false;
    }

    std::size_t elements = 1;
    for (const int64_t dim : value_shape) {
        const std::size_t size = static_cast<std::size_t>(dim);
        if (elements > std::numeric_limits<std::size_t>::max() / size) return false;
        elements *= size;
    }
    const std::size_t rows = static_cast<std::size_t>(out_dim);
    if (elements % rows != 0) return false;
    const std::size_t inner = elements / rows;
    const float* g = w.get(name + ".weight_g");
    const float* v = w.get(name + ".weight_v");
    if (!g || !v) return false;

    for (std::size_t row = 0; row < rows; ++row) {
        double norm_squared = 0.0;
        for (std::size_t i = 0; i < inner; ++i) {
            const double value = v[row * inner + i];
            norm_squared += value * value;
        }
        if (!(norm_squared > 0.0) || !std::isfinite(norm_squared)) {
            fprintf(stderr, "VITS: weight norm 'model.%s' has a degenerate row\n",
                    name.c_str());
            return false;
        }
        const double norm = std::sqrt(norm_squared);
        for (std::size_t i = 0; i < inner; ++i) {
            const double reconstructed =
                static_cast<double>(g[row]) * v[row * inner + i] / norm;
            if (!std::isfinite(reconstructed) ||
                std::abs(reconstructed) > std::numeric_limits<float>::max()) {
                fprintf(stderr,
                        "VITS: weight norm 'model.%s' reconstruction overflows\n",
                        name.c_str());
                return false;
            }
        }
    }
    return true;
}

bool validate_checkpoint(const Weights& w, int feature_dim, int& num_speakers) {
    bool ok = true;
    ok &= expect_finite_tensor(w, "enc_p.emb_phone.weight", {kHidden, feature_dim});
    ok &= expect_finite_tensor(w, "enc_p.emb_phone.bias", {kHidden});
    ok &= expect_finite_tensor(w, "enc_p.emb_pitch.weight", {256, kHidden});
    for (int layer = 0; layer < kNLayers; ++layer) {
        const std::string a = "enc_p.encoder.attn_layers." + std::to_string(layer) + ".";
        ok &= expect_finite_tensor(w, a + "emb_rel_k", {1, 2 * kWindow + 1, kHidden / kNHeads});
        ok &= expect_finite_tensor(w, a + "emb_rel_v", {1, 2 * kWindow + 1, kHidden / kNHeads});
        for (const char* proj : {"conv_q", "conv_k", "conv_v", "conv_o"}) {
            ok &= expect_finite_tensor(w, a + proj + ".weight", {kHidden, kHidden, 1});
            ok &= expect_finite_tensor(w, a + proj + ".bias", {kHidden});
        }
        const std::string f = "enc_p.encoder.ffn_layers." + std::to_string(layer) + ".";
        ok &= expect_finite_tensor(w, f + "conv_1.weight", {kFilter, kHidden, 3});
        ok &= expect_finite_tensor(w, f + "conv_1.bias", {kFilter});
        ok &= expect_finite_tensor(w, f + "conv_2.weight", {kHidden, kFilter, 3});
        ok &= expect_finite_tensor(w, f + "conv_2.bias", {kHidden});
        for (const char* norm : {"norm_layers_1", "norm_layers_2"}) {
            const std::string n = "enc_p.encoder." + std::string(norm) + "." +
                                  std::to_string(layer) + ".";
            ok &= expect_finite_tensor(w, n + "gamma", {kHidden});
            ok &= expect_finite_tensor(w, n + "beta", {kHidden});
        }
    }
    ok &= expect_finite_tensor(w, "enc_p.proj.weight", {2 * kInter, kHidden, 1});
    ok &= expect_finite_tensor(w, "enc_p.proj.bias", {2 * kInter});

    for (int coupling = 0; coupling < 4; ++coupling) {
        const std::string p = "flow.flows." + std::to_string(coupling * 2) + ".";
        ok &= expect_finite_tensor(w, p + "pre.weight", {kHidden, kInter / 2, 1});
        ok &= expect_finite_tensor(w, p + "pre.bias", {kHidden});
        ok &= expect_finite_tensor(w, p + "post.weight", {kInter / 2, kHidden, 1});
        ok &= expect_finite_tensor(w, p + "post.bias", {kInter / 2});
        const std::string e = p + "enc.";
        ok &= expect_weight_norm(w, e + "cond_layer", 2 * kHidden * 3,
                                 {2 * kHidden * 3, kGin, 1});
        ok &= expect_finite_tensor(w, e + "cond_layer.bias", {2 * kHidden * 3});
        for (int layer = 0; layer < 3; ++layer) {
            const std::string in = e + "in_layers." + std::to_string(layer) + ".";
            ok &= expect_weight_norm(w, in.substr(0, in.size() - 1), 2 * kHidden,
                                     {2 * kHidden, kHidden, 5});
            ok &= expect_finite_tensor(w, in + "bias", {2 * kHidden});
            const int rs_channels = layer < 2 ? 2 * kHidden : kHidden;
            const std::string rs = e + "res_skip_layers." + std::to_string(layer) + ".";
            ok &= expect_weight_norm(w, rs.substr(0, rs.size() - 1), rs_channels,
                                     {rs_channels, kHidden, 1});
            ok &= expect_finite_tensor(w, rs + "bias", {rs_channels});
        }
    }

    ok &= expect_finite_tensor(w, "dec.m_source.l_linear.weight", {1, 1});
    ok &= expect_finite_tensor(w, "dec.m_source.l_linear.bias", {1});
    ok &= expect_finite_tensor(w, "dec.conv_pre.weight", {512, kInter, 7});
    ok &= expect_finite_tensor(w, "dec.conv_pre.bias", {512});
    ok &= expect_finite_tensor(w, "dec.cond.weight", {512, kGin, 1});
    ok &= expect_finite_tensor(w, "dec.cond.bias", {512});
    const int rates[4] = {10, 10, 2, 2};
    const int kernels[4] = {16, 16, 4, 4};
    const int resblock_kernels[3] = {3, 7, 11};
    int channels = 512;
    for (int stage = 0; stage < 4; ++stage) {
        const int out_channels = channels / 2;
        const std::string up = "dec.ups." + std::to_string(stage) + ".";
        ok &= expect_weight_norm(w, up.substr(0, up.size() - 1), channels,
                                 {channels, out_channels, kernels[stage]});
        ok &= expect_finite_tensor(w, up + "bias", {out_channels});
        int source_kernel = 1;
        if (stage < 3) {
            int remaining_rate = 1;
            for (int next = stage + 1; next < 4; ++next) remaining_rate *= rates[next];
            source_kernel = remaining_rate * 2;
        }
        const std::string noise = "dec.noise_convs." + std::to_string(stage) + ".";
        ok &= expect_finite_tensor(w, noise + "weight", {out_channels, 1, source_kernel});
        ok &= expect_finite_tensor(w, noise + "bias", {out_channels});
        for (int block = 0; block < 3; ++block) {
            const int rb = stage * 3 + block;
            const int kernel = resblock_kernels[block];
            const std::string p = "dec.resblocks." + std::to_string(rb) + ".";
            for (int layer = 0; layer < 3; ++layer) {
                for (const char* group : {"convs1", "convs2"}) {
                    const std::string c = p + group + "." + std::to_string(layer) + ".";
                    ok &= expect_weight_norm(w, c.substr(0, c.size() - 1),
                                             out_channels,
                                             {out_channels, out_channels, kernel});
                    ok &= expect_finite_tensor(w, c + "bias", {out_channels});
                }
            }
        }
        channels = out_channels;
    }
    ok &= expect_finite_tensor(w, "dec.conv_post.weight", {1, channels, 7});

    const io::Tensor* speakers = w.meta("emb_g.weight");
    if (!speakers || speakers->shape.size() != 2 || speakers->shape[0] <= 0 ||
        speakers->shape[0] > std::numeric_limits<int>::max() ||
        speakers->shape[1] != kGin ||
        !expect_finite_tensor(w, "emb_g.weight", {speakers->shape[0], kGin})) {
        fprintf(stderr, "VITS: tensor 'model.emb_g.weight' has incompatible shape or dtype\n");
        ok = false;
    } else {
        num_speakers = static_cast<int>(speakers->shape[0]);
    }
    return ok;
}

bool supported_architecture(const SynthesizerConfig& config) {
    return (config.synthesis_mode == SynthesisMode::kSampled ||
            config.synthesis_mode == SynthesisMode::kMean) &&
           (config.version == ModelVersion::kV1 ||
            config.version == ModelVersion::kV2) &&
           config.spec_channels == 1025 && config.segment_size == 3200 &&
           config.inter_channels == kInter && config.hidden_channels == kHidden &&
           config.filter_channels == kFilter && config.n_heads == kNHeads &&
           config.n_layers == kNLayers && config.kernel_size == 3 &&
           config.p_dropout == 0 && !config.half_precision &&
           config.gin_channels == kGin && config.upsample_initial_channel == 512 &&
           config.resblock == "1" &&
           config.upsample_rates == std::vector<int>({10, 10, 2, 2}) &&
           config.upsample_kernel_sizes == std::vector<int>({16, 16, 4, 4}) &&
           config.resblock_kernel_sizes == std::vector<int>({3, 7, 11}) &&
           config.resblock_dilation_sizes ==
               std::vector<std::vector<int>>({{1, 3, 5}, {1, 3, 5}, {1, 3, 5}}) &&
           config.sample_rate == 40'000;
}

bool valid_speaker(int speaker_id, int num_speakers) {
    return speaker_id >= 0 && speaker_id < num_speakers;
}

bool valid_frame_count(int frames, int max_elements_per_frame,
                       bool uses_attention) {
    if (frames <= 0 || max_elements_per_frame <= 0) return false;
    // Current attention and CUDA operator boundaries use int element counts.
    // Reject before scanning caller memory or forming a signed product.
    const std::int64_t frame_count = frames;
    const std::int64_t max_int = std::numeric_limits<int>::max();
    if (frames > std::numeric_limits<int>::max() / max_elements_per_frame ||
        (uses_attention && frame_count > max_int / (2 * frame_count))) {
        return false;
    }
    return true;
}

bool valid_float_input(const float* values, int frames, int width,
                       int max_elements_per_frame, bool uses_attention) {
    if (!values || width <= 0 || max_elements_per_frame < width ||
        !valid_frame_count(frames, max_elements_per_frame, uses_attention)) {
        return false;
    }
    const std::size_t elements = static_cast<std::size_t>(frames) * width;
    for (std::size_t i = 0; i < elements; ++i) {
        if (!std::isfinite(values[i])) return false;
    }
    return true;
}

bool valid_features(const float* features, int frames, int feature_dim) {
    // The widest generator stage has 12,800 channel-time elements per input
    // frame; attention additionally forms T*(2T) int-indexed intermediates.
    return valid_float_input(features, frames, feature_dim, 12'800, true);
}

bool valid_pitch(const float* pitch, const float* pitchf, int frames) {
    if (!pitch || !pitchf || frames <= 0) return false;
    for (int frame = 0; frame < frames; ++frame) {
        if (!std::isfinite(pitch[frame]) || pitch[frame] < 0.0f || pitch[frame] > 255.0f ||
            !std::isfinite(pitchf[frame]) || pitchf[frame] < 0.0f) {
            return false;
        }
    }
    return true;
}

bool valid_coarse_pitch(const int* pitch, int frames) {
    if (!pitch || frames <= 0) return false;
    for (int frame = 0; frame < frames; ++frame)
        if (pitch[frame] < 0 || pitch[frame] > 255) return false;
    return true;
}

bool all_finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

// Reconstruct weight_norm weight: weight[o,inner] = g[o] * v[o,inner] / ||v[o]||.
std::vector<float> reconstruct_wn(const Weights& w, const std::string& name,
                                  int out_dim, int inner) {
    if (out_dim <= 0 || inner <= 0 ||
        static_cast<std::size_t>(out_dim) >
            std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(inner)) {
        return {};
    }
    const float* g = w.get(name + ".weight_g");
    const float* v = w.get(name + ".weight_v");
    if (!g || !v) return {};
    std::vector<float> out(static_cast<std::size_t>(out_dim) * inner);
    for (int o = 0; o < out_dim; ++o) {
        if (!std::isfinite(g[o])) return {};
        double n2 = 0.0;
        for (int i = 0; i < inner; ++i) {
            const float value = v[static_cast<std::size_t>(o) * inner + i];
            if (!std::isfinite(value)) return {};
            n2 += static_cast<double>(value) * value;
        }
        if (!(n2 > 0.0) || !std::isfinite(n2)) return {};
        const double norm = std::sqrt(n2);
        for (int i = 0; i < inner; ++i) {
            const std::size_t index = static_cast<std::size_t>(o) * inner + i;
            const double reconstructed =
                static_cast<double>(g[o]) * static_cast<double>(v[index]) / norm;
            if (!std::isfinite(reconstructed) ||
                std::abs(reconstructed) > std::numeric_limits<float>::max()) {
                return {};
            }
            out[index] = static_cast<float>(reconstructed);
        }
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

// Multi-head relative attention. Projection layout stays [C,T]; the score,
// softmax and value operations execute on device with one per-head workspace.
std::vector<float> rel_attention(const Weights& w, const std::string& p,
                                 const std::vector<float>& x, int T) {
    const int C = kHidden, H = kNHeads, dk = C / H;
    auto q = conv1x1(x, C, T, w.get(p + "conv_q.weight"), w.get(p + "conv_q.bias"), C);
    auto key = conv1x1(x, C, T, w.get(p + "conv_k.weight"), w.get(p + "conv_k.bias"), C);
    auto value = conv1x1(x, C, T, w.get(p + "conv_v.weight"), w.get(p + "conv_v.bias"), C);
    auto output = ops::relative_attention(q.data(), key.data(), value.data(),
                                          w.get(p + "emb_rel_k"), w.get(p + "emb_rel_v"),
                                          T, H, dk, kWindow);
    return conv1x1(output, C, T, w.get(p + "conv_o.weight"), w.get(p + "conv_o.bias"), C);
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
                                int feature_dim, const int* pitch,
                                std::vector<float>* logs_out) {
    // emb_phone: Linear(feature_dim,192). phone [T,feature_dim] -> [T,192]
    auto xtc = k::linear(phone, T, feature_dim, w.get("enc_p.emb_phone.weight"),
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
    if (cond_w.size() != static_cast<std::size_t>(2 * H * nLayers * kGin)) {
        return {};
    }
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
        if (in_w.size() != static_cast<std::size_t>(2 * H * H * kernel)) return {};
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
        if (rs_w.size() != static_cast<std::size_t>(rs_ch * H)) return {};
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
        if (h.size() != static_cast<std::size_t>(kHidden * T)) return {};
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

// SineGen: f0 [T] -> har_source [T*upp]. harmonic_num=0; configured NSF sampling.
std::vector<float> sine_source(const float* f0, int T, int upp, int sr,
                               const Weights& w, const SynthesizerConfig& config,
                               std::uint64_t frame_offset = 0) {
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
    // Restore the upstream voiced/unvoiced excitation before source linear/tanh.
    for (int t = 0; t < T; ++t) {
        float uv = (f0[t] > 0) ? 1.0f : 0.0f;
        for (int i = 0; i < upp; ++i) {
            const auto offset = static_cast<std::size_t>(t) * upp + i;
            sine[offset] *= uv;
            if (config.synthesis_mode == SynthesisMode::kSampled) {
                const float amplitude = uv > 0.0f ? 0.003f : 0.1f / 3.0f;
                sine[offset] += amplitude * detail::sampling_normal(
                    config.synthesis_seed, detail::kNsfStream,
                    frame_offset * static_cast<std::uint64_t>(upp) + offset);
            }
        }
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
        if (w1.size() != static_cast<std::size_t>(C * C * kernel)) return {};
        int pad1 = dil[j] * (kernel - 1) / 2;
        xt = op::conv1d(xt.data(), C, T, w1.data(), w.get(c1 + ".bias"), C, kernel, 1, pad1, dil[j], 1);
        op::leaky_relu_inplace(xt.data(), C * T, kLRelu);
        std::string c2 = p + "convs2." + std::to_string(j);
        auto w2 = reconstruct_wn(w, c2, C, C * kernel);
        if (w2.size() != static_cast<std::size_t>(C * C * kernel)) return {};
        int pad2 = (kernel - 1) / 2;
        xt = op::conv1d(xt.data(), C, T, w2.data(), w.get(c2 + ".bias"), C, kernel, 1, pad2, 1, 1);
        for (int i = 0; i < C * T; ++i) x[i] += xt[i];
    }
    return x;
}

// GeneratorNSF: z [inter, T], f0 [T] -> audio [T*400]
std::vector<float> generator(const Weights& w, std::vector<float> x, int T,
                             const float* f0, const float* g, int sr,
                             const SynthesizerConfig& config,
                             std::uint64_t frame_offset = 0) {
    const int upsample_rates[4] = {10, 10, 2, 2};
    const int upsample_kernels[4] = {16, 16, 4, 4};
    const int resblock_k[3] = {3, 7, 11};
    const std::vector<std::vector<int>> resblock_d = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    int upp = 400;

    // Sine source [T*upp]
    auto har = sine_source(f0, T, upp, sr, w, config, frame_offset);  // [T*upp]

    // conv_pre: Conv1d(192, 512, 7, pad 3)
    auto xc = op::conv1d(x.data(), kInter, T, w.get("dec.conv_pre.weight"),
                         w.get("dec.conv_pre.bias"), 512, 7, 1, 3, 1, 1);
    std::vector<float>().swap(x);  // The latent is consumed by conv_pre.
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
        if (up_w.size() != static_cast<std::size_t>(C * Cout * kk)) return {};
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

        // These stage inputs are no longer read. Release their allocations
        // before the residual branches create several full-resolution buffers.
        std::vector<float>().swap(xc);
        std::vector<float>().swap(up);
        std::vector<float>().swap(xsrc);

        // MRF: sum of 3 resblocks / 3
        std::vector<float> acc(Cout * use_len, 0.0f);
        for (int jr = 0; jr < 3; ++jr) {
            std::string rp = "dec.resblocks." + std::to_string(i * 3 + jr) + ".";
            auto rb = resblock1(w, rp, xn, Cout, use_len, resblock_k[jr], resblock_d[jr]);
            if (rb.size() != static_cast<std::size_t>(Cout * use_len)) return {};
            for (int idx = 0; idx < Cout * use_len; ++idx) acc[idx] += rb[idx];
        }
        for (auto& v : acc) v /= 3.0f;

        xc = std::move(acc);
        C = Cout;
        L = use_len;
    }

    // Reference GeneratorNSF uses the default slope 0.01 here; the earlier
    // upsampling/residual activations explicitly use kLRelu (0.1).
    op::leaky_relu_inplace(xc.data(), C * L, 0.01f);
    auto out = op::conv1d(xc.data(), C, L, w.get("dec.conv_post.weight"), nullptr, 1, 7, 1, 3, 1, 1);
    op::tanh_inplace(out.data(), (int)out.size());
    return out;  // [L] audio
}

}  // namespace

struct Synthesizer::ModelState {
    std::shared_ptr<const io::SafetensorsLoader> loader;
    CheckpointIdentity identity;
    std::string resolved_path;
};

void Synthesizer::reset() noexcept {
    initialized_ = false;
    model_state_.reset();
    num_speakers_ = 0;
    has_f0_ = false;
}

bool Synthesizer::init(const SynthesizerConfig& config) try {
    // Reinitialization is deliberately invalidating: neither an ordinary
    // validation failure nor an exception may retain the previous snapshot.
    reset();
    if (!config.has_f0) {
        fprintf(stderr, "VITS: non-F0 checkpoints are not supported by this synthesizer\n");
        return false;
    }
    if (config.model_path.empty() || !supported_architecture(config)) {
        fprintf(stderr, "VITS: invalid or unsupported synthesizer configuration\n");
        return false;
    }

    const int feature_dim = config.version == ModelVersion::kV1 ? 256 : 768;
    std::string resolved_path;
    Weights weights;
    int checkpoint_speakers = 0;
    if (!resolve_checkpoint_path(config.model_path, resolved_path) ||
        !weights.load(resolved_path)) {
        fprintf(stderr, "VITS: checkpoint validation failed for '%s'\n",
                config.model_path.c_str());
        return false;
    }
    const CheckpointIdentity* snapshot_identity =
        weights.loader->source_identity();
    CheckpointIdentity current_identity;
    if (!snapshot_identity ||
        !validate_checkpoint(weights, feature_dim, checkpoint_speakers) ||
        !read_checkpoint_identity(resolved_path, current_identity) ||
        !(current_identity == *snapshot_identity)) {
        fprintf(stderr, "VITS: checkpoint validation failed for '%s'\n",
                config.model_path.c_str());
        return false;
    }
    auto staged_state = std::make_shared<ModelState>();
    staged_state->loader = weights.loader;
    staged_state->identity = *snapshot_identity;
    staged_state->resolved_path = std::move(resolved_path);

    config_ = config;
    config_.spk_embed_dim = checkpoint_speakers;
    sample_rate_ = config.sample_rate;
    version_ = config.version;
    has_f0_ = true;
    num_speakers_ = checkpoint_speakers;
    feature_dim_ = feature_dim;
    model_state_ = std::move(staged_state);
    initialized_ = true;
    return true;
} catch (const std::exception&) {
    reset();
    return false;
} catch (...) {
    reset();
    return false;
}

bool Synthesizer::checkpoint_current() const {
    return model_state_ &&
           checkpoint_is_current(model_state_->resolved_path,
                                 model_state_->identity);
}

std::vector<float> Synthesizer::debug_text_encoder(const float* features, int frames,
                                                   const int* pitch_coarse) try {
    if (!initialized_ || !checkpoint_current() ||
        !valid_features(features, frames, feature_dim_) ||
        !valid_coarse_pitch(pitch_coarse, frames)) return {};
    Weights w;
    w.loader = model_state_->loader;
    auto result = text_encoder(w, features, frames, feature_dim_, pitch_coarse, nullptr);
    return all_finite(result) ? result : std::vector<float>{};
} catch (const std::exception&) {
    return {};
} catch (...) {
    return {};
}

std::vector<float> Synthesizer::debug_flow(const float* features, int frames,
                                           const int* pitch_coarse, int speaker_id) try {
    if (!initialized_ || !checkpoint_current() ||
        !valid_features(features, frames, feature_dim_) ||
        !valid_coarse_pitch(pitch_coarse, frames) ||
        !valid_speaker(speaker_id, num_speakers_)) return {};
    Weights w;
    w.loader = model_state_->loader;
    std::vector<float> logs;
    const bool sampled = config_.synthesis_mode == SynthesisMode::kSampled;
    auto m_p = text_encoder(w, features, frames, feature_dim_, pitch_coarse, sampled ? &logs : nullptr);
    if (sampled && !detail::sample_prior(m_p, logs, frames, config_.synthesis_seed)) return {};
    const float* emb_g = w.get("emb_g.weight");  // [num_spk, gin]
    std::vector<float> g(kGin);
    for (int i = 0; i < kGin; ++i) g[i] = emb_g[speaker_id * kGin + i];
    auto result = flow_reverse(w, std::move(m_p), frames, g.data());
    return all_finite(result) ? result : std::vector<float>{};
} catch (const std::exception&) {
    return {};
} catch (...) {
    return {};
}

std::vector<float> Synthesizer::debug_har(const float* f0, int frames) try {
    if (!initialized_ || !checkpoint_current() ||
        !valid_float_input(f0, frames, 1, 400, false)) return {};
    for (int frame = 0; frame < frames; ++frame) {
        if (f0[frame] < 0.0f) return {};
    }
    Weights w;
    w.loader = model_state_->loader;
    auto result = sine_source(f0, frames, 400, sample_rate_, w, config_);
    return all_finite(result) ? result : std::vector<float>{};
} catch (const std::exception&) {
    return {};
} catch (...) {
    return {};
}

std::vector<float> Synthesizer::debug_convpre(const float* z, int frames,
                                              int speaker_id) try {
    if (!initialized_ || !checkpoint_current() ||
        !valid_float_input(z, frames, kInter, 512, false) ||
        !valid_speaker(speaker_id, num_speakers_)) return {};
    Weights w;
    w.loader = model_state_->loader;
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
    return all_finite(xc) ? xc : std::vector<float>{};
} catch (const std::exception&) {
    return {};
} catch (...) {
    return {};
}

std::vector<float> Synthesizer::debug_gen_stage0(const float* z, const float* f0,
                                                 int frames, int speaker_id,
                                                 int which) try {
    if (!initialized_ || !checkpoint_current() ||
        !valid_float_input(z, frames, kInter, 2'560, false) ||
        !valid_float_input(f0, frames, 1, 400, false) ||
        which < 0 || which > 2 ||
        !valid_speaker(speaker_id, num_speakers_)) return {};
    for (int frame = 0; frame < frames; ++frame) {
        if (f0[frame] < 0.0f) return {};
    }
    Weights w;
    w.loader = model_state_->loader;
    const float* emb_g = w.get("emb_g.weight");
    std::vector<float> g(kGin);
    for (int i = 0; i < kGin; ++i) g[i] = emb_g[speaker_id * kGin + i];

    auto har = sine_source(f0, frames, 400, sample_rate_, w, config_);
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
    if (up_w.size() != static_cast<std::size_t>(C * Cout * kk)) return {};
    auto up = op::conv_transpose1d(xc.data(), C, L, up_w.data(),
                                   w.get("dec.ups.0.bias"), Cout, kk, u, pad);
    int Lout = (L - 1) * u - 2 * pad + kk;
    if (which == 0) {
        return all_finite(up) ? up : std::vector<float>{};  // [256, Lout]
    }

    int stride_f0 = 10 * 2 * 2;  // 40
    auto xsrc = op::conv1d(har.data(), 1, (int)har.size(),
                           w.get("dec.noise_convs.0.weight"), w.get("dec.noise_convs.0.bias"),
                           Cout, stride_f0 * 2, stride_f0, stride_f0 / 2, 1, 1);
    if (which == 1) return all_finite(xsrc) ? xsrc : std::vector<float>{};

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
        if (rb.size() != static_cast<std::size_t>(Cout * use_len)) return {};
        for (int i = 0; i < Cout * use_len; ++i) acc[i] += rb[i];
    }
    for (auto& v : acc) v /= 3.0f;
    return all_finite(acc) ? acc : std::vector<float>{};
} catch (const std::exception&) {
    return {};
} catch (...) {
    return {};
}

std::vector<float> Synthesizer::debug_decode(const float* z, const float* f0,
                                             int frames, int speaker_id) try {
    if (!initialized_ || !checkpoint_current() ||
        !valid_float_input(z, frames, kInter, 12'800, false) ||
        !valid_float_input(f0, frames, 1, 400, false) ||
        !valid_speaker(speaker_id, num_speakers_)) return {};
    for (int frame = 0; frame < frames; ++frame) {
        if (f0[frame] < 0.0f) return {};
    }
    Weights w;
    w.loader = model_state_->loader;
    const float* emb_g = w.get("emb_g.weight");
    std::vector<float> g(kGin);
    for (int i = 0; i < kGin; ++i) g[i] = emb_g[speaker_id * kGin + i];
    std::vector<float> x(z, z + kInter * frames);
    auto result = generator(w, std::move(x), frames, f0, g.data(), sample_rate_, config_);
    return all_finite(result) ? result : std::vector<float>{};
} catch (const std::exception&) {
    return {};
} catch (...) {
    return {};
}

std::vector<float> Synthesizer::debug_encp(const float* features, int frames,
                                           const int* pitch_coarse,
                                           bool want_logs) try {
    if (!initialized_ || !checkpoint_current() ||
        !valid_features(features, frames, feature_dim_) ||
        !valid_coarse_pitch(pitch_coarse, frames)) return {};
    Weights w;
    w.loader = model_state_->loader;
    std::vector<float> logs;
    auto m_p = text_encoder(w, features, frames, feature_dim_, pitch_coarse,
                            want_logs ? &logs : nullptr);
    if (want_logs) m_p.insert(m_p.end(), logs.begin(), logs.end());  // [m_p ; logs_p]
    return all_finite(m_p) ? m_p : std::vector<float>{};
} catch (const std::exception&) {
    return {};
} catch (...) {
    return {};
}

std::vector<float> Synthesizer::debug_flow_reverse(const float* x, int frames,
                                                   int speaker_id) try {
    if (!initialized_ || !checkpoint_current() ||
        !valid_float_input(x, frames, kInter, kFilter, false) ||
        !valid_speaker(speaker_id, num_speakers_)) return {};
    Weights w;
    w.loader = model_state_->loader;
    const float* emb_g = w.get("emb_g.weight");
    std::vector<float> g(kGin);
    for (int i = 0; i < kGin; ++i) g[i] = emb_g[speaker_id * kGin + i];
    std::vector<float> in(x, x + kInter * frames);
    auto result = flow_reverse(w, std::move(in), frames, g.data());
    return all_finite(result) ? result : std::vector<float>{};
} catch (const std::exception&) {
    return {};
} catch (...) {
    return {};
}

AudioBuffer Synthesizer::infer(const float* features, int frames,
                               const float* pitch, const float* pitchf,
                               int speaker_id) {
    return infer(features, frames, pitch, pitchf, speaker_id, 0);
}

AudioBuffer Synthesizer::infer(const float* features, int frames,
                               const float* pitch, const float* pitchf,
                               int speaker_id, std::uint64_t sampling_frame_offset) try {
    AudioBuffer result;
    result.sample_rate = sample_rate_;

    if (!initialized_ || !checkpoint_current() ||
        !detail::valid_sampling_range(sampling_frame_offset, frames) ||
        !valid_features(features, frames, feature_dim_) ||
        !valid_speaker(speaker_id, num_speakers_) ||
        !valid_pitch(pitch, pitchf, frames)) {
        return result;
    }

    Weights w;
    w.loader = model_state_->loader;

    // pitch here is coarse (int stored as float); pitchf is f0 in Hz.
    std::vector<int> pitch_coarse(frames);
    for (int t = 0; t < frames; ++t) pitch_coarse[t] = (int)std::lround(pitch[t]);

    // Speaker embedding
    const float* emb_g = w.get("emb_g.weight");
    std::vector<float> g(kGin);
    for (int i = 0; i < kGin; ++i) g[i] = emb_g[speaker_id * kGin + i];

    // enc_p -> sampled z_p; explicit mean mode preserves the legacy reference.
    std::vector<float> logs;
    const bool sampled = config_.synthesis_mode == SynthesisMode::kSampled;
    auto m_p = text_encoder(w, features, frames, feature_dim_, pitch_coarse.data(), sampled ? &logs : nullptr);
    if (sampled && !detail::sample_prior(m_p, logs, frames, config_.synthesis_seed,
                                        sampling_frame_offset)) return result;
    if (m_p.size() != static_cast<std::size_t>(kInter * frames)) return result;
    // flow reverse -> z
    auto z = flow_reverse(w, m_p, frames, g.data());
    if (z.size() != static_cast<std::size_t>(kInter * frames)) return result;
    // generator -> audio
    auto audio = generator(w, std::move(z), frames, pitchf, g.data(), sample_rate_,
                           config_, sampling_frame_offset);
    if (audio.empty() || !all_finite(audio)) return result;

    result.data = std::move(audio);
    return result;
} catch (const std::exception&) {
    AudioBuffer result;
    result.sample_rate = sample_rate_;
    return result;
} catch (...) {
    AudioBuffer result;
    result.sample_rate = sample_rate_;
    return result;
}

AudioBuffer Synthesizer::infer_stream(const float* features, int frames,
                                      const float* pitch, const float* pitchf,
                                      int speaker_id, int skip_head,
                                      int return_length) try {
    AudioBuffer result;
    result.sample_rate = sample_rate_;
    if (!initialized_ || skip_head < 0 || return_length <= 0 || skip_head > frames ||
        return_length > frames - skip_head) {
        return result;
    }

    auto full = infer(features, frames, pitch, pitchf, speaker_id);
    const std::size_t samples_per_frame = static_cast<std::size_t>(sample_rate_ / 100);
    const std::size_t begin = static_cast<std::size_t>(skip_head) * samples_per_frame;
    const std::size_t count = static_cast<std::size_t>(return_length) * samples_per_frame;
    if (samples_per_frame == 0 || begin > full.data.size() || count > full.data.size() - begin) {
        return result;
    }
    result.data.assign(full.data.begin() + static_cast<std::ptrdiff_t>(begin),
                       full.data.begin() + static_cast<std::ptrdiff_t>(begin + count));
    return result;
} catch (const std::exception&) {
    AudioBuffer result;
    result.sample_rate = sample_rate_;
    return result;
} catch (...) {
    AudioBuffer result;
    result.sample_rate = sample_rate_;
    return result;
}

}  // namespace voxmutatio::synthesizer
