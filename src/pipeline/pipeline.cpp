// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// End-to-end voice conversion pipeline orchestration.
// Audio -> HuBERT content -> RMVPE F0 -> VITS synthesis -> Audio.

#include "voxmutatio/pipeline/pipeline.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/io/safetensors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

namespace voxmutatio::pipeline {

namespace {

constexpr int feature_dim_for(ModelVersion version) {
    return version == ModelVersion::kV1 ? 256 : 768;
}

constexpr bool supported_version(ModelVersion version) {
    return version == ModelVersion::kV1 || version == ModelVersion::kV2;
}

constexpr bool supported_content_encoder(ContentEncoder encoder) {
    return encoder == ContentEncoder::kHubert || encoder == ContentEncoder::kWavlm;
}

constexpr bool supported_f0_method(F0Method method) {
    return method == F0Method::kRmvpe || method == F0Method::kFcpe;
}

bool expect_f32_tensor(const io::SafetensorsLoader& loader, const std::string& name,
                       std::initializer_list<int64_t> shape) {
    const io::Tensor* tensor = loader.get_tensor(name);
    if (!tensor || tensor->shape != std::vector<int64_t>(shape)) return false;
    std::size_t elements = 1;
    for (int64_t dim : shape) elements *= static_cast<std::size_t>(dim);
    return tensor->dtype == "F32" && tensor->data_nbytes == elements * sizeof(float);
}

bool validate_rmvpe_checkpoint(const std::string& path) {
    io::SafetensorsLoader loader;
    return !path.empty() && loader.load(path) &&
           expect_f32_tensor(loader, "cnn.weight", {3, 16, 3, 3}) &&
           expect_f32_tensor(loader, "cnn.bias", {3}) &&
           expect_f32_tensor(loader, "fc.0.gru.weight_ih_l0", {768, 384}) &&
           expect_f32_tensor(loader, "fc.0.gru.weight_hh_l0", {768, 256}) &&
           expect_f32_tensor(loader, "fc.1.weight", {360, 512}) &&
           expect_f32_tensor(loader, "fc.1.bias", {360});
}

bool validate_wavlm_checkpoint(const std::string& path) {
    io::SafetensorsLoader loader;
    return !path.empty() && loader.load(path) &&
           expect_f32_tensor(loader, "feature_projection.projection.weight", {768, 512}) &&
           expect_f32_tensor(loader, "encoder.layers.0.attention.rel_attn_embed.weight", {320, 12}) &&
           expect_f32_tensor(loader, "encoder.layers.11.feed_forward.output_dense.weight",
                             {768, 3072});
}

// Coarse pitch quantization (matches RVC f0_to_coarse), returns 1..255.
int f0_to_coarse(float f0) {
    const double f0_mel_min = 1127.0 * std::log(1.0 + 50.0 / 700.0);
    const double f0_mel_max = 1127.0 * std::log(1.0 + 1100.0 / 700.0);
    if (f0 <= 0.0f) return 1;
    double f0_mel = 1127.0 * std::log(1.0 + f0 / 700.0);
    f0_mel = (f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0;
    if (f0_mel <= 1.0) f0_mel = 1.0;
    if (f0_mel > 255.0) f0_mel = 255.0;
    int coarse = static_cast<int>(std::lround(f0_mel));
    return std::clamp(coarse, 1, 255);
}

// Nearest-neighbour 2x interpolation of features [T, D] -> [2T, D].
std::vector<float> interpolate_2x(const std::vector<float>& feats, int T, int D) {
    std::vector<float> out(2 * T * D);
    for (int t = 0; t < T; ++t) {
        std::memcpy(&out[(2 * t) * D], &feats[t * D], D * sizeof(float));
        std::memcpy(&out[(2 * t + 1) * D], &feats[t * D], D * sizeof(float));
    }
    return out;
}

// RVC-style F0 median filter (kernel = odd size; smooths octave jumps/breaks).
std::vector<float> median_filter(const std::vector<float>& f0, int kernel) {
    if (kernel < 3) return f0;
    int r = (kernel - 1) / 2, n = static_cast<int>(f0.size());
    std::vector<float> out(n), win;
    win.reserve(kernel);
    for (int i = 0; i < n; ++i) {
        win.clear();
        for (int j = -r; j <= r; ++j) win.push_back(f0[std::clamp(i + j, 0, n - 1)]);
        std::sort(win.begin(), win.end());
        out[i] = win[win.size() / 2];
    }
    return out;
}

// RVC input high-pass: 5th-order Butterworth @48Hz (fs=16k), zero-phase (filtfilt).
void highpass_filtfilt(std::vector<float>& x) {
    static const double b[6] = {9.6996064518e-01, -4.8498032259e+00, 9.6996064518e+00,
                                -9.6996064518e+00, 4.8498032259e+00, -9.6996064518e-01};
    static const double a[6] = {1.0, -4.9390018192e+00, 9.7578635267e+00,
                                -9.6395448494e+00, 4.7615067974e+00, -9.4082365321e-01};
    int n = static_cast<int>(x.size());
    if (n < 40) return;
    auto iir = [&](const std::vector<double>& in) {
        std::vector<double> out(in.size(), 0.0);
        for (int i = 0; i < static_cast<int>(in.size()); ++i) {
            double acc = b[0] * in[i];
            for (int j = 1; j < 6; ++j)
                if (i - j >= 0) acc += b[j] * in[i - j] - a[j] * out[i - j];
            out[i] = acc;
        }
        return out;
    };
    const int pad = 18;  // scipy filtfilt default padlen = 3*max(len(b),len(a))
    std::vector<double> ext(n + 2 * pad);
    for (int i = 0; i < pad; ++i) ext[i] = 2.0 * x[0] - x[pad - i];
    for (int i = 0; i < n; ++i) ext[pad + i] = x[i];
    for (int i = 0; i < pad; ++i) ext[pad + n + i] = 2.0 * x[n - 1] - x[n - 2 - i];
    auto fwd = iir(ext);
    std::reverse(fwd.begin(), fwd.end());
    auto bwd = iir(fwd);
    std::reverse(bwd.begin(), bwd.end());
    for (int i = 0; i < n; ++i) x[i] = static_cast<float>(bwd[pad + i]);
}

// Reflect-pad both ends by `pad` samples (np.pad mode='reflect'); pad clamped to n-1.
std::vector<float> reflect_pad(const std::vector<float>& x, int pad) {
    int n = static_cast<int>(x.size());
    if (n < 2 || pad <= 0) return x;
    pad = std::min(pad, n - 1);
    std::vector<float> out(n + 2 * pad);
    for (int i = 0; i < pad; ++i) out[i] = x[pad - i];
    std::memcpy(&out[pad], x.data(), n * sizeof(float));
    for (int i = 0; i < pad; ++i) out[pad + n + i] = x[n - 2 - i];
    return out;
}

// RVC: linear-interpolate F0 across unvoiced (f0==0) frames before pitch shift.
void interp_unvoiced_f0(std::vector<float>& f0) {
    int n = static_cast<int>(f0.size());
    int first = -1, last = -1;
    for (int i = 0; i < n; ++i) if (f0[i] > 0) { if (first < 0) first = i; last = i; }
    if (first < 0) return;  // all unvoiced: leave as-is
    for (int i = 0; i < first; ++i) f0[i] = f0[first];
    for (int i = last + 1; i < n; ++i) f0[i] = f0[last];
    int i = first;
    while (i <= last) {
        if (f0[i] > 0) { ++i; continue; }
        int a = i - 1, b = i;
        while (b <= last && f0[b] <= 0) ++b;
        float fa = f0[a], fb = f0[b];
        for (int k = a + 1; k < b; ++k)
            f0[k] = fa + (fb - fa) * static_cast<float>(k - a) / static_cast<float>(b - a);
        i = b;
    }
}

// RVC change_rms: match source's per-frame RMS envelope onto the output.
void change_rms_envelope(const std::vector<float>& src, int src_sr,
                         std::vector<float>& out, double rate) {
    int L = static_cast<int>(out.size());
    if (L == 0) return;
    auto rms_env = [](const std::vector<float>& y, int sr, int out_len) {
        int hop = std::max(1, sr / 2), frame = hop * 2;  // ~1s window, 0.5s hop
        int nf = std::max(1, (static_cast<int>(y.size()) + hop - 1) / hop);
        std::vector<float> e(nf);
        for (int f = 0; f < nf; ++f) {
            long s = static_cast<long>(f) * hop, en = std::min<long>(y.size(), s + frame);
            double acc = 0; long c = 0;
            for (long k = s; k < en; ++k) { acc += static_cast<double>(y[k]) * y[k]; ++c; }
            e[f] = static_cast<float>(std::sqrt(acc / std::max(1L, c)));
        }
        std::vector<float> r(out_len);
        for (int i = 0; i < out_len; ++i) {
            double pos = static_cast<double>(i) * (nf - 1) / std::max(1, out_len - 1);
            int i0 = static_cast<int>(pos); float fr = static_cast<float>(pos - i0);
            float v0 = e[std::min(i0, nf - 1)], v1 = e[std::min(i0 + 1, nf - 1)];
            r[i] = v0 + (v1 - v0) * fr;
        }
        return r;
    };
    auto r1 = rms_env(src, src_sr, L);
    auto r2 = rms_env(out, 40000, L);  // out_sr cancels in the envelope ratio
    for (int i = 0; i < L; ++i) {
        float denom = std::max(r2[i], 1e-6f);
        out[i] *= std::pow(r1[i], static_cast<float>(1.0 - rate)) *
                  std::pow(denom, static_cast<float>(rate - 1.0));
    }
}

}  // namespace

bool VoiceConversionPipeline::set_runtime_params(int f0_up_key, double formant_shift,
                                                  double index_rate, double rms_mix_rate,
                                                  double protect, int filter_radius) noexcept {
    const bool valid_filter = filter_radius == 0 ||
                              (filter_radius >= 3 && filter_radius <= 31 &&
                               filter_radius % 2 == 1);
    if (f0_up_key < -48 || f0_up_key > 48 ||
        !std::isfinite(formant_shift) || formant_shift != 0.0 ||
        !std::isfinite(index_rate) || index_rate < 0.0 || index_rate > 1.0 ||
        (initialized_ && index_rate > 0.0 && !index_loaded_) ||
        !std::isfinite(rms_mix_rate) || rms_mix_rate < 0.0 || rms_mix_rate > 1.0 ||
        !std::isfinite(protect) || protect < 0.0 || protect > 1.0 ||
        !valid_filter) {
        return false;
    }
    config_.f0_up_key = f0_up_key;
    config_.formant_shift = formant_shift;
    config_.index_rate = index_rate;
    config_.rms_mix_rate = rms_mix_rate;
    config_.protect = protect;
    config_.filter_radius = filter_radius;
    return true;
}

bool VoiceConversionPipeline::init(const VCConfig& config) {
    VoiceConversionPipeline staged;
    if (!staged.init_in_place(config)) {
        // Failed re-initialization invalidates and releases every old mmap,
        // checkpoint loader, host weight vector, and model configuration.
        *this = VoiceConversionPipeline{};
        return false;
    }
    *this = std::move(staged);
    return true;
}

bool VoiceConversionPipeline::init_in_place(const VCConfig& config) {
    initialized_ = false;
    index_loaded_ = false;
    wavlm_loaded_ = false;
    fcpe_loaded_ = false;
    config_ = config;
    if (!supported_version(config.version) ||
        !supported_content_encoder(config.content_encoder) ||
        !supported_f0_method(config.f0_method)) {
        return false;
    }
    feature_dim_ = feature_dim_for(config.version);

    const bool invalid_filter = config.filter_radius != 0 &&
                                (config.filter_radius < 3 || config.filter_radius > 31 ||
                                 config.filter_radius % 2 == 0);
    const bool invalid_runtime_range =
        config.f0_up_key < -48 || config.f0_up_key > 48 ||
        !std::isfinite(config.index_rate) || config.index_rate < 0.0 || config.index_rate > 1.0 ||
        !std::isfinite(config.rms_mix_rate) || config.rms_mix_rate < 0.0 || config.rms_mix_rate > 1.0 ||
        !std::isfinite(config.protect) || config.protect < 0.0 || config.protect > 1.0 ||
        !std::isfinite(config.edge_pad_sec) || config.edge_pad_sec < 0.0 ||
        config.edge_pad_sec > 2.0 || !std::isfinite(config.max_chunk_sec) ||
        config.max_chunk_sec < 1.0 || config.max_chunk_sec > 30.0 ||
        !std::isfinite(config.chunk_pad_sec) || config.chunk_pad_sec < 0.0 ||
        config.chunk_pad_sec > 5.0 || !std::isfinite(config.chunk_query_sec) ||
        config.chunk_query_sec < 0.0 || config.chunk_query_sec > 5.0 ||
        config.chunk_query_sec > config.max_chunk_sec / 2.0 ||
        config.model_sample_rate != 40'000;
    if (config.device != "cuda" || config.use_half_precision ||
        !std::isfinite(config.formant_shift) || config.formant_shift != 0.0 ||
        (config.index_rate > 0.0 && config.index_path.empty()) ||
        (config.target_sample_rate != 0 &&
         config.target_sample_rate != config.model_sample_rate) ||
        config.output_format != "wav" ||
        invalid_filter || invalid_runtime_range) {
        return false;
    }

    // The current synthesizer is the NSF/F0 architecture. Reject unsupported
    // non-F0 checkpoints at the boundary instead of accepting a configuration
    // that would later dereference absent pitch tensors.
    if (!config.has_f0) return false;

    if (config.content_encoder == ContentEncoder::kWavlm) {
        if (config.version != ModelVersion::kV2 ||
            !validate_wavlm_checkpoint(config.wavlm_model_path)) {
            return false;
        }
        wavlm_cfg_.model_path = config.wavlm_model_path;
        wavlm_cfg_.output_dim = 768;
        wavlm_cfg_.num_layers = 12;
        wavlm_loaded_ = wavlm_encoder_.init(wavlm_cfg_);
        if (!wavlm_loaded_) return false;
    } else {
        hubert_cfg_.model_path = config.hubert_model_path;
        hubert_cfg_.output_dim = feature_dim_;
        hubert_cfg_.use_final_proj = (config.version == ModelVersion::kV1);
        hubert_cfg_.num_layers = hubert_cfg_.use_final_proj ? 9 : 12;
        hubert_cfg_.half_precision = config.use_half_precision;
        if (!hubert_encoder_.init(hubert_cfg_)) return false;
    }

    if (config.f0_method == F0Method::kFcpe) {
        if (config.fcpe_model_path.empty()) return false;
        fcpe_cfg_.model_path = config.fcpe_model_path;
        fcpe_loaded_ = fcpe_extractor_.init(fcpe_cfg_);
        if (!fcpe_loaded_) return false;
    } else {
        if (!validate_rmvpe_checkpoint(config.rmvpe_model_path)) return false;
        f0_cfg_.model_path = config.rmvpe_model_path;
        if (!f0_extractor_.init(f0_cfg_)) return false;
    }

    synth_cfg_.model_path = config.synthesizer_model_path;
    synth_cfg_.version = config.version;
    synth_cfg_.has_f0 = config.has_f0;
    synth_cfg_.sample_rate = config.model_sample_rate;
    synth_cfg_.spk_embed_dim = config.num_speakers;
    if (!synth_.init(synth_cfg_)) return false;
    if (synth_.feature_dim() != feature_dim_) return false;
    config_.num_speakers = synth_.num_speakers();

    // Optional feature retrieval index
    if (!config.index_path.empty()) {
        index_loaded_ = feature_index_.load(config.index_path);
        if (!index_loaded_ || feature_index_.dim() != feature_dim_) return false;
    }

    initialized_ = true;
    return true;
}

// Dispatch F0 extraction to the model validated during init().
std::vector<float> VoiceConversionPipeline::extract_f0(const float* audio, int num_samples) {
    if (config_.f0_method == F0Method::kFcpe)
        return fcpe_extractor_.infer(audio, num_samples);
    return f0_extractor_.infer(audio, num_samples);
}

// Dispatch content extraction to the model validated during init().
std::vector<float> VoiceConversionPipeline::extract_content(const float* audio, int num_samples) {
    if (config_.content_encoder == ContentEncoder::kWavlm)
        return wavlm_encoder_.extract(audio, num_samples);
    return hubert_encoder_.extract(audio, num_samples);
}


// Mirrors the historical whole-clip path exactly (HuBERT -> index -> 2x-interp ->
// protect -> coarse-pitch -> VITS), so callers own F0 computation and stitching.
AudioBuffer VoiceConversionPipeline::synth_segment(
        const float* seg16k, int segn,
        const float* f0_slice, const std::uint8_t* uv_slice, int f0_len,
        int speaker_id, VCResult& timing) {
    using clock = std::chrono::high_resolution_clock;

    // Content features [T, feature_dim_]
    auto t0 = clock::now();
    auto feats = extract_content(seg16k, segn);
    timing.hubert_ms += std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    if (feats.empty()) return {};
    if (feats.size() % static_cast<std::size_t>(feature_dim_) != 0) return {};
    int T = static_cast<int>(feats.size()) / feature_dim_;

    // Keep pre-index features for consonant/breath protection.
    const bool do_protect = config_.protect < 0.5;
    std::vector<float> feats0;
    if (do_protect) feats0 = feats;

    // Feature retrieval blending (RVC index_rate)
    t0 = clock::now();
    if (index_loaded_ && config_.index_rate > 0.0 && feature_index_.dim() == feature_dim_) {
        auto retrieved = feature_index_.retrieve_weighted(feats.data(), T, 8, feature_dim_);
        // An explicitly loaded index is part of the requested conversion.  A
        // failed lookup must fail this segment instead of returning plausible
        // audio synthesized from the original features while reporting success.
        if (retrieved.size() != feats.size()) {
            timing.index_ms += std::chrono::duration<double, std::milli>(
                clock::now() - t0).count();
            timing.error_message = "index retrieval failed";
            return {};
        }
        index::blend_features(feats.data(), feats.data(), retrieved.data(),
                              T, feature_dim_, config_.index_rate);
    }
    timing.index_ms += std::chrono::duration<double, std::milli>(clock::now() - t0).count();

    // Interpolate features 2x (50Hz -> 100Hz)
    auto feats_up = interpolate_2x(feats, T, feature_dim_);
    std::vector<float> feats0_up;
    if (do_protect) feats0_up = interpolate_2x(feats0, T, feature_dim_);
    int Tf = 2 * T;

    // Align lengths to the provided F0 slice.
    int p_len = std::min(Tf, f0_len);
    if (p_len <= 0) return {};
    feats_up.resize(static_cast<size_t>(p_len) * feature_dim_);
    std::vector<float> f0(f0_slice, f0_slice + p_len);

    // Consonant protection: unvoiced frames keep original (pre-index) features.
    if (do_protect) {
        feats0_up.resize(static_cast<size_t>(p_len) * feature_dim_);
        const float pr = static_cast<float>(config_.protect);
        for (int t = 0; t < p_len; ++t) {
            float w = uv_slice[t] ? pr : 1.0f;  // RVC pitchff
            if (w >= 1.0f) continue;
            float* f = &feats_up[static_cast<size_t>(t) * feature_dim_];
            const float* g = &feats0_up[static_cast<size_t>(t) * feature_dim_];
            for (int d = 0; d < feature_dim_; ++d) f[d] = f[d] * w + g[d] * (1.0f - w);
        }
    }

    // Coarse pitch
    std::vector<float> pitch_coarse(p_len);
    for (int i = 0; i < p_len; ++i)
        pitch_coarse[i] = static_cast<float>(f0_to_coarse(f0[i]));

    // VITS synthesis
    t0 = clock::now();
    auto synth_audio = synth_.infer(feats_up.data(), p_len,
                                    pitch_coarse.data(), f0.data(), speaker_id);
    timing.synth_ms += std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    return synth_audio;
}

VCResult VoiceConversionPipeline::convert_long(std::vector<float>& audio16k, int speaker_id) {
    using clock = std::chrono::high_resolution_clock;
    VCResult result;
    auto t_start = clock::now();

    // High-pass the whole clip (same front-end as the short path).
    if (config_.apply_highpass) highpass_filtfilt(audio16k);
    std::vector<float> src_hp = audio16k;  // unpadded, for the final change_rms
    int n = static_cast<int>(audio16k.size());
    if (n < 2) { result.error_message = "audio too short"; return result; }

    // Reflect-pad the whole clip once so every chunk gets real context on both sides.
    const int hop = 160;
    int pad = std::max(hop, static_cast<int>(config_.chunk_pad_sec * 16000));
    pad = std::min(pad, n - 1);
    std::vector<float> apad = reflect_pad(audio16k, pad);
    int np = static_cast<int>(apad.size());

    // F0 once on the whole padded clip -> globally consistent pitch (RVC does this).
    auto t0 = clock::now();
    auto f0 = extract_f0(apad.data(), np);
    std::vector<std::uint8_t> uv(f0.size());
    for (size_t i = 0; i < f0.size(); ++i) uv[i] = (f0[i] <= 0.0f) ? 1 : 0;
    if (config_.interp_unvoiced) interp_unvoiced_f0(f0);
    if (config_.filter_radius >= 3) f0 = median_filter(f0, config_.filter_radius | 1);
    if (config_.f0_up_key != 0) {
        float factor = std::pow(2.0f, config_.f0_up_key / 12.0f);
        for (auto& v : f0) if (v > 0) v *= factor;
    }
    result.f0_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    int f0n = static_cast<int>(f0.size());

    // Per-frame absolute energy over the ORIGINAL (unpadded) clip for cut selection.
    int nf = n / hop;
    std::vector<float> energy(std::max(0, nf), 0.0f);
    for (int f = 0; f < nf; ++f) {
        double acc = 0.0;
        int base = f * hop;
        for (int k = 0; k < hop; ++k) acc += std::fabs(audio16k[base + k]);
        energy[f] = static_cast<float>(acc);
    }

    // Cut points near every max_chunk_sec, snapped to the quietest frame within query.
    int chunk = std::max(hop, static_cast<int>(config_.max_chunk_sec * 16000));
    int query = std::max(hop, static_cast<int>(config_.chunk_query_sec * 16000));
    std::vector<int> bounds{0};
    for (std::int64_t target = chunk; target < n; target += chunk) {
        const int lo = static_cast<int>(std::max<std::int64_t>(1, target - query));
        const int hi = static_cast<int>(
            std::min<std::int64_t>(n - 1, target + query));
        int lof = lo / hop, hif = std::min(nf - 1, hi / hop);
        int best = lof;
        float bestE = std::numeric_limits<float>::max();
        for (int f = lof; f <= hif; ++f)
            if (energy[f] < bestE) { bestE = energy[f]; best = f; }
        int pos = best * hop;
        if (pos > bounds.back() && pos < n) bounds.push_back(pos);
    }
    bounds.push_back(n);

    // Convert each chunk with its context pad, trim the pad, and stitch.
    // Cores are placed at their ABSOLUTE output positions (not by trimming both
    // ends) so per-chunk frame-count jitter cannot accumulate into time drift.
    const long msr = synth_cfg_.sample_rate;
    auto to_out = [msr](long s16k) { return s16k * msr / 16000; };
    const int left_tgt = static_cast<int>(to_out(pad));  // context to drop at chunk start
    std::vector<float> out;
    for (size_t i = 0; i + 1 < bounds.size(); ++i) {
        int s = bounds[i], e = bounds[i + 1];
        int a = s;                                 // padded-index start (left context)
        int b = std::min(np, e + 2 * pad + hop);   // +hop so HuBERT covers the core tail
        int segn = b - a;
        int fs = a / hop, fe = std::min(f0n, b / hop);
        int f0_len = fe - fs;
        if (segn <= 0 || f0_len <= 0) continue;
        auto piece = synth_segment(apad.data() + a, segn, f0.data() + fs, uv.data() + fs,
                                   f0_len, speaker_id, result);
        if (piece.data.empty()) {
            if (result.error_message.empty()) {
                result.error_message = "chunk synthesis failed";
            }
            return result;
        }
        if (piece.data.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            result.error_message = "chunk synthesis exceeded the supported sample range";
            return result;
        }
        // Absolute output span for this chunk's core [s, e).
        int core_len = static_cast<int>(to_out(e) - to_out(s));
        const auto& d = piece.data;
        int avail = static_cast<int>(d.size()) - left_tgt;
        int take = std::clamp(avail, 0, core_len);
        if (take > 0) out.insert(out.end(), d.begin() + left_tgt, d.begin() + left_tgt + take);
        out.insert(out.end(), static_cast<size_t>(core_len - take), 0.0f);  // keep absolute align
    }
    if (out.empty()) { result.error_message = "long-audio synthesis produced no output"; return result; }
    if (out.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        result.error_message = "long-audio synthesis exceeded the supported sample range";
        return result;
    }

    // Match the source's per-frame RMS envelope over the whole output (RVC change_rms).
    if (config_.rms_mix_rate < 1.0)
        change_rms_envelope(src_hp, 16000, out, config_.rms_mix_rate);

    if (!std::all_of(out.begin(), out.end(), [](float value) { return std::isfinite(value); })) {
        result.error_message = "long-audio synthesis produced non-finite samples";
        return result;
    }

    result.audio.data = std::move(out);
    result.audio.sample_rate = synth_cfg_.sample_rate;
    result.success = true;
    result.total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_start).count();
    return result;
}

VCResult VoiceConversionPipeline::convert_buffer(const AudioBuffer& input,
                                                 int speaker_id) {
    using clock = std::chrono::high_resolution_clock;
    VCResult result;
    auto t_start = clock::now();

    if (!initialized_) {
        result.error_message = "pipeline not initialized";
        return result;
    }
    // Long-path reflection can add at most 5 s on each side at 16 kHz, plus one
    // 160-sample tail hop. Also reserve index range for native-rate output.
    constexpr std::size_t kMaximumPaddingSamples = 2u * 5u * 16'000u + 160u;
    const std::uint64_t maximum_by_output =
        static_cast<std::uint64_t>(std::numeric_limits<int>::max()) * 16'000u /
        static_cast<std::uint64_t>(synth_cfg_.sample_rate);
    const std::size_t kMaximumUnpaddedSamples = static_cast<std::size_t>(std::min<std::uint64_t>(
        static_cast<std::uint64_t>(std::numeric_limits<int>::max()) - kMaximumPaddingSamples,
        maximum_by_output > kMaximumPaddingSamples
            ? maximum_by_output - kMaximumPaddingSamples
            : 0));
    if (input.sample_rate <= 0 || input.data.empty() ||
        input.data.size() > kMaximumUnpaddedSamples ||
        !std::all_of(input.data.begin(), input.data.end(),
                     [](float value) { return std::isfinite(value); })) {
        result.error_message = "invalid input audio buffer";
        return result;
    }
    if (speaker_id < 0 || speaker_id >= synth_.num_speakers()) {
        result.error_message = "speaker_id out of range";
        return result;
    }

    // 1. Resample input to 16kHz mono for feature extraction
    std::vector<float> audio16k;
    if (input.sample_rate != 16000) {
        audio16k = io::resample_linear(input.data.data(),
                                       static_cast<int>(input.data.size()),
                                       input.sample_rate, 16000);
    } else {
        audio16k = input.data;
    }
    if (audio16k.empty() || audio16k.size() > kMaximumUnpaddedSamples ||
        !std::all_of(audio16k.begin(), audio16k.end(),
                     [](float value) { return std::isfinite(value); })) {
        result.error_message = "input resampling failed";
        return result;
    }

    // Long clips: split at silence to bound each model/GPU invocation. The
    // offline path still retains whole-clip host buffers and the final output.
    if (config_.max_chunk_sec > 0.0 &&
        static_cast<double>(audio16k.size()) > config_.max_chunk_sec * 16000.0) {
        return convert_long(audio16k, speaker_id);
    }

    // 1b. RVC input high-pass (drop sub-48Hz rumble), zero-phase.
    if (config_.apply_highpass) highpass_filtfilt(audio16k);
    // 1c. Reflect-pad both ends to avoid edge artifacts (trimmed after synthesis).
    int pad16k = std::max(0, static_cast<int>(config_.edge_pad_sec * 16000));
    pad16k = std::min(pad16k, std::max(0, static_cast<int>(audio16k.size()) - 1));
    audio16k = reflect_pad(audio16k, pad16k);
    int n16k = static_cast<int>(audio16k.size());

    // 2. F0 (RMVPE or FCPE); pre-interp voicing mask drives consonant protection.
    auto t0 = clock::now();
    auto f0 = extract_f0(audio16k.data(), n16k);
    std::vector<uint8_t> uv(f0.size());
    for (size_t i = 0; i < f0.size(); ++i) uv[i] = (f0[i] <= 0.0f) ? 1 : 0;
    if (config_.interp_unvoiced) interp_unvoiced_f0(f0);
    if (config_.filter_radius >= 3) f0 = median_filter(f0, config_.filter_radius | 1);
    result.f0_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();

    // 3. Pitch shift
    if (config_.f0_up_key != 0) {
        float factor = std::pow(2.0f, config_.f0_up_key / 12.0f);
        for (auto& v : f0) if (v > 0) v *= factor;
    }

    // 4. HuBERT -> index -> 2x-interp -> protect -> coarse-pitch -> VITS synthesis.
    auto synth_audio = synth_segment(audio16k.data(), n16k, f0.data(), uv.data(),
                                     static_cast<int>(f0.size()), speaker_id, result);
    if (synth_audio.data.empty()) {
        if (result.error_message.empty()) {
            result.error_message = "segment synthesis failed";
        }
        return result;
    }
    if (synth_audio.data.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        result.error_message = "synthesis exceeded the supported sample range";
        return result;
    }

    // 9. Trim the reflect-pad region (map 16k pad -> model-sr pad).
    if (pad16k > 0 && !synth_audio.data.empty()) {
        int pad_tgt = static_cast<int>(static_cast<long>(pad16k) * synth_cfg_.sample_rate / 16000);
        auto& d = synth_audio.data;
        if (static_cast<int>(d.size()) > 2 * pad_tgt && pad_tgt > 0) {
            d.erase(d.end() - pad_tgt, d.end());
            d.erase(d.begin(), d.begin() + pad_tgt);
        }
    }

    // 10. Match the source's per-frame RMS envelope (RVC change_rms).
    if (config_.rms_mix_rate < 1.0 && !synth_audio.data.empty()) {
        std::vector<float> src_core;
        if (pad16k > 0 && n16k > 2 * pad16k)
            src_core.assign(audio16k.begin() + pad16k, audio16k.end() - pad16k);
        else
            src_core = audio16k;
        change_rms_envelope(src_core, 16000, synth_audio.data, config_.rms_mix_rate);
    }

    if (!std::all_of(synth_audio.data.begin(), synth_audio.data.end(),
                     [](float value) { return std::isfinite(value); })) {
        result.error_message = "synthesis produced non-finite samples";
        return result;
    }

    result.audio = std::move(synth_audio);
    result.audio.sample_rate = synth_cfg_.sample_rate;
    result.success = true;
    result.total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_start).count();
    return result;
}

VCResult VoiceConversionPipeline::convert_file(const std::string& input_path,
                                               const std::string& output_path,
                                               int speaker_id) {
    VCResult result;
    auto audio = io::read_audio(input_path, 16000);
    if (!audio.has_value()) {
        result.error_message = "failed to read input: " + input_path;
        return result;
    }
    result = convert_buffer(*audio, speaker_id);
    if (!result.success) return result;
    if (result.audio.data.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        result.success = false;
        result.error_message = "converted audio exceeds the WAV writer range";
        return result;
    }

    if (!io::write_audio(output_path, result.audio.data.data(),
                         static_cast<int>(result.audio.data.size()),
                         result.audio.sample_rate, config_.output_format)) {
        result.success = false;
        result.error_message = "failed to write output: " + output_path;
    }
    return result;
}

ModelVersion VoiceConversionPipeline::version() const { return config_.version; }
int VoiceConversionPipeline::num_speakers() const { return synth_.num_speakers(); }

}  // namespace voxmutatio::pipeline
