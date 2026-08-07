// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// End-to-end voice conversion pipeline orchestration.
// Audio -> HuBERT content -> RMVPE F0 -> VITS synthesis -> Audio.

#include "voxmutatio/pipeline/pipeline.h"
#include "voxmutatio/io/audio_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace voxmutatio::pipeline {

namespace {

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

double compute_rms(const float* x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)x[i] * x[i];
    return std::sqrt(s / std::max(1, n));
}

}  // namespace

bool VoiceConversionPipeline::init(const VCConfig& config) {
    config_ = config;

    hubert_cfg_.model_path = config.hubert_model_path;
    hubert_cfg_.output_dim = (config.version == ModelVersion::kV2) ? 768 : 768;
    hubert_cfg_.use_final_proj = (config.version == ModelVersion::kV1);
    hubert_cfg_.half_precision = config.use_half_precision;
    hubert_encoder_.init(hubert_cfg_);

    f0_cfg_.model_path = config.rmvpe_model_path;
    f0_extractor_.init(f0_cfg_);

    synth_cfg_.model_path = config.synthesizer_model_path;
    synth_cfg_.version = config.version;
    synth_cfg_.has_f0 = config.has_f0;
    synth_cfg_.sample_rate = config.model_sample_rate;
    synth_cfg_.spk_embed_dim = config.num_speakers;
    synth_.init(synth_cfg_);

    // Optional feature retrieval index
    index_loaded_ = false;
    if (!config.index_path.empty()) {
        index_loaded_ = feature_index_.load(config.index_path);
    }

    initialized_ = true;
    return true;
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

    // 1. Resample input to 16kHz mono for feature extraction
    std::vector<float> audio16k;
    if (input.sample_rate != 16000) {
        audio16k = io::resample_linear(input.data.data(),
                                       static_cast<int>(input.data.size()),
                                       input.sample_rate, 16000);
    } else {
        audio16k = input.data;
    }
    int n16k = static_cast<int>(audio16k.size());

    // 2. HuBERT content features [T, 768]
    auto t0 = clock::now();
    auto feats = hubert_encoder_.extract(audio16k.data(), n16k);
    result.hubert_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    if (feats.empty()) {
        result.error_message = "HuBERT extraction failed";
        return result;
    }
    int hop = 320;  // HuBERT hop at 16kHz
    int T = static_cast<int>(feats.size()) / 768;

    // 2b. Feature retrieval blending (RVC index_rate)
    t0 = clock::now();
    if (index_loaded_ && config_.index_rate > 0.0 &&
        feature_index_.dim() == 768) {
        auto retrieved = feature_index_.retrieve_weighted(feats.data(), T, 8, 768);
        if (!retrieved.empty()) {
            index::blend_features(feats.data(), feats.data(), retrieved.data(),
                                  T, 768, config_.index_rate);
        }
    }
    result.index_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();

    // 3. Interpolate features 2x (50Hz -> 100Hz)
    auto feats_up = interpolate_2x(feats, T, 768);
    int Tf = 2 * T;

    // 4. RMVPE F0 [Tr]
    t0 = clock::now();
    auto f0 = f0_extractor_.infer(audio16k.data(), n16k);
    result.f0_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();

    // 5. Pitch shift
    if (config_.f0_up_key != 0) {
        float factor = std::pow(2.0f, config_.f0_up_key / 12.0f);
        for (auto& v : f0) if (v > 0) v *= factor;
    }

    // 6. Align lengths
    int p_len = std::min(Tf, static_cast<int>(f0.size()));
    feats_up.resize(p_len * 768);
    f0.resize(p_len);

    // 7. Coarse pitch
    std::vector<float> pitch_coarse(p_len);
    for (int i = 0; i < p_len; ++i)
        pitch_coarse[i] = static_cast<float>(f0_to_coarse(f0[i]));

    // 8. VITS synthesis
    t0 = clock::now();
    auto synth_audio = synth_.infer(feats_up.data(), p_len,
                                    pitch_coarse.data(), f0.data(), speaker_id);
    result.synth_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();

    // 9. RMS mix (optional): scale output toward source energy
    if (config_.rms_mix_rate < 1.0 && !synth_audio.data.empty()) {
        double src_rms = compute_rms(audio16k.data(), n16k);
        double out_rms = compute_rms(synth_audio.data.data(),
                                     static_cast<int>(synth_audio.data.size()));
        if (out_rms > 1e-8) {
            double target = std::pow(src_rms, 1.0 - config_.rms_mix_rate) *
                            std::pow(out_rms, config_.rms_mix_rate);
            float scale = static_cast<float>(target / out_rms);
            for (auto& v : synth_audio.data) v *= scale;
        }
    }

    result.audio = std::move(synth_audio);
    result.audio.sample_rate = synth_cfg_.sample_rate;
    result.success = true;
    result.total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_start).count();
    (void)hop;
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

    if (!io::write_audio(output_path, result.audio.data.data(),
                         static_cast<int>(result.audio.data.size()),
                         result.audio.sample_rate, config_.output_format)) {
        result.success = false;
        result.error_message = "failed to write output: " + output_path;
    }
    return result;
}

ModelVersion VoiceConversionPipeline::version() const { return config_.version; }
int VoiceConversionPipeline::num_speakers() const { return config_.num_speakers; }

}  // namespace voxmutatio::pipeline
