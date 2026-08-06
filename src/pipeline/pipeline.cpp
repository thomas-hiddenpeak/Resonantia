// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/pipeline/pipeline.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>

namespace voxmutatio::pipeline {

namespace {

// Compute RMS energy of audio buffer
float compute_rms(const float* audio, int num_samples) {
    float sum_sq = 0.0f;
    for (int i = 0; i < num_samples; ++i) {
        sum_sq += audio[i] * audio[i];
    }
    return std::sqrt(sum_sq / num_samples);
}

// Apply RMS energy blending
void apply_rms_blend(float* output, const float* source, 
                     int num_samples, double rms_mix_rate) {
    if (rms_mix_rate <= 0.0 || rms_mix_rate >= 1.0) {
        // No blending needed
        if (rms_mix_rate >= 1.0) {
            std::memcpy(output, source, num_samples * sizeof(float));
        }
        return;
    }
    
    float source_rms = compute_rms(source, num_samples);
    
    // Target RMS would be computed from reference audio
    // For now, we just scale the output
    for (int i = 0; i < num_samples; ++i) {
        output[i] = output[i] * rms_mix_rate + source[i] * (1.0 - rms_mix_rate);
    }
}

// Apply unvoiced protection
void apply_uv_protection(std::vector<float>& f0, 
                         const std::vector<float>& source_f0,
                         double protect) {
    if (protect <= 0.0) {
        return;
    }
    
    for (size_t i = 0; i < f0.size(); ++i) {
        // If source frame is unvoiced (F0 = 0), protect it
        if (source_f0[i] == 0.0f && protect > 0.5) {
            f0[i] = 0.0f;
        }
    }
}

}  // namespace

bool VoiceConversionPipeline::init(const VCConfig& config) {
    config_ = config;
    
    // Initialize content encoder (HuBERT)
    hubert_cfg_.model_path = config_.hubert_model_path;
    hubert_cfg_.output_dim = (config_.version == ModelVersion::kV2) ? 768 : 256;
    hubert_cfg_.half_precision = config_.use_half_precision;
    
    if (!hubert_encoder_.init(hubert_cfg_)) {
        return false;
    }
    
    // Initialize F0 extractor (RMVPE)
    f0_cfg_.model_path = config_.rmvpe_model_path;
    
    if (!f0_extractor_.init(f0_cfg_)) {
        return false;
    }
    
    // Initialize feature index (optional)
    if (!config_.index_path.empty()) {
        if (!feature_index_.load(config_.index_path)) {
            // Index loading failed, continue without index
        }
    }
    
    // Initialize synthesizer
    synth_cfg_.model_path = config_.synthesizer_model_path;
    synth_cfg_.version = config_.version;
    synth_cfg_.has_f0 = config_.has_f0;
    synth_cfg_.half_precision = config_.use_half_precision;
    
    if (!synth_.init(synth_cfg_)) {
        return false;
    }
    
    // Auto-detect model metadata
    config_.version = synth_.version();
    config_.has_f0 = synth_.has_f0();
    config_.num_speakers = synth_.num_speakers();
    config_.model_sample_rate = synth_.sample_rate();
    
    initialized_ = true;
    return true;
}

VCResult VoiceConversionPipeline::convert_file(const std::string& input_path,
                                                const std::string& output_path,
                                                int speaker_id) {
    VCResult result;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Load audio
    auto audio = io::read_audio(input_path, 16000);
    if (!audio) {
        result.success = false;
        result.error_message = "Failed to load audio file: " + input_path;
        return result;
    }
    
    // Convert buffer
    VCResult buffer_result = convert_buffer(*audio, speaker_id);
    
    if (!buffer_result.success) {
        return buffer_result;
    }
    
    // Write output
    if (!io::write_audio(output_path, buffer_result.audio.data.data(),
                        static_cast<int>(buffer_result.audio.data.size()),
                        buffer_result.audio.sample_rate,
                        config_.output_format)) {
        result.success = false;
        result.error_message = "Failed to write audio file: " + output_path;
        return result;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    buffer_result.total_ms = std::chrono::duration<double, std::milli>(
        total_end - total_start).count();
    
    return buffer_result;
}

VCResult VoiceConversionPipeline::convert_buffer(const AudioBuffer& input,
                                                  int speaker_id) {
    VCResult result;
    
    if (!initialized_) {
        result.success = false;
        result.error_message = "Pipeline not initialized";
        return result;
    }
    
    if (input.sample_rate != 16000) {
        result.success = false;
        result.error_message = "Input audio must be 16kHz mono";
        return result;
    }
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Step 1: Extract content features (HuBERT)
    auto hubert_start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> source_features = hubert_encoder_.extract(
        input.data.data(), static_cast<int>(input.data.size()));
    
    auto hubert_end = std::chrono::high_resolution_clock::now();
    result.hubert_ms = std::chrono::duration<double, std::milli>(
        hubert_end - hubert_start).count();
    
    int frames = static_cast<int>(source_features.size()) / hubert_encoder_.output_dim();
    
    // Step 2: Extract F0 (RMVPE)
    auto f0_start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> source_f0;
    if (config_.has_f0) {
        source_f0 = f0_extractor_.infer(input.data.data(),
                                       static_cast<int>(input.data.size()));
        
        // Apply pitch shift
        if (config_.f0_up_key != 0) {
            source_f0 = f0::RmvpeExtractor::pitch_shift(source_f0, 
                                                         config_.f0_up_key);
        }
        
        // Apply unvoiced protection
        apply_uv_protection(source_f0, source_f0, config_.protect);
    }
    
    auto f0_end = std::chrono::high_resolution_clock::now();
    result.f0_ms = std::chrono::duration<double, std::milli>(
        f0_end - f0_start).count();
    
    // Step 3: Feature retrieval (optional)
    auto index_start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> target_features = source_features;
    
    if (feature_index_.valid() && config_.index_rate > 0.0) {
        // Search for nearest neighbors
        auto [distances, indices] = feature_index_.search(
            source_features.data(), frames, 
            static_cast<int>(config_.index_rate > 0.0 ? 1 : 0),
            hubert_encoder_.output_dim());
        
        // Reconstruct retrieved features
        std::vector<float> retrieved(frames * hubert_encoder_.output_dim());
        for (int i = 0; i < frames; ++i) {
            auto recon = feature_index_.reconstruct(indices[i], 
                                                    hubert_encoder_.output_dim());
            std::memcpy(retrieved.data() + i * hubert_encoder_.output_dim(),
                       recon.data(), hubert_encoder_.output_dim() * sizeof(float));
        }
        
        // Blend features
        index::blend_features(target_features.data(), source_features.data(),
                             retrieved.data(), frames,
                             hubert_encoder_.output_dim(),
                             config_.index_rate);
    }
    
    auto index_end = std::chrono::high_resolution_clock::now();
    result.index_ms = std::chrono::duration<double, std::milli>(
        index_end - index_start).count();
    
    // Step 4: Synthesize audio
    auto synth_start = std::chrono::high_resolution_clock::now();
    
    // Prepare pitch and pitchf arrays
    std::vector<float> pitch(frames, 0.0f);
    std::vector<float> pitchf(frames, 0.0f);
    
    if (config_.has_f0 && !source_f0.empty()) {
        // Convert F0 (Hz) to pitch (log scale) and pitchf (linear)
        for (int i = 0; i < frames && i < static_cast<int>(source_f0.size()); ++i) {
            if (source_f0[i] > 0.0f) {
                pitch[i] = std::log2(source_f0[i] / 10.0f);
                pitchf[i] = source_f0[i];
            }
        }
    }
    
    AudioBuffer synthesized = synth_.infer(
        target_features.data(), frames,
        pitch.data(), pitchf.data(),
        speaker_id);
    
    auto synth_end = std::chrono::high_resolution_clock::now();
    result.synth_ms = std::chrono::duration<double, std::milli>(
        synth_end - synth_start).count();
    
    // Step 5: Apply RMS energy blending
    if (config_.rms_mix_rate > 0.0 && config_.rms_mix_rate < 1.0) {
        apply_rms_blend(synthesized.data.data(), input.data.data(),
                       std::min(static_cast<int>(synthesized.data.size()),
                               static_cast<int>(input.data.size())),
                       config_.rms_mix_rate);
    }
    
    // Resample to target sample rate if needed
    if (config_.target_sample_rate > 0 && 
        config_.target_sample_rate != synthesized.sample_rate) {
        std::vector<float> resampled = io::resample_linear(
            synthesized.data.data(),
            static_cast<int>(synthesized.data.size()),
            synthesized.sample_rate,
            config_.target_sample_rate);
        
        synthesized.data = std::move(resampled);
        synthesized.sample_rate = config_.target_sample_rate;
    }
    
    result.audio = std::move(synthesized);
    result.success = true;
    
    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(
        total_end - total_start).count();
    
    return result;
}

ModelVersion VoiceConversionPipeline::version() const {
    return config_.version;
}

int VoiceConversionPipeline::num_speakers() const {
    return config_.num_speakers;
}

}  // namespace voxmutatio::pipeline
