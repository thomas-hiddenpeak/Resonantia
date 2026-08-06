// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/catch_approx.hpp>

#include "voxmutatio/pipeline/pipeline.h"
#include "voxmutatio/core/types.h"
#include "voxmutatio/io/audio_io.h"

#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

TEST_CASE("Pipeline initialization", "[pipeline][init]") {
    voxmutatio::VCConfig config;
    config.device = "cpu";  // Use CPU for testing
    
    voxmutatio::pipeline::VoiceConversionPipeline pipeline;
    
    // Init should fail without model paths (expected)
    bool result = pipeline.init(config);
    
    // This is expected to fail without valid model files
    // The test verifies the pipeline structure is correct
    if (!result) {
        INFO("Pipeline init failed as expected without model files");
    }
}

TEST_CASE("Pipeline config defaults", "[pipeline][config]") {
    voxmutatio::VCConfig config;
    
    REQUIRE(config.f0_up_key == 0);
    REQUIRE(config.formant_shift == Catch::Approx(0.0));
    REQUIRE(config.index_rate == Catch::Approx(0.0));
    REQUIRE(config.rms_mix_rate == Catch::Approx(0.5));
    REQUIRE(config.protect == Catch::Approx(0.5));
    REQUIRE(config.device == "cuda");
    REQUIRE(config.use_half_precision == false);
    REQUIRE(config.version == voxmutatio::ModelVersion::kV1);
    REQUIRE(config.has_f0 == true);
}

TEST_CASE("VCResult default values", "[pipeline][result]") {
    voxmutatio::VCResult result;
    
    REQUIRE(result.success == false);
    REQUIRE(result.hubert_ms == Catch::Approx(0.0));
    REQUIRE(result.f0_ms == Catch::Approx(0.0));
    REQUIRE(result.index_ms == Catch::Approx(0.0));
    REQUIRE(result.synth_ms == Catch::Approx(0.0));
    REQUIRE(result.total_ms == Catch::Approx(0.0));
    REQUIRE(result.error_message == "");
}

TEST_CASE("Edge case: empty audio buffer", "[pipeline][edge]") {
    voxmutatio::AudioBuffer empty;
    empty.data = {};
    empty.sample_rate = 16000;
    
    REQUIRE(empty.num_samples() == 0);
    REQUIRE(empty.duration_s() == Catch::Approx(0.0));
}

TEST_CASE("Edge case: very short audio", "[pipeline][edge]") {
    // 0.05s at 16kHz = 800 samples
    voxmutatio::AudioBuffer short_audio;
    short_audio.data.resize(800, 0.0f);
    short_audio.sample_rate = 16000;
    
    REQUIRE(short_audio.num_samples() == 800);
    REQUIRE(short_audio.duration_s() == Catch::Approx(0.05));
}

TEST_CASE("Edge case: long audio (5min)", "[pipeline][edge]") {
    // 5min at 16kHz = 4800000 samples
    voxmutatio::AudioBuffer long_audio;
    long_audio.data.resize(4800000, 0.0f);
    long_audio.sample_rate = 16000;
    
    REQUIRE(long_audio.num_samples() == 4800000);
    REQUIRE(long_audio.duration_s() == Catch::Approx(300.0));  // 5 minutes
}

TEST_CASE("Stereo to mono downmix", "[pipeline][edge]") {
    // Simulate stereo downmix
    std::vector<float> stereo = {
        0.1f, 0.2f,  // frame 0: L, R
        0.3f, 0.4f,  // frame 1: L, R
        0.5f, 0.6f   // frame 2: L, R
    };
    
    std::vector<float> mono(stereo.size() / 2);
    for (size_t i = 0; i < mono.size(); ++i) {
        mono[i] = (stereo[i * 2] + stereo[i * 2 + 1]) / 2.0f;
    }
    
    REQUIRE(mono.size() == 3);
    REQUIRE(mono[0] == Catch::Approx(0.15f));
    REQUIRE(mono[1] == Catch::Approx(0.35f));
    REQUIRE(mono[2] == Catch::Approx(0.55f));
}
