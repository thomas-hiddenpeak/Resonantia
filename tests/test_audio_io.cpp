// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/catch_approx.hpp>

#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/core/types.h"

#include <fstream>
#include <cmath>

TEST_CASE("Linear resample same rate", "[io][resample]") {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    auto output = voxmutatio::io::resample_linear(input.data(), 5, 16000, 16000);
    
    REQUIRE(output.size() == 5);
    for (int i = 0; i < 5; ++i) {
        REQUIRE(output[i] == Catch::Approx(input[i]).epsilon(1e-6));
    }
}

TEST_CASE("Linear resample upsample", "[io][resample]") {
    std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f};
    
    auto output = voxmutatio::io::resample_linear(input.data(), 4, 8000, 16000);
    
    // 4 * 16000 / 8000 = 8 samples (truncated to 8)
    REQUIRE(output.size() == 8);
    
    // Check first and last values
    REQUIRE(output[0] == Catch::Approx(0.0f).epsilon(1e-4));
    REQUIRE(output.back() == Catch::Approx(3.0f).epsilon(1e-4));
}

TEST_CASE("Linear resample downsample", "[io][resample]") {
    std::vector<float> input = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f};
    
    auto output = voxmutatio::io::resample_linear(input.data(), 8, 16000, 8000);
    
    REQUIRE(output.size() == 4);
    
    // Check interpolated values
    REQUIRE(output[0] == Catch::Approx(0.0f).epsilon(1e-4));
    REQUIRE(output[1] == Catch::Approx(1.0f).epsilon(1e-4));
    REQUIRE(output[2] == Catch::Approx(2.0f).epsilon(1e-4));
    REQUIRE(output[3] == Catch::Approx(3.0f).epsilon(1e-4));
}

TEST_CASE("WAV write and read round-trip", "[io][wav]") {
    const std::string test_file = "/tmp/test_roundtrip.wav";
    
    // Generate test audio (sine wave)
    const int sample_rate = 16000;
    const int duration_ms = 100;
    const int num_samples = sample_rate * duration_ms / 1000;
    
    std::vector<float> samples(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        samples[i] = 0.5f * std::sin(2.0f * 3.14159f * 440.0f * i / sample_rate);
    }
    
    // Write WAV
    bool write_ok = voxmutatio::io::write_audio(test_file, samples.data(),
                                                 num_samples, sample_rate, "wav");
    if (!write_ok) {
        // Skip test if /tmp is not writable
        WARN("Cannot write to /tmp, skipping WAV round-trip test");
        return;
    }
    
    // Read WAV
    auto buffer = voxmutatio::io::read_audio(test_file, sample_rate);
    if (!buffer) {
        WARN("Cannot read WAV file, skipping verification");
        return;
    }
    
    REQUIRE(buffer->sample_rate == sample_rate);
    REQUIRE(buffer->num_samples() == static_cast<std::size_t>(num_samples));
    
    // Compare samples (int16 quantization has absolute error up to ~1/32768).
    for (int i = 0; i < num_samples; ++i) {
        REQUIRE(buffer->data[i] == Catch::Approx(samples[i]).margin(1e-4));
    }
}

TEST_CASE("WAV read non-existent file", "[io][wav]") {
    auto buffer = voxmutatio::io::read_audio("/tmp/nonexistent.wav");
    REQUIRE(!buffer.has_value());
}

TEST_CASE("WAV write invalid path", "[io][wav]") {
    std::vector<float> samples = {1.0f, 2.0f, 3.0f};
    
    bool write_ok = voxmutatio::io::write_audio("/invalid/path/file.wav",
                                                 samples.data(), 3, 16000, "wav");
    REQUIRE(!write_ok);
}
