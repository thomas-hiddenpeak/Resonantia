// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/catch_approx.hpp>

#include "voxmutatio/core/types.h"
#include "voxmutatio/core/device.h"

TEST_CASE("Error code strings", "[core][types]") {
    REQUIRE(voxmutatio::error_code_string(voxmutatio::ErrorCode::kSuccess) == "Success");
    REQUIRE(voxmutatio::error_code_string(voxmutatio::ErrorCode::kInvalidInput) == "Invalid Input");
    REQUIRE(voxmutatio::error_code_string(voxmutatio::ErrorCode::kModelLoadFailed) == "Model Load Failed");
    REQUIRE(voxmutatio::error_code_string(voxmutatio::ErrorCode::kDeviceError) == "Device Error");
    REQUIRE(voxmutatio::error_code_string(voxmutatio::ErrorCode::kInferenceError) == "Inference Error");
    REQUIRE(voxmutatio::error_code_string(voxmutatio::ErrorCode::kIoError) == "I/O Error");
    REQUIRE(voxmutatio::error_code_string(voxmutatio::ErrorCode::kNotFound) == "Not Found");
}

TEST_CASE("AudioBuffer basic operations", "[core][types]") {
    voxmutatio::AudioBuffer buffer;
    buffer.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    buffer.sample_rate = 16000;
    
    REQUIRE(buffer.num_samples() == 5);
    REQUIRE(buffer.duration_s() == Catch::Approx(0.0003125));
    REQUIRE(buffer.ptr()[0] == 1.0f);
    REQUIRE(buffer.ptr()[4] == 5.0f);
}

TEST_CASE("AudioBuffer empty", "[core][types]") {
    voxmutatio::AudioBuffer buffer;
    
    REQUIRE(buffer.num_samples() == 0);
    REQUIRE(buffer.duration_s() == 0.0);
}

TEST_CASE("VCConfig defaults", "[core][types]") {
    voxmutatio::VCConfig config;
    
    REQUIRE(config.f0_up_key == 0);
    REQUIRE(config.formant_shift == Catch::Approx(0.0));
    REQUIRE(config.index_rate == Catch::Approx(0.0));
    REQUIRE(config.rms_mix_rate == Catch::Approx(0.5));
    REQUIRE(config.protect == Catch::Approx(0.5));
    REQUIRE(config.device == "cuda");
    REQUIRE(config.use_half_precision == false);
    REQUIRE(config.gpu_device == 0);
    REQUIRE(config.version == voxmutatio::ModelVersion::kV1);
    REQUIRE(config.has_f0 == true);
    REQUIRE(config.num_speakers == 1);
    REQUIRE(config.model_sample_rate == 40000);
}

TEST_CASE("Device CPU initialization", "[core][device]") {
    voxmutatio::Device device;
    auto err = device.init("cpu");
    
    REQUIRE(!err.has_value());
    REQUIRE(device.is_cuda() == false);
    REQUIRE(device.name() == "CPU");
    REQUIRE(device.compute_capability() == 0);
}

TEST_CASE("Device CUDA initialization", "[core][device]") {
    voxmutatio::Device device;
    auto err = device.init("cuda");
    
    // If CUDA is not available, error should be returned
    if (err.has_value()) {
        // No CUDA available, which is expected in test environment
        REQUIRE(device.is_cuda() == false);
    } else {
        // CUDA is available
        REQUIRE(device.is_cuda() == true);
        REQUIRE(!device.name().empty());
        REQUIRE(device.compute_capability() > 0);
    }
}

TEST_CASE("Device invalid device ID", "[core][device]") {
    voxmutatio::Device device;
    auto err = device.init("cuda", 999);
    
    // Should return error for invalid device ID
    REQUIRE(err.has_value());
}
