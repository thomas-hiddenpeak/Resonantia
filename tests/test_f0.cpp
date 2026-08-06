// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/catch_approx.hpp>

#include "voxmutatio/f0/rmvpe.h"
#include "voxmutatio/f0/fcpe.h"

#include <cmath>

TEST_CASE("Pitch shift calculation", "[f0][pitch-shift]") {
    std::vector<float> f0 = {100.0f, 200.0f, 0.0f, 300.0f};  // 0.0 = unvoiced
    
    // +12 semitones = octave up (2x frequency)
    auto shifted_up = voxmutatio::f0::RmvpeExtractor::pitch_shift(f0, 12);
    
    REQUIRE(shifted_up[0] == Catch::Approx(200.0f).epsilon(1e-4));
    REQUIRE(shifted_up[1] == Catch::Approx(400.0f).epsilon(1e-4));
    REQUIRE(shifted_up[2] == Catch::Approx(0.0f).epsilon(1e-4));  // Unvoiced stays 0
    REQUIRE(shifted_up[3] == Catch::Approx(600.0f).epsilon(1e-4));
    
    // -12 semitones = octave down (0.5x frequency)
    auto shifted_down = voxmutatio::f0::RmvpeExtractor::pitch_shift(f0, -12);
    
    REQUIRE(shifted_down[0] == Catch::Approx(50.0f).epsilon(1e-4));
    REQUIRE(shifted_down[1] == Catch::Approx(100.0f).epsilon(1e-4));
    REQUIRE(shifted_down[2] == Catch::Approx(0.0f).epsilon(1e-4));
    REQUIRE(shifted_down[3] == Catch::Approx(150.0f).epsilon(1e-4));
}

TEST_CASE("Pitch shift zero semitones", "[f0][pitch-shift]") {
    std::vector<float> f0 = {100.0f, 200.0f, 300.0f};
    
    auto shifted = voxmutatio::f0::RmvpeExtractor::pitch_shift(f0, 0);
    
    for (size_t i = 0; i < f0.size(); ++i) {
        REQUIRE(shifted[i] == Catch::Approx(f0[i]).epsilon(1e-6));
    }
}

TEST_CASE("Pitch shift +7 semitones (perfect fifth)", "[f0][pitch-shift]") {
    std::vector<float> f0 = {100.0f};
    
    auto shifted = voxmutatio::f0::RmvpeExtractor::pitch_shift(f0, 7);
    
    // 100 * 2^(7/12) ≈ 149.83 Hz
    REQUIRE(shifted[0] == Catch::Approx(149.83f).epsilon(1e-2));
}

TEST_CASE("FCPE pitch shift matches RMVPE", "[f0][pitch-shift]") {
    std::vector<float> f0 = {100.0f, 150.0f, 200.0f, 0.0f, 250.0f};
    
    auto rmvpe_shifted = voxmutatio::f0::RmvpeExtractor::pitch_shift(f0, 5);
    auto fcpe_shifted = voxmutatio::f0::FcpeExtractor::pitch_shift(f0, 5);
    
    REQUIRE(rmvpe_shifted.size() == fcpe_shifted.size());
    
    for (size_t i = 0; i < f0.size(); ++i) {
        REQUIRE(rmvpe_shifted[i] == Catch::Approx(fcpe_shifted[i]).epsilon(1e-6));
    }
}
