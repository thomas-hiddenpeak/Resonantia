// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/catch_approx.hpp>

#include "voxmutatio/index/cuda_flat_index.h"

#include <cmath>
#include <cstring>

TEST_CASE("Feature blend rate 0.0", "[index][blend]") {
    const int frames = 10;
    const int dim = 4;
    
    std::vector<float> original(frames * dim);
    std::vector<float> retrieved(frames * dim);
    std::vector<float> output(frames * dim);
    
    // Fill with test data
    for (int i = 0; i < frames * dim; ++i) {
        original[i] = static_cast<float>(i);
        retrieved[i] = static_cast<float>(i) * 2.0f;
    }
    
    // Blend with rate 0.0 (original only)
    voxmutatio::index::blend_features(output.data(), original.data(),
                                      retrieved.data(), frames, dim, 0.0);
    
    for (int i = 0; i < frames * dim; ++i) {
        REQUIRE(output[i] == Catch::Approx(original[i]).epsilon(1e-6));
    }
}

TEST_CASE("Feature blend rate 1.0", "[index][blend]") {
    const int frames = 10;
    const int dim = 4;
    
    std::vector<float> original(frames * dim);
    std::vector<float> retrieved(frames * dim);
    std::vector<float> output(frames * dim);
    
    // Fill with test data
    for (int i = 0; i < frames * dim; ++i) {
        original[i] = static_cast<float>(i);
        retrieved[i] = static_cast<float>(i) * 2.0f;
    }
    
    // Blend with rate 1.0 (retrieved only)
    voxmutatio::index::blend_features(output.data(), original.data(),
                                      retrieved.data(), frames, dim, 1.0);
    
    for (int i = 0; i < frames * dim; ++i) {
        REQUIRE(output[i] == Catch::Approx(retrieved[i]).epsilon(1e-6));
    }
}

TEST_CASE("Feature blend rate 0.5", "[index][blend]") {
    const int frames = 10;
    const int dim = 4;
    
    std::vector<float> original(frames * dim);
    std::vector<float> retrieved(frames * dim);
    std::vector<float> output(frames * dim);
    
    // Fill with test data
    for (int i = 0; i < frames * dim; ++i) {
        original[i] = static_cast<float>(i);
        retrieved[i] = static_cast<float>(i) * 2.0f;
    }
    
    // Blend with rate 0.5 (50/50 mix)
    voxmutatio::index::blend_features(output.data(), original.data(),
                                      retrieved.data(), frames, dim, 0.5);
    
    for (int i = 0; i < frames * dim; ++i) {
        float expected = original[i] * 0.5f + retrieved[i] * 0.5f;
        REQUIRE(output[i] == Catch::Approx(expected).epsilon(1e-6));
    }
}

TEST_CASE("Feature blend with different rates", "[index][blend]") {
    const int frames = 5;
    const int dim = 2;
    
    std::vector<float> original(frames * dim, 1.0f);
    std::vector<float> retrieved(frames * dim, 3.0f);
    std::vector<float> output(frames * dim);
    
    // Test rate 0.3
    voxmutatio::index::blend_features(output.data(), original.data(),
                                      retrieved.data(), frames, dim, 0.3);
    
    for (int i = 0; i < frames * dim; ++i) {
        float expected = 1.0f * 0.7f + 3.0f * 0.3f;  // 1.6f
        REQUIRE(output[i] == Catch::Approx(expected).epsilon(1e-6));
    }
}

TEST_CASE("CudaFlatIndex invalid file", "[index][cuda-flat]") {
    voxmutatio::index::CudaFlatIndex index;
    
    bool loaded = index.load("/tmp/nonexistent.index");
    REQUIRE(!loaded);
    REQUIRE(index.valid() == false);
}
