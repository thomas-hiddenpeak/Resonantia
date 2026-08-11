// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/catch_approx.hpp>

#include "voxmutatio/index/cuda_flat_index.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <utility>

namespace {

struct TempIndex {
    TempIndex() {
        static unsigned counter = 0;
        path = "/tmp/resonantia_index_" +
               std::to_string(static_cast<long long>(getpid())) + "_" +
               std::to_string(counter++) + ".bin";
    }
    ~TempIndex() { std::remove(path.c_str()); }
    TempIndex(const TempIndex&) = delete;
    TempIndex& operator=(const TempIndex&) = delete;
    std::string path;
};

void append_u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8)
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_float(std::vector<std::uint8_t>& bytes, float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<std::uint8_t>(bits >> shift));
}

std::vector<std::uint8_t> index_bytes(std::uint64_t ntotal,
                                      std::uint64_t dim,
                                      std::initializer_list<float> values) {
    std::vector<std::uint8_t> bytes;
    append_u64(bytes, ntotal);
    append_u64(bytes, dim);
    for (float value : values) append_float(bytes, value);
    return bytes;
}

bool write_binary(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(file);
}

}  // namespace

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

TEST_CASE("Feature blend rejects invalid bounds", "[index][blend][boundary]") {
    const float original[] = {1.0f, 2.0f};
    const float retrieved[] = {3.0f, 4.0f};
    float output[] = {9.0f, 9.0f};

    voxmutatio::index::blend_features(output, original, retrieved, -1, 2, 0.5);
    REQUIRE(output[0] == 9.0f);
    voxmutatio::index::blend_features(output, original, retrieved, 1, -2, 0.5);
    REQUIRE(output[0] == 9.0f);
    voxmutatio::index::blend_features(output, original, retrieved, 1, 2, -0.1);
    REQUIRE(output[0] == 9.0f);
    voxmutatio::index::blend_features(output, original, retrieved, 1, 2,
                                      std::numeric_limits<double>::quiet_NaN());
    REQUIRE(output[0] == 9.0f);
    voxmutatio::index::blend_features(nullptr, original, retrieved, 1, 2, 0.5);
}

TEST_CASE("CudaFlatIndex invalid file", "[index][cuda-flat]") {
    voxmutatio::index::CudaFlatIndex index;
    
    bool loaded = index.load("/tmp/nonexistent.index");
    REQUIRE(!loaded);
    REQUIRE(index.valid() == false);
}

TEST_CASE("CudaFlatIndex rejects malformed headers and ranges",
          "[index][cuda-flat][boundary]") {
    using voxmutatio::index::CudaFlatIndex;
    static_assert(!std::is_copy_constructible_v<CudaFlatIndex>);
    static_assert(!std::is_copy_assignable_v<CudaFlatIndex>);
    static_assert(std::is_nothrow_move_constructible_v<CudaFlatIndex>);
    static_assert(std::is_nothrow_move_assignable_v<CudaFlatIndex>);

    const auto rejected = [](std::vector<std::uint8_t> bytes) {
        TempIndex file;
        REQUIRE(write_binary(file.path, bytes));
        CudaFlatIndex index;
        REQUIRE_FALSE(index.load(file.path));
        REQUIRE_FALSE(index.valid());
        REQUIRE(index.total_vectors() == 0);
        REQUIRE(index.dim() == 0);
    };

    SECTION("negative vector count") {
        rejected(index_bytes(std::numeric_limits<std::uint64_t>::max(), 2, {}));
    }
    SECTION("negative dimension") {
        rejected(index_bytes(1, std::numeric_limits<std::uint64_t>::max(), {}));
    }
    SECTION("zero fields") {
        rejected(index_bytes(0, 2, {}));
        rejected(index_bytes(1, 0, {}));
    }
    SECTION("size multiplication overflow") {
        rejected(index_bytes(static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()),
                             static_cast<std::uint64_t>(std::numeric_limits<int>::max()), {}));
    }
    SECTION("truncated payload") {
        rejected(index_bytes(2, 2, {1.0f, 2.0f, 3.0f}));
    }
    SECTION("trailing payload") {
        rejected(index_bytes(1, 1, {1.0f, 2.0f}));
    }
    SECTION("non-finite database value") {
        rejected(index_bytes(1, 1, {std::numeric_limits<float>::quiet_NaN()}));
    }
}

TEST_CASE("CudaFlatIndex validates query bounds and supports move",
          "[index][cuda-flat][boundary]") {
    using voxmutatio::index::CudaFlatIndex;
    TempIndex file;
    REQUIRE(write_binary(file.path, index_bytes(2, 2, {0.0f, 0.0f, 1.0f, 1.0f})));

    CudaFlatIndex index;
    REQUIRE(index.load(file.path));
    const float query[] = {0.0f, 0.0f};
    REQUIRE(index.search(query, 1, 0, 2).first.empty());
    REQUIRE(index.search(query, 1, -1, 2).first.empty());
    REQUIRE(index.search(query, 1, 3, 2).first.empty());
    REQUIRE(index.search(query, 0, 1, 2).first.empty());
    REQUIRE(index.search(query, -1, 1, 2).first.empty());
    REQUIRE(index.search(nullptr, 1, 1, 2).first.empty());
    REQUIRE(index.search(query, 1, 1, -2).first.empty());
    REQUIRE(index.retrieve_weighted(query, 1, -1, 2).empty());
    REQUIRE(index.reconstruct(-1, 2).empty());

    auto [distances, indices] = index.search(query, 1, 2, 2);
    REQUIRE(distances.size() == 2);
    REQUIRE(indices.size() == 2);
    REQUIRE(indices[0] == 0);

    CudaFlatIndex moved(std::move(index));
    REQUIRE_FALSE(index.valid());
    REQUIRE(index.total_vectors() == 0);
    REQUIRE(moved.valid());
    REQUIRE(moved.total_vectors() == 2);

    // Replacement with a malformed file clears the previous mapping/device copy.
    TempIndex invalid;
    REQUIRE(write_binary(invalid.path, index_bytes(2, 2, {1.0f})));
    REQUIRE_FALSE(moved.load(invalid.path));
    REQUIRE_FALSE(moved.valid());
    REQUIRE(moved.total_vectors() == 0);
    REQUIRE(moved.dim() == 0);
}
