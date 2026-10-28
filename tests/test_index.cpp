// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/catch_approx.hpp>

#include "voxmutatio/index/cuda_flat_index.h"

#include "subprocess_test.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <sys/stat.h>
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

std::vector<std::uint8_t> repeated_index_bytes(std::uint64_t ntotal,
                                               std::uint64_t dim,
                                               float value) {
    std::vector<std::uint8_t> bytes;
    append_u64(bytes, ntotal);
    append_u64(bytes, dim);
    for (std::uint64_t row = 0; row < ntotal; ++row) {
        for (std::uint64_t column = 0; column < dim; ++column) {
            append_float(bytes, value);
        }
    }
    return bytes;
}

bool write_binary(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(file);
}

}  // namespace

TEST_CASE("Flat Index multi-query search preserves scalar distances and ties",
          "[index][cuda-flat][exact]") {
    constexpr int rows = 257, dim = 33, queries = 19, k = 8;
    std::vector<float> data(rows * dim), input(queries * dim);
    std::vector<std::uint8_t> bytes;
    append_u64(bytes, rows);
    append_u64(bytes, dim);
    for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < dim; ++d) {
            // Duplicate rows deliberately exercise equal-distance top-k ordering.
            const float value = ((r % 37) * 13 + d * 7) % 101 / 19.0f;
            data[r * dim + d] = value;
            append_float(bytes, value);
        }
    }
    for (int i = 0; i < queries * dim; ++i)
        input[i] = (i * 17 % 83) / 23.0f;
    std::copy_n(data.begin(), dim, input.begin());  // exact zero-distance match
    TempIndex file;
    REQUIRE(write_binary(file.path, bytes));
    voxmutatio::index::CudaFlatIndex index;
    REQUIRE(index.load(file.path));

    std::vector<float> expected_distances(queries * k), scratch(rows);
    std::vector<int64_t> expected_indices(queries * k), order(rows);
    for (int q = 0; q < queries; ++q) {
        for (int r = 0; r < rows; ++r) {
            float distance = 0;
            for (int d = 0; d < dim; ++d) {
                const float difference = input[q * dim + d] - data[r * dim + d];
                distance += difference * difference;
            }
            scratch[r] = distance;
        }
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](int64_t a, int64_t b) { return scratch[a] < scratch[b]; });
        for (int j = 0; j < k; ++j) {
            expected_indices[q * k + j] = order[j];
            expected_distances[q * k + j] = scratch[order[j]];
        }
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
        const auto [distances, indices] = index.search(input.data(), queries, k, dim);
        REQUIRE(indices == expected_indices);
        REQUIRE(distances == expected_distances);
    }
    input.back() = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(index.search(input.data(), queries, k, dim).first.empty());
}

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

TEST_CASE("hidden CudaFlatIndex FIFO rejection child",
          "[.index-fifo-child]") {
    TempIndex fifo;
    REQUIRE(::mkfifo(fifo.path.c_str(), 0600) == 0);
    voxmutatio::index::CudaFlatIndex index;
    REQUIRE_FALSE(index.load(fifo.path));
    REQUIRE_FALSE(index.valid());
}

TEST_CASE("CudaFlatIndex promptly rejects a no-writer FIFO",
          "[index][cuda-flat][io-contract]") {
    const auto child = resonantia_test::run_self_test(
        "hidden CudaFlatIndex FIFO rejection child", 2'000);
    INFO("timed_out=" << child.timed_out << " status=" << child.status
                       << " launch_error=" << child.launch_error
                       << " wait_error=" << child.wait_error);
    REQUIRE(child.exited_with(0));
}

TEST_CASE("hidden CudaFlatIndex truncate rejection child",
          "[.index-truncate-child]") {
    TempIndex file;
    REQUIRE(write_binary(file.path, repeated_index_bytes(4096, 2, 1.0f)));

    voxmutatio::index::CudaFlatIndex index;
    REQUIRE(index.load(file.path));
    REQUIRE(::truncate(file.path.c_str(), 16) == 0);

    const float query[] = {0.0f, 0.0f};
    const auto [distances, indices] = index.search(query, 1, 1, 2);
    REQUIRE(distances.size() == 1);
    REQUIRE(indices.size() == 1);
    REQUIRE(distances[0] == Catch::Approx(2.0f));
}

TEST_CASE("CudaFlatIndex snapshot survives source truncation",
          "[index][cuda-flat][io-contract]") {
    const auto child = resonantia_test::run_self_test(
        "hidden CudaFlatIndex truncate rejection child", 2'000);
    INFO("timed_out=" << child.timed_out << " status=" << child.status
                       << " launch_error=" << child.launch_error
                       << " wait_error=" << child.wait_error);
    REQUIRE(child.exited_with(0));
}

TEST_CASE("CudaFlatIndex snapshot ignores later same-size source mutation",
          "[index][cuda-flat][io-contract]") {
    TempIndex file;
    REQUIRE(write_binary(file.path,
                         index_bytes(2, 2, {0.0f, 0.0f, 1.0f, 1.0f})));

    voxmutatio::index::CudaFlatIndex index;
    REQUIRE(index.load(file.path));
    REQUIRE(write_binary(file.path,
                         index_bytes(2, 2, {2.0f, 2.0f, 3.0f, 3.0f})));

    const float query[] = {0.0f, 0.0f};
    const auto [distances, indices] = index.search(query, 1, 1, 2);
    REQUIRE(distances.size() == 1);
    REQUIRE(indices.size() == 1);
    REQUIRE(indices[0] == 0);
    REQUIRE(distances[0] == Catch::Approx(0.0f));
}

TEST_CASE("CudaFlatIndex snapshot ignores later atomic path replacement",
          "[index][cuda-flat][io-contract]") {
    TempIndex file;
    TempIndex replacement;
    REQUIRE(write_binary(file.path,
                         index_bytes(2, 2, {0.0f, 0.0f, 1.0f, 1.0f})));
    REQUIRE(write_binary(replacement.path,
                         index_bytes(2, 2, {4.0f, 4.0f, 5.0f, 5.0f})));

    voxmutatio::index::CudaFlatIndex index;
    REQUIRE(index.load(file.path));
    REQUIRE(::rename(replacement.path.c_str(), file.path.c_str()) == 0);

    const float query[] = {0.0f, 0.0f};
    const auto [distances, indices] = index.search(query, 1, 1, 2);
    REQUIRE(distances.size() == 1);
    REQUIRE(indices.size() == 1);
    REQUIRE(indices[0] == 0);
    REQUIRE(distances[0] == Catch::Approx(0.0f));
}
