// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/index/cuda_flat_index.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace voxmutatio::index {

namespace {

constexpr std::size_t kHeaderSize = sizeof(std::int64_t) * 2;

bool checked_add(std::size_t a, std::size_t b, std::size_t& result) {
    if (b > std::numeric_limits<std::size_t>::max() - a) return false;
    result = a + b;
    return true;
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t& result) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
    result = a * b;
    return true;
}

std::uint64_t read_u64_le(const std::uint8_t* data) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(data[i]) << (i * 8);
    }
    return value;
}

bool read_header(int fd, std::uint8_t (&header)[kHeaderSize]) {
    std::size_t offset = 0;
    while (offset < kHeaderSize) {
        const ssize_t count = pread(fd, header + offset, kHeaderSize - offset,
                                    static_cast<off_t>(offset));
        if (count <= 0) return false;
        offset += static_cast<std::size_t>(count);
    }
    return true;
}

void compute_l2_distances_host(const float* queries, const float* database,
                               float* distances, std::size_t n_queries,
                               std::size_t n_database, std::size_t dim) {
    for (std::size_t i = 0; i < n_queries; ++i) {
        for (std::size_t j = 0; j < n_database; ++j) {
            float distance = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
                const float difference = queries[i * dim + d] - database[j * dim + d];
                distance += difference * difference;
            }
            distances[i * n_database + j] = distance;
        }
    }
}

}  // namespace

CudaFlatIndex::CudaFlatIndex(CudaFlatIndex&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      map_base_(std::exchange(other.map_base_, nullptr)),
      map_size_(std::exchange(other.map_size_, 0)),
      ntotal_(std::exchange(other.ntotal_, 0)),
      dim_(std::exchange(other.dim_, 0)) {}

CudaFlatIndex& CudaFlatIndex::operator=(CudaFlatIndex&& other) noexcept {
    if (this != &other) {
        release();
        data_ = std::exchange(other.data_, nullptr);
        map_base_ = std::exchange(other.map_base_, nullptr);
        map_size_ = std::exchange(other.map_size_, 0);
        ntotal_ = std::exchange(other.ntotal_, 0);
        dim_ = std::exchange(other.dim_, 0);
    }
    return *this;
}

void CudaFlatIndex::release() noexcept {
    if (map_base_) munmap(map_base_, map_size_);
    data_ = nullptr;
    map_base_ = nullptr;
    map_size_ = 0;
    ntotal_ = 0;
    dim_ = 0;
}

bool CudaFlatIndex::load(const std::string& path) {
    release();

    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;

    struct stat st {};
    std::uint8_t header[kHeaderSize] {};
    if (fstat(fd, &st) < 0 || !S_ISREG(st.st_mode) ||
        st.st_size < static_cast<off_t>(kHeaderSize) ||
        static_cast<std::uintmax_t>(st.st_size) > std::numeric_limits<std::size_t>::max() ||
        !read_header(fd, header)) {
        close(fd);
        return false;
    }

    const std::uint64_t ntotal_u64 = read_u64_le(header);
    const std::uint64_t dim_u64 = read_u64_le(header + sizeof(std::int64_t));
    if (ntotal_u64 == 0 || ntotal_u64 > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()) ||
        dim_u64 == 0 || dim_u64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        ntotal_u64 > std::numeric_limits<std::size_t>::max()) {
        close(fd);
        return false;
    }

    std::size_t element_count = 0;
    std::size_t data_size = 0;
    std::size_t expected_size = 0;
    if (!checked_mul(static_cast<std::size_t>(ntotal_u64),
                     static_cast<std::size_t>(dim_u64), element_count) ||
        !checked_mul(element_count, sizeof(float), data_size) ||
        !checked_add(kHeaderSize, data_size, expected_size) ||
        expected_size != static_cast<std::size_t>(st.st_size)) {
        close(fd);
        return false;
    }

    map_size_ = expected_size;
    map_base_ = mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map_base_ == MAP_FAILED) {
        map_base_ = nullptr;
        map_size_ = 0;
        return false;
    }

    data_ = reinterpret_cast<const float*>(
        static_cast<const char*>(map_base_) + kHeaderSize);
    for (std::size_t i = 0; i < element_count; ++i) {
        if (!std::isfinite(data_[i])) {
            release();
            return false;
        }
    }

    ntotal_ = static_cast<int64_t>(ntotal_u64);
    dim_ = static_cast<int>(dim_u64);
    madvise(map_base_, map_size_, MADV_RANDOM);

    return true;
}

std::pair<std::vector<float>, std::vector<int64_t>>
CudaFlatIndex::search(const float* queries, int n_queries, int k, int dim) try {
    if (!valid() || queries == nullptr || n_queries <= 0 || k <= 0 || dim <= 0 ||
        dim != dim_ || static_cast<int64_t>(k) > ntotal_) {
        return {{}, {}};
    }

    const std::size_t query_count = static_cast<std::size_t>(n_queries);
    const std::size_t database_count = static_cast<std::size_t>(ntotal_);
    const std::size_t dimension = static_cast<std::size_t>(dim);
    std::size_t query_elements = 0;
    std::size_t result_count = 0;
    if (!checked_mul(query_count, dimension, query_elements) ||
        !checked_mul(query_count, static_cast<std::size_t>(k), result_count) ||
        result_count > std::vector<float>().max_size() ||
        result_count > std::vector<int64_t>().max_size()) {
        return {{}, {}};
    }
    for (std::size_t i = 0; i < query_elements; ++i) {
        if (!std::isfinite(queries[i])) return {{}, {}};
    }

    std::vector<float> result_distances(result_count);
    std::vector<int64_t> result_indices(result_count);
    std::vector<float> distances(database_count);
    std::vector<int64_t> indices(database_count);
    std::iota(indices.begin(), indices.end(), 0);

    for (std::size_t i = 0; i < query_count; ++i) {
        compute_l2_distances_host(queries + i * dimension, data_, distances.data(), 1,
                                  database_count, dimension);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
            [&distances](int64_t a, int64_t b) {
                return distances[a] < distances[b];
            });
        for (int j = 0; j < k; ++j) {
            result_distances[i * k + j] = distances[indices[j]];
            result_indices[i * k + j] = indices[j];
        }
    }
    return {std::move(result_distances), std::move(result_indices)};
} catch (const std::bad_alloc&) {
    return {{}, {}};
} catch (const std::length_error&) {
    return {{}, {}};
}

std::vector<float> CudaFlatIndex::reconstruct(int64_t idx, int dim) try {
    if (!valid() || idx < 0 || idx >= ntotal_ || dim <= 0 || dim != dim_) return {};
    const std::size_t dimension = static_cast<std::size_t>(dim_);
    const std::size_t offset = static_cast<std::size_t>(idx) * dimension;
    std::vector<float> result(dimension);
    std::memcpy(result.data(), data_ + offset, dimension * sizeof(float));
    return result;
} catch (const std::bad_alloc&) {
    return {};
} catch (const std::length_error&) {
    return {};
}

std::vector<float> CudaFlatIndex::retrieve_weighted(const float* queries,
                                                     int n_queries, int k, int dim) try {
    if (!valid() || queries == nullptr || n_queries <= 0 || k <= 0 || dim <= 0 ||
        dim != dim_ || ntotal_ <= 0) {
        return {};
    }
    k = static_cast<int>(std::min<int64_t>(k, ntotal_));

    auto [distances, indices] = search(queries, n_queries, k, dim);
    if (indices.empty()) return {};

    std::size_t output_count = 0;
    if (!checked_mul(static_cast<std::size_t>(n_queries),
                     static_cast<std::size_t>(dim), output_count) ||
        output_count > std::vector<float>().max_size()) {
        return {};
    }
    std::vector<float> output(output_count, 0.0f);
    for (int i = 0; i < n_queries; ++i) {
        std::vector<double> weights(static_cast<std::size_t>(k));
        double weight_sum = 0.0;
        for (int j = 0; j < k; ++j) {
            const double distance = distances[static_cast<std::size_t>(i) * k + j];
            double weight = 1.0 / (distance + 1e-8);
            weight *= weight;
            weights[j] = weight;
            weight_sum += weight;
        }
        if (!std::isfinite(weight_sum) || weight_sum <= 0.0) return {};
        for (int j = 0; j < k; ++j) {
            const double weight = weights[j] / weight_sum;
            const float* vector = data_ +
                static_cast<std::size_t>(indices[static_cast<std::size_t>(i) * k + j]) * dim_;
            for (int d = 0; d < dim; ++d) {
                output[static_cast<std::size_t>(i) * dim + d] +=
                    static_cast<float>(weight * vector[d]);
            }
        }
    }
    return output;
} catch (const std::bad_alloc&) {
    return {};
} catch (const std::length_error&) {
    return {};
}

CudaFlatIndex::~CudaFlatIndex() {
    release();
}

void blend_features(float* output, const float* original,
                    const float* retrieved, int frames, int dim,
                    double rate) {
    if (output == nullptr || original == nullptr || retrieved == nullptr ||
        frames <= 0 || dim <= 0 || !std::isfinite(rate) || rate < 0.0 || rate > 1.0) {
        return;
    }
    std::size_t total = 0;
    if (!checked_mul(static_cast<std::size_t>(frames),
                     static_cast<std::size_t>(dim), total)) {
        return;
    }
    for (std::size_t i = 0; i < total; ++i) {
        output[i] = static_cast<float>(original[i] * (1.0 - rate) + retrieved[i] * rate);
    }
}

}  // namespace voxmutatio::index
