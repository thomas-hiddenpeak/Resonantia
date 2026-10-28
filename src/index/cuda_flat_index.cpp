// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/index/cuda_flat_index.h"

#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

namespace voxmutatio::index {

namespace {

constexpr std::size_t kHeaderSize = sizeof(std::int64_t) * 2;
constexpr std::size_t kReadChunkBytes = 1U << 20;

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
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        offset += static_cast<std::size_t>(count);
    }
    return true;
}

bool read_exact_at(int fd, void* output, std::size_t byte_count,
                   std::size_t byte_offset) noexcept {
    if (byte_count == 0) return true;
    if (output == nullptr ||
        byte_offset > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) ||
        byte_count > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) -
                         byte_offset) {
        return false;
    }
    auto* destination = static_cast<std::uint8_t*>(output);
    std::size_t completed = 0;
    while (completed < byte_count) {
        const ssize_t count = ::pread(
            fd, destination + completed, byte_count - completed,
            static_cast<off_t>(byte_offset + completed));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        completed += static_cast<std::size_t>(count);
    }
    return true;
}

bool write_exact_at(int fd, const void* input, std::size_t byte_count,
                    std::size_t byte_offset) noexcept {
    if (byte_count == 0) return true;
    if (input == nullptr ||
        byte_offset > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) ||
        byte_count > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) -
                         byte_offset) {
        return false;
    }
    const auto* source = static_cast<const std::uint8_t*>(input);
    std::size_t completed = 0;
    while (completed < byte_count) {
        const ssize_t count = ::pwrite(
            fd, source + completed, byte_count - completed,
            static_cast<off_t>(byte_offset + completed));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        completed += static_cast<std::size_t>(count);
    }
    return true;
}

class ScopedFd {
public:
    explicit ScopedFd(int fd = -1) noexcept : fd_(fd) {}
    ~ScopedFd() {
        if (fd_ >= 0) ::close(fd_);
    }
    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;
    [[nodiscard]] int get() const noexcept { return fd_; }
    [[nodiscard]] int release() noexcept { return std::exchange(fd_, -1); }

private:
    int fd_ = -1;
};

ScopedFd create_private_snapshot(const std::filesystem::path& source_path) {
    const std::filesystem::path parent = source_path.parent_path();
#ifdef O_TMPFILE
    if (!parent.empty()) {
        const int descriptor = ::open(parent.c_str(),
                                      O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
        if (descriptor >= 0) return ScopedFd(descriptor);
    }
#endif

    char temporary[] = "/tmp/resonantia-index-XXXXXX";
    const int descriptor = ::mkostemp(temporary, O_CLOEXEC);
    if (descriptor < 0) return ScopedFd();
    if (::unlink(temporary) != 0) {
        const int saved_error = errno;
        ::close(descriptor);
        errno = saved_error;
        return ScopedFd();
    }
    return ScopedFd(descriptor);
}

bool clone_or_copy_source(int source_fd, int snapshot_fd,
                          std::size_t byte_count, bool& cloned) {
    cloned = false;
    if (::ioctl(snapshot_fd, FICLONE, source_fd) == 0) {
        struct stat cloned_stat {};
        if (::fstat(snapshot_fd, &cloned_stat) == 0 &&
            cloned_stat.st_size >= 0 &&
            static_cast<std::uintmax_t>(cloned_stat.st_size) == byte_count) {
            cloned = true;
            return true;
        }
    }

    if (::ftruncate(snapshot_fd, 0) != 0 ||
        ::ftruncate(snapshot_fd, static_cast<off_t>(byte_count)) != 0) {
        return false;
    }
    std::vector<std::uint8_t> buffer(
        std::min<std::size_t>(byte_count, kReadChunkBytes));
    for (std::size_t offset = 0; offset < byte_count;) {
        const std::size_t count = std::min(buffer.size(), byte_count - offset);
        if (!read_exact_at(source_fd, buffer.data(), count, offset) ||
            !write_exact_at(snapshot_fd, buffer.data(), count, offset)) {
            return false;
        }
        offset += count;
    }
    return true;
}

bool same_file_contents(int left_fd, int right_fd, std::size_t byte_count) {
    std::vector<std::uint8_t> left(
        std::min<std::size_t>(byte_count, kReadChunkBytes));
    std::vector<std::uint8_t> right(left.size());
    for (std::size_t offset = 0; offset < byte_count;) {
        const std::size_t count = std::min(left.size(), byte_count - offset);
        if (!read_exact_at(left_fd, left.data(), count, offset) ||
            !read_exact_at(right_fd, right.data(), count, offset) ||
            std::memcmp(left.data(), right.data(), count) != 0) {
            return false;
        }
        offset += count;
    }
    return true;
}

struct FileIdentity {
    std::uint64_t device = 0;
    std::uint64_t inode = 0;
    std::uint64_t size = 0;
    std::int64_t modified_seconds = 0;
    std::int64_t modified_nanoseconds = 0;
    std::int64_t changed_seconds = 0;
    std::int64_t changed_nanoseconds = 0;
};

bool identity_from_stat(const struct stat& value, FileIdentity& identity) noexcept {
    if (!S_ISREG(value.st_mode) || value.st_size < 0) return false;
    identity.device = static_cast<std::uint64_t>(value.st_dev);
    identity.inode = static_cast<std::uint64_t>(value.st_ino);
    identity.size = static_cast<std::uint64_t>(value.st_size);
    identity.modified_seconds = static_cast<std::int64_t>(value.st_mtim.tv_sec);
    identity.modified_nanoseconds = static_cast<std::int64_t>(value.st_mtim.tv_nsec);
    identity.changed_seconds = static_cast<std::int64_t>(value.st_ctim.tv_sec);
    identity.changed_nanoseconds = static_cast<std::int64_t>(value.st_ctim.tv_nsec);
    return true;
}

bool same_identity(const FileIdentity& left, const FileIdentity& right) noexcept {
    return left.device == right.device && left.inode == right.inode &&
           left.size == right.size &&
           left.modified_seconds == right.modified_seconds &&
           left.modified_nanoseconds == right.modified_nanoseconds &&
           left.changed_seconds == right.changed_seconds &&
           left.changed_nanoseconds == right.changed_nanoseconds;
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
    if (map_base_) ::munmap(map_base_, map_size_);
    data_ = nullptr;
    map_base_ = nullptr;
    map_size_ = 0;
    ntotal_ = 0;
    dim_ = 0;
}

bool CudaFlatIndex::load(const std::string& path) try {
    release();

    const std::string resolved_path =
        std::filesystem::absolute(std::filesystem::path(path))
            .lexically_normal()
            .string();
    ScopedFd source_fd(::open(resolved_path.c_str(),
                              O_RDONLY | O_CLOEXEC | O_NONBLOCK));
    if (source_fd.get() < 0) return false;

    struct stat st {};
    std::uint8_t header[kHeaderSize] {};
    FileIdentity initial_identity;
    if (::fstat(source_fd.get(), &st) < 0 ||
        !identity_from_stat(st, initial_identity) ||
        st.st_size < static_cast<off_t>(kHeaderSize) ||
        static_cast<std::uintmax_t>(st.st_size) > std::numeric_limits<std::size_t>::max() ||
        !read_header(source_fd.get(), header)) return false;

    const std::uint64_t ntotal_u64 = read_u64_le(header);
    const std::uint64_t dim_u64 = read_u64_le(header + sizeof(std::int64_t));
    if (ntotal_u64 == 0 || ntotal_u64 > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()) ||
        dim_u64 == 0 || dim_u64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        ntotal_u64 > std::numeric_limits<std::size_t>::max()) {
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
        return false;
    }

    ScopedFd snapshot_fd(
        create_private_snapshot(std::filesystem::path(resolved_path)).release());
    bool cloned = false;
    if (snapshot_fd.get() < 0 ||
        !clone_or_copy_source(source_fd.get(), snapshot_fd.get(), expected_size,
                              cloned) ||
        (!cloned && !same_file_contents(source_fd.get(), snapshot_fd.get(),
                                        expected_size))) {
        return false;
    }

    struct stat final_descriptor_stat {};
    struct stat final_path_stat {};
    FileIdentity final_descriptor_identity;
    FileIdentity final_path_identity;
    if (::fstat(source_fd.get(), &final_descriptor_stat) != 0 ||
        ::stat(resolved_path.c_str(), &final_path_stat) != 0 ||
        !identity_from_stat(final_descriptor_stat, final_descriptor_identity) ||
        !identity_from_stat(final_path_stat, final_path_identity) ||
        !same_identity(initial_identity, final_descriptor_identity) ||
        !same_identity(initial_identity, final_path_identity)) {
        return false;
    }

    map_size_ = expected_size;
    map_base_ = ::mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE,
                       snapshot_fd.get(), 0);
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
    ::madvise(map_base_, map_size_, MADV_RANDOM);
    return true;
} catch (...) {
    release();
    return false;
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
    // Queries are independent. Preserve each scalar accumulation and partial_sort
    // (including its tie behavior), with at most four O(database_count) scratch
    // sets. No shared pool, model-sized device copy, or Nqueries x Ndatabase matrix.
    const std::size_t worker_count = query_count >= 4 && database_count >= 256
        ? std::min<std::size_t>(4, std::max(1u, std::thread::hardware_concurrency()))
        : 1;
    std::atomic<bool> failed{false};
    const auto search_queries = [&](std::size_t worker) noexcept {
        try {
            std::vector<float> distances(database_count);
            std::vector<int64_t> indices(database_count);
            for (std::size_t i = worker; i < query_count && !failed.load();
                 i += worker_count) {
                for (std::size_t j = 0; j < database_count; ++j) {
                    float distance = 0.0f;
                    for (std::size_t d = 0; d < dimension; ++d) {
                        const float difference =
                            queries[i * dimension + d] - data_[j * dimension + d];
                        distance += difference * difference;
                    }
                    distances[j] = distance;
                }
                std::iota(indices.begin(), indices.end(), 0);
                std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&distances](int64_t a, int64_t b) {
                        return distances[static_cast<std::size_t>(a)] <
                               distances[static_cast<std::size_t>(b)];
                    });
                for (int j = 0; j < k; ++j) {
                    result_distances[i * k + j] =
                        distances[static_cast<std::size_t>(indices[j])];
                    result_indices[i * k + j] = indices[j];
                }
            }
        } catch (...) {
            // Never allow an allocation failure in a worker to terminate the
            // process, and never publish a partially populated search result.
            failed.store(true);
        }
    };
    {
        std::vector<std::jthread> workers;
        workers.reserve(worker_count - 1);
        for (std::size_t worker = 1; worker < worker_count; ++worker)
            workers.emplace_back(search_queries, worker);
        search_queries(0);
        // jthread also joins already-started workers if thread creation throws.
    }
    if (failed.load()) return {{}, {}};
    return {std::move(result_distances), std::move(result_indices)};
} catch (const std::bad_alloc&) {
    return {{}, {}};
} catch (const std::length_error&) {
    return {{}, {}};
} catch (const std::system_error&) {
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
                static_cast<std::size_t>(
                    indices[static_cast<std::size_t>(i) * k + j]) * dim_;
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
