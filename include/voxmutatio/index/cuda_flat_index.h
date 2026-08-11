#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace voxmutatio::index {

/// Immutable-snapshot exact Flat Index for feature retrieval (zero FAISS
/// dependency).
///
/// load() copies or reflinks the validated source into a private, unlinked
/// disk-backed snapshot. Search retains the existing mmap exact-scan behavior,
/// while later replacement, truncation, or overwrite of the source path cannot
/// mutate the active index or raise SIGBUS. The snapshot consumes index-sized
/// temporary filesystem capacity when copy-on-write cloning is unavailable
/// (which may be memory/swap-backed on tmpfs), but does not explicitly allocate
/// an index-sized anonymous-memory duplicate.
class CudaFlatIndex {
public:
    CudaFlatIndex() = default;
    CudaFlatIndex(const CudaFlatIndex&) = delete;
    CudaFlatIndex& operator=(const CudaFlatIndex&) = delete;
    CudaFlatIndex(CudaFlatIndex&& other) noexcept;
    CudaFlatIndex& operator=(CudaFlatIndex&& other) noexcept;

    /// Load a feature index from disk
    bool load(const std::string& path);

    /// Search K nearest neighbors for each query vector without duplicating the
    /// database in device memory.
    /// queries: [N, dim] row-major
    /// Returns {distances, indices} each [N, K] row-major
    std::pair<std::vector<float>, std::vector<int64_t>>
    search(const float* queries, int n_queries, int k, int dim);

    /// Reconstruct a vector by its index
    std::vector<float> reconstruct(int64_t idx, int dim);

    /// RVC-style weighted k-NN retrieval.
    /// For each query, finds k nearest neighbors and returns their weighted
    /// average, where weight = (1/dist)^2 normalized (matches RVC infer).
    /// queries: [n, dim] -> returns [n, dim].
    std::vector<float> retrieve_weighted(const float* queries, int n_queries,
                                         int k, int dim);

    /// Get total number of vectors in the index
    [[nodiscard]] int64_t total_vectors() const noexcept { return ntotal_; }

    /// Get dimensionality
    [[nodiscard]] int dim() const noexcept { return dim_; }

    /// Check if index is loaded
    [[nodiscard]] bool valid() const noexcept {
        return data_ != nullptr && map_base_ != nullptr && ntotal_ > 0 && dim_ > 0;
    }

    ~CudaFlatIndex();

private:
    void release() noexcept;

    const float* data_ = nullptr; // points into map_base_ (past 16-byte header)
    void* map_base_ = nullptr;     // mmap base of the private disk snapshot
    std::size_t map_size_ = 0;
    int64_t ntotal_ = 0;
    int dim_ = 0;
};

/// Blend original features with retrieved index features
/// original: [T, dim]
/// retrieved: [T, dim]
/// rate: blend factor [0.0 = original only, 1.0 = retrieved only]
void blend_features(float* output, const float* original,
                    const float* retrieved, int frames, int dim,
                    double rate);

}  // namespace voxmutatio::index
