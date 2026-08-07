#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace voxmutatio::index {

/// CUDA-based Flat Index for feature retrieval (zero FAISS dependency)
class CudaFlatIndex {
public:
    /// Load a feature index from disk
    bool load(const std::string& path);

    /// Search K nearest neighbors for each query vector on GPU
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
    [[nodiscard]] bool valid() const noexcept { return data_ != nullptr; }

    ~CudaFlatIndex();

private:
    float* data_ = nullptr;       // points into map_base_ (past 16-byte header)
    float* device_data_ = nullptr; // device copy
    void* map_base_ = nullptr;     // mmap base (whole file)
    std::size_t map_size_ = 0;     // mmap size
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
