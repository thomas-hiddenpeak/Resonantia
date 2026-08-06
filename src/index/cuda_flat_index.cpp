// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/index/cuda_flat_index.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace voxmutatio::index {

namespace {

// Host-side L2 distance computation (GPU version in .cu file)
void compute_l2_distances_host(const float* queries, const float* database,
                                float* distances, int n_queries,
                                int n_database, int dim) {
    for (int i = 0; i < n_queries; ++i) {
        for (int j = 0; j < n_database; ++j) {
            float dist = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = queries[i * dim + d] - database[j * dim + d];
                dist += diff * diff;
            }
            distances[i * n_database + j] = dist;
        }
    }
}

}  // namespace

bool CudaFlatIndex::load(const std::string& path) {
    // Free existing resources
    if (data_) {
        munmap(data_, ntotal_ * dim_ * sizeof(float));
        data_ = nullptr;
    }
    if (device_data_) {
        cudaFree(device_data_);
        device_data_ = nullptr;
    }
    
    // Open file
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return false;
    }
    
    // Read header (ntotal, dim)
    if (st.st_size < sizeof(int64_t) * 2) {
        close(fd);
        return false;
    }
    
    void* header = mmap(nullptr, sizeof(int64_t) * 2, PROT_READ, MAP_PRIVATE, fd, 0);
    if (header == MAP_FAILED) {
        close(fd);
        return false;
    }
    
    const int64_t* header_data = static_cast<const int64_t*>(header);
    ntotal_ = header_data[0];
    dim_ = header_data[1];
    
    munmap(header, sizeof(int64_t) * 2);
    
    // Validate
    std::size_t expected_size = sizeof(int64_t) * 2 + 
                                ntotal_ * dim_ * sizeof(float);
    if (static_cast<std::size_t>(st.st_size) < expected_size) {
        close(fd);
        return false;
    }
    
    // Mmap data
    std::size_t data_size = ntotal_ * dim_ * sizeof(float);
    data_ = static_cast<float*>(
        mmap(nullptr, data_size, PROT_READ, MAP_PRIVATE, fd, 
             sizeof(int64_t) * 2));
    close(fd);
    
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        return false;
    }
    
    // Advise kernel about random access pattern
    madvise(data_, data_size, MADV_RANDOM);
    
    // Copy to GPU
    cudaError_t err = cudaMalloc(&device_data_, data_size);
    if (err != cudaSuccess) {
        munmap(data_, data_size);
        data_ = nullptr;
        return false;
    }
    
    err = cudaMemcpy(device_data_, data_, data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(device_data_);
        device_data_ = nullptr;
        munmap(data_, data_size);
        data_ = nullptr;
        return false;
    }
    
    return true;
}

std::pair<std::vector<float>, std::vector<int64_t>>
CudaFlatIndex::search(const float* queries, int n_queries, int k, int dim) {
    if (!valid() || dim != dim_) {
        return {{}, {}};
    }
    
    // Compute L2 distances on host (GPU version uses device_data_)
    std::vector<float> distances(n_queries * ntotal_);
    compute_l2_distances_host(queries, data_, distances.data(),
                              n_queries, ntotal_, dim);
    
    // Find top-K on host (for now)
    std::vector<float> result_dists(n_queries * k);
    std::vector<int64_t> result_idxs(n_queries * k);
    
    for (int i = 0; i < n_queries; ++i) {
        // Create index array
        std::vector<int64_t> idxs(ntotal_);
        std::iota(idxs.begin(), idxs.end(), 0);
        
        // Partial sort to find top-K
        const float* q_dists = distances.data() + i * ntotal_;
        
        std::partial_sort(idxs.begin(), idxs.begin() + k, idxs.end(),
            [q_dists](int64_t a, int64_t b) {
                return q_dists[a] < q_dists[b];
            });
        
        for (int j = 0; j < k; ++j) {
            result_dists[i * k + j] = q_dists[idxs[j]];
            result_idxs[i * k + j] = idxs[j];
        }
    }
    
    return {result_dists, result_idxs};
}

std::vector<float> CudaFlatIndex::reconstruct(int64_t idx, int dim) {
    if (!valid() || idx < 0 || idx >= ntotal_ || dim != dim_) {
        return {};
    }
    
    std::vector<float> result(dim);
    std::memcpy(result.data(), data_ + idx * dim_, dim * sizeof(float));
    return result;
}

CudaFlatIndex::~CudaFlatIndex() {
    if (data_) {
        munmap(data_, ntotal_ * dim_ * sizeof(float));
        data_ = nullptr;
    }
    if (device_data_) {
        cudaFree(device_data_);
        device_data_ = nullptr;
    }
}

void blend_features(float* output, const float* original,
                    const float* retrieved, int frames, int dim,
                    double rate) {
    std::size_t total = frames * dim;
    
    for (std::size_t i = 0; i < total; ++i) {
        output[i] = original[i] * (1.0 - rate) + retrieved[i] * rate;
    }
}

}  // namespace voxmutatio::index
