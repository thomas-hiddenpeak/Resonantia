// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// RAII owner for device (GPU) float memory. Replaces raw cudaMalloc/cudaFree.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace voxmutatio::core {

/// Move-only RAII wrapper around a device float buffer.
class CudaBuffer {
 public:
  CudaBuffer() = default;
  explicit CudaBuffer(std::size_t n) { allocate(n); }
  ~CudaBuffer() { free_(); }

  CudaBuffer(CudaBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
  }
  CudaBuffer& operator=(CudaBuffer&& o) noexcept {
    if (this != &o) {
      free_();
      ptr_ = o.ptr_;
      size_ = o.size_;
      o.ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  /// (Re)allocate to hold n floats. Contents undefined.
  void allocate(std::size_t n) {
    free_();
    if (n == 0) return;
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      throw std::length_error("CudaBuffer allocation size overflow");
    }
    cudaError_t e = cudaMalloc(&ptr_, n * sizeof(float));
    if (e != cudaSuccess) {
      ptr_ = nullptr;
      size_ = 0;
      throw_cuda_("cudaMalloc", e, n);
    }
    size_ = n;
  }

  /// Allocate only if current capacity differs.
  void resize(std::size_t n) {
    if (n != size_) allocate(n);
  }

  /// Zero all bytes (float 0.0f).
  void zero() {
    if (!ptr_) return;
    cudaError_t e = cudaMemset(ptr_, 0, size_ * sizeof(float));
    if (e != cudaSuccess) throw_cuda_("cudaMemset", e, size_);
  }

  void copy_from_host(const float* src, std::size_t n) {
    if (n == 0) return;
    if (!src) throw std::invalid_argument("CudaBuffer::copy_from_host null source");
    if (n > size_) allocate(n);
    cudaError_t e = cudaMemcpy(ptr_, src, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) throw_cuda_("cudaMemcpy(H2D)", e, n);
  }
  void copy_to_host(float* dst, std::size_t n) const {
    if (n == 0) return;
    if (!dst) throw std::invalid_argument("CudaBuffer::copy_to_host null destination");
    if (!ptr_ || n > size_) throw std::out_of_range("CudaBuffer::copy_to_host exceeds buffer");
    cudaError_t e = cudaMemcpy(dst, ptr_, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) throw_cuda_("cudaMemcpy(D2H)", e, n);
  }
  void copy_from_device(const float* src, std::size_t n) {
    if (n == 0) return;
    if (!src) throw std::invalid_argument("CudaBuffer::copy_from_device null source");
    if (n > size_) allocate(n);
    cudaError_t e = cudaMemcpy(ptr_, src, n * sizeof(float), cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess) throw_cuda_("cudaMemcpy(D2D)", e, n);
  }

  [[nodiscard]] float* data() noexcept { return ptr_; }
  [[nodiscard]] const float* data() const noexcept { return ptr_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

 private:
  [[noreturn]] static void throw_cuda_(const char* operation, cudaError_t error,
                                      std::size_t elements) {
    throw std::runtime_error(std::string("CudaBuffer ") + operation + " failed for " +
                             std::to_string(elements) + " floats: " +
                             cudaGetErrorString(error));
  }

  void free_() {
    if (ptr_) {
      cudaError_t e = cudaFree(ptr_);
      if (e != cudaSuccess) {
        // Destructors and move assignment cannot safely throw. Preserve the first
        // CUDA failure in stderr while ensuring ownership is cleared exactly once.
        fprintf(stderr, "CudaBuffer: cudaFree failed: %s\n", cudaGetErrorString(e));
      }
    }
    ptr_ = nullptr;
    size_ = 0;
  }
  float* ptr_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace voxmutatio::core
