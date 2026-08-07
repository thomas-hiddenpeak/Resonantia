// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// RAII owner for device (GPU) float memory. Replaces raw cudaMalloc/cudaFree.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
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
    cudaError_t e = cudaMalloc(&ptr_, n * sizeof(float));
    if (e != cudaSuccess) {
      fprintf(stderr, "CudaBuffer: cudaMalloc(%zu) failed: %s\n", n,
              cudaGetErrorString(e));
      ptr_ = nullptr;
      size_ = 0;
      return;
    }
    size_ = n;
  }

  /// Allocate only if current capacity differs.
  void resize(std::size_t n) {
    if (n != size_) allocate(n);
  }

  /// Zero all bytes (float 0.0f).
  void zero() {
    if (ptr_) cudaMemset(ptr_, 0, size_ * sizeof(float));
  }

  void copy_from_host(const float* src, std::size_t n) {
    if (n > size_) allocate(n);
    cudaMemcpy(ptr_, src, n * sizeof(float), cudaMemcpyHostToDevice);
  }
  void copy_to_host(float* dst, std::size_t n) const {
    cudaMemcpy(dst, ptr_, n * sizeof(float), cudaMemcpyDeviceToHost);
  }
  void copy_from_device(const float* src, std::size_t n) {
    if (n > size_) allocate(n);
    cudaMemcpy(ptr_, src, n * sizeof(float), cudaMemcpyDeviceToDevice);
  }

  [[nodiscard]] float* data() noexcept { return ptr_; }
  [[nodiscard]] const float* data() const noexcept { return ptr_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

 private:
  void free_() {
    if (ptr_) cudaFree(ptr_);
    ptr_ = nullptr;
    size_ = 0;
  }
  float* ptr_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace voxmutatio::core
