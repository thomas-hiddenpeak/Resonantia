// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace resonantia_test::cuda_failure {

// This controller is private to one serial Catch2/CUDA test executable. Its
// fixed-capacity bookkeeping performs no host allocation on noexcept paths.
enum class Operation : std::size_t {
    kAllocation = 0,
    kCopy,
    kLaunch,
    kSynchronize,
    kCount,
};

// Reset only the configured fault and per-operation observations. Live device
// allocations deliberately remain tracked so a reset cannot hide a leak.
void reset() noexcept;
void fail_nth(Operation operation, std::size_t invocation) noexcept;

[[nodiscard]] std::size_t call_count(Operation operation) noexcept;
[[nodiscard]] std::size_t live_allocation_count() noexcept;
[[nodiscard]] std::size_t peak_live_allocation_count() noexcept;

cudaError_t allocate_bytes(void** pointer, std::size_t bytes) noexcept;

template <typename T>
cudaError_t allocate(T** pointer, std::size_t bytes) noexcept {
    return allocate_bytes(reinterpret_cast<void**>(pointer), bytes);
}

cudaError_t release(void* pointer) noexcept;
cudaError_t copy(void* destination, const void* source, std::size_t bytes,
                 cudaMemcpyKind kind) noexcept;
cudaError_t before_launch() noexcept;
cudaError_t synchronize() noexcept;

}  // namespace resonantia_test::cuda_failure
