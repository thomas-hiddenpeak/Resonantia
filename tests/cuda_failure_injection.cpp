// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "cuda_failure_injection.h"

#include <algorithm>
#include <array>

namespace resonantia_test::cuda_failure {
namespace {

constexpr std::size_t operation_count =
    static_cast<std::size_t>(Operation::kCount);
constexpr std::size_t max_tracked_allocations = 64;

struct State {
    std::array<std::size_t, operation_count> calls{};
    std::array<std::size_t, operation_count> fail_at{};
    std::array<void*, max_tracked_allocations> live_allocations{};
    std::size_t live_allocation_count = 0;
    std::size_t peak_live_allocations = 0;
};

State& state() {
    static State value;
    return value;
}

std::size_t index(Operation operation) noexcept {
    return static_cast<std::size_t>(operation);
}

bool valid(Operation operation) noexcept {
    return index(operation) < operation_count;
}

bool should_fail(Operation operation) noexcept {
    auto& value = state();
    const auto operation_index = index(operation);
    const auto invocation = ++value.calls[operation_index];
    return value.fail_at[operation_index] != 0 &&
           invocation == value.fail_at[operation_index];
}

}  // namespace

void reset() noexcept {
    auto& value = state();
    value.calls.fill(0);
    value.fail_at.fill(0);
    value.peak_live_allocations = value.live_allocation_count;
}

void fail_nth(Operation operation, std::size_t invocation) noexcept {
    if (!valid(operation)) return;
    state().fail_at[index(operation)] = invocation;
}

std::size_t call_count(Operation operation) noexcept {
    if (!valid(operation)) return 0;
    return state().calls[index(operation)];
}

std::size_t live_allocation_count() noexcept {
    return state().live_allocation_count;
}

std::size_t peak_live_allocation_count() noexcept {
    return state().peak_live_allocations;
}

cudaError_t allocate_bytes(void** pointer, std::size_t bytes) noexcept {
    if (pointer == nullptr) return cudaErrorInvalidValue;
    *pointer = nullptr;
    if (should_fail(Operation::kAllocation)) return cudaErrorMemoryAllocation;

    const cudaError_t error = cudaMalloc(pointer, bytes);
    if (error == cudaSuccess) {
        auto& value = state();
        if (value.live_allocation_count == value.live_allocations.size()) {
            const cudaError_t cleanup_error = cudaFree(*pointer);
            *pointer = nullptr;
            return cleanup_error == cudaSuccess ? cudaErrorMemoryAllocation
                                                : cleanup_error;
        }
        value.live_allocations[value.live_allocation_count++] = *pointer;
        value.peak_live_allocations =
            std::max(value.peak_live_allocations, value.live_allocation_count);
    }
    return error;
}

cudaError_t release(void* pointer) noexcept {
    if (pointer == nullptr) return cudaSuccess;
    const cudaError_t error = cudaFree(pointer);
    if (error == cudaSuccess) {
        auto& value = state();
        const auto begin = value.live_allocations.begin();
        const auto end = begin + value.live_allocation_count;
        const auto found = std::find(begin, end, pointer);
        if (found != end) {
            *found = value.live_allocations[--value.live_allocation_count];
            value.live_allocations[value.live_allocation_count] = nullptr;
        }
    }
    return error;
}

cudaError_t copy(void* destination, const void* source, std::size_t bytes,
                 cudaMemcpyKind kind) noexcept {
    if (should_fail(Operation::kCopy)) return cudaErrorInvalidValue;
    return cudaMemcpy(destination, source, bytes, kind);
}

cudaError_t before_launch() noexcept {
    if (should_fail(Operation::kLaunch)) return cudaErrorLaunchFailure;
    return cudaSuccess;
}

cudaError_t synchronize() noexcept {
    if (should_fail(Operation::kSynchronize)) return cudaErrorLaunchFailure;
    return cudaDeviceSynchronize();
}

}  // namespace resonantia_test::cuda_failure
