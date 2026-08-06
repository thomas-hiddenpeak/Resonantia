// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/core/device.h"

#include <cuda_runtime.h>

#include <optional>
#include <string>

namespace voxmutatio {

std::optional<std::string> Device::init(const std::string& device_str,
                                        int device_id) {
    if (device_str == "cpu") {
        is_cuda_ = false;
        name_ = "CPU";
        cc_ = 0;
        total_memory_ = 0;
        return std::nullopt;
    }

    // Query CUDA device count
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return "No CUDA devices available: " +
               std::string(cudaGetErrorString(err));
    }

    if (device_id < 0 || device_id >= device_count) {
        return "CUDA device ID " + std::to_string(device_id) +
               " out of range [0, " + std::to_string(device_count - 1) + "]";
    }

    device_id_ = device_id;
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return "Failed to set CUDA device: " +
               std::string(cudaGetErrorString(err));
    }

    // Get device properties
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        return "Failed to get CUDA device properties: " +
               std::string(cudaGetErrorString(err));
    }

    is_cuda_ = true;
    name_ = props.name;
    cc_ = props.major * 10 + props.minor;
    total_memory_ = props.totalGlobalMem;

    return std::nullopt;
}

}  // namespace voxmutatio
