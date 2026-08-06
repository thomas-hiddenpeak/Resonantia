#pragma once

#include <memory>
#include <optional>
#include <string>

namespace voxmutatio {

/// CUDA device discovery and management
class Device {
public:
    /// Initialize CUDA device
    std::optional<std::string> init(const std::string& device_str,
                                    int device_id = 0);

    /// Check if CUDA is available
    [[nodiscard]] bool is_cuda() const noexcept { return is_cuda_; }

    /// Get device name
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    /// Get compute capability
    [[nodiscard]] int compute_capability() const noexcept { return cc_; }

    /// Get total GPU memory (bytes)
    [[nodiscard]] std::size_t total_memory() const noexcept {
        return total_memory_;
    }

private:
    bool is_cuda_ = false;
    std::string name_;
    int cc_ = 0;
    std::size_t total_memory_ = 0;
    int device_id_ = 0;
};

}  // namespace voxmutatio
