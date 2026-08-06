#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace voxmutatio::io {

/// A tensor stored in safetensors format
struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;  // in bytes
    std::size_t data_offset;       // byte offset in file
    std::size_t data_nbytes;
};

/// Safetensors file loader (zero-copy mmap)
class SafetensorsLoader {
public:
    /// Load and parse a safetensors file
    bool load(const std::string& path);

    /// Get tensor metadata by name
    const Tensor* get_tensor(const std::string& name) const;

    /// Get all tensor names
    std::vector<std::string> tensor_names() const;

    /// Get raw data pointer for a tensor (const, mmap-backed)
    const uint8_t* data(const std::string& name) const;

    /// Check if file is loaded
    [[nodiscard]] bool valid() const noexcept { return data_ != nullptr; }

    /// Get total data size (bytes)
    [[nodiscard]] std::size_t data_size() const noexcept { return data_size_; }

    ~SafetensorsLoader();

private:
    void* data_ = nullptr;         // mmap pointer
    std::size_t data_size_ = 0;
    std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace voxmutatio::io
