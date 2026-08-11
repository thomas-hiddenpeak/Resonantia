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
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;  // in bytes
    std::size_t data_offset = 0;   // byte offset in file
    std::size_t data_nbytes = 0;
};

/// Safetensors file loader (zero-copy mmap)
class SafetensorsLoader {
public:
    SafetensorsLoader() = default;
    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;
    SafetensorsLoader(SafetensorsLoader&& other) noexcept;
    SafetensorsLoader& operator=(SafetensorsLoader&& other) noexcept;

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
    void reset() noexcept;

    void* data_ = nullptr;         // mmap pointer
    std::size_t data_size_ = 0;
    std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace voxmutatio::io
