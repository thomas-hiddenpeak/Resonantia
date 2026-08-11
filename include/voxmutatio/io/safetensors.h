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

/// Identity of the source file descriptor used to create an anonymous
/// safetensors snapshot.
struct SafetensorsSourceIdentity {
    std::uintmax_t device = 0;
    std::uintmax_t inode = 0;
    std::uintmax_t mode = 0;
    std::uintmax_t size = 0;
    std::int64_t modified_seconds = 0;
    std::int64_t modified_nanoseconds = 0;
    std::int64_t changed_seconds = 0;
    std::int64_t changed_nanoseconds = 0;

    bool operator==(const SafetensorsSourceIdentity&) const noexcept = default;
};

/// Safetensors loader backed by an owned memory mapping.
class SafetensorsLoader {
public:
    SafetensorsLoader() = default;
    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;
    SafetensorsLoader(SafetensorsLoader&& other) noexcept;
    SafetensorsLoader& operator=(SafetensorsLoader&& other) noexcept;

    /// Load and parse a safetensors file
    bool load(const std::string& path);

    /// Load a stable, read-only anonymous snapshot of a safetensors file.
    ///
    /// The source must remain the same regular file for the complete copy. The
    /// snapshot remains valid independently of later source-file truncation or
    /// replacement, until this loader is destroyed or loaded again. max_bytes
    /// is a required caller-owned resource bound checked before allocation.
    bool load_snapshot(const std::string& path, std::size_t max_bytes);

    /// Get the opened-fd identity captured by a successful load_snapshot().
    /// Returns nullptr after load(), a failed load, or a move from this object.
    [[nodiscard]] const SafetensorsSourceIdentity* source_identity() const noexcept {
        return source_identity_valid_ ? &source_identity_ : nullptr;
    }

    /// Get tensor metadata by name
    const Tensor* get_tensor(const std::string& name) const;

    /// Get all tensor names
    std::vector<std::string> tensor_names() const;

    /// Get raw data pointer for a tensor (const, mapping-backed)
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
    SafetensorsSourceIdentity source_identity_;
    bool source_identity_valid_ = false;
};

}  // namespace voxmutatio::io
