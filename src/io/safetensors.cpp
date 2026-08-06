// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/io/safetensors.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace voxmutatio::io {

namespace {

// Parse safetensors header (JSON format)
// Header format: 8-byte little-endian uint64 length + JSON body
bool parse_header(const uint8_t* data, std::size_t data_size,
                  std::size_t& header_len,
                  std::unordered_map<std::string, Tensor>& tensors) {
    if (data_size < 8) {
        return false;
    }

    // Read header length (little-endian uint64)
    uint64_t header_length = 0;
    std::memcpy(&header_length, data, 8);
    
    if (header_length == 0 || header_length + 8 > data_size) {
        return false;
    }

    // Parse JSON header (simple manual parser for safetensors format)
    const char* json = reinterpret_cast<const char*>(data + 8);
    
    // Skip opening '{'
    const char* p = json + 1;
    
    while (*p && *p != '}') {
        // Skip whitespace and commas
        while (*p && (*p == ' ' || *p == ',' || *p == '\n')) ++p;
        if (*p == '}') break;

        // Parse tensor name (quoted string)
        if (*p != '"') break;
        ++p; // skip opening quote
        
        const char* name_start = p;
        while (*p && *p != '"') ++p;
        std::string name(name_start, p - name_start);
        ++p; // skip closing quote

        // Skip ': '
        while (*p && (*p == ' ' || *p == ':')) ++p;

        // Parse object {dtype, shape, data_offsets}
        if (*p != '{') break;
        ++p; // skip opening '{'

        Tensor t;
        t.name = name;
        
        while (*p && *p != '}') {
            // Skip whitespace and commas
            while (*p && (*p == ' ' || *p == ',')) ++p;
            if (*p == '}') break;

            // Parse key
            if (*p != '"') break;
            ++p;
            const char* key_start = p;
            while (*p && *p != '"') ++p;
            std::string key(key_start, p - key_start);
            ++p; // skip closing quote

            // Skip ': '
            while (*p && (*p == ' ' || *p == ':')) ++p;

            if (key == "dtype") {
                // Skip dtype value (we don't need it for now)
                if (*p == '"') {
                    ++p;
                    while (*p && *p != '"') ++p;
                    ++p;
                }
            } else if (key == "shape") {
                // Parse [dim1, dim2, ...]
                if (*p != '[') break;
                ++p;
                while (*p && *p != ']') {
                    while (*p && *p != ',') {
                        if (*p >= '0' && *p <= '9') {
                            long long val = 0;
                            while (*p >= '0' && *p <= '9') {
                                val = val * 10 + (*p - '0');
                                ++p;
                            }
                            t.shape.push_back(static_cast<int64_t>(val));
                            continue;
                        }
                        ++p;
                    }
                    if (*p == ',') ++p;
                }
                if (*p == ']') ++p;
            } else if (key == "data_offsets") {
                // Parse [start_offset, end_offset]
                // Note: offsets are relative to the start of the data section (after header + 8 bytes)
                if (*p != '[') break;
                ++p;
                
                // Skip whitespace
                while (*p && (*p == ' ' || *p == '\n')) ++p;
                
                // Read start offset
                long long start_offset = 0;
                {
                    const char* num_start = p;
                    while (*p >= '0' && *p <= '9') ++p;
                    if (num_start != p) {
                        start_offset = std::stoll(std::string(num_start, p));
                    }
                }
                
                // Skip comma and whitespace
                while (*p && (*p == ' ' || *p == ',' || *p == '\n')) ++p;
                
                // Read end offset
                long long end_offset = 0;
                {
                    const char* num_start = p;
                    while (*p >= '0' && *p <= '9') ++p;
                    if (num_start != p) {
                        end_offset = std::stoll(std::string(num_start, p));
                    }
                }
                
                // Data section starts after header (8 bytes length + header_length bytes JSON)
                std::size_t data_section_start = header_length + 8;
                
                t.data_offset = data_section_start + static_cast<std::size_t>(start_offset);
                t.data_nbytes = static_cast<std::size_t>(end_offset - start_offset);
                
                // Calculate strides (row-major, float32)
                if (!t.shape.empty()) {
                    int64_t stride = 4; // float32 = 4 bytes
                    for (int64_t i = t.shape.size() - 1; i >= 0; --i) {
                        t.strides.push_back(stride);
                        stride *= t.shape[i];
                    }
                    std::reverse(t.strides.begin(), t.strides.end());
                }
                
                if (*p == ',') ++p;
            }
        }
        if (*p == '}') ++p;

        tensors[name] = t;
    }

    header_len = header_length + 8;
    return true;
}

}  // namespace

bool SafetensorsLoader::load(const std::string& path) {
    // Close existing file if any
    if (data_) {
        munmap(data_, data_size_);
        data_ = nullptr;
        data_size_ = 0;
        tensors_.clear();
    }

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return false;
    }

    data_size_ = static_cast<std::size_t>(st.st_size);

    data_ = mmap(nullptr, data_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        data_size_ = 0;
        return false;
    }

    // Advise kernel about sequential access pattern
    madvise(data_, data_size_, MADV_SEQUENTIAL);

    std::size_t header_len = 0;
    if (!parse_header(static_cast<const uint8_t*>(data_), data_size_,
                      header_len, tensors_)) {
        munmap(data_, data_size_);
        data_ = nullptr;
        data_size_ = 0;
        return false;
    }

    return true;
}

const Tensor* SafetensorsLoader::get_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it != tensors_.end() ? &it->second : nullptr;
}

std::vector<std::string> SafetensorsLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& [name, _] : tensors_) {
        names.push_back(name);
    }
    return names;
}

const uint8_t* SafetensorsLoader::data(const std::string& name) const {
    const Tensor* t = get_tensor(name);
    if (!t || !data_) {
        return nullptr;
    }
    return static_cast<const uint8_t*>(data_) + t->data_offset;
}

SafetensorsLoader::~SafetensorsLoader() {
    if (data_) {
        munmap(data_, data_size_);
        data_ = nullptr;
        data_size_ = 0;
    }
}

}  // namespace voxmutatio::io
