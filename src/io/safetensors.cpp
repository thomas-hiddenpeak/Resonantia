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

// Skip whitespace
inline void skip_ws(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) ++p;
}

// Parse a JSON string (assumes *p == '"'). Returns content, advances past closing quote.
std::string parse_string(const char*& p, const char* end) {
    std::string out;
    if (p >= end || *p != '"') return out;
    ++p;  // opening quote
    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) ++p;  // skip escape
        out.push_back(*p);
        ++p;
    }
    if (p < end && *p == '"') ++p;  // closing quote
    return out;
}

// Parse a JSON integer, advancing p.
long long parse_int(const char*& p, const char* end) {
    skip_ws(p, end);
    bool neg = false;
    if (p < end && *p == '-') { neg = true; ++p; }
    long long v = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        v = v * 10 + (*p - '0');
        ++p;
    }
    return neg ? -v : v;
}

// Skip a JSON value (object, array, string, or number) at *p.
void skip_value(const char*& p, const char* end) {
    skip_ws(p, end);
    if (p >= end) return;
    if (*p == '{') {
        int depth = 0;
        do {
            if (*p == '{') ++depth;
            else if (*p == '}') --depth;
            else if (*p == '"') { parse_string(p, end); continue; }
            ++p;
        } while (p < end && depth > 0);
    } else if (*p == '[') {
        int depth = 0;
        do {
            if (*p == '[') ++depth;
            else if (*p == ']') --depth;
            else if (*p == '"') { parse_string(p, end); continue; }
            ++p;
        } while (p < end && depth > 0);
    } else if (*p == '"') {
        parse_string(p, end);
    } else {
        while (p < end && *p != ',' && *p != '}' && *p != ']') ++p;
    }
}

// Parse safetensors header (JSON format)
// Header format: 8-byte little-endian uint64 length + JSON body
bool parse_header(const uint8_t* data, std::size_t data_size,
                  std::size_t& header_len,
                  std::unordered_map<std::string, Tensor>& tensors) {
    if (data_size < 8) {
        return false;
    }

    uint64_t header_length = 0;
    std::memcpy(&header_length, data, 8);

    if (header_length == 0 || header_length + 8 > data_size) {
        return false;
    }

    const char* p = reinterpret_cast<const char*>(data + 8);
    const char* end = p + header_length;
    const std::size_t data_section_start = header_length + 8;

    skip_ws(p, end);
    if (p >= end || *p != '{') return false;
    ++p;  // opening '{'

    while (p < end) {
        skip_ws(p, end);
        if (p >= end || *p == '}') break;
        if (*p == ',') { ++p; continue; }
        if (*p != '"') break;

        std::string name = parse_string(p, end);
        skip_ws(p, end);
        if (p < end && *p == ':') ++p;
        skip_ws(p, end);

        // Skip the __metadata__ entry (maps to a string dict, not a tensor)
        if (name == "__metadata__") {
            skip_value(p, end);
            continue;
        }

        if (p >= end || *p != '{') break;
        ++p;  // tensor object '{'

        Tensor t;
        t.name = name;
        long long start_offset = 0, end_offset = 0;

        while (p < end) {
            skip_ws(p, end);
            if (p >= end || *p == '}') break;
            if (*p == ',') { ++p; continue; }
            if (*p != '"') break;

            std::string key = parse_string(p, end);
            skip_ws(p, end);
            if (p < end && *p == ':') ++p;
            skip_ws(p, end);

            if (key == "dtype") {
                parse_string(p, end);
            } else if (key == "shape") {
                if (p < end && *p == '[') {
                    ++p;
                    while (p < end && *p != ']') {
                        skip_ws(p, end);
                        if (*p == ']') break;
                        long long dim = parse_int(p, end);
                        t.shape.push_back(static_cast<int64_t>(dim));
                        skip_ws(p, end);
                        if (p < end && *p == ',') ++p;
                    }
                    if (p < end && *p == ']') ++p;
                }
            } else if (key == "data_offsets") {
                if (p < end && *p == '[') {
                    ++p;
                    start_offset = parse_int(p, end);
                    skip_ws(p, end);
                    if (p < end && *p == ',') ++p;
                    end_offset = parse_int(p, end);
                    skip_ws(p, end);
                    if (p < end && *p == ']') ++p;
                }
            } else {
                skip_value(p, end);
            }
        }
        if (p < end && *p == '}') ++p;  // close tensor object

        t.data_offset = data_section_start + static_cast<std::size_t>(start_offset);
        t.data_nbytes = static_cast<std::size_t>(end_offset - start_offset);

        // Row-major strides (float32)
        if (!t.shape.empty()) {
            int64_t stride = 4;
            for (int64_t i = static_cast<int64_t>(t.shape.size()) - 1; i >= 0; --i) {
                t.strides.push_back(stride);
                stride *= t.shape[i];
            }
            std::reverse(t.strides.begin(), t.strides.end());
        }

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
