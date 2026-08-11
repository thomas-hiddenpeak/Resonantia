// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/io/safetensors.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace voxmutatio::io {

namespace {

constexpr std::uint64_t kMaxHeaderSize = 100'000'000;
constexpr std::size_t kMaxTensorRank = 32;
constexpr int kMaxJsonDepth = 64;
constexpr std::size_t kSnapshotReadChunk = 1U << 20;

class ScopedFd {
public:
    explicit ScopedFd(int fd = -1) noexcept : fd_(fd) {}
    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;
    ~ScopedFd() {
        if (fd_ >= 0) ::close(fd_);
    }

    [[nodiscard]] int get() const noexcept { return fd_; }
    [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }

private:
    int fd_ = -1;
};

class ScopedMapping {
public:
    ScopedMapping(void* data, std::size_t size) noexcept
        : data_(data), size_(size) {}
    ScopedMapping(const ScopedMapping&) = delete;
    ScopedMapping& operator=(const ScopedMapping&) = delete;
    ~ScopedMapping() {
        if (data_) ::munmap(data_, size_);
    }

    [[nodiscard]] void* get() const noexcept { return data_; }
    [[nodiscard]] void* release() noexcept {
        return std::exchange(data_, nullptr);
    }

private:
    void* data_ = nullptr;
    std::size_t size_ = 0;
};

int open_readonly(const std::string& path) noexcept {
    int fd = -1;
    do {
        // O_NONBLOCK is ignored for regular files, but prevents a path that
        // names a no-writer FIFO from blocking before fstat rejects it.
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    } while (fd < 0 && errno == EINTR);
    return fd;
}

bool read_file_status(int fd, struct stat& status) noexcept {
    int result = -1;
    do {
        result = ::fstat(fd, &status);
    } while (result < 0 && errno == EINTR);
    return result == 0;
}

bool checked_regular_file_size(const struct stat& status,
                               std::size_t& size) noexcept {
    if (!S_ISREG(status.st_mode) ||
        status.st_size < static_cast<off_t>(sizeof(std::uint64_t)) ||
        static_cast<std::uintmax_t>(status.st_size) >
            std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    size = static_cast<std::size_t>(status.st_size);
    return true;
}

bool same_file_status(const struct stat& before,
                      const struct stat& after) noexcept {
    return S_ISREG(after.st_mode) && before.st_dev == after.st_dev &&
           before.st_ino == after.st_ino && before.st_mode == after.st_mode &&
           before.st_size == after.st_size &&
           before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
           before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
           before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
           before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
}

SafetensorsSourceIdentity source_identity_from(
    const struct stat& status) noexcept {
    SafetensorsSourceIdentity identity;
    identity.device = static_cast<std::uintmax_t>(status.st_dev);
    identity.inode = static_cast<std::uintmax_t>(status.st_ino);
    identity.mode = static_cast<std::uintmax_t>(status.st_mode);
    identity.size = static_cast<std::uintmax_t>(status.st_size);
    identity.modified_seconds =
        static_cast<std::int64_t>(status.st_mtim.tv_sec);
    identity.modified_nanoseconds =
        static_cast<std::int64_t>(status.st_mtim.tv_nsec);
    identity.changed_seconds =
        static_cast<std::int64_t>(status.st_ctim.tv_sec);
    identity.changed_nanoseconds =
        static_cast<std::int64_t>(status.st_ctim.tv_nsec);
    return identity;
}

bool checked_add(std::size_t a, std::size_t b, std::size_t& result) {
    if (b > std::numeric_limits<std::size_t>::max() - a) return false;
    result = a + b;
    return true;
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t& result) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
    result = a * b;
    return true;
}

std::uint64_t read_u64_le(const std::uint8_t* data) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(data[i]) << (i * 8);
    }
    return value;
}

void skip_ws(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) ++p;
}

bool parse_hex4(const char*& p, const char* end, std::uint32_t& value) {
    if (end - p < 4) return false;
    value = 0;
    for (int i = 0; i < 4; ++i) {
        const unsigned char c = static_cast<unsigned char>(*p++);
        value <<= 4;
        if (c >= '0' && c <= '9') value |= c - '0';
        else if (c >= 'a' && c <= 'f') value |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') value |= c - 'A' + 10;
        else return false;
    }
    return true;
}

bool append_utf8(std::uint32_t codepoint, std::string& out) {
    if (codepoint <= 0x7f) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        if (codepoint >= 0xd800 && codepoint <= 0xdfff) return false;
        out.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0x10ffff) {
        out.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        return false;
    }
    return true;
}

bool parse_string(const char*& p, const char* end, std::string& out) {
    out.clear();
    if (p >= end || *p != '"') return false;
    ++p;
    while (p < end) {
        const unsigned char c = static_cast<unsigned char>(*p++);
        if (c == '"') return true;
        if (c < 0x20) return false;
        if (c != '\\') {
            out.push_back(static_cast<char>(c));
            continue;
        }
        if (p >= end) return false;
        const char escaped = *p++;
        switch (escaped) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                std::uint32_t codepoint = 0;
                if (!parse_hex4(p, end, codepoint)) return false;
                if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                    if (end - p < 2 || p[0] != '\\' || p[1] != 'u') return false;
                    p += 2;
                    std::uint32_t low = 0;
                    if (!parse_hex4(p, end, low) || low < 0xdc00 || low > 0xdfff) return false;
                    codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                    return false;
                }
                if (!append_utf8(codepoint, out)) return false;
                break;
            }
            default: return false;
        }
    }
    return false;
}

bool parse_uint64(const char*& p, const char* end, std::uint64_t& value) {
    skip_ws(p, end);
    if (p >= end || *p < '0' || *p > '9') return false;
    if (*p == '0' && p + 1 < end && p[1] >= '0' && p[1] <= '9') return false;
    value = 0;
    do {
        const unsigned digit = static_cast<unsigned>(*p - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) return false;
        value = value * 10 + digit;
        ++p;
    } while (p < end && *p >= '0' && *p <= '9');
    return true;
}

bool skip_value(const char*& p, const char* end, int depth);

bool skip_array(const char*& p, const char* end, int depth) {
    if (depth > kMaxJsonDepth || p >= end || *p != '[') return false;
    ++p;
    skip_ws(p, end);
    if (p < end && *p == ']') { ++p; return true; }
    while (p < end) {
        if (!skip_value(p, end, depth + 1)) return false;
        skip_ws(p, end);
        if (p < end && *p == ']') { ++p; return true; }
        if (p >= end || *p != ',') return false;
        ++p;
        skip_ws(p, end);
    }
    return false;
}

bool skip_object(const char*& p, const char* end, int depth) {
    if (depth > kMaxJsonDepth || p >= end || *p != '{') return false;
    ++p;
    skip_ws(p, end);
    if (p < end && *p == '}') { ++p; return true; }
    std::unordered_set<std::string> keys;
    while (p < end) {
        std::string key;
        if (!parse_string(p, end, key) || !keys.insert(key).second) return false;
        skip_ws(p, end);
        if (p >= end || *p++ != ':') return false;
        if (!skip_value(p, end, depth + 1)) return false;
        skip_ws(p, end);
        if (p < end && *p == '}') { ++p; return true; }
        if (p >= end || *p != ',') return false;
        ++p;
        skip_ws(p, end);
    }
    return false;
}

bool skip_number(const char*& p, const char* end) {
    if (p < end && *p == '-') ++p;
    if (p >= end) return false;
    if (*p == '0') {
        ++p;
    } else if (*p >= '1' && *p <= '9') {
        do { ++p; } while (p < end && *p >= '0' && *p <= '9');
    } else {
        return false;
    }
    if (p < end && *p == '.') {
        ++p;
        if (p >= end || *p < '0' || *p > '9') return false;
        do { ++p; } while (p < end && *p >= '0' && *p <= '9');
    }
    if (p < end && (*p == 'e' || *p == 'E')) {
        ++p;
        if (p < end && (*p == '+' || *p == '-')) ++p;
        if (p >= end || *p < '0' || *p > '9') return false;
        do { ++p; } while (p < end && *p >= '0' && *p <= '9');
    }
    return true;
}

bool skip_value(const char*& p, const char* end, int depth) {
    if (depth > kMaxJsonDepth) return false;
    skip_ws(p, end);
    if (p >= end) return false;
    if (*p == '{') return skip_object(p, end, depth);
    if (*p == '[') return skip_array(p, end, depth);
    if (*p == '"') { std::string ignored; return parse_string(p, end, ignored); }
    if (*p == '-' || (*p >= '0' && *p <= '9')) return skip_number(p, end);
    constexpr const char* literals[] = {"true", "false", "null"};
    for (const char* literal : literals) {
        const std::size_t n = std::strlen(literal);
        if (static_cast<std::size_t>(end - p) >= n && std::memcmp(p, literal, n) == 0) {
            p += n;
            return true;
        }
    }
    return false;
}

bool parse_shape(const char*& p, const char* end, std::vector<int64_t>& shape) {
    if (p >= end || *p != '[') return false;
    ++p;
    skip_ws(p, end);
    if (p < end && *p == ']') { ++p; return true; }
    while (p < end) {
        if (shape.size() >= kMaxTensorRank) return false;
        std::uint64_t dim = 0;
        if (!parse_uint64(p, end, dim) || dim > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
            return false;
        }
        shape.push_back(static_cast<int64_t>(dim));
        skip_ws(p, end);
        if (p < end && *p == ']') { ++p; return true; }
        if (p >= end || *p != ',') return false;
        ++p;
        skip_ws(p, end);
    }
    return false;
}

bool parse_offsets(const char*& p, const char* end,
                   std::uint64_t& start, std::uint64_t& finish) {
    if (p >= end || *p != '[') return false;
    ++p;
    if (!parse_uint64(p, end, start)) return false;
    skip_ws(p, end);
    if (p >= end || *p++ != ',') return false;
    if (!parse_uint64(p, end, finish)) return false;
    skip_ws(p, end);
    if (p >= end || *p++ != ']') return false;
    return true;
}

bool parse_tensor(const char*& p, const char* end, const std::string& name,
                  std::size_t data_start, std::size_t data_section_size,
                  Tensor& tensor, std::pair<std::size_t, std::size_t>& range) {
    if (p >= end || *p != '{') return false;
    ++p;
    tensor.name = name;
    bool has_dtype = false;
    bool has_shape = false;
    bool has_offsets = false;
    std::uint64_t start = 0;
    std::uint64_t finish = 0;
    std::unordered_set<std::string> keys;

    skip_ws(p, end);
    if (p < end && *p == '}') return false;
    while (p < end) {
        std::string key;
        if (!parse_string(p, end, key) || !keys.insert(key).second) return false;
        skip_ws(p, end);
        if (p >= end || *p++ != ':') return false;
        skip_ws(p, end);
        if (key == "dtype") {
            if (!parse_string(p, end, tensor.dtype)) return false;
            has_dtype = true;
        } else if (key == "shape") {
            if (!parse_shape(p, end, tensor.shape)) return false;
            has_shape = true;
        } else if (key == "data_offsets") {
            if (!parse_offsets(p, end, start, finish)) return false;
            has_offsets = true;
        } else if (!skip_value(p, end, 1)) {
            return false;
        }
        skip_ws(p, end);
        if (p < end && *p == '}') { ++p; break; }
        if (p >= end || *p != ',') return false;
        ++p;
        skip_ws(p, end);
    }

    // Every current consumer exposes the bytes as float*, so accepting another
    // dtype would silently reinterpret the tensor with the wrong element size.
    if (!has_dtype || !has_shape || !has_offsets || tensor.dtype != "F32") return false;
    if (start > finish || finish > data_section_size) return false;
    if (start > std::numeric_limits<std::size_t>::max() ||
        finish > std::numeric_limits<std::size_t>::max()) return false;

    std::size_t elements = 1;
    tensor.strides.resize(tensor.shape.size());
    std::size_t stride = sizeof(float);
    for (std::size_t i = tensor.shape.size(); i-- > 0;) {
        if (stride > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) return false;
        tensor.strides[i] = static_cast<int64_t>(stride);
        const auto dim = static_cast<std::size_t>(tensor.shape[i]);
        if (!checked_mul(stride, dim, stride)) return false;
        if (!checked_mul(elements, dim, elements)) return false;
    }
    std::size_t expected_bytes = 0;
    if (!checked_mul(elements, sizeof(float), expected_bytes)) return false;
    const std::size_t start_size = static_cast<std::size_t>(start);
    const std::size_t finish_size = static_cast<std::size_t>(finish);
    if (finish_size - start_size != expected_bytes) return false;
    if (!checked_add(data_start, start_size, tensor.data_offset)) return false;
    tensor.data_nbytes = expected_bytes;
    range = {start_size, finish_size};
    return true;
}

// Parse safetensors header: 8-byte little-endian uint64 length + JSON body.
bool parse_header(const std::uint8_t* data, std::size_t file_size,
                  std::unordered_map<std::string, Tensor>& tensors) {
    if (file_size < sizeof(std::uint64_t)) return false;
    const std::uint64_t header_length = read_u64_le(data);
    if (header_length == 0 || header_length > kMaxHeaderSize ||
        header_length > file_size - sizeof(std::uint64_t) ||
        header_length > std::numeric_limits<std::size_t>::max()) {
        return false;
    }

    const std::size_t header_size = static_cast<std::size_t>(header_length);
    const std::size_t data_start = sizeof(std::uint64_t) + header_size;
    const std::size_t data_section_size = file_size - data_start;
    const char* p = reinterpret_cast<const char*>(data + sizeof(std::uint64_t));
    const char* end = p + header_size;
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    bool has_metadata = false;

    skip_ws(p, end);
    if (p >= end || *p++ != '{') return false;
    skip_ws(p, end);
    if (p < end && *p == '}') {
        ++p;
    } else {
        while (p < end) {
            std::string name;
            if (!parse_string(p, end, name)) return false;
            skip_ws(p, end);
            if (p >= end || *p++ != ':') return false;
            skip_ws(p, end);
            if (name == "__metadata__") {
                if (has_metadata || !skip_object(p, end, 1)) return false;
                has_metadata = true;
            } else {
                Tensor tensor;
                std::pair<std::size_t, std::size_t> range;
                if (!parse_tensor(p, end, name, data_start, data_section_size,
                                  tensor, range)) return false;
                if (!tensors.emplace(name, std::move(tensor)).second) return false;
                ranges.push_back(range);
            }
            skip_ws(p, end);
            if (p < end && *p == '}') { ++p; break; }
            if (p >= end || *p != ',') return false;
            ++p;
            skip_ws(p, end);
        }
    }
    skip_ws(p, end);
    if (p != end) return false;

    std::sort(ranges.begin(), ranges.end());
    std::size_t covered = 0;
    for (const auto& range : ranges) {
        // Safetensors payloads are contiguous: gaps, overlap, and unindexed
        // trailing bytes indicate a malformed or ambiguous file.
        if (range.first != covered || range.second < range.first) return false;
        covered = range.second;
    }
    return covered == data_section_size;
}

}  // namespace

SafetensorsLoader::SafetensorsLoader(SafetensorsLoader&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      data_size_(std::exchange(other.data_size_, 0)),
      tensors_(std::move(other.tensors_)),
      source_identity_(other.source_identity_),
      source_identity_valid_(
          std::exchange(other.source_identity_valid_, false)) {}

SafetensorsLoader& SafetensorsLoader::operator=(SafetensorsLoader&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = std::exchange(other.data_, nullptr);
        data_size_ = std::exchange(other.data_size_, 0);
        tensors_ = std::move(other.tensors_);
        source_identity_ = other.source_identity_;
        source_identity_valid_ =
            std::exchange(other.source_identity_valid_, false);
    }
    return *this;
}

void SafetensorsLoader::reset() noexcept {
    if (data_) munmap(data_, data_size_);
    data_ = nullptr;
    data_size_ = 0;
    tensors_.clear();
    source_identity_ = {};
    source_identity_valid_ = false;
}

bool SafetensorsLoader::load(const std::string& path) try {
    reset();

    ScopedFd fd(open_readonly(path));
    if (!fd.valid()) return false;

    struct stat st {};
    std::size_t staged_size = 0;
    if (!read_file_status(fd.get(), st) ||
        !checked_regular_file_size(st, staged_size)) {
        return false;
    }

    void* mapped = ::mmap(nullptr, staged_size, PROT_READ, MAP_PRIVATE,
                          fd.get(), 0);
    if (mapped == MAP_FAILED) return false;
    ScopedMapping staged_mapping(mapped, staged_size);

    std::unordered_map<std::string, Tensor> parsed;
    if (!parse_header(static_cast<const std::uint8_t*>(staged_mapping.get()),
                      staged_size, parsed)) return false;

    tensors_ = std::move(parsed);
    data_size_ = staged_size;
    data_ = staged_mapping.release();
    ::madvise(data_, data_size_, MADV_SEQUENTIAL);
    return true;
} catch (const std::bad_alloc&) {
    reset();
    return false;
} catch (const std::length_error&) {
    reset();
    return false;
}

bool SafetensorsLoader::load_snapshot(const std::string& path,
                                      std::size_t max_bytes) try {
    reset();

    ScopedFd fd(open_readonly(path));
    if (!fd.valid()) return false;

    struct stat before {};
    std::size_t staged_size = 0;
    if (!read_file_status(fd.get(), before) ||
        !checked_regular_file_size(before, staged_size) ||
        staged_size > max_bytes) {
        return false;
    }

    void* mapped = ::mmap(nullptr, staged_size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped == MAP_FAILED) return false;
    ScopedMapping staged_mapping(mapped, staged_size);

    auto* destination = static_cast<std::uint8_t*>(staged_mapping.get());
    std::size_t copied = 0;
    while (copied < staged_size) {
        const std::size_t requested =
            std::min(kSnapshotReadChunk, staged_size - copied);
        ssize_t read_size = -1;
        do {
            read_size = ::pread(fd.get(), destination + copied, requested,
                                static_cast<off_t>(copied));
        } while (read_size < 0 && errno == EINTR);
        if (read_size <= 0) return false;
        copied += static_cast<std::size_t>(read_size);
    }

    struct stat after {};
    if (!read_file_status(fd.get(), after) ||
        !same_file_status(before, after) ||
        ::mprotect(staged_mapping.get(), staged_size, PROT_READ) != 0) {
        return false;
    }
    const SafetensorsSourceIdentity staged_identity =
        source_identity_from(after);

    std::unordered_map<std::string, Tensor> parsed;
    if (!parse_header(static_cast<const std::uint8_t*>(staged_mapping.get()),
                      staged_size, parsed)) return false;

    tensors_ = std::move(parsed);
    source_identity_ = staged_identity;
    source_identity_valid_ = true;
    data_size_ = staged_size;
    data_ = staged_mapping.release();
    return true;
} catch (const std::bad_alloc&) {
    reset();
    return false;
} catch (const std::length_error&) {
    reset();
    return false;
}

const Tensor* SafetensorsLoader::get_tensor(const std::string& name) const {
    const auto it = tensors_.find(name);
    return it != tensors_.end() ? &it->second : nullptr;
}

std::vector<std::string> SafetensorsLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& [name, _] : tensors_) names.push_back(name);
    return names;
}

const std::uint8_t* SafetensorsLoader::data(const std::string& name) const {
    const Tensor* tensor = get_tensor(name);
    if (!tensor || !data_ || tensor->data_offset > data_size_ ||
        tensor->data_nbytes > data_size_ - tensor->data_offset) {
        return nullptr;
    }
    return static_cast<const std::uint8_t*>(data_) + tensor->data_offset;
}

SafetensorsLoader::~SafetensorsLoader() {
    reset();
}

}  // namespace voxmutatio::io
