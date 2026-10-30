// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "bounded_audio_io.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <new>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace voxmutatio::io::detail {

namespace {

constexpr std::uint16_t kPcm = 1;
constexpr std::uint16_t kIeeeFloat = 3;
constexpr std::uint16_t kMaxChannels = 256;
constexpr std::size_t kDecodeBufferBytes = 64U * 1024U;
constexpr std::size_t kIoChunkBytes = 1U * 1024U * 1024U;
constexpr std::size_t kPcmWriteBufferBytes = 8192;
constexpr unsigned kTemporaryCreateAttempts = 128;
constexpr std::uint64_t kWavHeaderBytes = 44;

std::atomic<std::uint64_t> g_temporary_sequence{0};

struct FileIdentity {
    dev_t device = 0;
    ino_t inode = 0;
    off_t size = 0;
    mode_t mode = 0;
    nlink_t links = 0;
    timespec mtime{};
    timespec ctime{};
};

bool checked_add(std::uint64_t a, std::uint64_t b, std::uint64_t& result) noexcept {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) return false;
    result = a + b;
    return true;
}

bool checked_mul(std::uint64_t a, std::uint64_t b, std::uint64_t& result) noexcept {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return false;
    result = a * b;
    return true;
}

bool same_timespec(const timespec& left, const timespec& right) noexcept {
    return left.tv_sec == right.tv_sec && left.tv_nsec == right.tv_nsec;
}

bool same_object(const FileIdentity& left, const FileIdentity& right) noexcept {
    return left.device == right.device && left.inode == right.inode;
}

bool same_identity(const FileIdentity& left, const FileIdentity& right) noexcept {
    return same_object(left, right) && left.size == right.size && left.mode == right.mode &&
           left.links == right.links && same_timespec(left.mtime, right.mtime) &&
           same_timespec(left.ctime, right.ctime);
}

bool capture_identity(int fd, FileIdentity& identity, bool require_regular = true) noexcept {
    struct stat status {};
    if (::fstat(fd, &status) != 0 || status.st_size < 0 ||
        (require_regular && !S_ISREG(status.st_mode))) {
        return false;
    }
    identity.device = status.st_dev;
    identity.inode = status.st_ino;
    identity.size = status.st_size;
    identity.mode = status.st_mode;
    identity.links = status.st_nlink;
    identity.mtime = status.st_mtim;
    identity.ctime = status.st_ctim;
    return true;
}

bool checked_file_range(std::uint64_t offset, std::size_t size) noexcept {
    std::uint64_t end = 0;
    return checked_add(offset, static_cast<std::uint64_t>(size), end) &&
           offset <= static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) &&
           end <= static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
}

bool pread_exact(int fd, void* output, std::size_t size, std::uint64_t offset) noexcept {
    if (size == 0) return true;
    if (!output || !checked_file_range(offset, size)) return false;

    auto* cursor = static_cast<std::uint8_t*>(output);
    std::size_t remaining = size;
    while (remaining > 0) {
        const std::size_t chunk = std::min(remaining, kIoChunkBytes);
        const ssize_t received = ::pread(
            fd, cursor, chunk, static_cast<off_t>(offset));
        if (received < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (received == 0) return false;
        const auto consumed = static_cast<std::size_t>(received);
        cursor += consumed;
        remaining -= consumed;
        offset += consumed;
    }
    return true;
}

bool pwrite_exact(int fd, const void* input, std::size_t size,
                  std::uint64_t offset) noexcept {
    if (size == 0) return true;
    if (!input || !checked_file_range(offset, size)) return false;

    const auto* cursor = static_cast<const std::uint8_t*>(input);
    std::size_t remaining = size;
    while (remaining > 0) {
        const std::size_t chunk = std::min(remaining, kIoChunkBytes);
        const ssize_t written = ::pwrite(
            fd, cursor, chunk, static_cast<off_t>(offset));
        if (written < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (written == 0) return false;
        const auto consumed = static_cast<std::size_t>(written);
        cursor += consumed;
        remaining -= consumed;
        offset += consumed;
    }
    return true;
}

bool fsync_retry(int fd) noexcept {
    int result = 0;
    do {
        result = ::fsync(fd);
    } while (result != 0 && errno == EINTR);
    return result == 0;
}

bool close_checked(int& fd) noexcept {
    if (fd < 0) return true;
    const int owned = std::exchange(fd, -1);
    return ::close(owned) == 0;
}

void close_ignored(int& fd) noexcept {
    if (fd < 0) return;
    const int owned = std::exchange(fd, -1);
    (void)::close(owned);
}

std::uint16_t read_u16_le(const std::uint8_t* data) noexcept {
    return static_cast<std::uint16_t>(data[0]) |
           (static_cast<std::uint16_t>(data[1]) << 8);
}

std::uint32_t read_u32_le(const std::uint8_t* data) noexcept {
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8) |
           (static_cast<std::uint32_t>(data[2]) << 16) |
           (static_cast<std::uint32_t>(data[3]) << 24);
}

void write_u16_le(std::uint8_t* output, std::uint16_t value) noexcept {
    output[0] = static_cast<std::uint8_t>(value);
    output[1] = static_cast<std::uint8_t>(value >> 8);
}

void write_u32_le(std::uint8_t* output, std::uint32_t value) noexcept {
    output[0] = static_cast<std::uint8_t>(value);
    output[1] = static_cast<std::uint8_t>(value >> 8);
    output[2] = static_cast<std::uint8_t>(value >> 16);
    output[3] = static_cast<std::uint8_t>(value >> 24);
}

std::string temporary_name(const char* prefix) {
    const auto sequence = g_temporary_sequence.fetch_add(1, std::memory_order_relaxed);
    return std::string(prefix) + std::to_string(static_cast<long long>(::getpid())) + "-" +
           std::to_string(sequence) + ".tmp";
}

std::filesystem::path normalized_directory(const std::filesystem::path& directory) {
    return directory.empty() ? std::filesystem::path(".") : directory;
}

bool path_matches_object(int directory_fd, const std::string& name,
                         const FileIdentity& identity) noexcept {
    struct stat status {};
    if (::fstatat(directory_fd, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
        return false;
    }
    return status.st_dev == identity.device && status.st_ino == identity.inode;
}

struct WavOutputLayout {
    std::uint32_t data_bytes = 0;
    std::uint32_t riff_size = 0;
    std::uint32_t byte_rate = 0;
};

bool wav_output_layout(int sample_rate, std::uint64_t samples,
                       WavOutputLayout& layout) noexcept {
    if (sample_rate <= 0) return false;
    std::uint64_t data_bytes = 0;
    std::uint64_t riff_size = 0;
    const std::uint64_t byte_rate =
        static_cast<std::uint64_t>(sample_rate) * sizeof(std::int16_t);
    if (!checked_mul(samples, sizeof(std::int16_t), data_bytes) ||
        !checked_add(36, data_bytes, riff_size) ||
        data_bytes > std::numeric_limits<std::uint32_t>::max() ||
        riff_size > std::numeric_limits<std::uint32_t>::max() ||
        byte_rate > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    layout.data_bytes = static_cast<std::uint32_t>(data_bytes);
    layout.riff_size = static_cast<std::uint32_t>(riff_size);
    layout.byte_rate = static_cast<std::uint32_t>(byte_rate);
    return true;
}

std::array<std::uint8_t, kWavHeaderBytes> make_wav_header(
        int sample_rate, const WavOutputLayout& layout, bool final_sizes) noexcept {
    std::array<std::uint8_t, kWavHeaderBytes> header{};
    std::memcpy(header.data(), "RIFF", 4);
    write_u32_le(header.data() + 4, final_sizes ? layout.riff_size : 0);
    std::memcpy(header.data() + 8, "WAVEfmt ", 8);
    write_u32_le(header.data() + 16, 16);
    write_u16_le(header.data() + 20, kPcm);
    write_u16_le(header.data() + 22, 1);
    write_u32_le(header.data() + 24, static_cast<std::uint32_t>(sample_rate));
    write_u32_le(header.data() + 28, layout.byte_rate);
    write_u16_le(header.data() + 32, sizeof(std::int16_t));
    write_u16_le(header.data() + 34, 16);
    std::memcpy(header.data() + 36, "data", 4);
    write_u32_le(header.data() + 40, final_sizes ? layout.data_bytes : 0);
    return header;
}

}  // namespace

struct WavChunkReader::State {
    ~State() { close_ignored(fd); }

    int fd = -1;
    FileIdentity identity;
    WavMetadata metadata;
    std::uint64_t data_offset = 0;
    std::uint64_t data_bytes = 0;
    std::uint16_t block_align = 0;
    std::uint16_t audio_format = 0;
};

WavChunkReader::WavChunkReader() noexcept = default;
WavChunkReader::~WavChunkReader() = default;
WavChunkReader::WavChunkReader(WavChunkReader&&) noexcept = default;
WavChunkReader& WavChunkReader::operator=(WavChunkReader&&) noexcept = default;

bool WavChunkReader::open(const std::filesystem::path& path, bool no_follow) noexcept try {
    close();
    const std::string path_text = path.string();
    if (path_text.empty() || path_text.find('\0') != std::string::npos) return false;

    auto staged = std::make_unique<State>();
    do {
        // Reject FIFOs/devices at the subsequent fstat boundary without ever
        // waiting for a writer. O_NONBLOCK does not change regular-file pread.
        staged->fd = ::open(
            path_text.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK | (no_follow ? O_NOFOLLOW : 0));
    } while (staged->fd < 0 && errno == EINTR);
    if (staged->fd < 0 || !capture_identity(staged->fd, staged->identity) ||
        staged->identity.size < 12) {
        return false;
    }
    const auto file_size = static_cast<std::uint64_t>(staged->identity.size);

    std::array<std::uint8_t, 12> riff{};
    if (!pread_exact(staged->fd, riff.data(), riff.size(), 0) ||
        std::memcmp(riff.data(), "RIFF", 4) != 0 ||
        std::memcmp(riff.data() + 8, "WAVE", 4) != 0) {
        return false;
    }
    const std::uint32_t riff_size = read_u32_le(riff.data() + 4);
    std::uint64_t riff_end = 0;
    if (riff_size < 4 || !checked_add(8, riff_size, riff_end) || riff_end > file_size) {
        return false;
    }

    bool has_fmt = false;
    bool has_data = false;
    std::uint16_t channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint32_t byte_rate = 0;
    std::uint16_t bits_per_sample = 0;
    std::uint64_t cursor = 12;
    while (cursor < riff_end) {
        std::uint64_t chunk_data = 0;
        if (!checked_add(cursor, 8, chunk_data) || chunk_data > riff_end) return false;
        std::array<std::uint8_t, 8> chunk_header{};
        if (!pread_exact(staged->fd, chunk_header.data(), chunk_header.size(), cursor)) {
            return false;
        }
        const std::uint32_t chunk_size = read_u32_le(chunk_header.data() + 4);
        std::uint64_t chunk_end = 0;
        if (!checked_add(chunk_data, chunk_size, chunk_end) || chunk_end > riff_end) {
            return false;
        }
        std::uint64_t padded_end = chunk_end;
        if ((chunk_size & 1U) != 0 &&
            (!checked_add(padded_end, 1, padded_end) || padded_end > riff_end)) {
            return false;
        }

        if (std::memcmp(chunk_header.data(), "fmt ", 4) == 0) {
            if (has_fmt || chunk_size < 16) return false;
            std::array<std::uint8_t, 16> format{};
            if (!pread_exact(staged->fd, format.data(), format.size(), chunk_data)) {
                return false;
            }
            staged->audio_format = read_u16_le(format.data());
            channels = read_u16_le(format.data() + 2);
            sample_rate = read_u32_le(format.data() + 4);
            byte_rate = read_u32_le(format.data() + 8);
            staged->block_align = read_u16_le(format.data() + 12);
            bits_per_sample = read_u16_le(format.data() + 14);

            if ((staged->audio_format != kPcm && staged->audio_format != kIeeeFloat) ||
                channels == 0 || channels > kMaxChannels || sample_rate == 0 ||
                sample_rate > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
                (staged->audio_format == kPcm && bits_per_sample != 16) ||
                (staged->audio_format == kIeeeFloat && bits_per_sample != 32)) {
                return false;
            }
            const std::uint64_t bytes_per_sample = bits_per_sample / 8;
            std::uint64_t expected_align = 0;
            std::uint64_t expected_rate = 0;
            if (!checked_mul(channels, bytes_per_sample, expected_align) ||
                expected_align > std::numeric_limits<std::uint16_t>::max() ||
                staged->block_align != expected_align ||
                !checked_mul(sample_rate, expected_align, expected_rate) ||
                expected_rate > std::numeric_limits<std::uint32_t>::max() ||
                byte_rate != expected_rate) {
                return false;
            }
            has_fmt = true;
        } else if (std::memcmp(chunk_header.data(), "data", 4) == 0) {
            if (has_data) return false;
            staged->data_offset = chunk_data;
            staged->data_bytes = chunk_size;
            has_data = true;
        }
        cursor = padded_end;
    }

    if (cursor != riff_end || !has_fmt || !has_data || staged->block_align == 0 ||
        staged->data_bytes % staged->block_align != 0) {
        return false;
    }

    staged->metadata.sample_rate = sample_rate;
    staged->metadata.channels = channels;
    staged->metadata.bits_per_sample = bits_per_sample;
    staged->metadata.encoding = staged->audio_format == kPcm
        ? WavEncoding::kPcm16 : WavEncoding::kFloat32;
    staged->metadata.frame_count = staged->data_bytes / staged->block_align;

    FileIdentity after;
    if (!capture_identity(staged->fd, after) || !same_identity(staged->identity, after)) {
        return false;
    }
    state_ = std::move(staged);
    return true;
} catch (...) {
    close();
    return false;
}

void WavChunkReader::close() noexcept { state_.reset(); }

bool WavChunkReader::is_open() const noexcept { return state_ && state_->fd >= 0; }

WavMetadata WavChunkReader::metadata() const noexcept {
    return state_ ? state_->metadata : WavMetadata{};
}

bool WavChunkReader::verify_unchanged() noexcept {
    if (!state_ || state_->fd < 0) return false;
    FileIdentity current;
    if (!capture_identity(state_->fd, current) ||
        !same_identity(state_->identity, current)) {
        close();
        return false;
    }
    return true;
}

bool WavChunkReader::read_mono_frames(std::uint64_t first_frame,
                                      std::size_t frame_count,
                                      std::vector<float>& output) noexcept try {
    output.clear();
    if (!verify_unchanged()) return false;
    State* const state = state_.get();

    std::uint64_t end_frame = 0;
    if (!checked_add(first_frame, static_cast<std::uint64_t>(frame_count), end_frame) ||
        end_frame > state->metadata.frame_count || frame_count > output.max_size()) {
        return false;
    }
    if (frame_count == 0) return verify_unchanged();
    output.resize(frame_count);

    std::array<std::uint8_t, kDecodeBufferBytes> encoded{};
    const std::size_t frames_per_read =
        std::max<std::size_t>(1, encoded.size() / state->block_align);
    const std::size_t bytes_per_sample = state->metadata.bits_per_sample / 8;
    std::size_t completed = 0;
    while (completed < frame_count) {
        const std::size_t frames = std::min(frame_count - completed, frames_per_read);
        std::uint64_t native_frame = 0;
        std::uint64_t byte_offset = 0;
        std::uint64_t byte_count_u64 = 0;
        if (!checked_add(first_frame, completed, native_frame) ||
            !checked_mul(native_frame, state->block_align, byte_offset) ||
            !checked_add(state->data_offset, byte_offset, byte_offset) ||
            !checked_mul(frames, state->block_align, byte_count_u64) ||
            byte_count_u64 > encoded.size()) {
            output.clear();
            return false;
        }
        const auto byte_count = static_cast<std::size_t>(byte_count_u64);
        if (!pread_exact(state->fd, encoded.data(), byte_count, byte_offset)) {
            output.clear();
            close();
            return false;
        }

        for (std::size_t frame = 0; frame < frames; ++frame) {
            double mono = 0.0;
            const std::size_t frame_offset = frame * state->block_align;
            for (std::size_t channel = 0; channel < state->metadata.channels; ++channel) {
                const std::uint8_t* value =
                    encoded.data() + frame_offset + channel * bytes_per_sample;
                if (state->audio_format == kPcm) {
                    const std::uint16_t bits = read_u16_le(value);
                    const std::int32_t pcm = bits >= 0x8000U
                        ? static_cast<std::int32_t>(bits) - 0x10000
                        : static_cast<std::int32_t>(bits);
                    mono += static_cast<double>(pcm) / 32768.0;
                } else {
                    const std::uint32_t bits = read_u32_le(value);
                    float sample = 0.0f;
                    std::memcpy(&sample, &bits, sizeof(sample));
                    if (!std::isfinite(sample)) {
                        output.clear();
                        (void)verify_unchanged();
                        return false;
                    }
                    mono += sample;
                }
            }
            const float mixed = static_cast<float>(
                mono / static_cast<double>(state->metadata.channels));
            if (!std::isfinite(mixed)) {
                output.clear();
                (void)verify_unchanged();
                return false;
            }
            output[completed + frame] = mixed;
        }
        completed += frames;
    }

    if (!verify_unchanged()) {
        output.clear();
        return false;
    }
    return true;
} catch (...) {
    output.clear();
    return false;
}

struct RawSampleSpoolStorage::State {
    ~State() {
        if (directory_fd >= 0 && !name.empty() &&
            path_matches_object(directory_fd, name, identity)) {
            (void)::unlinkat(directory_fd, name.c_str(), 0);
        }
        close_ignored(fd);
        close_ignored(directory_fd);
    }

    int fd = -1;
    int directory_fd = -1;
    std::filesystem::path directory;
    std::string name;
    FileIdentity identity;
    std::uint64_t sample_count = 0;
    bool failed = false;
};

RawSampleSpoolStorage::RawSampleSpoolStorage(std::size_t element_size) noexcept
    : element_size_(element_size) {}
RawSampleSpoolStorage::~RawSampleSpoolStorage() = default;
RawSampleSpoolStorage::RawSampleSpoolStorage(RawSampleSpoolStorage&&) noexcept = default;
RawSampleSpoolStorage& RawSampleSpoolStorage::operator=(RawSampleSpoolStorage&&) noexcept = default;

bool RawSampleSpoolStorage::create_storage(
        const std::filesystem::path& directory) noexcept try {
    close_storage();
    if (element_size_ != sizeof(float) && element_size_ != sizeof(double)) return false;

    auto staged = std::make_unique<State>();
    staged->directory = normalized_directory(directory);
    const std::string directory_text = staged->directory.string();
    if (directory_text.empty() || directory_text.find('\0') != std::string::npos) return false;
    do {
        staged->directory_fd = ::open(
            directory_text.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    } while (staged->directory_fd < 0 && errno == EINTR);
    if (staged->directory_fd < 0) return false;

    for (unsigned attempt = 0; attempt < kTemporaryCreateAttempts; ++attempt) {
        staged->name = temporary_name(".resonantia-raw-");
        do {
            staged->fd = ::openat(staged->directory_fd, staged->name.c_str(),
                                  O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                                  0600);
        } while (staged->fd < 0 && errno == EINTR);
        if (staged->fd >= 0) break;
        if (errno != EEXIST) return false;
    }
    if (staged->fd < 0 || !capture_identity(staged->fd, staged->identity) ||
        staged->identity.size != 0) {
        return false;
    }
    state_ = std::move(staged);
    return true;
} catch (...) {
    close_storage();
    return false;
}

void RawSampleSpoolStorage::close_storage() noexcept { state_.reset(); }

bool RawSampleSpoolStorage::is_open_storage() const noexcept {
    return state_ && state_->fd >= 0 && !state_->failed;
}

std::uint64_t RawSampleSpoolStorage::sample_count_storage() const noexcept {
    return state_ ? state_->sample_count : 0;
}

std::filesystem::path RawSampleSpoolStorage::path_storage() const {
    return state_ ? state_->directory / state_->name : std::filesystem::path{};
}

bool RawSampleSpoolStorage::append_storage(const void* samples,
                                           std::size_t count) noexcept {
    if (!is_open_storage() || (count > 0 && !samples)) return false;
    if (count == 0) return true;
    State& state = *state_;

    FileIdentity before;
    std::uint64_t expected_bytes = 0;
    if (!capture_identity(state.fd, before) || !same_identity(state.identity, before) ||
        !checked_mul(state.sample_count, element_size_, expected_bytes) ||
        expected_bytes != static_cast<std::uint64_t>(before.size)) {
        state.failed = true;
        return false;
    }

    std::uint64_t new_count = 0;
    std::uint64_t append_bytes_u64 = 0;
    std::uint64_t new_bytes = 0;
    if (!checked_add(state.sample_count, static_cast<std::uint64_t>(count), new_count) ||
        !checked_mul(static_cast<std::uint64_t>(count), element_size_, append_bytes_u64) ||
        append_bytes_u64 > std::numeric_limits<std::size_t>::max() ||
        !checked_mul(new_count, element_size_, new_bytes) ||
        new_bytes > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        !pwrite_exact(state.fd, samples, static_cast<std::size_t>(append_bytes_u64),
                      expected_bytes)) {
        state.failed = true;
        return false;
    }

    FileIdentity after;
    if (!capture_identity(state.fd, after) || !same_object(state.identity, after) ||
        static_cast<std::uint64_t>(after.size) != new_bytes) {
        state.failed = true;
        return false;
    }
    state.identity = after;
    state.sample_count = new_count;
    return true;
}

bool RawSampleSpoolStorage::read_storage(std::uint64_t first_sample,
                                         std::size_t count, void* output) noexcept {
    if (!is_open_storage() || (count > 0 && !output)) return false;
    State& state = *state_;
    std::uint64_t end_sample = 0;
    if (!checked_add(first_sample, static_cast<std::uint64_t>(count), end_sample) ||
        end_sample > state.sample_count) {
        return false;
    }

    FileIdentity before;
    if (!capture_identity(state.fd, before) || !same_identity(state.identity, before)) {
        state.failed = true;
        return false;
    }
    std::uint64_t offset = 0;
    std::uint64_t byte_count_u64 = 0;
    if (!checked_mul(first_sample, element_size_, offset) ||
        !checked_mul(static_cast<std::uint64_t>(count), element_size_, byte_count_u64) ||
        byte_count_u64 > std::numeric_limits<std::size_t>::max() ||
        !pread_exact(state.fd, output, static_cast<std::size_t>(byte_count_u64), offset)) {
        state.failed = true;
        return false;
    }
    FileIdentity after;
    if (!capture_identity(state.fd, after) || !same_identity(state.identity, after)) {
        state.failed = true;
        return false;
    }
    return true;
}

struct AtomicWavStreamWriter::State {
    ~State() {
        if (directory_fd >= 0 && !staging_name.empty() &&
            path_matches_object(directory_fd, staging_name, identity)) {
            (void)::unlinkat(directory_fd, staging_name.c_str(), 0);
        }
        close_ignored(fd);
        close_ignored(directory_fd);
    }

    int fd = -1;
    int directory_fd = -1;
    std::filesystem::path directory;
    std::string staging_name;
    std::string target_name;
    FileIdentity identity;
    WavOutputLayout layout;
    int sample_rate = 0;
    std::uint64_t expected_samples = 0;
    std::uint64_t samples_written = 0;
    bool failed = false;
};

AtomicWavStreamWriter::AtomicWavStreamWriter() noexcept = default;
AtomicWavStreamWriter::~AtomicWavStreamWriter() = default;
AtomicWavStreamWriter::AtomicWavStreamWriter(AtomicWavStreamWriter&&) noexcept = default;
AtomicWavStreamWriter& AtomicWavStreamWriter::operator=(
    AtomicWavStreamWriter&&) noexcept = default;

bool AtomicWavStreamWriter::open(const std::filesystem::path& target,
                                 int sample_rate,
                                 std::uint64_t expected_samples) noexcept try {
    abort();
    const std::string target_text = target.string();
    if (target_text.empty() || target_text.find('\0') != std::string::npos ||
        target.filename().empty()) {
        return false;
    }
    WavOutputLayout layout;
    if (!wav_output_layout(sample_rate, expected_samples, layout)) return false;

    auto staged = std::make_unique<State>();
    staged->directory = normalized_directory(target.parent_path());
    staged->target_name = target.filename().string();
    const std::string directory_text = staged->directory.string();
    if (directory_text.empty() || directory_text.find('\0') != std::string::npos ||
        staged->target_name.empty() ||
        staged->target_name.find('\0') != std::string::npos) {
        return false;
    }
    do {
        staged->directory_fd = ::open(
            directory_text.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    } while (staged->directory_fd < 0 && errno == EINTR);
    if (staged->directory_fd < 0) return false;

    for (unsigned attempt = 0; attempt < kTemporaryCreateAttempts; ++attempt) {
        staged->staging_name = temporary_name(".resonantia-wav-stream-");
        do {
            staged->fd = ::openat(staged->directory_fd, staged->staging_name.c_str(),
                                  O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                                  0666);
        } while (staged->fd < 0 && errno == EINTR);
        if (staged->fd >= 0) break;
        if (errno != EEXIST) return false;
    }
    if (staged->fd < 0) return false;

    const auto header = make_wav_header(sample_rate, layout, false);
    if (!pwrite_exact(staged->fd, header.data(), header.size(), 0) ||
        !capture_identity(staged->fd, staged->identity) ||
        staged->identity.size != static_cast<off_t>(kWavHeaderBytes)) {
        return false;
    }
    staged->layout = layout;
    staged->sample_rate = sample_rate;
    staged->expected_samples = expected_samples;
    state_ = std::move(staged);
    return true;
} catch (...) {
    abort();
    return false;
}

bool AtomicWavStreamWriter::append(std::span<const float> samples) noexcept {
    return append_impl(samples,false);
}
bool AtomicWavStreamWriter::append_decoded_pcm16(std::span<const float> samples) noexcept {
    return append_impl(samples,true);
}
bool AtomicWavStreamWriter::append_impl(std::span<const float> samples,bool exact_pcm16) noexcept {
    if (!state_ || state_->fd < 0 || state_->failed) return false;
    if (samples.empty()) return true;
    for (const float value : samples) {
        if (!std::isfinite(value)||(exact_pcm16&&(value < -1.0f || value > 32767.0f/32768 || std::trunc(value*32768)!=value*32768))) {
            state_->failed = true;
            return false;
        }
    }

    State& state = *state_;
    std::uint64_t new_count = 0;
    if (!checked_add(state.samples_written, samples.size(), new_count) ||
        new_count > state.expected_samples) {
        state.failed = true;
        return false;
    }
    FileIdentity before;
    std::uint64_t current_data_bytes = 0;
    std::uint64_t expected_size = 0;
    if (!checked_mul(state.samples_written, sizeof(std::int16_t), current_data_bytes) ||
        !checked_add(kWavHeaderBytes, current_data_bytes, expected_size) ||
        !capture_identity(state.fd, before) || !same_identity(state.identity, before) ||
        static_cast<std::uint64_t>(before.size) != expected_size) {
        state.failed = true;
        return false;
    }

    std::array<std::uint8_t, kPcmWriteBufferBytes> encoded{};
    std::size_t encoded_size = 0;
    std::uint64_t write_offset = expected_size;
    const auto flush = [&]() noexcept {
        if (!pwrite_exact(state.fd, encoded.data(), encoded_size, write_offset)) return false;
        write_offset += encoded_size;
        encoded_size = 0;
        return true;
    };
    for (const float input : samples) {
        const float sample = std::clamp(input, -1.0f, 1.0f);
        const auto pcm = static_cast<std::uint16_t>(
            static_cast<std::int16_t>(exact_pcm16 ? input * 32768.0f : sample * 32767.0f));
        encoded[encoded_size++] = static_cast<std::uint8_t>(pcm);
        encoded[encoded_size++] = static_cast<std::uint8_t>(pcm >> 8);
        if (encoded_size == encoded.size() && !flush()) {
            state.failed = true;
            return false;
        }
    }
    if (encoded_size > 0 && !flush()) {
        state.failed = true;
        return false;
    }

    std::uint64_t final_data_bytes = 0;
    std::uint64_t final_size = 0;
    FileIdentity after;
    if (!checked_mul(new_count, sizeof(std::int16_t), final_data_bytes) ||
        !checked_add(kWavHeaderBytes, final_data_bytes, final_size) ||
        !capture_identity(state.fd, after) || !same_object(state.identity, after) ||
        static_cast<std::uint64_t>(after.size) != final_size) {
        state.failed = true;
        return false;
    }
    state.identity = after;
    state.samples_written = new_count;
    return true;
}

bool AtomicWavStreamWriter::commit() noexcept {
    if (!state_ || state_->fd < 0 || state_->failed ||
        state_->samples_written != state_->expected_samples) {
        abort();
        return false;
    }
    State& state = *state_;
    FileIdentity before;
    std::uint64_t expected_size = 0;
    if (!checked_add(kWavHeaderBytes, state.layout.data_bytes, expected_size) ||
        !capture_identity(state.fd, before) || !same_identity(state.identity, before) ||
        static_cast<std::uint64_t>(before.size) != expected_size) {
        abort();
        return false;
    }

    const auto header = make_wav_header(state.sample_rate, state.layout, true);
    FileIdentity after_header;
    if (!pwrite_exact(state.fd, header.data(), header.size(), 0) ||
        !capture_identity(state.fd, after_header) ||
        !same_object(state.identity, after_header) ||
        static_cast<std::uint64_t>(after_header.size) != expected_size ||
        !fsync_retry(state.fd) || !close_checked(state.fd)) {
        abort();
        return false;
    }
    state.identity = after_header;
    if (!path_matches_object(state.directory_fd, state.staging_name, state.identity) ||
        ::renameat(state.directory_fd, state.staging_name.c_str(),
                   state.directory_fd, state.target_name.c_str()) != 0) {
        abort();
        return false;
    }

    // rename is the irreversible runtime commit point. Directory fsync is a
    // best-effort namespace durability improvement; a failure cannot honestly
    // be reported as a preserved-target failure after replacement occurred.
    state.staging_name.clear();
    (void)fsync_retry(state.directory_fd);
    close_ignored(state.directory_fd);
    state_.reset();
    return true;
}

void AtomicWavStreamWriter::abort() noexcept { state_.reset(); }

bool AtomicWavStreamWriter::is_open() const noexcept {
    return state_ && state_->fd >= 0 && !state_->failed;
}

std::uint64_t AtomicWavStreamWriter::expected_samples() const noexcept {
    return state_ ? state_->expected_samples : 0;
}

std::uint64_t AtomicWavStreamWriter::samples_written() const noexcept {
    return state_ ? state_->samples_written : 0;
}

std::filesystem::path AtomicWavStreamWriter::staging_path() const {
    return state_ ? state_->directory / state_->staging_name : std::filesystem::path{};
}

}  // namespace voxmutatio::io::detail
