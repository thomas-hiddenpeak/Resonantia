// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/io/audio_io.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace voxmutatio::io {

namespace {

constexpr std::uint16_t kPcm = 1;
constexpr std::uint16_t kIeeeFloat = 3;
constexpr std::uint16_t kMaxChannels = 256;
constexpr std::size_t kPcmWriteBufferBytes = 8192;
constexpr unsigned kStagingCreateAttempts = 128;
std::atomic<std::uint64_t> g_staging_sequence {0};

std::uint16_t read_u16_le(const std::uint8_t* data) {
    return static_cast<std::uint16_t>(data[0]) |
           (static_cast<std::uint16_t>(data[1]) << 8);
}

std::uint32_t read_u32_le(const std::uint8_t* data) {
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8) |
           (static_cast<std::uint32_t>(data[2]) << 16) |
           (static_cast<std::uint32_t>(data[3]) << 24);
}

bool checked_add(std::uint64_t a, std::uint64_t b, std::uint64_t& result) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) return false;
    result = a + b;
    return true;
}

bool checked_mul(std::uint64_t a, std::uint64_t b, std::uint64_t& result) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return false;
    result = a * b;
    return true;
}

bool read_exact(std::ifstream& file, void* output, std::size_t size) {
    if (size == 0) return true;
    if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }
    file.read(static_cast<char*>(output), static_cast<std::streamsize>(size));
    return file.good() || (file.eof() && static_cast<std::size_t>(file.gcount()) == size);
}

bool seek_to(std::ifstream& file, std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return false;
    }
    file.clear();
    file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    return static_cast<bool>(file);
}

struct WavFormat {
    std::uint16_t audio_format = 0;
    std::uint16_t channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint32_t byte_rate = 0;
    std::uint16_t block_align = 0;
    std::uint16_t bits_per_sample = 0;
};

bool validate_format(const WavFormat& format) {
    if ((format.audio_format != kPcm && format.audio_format != kIeeeFloat) ||
        format.channels == 0 || format.channels > kMaxChannels ||
        format.sample_rate == 0 ||
        format.sample_rate > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    if ((format.audio_format == kPcm && format.bits_per_sample != 16) ||
        (format.audio_format == kIeeeFloat && format.bits_per_sample != 32)) {
        return false;
    }

    const std::uint64_t bytes_per_sample = format.bits_per_sample / 8;
    std::uint64_t expected_align = 0;
    std::uint64_t expected_rate = 0;
    return checked_mul(format.channels, bytes_per_sample, expected_align) &&
           expected_align <= std::numeric_limits<std::uint16_t>::max() &&
           format.block_align == expected_align &&
           checked_mul(format.sample_rate, expected_align, expected_rate) &&
           expected_rate <= std::numeric_limits<std::uint32_t>::max() &&
           format.byte_rate == expected_rate;
}

bool read_wav(const std::string& path, std::vector<float>& samples,
              int& sample_rate, SampleFormat& source_format, bool stereo = false) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    file.seekg(0, std::ios::end);
    const std::streamoff stream_size = file.tellg();
    if (stream_size < 12) return false;
    const auto file_size = static_cast<std::uint64_t>(stream_size);
    if (stereo && file_size > (128u << 20)) return false;
    if (!seek_to(file, 0)) return false;

    std::array<std::uint8_t, 12> riff {};
    if (!read_exact(file, riff.data(), riff.size()) ||
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
    WavFormat format;
    std::uint64_t data_offset = 0;
    std::uint32_t data_size = 0;
    std::uint64_t cursor = 12;

    while (cursor < riff_end) {
        std::uint64_t chunk_header_end = 0;
        if (!checked_add(cursor, 8, chunk_header_end) || chunk_header_end > riff_end ||
            !seek_to(file, cursor)) {
            return false;
        }
        std::array<std::uint8_t, 8> chunk_header {};
        if (!read_exact(file, chunk_header.data(), chunk_header.size())) return false;
        const std::uint32_t chunk_size = read_u32_le(chunk_header.data() + 4);
        const std::uint64_t chunk_data = chunk_header_end;
        std::uint64_t chunk_end = 0;
        if (!checked_add(chunk_data, chunk_size, chunk_end) || chunk_end > riff_end) return false;
        std::uint64_t padded_end = chunk_end;
        if ((chunk_size & 1U) != 0 && (!checked_add(padded_end, 1, padded_end) || padded_end > riff_end)) {
            return false;
        }

        if (std::memcmp(chunk_header.data(), "fmt ", 4) == 0) {
            if (has_fmt || chunk_size < 16 || !seek_to(file, chunk_data)) return false;
            std::array<std::uint8_t, 16> body {};
            if (!read_exact(file, body.data(), body.size())) return false;
            format.audio_format = read_u16_le(body.data());
            format.channels = read_u16_le(body.data() + 2);
            format.sample_rate = read_u32_le(body.data() + 4);
            format.byte_rate = read_u32_le(body.data() + 8);
            format.block_align = read_u16_le(body.data() + 12);
            format.bits_per_sample = read_u16_le(body.data() + 14);
            if (!validate_format(format)) return false;
            has_fmt = true;
        } else if (std::memcmp(chunk_header.data(), "data", 4) == 0) {
            if (has_data) return false;
            data_offset = chunk_data;
            data_size = chunk_size;
            has_data = true;
        }
        cursor = padded_end;
    }

    if (cursor != riff_end || !has_fmt || !has_data ||
        data_size % format.block_align != 0) {
        return false;
    }

    const std::size_t frame_count = data_size / format.block_align;
    if (stereo && (format.channels > 2 ||
        frame_count > static_cast<std::uint64_t>(format.sample_rate) * 600)) return false;
    if (frame_count > samples.max_size() || data_size > std::vector<std::uint8_t>().max_size()) {
        return false;
    }
    std::vector<std::uint8_t> encoded(data_size);
    if (!seek_to(file, data_offset) || !read_exact(file, encoded.data(), encoded.size())) return false;

    samples.assign(frame_count * (stereo ? 2 : 1), 0.0f);
    const std::size_t bytes_per_sample = format.bits_per_sample / 8;
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        double mono = 0.0;
        const std::size_t frame_offset = frame * format.block_align;
        for (std::size_t channel = 0; channel < format.channels; ++channel) {
            double decoded = 0;
            const std::uint8_t* value = encoded.data() + frame_offset + channel * bytes_per_sample;
            if (format.audio_format == kPcm) {
                const auto bits = read_u16_le(value);
                const std::int32_t pcm = bits >= 0x8000U
                    ? static_cast<std::int32_t>(bits) - 0x10000
                    : static_cast<std::int32_t>(bits);
                decoded = static_cast<double>(pcm) / 32768.0;
            } else {
                const std::uint32_t bits = read_u32_le(value);
                float sample = 0.0f;
                std::memcpy(&sample, &bits, sizeof(sample));
                if (!std::isfinite(sample)) return false;
                decoded = sample;
            }
            mono += decoded;
            if (stereo) samples[frame * 2 + channel] = static_cast<float>(decoded);
        }
        if (!stereo) samples[frame] = static_cast<float>(mono / format.channels);
        else if (format.channels == 1) samples[frame * 2 + 1] = samples[frame * 2];
    }

    sample_rate = static_cast<int>(format.sample_rate);
    source_format = format.audio_format == kPcm ? SampleFormat::kInt16
                                                : SampleFormat::kFloat32;
    return true;
}

bool write_bytes(int fd, const void* data, std::size_t size) {
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    while (size > 0) {
        const ssize_t written = ::write(fd, cursor, size);
        if (written < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (written == 0) return false;
        cursor += written;
        size -= static_cast<std::size_t>(written);
    }
    return true;
}

bool write_u16_le(int fd, std::uint16_t value) {
    const std::array<std::uint8_t, 2> bytes = {
        static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8)};
    return write_bytes(fd, bytes.data(), bytes.size());
}

bool write_u32_le(int fd, std::uint32_t value) {
    const std::array<std::uint8_t, 4> bytes = {
        static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8),
        static_cast<std::uint8_t>(value >> 16), static_cast<std::uint8_t>(value >> 24)};
    return write_bytes(fd, bytes.data(), bytes.size());
}

struct WavOutputLayout {
    std::uint32_t data_bytes = 0;
    std::uint32_t riff_size = 0;
    std::uint32_t byte_rate = 0;
};

bool validate_wav_output(const float* samples, int num_samples, int sample_rate,
                         WavOutputLayout& layout, int channels = 1, bool float32 = false) {
    if (num_samples < 0 || sample_rate <= 0 || (num_samples > 0 && samples == nullptr)) {
        return false;
    }
    for (int i = 0; i < num_samples; ++i) {
        if (!std::isfinite(samples[i])) return false;
    }

    std::uint64_t data_bytes = 0;
    const unsigned width = float32 ? 4 : 2;
    if (!checked_mul(static_cast<std::uint64_t>(num_samples), width, data_bytes) ||
        data_bytes > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    std::uint64_t riff_size = 0;
    if (!checked_add(36, data_bytes, riff_size) ||
        riff_size > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    const std::uint64_t byte_rate = static_cast<std::uint64_t>(sample_rate) * width * channels;
    if (byte_rate > std::numeric_limits<std::uint32_t>::max()) return false;

    layout.data_bytes = static_cast<std::uint32_t>(data_bytes);
    layout.riff_size = static_cast<std::uint32_t>(riff_size);
    layout.byte_rate = static_cast<std::uint32_t>(byte_rate);
    return true;
}

class StagedFile {
public:
    explicit StagedFile(const std::filesystem::path& parent) {
        for (unsigned attempt = 0; attempt < kStagingCreateAttempts; ++attempt) {
            const auto sequence = g_staging_sequence.fetch_add(1, std::memory_order_relaxed);
            path_ = (parent / (".resonantia-wav-" +
                               std::to_string(static_cast<long long>(::getpid())) + "-" +
                               std::to_string(sequence) + ".tmp")).string();
            do {
                fd_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
            } while (fd_ < 0 && errno == EINTR);
            if (fd_ >= 0) return;
            if (errno != EEXIST) break;
        }
        path_.clear();
    }

    ~StagedFile() {
        close_without_commit();
        remove_staging();
    }

    StagedFile(const StagedFile&) = delete;
    StagedFile& operator=(const StagedFile&) = delete;

    [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }
    [[nodiscard]] int fd() const noexcept { return fd_; }

    bool commit(const std::string& target) noexcept {
        if (!flush_and_close()) return false;
        if (::rename(path_.c_str(), target.c_str()) != 0) return false;
        path_.clear();
        return true;
    }

private:
    bool flush_and_close() noexcept {
        if (fd_ < 0) return false;
        int result = 0;
        do {
            result = ::fsync(fd_);
        } while (result != 0 && errno == EINTR);
        if (result != 0) return false;

        const int fd = std::exchange(fd_, -1);
        return ::close(fd) == 0;
    }

    void close_without_commit() noexcept {
        if (fd_ < 0) return;
        const int fd = std::exchange(fd_, -1);
        (void)::close(fd);
    }

    void remove_staging() noexcept {
        if (path_.empty()) return;
        int result = 0;
        do {
            result = ::unlink(path_.c_str());
        } while (result != 0 && errno == EINTR);
    }

    std::string path_;
    int fd_ = -1;
};

bool write_wav_contents(int fd, const float* samples, int num_samples,
                        int sample_rate, const WavOutputLayout& layout,
                        int channels = 1, bool float32 = false) {
    if (!write_bytes(fd, "RIFF", 4) ||
        !write_u32_le(fd, layout.riff_size) ||
        !write_bytes(fd, "WAVEfmt ", 8) ||
        !write_u32_le(fd, 16) ||
        !write_u16_le(fd, float32 ? kIeeeFloat : kPcm) ||
        !write_u16_le(fd, channels) ||
        !write_u32_le(fd, static_cast<std::uint32_t>(sample_rate)) ||
        !write_u32_le(fd, layout.byte_rate) ||
        !write_u16_le(fd, channels * (float32 ? 4 : 2)) ||
        !write_u16_le(fd, float32 ? 32 : 16) ||
        !write_bytes(fd, "data", 4) ||
        !write_u32_le(fd, layout.data_bytes)) {
        return false;
    }

    std::array<std::uint8_t, kPcmWriteBufferBytes> encoded {};
    std::size_t encoded_size = 0;
    for (int i = 0; i < num_samples; ++i) {
        if (float32) {
            std::uint32_t bits;
            std::memcpy(&bits, samples + i, sizeof bits);
            for (unsigned shift = 0; shift < 32; shift += 8)
                encoded[encoded_size++] = static_cast<std::uint8_t>(bits >> shift);
            if (encoded_size == encoded.size()) {
                if (!write_bytes(fd, encoded.data(), encoded_size)) return false;
                encoded_size = 0;
            }
            continue;
        }
        const float sample = std::clamp(samples[i], -1.0f, 1.0f);
        const auto pcm = static_cast<std::uint16_t>(
            static_cast<std::int16_t>(sample * 32767.0f));
        encoded[encoded_size++] = static_cast<std::uint8_t>(pcm);
        encoded[encoded_size++] = static_cast<std::uint8_t>(pcm >> 8);
        if (encoded_size == encoded.size()) {
            if (!write_bytes(fd, encoded.data(), encoded_size)) return false;
            encoded_size = 0;
        }
    }
    return write_bytes(fd, encoded.data(), encoded_size);
}

bool write_wav(const std::string& path, const float* samples,
               int num_samples, int sample_rate, int channels = 1, bool float32 = false) try {
    if (path.empty() || path.find('\0') != std::string::npos) return false;

    WavOutputLayout layout;
    if (!validate_wav_output(samples, num_samples, sample_rate, layout, channels, float32)) return false;

    const std::filesystem::path target(path);
    if (target.filename().empty()) return false;
    std::filesystem::path parent = target.parent_path();
    if (parent.empty()) parent = ".";
    StagedFile staged(parent);
    if (!staged.valid() ||
        !write_wav_contents(staged.fd(), samples, num_samples, sample_rate, layout, channels, float32)) {
        return false;
    }
    return staged.commit(path);
} catch (...) {
    return false;
}

}  // namespace

std::optional<StereoAudioBuffer> read_stereo_audio(const std::string& path, int target_sr) try {
    if (target_sr < 8000 || target_sr > 192000) return std::nullopt;
    StereoAudioBuffer buffer;
    SampleFormat format;
    if (!read_wav(path, buffer.data, buffer.sample_rate, format, true)) return std::nullopt;
    if (buffer.sample_rate != target_sr && !buffer.data.empty()) {
        std::vector<float> left(buffer.frames()), right(buffer.frames());
        for (std::size_t i = 0; i < left.size(); ++i) {
            left[i] = buffer.data[2*i]; right[i] = buffer.data[2*i+1];
        }
        left = resample_linear(left.data(), static_cast<int>(left.size()), buffer.sample_rate, target_sr);
        right = resample_linear(right.data(), static_cast<int>(right.size()), buffer.sample_rate, target_sr);
        if (left.empty() || left.size() != right.size()) return std::nullopt;
        buffer.data.resize(left.size() * 2);
        for (std::size_t i = 0; i < left.size(); ++i) {
            buffer.data[2*i] = left[i]; buffer.data[2*i+1] = right[i];
        }
    }
    buffer.sample_rate = target_sr;
    return buffer;
} catch (...) { return std::nullopt; }

bool write_stereo_audio(const std::string& path, const float* samples, int frames,
                        int sample_rate, bool float32) {
    if (sample_rate < 8000 || sample_rate > 192000 || frames < 0 ||
        frames > sample_rate * 600 || frames > std::numeric_limits<int>::max()/2) return false;
    return write_wav(path, samples, frames * 2, sample_rate, 2, float32);
}

std::optional<AudioBuffer> read_audio(const std::string& path, int target_sr) try {
    if (target_sr <= 0) return std::nullopt;

    std::vector<float> samples;
    int native_sr = 0;
    SampleFormat source_format = SampleFormat::kFloat32;
    if (!read_wav(path, samples, native_sr, source_format)) return std::nullopt;

    AudioBuffer buffer;
    buffer.data = std::move(samples);
    buffer.sample_rate = native_sr;
    buffer.source_format = source_format;

    if (native_sr != target_sr) {
        std::vector<float> resampled = resample_linear(
            buffer.data.data(), static_cast<int>(buffer.data.size()), native_sr, target_sr);
        if (!buffer.data.empty() && resampled.empty()) return std::nullopt;
        buffer.data = std::move(resampled);
        buffer.sample_rate = target_sr;
    }
    return buffer;
} catch (const std::bad_alloc&) {
    return std::nullopt;
} catch (const std::length_error&) {
    return std::nullopt;
}

bool write_audio(const std::string& path, const float* samples,
                 int num_samples, int sample_rate, const std::string& format) {
    return format == "wav" && write_wav(path, samples, num_samples, sample_rate);
}

std::vector<float> resample_linear(const float* input, int input_len,
                                   int src_sr, int dst_sr) try {
    if (input == nullptr || input_len <= 0 || src_sr <= 0 || dst_sr <= 0) return {};
    if (src_sr == dst_sr) return std::vector<float>(input, input + input_len);

    std::uint64_t scaled_length = 0;
    if (!checked_mul(static_cast<std::uint64_t>(input_len),
                     static_cast<std::uint64_t>(dst_sr), scaled_length)) {
        return {};
    }
    const std::uint64_t output_length_u64 = scaled_length / static_cast<std::uint64_t>(src_sr);
    if (output_length_u64 == 0 ||
        output_length_u64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        output_length_u64 > std::vector<float>().max_size()) {
        return {};
    }
    const std::size_t output_length = static_cast<std::size_t>(output_length_u64);
    std::vector<float> output(output_length);
    for (std::size_t i = 0; i < output_length; ++i) {
        const long double position = static_cast<long double>(i) * src_sr / dst_sr;
        std::size_t index = static_cast<std::size_t>(position);
        if (index >= static_cast<std::size_t>(input_len)) index = input_len - 1;
        const long double fraction = position - static_cast<long double>(index);
        if (index + 1 < static_cast<std::size_t>(input_len)) {
            output[i] = static_cast<float>(input[index] * (1.0L - fraction) +
                                           input[index + 1] * fraction);
        } else {
            output[i] = input[index];
        }
    }
    return output;
} catch (const std::bad_alloc&) {
    return {};
} catch (const std::length_error&) {
    return {};
}

}  // namespace voxmutatio::io
