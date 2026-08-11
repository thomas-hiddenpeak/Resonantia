// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace voxmutatio::io::detail {

enum class WavEncoding : std::uint8_t {
    kPcm16,
    kFloat32,
};

struct WavMetadata {
    std::uint32_t sample_rate = 0;
    std::uint16_t channels = 0;
    std::uint16_t bits_per_sample = 0;
    WavEncoding encoding = WavEncoding::kPcm16;
    std::uint64_t frame_count = 0;
};

// Private seekable WAV seam for the bounded offline pipeline. It never mmap's
// caller-controlled audio. Failed open leaves the reader closed; a source
// identity/size change invalidates the reader and makes all later reads fail.
class WavChunkReader {
public:
    WavChunkReader() noexcept;
    ~WavChunkReader();

    WavChunkReader(WavChunkReader&&) noexcept;
    WavChunkReader& operator=(WavChunkReader&&) noexcept;
    WavChunkReader(const WavChunkReader&) = delete;
    WavChunkReader& operator=(const WavChunkReader&) = delete;

    bool open(const std::filesystem::path& path) noexcept;
    void close() noexcept;

    [[nodiscard]] bool is_open() const noexcept;
    [[nodiscard]] WavMetadata metadata() const noexcept;

    // Re-check the opened fd after a complete snapshot pass. This is required
    // even when the caller did not consume the native payload's last frame.
    bool verify_unchanged() noexcept;

    // Decode [first_frame, first_frame + frame_count) and downmix to finite
    // mono float32. Output is cleared on every failure.
    bool read_mono_frames(std::uint64_t first_frame, std::size_t frame_count,
                          std::vector<float>& output) noexcept;

private:
    struct State;
    std::unique_ptr<State> state_;
};

class RawSampleSpoolStorage {
public:
    explicit RawSampleSpoolStorage(std::size_t element_size) noexcept;
    ~RawSampleSpoolStorage();

    RawSampleSpoolStorage(RawSampleSpoolStorage&&) noexcept;
    RawSampleSpoolStorage& operator=(RawSampleSpoolStorage&&) noexcept;
    RawSampleSpoolStorage(const RawSampleSpoolStorage&) = delete;
    RawSampleSpoolStorage& operator=(const RawSampleSpoolStorage&) = delete;

protected:
    bool create_storage(const std::filesystem::path& directory) noexcept;
    void close_storage() noexcept;
    [[nodiscard]] bool is_open_storage() const noexcept;
    [[nodiscard]] std::uint64_t sample_count_storage() const noexcept;
    [[nodiscard]] std::filesystem::path path_storage() const;
    bool append_storage(const void* samples, std::size_t count) noexcept;
    bool read_storage(std::uint64_t first_sample, std::size_t count,
                      void* output) noexcept;

private:
    struct State;
    std::unique_ptr<State> state_;
    std::size_t element_size_ = 0;
};

// Raw float/double temporary storage with bounded random reads. Files use mode
// 0600 and are unlinked by RAII; values are host-native finite IEEE samples.
template <typename Sample>
class RawSampleSpool final : private RawSampleSpoolStorage {
    static_assert(std::is_same_v<Sample, float> || std::is_same_v<Sample, double>);

public:
    RawSampleSpool() noexcept : RawSampleSpoolStorage(sizeof(Sample)) {}
    ~RawSampleSpool() = default;

    RawSampleSpool(RawSampleSpool&&) noexcept = default;
    RawSampleSpool& operator=(RawSampleSpool&&) noexcept = default;
    RawSampleSpool(const RawSampleSpool&) = delete;
    RawSampleSpool& operator=(const RawSampleSpool&) = delete;

    bool create(const std::filesystem::path& directory) noexcept {
        return create_storage(directory);
    }
    void close() noexcept { close_storage(); }

    [[nodiscard]] bool is_open() const noexcept { return is_open_storage(); }
    [[nodiscard]] std::uint64_t sample_count() const noexcept {
        return sample_count_storage();
    }
    [[nodiscard]] std::filesystem::path path() const { return path_storage(); }

    bool append(std::span<const Sample> samples) noexcept {
        for (const Sample value : samples) {
            if (!std::isfinite(value)) return false;
        }
        return append_storage(samples.data(), samples.size());
    }

    bool read(std::uint64_t first_sample, std::size_t count,
              std::vector<Sample>& output) noexcept {
        output.clear();
        if (count > output.max_size()) return false;
        try {
            output.resize(count);
        } catch (...) {
            output.clear();
            return false;
        }
        if (!read_storage(first_sample, count, output.data())) {
            output.clear();
            return false;
        }
        for (const Sample value : output) {
            if (!std::isfinite(value)) {
                output.clear();
                return false;
            }
        }
        return true;
    }
};

using RawFloatSpool = RawSampleSpool<float>;
using RawDoubleSpool = RawSampleSpool<double>;

// Incremental mono PCM16 WAV publication. The staging file is created in the
// target directory with 0666+umask. Exact count, header backpatch and file
// fsync/close precede rename. Rename is the irreversible runtime commit point;
// parent-directory fsync is best-effort and is not a crash-durability promise.
class AtomicWavStreamWriter {
public:
    AtomicWavStreamWriter() noexcept;
    ~AtomicWavStreamWriter();

    AtomicWavStreamWriter(AtomicWavStreamWriter&&) noexcept;
    AtomicWavStreamWriter& operator=(AtomicWavStreamWriter&&) noexcept;
    AtomicWavStreamWriter(const AtomicWavStreamWriter&) = delete;
    AtomicWavStreamWriter& operator=(const AtomicWavStreamWriter&) = delete;

    bool open(const std::filesystem::path& target, int sample_rate,
              std::uint64_t expected_samples) noexcept;
    bool append(std::span<const float> samples) noexcept;
    bool commit() noexcept;
    void abort() noexcept;

    [[nodiscard]] bool is_open() const noexcept;
    [[nodiscard]] std::uint64_t expected_samples() const noexcept;
    [[nodiscard]] std::uint64_t samples_written() const noexcept;
    [[nodiscard]] std::filesystem::path staging_path() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace voxmutatio::io::detail
