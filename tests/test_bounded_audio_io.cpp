// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "io/bounded_audio_io.h"
#include "subprocess_test.h"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
using voxmutatio::io::detail::AtomicWavStreamWriter;
using voxmutatio::io::detail::RawDoubleSpool;
using voxmutatio::io::detail::RawFloatSpool;
using voxmutatio::io::detail::WavChunkReader;
using voxmutatio::io::detail::WavEncoding;

class TempDirectory {
public:
    TempDirectory() {
        std::array<char, 64> pattern{};
        const std::string value = "/tmp/resonantia-bounded-io-XXXXXX";
        std::memcpy(pattern.data(), value.c_str(), value.size() + 1);
        char* created = ::mkdtemp(pattern.data());
        if (created) path_ = created;
    }
    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

    [[nodiscard]] const fs::path& path() const noexcept { return path_; }

private:
    fs::path path_;
};

class ScopedUmask {
public:
    explicit ScopedUmask(mode_t replacement) : previous_(::umask(replacement)) {}
    ScopedUmask(const ScopedUmask&) = delete;
    ScopedUmask& operator=(const ScopedUmask&) = delete;
    ~ScopedUmask() { (void)::umask(previous_); }

private:
    mode_t previous_;
};

void append_u16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8));
}

void append_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8));
    bytes.push_back(static_cast<std::uint8_t>(value >> 16));
    bytes.push_back(static_cast<std::uint8_t>(value >> 24));
}

void patch_u32(std::vector<std::uint8_t>& bytes, std::size_t offset,
               std::uint32_t value) {
    REQUIRE(offset + 4 <= bytes.size());
    bytes[offset] = static_cast<std::uint8_t>(value);
    bytes[offset + 1] = static_cast<std::uint8_t>(value >> 8);
    bytes[offset + 2] = static_cast<std::uint8_t>(value >> 16);
    bytes[offset + 3] = static_cast<std::uint8_t>(value >> 24);
}

bool write_bytes(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(output);
}

std::vector<std::uint8_t> read_bytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::vector<std::uint8_t> pcm16_wav(std::uint32_t sample_rate,
                                    std::uint16_t channels,
                                    std::span<const std::int16_t> samples,
                                    bool add_unknown_chunk = false) {
    REQUIRE(channels > 0);
    REQUIRE(samples.size() % channels == 0);
    std::vector<std::uint8_t> bytes;
    bytes.insert(bytes.end(), {'R', 'I', 'F', 'F'});
    append_u32(bytes, 0);
    bytes.insert(bytes.end(), {'W', 'A', 'V', 'E'});
    bytes.insert(bytes.end(), {'f', 'm', 't', ' '});
    append_u32(bytes, 16);
    append_u16(bytes, 1);
    append_u16(bytes, channels);
    append_u32(bytes, sample_rate);
    const std::uint16_t block_align = channels * sizeof(std::int16_t);
    append_u32(bytes, sample_rate * block_align);
    append_u16(bytes, block_align);
    append_u16(bytes, 16);
    if (add_unknown_chunk) {
        bytes.insert(bytes.end(), {'J', 'U', 'N', 'K'});
        append_u32(bytes, 3);
        bytes.insert(bytes.end(), {7, 8, 9, 0});
    }
    bytes.insert(bytes.end(), {'d', 'a', 't', 'a'});
    append_u32(bytes, static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t)));
    for (const std::int16_t sample : samples) {
        append_u16(bytes, static_cast<std::uint16_t>(sample));
    }
    patch_u32(bytes, 4, static_cast<std::uint32_t>(bytes.size() - 8));
    return bytes;
}

std::vector<std::uint8_t> float32_wav(std::uint32_t sample_rate,
                                      std::span<const float> samples) {
    std::vector<std::uint8_t> bytes;
    bytes.insert(bytes.end(), {'R', 'I', 'F', 'F'});
    append_u32(bytes, 0);
    bytes.insert(bytes.end(), {'W', 'A', 'V', 'E'});
    bytes.insert(bytes.end(), {'f', 'm', 't', ' '});
    append_u32(bytes, 16);
    append_u16(bytes, 3);
    append_u16(bytes, 1);
    append_u32(bytes, sample_rate);
    append_u32(bytes, sample_rate * sizeof(float));
    append_u16(bytes, sizeof(float));
    append_u16(bytes, 32);
    bytes.insert(bytes.end(), {'d', 'a', 't', 'a'});
    append_u32(bytes, static_cast<std::uint32_t>(samples.size() * sizeof(float)));
    for (const float sample : samples) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        append_u32(bytes, bits);
    }
    patch_u32(bytes, 4, static_cast<std::uint32_t>(bytes.size() - 8));
    return bytes;
}

mode_t permissions(const fs::path& path) {
    struct stat status {};
    REQUIRE(::stat(path.c_str(), &status) == 0);
    return status.st_mode & 0777;
}

}  // namespace

TEST_CASE("Bounded WAV no-writer FIFO subprocess helper",
          "[.][bounded-io-fifo-helper]") {
    const char* fifo_path = std::getenv("RESONANTIA_SUBPROCESS_RESULT_PATH");
    REQUIRE(fifo_path != nullptr);
    WavChunkReader reader;
    CHECK_FALSE(reader.open(fifo_path));
    CHECK_FALSE(reader.is_open());
}

TEST_CASE("Bounded WAV reader promptly rejects a no-writer FIFO",
          "[bounded-io][reader][contract]") {
    TempDirectory directory;
    REQUIRE(!directory.path().empty());
    const fs::path fifo = directory.path() / "checkpoint.fifo";
    REQUIRE(::mkfifo(fifo.c_str(), 0600) == 0);

    const auto child = resonantia_test::run_self_test(
        "Bounded WAV no-writer FIFO subprocess helper", 2'000,
        fifo.c_str());
    INFO("launch_error=" << child.launch_error
         << " wait_error=" << child.wait_error
         << " timed_out=" << child.timed_out
         << " status=" << child.status);
    CHECK_FALSE(child.timed_out);
    REQUIRE(child.exited_with(0));
}

TEST_CASE("Bounded WAV reader validates metadata, ranges and fd identity",
          "[bounded-io][reader][contract]") {
    TempDirectory directory;
    REQUIRE(!directory.path().empty());
    const fs::path input = directory.path() / "stereo.wav";
    const std::vector<std::int16_t> interleaved = {
        16384, 0,
        -16384, 16384,
        32767, 32767,
        -32768, -32768,
    };
    REQUIRE(write_bytes(input, pcm16_wav(48000, 2, interleaved, true)));

    WavChunkReader reader;
    REQUIRE(reader.open(input));
    const auto metadata = reader.metadata();
    CHECK(metadata.sample_rate == 48000);
    CHECK(metadata.channels == 2);
    CHECK(metadata.bits_per_sample == 16);
    CHECK(metadata.encoding == WavEncoding::kPcm16);
    CHECK(metadata.frame_count == 4);

    std::vector<float> decoded{42.0f};
    REQUIRE(reader.read_mono_frames(1, 2, decoded));
    REQUIRE(decoded.size() == 2);
    CHECK(decoded[0] == Catch::Approx(0.0f));
    CHECK(decoded[1] == Catch::Approx(32767.0f / 32768.0f));
    REQUIRE(reader.read_mono_frames(4, 0, decoded));
    CHECK(decoded.empty());

    decoded = {42.0f};
    CHECK_FALSE(reader.read_mono_frames(4, 1, decoded));
    CHECK(decoded.empty());
    CHECK_FALSE(reader.read_mono_frames(std::numeric_limits<std::uint64_t>::max(),
                                        1, decoded));
    CHECK(decoded.empty());
    REQUIRE(reader.verify_unchanged());

    // A same-fd truncate is observed before any access and invalidates state;
    // no mmap/SIGBUS path exists and the caller never receives partial output.
    REQUIRE(::truncate(input.c_str(), 16) == 0);
    decoded = {42.0f};
    CHECK_FALSE(reader.read_mono_frames(0, 1, decoded));
    CHECK(decoded.empty());
    CHECK_FALSE(reader.is_open());
    CHECK_FALSE(reader.verify_unchanged());
}

TEST_CASE("Bounded WAV reader rejects truncated and non-finite payloads",
          "[bounded-io][reader][contract]") {
    TempDirectory directory;
    REQUIRE(!directory.path().empty());

    const fs::path truncated = directory.path() / "truncated.wav";
    const std::array<std::int16_t, 2> truncated_samples = {1, 2};
    auto malformed = pcm16_wav(16000, 1, truncated_samples);
    malformed.resize(malformed.size() - 1);
    REQUIRE(write_bytes(truncated, malformed));
    WavChunkReader reader;
    CHECK_FALSE(reader.open(truncated));
    CHECK_FALSE(reader.is_open());

    const fs::path non_finite = directory.path() / "non-finite.wav";
    const std::array<float, 2> values = {
        0.25f, std::numeric_limits<float>::quiet_NaN()};
    REQUIRE(write_bytes(non_finite, float32_wav(16000, values)));
    REQUIRE(reader.open(non_finite));
    CHECK(reader.metadata().encoding == WavEncoding::kFloat32);
    std::vector<float> output{42.0f};
    CHECK_FALSE(reader.read_mono_frames(0, values.size(), output));
    CHECK(output.empty());

    // Explicit final verification detects fd metadata/identity drift even when
    // a resampler did not request the native payload tail. Toggle a mode bit as
    // a deterministic ctime change after the same-size rewrite.
    const fs::path rewritten = directory.path() / "rewritten.wav";
    const std::array<std::int16_t, 2> rewritten_samples = {3, 4};
    auto stable = pcm16_wav(16000, 1, rewritten_samples);
    REQUIRE(write_bytes(rewritten, stable));
    REQUIRE(reader.open(rewritten));
    stable.back() ^= 1U;
    REQUIRE(write_bytes(rewritten, stable));
    REQUIRE(::chmod(rewritten.c_str(), permissions(rewritten) ^ 0100) == 0);
    CHECK_FALSE(reader.verify_unchanged());
    CHECK_FALSE(reader.is_open());
}

TEST_CASE("Raw float and double spools are finite, checked and RAII-cleaned",
          "[bounded-io][spool][contract]") {
    TempDirectory directory;
    REQUIRE(!directory.path().empty());
    fs::path float_path;
    fs::path double_path;

    {
        RawFloatSpool spool;
        REQUIRE(spool.create(directory.path()));
        float_path = spool.path();
        REQUIRE(fs::exists(float_path));
        CHECK(permissions(float_path) == 0600);
        const std::array<float, 4> first = {1.0f, 2.0f, 3.0f, 4.0f};
        REQUIRE(spool.append(first));
        CHECK(spool.sample_count() == first.size());

        std::vector<float> reverse_block;
        REQUIRE(spool.read(2, 2, reverse_block));
        REQUIRE(reverse_block.size() == 2);
        CHECK(reverse_block[0] == Catch::Approx(3.0f));
        CHECK(reverse_block[1] == Catch::Approx(4.0f));
        CHECK_FALSE(spool.read(std::numeric_limits<std::uint64_t>::max(),
                               1, reverse_block));
        CHECK(reverse_block.empty());

        const float invalid = std::numeric_limits<float>::infinity();
        CHECK_FALSE(spool.append(std::span<const float>(&invalid, 1)));
        CHECK(spool.sample_count() == first.size());
        const float tail = 5.0f;
        REQUIRE(spool.append(std::span<const float>(&tail, 1)));
        CHECK(spool.sample_count() == 5);
    }
    CHECK_FALSE(fs::exists(float_path));

    {
        RawDoubleSpool spool;
        REQUIRE(spool.create(directory.path()));
        double_path = spool.path();
        CHECK(permissions(double_path) == 0600);
        const std::array<double, 3> values = {0.5, -1.25, 8.0};
        REQUIRE(spool.append(values));
        std::vector<double> result;
        REQUIRE(spool.read(0, values.size(), result));
        CHECK(result == std::vector<double>(values.begin(), values.end()));
        CHECK_FALSE(spool.read(std::numeric_limits<std::uint64_t>::max(), 1, result));
        CHECK(result.empty());
        CHECK_FALSE(spool.read(std::numeric_limits<std::uint64_t>::max(),
                               std::numeric_limits<std::size_t>::max(), result));
        CHECK(result.empty());

        REQUIRE(::truncate(double_path.c_str(), 0) == 0);
        CHECK_FALSE(spool.read(0, 1, result));
        CHECK_FALSE(spool.is_open());
    }
    CHECK_FALSE(fs::exists(double_path));
}

TEST_CASE("Incremental WAV writer commits exact output with 0666 plus umask",
          "[bounded-io][writer][contract]") {
    TempDirectory directory;
    REQUIRE(!directory.path().empty());
    const fs::path target = directory.path() / "same-path.wav";
    const std::array<std::int16_t, 2> old_samples = {111, 222};
    REQUIRE(write_bytes(target, pcm16_wav(16000, 1, old_samples)));

    WavChunkReader old_reader;
    REQUIRE(old_reader.open(target));
    std::vector<float> old_audio;
    REQUIRE(old_reader.read_mono_frames(0, 2, old_audio));

    AtomicWavStreamWriter writer;
    fs::path staging;
    {
        ScopedUmask mask(0022);
        REQUIRE(writer.open(target, 40000, 4));
        staging = writer.staging_path();
    }
    REQUIRE(fs::exists(staging));
    const std::array<float, 2> first_output = {0.0f, 0.5f};
    const std::array<float, 2> second_output = {-0.5f, 1.0f};
    REQUIRE(writer.append(first_output));
    REQUIRE(writer.append(second_output));
    CHECK(writer.samples_written() == 4);

    // The source fd remains readable until the atomic rename commit point.
    std::vector<float> still_old;
    REQUIRE(old_reader.read_mono_frames(0, 2, still_old));
    CHECK(still_old == old_audio);
    REQUIRE(writer.commit());
    CHECK_FALSE(fs::exists(staging));
    CHECK(permissions(target) == 0644);
    CHECK_FALSE(old_reader.verify_unchanged());  // old inode was replaced/unlinked

    WavChunkReader committed;
    REQUIRE(committed.open(target));
    CHECK(committed.metadata().sample_rate == 40000);
    CHECK(committed.metadata().channels == 1);
    CHECK(committed.metadata().frame_count == 4);
    std::vector<float> output;
    REQUIRE(committed.read_mono_frames(0, 4, output));
    REQUIRE(output.size() == 4);
    CHECK(output[0] == Catch::Approx(0.0f).margin(1.0f / 32768.0f));
    CHECK(output[1] == Catch::Approx(0.5f).margin(1.0f / 32768.0f));
    CHECK(output[2] == Catch::Approx(-0.5f).margin(1.0f / 32768.0f));
    CHECK(output[3] == Catch::Approx(32767.0f / 32768.0f));
}

TEST_CASE("Incremental WAV writer preserves targets before rename commit",
          "[bounded-io][writer][contract]") {
    TempDirectory directory;
    REQUIRE(!directory.path().empty());
    const fs::path target = directory.path() / "preserved.bin";
    const std::vector<std::uint8_t> sentinel = {'o', 'l', 'd', '-', 't', 'a', 'r', 'g', 'e', 't'};
    REQUIRE(write_bytes(target, sentinel));

    fs::path under_staging;
    {
        AtomicWavStreamWriter under;
        REQUIRE(under.open(target, 40000, 2));
        under_staging = under.staging_path();
        const std::array<float, 1> one_sample = {0.25f};
        REQUIRE(under.append(one_sample));
        CHECK_FALSE(under.commit());
        CHECK_FALSE(fs::exists(under_staging));
        CHECK(read_bytes(target) == sentinel);
    }

    fs::path over_staging;
    {
        AtomicWavStreamWriter over;
        REQUIRE(over.open(target, 40000, 1));
        over_staging = over.staging_path();
        const std::array<float, 1> exact = {0.0f};
        REQUIRE(over.append(exact));
        CHECK(over.samples_written() == 1);
        CHECK_FALSE(over.append(exact));
        CHECK_FALSE(over.is_open());
        CHECK_FALSE(over.commit());
        CHECK_FALSE(fs::exists(over_staging));
        CHECK(read_bytes(target) == sentinel);
    }

    fs::path invalid_staging;
    {
        AtomicWavStreamWriter invalid;
        REQUIRE(invalid.open(target, 40000, 1));
        invalid_staging = invalid.staging_path();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        CHECK_FALSE(invalid.append(std::span<const float>(&nan, 1)));
        const float finite = 0.0f;
        CHECK_FALSE(invalid.append(std::span<const float>(&finite, 1)));
        CHECK_FALSE(invalid.commit());
    }
    CHECK_FALSE(fs::exists(invalid_staging));
    CHECK(read_bytes(target) == sentinel);

    AtomicWavStreamWriter overflow;
    CHECK_FALSE(overflow.open(target, 40000,
                              std::numeric_limits<std::uint64_t>::max()));
    CHECK(read_bytes(target) == sentinel);
}
