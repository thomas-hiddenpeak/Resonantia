// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/catch_approx.hpp>

#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/io/safetensors.h"
#include "voxmutatio/core/types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

struct TempFile {
    explicit TempFile(const char* suffix) {
        static unsigned counter = 0;
        path = "/tmp/resonantia_io_" + std::to_string(static_cast<long long>(getpid())) +
               "_" + std::to_string(counter++) + suffix;
    }
    ~TempFile() { std::remove(path.c_str()); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    std::string path;
};

struct TempDir {
    TempDir() {
        static unsigned counter = 0;
        path = "/tmp/resonantia_io_dir_" +
               std::to_string(static_cast<long long>(getpid())) + "_" +
               std::to_string(counter++);
        if (!std::filesystem::create_directory(path)) {
            throw std::runtime_error("failed to create temporary directory");
        }
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    std::string path;
};

void append_u16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8));
}

void append_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8)
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void set_u32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8)
        bytes[offset++] = static_cast<std::uint8_t>(value >> shift);
}

void set_u64(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8)
        bytes[offset++] = static_cast<std::uint8_t>(value >> shift);
}

void append_tag(std::vector<std::uint8_t>& bytes, const char* tag) {
    bytes.insert(bytes.end(), tag, tag + 4);
}

bool write_binary(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(file);
}

std::vector<std::uint8_t> read_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::vector<std::string> directory_entries(const std::string& path) {
    std::vector<std::string> entries;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        entries.push_back(entry.path().filename().string());
    }
    std::sort(entries.begin(), entries.end());
    return entries;
}

std::vector<std::uint8_t> make_wav(std::uint16_t format,
                                   std::uint16_t channels,
                                   std::uint32_t sample_rate,
                                   std::uint32_t byte_rate,
                                   std::uint16_t block_align,
                                   std::uint16_t bits_per_sample,
                                   const std::vector<std::uint8_t>& payload,
                                   bool odd_sized_unknown_chunk = false) {
    std::vector<std::uint8_t> bytes;
    append_tag(bytes, "RIFF");
    append_u32(bytes, 0);
    append_tag(bytes, "WAVE");
    if (odd_sized_unknown_chunk) {
        append_tag(bytes, "JUNK");
        append_u32(bytes, 1);
        bytes.push_back(0x7f);
        bytes.push_back(0);  // RIFF chunks are word-aligned.
    }
    append_tag(bytes, "fmt ");
    append_u32(bytes, 16);
    append_u16(bytes, format);
    append_u16(bytes, channels);
    append_u32(bytes, sample_rate);
    append_u32(bytes, byte_rate);
    append_u16(bytes, block_align);
    append_u16(bytes, bits_per_sample);
    append_tag(bytes, "data");
    append_u32(bytes, static_cast<std::uint32_t>(payload.size()));
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    if ((payload.size() & 1U) != 0) bytes.push_back(0);
    set_u32(bytes, 4, static_cast<std::uint32_t>(bytes.size() - 8));
    return bytes;
}

std::vector<std::uint8_t> make_safetensors(std::string header,
                                           const std::vector<std::uint8_t>& payload) {
    while ((header.size() & 7U) != 0) header.push_back(' ');
    std::vector<std::uint8_t> bytes;
    append_u64(bytes, header.size());
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    return bytes;
}

std::vector<std::uint8_t> float_payload(std::initializer_list<float> values) {
    std::vector<std::uint8_t> bytes;
    for (float value : values) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        append_u32(bytes, bits);
    }
    return bytes;
}

}  // namespace

TEST_CASE("Linear resample same rate", "[io][resample]") {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    auto output = voxmutatio::io::resample_linear(input.data(), 5, 16000, 16000);
    
    REQUIRE(output.size() == 5);
    for (int i = 0; i < 5; ++i) {
        REQUIRE(output[i] == Catch::Approx(input[i]).epsilon(1e-6));
    }
}

TEST_CASE("Linear resample upsample", "[io][resample]") {
    std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f};
    
    auto output = voxmutatio::io::resample_linear(input.data(), 4, 8000, 16000);
    
    // 4 * 16000 / 8000 = 8 samples (truncated to 8)
    REQUIRE(output.size() == 8);
    
    // Check first and last values
    REQUIRE(output[0] == Catch::Approx(0.0f).epsilon(1e-4));
    REQUIRE(output.back() == Catch::Approx(3.0f).epsilon(1e-4));
}

TEST_CASE("Linear resample downsample", "[io][resample]") {
    std::vector<float> input = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f};
    
    auto output = voxmutatio::io::resample_linear(input.data(), 8, 16000, 8000);
    
    REQUIRE(output.size() == 4);
    
    // Check interpolated values
    REQUIRE(output[0] == Catch::Approx(0.0f).epsilon(1e-4));
    REQUIRE(output[1] == Catch::Approx(1.0f).epsilon(1e-4));
    REQUIRE(output[2] == Catch::Approx(2.0f).epsilon(1e-4));
    REQUIRE(output[3] == Catch::Approx(3.0f).epsilon(1e-4));
}

TEST_CASE("Linear resample rejects invalid parameters", "[io][resample][boundary]") {
    const std::array<float, 2> input = {0.0f, 1.0f};
    REQUIRE(voxmutatio::io::resample_linear(nullptr, 2, 16000, 8000).empty());
    REQUIRE(voxmutatio::io::resample_linear(input.data(), 0, 16000, 8000).empty());
    REQUIRE(voxmutatio::io::resample_linear(input.data(), -1, 16000, 8000).empty());
    REQUIRE(voxmutatio::io::resample_linear(input.data(), 2, 0, 8000).empty());
    REQUIRE(voxmutatio::io::resample_linear(input.data(), 2, 16000, -1).empty());
}

TEST_CASE("WAV write and read round-trip", "[io][wav]") {
    TempFile test_file(".wav");
    
    // Generate test audio (sine wave)
    const int sample_rate = 16000;
    const int duration_ms = 100;
    const int num_samples = sample_rate * duration_ms / 1000;
    
    std::vector<float> samples(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        samples[i] = 0.5f * std::sin(2.0f * 3.14159f * 440.0f * i / sample_rate);
    }
    
    // Write WAV
    bool write_ok = voxmutatio::io::write_audio(test_file.path, samples.data(),
                                                 num_samples, sample_rate, "wav");
    REQUIRE(write_ok);
    
    // Read WAV
    auto buffer = voxmutatio::io::read_audio(test_file.path, sample_rate);
    REQUIRE(buffer.has_value());
    REQUIRE(buffer->sample_rate == sample_rate);
    REQUIRE(buffer->source_format == voxmutatio::SampleFormat::kInt16);
    REQUIRE(buffer->num_samples() == static_cast<std::size_t>(num_samples));
    
    // Compare samples (int16 quantization has absolute error up to ~1/32768).
    for (int i = 0; i < num_samples; ++i) {
        REQUIRE(buffer->data[i] == Catch::Approx(samples[i]).margin(1e-4));
    }
}

TEST_CASE("WAV output atomically replaces an existing file", "[io][wav][atomic]") {
    TempDir directory;
    const std::string output = directory.path + "/output.wav";
    const std::string old_link = directory.path + "/old-output";
    const std::vector<std::uint8_t> old_contents = {'o', 'l', 'd'};
    REQUIRE(write_binary(output, old_contents));
    REQUIRE(::link(output.c_str(), old_link.c_str()) == 0);
    struct stat old_status {};
    REQUIRE(::stat(old_link.c_str(), &old_status) == 0);

    const std::array<float, 4> samples = {-0.5f, 0.0f, 0.5f, 1.0f};
    REQUIRE(voxmutatio::io::write_audio(
        output, samples.data(), static_cast<int>(samples.size()), 16000));

    // Atomic replacement gives the destination a new inode. A writer that truncates
    // the destination in place would also overwrite the hard-linked old contents.
    REQUIRE(read_binary(old_link) == old_contents);
    struct stat output_status {};
    REQUIRE(::stat(output.c_str(), &output_status) == 0);
    REQUIRE((output_status.st_mode & 0777) == (old_status.st_mode & 0777));
    auto audio = voxmutatio::io::read_audio(output, 16000);
    REQUIRE(audio.has_value());
    REQUIRE(audio->num_samples() == samples.size());
}

TEST_CASE("WAV write failures preserve the destination and remove staging files",
          "[io][wav][atomic][boundary]") {
    TempDir directory;
    const std::vector<std::uint8_t> old_contents = {'k', 'e', 'e', 'p'};

    SECTION("invalid samples") {
        const std::string output = directory.path + "/output.wav";
        REQUIRE(write_binary(output, old_contents));
        const auto entries_before = directory_entries(directory.path);
        const float invalid = std::numeric_limits<float>::quiet_NaN();

        REQUIRE_FALSE(voxmutatio::io::write_audio(output, &invalid, 1, 16000));

        REQUIRE(read_binary(output) == old_contents);
        REQUIRE(directory_entries(directory.path) == entries_before);
    }

    SECTION("commit target is a directory") {
        const std::string output = directory.path + "/output.wav";
        REQUIRE(std::filesystem::create_directory(output));
        const std::string marker = output + "/marker";
        REQUIRE(write_binary(marker, old_contents));
        const auto entries_before = directory_entries(directory.path);

        const std::array<float, 2> samples = {0.0f, 0.25f};
        REQUIRE_FALSE(voxmutatio::io::write_audio(
            output, samples.data(), static_cast<int>(samples.size()), 16000));

        REQUIRE(std::filesystem::is_directory(output));
        REQUIRE(read_binary(marker) == old_contents);
        REQUIRE(directory_entries(directory.path) == entries_before);
    }
}

TEST_CASE("WAV parser handles padded chunks and downmixes channels",
          "[io][wav][boundary]") {
    std::vector<std::uint8_t> pcm;
    append_u16(pcm, static_cast<std::uint16_t>(32767));
    append_u16(pcm, static_cast<std::uint16_t>(32767));
    append_u16(pcm, static_cast<std::uint16_t>(-32768));
    append_u16(pcm, static_cast<std::uint16_t>(0));
    auto wav = make_wav(/*PCM=*/1, /*channels=*/2, /*sample_rate=*/8000,
                        /*byte_rate=*/32000, /*block_align=*/4,
                        /*bits=*/16, pcm, /*odd unknown chunk=*/true);
    TempFile file(".wav");
    REQUIRE(write_binary(file.path, wav));

    auto audio = voxmutatio::io::read_audio(file.path, 8000);
    REQUIRE(audio.has_value());
    REQUIRE(audio->num_samples() == 2);
    REQUIRE(audio->data[0] == Catch::Approx(32767.0 / 32768.0).margin(1e-6));
    REQUIRE(audio->data[1] == Catch::Approx(-0.5).margin(1e-6));
}

TEST_CASE("WAV parser rejects inconsistent fmt and incomplete data",
          "[io][wav][boundary]") {
    const std::vector<std::uint8_t> one_stereo_frame = {0, 0, 0, 0};

    SECTION("block alignment") {
        auto wav = make_wav(1, 2, 8000, 24000, 3, 16, one_stereo_frame);
        TempFile file(".wav");
        REQUIRE(write_binary(file.path, wav));
        REQUIRE_FALSE(voxmutatio::io::read_audio(file.path, 8000).has_value());
    }
    SECTION("data is not a whole frame") {
        auto wav = make_wav(1, 2, 8000, 32000, 4, 16, {0, 0});
        TempFile file(".wav");
        REQUIRE(write_binary(file.path, wav));
        REQUIRE_FALSE(voxmutatio::io::read_audio(file.path, 8000).has_value());
    }
    SECTION("declared chunk exceeds RIFF range") {
        auto wav = make_wav(1, 2, 8000, 32000, 4, 16, one_stereo_frame);
        wav.resize(wav.size() - 2);
        set_u32(wav, 4, static_cast<std::uint32_t>(wav.size() - 8));
        TempFile file(".wav");
        REQUIRE(write_binary(file.path, wav));
        REQUIRE_FALSE(voxmutatio::io::read_audio(file.path, 8000).has_value());
    }
    SECTION("unsupported sample encoding") {
        auto wav = make_wav(1, 1, 8000, 3000, 3, 24, {0, 0, 0});
        TempFile file(".wav");
        REQUIRE(write_binary(file.path, wav));
        REQUIRE_FALSE(voxmutatio::io::read_audio(file.path, 8000).has_value());
    }
    SECTION("invalid native sample rate") {
        auto wav = make_wav(1, 1, 0, 0, 2, 16, {0, 0});
        TempFile file(".wav");
        REQUIRE(write_binary(file.path, wav));
        REQUIRE_FALSE(voxmutatio::io::read_audio(file.path, 8000).has_value());
    }
    SECTION("non-finite float PCM") {
        auto wav = make_wav(3, 1, 8000, 32000, 4, 32,
                            float_payload({std::numeric_limits<float>::quiet_NaN()}));
        TempFile file(".wav");
        REQUIRE(write_binary(file.path, wav));
        REQUIRE_FALSE(voxmutatio::io::read_audio(file.path, 8000).has_value());
    }
}

TEST_CASE("WAV read non-existent file", "[io][wav]") {
    TempFile file(".wav");
    auto buffer = voxmutatio::io::read_audio(file.path);
    REQUIRE(!buffer.has_value());
}

TEST_CASE("WAV write invalid path", "[io][wav]") {
    std::vector<float> samples = {1.0f, 2.0f, 3.0f};
    
    bool write_ok = voxmutatio::io::write_audio("/invalid/path/file.wav",
                                                 samples.data(), 3, 16000, "wav");
    REQUIRE(!write_ok);
}

TEST_CASE("WAV API rejects invalid write and read arguments", "[io][wav][boundary]") {
    TempFile file(".wav");
    const float finite = 0.0f;
    const float not_finite = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_FALSE(voxmutatio::io::write_audio(file.path, nullptr, 1, 16000));
    REQUIRE_FALSE(voxmutatio::io::write_audio(file.path, &finite, -1, 16000));
    REQUIRE_FALSE(voxmutatio::io::write_audio(file.path, &finite, 1, 0));
    REQUIRE_FALSE(voxmutatio::io::write_audio(file.path, &not_finite, 1, 16000));
    REQUIRE_FALSE(voxmutatio::io::write_audio(file.path, &finite, 1, 16000, "flac"));
    REQUIRE_FALSE(voxmutatio::io::read_audio(file.path, 0).has_value());
}

TEST_CASE("Safetensors validates dtype shape offsets and payload range",
          "[io][safetensors][boundary]") {
    using voxmutatio::io::SafetensorsLoader;
    static_assert(!std::is_copy_constructible_v<SafetensorsLoader>);
    static_assert(std::is_nothrow_move_constructible_v<SafetensorsLoader>);

    const std::string valid_header =
        R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
    TempFile valid_file(".safetensors");
    REQUIRE(write_binary(valid_file.path,
                         make_safetensors(valid_header, float_payload({1.25f, -2.5f}))));

    SafetensorsLoader loader;
    REQUIRE(loader.load(valid_file.path));
    const auto* tensor = loader.get_tensor("x");
    REQUIRE(tensor != nullptr);
    REQUIRE(tensor->dtype == "F32");
    REQUIRE(tensor->shape == std::vector<int64_t>{2});
    REQUIRE(tensor->strides == std::vector<int64_t>{4});
    REQUIRE(tensor->data_nbytes == 8);
    const auto* values = reinterpret_cast<const float*>(loader.data("x"));
    REQUIRE(values != nullptr);
    REQUIRE(values[0] == Catch::Approx(1.25f));
    REQUIRE(values[1] == Catch::Approx(-2.5f));

    SafetensorsLoader moved(std::move(loader));
    REQUIRE_FALSE(loader.valid());
    REQUIRE(moved.valid());

    const auto rejected = [](const std::string& header,
                             const std::vector<std::uint8_t>& payload) {
        TempFile file(".safetensors");
        REQUIRE(write_binary(file.path, make_safetensors(header, payload)));
        SafetensorsLoader candidate;
        REQUIRE_FALSE(candidate.load(file.path));
        REQUIRE_FALSE(candidate.valid());
    };

    SECTION("unsupported dtype") {
        rejected(R"({"x":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}})",
                 float_payload({1.0f}));
    }
    SECTION("negative shape") {
        rejected(R"({"x":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}})", {});
    }
    SECTION("shape multiplication overflow") {
        rejected(R"({"x":{"dtype":"F32","shape":[9223372036854775807,2],"data_offsets":[0,0]}})", {});
    }
    SECTION("shape and byte count disagree") {
        rejected(R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})",
                 float_payload({1.0f}));
    }
    SECTION("offset is outside payload") {
        rejected(R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
                 float_payload({1.0f}));
    }
    SECTION("offsets overlap") {
        const std::string header =
            R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},)"
            R"("b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
        rejected(header,
                 float_payload({1.0f}));
    }
    SECTION("unindexed trailing payload") {
        rejected(R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
                 float_payload({1.0f, 2.0f}));
    }
    SECTION("duplicate tensor key") {
        const std::string header =
            R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},)"
            R"("x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})";
        rejected(header,
                 float_payload({1.0f, 2.0f}));
    }
    SECTION("declared header exceeds file") {
        auto bytes = make_safetensors(valid_header, float_payload({1.0f, 2.0f}));
        set_u64(bytes, 0, static_cast<std::uint64_t>(bytes.size() + 1));
        TempFile file(".safetensors");
        REQUIRE(write_binary(file.path, bytes));
        SafetensorsLoader candidate;
        REQUIRE_FALSE(candidate.load(file.path));
    }

    // A failed replacement load must not leave the old mapping exposed.
    TempFile invalid_file(".safetensors");
    REQUIRE(write_binary(invalid_file.path, make_safetensors("{}", float_payload({1.0f}))));
    REQUIRE_FALSE(moved.load(invalid_file.path));
    REQUIRE_FALSE(moved.valid());
    REQUIRE(moved.data("x") == nullptr);
}
