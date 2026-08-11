// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>

#include "pipeline/bounded_pipeline_detail.h"
#include "voxmutatio/io/audio_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace bounded_io = voxmutatio::io::detail;
namespace bounded_pipeline = voxmutatio::pipeline::detail;

class TempDirectory {
public:
    TempDirectory() {
        std::array<char, 64> pattern{};
        constexpr char value[] = "/tmp/resonantia-bounded-dsp-XXXXXX";
        std::memcpy(pattern.data(), value, sizeof(value));
        if (char* created = ::mkdtemp(pattern.data())) path_ = created;
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

bool write_stereo_pcm16(const fs::path& path, std::uint32_t sample_rate,
                        std::span<const std::int16_t> interleaved) {
    if (interleaved.size() % 2 != 0 || interleaved.size() >
            std::numeric_limits<std::uint32_t>::max() / sizeof(std::int16_t)) {
        return false;
    }
    const auto data_bytes = static_cast<std::uint32_t>(
        interleaved.size() * sizeof(std::int16_t));
    std::vector<std::uint8_t> bytes;
    bytes.reserve(44 + data_bytes);
    bytes.insert(bytes.end(), {'R', 'I', 'F', 'F'});
    append_u32(bytes, 36 + data_bytes);
    bytes.insert(bytes.end(), {'W', 'A', 'V', 'E'});
    bytes.insert(bytes.end(), {'f', 'm', 't', ' '});
    append_u32(bytes, 16);
    append_u16(bytes, 1);
    append_u16(bytes, 2);
    append_u32(bytes, sample_rate);
    append_u32(bytes, sample_rate * 4);
    append_u16(bytes, 4);
    append_u16(bytes, 16);
    bytes.insert(bytes.end(), {'d', 'a', 't', 'a'});
    append_u32(bytes, data_bytes);
    for (const auto sample : interleaved) {
        append_u16(bytes, static_cast<std::uint16_t>(sample));
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(stream);
}

std::vector<float> legacy_highpass(std::vector<float> audio) {
    static constexpr std::array<double, 6> b = {
        9.6996064518e-01, -4.8498032259e+00, 9.6996064518e+00,
        -9.6996064518e+00, 4.8498032259e+00, -9.6996064518e-01};
    static constexpr std::array<double, 6> a = {
        1.0, -4.9390018192e+00, 9.7578635267e+00,
        -9.6395448494e+00, 4.7615067974e+00, -9.4082365321e-01};
    if (audio.size() < 40) return audio;
    const auto iir = [](const std::vector<double>& input) {
        std::vector<double> output(input.size(), 0.0);
        for (std::size_t index = 0; index < input.size(); ++index) {
            double value = b[0] * input[index];
            for (std::size_t order = 1; order < b.size(); ++order) {
                if (index >= order) {
                    value += b[order] * input[index - order] -
                             a[order] * output[index - order];
                }
            }
            output[index] = value;
        }
        return output;
    };
    constexpr std::size_t pad = 18;
    std::vector<double> extended(audio.size() + 2 * pad);
    for (std::size_t index = 0; index < pad; ++index) {
        extended[index] = 2.0 * audio.front() - audio[pad - index];
    }
    for (std::size_t index = 0; index < audio.size(); ++index) {
        extended[pad + index] = audio[index];
    }
    for (std::size_t index = 0; index < pad; ++index) {
        extended[pad + audio.size() + index] =
            2.0 * audio.back() - audio[audio.size() - 2 - index];
    }
    auto forward = iir(extended);
    std::reverse(forward.begin(), forward.end());
    auto backward = iir(forward);
    std::reverse(backward.begin(), backward.end());
    for (std::size_t index = 0; index < audio.size(); ++index) {
        audio[index] = static_cast<float>(backward[pad + index]);
    }
    return audio;
}

std::vector<float> legacy_rms(const std::vector<float>& source,
                              std::vector<float> output, double rate) {
    const auto envelope = [](const std::vector<float>& audio, int sample_rate,
                             std::size_t output_samples) {
        const std::size_t hop = static_cast<std::size_t>(sample_rate / 2);
        const std::size_t frame = 2 * hop;
        const std::size_t record_count = std::max<std::size_t>(
            1, (audio.size() + hop - 1) / hop);
        std::vector<float> records(record_count);
        for (std::size_t record = 0; record < record_count; ++record) {
            const std::size_t first = record * hop;
            const std::size_t end = std::min(audio.size(), first + frame);
            double sum = 0.0;
            for (std::size_t index = first; index < end; ++index) {
                sum += static_cast<double>(audio[index]) * audio[index];
            }
            records[record] = static_cast<float>(
                std::sqrt(sum / std::max<std::size_t>(1, end - first)));
        }
        std::vector<float> result(output_samples);
        for (std::size_t index = 0; index < output_samples; ++index) {
            const double position = static_cast<double>(index) *
                static_cast<double>(record_count - 1) /
                static_cast<double>(std::max<std::size_t>(1, output_samples - 1));
            const auto first = static_cast<std::size_t>(position);
            const float fraction = static_cast<float>(position - first);
            const float left = records[first];
            const float right = records[std::min(first + 1, record_count - 1)];
            result[index] = left + (right - left) * fraction;
        }
        return result;
    };
    const auto source_rms = envelope(source, 16'000, output.size());
    const auto output_rms = envelope(output, 40'000, output.size());
    for (std::size_t index = 0; index < output.size(); ++index) {
        const float denominator = std::max(output_rms[index], 1e-6f);
        output[index] *=
            std::pow(source_rms[index], static_cast<float>(1.0 - rate)) *
            std::pow(denominator, static_cast<float>(rate - 1.0));
    }
    return output;
}

void require_spool_equals(bounded_io::RawFloatSpool& spool,
                          const std::vector<float>& expected,
                          float tolerance) {
    REQUIRE(spool.sample_count() == expected.size());
    std::vector<float> actual;
    REQUIRE(spool.read(0, expected.size(), actual));
    REQUIRE(actual.size() == expected.size());
    float maximum_error = 0.0f;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum_error = std::max(maximum_error,
            std::abs(actual[index] - expected[index]));
    }
    INFO("maximum_error=" << maximum_error << " tolerance=" << tolerance);
    CHECK(maximum_error <= tolerance);
}

}  // namespace

TEST_CASE("Bounded stable snapshot matches legacy global-phase resampling",
          "[bounded-pipeline][dsp][resample][contract]") {
    TempDirectory directory;
    REQUIRE(!directory.path().empty());
    for (const std::uint32_t sample_rate : {48'000U, 44'100U}) {
        const std::size_t frames = sample_rate == 48'000U ? 20'003 : 25'013;
        std::vector<std::int16_t> interleaved(2 * frames);
        for (std::size_t frame = 0; frame < frames; ++frame) {
            const double phase = static_cast<double>(frame) * 0.017;
            interleaved[2 * frame] = static_cast<std::int16_t>(
                std::lround(22'000.0 * std::sin(phase)));
            interleaved[2 * frame + 1] = static_cast<std::int16_t>(
                std::lround(9'000.0 * std::cos(phase * 0.37)));
        }
        const fs::path input = directory.path() /
            ("stereo-" + std::to_string(sample_rate) + ".wav");
        REQUIRE(write_stereo_pcm16(input, sample_rate, interleaved));
        auto legacy = voxmutatio::io::read_audio(input.string(), 16'000);
        REQUIRE(legacy.has_value());

        bounded_io::WavChunkReader reader;
        REQUIRE(reader.open(input));
        const std::uint64_t expected_samples =
            reader.metadata().frame_count * 16'000 / reader.metadata().sample_rate;
        bounded_io::RawFloatSpool stable;
        std::string error;
        REQUIRE(bounded_pipeline::resample_wav_to_stable_snapshot(
            reader, directory.path(), stable, expected_samples, error));
        INFO("sample_rate=" << sample_rate << " error=" << error);
        require_spool_equals(stable, legacy->data, 2e-7f);
    }
}

TEST_CASE("Bounded double filtfilt matches the whole-buffer recurrence",
          "[bounded-pipeline][dsp][highpass][contract]") {
    TempDirectory directory;
    REQUIRE(!directory.path().empty());
    for (const std::size_t samples : {31U, 20'003U}) {
        std::vector<float> source(samples);
        for (std::size_t index = 0; index < source.size(); ++index) {
            source[index] = static_cast<float>(
                0.3 * std::sin(index * 0.013) +
                0.1 * std::cos(index * 0.0007) +
                0.00001 * static_cast<double>(index));
        }
        const auto expected = legacy_highpass(source);
        bounded_io::RawFloatSpool source_spool;
        bounded_io::RawFloatSpool filtered;
        REQUIRE(source_spool.create(directory.path()));
        REQUIRE(source_spool.append(source));
        std::string error;
        REQUIRE(bounded_pipeline::filter_stable_source(
            source_spool, directory.path(), true, filtered, error));
        INFO("samples=" << samples << " error=" << error);
        require_spool_equals(filtered, expected, 2e-6f);
    }
}

TEST_CASE("Bounded quiet planner emits a sub-hop final core",
          "[bounded-pipeline][planner][contract]") {
    TempDirectory directory;
    REQUIRE(!directory.path().empty());
    bounded_io::RawFloatSpool source;
    REQUIRE(source.create(directory.path()));
    const std::array<float, 2> samples = {0.25f, -0.25f};
    REQUIRE(source.append(samples));
    std::uint64_t start = 99;
    std::uint64_t end = 99;
    bool done = true;
    std::string error;
    REQUIRE(bounded_pipeline::plan_first_quiet_core(
        source, samples.size(), 160, 160, 480,
        start, end, done, error));
    INFO(error);
    CHECK_FALSE(done);
    CHECK(start == 0);
    CHECK(end == samples.size());
}

TEST_CASE("Bounded quiet planner preserves legacy float-energy tie breaking",
          "[bounded-pipeline][planner][contract]") {
    TempDirectory directory;
    REQUIRE(!directory.path().empty());
    constexpr std::size_t hop = 160;
    std::vector<float> audio(6 * hop, 0.25f);
    std::fill(audio.begin() + hop, audio.begin() + 2 * hop, 0.0100000f);
    std::fill(audio.begin() + 2 * hop, audio.begin() + 3 * hop, 0.0100000f);
    audio[2 * hop] = std::nextafter(0.0100000f, 0.0f);
    std::fill(audio.begin() + 3 * hop, audio.begin() + 4 * hop, 0.02f);

    const auto double_energy = [&](std::size_t frame) {
        double sum = 0.0;
        for (std::size_t sample = 0; sample < hop; ++sample) {
            sum += std::fabs(audio[frame * hop + sample]);
        }
        return sum;
    };
    const double first_double = double_energy(1);
    const double second_double = double_energy(2);
    REQUIRE(second_double < first_double);
    REQUIRE(static_cast<float>(second_double) ==
            static_cast<float>(first_double));

    std::size_t expected_frame = 1;
    float best_energy = std::numeric_limits<float>::max();
    for (std::size_t frame = 1; frame <= 3; ++frame) {
        double sum = 0.0;
        for (std::size_t sample = 0; sample < hop; ++sample) {
            sum += std::fabs(audio[frame * hop + sample]);
        }
        const float energy = static_cast<float>(sum);
        if (energy < best_energy) {
            best_energy = energy;
            expected_frame = frame;
        }
    }

    bounded_io::RawFloatSpool source;
    REQUIRE(source.create(directory.path()));
    REQUIRE(source.append(audio));
    std::uint64_t start = 99;
    std::uint64_t end = 99;
    bool done = true;
    std::string error;
    REQUIRE(bounded_pipeline::plan_first_quiet_core(
        source, audio.size(), 2 * hop, hop, 4 * hop,
        start, end, done, error));
    INFO(error);
    CHECK_FALSE(done);
    CHECK(start == 0);
    CHECK(expected_frame == 1);
    CHECK(end == hop);
}

TEST_CASE("Bounded global RMS writer matches the legacy envelope and PCM rule",
          "[bounded-pipeline][dsp][rms][contract]") {
    TempDirectory directory;
    REQUIRE(!directory.path().empty());
    constexpr std::size_t source_samples = 20'003;
    constexpr std::size_t output_samples = source_samples * 5 / 2;
    std::vector<float> source(source_samples);
    std::vector<float> output(output_samples);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(
            (0.1 + 0.4 * static_cast<double>(index) / source.size()) *
            std::sin(index * 0.031));
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
        output[index] = static_cast<float>(
            (0.45 - 0.25 * static_cast<double>(index) / output.size()) *
            std::cos(index * 0.019));
    }
    const auto expected_float = legacy_rms(source, output, 0.35);

    bounded_io::RawFloatSpool source_spool;
    bounded_io::RawFloatSpool output_spool;
    REQUIRE(source_spool.create(directory.path()));
    REQUIRE(output_spool.create(directory.path()));
    REQUIRE(source_spool.append(source));
    REQUIRE(output_spool.append(output));
    const fs::path target = directory.path() / "rms.wav";
    bounded_io::AtomicWavStreamWriter writer;
    REQUIRE(writer.open(target, 40'000, output_samples));
    std::string error;
    REQUIRE(bounded_pipeline::write_global_rms_output(
        source_spool, output_spool, directory.path(), 0.35, writer, error));
    INFO(error);
    REQUIRE(writer.commit());
    auto actual = voxmutatio::io::read_audio(target.string(), 40'000);
    REQUIRE(actual.has_value());
    REQUIRE(actual->data.size() == expected_float.size());

    float maximum_error = 0.0f;
    for (std::size_t index = 0; index < actual->data.size(); ++index) {
        const float clamped = std::clamp(expected_float[index], -1.0f, 1.0f);
        const auto pcm = static_cast<std::int16_t>(clamped * 32767.0f);
        const float expected = static_cast<float>(pcm) / 32768.0f;
        maximum_error = std::max(
            maximum_error, std::abs(actual->data[index] - expected));
    }
    INFO("maximum_pcm_error=" << maximum_error);
    CHECK(maximum_error == 0.0f);
}
