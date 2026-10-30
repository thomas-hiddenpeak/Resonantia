// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>

#include "pipeline/bounded_pipeline_detail.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/f0/pitch_review.h"

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

std::vector<float> centered_rms(const std::vector<float>& source,
                              std::vector<float> output, double rate) {
    const auto envelope = [](const std::vector<float>& audio, int sample_rate,
                             std::size_t output_samples) {
        const std::size_t hop = static_cast<std::size_t>(sample_rate / 2);
        const std::size_t frame = 2 * hop;
        const std::size_t record_count = audio.size() / hop + 1;
        std::vector<float> records(record_count);
        for (std::size_t record = 0; record < record_count; ++record) {
            const std::size_t center = record * hop;
            const std::size_t first = center >= frame / 2 ? center - frame / 2 : 0;
            const std::size_t end = std::min(audio.size(), center + frame / 2);
            double sum = 0.0;
            for (std::size_t index = first; index < end; ++index) {
                sum += static_cast<double>(audio[index]) * audio[index];
            }
            records[record] = static_cast<float>(
                std::sqrt(sum / frame));
        }
        std::vector<float> result(output_samples);
        for (std::size_t index = 0; index < output_samples; ++index) {
            const double position = std::clamp((static_cast<double>(index) + 0.5) *
                record_count / output_samples - 0.5, 0.0, static_cast<double>(record_count - 1));
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

TEST_CASE("Default F0 processing preserves unvoiced consonants and silence",
          "[bounded-pipeline][dsp][f0][contract]") {
    TempDirectory directory;
    REQUIRE(!directory.path().empty());
    voxmutatio::VCConfig config;
    std::vector<float> input = {0, 0, 100, 0, 0, 0, 200, 0, 0};
    std::vector<float> expected = input;

    SECTION("default preserves leading interior and trailing unvoiced frames") {}
    SECTION("pitch shift changes only voiced frames") {
        config.f0_up_key = 12;
        for (auto& value : expected) value *= 2;
    }
    SECTION("Web median filter retains unvoiced runs") {
        config.filter_radius = 3;
        input = {0, 0, 100, 110, 120, 0, 0, 0, 220, 210, 200, 0, 0};
        expected = {0, 0, 100, 110, 110, 0, 0, 0, 210, 210, 200, 0, 0};
    }
    SECTION("long silence spans spool blocks") {
        input.assign(40'003, 0);
        input.front() = 100;
        input.back() = 200;
        expected = input;
    }
    SECTION("single unvoiced frame") {
        input = expected = {0};
    }
    SECTION("all-unvoiced compatibility interpolation stays silent") {
        config.interp_unvoiced = true;
        input = expected = {0, 0, 0, 0};
    }
    SECTION("explicit interpolation retains the historical behavior") {
        config.interp_unvoiced = true;
        expected = {100, 100, 100, 125, 150, 175, 200, 200, 200};
    }
    bounded_io::RawFloatSpool raw, processed;
    REQUIRE(raw.create(directory.path()));
    REQUIRE(raw.append(input));
    std::string error;
    REQUIRE(bounded_pipeline::process_f0(raw, directory.path(), config, processed, error));
    INFO(error);
    require_spool_equals(processed, expected, 1e-5f);
    require_spool_equals(raw, input, 0.0f);
}

TEST_CASE("F0 spool creation failure preserves raw input",
          "[bounded-pipeline][dsp][f0][contract]") {
    TempDirectory directory;
    REQUIRE(!directory.path().empty());
    bounded_io::RawFloatSpool raw, processed;
    const std::vector<float> input = {0, 120, 0};
    REQUIRE(raw.create(directory.path()));
    REQUIRE(raw.append(input));
    const auto blocked = directory.path() / "not-a-directory";
    { std::ofstream file(blocked); file << "occupied"; REQUIRE(file.good()); }
    std::string error;
    CHECK_FALSE(bounded_pipeline::process_f0(raw, blocked, {}, processed, error));
    CHECK_FALSE(error.empty());
    require_spool_equals(raw, input, 0.0f);
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

TEST_CASE("Bounded global RMS writer matches the centered envelope and PCM rule",
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
    const auto expected_float = centered_rms(source, output, 0.35);

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

// Independent librosa 0.11 / torch upstream change_rms values; generator and
// provenance in dynamics-20261026/operator-goldens.json, not a C++ self-reference.
TEST_CASE("RMS matches independent centered half-pixel reference", "[bounded-pipeline][dsp][rms][contract]") {
    TempDirectory directory;
    std::vector<float> source(32003), output(80007);
    constexpr float levels1[]={.05f,.4f,.1f,.25f},levels2[]={.3f,.1f,.2f};
    for(std::size_t i=0;i<source.size();++i)source[i]=levels1[(i/4000)%4];
    for(std::size_t i=0;i<output.size();++i)output[i]=levels2[(i/7000)%3];
    constexpr std::array<std::size_t,13> points={0,1,799,800,7999,8000,16000,32767,32768,65535,65536,79999,80006};
    constexpr std::array<std::array<float,13>,3> expected={{
    {{3.944427669e-01f,3.944427669e-01f,3.944427669e-01f,3.944427669e-01f,1.314809173e-01f,1.314809173e-01f,2.399103194e-01f,1.118033901e-01f,1.118033901e-01f,2.986869514e-01f,2.986822724e-01f,1.756208092e-01f,1.756208092e-01f}},
    {{3.439953923e-01f,3.439953923e-01f,3.439953923e-01f,3.439953923e-01f,1.146651283e-01f,1.146651283e-01f,2.190480977e-01f,1.057371274e-01f,1.057371274e-01f,2.993427515e-01f,2.993404269e-01f,1.874144077e-01f,1.874144077e-01f}},
    {{3.000000119e-01f,3.000000119e-01f,3.000000119e-01f,3.000000119e-01f,1.000000015e-01f,1.000000015e-01f,2.000000030e-01f,1.000000015e-01f,1.000000015e-01f,3.000000119e-01f,3.000000119e-01f,2.000000030e-01f,2.000000030e-01f}}
    }};
    for(int mode=0;mode<3;++mode) {
        const double rate=mode*.5;
        bounded_io::RawFloatSpool a,b;
        REQUIRE(a.create(directory.path()));REQUIRE(b.create(directory.path()));
        REQUIRE(a.append(source));REQUIRE(b.append(output));
        const auto path=directory.path()/("independent-"+std::to_string(mode)+".wav");
        bounded_io::AtomicWavStreamWriter writer;
        REQUIRE(writer.open(path,40000,output.size()));std::string error;
        REQUIRE(bounded_pipeline::write_global_rms_output(a,b,directory.path(),rate,writer,error));
        REQUIRE(writer.commit());const auto actual=voxmutatio::io::read_audio(path.string(),40000);
        REQUIRE(actual);REQUIRE(actual->data.size()==output.size());
        for(std::size_t i=0;i<points.size();++i) {
            INFO("rate="<<rate<<" sample="<<points[i]);
            const auto pcm=static_cast<std::int16_t>(std::clamp(expected[mode][i],-1.f,1.f)*32767.f);
            CHECK(std::abs(actual->data[points[i]]-float(pcm)/32768.f)<=2.f/32768.f);
        }
    }
}

TEST_CASE("Manual pitch corrections preserve endpoints, UV and unrelated frames", "[bounded][pitch]") {
    TempDirectory directory;
    std::vector<float> f0(800, 100.0f);
    f0[301] = 400; f0[500] = 0;
    bounded_io::RawFloatSpool raw, processed;
    REQUIRE(raw.create(directory.path())); REQUIRE(raw.append(f0));
    voxmutatio::VCConfig config; config.filter_radius = 0;
    REQUIRE(voxmutatio::parse_pitch_edits("smooth:1:1.12;shift:2:2.06:12", config.pitch_edits));
    std::string error;
    REQUIRE(bounded_pipeline::process_f0(raw, directory.path(), config, processed, error, 2, 4));
    std::vector<float> output; REQUIRE(processed.read(0, f0.size(), output));
    for (size_t i = 0; i < f0.size(); ++i) {
        if (i > 300 && i < 312) CHECK(std::abs(output[i] - 100) < 0.001f);
        else if (i >= 400 && i < 406) CHECK(output[i] == 200);
        else CHECK(output[i] == f0[i]);
    }
    for (const auto text : {"smooth:3:3.12", "smooth:3.95:4.1"}) {
        REQUIRE(voxmutatio::parse_pitch_edits(text, config.pitch_edits));
        CHECK_FALSE(bounded_pipeline::process_f0(raw, directory.path(), config, processed, error, 2, 4));
    }
    for (const auto text : {"smooth:nan:1", "shift:0:1:25", "smooth:1:0", "smooth:0:3",
                            "smooth:0:1;smooth:0.5:1.5", "shift:0:1:12;", "smooth:0:1:2"})
        CHECK_FALSE(voxmutatio::parse_pitch_edits(text, config.pitch_edits));
}

TEST_CASE("pitch review curve matches the actual bounded postprocessor exactly", "[bounded][pitch]") {
    TempDirectory directory;std::vector<float> f0(1401,200);
    for(std::size_t i=0;i<f0.size();++i)f0[i]=(i%71<5)?0.f:static_cast<float>(150+30*std::sin(i*.17));
    bounded_io::RawFloatSpool raw;REQUIRE(raw.create(directory.path()));REQUIRE(raw.append(f0));
    for(int filter:{0,3,5,31})for(int pitch:{-12,0,7}) {
        voxmutatio::VCConfig c;c.filter_radius=filter;c.f0_up_key=pitch;
        c.pitch_edits={{voxmutatio::PitchEditMode::kShift,3,3.18,12}};
        bounded_io::RawFloatSpool result;std::string error;
        REQUIRE(bounded_pipeline::process_f0(raw,directory.path(),c,result,error,2,10));
        std::vector<float> values;REQUIRE(result.read(0,result.sample_count(),values));
        CHECK(values==voxmutatio::f0::prepare_pitch_curve(f0,filter,pitch,c.pitch_edits,2,10));
    }
}
