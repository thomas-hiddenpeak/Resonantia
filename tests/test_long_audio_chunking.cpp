/**
 * @file test_long_audio_chunking.cpp
 * @brief Long-audio chunked inference must match the whole-clip pass.
 *
 * The production pipeline splits clips longer than `max_chunk_sec` at silence,
 * converts each chunk with context padding, and stitches the result. This test
 * tiles a real speech fixture to roughly 18 seconds, past the configured chunk
 * threshold, and asserts the chunked output is numerically consistent with a
 * single whole-clip pass. It does not establish five-minute resource bounds.
 */
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "voxmutatio/pipeline/pipeline.h"
#include "voxmutatio/io/audio_io.h"
#include "pipeline/bounded_pipeline_detail.h"
#include "subprocess_test.h"
#include "test_artifacts.h"

namespace {

double correlation(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = std::min(a.size(), b.size());
    if (n == 0) return 0.0;
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double c = 0, va = 0, vb = 0;
    for (size_t i = 0; i < n; ++i) {
        double da = a[i] - ma, db = b[i] - mb;
        c += da * db; va += da * da; vb += db * db;
    }
    return c / (std::sqrt(va * vb) + 1e-12);
}

// In-place iterative radix-2 FFT.
void fft(std::vector<std::complex<float>>& a) {
    int n = static_cast<int>(a.size());
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * static_cast<float>(M_PI) / len;
        std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int k = 0; k < len / 2; ++k) {
                std::complex<float> u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Phase-invariant descriptor: Hann-windowed magnitude spectrogram, flattened.
// NSF vocoders reset harmonic phase per inference call, so chunked and whole-clip
// outputs are perceptually identical but waveform-decorrelated; magnitude removes
// that phase and lets us verify spectral content + coarse time alignment.
std::vector<float> mag_spectrogram(const std::vector<float>& x, int fftn, int hop) {
    if (x.size() < static_cast<size_t>(fftn)) return {};
    int nf = (static_cast<int>(x.size()) - fftn) / hop + 1;
    std::vector<float> win(fftn);
    for (int i = 0; i < fftn; ++i)
        win[i] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * i / (fftn - 1));
    std::vector<float> out;
    out.reserve(static_cast<size_t>(nf) * (fftn / 2 + 1));
    std::vector<std::complex<float>> buf(fftn);
    for (int f = 0; f < nf; ++f) {
        int base = f * hop;
        for (int i = 0; i < fftn; ++i) buf[i] = {x[base + i] * win[i], 0.0f};
        fft(buf);
        for (int k = 0; k <= fftn / 2; ++k) out.push_back(std::abs(buf[k]));
    }
    return out;
}

voxmutatio::VCConfig base_config() {
    using namespace voxmutatio;
    VCConfig cfg;
    cfg.hubert_model_path = "../models/hubert_base/model.safetensors";
    cfg.rmvpe_model_path = "../models/rmvpe.safetensors";
    cfg.synthesizer_model_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    cfg.version = ModelVersion::kV2;
    cfg.has_f0 = true;
    cfg.model_sample_rate = 40000;
    cfg.num_speakers = 109;
    cfg.rms_mix_rate = 1.0;   // isolate chunking (no RMS envelope matching)
    cfg.protect = 0.5;        // protect disabled (>=0.5)
    cfg.apply_highpass = true;
    cfg.interp_unvoiced = true;
    cfg.filter_radius = 3;
    return cfg;
}

struct ResourceMetrics {
    std::uint64_t input_samples = 0;
    std::uint64_t device_total_bytes = 0;
    std::uint64_t device_baseline_free_bytes = 0;
    std::uint64_t device_minimum_free_bytes = 0;

    [[nodiscard]] std::uint64_t peak_device_growth_bytes() const noexcept {
        return device_baseline_free_bytes > device_minimum_free_bytes
            ? device_baseline_free_bytes - device_minimum_free_bytes : 0;
    }
};

class DeviceMemoryObserver {
public:
    DeviceMemoryObserver() {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        valid_ = cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
        if (!valid_) return;
        total_bytes_ = total_bytes;
        baseline_free_bytes_ = free_bytes;
        minimum_free_bytes_.store(free_bytes, std::memory_order_relaxed);
        worker_ = std::thread([this] {
            while (!stop_.load(std::memory_order_relaxed)) {
                sample();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            sample();
        });
    }

    DeviceMemoryObserver(const DeviceMemoryObserver&) = delete;
    DeviceMemoryObserver& operator=(const DeviceMemoryObserver&) = delete;

    ~DeviceMemoryObserver() { stop(); }

    [[nodiscard]] bool valid() const noexcept { return valid_; }

    ResourceMetrics stop() noexcept {
        stop_.store(true, std::memory_order_relaxed);
        if (worker_.joinable()) worker_.join();
        ResourceMetrics metrics;
        metrics.device_total_bytes = total_bytes_;
        metrics.device_baseline_free_bytes = baseline_free_bytes_;
        metrics.device_minimum_free_bytes =
            minimum_free_bytes_.load(std::memory_order_relaxed);
        return metrics;
    }

private:
    void sample() noexcept {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return;
        std::size_t minimum = minimum_free_bytes_.load(std::memory_order_relaxed);
        while (free_bytes < minimum &&
               !minimum_free_bytes_.compare_exchange_weak(
                   minimum, free_bytes, std::memory_order_relaxed)) {}
    }

    bool valid_ = false;
    std::size_t total_bytes_ = 0;
    std::size_t baseline_free_bytes_ = 0;
    std::atomic<std::size_t> minimum_free_bytes_{0};
    std::atomic<bool> stop_{false};
    std::thread worker_;
};

void run_bounded_resource_helper(int repetitions) {
    using namespace voxmutatio;
    auto fixture = io::read_audio("../tests/fixtures/speech_librispeech.wav", 16'000);
    REQUIRE(fixture.has_value());
    REQUIRE(fixture->data.size() <=
            std::numeric_limits<std::uint64_t>::max() /
                static_cast<std::uint64_t>(repetitions));
    const std::uint64_t input_samples =
        fixture->data.size() * static_cast<std::uint64_t>(repetitions);

    resonantia_test::ScopedArtifact input(
        resonantia_test::unique_artifact_path("bounded-resource-input", ".wav"));
    resonantia_test::ScopedArtifact output(
        resonantia_test::unique_artifact_path("bounded-resource-output", ".wav"));
    // Stream the tiled fixture so the child resource high-water mark does not
    // itself contain an O(duration) test-construction AudioBuffer.
    io::detail::AtomicWavStreamWriter input_writer;
    REQUIRE(input_writer.open(input.path(), 16'000, input_samples));
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        REQUIRE(input_writer.append(fixture->data));
    }
    REQUIRE(input_writer.commit());

    DeviceMemoryObserver observer;
    REQUIRE(observer.valid());
    VCConfig config = base_config();
    config.max_chunk_sec = 6.0;
    config.chunk_pad_sec = 2.0;
    config.chunk_query_sec = 1.0;
    config.rms_mix_rate = 0.5;
    pipeline::VoiceConversionPipeline pipeline;
    REQUIRE(pipeline.init(config));
    const auto result = pipeline.convert_file_bounded(
        input.path().string(), output.path().string(), 0);
    REQUIRE(result.success);
    auto metrics = observer.stop();
    metrics.input_samples = input_samples;
    REQUIRE(metrics.device_total_bytes > 0);
    REQUIRE(metrics.device_baseline_free_bytes >= metrics.device_minimum_free_bytes);

    const char* result_path = std::getenv("RESONANTIA_SUBPROCESS_RESULT_PATH");
    REQUIRE(result_path != nullptr);
    std::ofstream stream(result_path, std::ios::trunc);
    REQUIRE(stream << metrics.input_samples << ' '
                   << metrics.device_total_bytes << ' '
                   << metrics.device_baseline_free_bytes << ' '
                   << metrics.device_minimum_free_bytes << '\n');
}

bool read_resource_metrics(const std::filesystem::path& path,
                           ResourceMetrics& metrics) {
    std::ifstream stream(path);
    return static_cast<bool>(
        stream >> metrics.input_samples
               >> metrics.device_total_bytes
               >> metrics.device_baseline_free_bytes
               >> metrics.device_minimum_free_bytes);
}

void require_bounded_matches_legacy(voxmutatio::VCConfig config,
                                    int repetitions,
                                    const char* artifact_stem) {
    using namespace voxmutatio;
    auto fixture = io::read_audio("../tests/fixtures/speech_librispeech.wav", 16'000);
    REQUIRE(fixture.has_value());
    AudioBuffer clip;
    clip.sample_rate = 16'000;
    clip.data.reserve(fixture->data.size() * static_cast<std::size_t>(repetitions));
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        clip.data.insert(clip.data.end(), fixture->data.begin(), fixture->data.end());
    }

    resonantia_test::ScopedArtifact input(
        resonantia_test::unique_artifact_path(artifact_stem, "-input.wav"));
    resonantia_test::ScopedArtifact output(
        resonantia_test::unique_artifact_path(artifact_stem, "-output.wav"));
    REQUIRE(io::write_audio(input.path().string(), clip.data.data(),
                            static_cast<int>(clip.data.size()), clip.sample_rate));
    auto persisted = io::read_audio(input.path().string(), 16'000);
    REQUIRE(persisted.has_value());

    pipeline::VoiceConversionPipeline bounded_pipeline;
    REQUIRE(bounded_pipeline.init(config));
    const auto bounded = bounded_pipeline.convert_file_bounded(
        input.path().string(), output.path().string(), 0);
    INFO(bounded.error_message);
    REQUIRE(bounded.success);
    REQUIRE(bounded.audio.data.empty());
    REQUIRE(bounded.audio.sample_rate == 40'000);
    auto bounded_audio = io::read_audio(output.path().string(), 40'000);
    REQUIRE(bounded_audio.has_value());
    REQUIRE(!bounded_audio->data.empty());
    REQUIRE(std::all_of(
        bounded_audio->data.begin(), bounded_audio->data.end(),
        [](float value) { return std::isfinite(value); }));

    pipeline::VoiceConversionPipeline legacy_pipeline;
    REQUIRE(legacy_pipeline.init(config));
    const auto legacy = legacy_pipeline.convert_buffer(*persisted, 0);
    INFO(legacy.error_message);
    REQUIRE(legacy.success);
    REQUIRE(!legacy.audio.data.empty());
    const double length_ratio = static_cast<double>(bounded_audio->data.size()) /
                                static_cast<double>(legacy.audio.data.size());
    CHECK(length_ratio > 0.98);
    CHECK(length_ratio < 1.02);
    const auto bounded_spectrum = mag_spectrogram(
        bounded_audio->data, 1024, 512);
    const auto legacy_spectrum = mag_spectrogram(legacy.audio.data, 1024, 512);
    const double spectrum_correlation = correlation(
        bounded_spectrum, legacy_spectrum);
    std::printf("bounded_reference version=%d f0=%d length_ratio=%.8f "
                "spectrum_correlation=%.8f\n",
                static_cast<int>(config.version),
                static_cast<int>(config.f0_method),
                length_ratio, spectrum_correlation);
    CHECK(spectrum_correlation > 0.98);
}

}  // namespace

TEST_CASE("Bounded offline file conversion is additive and keeps the buffer reference",
          "[bounded-file][chunk][integration]") {
    using namespace voxmutatio;
    using Pipeline = pipeline::VoiceConversionPipeline;

    auto fixture = io::read_audio("../tests/fixtures/speech_librispeech.wav", 16000);
    REQUIRE(fixture.has_value());

    AudioBuffer clip;
    clip.sample_rate = 16000;
    for (int rep = 0; rep < 3; ++rep)
        clip.data.insert(clip.data.end(), fixture->data.begin(), fixture->data.end());

    resonantia_test::ScopedArtifact input_path(
        resonantia_test::unique_artifact_path("bounded-long-input", ".wav"));
    REQUIRE(io::write_audio(input_path.path().string(), clip.data.data(),
                            static_cast<int>(clip.data.size()), clip.sample_rate));

    // Use the decoded file samples for both paths so PCM16 input quantization is
    // not mistaken for a windowing difference.
    auto persisted = io::read_audio(input_path.path().string(), 16000);
    REQUIRE(persisted.has_value());
    resonantia_test::ScopedArtifact legacy_input_path(
        resonantia_test::unique_artifact_path("legacy-long-input", ".wav"));
    resonantia_test::ScopedArtifact legacy_output_path(
        resonantia_test::unique_artifact_path("legacy-long-output", ".wav"));
    REQUIRE(io::write_audio(
        legacy_input_path.path().string(), persisted->data.data(),
        static_cast<int>(persisted->data.size()), persisted->sample_rate));

    VCConfig cfg = base_config();
    cfg.max_chunk_sec = 6.0;
    cfg.chunk_pad_sec = 2.0;
    cfg.chunk_query_sec = 1.0;
    cfg.rms_mix_rate = 0.5;  // exercise the global source/output RMS pass

    Pipeline bounded_pipe;
    REQUIRE(bounded_pipe.init(cfg));
    // The bounded reader snapshots and closes the source before commit, so a
    // successful in-place conversion is a supported atomic replacement.
    const auto bounded = bounded_pipe.convert_file_bounded(
        input_path.path().string(), input_path.path().string(), 0);
    REQUIRE(bounded.success);
    CHECK(bounded.audio.data.empty());
    CHECK(bounded.audio.sample_rate == 40000);

    auto bounded_audio = io::read_audio(input_path.path().string(), 40000);
    REQUIRE(bounded_audio.has_value());
    REQUIRE(!bounded_audio->data.empty());
    for (float value : bounded_audio->data) REQUIRE(std::isfinite(value));

    Pipeline legacy_pipe;
    REQUIRE(legacy_pipe.init(cfg));
    const auto legacy = legacy_pipe.convert_file(
        legacy_input_path.path().string(), legacy_output_path.path().string(), 0);
    REQUIRE(legacy.success);
    REQUIRE(!legacy.audio.data.empty());
    REQUIRE(legacy.audio.sample_rate == 40'000);
    auto legacy_file = io::read_audio(legacy_output_path.path().string(), 40'000);
    REQUIRE(legacy_file.has_value());
    REQUIRE(legacy_file->data.size() == legacy.audio.data.size());
    float legacy_pcm_error = 0.0f;
    for (std::size_t index = 0; index < legacy_file->data.size(); ++index) {
        const float clamped = std::clamp(legacy.audio.data[index], -1.0f, 1.0f);
        const auto pcm = static_cast<std::int16_t>(clamped * 32767.0f);
        const float expected = static_cast<float>(pcm) / 32768.0f;
        legacy_pcm_error = std::max(
            legacy_pcm_error, std::abs(legacy_file->data[index] - expected));
    }
    INFO("legacy_pcm_error=" << legacy_pcm_error);
    CHECK(legacy_pcm_error == 0.0f);

    const double len_ratio = static_cast<double>(bounded_audio->data.size()) /
                             static_cast<double>(legacy.audio.data.size());
    CHECK(len_ratio > 0.98);
    CHECK(len_ratio < 1.02);

    const auto bounded_spec = mag_spectrogram(bounded_audio->data, 1024, 512);
    const auto legacy_spec = mag_spectrogram(legacy.audio.data, 1024, 512);
    const double bounded_spec_correlation = correlation(bounded_spec, legacy_spec);
    std::printf("bounded_same_path_rmvpe_v2 length_ratio=%.8f "
                "spectrum_correlation=%.8f legacy_pcm_error=%.8f\n",
                len_ratio, bounded_spec_correlation, legacy_pcm_error);
    CHECK(bounded_spec_correlation > 0.98);
}

TEST_CASE("Long-audio chunking matches whole-clip pass", "[chunk][integration]") {
    using namespace voxmutatio;

    auto fixture = io::read_audio("../tests/fixtures/speech_librispeech.wav", 16000);
    REQUIRE(fixture.has_value());

    // Tile the fixture past the chunk threshold (~18s -> 3 chunks at 6s cuts).
    AudioBuffer clip;
    clip.sample_rate = 16000;
    for (int rep = 0; rep < 3; ++rep)
        clip.data.insert(clip.data.end(), fixture->data.begin(), fixture->data.end());
    const double clip_sec = static_cast<double>(clip.data.size()) / 16000.0;
    REQUIRE(clip_sec > 12.0);

    // A) Chunked path: force multiple silence-cut segments with 2s context pad.
    VCConfig chunk_cfg = base_config();
    chunk_cfg.max_chunk_sec = 6.0;
    chunk_cfg.chunk_pad_sec = 2.0;
    chunk_cfg.chunk_query_sec = 1.0;
    pipeline::VoiceConversionPipeline chunk_pipe;
    REQUIRE(chunk_pipe.init(chunk_cfg));
    auto chunked = chunk_pipe.convert_buffer(clip, /*speaker_id=*/0);
    REQUIRE(chunked.success);
    REQUIRE(!chunked.audio.data.empty());
    for (float v : chunked.audio.data) { REQUIRE(!std::isnan(v)); REQUIRE(!std::isinf(v)); }

    // B) Reference whole-clip pass: keep the valid chunk threshold above this
    // fixture's duration, and match the 2s outer reflect pad.
    VCConfig whole_cfg = base_config();
    whole_cfg.max_chunk_sec = 30.0;
    whole_cfg.edge_pad_sec = 2.0;    // same outer context as the chunk path
    pipeline::VoiceConversionPipeline whole_pipe;
    REQUIRE(whole_pipe.init(whole_cfg));
    auto whole = whole_pipe.convert_buffer(clip, /*speaker_id=*/0);
    REQUIRE(whole.success);
    REQUIRE(!whole.audio.data.empty());

    // Lengths should match within a few percent (chunk boundaries quantize to frames).
    double len_ratio = static_cast<double>(chunked.audio.data.size()) /
                       static_cast<double>(whole.audio.data.size());
    CHECK(len_ratio > 0.98);
    CHECK(len_ratio < 1.02);

    // Phase-invariant spectral match: the chunked output must reproduce the
    // whole-clip spectrum (content + coarse time alignment). Raw-waveform corr
    // is expected to be low because the NSF vocoder resets harmonic phase per
    // chunk, so it is reported for information only. The residual spectral gap
    // is inherent: HuBERT uses global self-attention, so a chunk's content
    // features legitimately differ from a whole pass (RVC chunks HuBERT too).
    auto sc = mag_spectrogram(chunked.audio.data, 1024, 512);
    auto sw = mag_spectrogram(whole.audio.data, 1024, 512);
    double spec_corr = correlation(sc, sw);
    double wave_corr = correlation(chunked.audio.data, whole.audio.data);
    std::printf("spec_corr=%.4f  wave_corr=%.4f  (chunk=%zu whole=%zu)\n",
                spec_corr, wave_corr, chunked.audio.data.size(), whole.audio.data.size());
    CHECK(spec_corr > 0.98);
}

TEST_CASE("FCPE v1 bounded file output matches the legacy buffer reference",
          "[.][bounded-file][fcpe][v1][integration]") {
    using namespace voxmutatio;
    VCConfig config = base_config();
    config.version = ModelVersion::kV1;
    config.synthesizer_model_path =
        "../models/pretrained/pretrained/f0G40k.safetensors";
    config.f0_method = F0Method::kFcpe;
    config.fcpe_model_path = "../models/fcpe/model.safetensors";
    config.max_chunk_sec = 6.0;
    config.chunk_pad_sec = 2.0;
    config.chunk_query_sec = 1.0;
    config.rms_mix_rate = 0.5;
    require_bounded_matches_legacy(config, 3, "bounded-fcpe-v1");
}

TEST_CASE("WavLM bounded profile fails before payload or staging I/O",
          "[.][bounded-file][wavlm][contract][integration]") {
    using namespace voxmutatio;
    VCConfig config = base_config();
    config.content_encoder = ContentEncoder::kWavlm;
    config.wavlm_model_path = "../models/wavlm/model.safetensors";
    pipeline::VoiceConversionPipeline pipeline;
    REQUIRE(pipeline.init(config));

    const auto directory_path =
        resonantia_test::unique_artifact_path("bounded-wavlm-preflight", "");
    resonantia_test::ScopedArtifact directory(directory_path);
    REQUIRE(std::filesystem::create_directory(directory_path));
    resonantia_test::ScopedArtifact target(directory_path / "old-target.wav");
    const std::array<char, 10> sentinel = {
        'o', 'l', 'd', '-', 't', 'a', 'r', 'g', 'e', 't'};
    {
        std::ofstream stream(target.path(), std::ios::binary | std::ios::trunc);
        REQUIRE(stream.write(sentinel.data(), sentinel.size()));
    }
    const auto result = pipeline.convert_file_bounded(
        (directory_path / "missing-input.wav").string(),
        target.path().string(), 0);
    CHECK_FALSE(result.success);
    CHECK(result.error_message == "bounded WavLM profile is not qualified");
    std::ifstream persisted(target.path(), std::ios::binary);
    const std::vector<char> bytes{
        std::istreambuf_iterator<char>(persisted),
        std::istreambuf_iterator<char>()};
    CHECK(bytes == std::vector<char>(sentinel.begin(), sentinel.end()));
    std::vector<std::filesystem::path> entries;
    for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
        entries.push_back(entry.path().filename());
    }
    REQUIRE(entries.size() == 1);
    CHECK(entries.front() == target.path().filename());
}

TEST_CASE("Bounded resource short subprocess helper",
          "[.][bounded-resource-helper]") {
    run_bounded_resource_helper(3);
}

TEST_CASE("Bounded resource long subprocess helper",
          "[.][bounded-resource-helper]") {
    run_bounded_resource_helper(6);
}

TEST_CASE("Bounded file working set is duration independent",
          "[.][bounded-resource][integration]") {
    resonantia_test::ScopedArtifact short_metrics_path(
        resonantia_test::unique_artifact_path("bounded-resource-short", ".txt"));
    resonantia_test::ScopedArtifact long_metrics_path(
        resonantia_test::unique_artifact_path("bounded-resource-long", ".txt"));
    const auto short_run = resonantia_test::run_self_test(
        "Bounded resource short subprocess helper", 600'000,
        short_metrics_path.path().c_str());
    INFO("short launch_error=" << short_run.launch_error
         << " wait_error=" << short_run.wait_error
         << " timed_out=" << short_run.timed_out
         << " status=" << short_run.status);
    REQUIRE(short_run.exited_with(0));
    const auto long_run = resonantia_test::run_self_test(
        "Bounded resource long subprocess helper", 600'000,
        long_metrics_path.path().c_str());
    INFO("long launch_error=" << long_run.launch_error
         << " wait_error=" << long_run.wait_error
         << " timed_out=" << long_run.timed_out
         << " status=" << long_run.status);
    REQUIRE(long_run.exited_with(0));

    ResourceMetrics short_metrics;
    ResourceMetrics long_metrics;
    REQUIRE(read_resource_metrics(short_metrics_path.path(), short_metrics));
    REQUIRE(read_resource_metrics(long_metrics_path.path(), long_metrics));
    REQUIRE(short_run.max_rss_kib > 0);
    REQUIRE(long_run.max_rss_kib > 0);
    REQUIRE(long_metrics.input_samples > short_metrics.input_samples);

    constexpr std::uint64_t kHostSlackKib = 64 * 1024;
    constexpr std::uint64_t kDeviceSlackBytes = 256ULL * 1024 * 1024;
    constexpr std::uint64_t kRmvpeF0InputHardBound =
        voxmutatio::pipeline::detail::kF0CoreSamples +
        2 * voxmutatio::pipeline::detail::kRmvpeContextSamples;
    const auto short_device_peak = short_metrics.peak_device_growth_bytes();
    const auto long_device_peak = long_metrics.peak_device_growth_bytes();
    int device = -1;
    cudaDeviceProp device_properties{};
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    REQUIRE(cudaGetDeviceProperties(&device_properties, device) == cudaSuccess);
#ifdef NDEBUG
    constexpr const char* kBuildMode = "Release";
#else
    constexpr const char* kBuildMode = "Debug";
#endif
    std::printf(
        "bounded_resource short_samples=%llu long_samples=%llu "
        "short_peak_rss_kib=%ld long_peak_rss_kib=%ld "
        "short_device_growth_bytes=%llu long_device_growth_bytes=%llu "
        "device_total_bytes=%llu device_name=\"%s\" compute_capability=%d.%d "
        "build_mode=%s rmvpe_f0_input_hard_bound=%llu host_slack_kib=%llu "
        "device_slack_bytes=%llu\n",
        static_cast<unsigned long long>(short_metrics.input_samples),
        static_cast<unsigned long long>(long_metrics.input_samples),
        short_run.max_rss_kib, long_run.max_rss_kib,
        static_cast<unsigned long long>(short_device_peak),
        static_cast<unsigned long long>(long_device_peak),
        static_cast<unsigned long long>(long_metrics.device_total_bytes),
        device_properties.name, device_properties.major, device_properties.minor,
        kBuildMode,
        static_cast<unsigned long long>(kRmvpeF0InputHardBound),
        static_cast<unsigned long long>(kHostSlackKib),
        static_cast<unsigned long long>(kDeviceSlackBytes));
    CHECK(static_cast<std::uint64_t>(long_run.max_rss_kib) <=
          static_cast<std::uint64_t>(short_run.max_rss_kib) + kHostSlackKib);
    CHECK(long_device_peak <= short_device_peak + kDeviceSlackBytes);
}
