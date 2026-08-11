#include <catch2/catch_test_macros.hpp>

#include "tools/realtime/benchmark_report.h"

#include <algorithm>
#include <array>
#include <string>

namespace benchmark = resonantia::realtime::benchmark;

namespace {

benchmark::LatencySummary valid_latency(double base_ms = 1.0) {
    return {
        .count = 100,
        .p50_ms = base_ms,
        .p95_ms = base_ms + 1.0,
        .p99_ms = base_ms + 2.0,
        .max_ms = base_ms + 3.0,
    };
}

benchmark::BenchmarkReport valid_report() {
    benchmark::BenchmarkReport report;
    report.schema_version = benchmark::kBenchmarkReportSchemaVersion;
    report.build = benchmark::BuildMetadata{
        .revision = "0123456789abcdef0123456789abcdef01234567",
        .tree_state = "dirty",
        .build_type = "Release",
        .compiler = "GNU 13.2.0",
        .cuda_version = "13.2.78",
        .compile_flags = "-O3 -DNDEBUG",
    };
    report.environment = benchmark::EnvironmentMetadata{
        .machine = "NVIDIA Jetson Orin NX",
        .os = "Linux 6.8.12-1021-tegra",
        .architecture = "aarch64 sm_87",
        .power_mode = "MAXN",
        .clocks = "locked",
    };
    report.profile = benchmark::ProfileMetadata{
        .name = "orin-v2-hubert-fcpe-40k-indexed",
        .input_sample_rate = 48'000,
        .output_sample_rate = 40'000,
        .window_ms = 200,
        .emit_hop_ms = 20,
        .lookahead_ms = 40,
        .measured_duration_seconds = 600.0,
    };
    report.artifacts = benchmark::ArtifactHashes{
        .input_audio_sha256 = std::string(64, '1'),
        .hubert_sha256 = std::string(64, '2'),
        .f0_sha256 = std::string(64, '3'),
        .synthesizer_sha256 = std::string(64, '4'),
    };
    report.index = benchmark::IndexMetadata{
        .sha256 = std::string(64, '5'),
        .rows = 10'000,
        .dimension = 768,
        .k = 8,
        .index_rate = 0.75,
        .query_batch = 2,
    };
    report.warmup = benchmark::WarmupMetadata{
        .duration_seconds = 5.0,
        .chunks = 250,
    };
    const std::array names{
        "queue_wait", "hubert", "f0", "index", "synth", "gpu_total",
        "output_queue_device", "capture_to_playback",
    };
    for (std::size_t i = 0; i < names.size(); ++i) {
        report.stages.push_back({names[i], valid_latency(static_cast<double>(i + 1))});
    }
    report.retrieval = benchmark::RetrievalMetrics{
        .mode = "index_enabled",
        .fallback_policy = "disabled",
        .fallback_triggered = false,
        .fallback_reason = "none",
        .index_queries = 30'000,
        .index_enabled_hops = 30'000,
        .index_bypassed_hops = 0,
        .fallback_transitions = 0,
    };
    report.counters = benchmark::RunCounters{
        .input_hops_dropped = 0,
        .output_hops_concealed = 0,
        .capture_overruns = 0,
        .playback_underruns = 0,
        .alsa_xruns = 0,
        .alsa_recoveries = 0,
    };
    report.resources = benchmark::ResourceMetrics{
        .peak_rss_bytes = 900'000'000,
        .steady_rss_bytes = 850'000'000,
        .peak_cuda_allocated_bytes = 700'000'000,
        .steady_cuda_allocated_bytes = 650'000'000,
        .system_memory_delta_bytes = 800'000'000,
        .thermal_start_celsius = 48.0,
        .thermal_peak_celsius = 65.0,
        .thermal_samples = 600,
        .thermal_throttled = false,
    };
    return report;
}

bool has_error(const benchmark::ValidationResult& result, const std::string& field) {
    return std::any_of(result.errors.begin(), result.errors.end(), [&](const auto& error) {
        return error.find(field) != std::string::npos;
    });
}

}  // namespace

TEST_CASE("A complete realtime benchmark report satisfies the internal schema",
          "[realtime][benchmark][contract]") {
    CHECK(benchmark::validate_benchmark_report(valid_report()).ok());
}

TEST_CASE("Missing benchmark provenance and profile fields fail validation",
          "[realtime][benchmark][contract]") {
    auto report = valid_report();
    report.build.reset();
    report.environment.reset();
    report.profile->emit_hop_ms.reset();
    report.artifacts->hubert_sha256.clear();

    const auto result = benchmark::validate_benchmark_report(report);
    CHECK_FALSE(result.ok());
    CHECK(has_error(result, "build"));
    CHECK(has_error(result, "environment"));
    CHECK(has_error(result, "profile.emit_hop_ms"));
    CHECK(has_error(result, "artifacts.hubert_sha256"));
}

TEST_CASE("Index identity and query envelope are mandatory and consistent",
          "[realtime][benchmark][contract]") {
    auto report = valid_report();
    report.index->sha256.clear();
    report.index->rows.reset();
    report.index->dimension.reset();
    report.index->k = 20'000;
    report.index->index_rate.reset();
    report.index->query_batch.reset();

    const auto result = benchmark::validate_benchmark_report(report);
    CHECK_FALSE(result.ok());
    CHECK(has_error(result, "index.sha256"));
    CHECK(has_error(result, "index.rows"));
    CHECK(has_error(result, "index.dimension"));
    CHECK(has_error(result, "index.k"));
    CHECK(has_error(result, "index.index_rate"));
    CHECK(has_error(result, "index.query_batch"));
}

TEST_CASE("Warmup and every required latency percentile are mandatory",
          "[realtime][benchmark][contract]") {
    auto report = valid_report();
    report.warmup.reset();
    report.stages.erase(report.stages.begin() + 3);  // index
    report.stages.front().latency.p99_ms.reset();
    report.stages[1].latency.p50_ms = 5.0;
    report.stages[1].latency.p95_ms = 4.0;

    const auto result = benchmark::validate_benchmark_report(report);
    CHECK_FALSE(result.ok());
    CHECK(has_error(result, "warmup"));
    CHECK(has_error(result, "stages.index"));
    CHECK(has_error(result, "stages.queue_wait.p99_ms"));
    CHECK(has_error(result, "stages.hubert.percentiles"));
}

TEST_CASE("Retrieval, drop, xrun, memory and thermal evidence cannot be omitted",
          "[realtime][benchmark][contract]") {
    auto report = valid_report();
    report.retrieval->fallback_triggered.reset();
    report.counters->input_hops_dropped.reset();
    report.counters->alsa_xruns.reset();
    report.resources->peak_rss_bytes.reset();
    report.resources->peak_cuda_allocated_bytes.reset();
    report.resources->thermal_peak_celsius.reset();

    const auto result = benchmark::validate_benchmark_report(report);
    CHECK_FALSE(result.ok());
    CHECK(has_error(result, "retrieval.fallback_triggered"));
    CHECK(has_error(result, "counters.input_hops_dropped"));
    CHECK(has_error(result, "counters.alsa_xruns"));
    CHECK(has_error(result, "resources.peak_rss_bytes"));
    CHECK(has_error(result, "resources.peak_cuda_allocated_bytes"));
    CHECK(has_error(result, "resources.thermal_peak_celsius"));
}

TEST_CASE("No-index and sticky fallback reports remain explicit",
          "[realtime][benchmark][contract]") {
    SECTION("index-enabled run cannot report a fallback reason") {
        auto report = valid_report();
        report.retrieval->fallback_reason = "sustained_index_deadline";

        const auto result = benchmark::validate_benchmark_report(report);
        CHECK_FALSE(result.ok());
        CHECK(has_error(result, "retrieval.fallback_reason"));
    }

    SECTION("explicit no-index diagnostic") {
        auto report = valid_report();
        report.index.reset();
        report.retrieval->mode = "no_index_diagnostic";
        report.retrieval->index_queries = 0;
        report.retrieval->index_enabled_hops = 0;
        report.retrieval->index_bypassed_hops = 30'000;
        CHECK(benchmark::validate_benchmark_report(report).ok());
    }

    SECTION("no-index diagnostic cannot report a fallback reason") {
        auto report = valid_report();
        report.index.reset();
        report.retrieval->mode = "no_index_diagnostic";
        report.retrieval->fallback_reason = "sustained_index_deadline";
        report.retrieval->index_queries = 0;
        report.retrieval->index_enabled_hops = 0;
        report.retrieval->index_bypassed_hops = 30'000;

        const auto result = benchmark::validate_benchmark_report(report);
        CHECK_FALSE(result.ok());
        CHECK(has_error(result, "retrieval.fallback_reason"));
    }

    SECTION("no-index diagnostic cannot claim an active index") {
        auto report = valid_report();
        report.index->index_rate = 0.0;
        report.retrieval->mode = "no_index_diagnostic";
        report.retrieval->index_queries = 0;
        report.retrieval->index_enabled_hops = 0;
        report.retrieval->index_bypassed_hops = 30'000;

        const auto result = benchmark::validate_benchmark_report(report);
        CHECK_FALSE(result.ok());
        CHECK(has_error(result, "index"));
    }

    SECTION("fallback requires authorization, reason and transition") {
        auto report = valid_report();
        report.retrieval->mode = "sticky_no_index_fallback";
        report.retrieval->fallback_policy = "allow_sticky_no_index";
        report.retrieval->fallback_triggered = true;
        report.retrieval->fallback_reason.clear();
        report.retrieval->index_enabled_hops = 20'000;
        report.retrieval->index_bypassed_hops = 10'000;
        report.retrieval->fallback_transitions = 0;

        const auto result = benchmark::validate_benchmark_report(report);
        CHECK_FALSE(result.ok());
        CHECK(has_error(result, "retrieval.fallback_reason"));
        CHECK(has_error(result, "retrieval.fallback_transitions"));
    }


    SECTION("sticky fallback must prove an indexed prefix") {
        auto report = valid_report();
        report.index->index_rate = 0.0;
        report.retrieval->mode = "sticky_no_index_fallback";
        report.retrieval->fallback_policy = "allow_sticky_no_index";
        report.retrieval->fallback_triggered = true;
        report.retrieval->fallback_reason = "sustained_index_deadline";
        report.retrieval->index_queries = 0;
        report.retrieval->index_enabled_hops = 0;
        report.retrieval->index_bypassed_hops = 10'000;
        report.retrieval->fallback_transitions = 1;

        const auto result = benchmark::validate_benchmark_report(report);
        CHECK_FALSE(result.ok());
        CHECK(has_error(result, "index.index_rate"));
        CHECK(has_error(result, "retrieval.index_queries"));
        CHECK(has_error(result, "retrieval.index_enabled_hops"));
    }
}
