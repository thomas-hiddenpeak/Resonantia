#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace resonantia::realtime::benchmark {

inline constexpr std::uint32_t kBenchmarkReportSchemaVersion = 1;

struct BuildMetadata {
    std::string revision;
    std::string tree_state;
    std::string build_type;
    std::string compiler;
    std::string cuda_version;
    std::string compile_flags;
};

struct EnvironmentMetadata {
    std::string machine;
    std::string os;
    std::string architecture;
    std::string power_mode;
    std::string clocks;
};

struct ProfileMetadata {
    std::string name;
    std::optional<int> input_sample_rate;
    std::optional<int> output_sample_rate;
    std::optional<int> window_ms;
    std::optional<int> emit_hop_ms;
    std::optional<int> lookahead_ms;
    std::optional<double> measured_duration_seconds;
};

struct ArtifactHashes {
    std::string input_audio_sha256;
    std::string hubert_sha256;
    std::string f0_sha256;
    std::string synthesizer_sha256;
};

struct IndexMetadata {
    std::string sha256;
    std::optional<std::uint64_t> rows;
    std::optional<std::uint32_t> dimension;
    std::optional<std::uint32_t> k;
    std::optional<double> index_rate;
    std::optional<std::uint32_t> query_batch;
};

struct WarmupMetadata {
    std::optional<double> duration_seconds;
    std::optional<std::uint64_t> chunks;
};

struct LatencySummary {
    std::optional<std::uint64_t> count;
    std::optional<double> p50_ms;
    std::optional<double> p95_ms;
    std::optional<double> p99_ms;
    std::optional<double> max_ms;
};

struct StageLatency {
    std::string name;
    LatencySummary latency;
};

struct RetrievalMetrics {
    std::string mode;
    std::string fallback_policy;
    std::optional<bool> fallback_triggered;
    std::string fallback_reason;
    std::optional<std::uint64_t> index_queries;
    std::optional<std::uint64_t> index_enabled_hops;
    std::optional<std::uint64_t> index_bypassed_hops;
    std::optional<std::uint64_t> fallback_transitions;
};

struct RunCounters {
    std::optional<std::uint64_t> input_hops_dropped;
    std::optional<std::uint64_t> output_hops_concealed;
    std::optional<std::uint64_t> capture_overruns;
    std::optional<std::uint64_t> playback_underruns;
    std::optional<std::uint64_t> alsa_xruns;
    std::optional<std::uint64_t> alsa_recoveries;
};

struct ResourceMetrics {
    std::optional<std::uint64_t> peak_rss_bytes;
    std::optional<std::uint64_t> steady_rss_bytes;
    std::optional<std::uint64_t> peak_cuda_allocated_bytes;
    std::optional<std::uint64_t> steady_cuda_allocated_bytes;
    std::optional<std::int64_t> system_memory_delta_bytes;
    std::optional<double> thermal_start_celsius;
    std::optional<double> thermal_peak_celsius;
    std::optional<std::uint64_t> thermal_samples;
    std::optional<bool> thermal_throttled;
};

// Internal construction-time schema for R0 benchmark runners.  It deliberately
// lives under tools/ and is not part of the installed voxmutatio API.
struct BenchmarkReport {
    std::uint32_t schema_version = 0;
    std::optional<BuildMetadata> build;
    std::optional<EnvironmentMetadata> environment;
    std::optional<ProfileMetadata> profile;
    std::optional<ArtifactHashes> artifacts;
    std::optional<IndexMetadata> index;
    std::optional<WarmupMetadata> warmup;
    std::vector<StageLatency> stages;
    std::optional<RetrievalMetrics> retrieval;
    std::optional<RunCounters> counters;
    std::optional<ResourceMetrics> resources;
};

struct ValidationResult {
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

[[nodiscard]] ValidationResult validate_benchmark_report(const BenchmarkReport& report);

}  // namespace resonantia::realtime::benchmark
