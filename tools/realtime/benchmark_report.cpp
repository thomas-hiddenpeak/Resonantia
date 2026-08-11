#include "tools/realtime/benchmark_report.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <string_view>
#include <unordered_map>

namespace resonantia::realtime::benchmark {
namespace {

void add_error(ValidationResult& result, std::string_view field,
               std::string_view detail) {
    result.errors.emplace_back(std::string(field) + ": " + std::string(detail));
}

bool is_sha256(std::string_view value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](unsigned char ch) {
               return std::isxdigit(ch) != 0;
           });
}

bool finite_positive(const std::optional<double>& value) {
    return value && std::isfinite(*value) && *value > 0.0;
}

bool finite_nonnegative(const std::optional<double>& value) {
    return value && std::isfinite(*value) && *value >= 0.0;
}

template <typename T>
bool positive(const std::optional<T>& value) {
    return value && *value > 0;
}

template <typename T>
void require_present(const std::optional<T>& value, std::string_view field,
                     ValidationResult& result) {
    if (!value) add_error(result, field, "missing");
}

void require_text(const std::string& value, std::string_view field,
                  ValidationResult& result) {
    if (value.empty()) add_error(result, field, "missing or empty");
}

void require_sha256(const std::string& value, std::string_view field,
                    ValidationResult& result) {
    if (!is_sha256(value)) add_error(result, field, "must be 64 hexadecimal characters");
}

void validate_latency(const StageLatency& stage, ValidationResult& result) {
    const std::string prefix = "stages." + stage.name + ".";
    if (!positive(stage.latency.count)) {
        add_error(result, prefix + "count", "missing or zero");
    }

    const std::array<std::pair<std::string_view, const std::optional<double>*>, 4> values{{
        {"p50_ms", &stage.latency.p50_ms},
        {"p95_ms", &stage.latency.p95_ms},
        {"p99_ms", &stage.latency.p99_ms},
        {"max_ms", &stage.latency.max_ms},
    }};
    for (const auto& [name, value] : values) {
        if (!finite_nonnegative(*value)) {
            add_error(result, prefix + std::string(name), "missing, negative, or non-finite");
        }
    }

    if (finite_nonnegative(stage.latency.p50_ms) &&
        finite_nonnegative(stage.latency.p95_ms) &&
        finite_nonnegative(stage.latency.p99_ms) &&
        finite_nonnegative(stage.latency.max_ms) &&
        !(*stage.latency.p50_ms <= *stage.latency.p95_ms &&
          *stage.latency.p95_ms <= *stage.latency.p99_ms &&
          *stage.latency.p99_ms <= *stage.latency.max_ms)) {
        add_error(result, prefix + "percentiles", "must satisfy p50 <= p95 <= p99 <= max");
    }
}

}  // namespace

ValidationResult validate_benchmark_report(const BenchmarkReport& report) {
    ValidationResult result;

    const bool index_enabled =
        report.retrieval && report.retrieval->mode == "index_enabled";
    const bool no_index =
        report.retrieval && report.retrieval->mode == "no_index_diagnostic";
    const bool sticky_fallback =
        report.retrieval && report.retrieval->mode == "sticky_no_index_fallback";

    if (report.schema_version != kBenchmarkReportSchemaVersion) {
        add_error(result, "schema_version", "unsupported or missing");
    }

    if (!report.build) {
        add_error(result, "build", "missing");
    } else {
        require_text(report.build->revision, "build.revision", result);
        require_text(report.build->tree_state, "build.tree_state", result);
        require_text(report.build->build_type, "build.build_type", result);
        require_text(report.build->compiler, "build.compiler", result);
        require_text(report.build->cuda_version, "build.cuda_version", result);
        require_text(report.build->compile_flags, "build.compile_flags", result);
    }

    if (!report.environment) {
        add_error(result, "environment", "missing");
    } else {
        require_text(report.environment->machine, "environment.machine", result);
        require_text(report.environment->os, "environment.os", result);
        require_text(report.environment->architecture, "environment.architecture", result);
        require_text(report.environment->power_mode, "environment.power_mode", result);
        require_text(report.environment->clocks, "environment.clocks", result);
    }

    if (!report.profile) {
        add_error(result, "profile", "missing");
    } else {
        require_text(report.profile->name, "profile.name", result);
        if (!positive(report.profile->input_sample_rate)) {
            add_error(result, "profile.input_sample_rate", "missing or non-positive");
        }
        if (!positive(report.profile->output_sample_rate)) {
            add_error(result, "profile.output_sample_rate", "missing or non-positive");
        }
        if (!positive(report.profile->window_ms)) {
            add_error(result, "profile.window_ms", "missing or non-positive");
        }
        if (!positive(report.profile->emit_hop_ms)) {
            add_error(result, "profile.emit_hop_ms", "missing or non-positive");
        }
        if (!report.profile->lookahead_ms || *report.profile->lookahead_ms < 0) {
            add_error(result, "profile.lookahead_ms", "missing or negative");
        }
        if (!finite_positive(report.profile->measured_duration_seconds)) {
            add_error(result, "profile.measured_duration_seconds",
                      "missing, non-positive, or non-finite");
        }
        if (positive(report.profile->window_ms) && positive(report.profile->emit_hop_ms) &&
            report.profile->lookahead_ms &&
            static_cast<std::int64_t>(*report.profile->window_ms) <
                static_cast<std::int64_t>(*report.profile->emit_hop_ms) +
                    static_cast<std::int64_t>(*report.profile->lookahead_ms)) {
            add_error(result, "profile.window_ms", "smaller than hop plus lookahead");
        }
    }

    if (!report.artifacts) {
        add_error(result, "artifacts", "missing");
    } else {
        require_sha256(report.artifacts->input_audio_sha256,
                       "artifacts.input_audio_sha256", result);
        require_sha256(report.artifacts->hubert_sha256,
                       "artifacts.hubert_sha256", result);
        require_sha256(report.artifacts->f0_sha256,
                       "artifacts.f0_sha256", result);
        require_sha256(report.artifacts->synthesizer_sha256,
                       "artifacts.synthesizer_sha256", result);
    }

    if (no_index) {
        if (report.index) {
            add_error(result, "index",
                      "must be absent for no_index_diagnostic mode");
        }
    } else if (!report.index) {
        add_error(result, "index", "missing for an index-capable retrieval mode");
    } else {
        require_sha256(report.index->sha256, "index.sha256", result);
        if (!positive(report.index->rows)) {
            add_error(result, "index.rows", "missing or zero");
        }
        if (!positive(report.index->dimension)) {
            add_error(result, "index.dimension", "missing or zero");
        }
        if (!positive(report.index->k)) {
            add_error(result, "index.k", "missing or zero");
        } else if (!report.index->rows || *report.index->k > *report.index->rows) {
            add_error(result, "index.k", "cannot exceed rows or be validated without rows");
        }
        if (!finite_nonnegative(report.index->index_rate) ||
            (report.index->index_rate && *report.index->index_rate > 1.0)) {
            add_error(result, "index.index_rate", "missing, non-finite, or outside [0, 1]");
        } else if ((index_enabled || sticky_fallback) &&
                   *report.index->index_rate <= 0.0) {
            add_error(result, "index.index_rate",
                      "must be positive for an index-capable retrieval mode");
        }
        if (!positive(report.index->query_batch)) {
            add_error(result, "index.query_batch", "missing or zero");
        }
    }

    if (!report.warmup) {
        add_error(result, "warmup", "missing");
    } else {
        if (!finite_positive(report.warmup->duration_seconds)) {
            add_error(result, "warmup.duration_seconds", "missing, non-positive, or non-finite");
        }
        if (!positive(report.warmup->chunks)) {
            add_error(result, "warmup.chunks", "missing or zero");
        }
    }

    std::unordered_map<std::string, std::size_t> stage_counts;
    for (const auto& stage : report.stages) {
        if (stage.name.empty()) {
            add_error(result, "stages.name", "missing or empty");
            continue;
        }
        ++stage_counts[stage.name];
        validate_latency(stage, result);
    }
    constexpr std::array<std::string_view, 8> kRequiredStages{
        "queue_wait", "hubert", "f0", "index", "synth", "gpu_total",
        "output_queue_device", "capture_to_playback",
    };
    for (const auto stage : kRequiredStages) {
        const auto found = stage_counts.find(std::string(stage));
        if (found == stage_counts.end()) {
            add_error(result, "stages." + std::string(stage), "missing");
        } else if (found->second != 1) {
            add_error(result, "stages." + std::string(stage), "must occur exactly once");
        }
    }

    if (!report.retrieval) {
        add_error(result, "retrieval", "missing");
    } else {
        const auto& retrieval = *report.retrieval;
        if (!index_enabled && !no_index && !sticky_fallback) {
            add_error(result, "retrieval.mode", "unsupported or missing");
        }
        if (retrieval.fallback_policy != "disabled" &&
            retrieval.fallback_policy != "allow_sticky_no_index") {
            add_error(result, "retrieval.fallback_policy", "unsupported or missing");
        }
        require_present(retrieval.fallback_triggered,
                        "retrieval.fallback_triggered", result);
        require_text(retrieval.fallback_reason, "retrieval.fallback_reason", result);
        require_present(retrieval.index_queries, "retrieval.index_queries", result);
        require_present(retrieval.index_enabled_hops,
                        "retrieval.index_enabled_hops", result);
        require_present(retrieval.index_bypassed_hops,
                        "retrieval.index_bypassed_hops", result);
        require_present(retrieval.fallback_transitions,
                        "retrieval.fallback_transitions", result);

        if (!sticky_fallback && retrieval.fallback_triggered &&
            *retrieval.fallback_triggered) {
            add_error(result, "retrieval.fallback_triggered",
                      "does not match the retrieval mode");
        }
        if (!sticky_fallback && retrieval.fallback_reason != "none") {
            add_error(result, "retrieval.fallback_reason",
                      "must be none when no fallback occurred");
        }

        if (index_enabled) {
            if (!positive(retrieval.index_queries)) {
                add_error(result, "retrieval.index_queries",
                          "must be positive in index_enabled mode");
            }
            if (!positive(retrieval.index_enabled_hops)) {
                add_error(result, "retrieval.index_enabled_hops",
                          "must be positive in index_enabled mode");
            }
            if (retrieval.index_bypassed_hops &&
                *retrieval.index_bypassed_hops != 0) {
                add_error(result, "retrieval.index_bypassed_hops",
                          "must be zero in index_enabled mode");
            }
            if (retrieval.fallback_transitions &&
                *retrieval.fallback_transitions != 0) {
                add_error(result, "retrieval.fallback_transitions",
                          "must be zero in index_enabled mode");
            }
        } else if (no_index) {
            if (retrieval.fallback_policy != "disabled") {
                add_error(result, "retrieval.fallback_policy",
                          "must be disabled in no_index_diagnostic mode");
            }
            if (retrieval.index_queries && *retrieval.index_queries != 0) {
                add_error(result, "retrieval.index_queries",
                          "must be zero in no_index_diagnostic mode");
            }
            if (retrieval.index_enabled_hops &&
                *retrieval.index_enabled_hops != 0) {
                add_error(result, "retrieval.index_enabled_hops",
                          "must be zero in no_index_diagnostic mode");
            }
            if (!positive(retrieval.index_bypassed_hops)) {
                add_error(result, "retrieval.index_bypassed_hops",
                          "must be positive in no_index_diagnostic mode");
            }
            if (retrieval.fallback_transitions &&
                *retrieval.fallback_transitions != 0) {
                add_error(result, "retrieval.fallback_transitions",
                          "must be zero in no_index_diagnostic mode");
            }
        } else if (sticky_fallback) {
            if (retrieval.fallback_policy != "allow_sticky_no_index") {
                add_error(result, "retrieval.fallback_policy",
                          "sticky fallback was not authorized");
            }
            if (!retrieval.fallback_triggered || !*retrieval.fallback_triggered) {
                add_error(result, "retrieval.fallback_triggered",
                          "must be true in sticky fallback mode");
            }
            if (retrieval.fallback_reason.empty() || retrieval.fallback_reason == "none") {
                add_error(result, "retrieval.fallback_reason",
                          "must identify the fallback trigger");
            }
            if (!retrieval.fallback_transitions ||
                *retrieval.fallback_transitions != 1) {
                add_error(result, "retrieval.fallback_transitions",
                          "must contain exactly one sticky transition");
            }
            if (!positive(retrieval.index_queries)) {
                add_error(result, "retrieval.index_queries",
                          "must prove a non-empty indexed prefix");
            }
            if (!positive(retrieval.index_enabled_hops)) {
                add_error(result, "retrieval.index_enabled_hops",
                          "must prove a non-empty indexed prefix");
            }
            if (!positive(retrieval.index_bypassed_hops)) {
                add_error(result, "retrieval.index_bypassed_hops",
                          "must prove a non-empty fallback suffix");
            }
        }
    }

    if (!report.counters) {
        add_error(result, "counters", "missing");
    } else {
        require_present(report.counters->input_hops_dropped,
                        "counters.input_hops_dropped", result);
        require_present(report.counters->output_hops_concealed,
                        "counters.output_hops_concealed", result);
        require_present(report.counters->capture_overruns,
                        "counters.capture_overruns", result);
        require_present(report.counters->playback_underruns,
                        "counters.playback_underruns", result);
        require_present(report.counters->alsa_xruns, "counters.alsa_xruns", result);
        require_present(report.counters->alsa_recoveries,
                        "counters.alsa_recoveries", result);
        if (report.counters->alsa_xruns && report.counters->alsa_recoveries &&
            *report.counters->alsa_recoveries > *report.counters->alsa_xruns) {
            add_error(result, "counters.alsa_recoveries", "cannot exceed alsa_xruns");
        }
    }

    if (!report.resources) {
        add_error(result, "resources", "missing");
    } else {
        const auto& resources = *report.resources;
        if (!positive(resources.peak_rss_bytes)) {
            add_error(result, "resources.peak_rss_bytes", "missing or zero");
        }
        if (!positive(resources.steady_rss_bytes)) {
            add_error(result, "resources.steady_rss_bytes", "missing or zero");
        }
        if (!positive(resources.peak_cuda_allocated_bytes)) {
            add_error(result, "resources.peak_cuda_allocated_bytes", "missing or zero");
        }
        if (!positive(resources.steady_cuda_allocated_bytes)) {
            add_error(result, "resources.steady_cuda_allocated_bytes", "missing or zero");
        }
        require_present(resources.system_memory_delta_bytes,
                        "resources.system_memory_delta_bytes", result);
        if (!finite_nonnegative(resources.thermal_start_celsius)) {
            add_error(result, "resources.thermal_start_celsius",
                      "missing, negative, or non-finite");
        }
        if (!finite_nonnegative(resources.thermal_peak_celsius)) {
            add_error(result, "resources.thermal_peak_celsius",
                      "missing, negative, or non-finite");
        }
        if (!positive(resources.thermal_samples)) {
            add_error(result, "resources.thermal_samples", "missing or zero");
        }
        require_present(resources.thermal_throttled,
                        "resources.thermal_throttled", result);
        if (resources.peak_rss_bytes && resources.steady_rss_bytes &&
            *resources.steady_rss_bytes > *resources.peak_rss_bytes) {
            add_error(result, "resources.steady_rss_bytes", "cannot exceed peak RSS");
        }
        if (resources.peak_cuda_allocated_bytes && resources.steady_cuda_allocated_bytes &&
            *resources.steady_cuda_allocated_bytes > *resources.peak_cuda_allocated_bytes) {
            add_error(result, "resources.steady_cuda_allocated_bytes",
                      "cannot exceed peak CUDA allocation");
        }
        if (finite_nonnegative(resources.thermal_start_celsius) &&
            finite_nonnegative(resources.thermal_peak_celsius) &&
            *resources.thermal_peak_celsius < *resources.thermal_start_celsius) {
            add_error(result, "resources.thermal_peak_celsius",
                      "cannot be lower than the starting temperature");
        }
    }

    return result;
}

}  // namespace resonantia::realtime::benchmark
