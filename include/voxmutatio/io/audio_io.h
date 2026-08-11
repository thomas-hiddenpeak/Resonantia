#pragma once

#include <optional>
#include <string>
#include <vector>

#include "voxmutatio/core/types.h"

namespace voxmutatio::io {

/// Read a WAV file, downmix it to mono, and return float32 PCM at target_sr.
/// Returns std::nullopt for malformed/unsupported input or an invalid rate.
std::optional<AudioBuffer> read_audio(const std::string& path,
                                      int target_sr = 16'000);

/// Write mono float32 PCM. WAV is currently the supported output format.
/// A successful write atomically replaces the destination; a failure preserves it.
bool write_audio(const std::string& path, const float* samples,
                 int num_samples, int sample_rate,
                 const std::string& format = "wav");

/// Resample audio using linear interpolation (simple, for I/O only).
/// Returns an empty vector for invalid pointers, lengths, or sample rates.
std::vector<float> resample_linear(const float* input, int input_len,
                                   int src_sr, int dst_sr);

}  // namespace voxmutatio::io
