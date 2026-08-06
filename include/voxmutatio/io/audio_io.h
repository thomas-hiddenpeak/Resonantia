#pragma once

#include <optional>
#include <string>
#include <vector>

#include "voxmutatio/core/types.h"

namespace voxmutatio::io {

/// Read an audio file and return mono float32 PCM at 16kHz
std::optional<AudioBuffer> read_audio(const std::string& path,
                                      int target_sr = 16'000);

/// Write float32 PCM to a WAV or FLAC file
bool write_audio(const std::string& path, const float* samples,
                 int num_samples, int sample_rate,
                 const std::string& format = "wav");

/// Resample audio using linear interpolation (simple, for I/O only)
std::vector<float> resample_linear(const float* input, int input_len,
                                   int src_sr, int dst_sr);

}  // namespace voxmutatio::io
