#pragma once

#include <optional>
#include <string>
#include <vector>

#include "voxmutatio/core/types.h"

namespace voxmutatio::io {

struct AudioFileInfo { std::uint64_t frames=0; int sample_rate=0, channels=0; };
/// Header-only, seekable WAV inspection; malformed files/final symlinks rejected.
std::optional<AudioFileInfo> inspect_audio(const std::string& path);
/// Native-rate mono excerpt, 0 < end-start <=15s, no resampling or normalization.
/// Reads only the selected samples; rejects out-of-file intervals/nonfinite data.
std::optional<AudioBuffer> read_audio_excerpt(const std::string& path,double start,double end);
/// Atomic native-rate mono PCM16 excerpt, <=15s; preserves every selected PCM
/// sample exactly. Rejects non-PCM16/stereo input rather than silently remixing.
bool write_audio_excerpt(const std::string& input,const std::string& output,double start,double end);

/// Owned interleaved L/R PCM. Mono input is duplicated; >2 channels rejected.
/// Limited to 600 seconds and 128 MiB encoded WAV for the cleanup workflow.
struct StereoAudioBuffer {
    std::vector<float> data;
    int sample_rate = 44'100;
    [[nodiscard]] std::size_t frames() const noexcept { return data.size() / 2; }
};
std::optional<StereoAudioBuffer> read_stereo_audio(const std::string& path,
                                                 int target_sr = 44'100);
/// Atomic stereo WAV output; float32 preserves separation intermediates.
/// PCM16 is suitable for playback. Nonfinite/oversized inputs fail unchanged.
bool write_stereo_audio(const std::string& path, const float* interleaved,
                        int frames, int sample_rate, bool float32 = false);

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
