// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/io/audio_io.h"

#include <fstream>
#include <cstring>
#include <sstream>
#include <algorithm>

namespace voxmutatio::io {

namespace {

// WAV file header structure
struct WaveHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t file_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
};

struct FmtChunk {
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t chunk_size = 16;
    uint16_t audio_format;  // 1 = PCM
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

struct DataChunk {
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t chunk_size;
};

bool read_wav(const std::string& path, std::vector<float>& samples,
              int& sample_rate) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    WaveHeader wave_header;
    file.read(reinterpret_cast<char*>(&wave_header), sizeof(wave_header));
    if (!file || std::memcmp(wave_header.riff, "RIFF", 4) != 0 ||
        std::memcmp(wave_header.wave, "WAVE", 4) != 0) {
        return false;
    }

    // Find data chunk
    FmtChunk fmt_chunk;
    DataChunk data_chunk;
    
    char chunk_id[5] = {0};
    uint32_t chunk_size = 0;
    
    bool has_fmt = false;
    bool has_data = false;
    
    while (file.peek() != EOF) {
        file.read(chunk_id, 4);
        file.read(reinterpret_cast<char*>(&chunk_size), 4);
        
        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            file.read(reinterpret_cast<char*>(&fmt_chunk), sizeof(fmt_chunk));
            has_fmt = true;
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            data_chunk.chunk_size = chunk_size;
            has_data = true;
            break;
        } else {
            // Skip unknown chunk
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    if (!has_fmt || !has_data) {
        return false;
    }

    sample_rate = fmt_chunk.sample_rate;
    
    // Read audio data
    int num_samples = data_chunk.chunk_size / (fmt_chunk.bits_per_sample / 8);
    samples.resize(num_samples);

    if (fmt_chunk.bits_per_sample == 16) {
        std::vector<int16_t> int16_samples(num_samples);
        file.read(reinterpret_cast<char*>(int16_samples.data()),
                  num_samples * sizeof(int16_t));
        
        // Convert int16 to float32
        for (int i = 0; i < num_samples; ++i) {
            samples[i] = static_cast<float>(int16_samples[i]) / 32768.0f;
        }
    } else if (fmt_chunk.bits_per_sample == 32 && 
               fmt_chunk.audio_format == 3) {  // IEEE float
        file.read(reinterpret_cast<char*>(samples.data()),
                  num_samples * sizeof(float));
    } else {
        return false;  // Unsupported format
    }

    return true;
}

bool write_wav(const std::string& path, const float* samples,
               int num_samples, int sample_rate) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Write headers (data will be updated later)
    WaveHeader wave_header;
    FmtChunk fmt_chunk;
    DataChunk data_chunk;

    fmt_chunk.audio_format = 1;  // PCM
    fmt_chunk.num_channels = 1;
    fmt_chunk.sample_rate = sample_rate;
    fmt_chunk.bits_per_sample = 16;
    fmt_chunk.byte_rate = sample_rate * fmt_chunk.num_channels * 
                         (fmt_chunk.bits_per_sample / 8);
    fmt_chunk.block_align = fmt_chunk.num_channels * 
                           (fmt_chunk.bits_per_sample / 8);

    data_chunk.chunk_size = num_samples * (fmt_chunk.bits_per_sample / 8);

    wave_header.file_size = 4 + (8 + fmt_chunk.chunk_size) + 
                           (8 + data_chunk.chunk_size);

    file.write(reinterpret_cast<const char*>(&wave_header), sizeof(wave_header));
    file.write(reinterpret_cast<const char*>(&fmt_chunk), sizeof(fmt_chunk));
    file.write(reinterpret_cast<const char*>(&data_chunk), sizeof(data_chunk));

    // Convert float32 to int16 and write
    for (int i = 0; i < num_samples; ++i) {
        float sample = std::clamp(samples[i], -1.0f, 1.0f);
        int16_t int16_sample = static_cast<int16_t>(sample * 32767.0f);
        file.write(reinterpret_cast<const char*>(&int16_sample), sizeof(int16_t));
    }

    return true;
}

}  // namespace

std::optional<AudioBuffer> read_audio(const std::string& path,
                                      int target_sr) {
    std::vector<float> samples;
    int native_sr = 0;

    if (!read_wav(path, samples, native_sr)) {
        return std::nullopt;
    }

    AudioBuffer buffer;
    buffer.data = std::move(samples);
    buffer.sample_rate = native_sr;
    buffer.source_format = SampleFormat::kFloat32;

    // Resample if needed
    if (native_sr != target_sr) {
        std::vector<float> resampled = resample_linear(
            buffer.data.data(), static_cast<int>(buffer.data.size()),
            native_sr, target_sr);
        buffer.data = std::move(resampled);
        buffer.sample_rate = target_sr;
    }

    return buffer;
}

bool write_audio(const std::string& path, const float* samples,
                 int num_samples, int sample_rate,
                 const std::string& format) {
    if (format == "wav") {
        return write_wav(path, samples, num_samples, sample_rate);
    }
    // FLAC support would be added here
    return false;
}

std::vector<float> resample_linear(const float* input, int input_len,
                                   int src_sr, int dst_sr) {
    if (src_sr == dst_sr) {
        return std::vector<float>(input, input + input_len);
    }

    double ratio = static_cast<double>(dst_sr) / static_cast<double>(src_sr);
    int output_len = static_cast<int>(input_len * ratio);

    std::vector<float> output(output_len);

    for (int i = 0; i < output_len; ++i) {
        double pos = i / ratio;
        int idx = static_cast<int>(pos);
        double frac = pos - idx;

        if (idx < input_len - 1) {
            output[i] = input[idx] * (1.0 - frac) + input[idx + 1] * frac;
        } else {
            output[i] = input[std::min(idx, input_len - 1)];
        }
    }

    return output;
}

}  // namespace voxmutatio::io
