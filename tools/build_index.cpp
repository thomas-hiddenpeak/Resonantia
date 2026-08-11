/**
 * @file build_index.cpp
 * @brief Build a feature retrieval index from a directory of audio files.
 *
 * Extracts HuBERT features (v2, 768-dim) from each audio file and writes a
 * flat index binary: [int64 ntotal][int64 dim][float32 data].
 * Pure C++/CUDA — no Python at runtime.
 *
 * Usage:
 *   build_index --hubert <path> --input-dir <dir> --output <index.bin>
 */
#include <algorithm>
#include <bit>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "voxmutatio/content/hubert_encoder.h"
#include "voxmutatio/io/audio_io.h"

namespace fs = std::filesystem;

namespace {

static_assert(std::endian::native == std::endian::little,
              "the memory-mapped Flat Index payload requires a little-endian host");

void write_u64_le(std::ostream& out, std::uint64_t value) {
    char bytes[8];
    for (int i = 0; i < 8; ++i)
        bytes[i] = static_cast<char>((value >> (8 * i)) & 0xffu);
    out.write(bytes, sizeof(bytes));
}

void print_usage() {
    std::cout << "Usage:\n"
              << "  build_index --hubert <path> --input-dir <dir> --output <index.bin>\n\n"
              << "Extracts HuBERT v2 features from all .wav files in the\n"
              << "directory and writes a flat feature index.\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string hubert_path, input_dir, output_path;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--hubert" && i + 1 < argc) hubert_path = argv[++i];
        else if (a == "--input-dir" && i + 1 < argc) input_dir = argv[++i];
        else if (a == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (a == "--help" || a == "-h") { print_usage(); return 0; }
    }

    if (hubert_path.empty() || input_dir.empty() || output_path.empty()) {
        print_usage();
        return 1;
    }
    std::error_code input_error;
    if (!fs::is_directory(input_dir, input_error) || input_error) {
        std::cerr << "Input directory is unavailable: " << input_dir << "\n";
        return 1;
    }

    voxmutatio::content::HubertConfig cfg;
    cfg.model_path = hubert_path;
    cfg.output_dim = 768;
    cfg.use_final_proj = false;
    cfg.num_layers = 12;

    voxmutatio::content::HubertEncoder encoder;
    if (!encoder.init(cfg)) {
        std::cerr << "Failed to init HuBERT\n";
        return 1;
    }

    std::vector<float> all_feats;
    const int dim = 768;
    int64_t ntotal = 0;
    int file_count = 0;

    std::error_code iteration_error;
    fs::directory_iterator iterator(input_dir, iteration_error), end;
    for (; !iteration_error && iterator != end; iterator.increment(iteration_error)) {
        const auto& entry = *iterator;
        std::error_code type_error;
        if (!entry.is_regular_file(type_error) || type_error) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (ext != ".wav") continue;

        auto audio = voxmutatio::io::read_audio(entry.path().string(), 16000);
        if (!audio.has_value() || audio->data.empty()) {
            std::cerr << "  skip (read failed): " << entry.path() << "\n";
            continue;
        }

        auto feats = encoder.extract(audio->data.data(),
                                     static_cast<int>(audio->data.size()));
        if (feats.empty()) continue;
        int frames = static_cast<int>(feats.size()) / dim;

        all_feats.insert(all_feats.end(), feats.begin(), feats.end());
        ntotal += frames;
        ++file_count;
        std::cout << "  " << entry.path().filename().string() << ": "
                  << frames << " frames\n";
    }
    if (iteration_error) {
        std::cerr << "Failed while scanning input directory: "
                  << iteration_error.message() << "\n";
        return 1;
    }

    if (ntotal == 0) {
        std::cerr << "No features extracted\n";
        return 1;
    }

    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Cannot write: " << output_path << "\n";
        return 1;
    }
    write_u64_le(out, static_cast<std::uint64_t>(ntotal));
    write_u64_le(out, static_cast<std::uint64_t>(dim));
    out.write(reinterpret_cast<const char*>(all_feats.data()),
              all_feats.size() * sizeof(float));
    out.flush();
    if (!out) {
        std::cerr << "Failed while writing index: " << output_path << "\n";
        return 1;
    }
    out.close();
    if (!out) {
        std::cerr << "Failed while closing index: " << output_path << "\n";
        return 1;
    }

    std::cout << "Wrote index: " << output_path << " (" << file_count
              << " files, " << ntotal << " vectors x " << dim << ")\n";
    return 0;
}
