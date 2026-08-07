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

void print_usage() {
    std::cout << "Usage:\n"
              << "  build_index --hubert <path> --input-dir <dir> --output <index.bin>\n\n"
              << "Extracts HuBERT v2 features from all .wav/.flac files in the\n"
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

    voxmutatio::content::HubertConfig cfg;
    cfg.model_path = hubert_path;
    cfg.output_dim = 768;
    cfg.use_final_proj = false;

    voxmutatio::content::HubertEncoder encoder;
    if (!encoder.init(cfg)) {
        std::cerr << "Failed to init HuBERT\n";
        return 1;
    }

    std::vector<float> all_feats;
    const int dim = 768;
    int64_t ntotal = 0;
    int file_count = 0;

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        auto ext = entry.path().extension().string();
        if (ext != ".wav" && ext != ".flac") continue;

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

    if (ntotal == 0) {
        std::cerr << "No features extracted\n";
        return 1;
    }

    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Cannot write: " << output_path << "\n";
        return 1;
    }
    int64_t dim64 = dim;
    out.write(reinterpret_cast<const char*>(&ntotal), 8);
    out.write(reinterpret_cast<const char*>(&dim64), 8);
    out.write(reinterpret_cast<const char*>(all_feats.data()),
              all_feats.size() * sizeof(float));

    std::cout << "Wrote index: " << output_path << " (" << file_count
              << " files, " << ntotal << " vectors x " << dim << ")\n";
    return 0;
}
