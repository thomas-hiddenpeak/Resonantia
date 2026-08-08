/**
 * @file vc_batch.cpp
 * @brief Batch voice conversion CLI tool.
 *
 * Usage:
 *   vc_batch --input-dir <dir> --output-dir <dir> [options]
 */
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>

#include "voxmutatio/core/types.h"
#include "voxmutatio/core/device.h"
#include "voxmutatio/pipeline/pipeline.h"

namespace fs = std::filesystem;

namespace {

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  vc_batch --input-dir <dir> --output-dir <dir> [options]\n\n"
        << "Options:\n"
        << "  --input-dir <dir>     Input directory containing audio files\n"
        << "  --output-dir <dir>    Output directory for converted audio\n"
        << "  --hubert <path>       HuBERT model safetensors directory\n"
        << "  --model <path>        VITS synthesizer safetensors path\n"
        << "  --index <path>        CUDA Flat Index file (optional)\n"
        << "  --rmvpe <path>        RMVPE model safetensors (optional)\n"
        << "  --speaker <id>        Speaker ID (default: 0)\n"
        << "  --pitch <semitones>   Pitch shift in semitones (default: 0)\n"
        << "  --index-rate <0-1>    Index blend rate (default: 0.0)\n"
        << "  --rms-mix <0-1>       RMS energy mix rate (default: 0.5)\n"
        << "  --protect <0-1>       Unvoiced protection (default: 0.5)\n"
        << "  --filter-radius <n>   F0 median filter radius (default: 3)\n"
        << "  --version <v1|v2>     Model version (default: v1)\n"
        << "  --speakers <n>        Speaker embedding count (default: 1)\n"
        << "  --sr <hz>             Model output sample rate (default: 40000)\n"
        << "  --recursive           Recurse into subdirectories\n"
        << "  --half                Use FP16 inference\n"
        << "  --device <cuda|cpu>   Compute device (default: cuda)\n"
        << "  --help                Show this help\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
    }

    voxmutatio::VCConfig config;
    std::string input_dir, output_dir;
    int speaker_id = 0;
    bool recursive = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input-dir" && i + 1 < argc) input_dir = argv[++i];
        else if (arg == "--output-dir" && i + 1 < argc) output_dir = argv[++i];
        else if (arg == "--hubert" && i + 1 < argc) config.hubert_model_path = argv[++i];
        else if (arg == "--model" && i + 1 < argc) config.synthesizer_model_path = argv[++i];
        else if (arg == "--index" && i + 1 < argc) config.index_path = argv[++i];
        else if (arg == "--rmvpe" && i + 1 < argc) config.rmvpe_model_path = argv[++i];
        else if (arg == "--speaker" && i + 1 < argc) speaker_id = std::atoi(argv[++i]);
        else if (arg == "--pitch" && i + 1 < argc) config.f0_up_key = std::atoi(argv[++i]);
        else if (arg == "--index-rate" && i + 1 < argc) config.index_rate = std::atof(argv[++i]);
        else if (arg == "--rms-mix" && i + 1 < argc) config.rms_mix_rate = std::atof(argv[++i]);
        else if (arg == "--protect" && i + 1 < argc) config.protect = std::atof(argv[++i]);
        else if (arg == "--filter-radius" && i + 1 < argc) config.filter_radius = std::atoi(argv[++i]);
        else if (arg == "--version" && i + 1 < argc) {
            std::string v = argv[++i];
            config.version = (v == "v2") ? voxmutatio::ModelVersion::kV2 : voxmutatio::ModelVersion::kV1;
        }
        else if (arg == "--speakers" && i + 1 < argc) config.num_speakers = std::atoi(argv[++i]);
        else if (arg == "--sr" && i + 1 < argc) config.model_sample_rate = std::atoi(argv[++i]);
        else if (arg == "--recursive") recursive = true;
        else if (arg == "--half") config.use_half_precision = true;
        else if (arg == "--device" && i + 1 < argc) config.device = argv[++i];
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    if (input_dir.empty() || output_dir.empty()) {
        std::cerr << "error: --input-dir and --output-dir are required\n";
        return 1;
    }
    if (config.hubert_model_path.empty()) {
        std::cerr << "error: --hubert is required\n";
        return 1;
    }
    if (config.synthesizer_model_path.empty()) {
        std::cerr << "error: --model is required\n";
        return 1;
    }

    // Initialize device
    voxmutatio::Device device;
    if (auto err = device.init(config.device, config.gpu_device)) {
        std::cerr << "CUDA init error: " << *err << "\n";
        return 1;
    }
    std::cout << "Device: " << device.name() << "\n";

    // Initialize pipeline
    voxmutatio::pipeline::VoiceConversionPipeline pipeline;
    if (!pipeline.init(config)) {
        std::cerr << "Pipeline init failed\n";
        return 1;
    }

    // Create output directory
    fs::create_directories(output_dir);

    // Collect audio files (optionally recursive), deterministic order.
    std::vector<std::string> audio_files;
    auto collect = [&](auto it) {
        for (const auto& entry : it) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".wav" || ext == ".flac") audio_files.push_back(entry.path().string());
        }
    };
    if (recursive) collect(fs::recursive_directory_iterator(input_dir));
    else collect(fs::directory_iterator(input_dir));
    std::sort(audio_files.begin(), audio_files.end());

    std::cout << "Found " << audio_files.size() << " audio files\n";
    
    int success_count = 0;
    int fail_count = 0;
    int idx = 0;

    for (const auto& input_path : audio_files) {
        ++idx;
        std::string output_path = output_dir + "/" + fs::path(input_path).stem().string() + ".wav";
        std::cout << "[" << idx << "/" << audio_files.size() << "] "
                  << fs::path(input_path).filename().string() << " ... " << std::flush;
        auto result = pipeline.convert_file(input_path, output_path, speaker_id);
        if (result.success) {
            ++success_count;
            std::cout << "ok (" << static_cast<long>(result.total_ms) << " ms)\n";
        } else {
            ++fail_count;
            std::cout << "FAILED: " << result.error_message << "\n";
        }
    }

    std::cout << "\nBatch complete: " << success_count << " succeeded, " 
              << fail_count << " failed\n";

    return fail_count > 0 ? 1 : 0;
}
