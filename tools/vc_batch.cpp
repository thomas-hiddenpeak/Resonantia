/**
 * @file vc_batch.cpp
 * @brief Batch voice conversion CLI tool.
 *
 * Usage:
 *   vc_batch --input-dir <dir> --output-dir <dir> [options]
 */
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

    // Collect audio files
    std::vector<std::string> audio_files;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        std::string ext = entry.path().extension().string();
        if (ext == ".wav" || ext == ".flac") {
            audio_files.push_back(entry.path().string());
        }
    }

    std::cout << "Found " << audio_files.size() << " audio files\n";
    
    int success_count = 0;
    int fail_count = 0;

    for (const auto& input_path : audio_files) {
        std::string output_path = output_dir + "/" + fs::path(input_path).stem().string() + ".wav";
        
        std::cout << "Converting: " << fs::path(input_path).filename().string() << "\r\033[K";
        
        auto result = pipeline.convert_file(input_path, output_path, speaker_id);
        
        if (result.success) {
            success_count++;
        } else {
            fail_count++;
            std::cerr << "\nFailed: " << input_path << " - " << result.error_message << "\n";
        }
    }

    std::cout << "\nBatch complete: " << success_count << " succeeded, " 
              << fail_count << " failed\n";

    return fail_count > 0 ? 1 : 0;
}
