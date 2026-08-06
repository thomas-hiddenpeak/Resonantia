/**
 * @file vc_convert.cpp
 * @brief Single-file offline voice conversion CLI tool.
 *
 * Usage:
 *   vc_convert --hubert <path> --model <path> --input <wav> --output <wav> [options]
 */
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <filesystem>

#include "voxmutatio/core/types.h"
#include "voxmutatio/core/device.h"
#include "voxmutatio/pipeline/pipeline.h"

namespace {

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  vc_convert --hubert <path> --model <path> --input <wav> --output <wav> [options]\n\n"
        << "Options:\n"
        << "  --hubert <path>       HuBERT model safetensors directory\n"
        << "  --model <path>        VITS synthesizer safetensors path\n"
        << "  --input <wav>         Input audio file path\n"
        << "  --output <wav>        Output audio file path\n"
        << "  --index <path>        CUDA Flat Index file (optional)\n"
        << "  --rmvpe <path>        RMVPE model safetensors (optional)\n"
        << "  --speaker <id>        Speaker ID (default: 0)\n"
        << "  --pitch <semitones>   Pitch shift in semitones (default: 0)\n"
        << "  --formant <semitones> Formant shift (default: 0)\n"
        << "  --index-rate <0-1>    Index blend rate (default: 0.0)\n"
        << "  --rms-mix <0-1>       RMS energy mix rate (default: 0.5)\n"
        << "  --protect <0-1>       Unvoiced protection (default: 0.5)\n"
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

    // Check for --help
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
    }

    voxmutatio::VCConfig config;
    std::string input_path;
    std::string output_path;
    int speaker_id = 0;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--hubert" && i + 1 < argc) config.hubert_model_path = argv[++i];
        else if (arg == "--model" && i + 1 < argc) config.synthesizer_model_path = argv[++i];
        else if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--index" && i + 1 < argc) config.index_path = argv[++i];
        else if (arg == "--rmvpe" && i + 1 < argc) config.rmvpe_model_path = argv[++i];
        else if (arg == "--speaker" && i + 1 < argc) speaker_id = std::atoi(argv[++i]);
        else if (arg == "--pitch" && i + 1 < argc) config.f0_up_key = std::atoi(argv[++i]);
        else if (arg == "--formant" && i + 1 < argc) config.formant_shift = std::atof(argv[++i]);
        else if (arg == "--index-rate" && i + 1 < argc) config.index_rate = std::atof(argv[++i]);
        else if (arg == "--rms-mix" && i + 1 < argc) config.rms_mix_rate = std::atof(argv[++i]);
        else if (arg == "--protect" && i + 1 < argc) config.protect = std::atof(argv[++i]);
        else if (arg == "--half") config.use_half_precision = true;
        else if (arg == "--device" && i + 1 < argc) config.device = argv[++i];
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    // Validate required args
    if (config.hubert_model_path.empty()) {
        std::cerr << "error: --hubert is required\n";
        return 1;
    }
    if (config.synthesizer_model_path.empty()) {
        std::cerr << "error: --model is required\n";
        return 1;
    }
    if (input_path.empty()) {
        std::cerr << "error: --input is required\n";
        return 1;
    }
    if (output_path.empty()) {
        std::cerr << "error: --output is required\n";
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

    // Run conversion
    std::cout << "Converting: " << input_path << " -> " << output_path << "\n";
    auto result = pipeline.convert_file(input_path, output_path, speaker_id);

    if (!result.success) {
        std::cerr << "Conversion failed: " << result.error_message << "\n";
        return 1;
    }

    std::cout << "Done (" << result.total_ms << " ms)\n"
              << "  HuBERT:  " << result.hubert_ms << " ms\n"
              << "  F0:      " << result.f0_ms << " ms\n"
              << "  Index:   " << result.index_ms << " ms\n"
              << "  Synth:   " << result.synth_ms << " ms\n";
    return 0;
}
