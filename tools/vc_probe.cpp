/**
 * @file vc_probe.cpp
 * @brief Numerical alignment probe tool.
 *
 * Prints HuBERT feature dimensions and summary statistics for a WAV input.
 */
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "voxmutatio/core/types.h"
#include "voxmutatio/core/device.h"
#include "voxmutatio/content/hubert_encoder.h"
#include "voxmutatio/io/audio_io.h"

namespace {

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  vc_probe --hubert <path> --input <wav>\n\n"
        << "Options:\n"
        << "  --hubert <path>       HuBERT model safetensors directory\n"
        << "  --input <wav>         Input audio file\n"
        << "  --device <cuda>       CUDA FP32 is the supported runtime\n"
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

    std::string hubert_path, input_path;
    std::string device = "cuda";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--hubert" && i + 1 < argc) hubert_path = argv[++i];
        else if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (arg == "--reference") {
            std::cerr << "error: NPZ reference comparison is not implemented\n";
            return 1;
        }
        else if (arg == "--device" && i + 1 < argc) {
            device = argv[++i];
            if (device != "cuda") {
                std::cerr << "error: only the CUDA backend is implemented\n";
                return 1;
            }
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    if (hubert_path.empty() || input_path.empty()) {
        std::cerr << "error: --hubert and --input are required\n";
        return 1;
    }

    // Initialize device
    voxmutatio::Device device_mgr;
    if (auto err = device_mgr.init(device)) {
        std::cerr << "Device init error: " << *err << "\n";
        return 1;
    }
    std::cout << "Device: " << device_mgr.name() << "\n";

    // Load audio
    auto audio = voxmutatio::io::read_audio(input_path, 16000);
    if (!audio) {
        std::cerr << "Failed to load audio: " << input_path << "\n";
        return 1;
    }
    std::cout << "Audio: " << audio->num_samples() << " samples, "
              << audio->duration_s() << "s\n";

    // Initialize HuBERT encoder
    voxmutatio::content::HubertConfig hubert_cfg;
    hubert_cfg.model_path = hubert_path;
    hubert_cfg.output_dim = 256;  // v1
    
    voxmutatio::content::HubertEncoder encoder;
    if (!encoder.init(hubert_cfg)) {
        std::cerr << "Failed to initialize HuBERT encoder\n";
        return 1;
    }

    // Extract features
    std::cout << "Extracting features...\n";
    auto features = encoder.extract(audio->data.data(), 
                                     static_cast<int>(audio->num_samples()));
    if (features.empty()) {
        std::cerr << "Feature extraction failed\n";
        return 1;
    }
    
    int num_frames = static_cast<int>(features.size()) / encoder.output_dim();
    std::cout << "Features: " << num_frames << " frames x " 
              << encoder.output_dim() << " dim\n";

    // Print feature statistics
    double mean = 0.0;
    double std_dev = 0.0;
    for (const auto& f : features) {
        mean += f;
    }
    mean /= features.size();
    
    for (const auto& f : features) {
        std_dev += (f - mean) * (f - mean);
    }
    std_dev = std::sqrt(std_dev / features.size());
    
    std::cout << "\n=== Feature Statistics ===\n";
    std::cout << "Mean: " << mean << "\n";
    std::cout << "StdDev: " << std_dev << "\n";

    return 0;
}
