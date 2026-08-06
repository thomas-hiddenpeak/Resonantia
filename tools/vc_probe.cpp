/**
 * @file vc_probe.cpp
 * @brief Numerical alignment probe tool.
 *
 * Usage:
 *   vc_probe --hubert <path> --input <wav> [--reference <npz>]
 *
 * Verifies C++ implementation matches Python reference output.
 */
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <numeric>

#include "voxmutatio/core/types.h"
#include "voxmutatio/core/device.h"
#include "voxmutatio/content/hubert_encoder.h"
#include "voxmutatio/io/audio_io.h"

namespace {

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  vc_probe --hubert <path> --input <wav> [--reference <npz>]\n\n"
        << "Options:\n"
        << "  --hubert <path>       HuBERT model safetensors directory\n"
        << "  --input <wav>         Input audio file\n"
        << "  --reference <npz>     Python reference output (optional)\n"
        << "  --device <cuda|cpu>   Compute device (default: cuda)\n"
        << "  --help                Show this help\n";
}

// Compute L2 distance between two vectors
double l2_distance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return -1.0;
    }
    
    double sum_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}

// Compute mean absolute error
double mean_absolute_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return -1.0;
    }
    
    double sum_abs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum_abs += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return sum_abs / a.size();
}

// Simple NPZ reader for single array files
std::vector<float> read_npz_array(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return {};
    }
    
    // This is a simplified NPZ reader
    // Full implementation would parse ZIP format and NumPy headers
    std::cout << "Warning: NPZ reference loading not yet implemented\n";
    return {};
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

    std::string hubert_path, input_path, reference_path;
    std::string device = "cuda";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--hubert" && i + 1 < argc) hubert_path = argv[++i];
        else if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (arg == "--reference" && i + 1 < argc) reference_path = argv[++i];
        else if (arg == "--device" && i + 1 < argc) device = argv[++i];
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
    
    int num_frames = static_cast<int>(features.size()) / encoder.output_dim();
    std::cout << "Features: " << num_frames << " frames x " 
              << encoder.output_dim() << " dim\n";

    // Compare with reference if provided
    if (!reference_path.empty()) {
        auto reference = read_npz_array(reference_path);
        
        if (reference.empty()) {
            std::cerr << "Failed to load reference\n";
            return 1;
        }
        
        if (features.size() != reference.size()) {
            std::cerr << "Size mismatch: " << features.size() 
                      << " vs " << reference.size() << "\n";
            return 1;
        }
        
        double l2 = l2_distance(features, reference);
        double mae = mean_absolute_error(features, reference);
        
        std::cout << "\n=== Numerical Alignment ===\n";
        std::cout << "L2 Distance: " << l2 << " (target: < 1e-4)\n";
        std::cout << "MAE: " << mae << " (target: < 1e-4)\n";
        
        bool pass = (l2 < 1e-4);
        std::cout << "Status: " << (pass ? "PASS" : "FAIL") << "\n";
        
        return pass ? 0 : 1;
    }

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
