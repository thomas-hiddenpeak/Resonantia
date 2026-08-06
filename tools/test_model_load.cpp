/**
 * @file test_model_load.cpp
 * @brief Test safetensors model loading.
 */
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "voxmutatio/io/safetensors.h"

namespace {

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  test_model_load <path.safetensors>\n\n"
        << "Prints model info from safetensors file.\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string path = argv[1];
    
    voxmutatio::io::SafetensorsLoader loader;
    
    std::cout << "Loading: " << path << std::endl;
    
    if (!loader.load(path)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << loader.tensor_names().size() << " tensors" << std::endl;
    std::cout << "Total size: " << loader.data_size() / 1024 / 1024 << " MB" << std::endl;
    
    // Print first 20 tensors
    int count = 0;
    for (const auto& name : loader.tensor_names()) {
        if (count >= 20) break;
        
        const auto* tensor = loader.get_tensor(name);
        if (!tensor) continue;
        
        std::string shape_str;
        for (size_t i = 0; i < tensor->shape.size(); ++i) {
            if (i > 0) shape_str += " x ";
            shape_str += std::to_string(tensor->shape[i]);
        }
        
        std::cout << "  " << name << ": [" << shape_str << "]" << std::endl;
        ++count;
    }
    
    if (loader.tensor_names().size() > 20) {
        std::cout << "  ... (" << (loader.tensor_names().size() - 20) << " more)" << std::endl;
    }
    
    return 0;
}
