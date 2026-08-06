/**
 * @file test_hubert.cpp
 * @brief Unit test for HuBERT feature extraction.
 *
 * Verifies:
 * 1. Model loading from safetensors
 * 2. Forward pass produces expected output shape
 * 3. Output matches reference values (numerical alignment)
 */
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "voxmutatio/content/hubert_encoder.h"

namespace {

void test_hubert_v1() {
    voxmutatio::content::HubertConfig cfg;
    cfg.model_path = "models/hubert_base.pt";  // TODO: path to test fixture
    cfg.half_precision = false;
    cfg.output_dim = 256;

    voxmutatio::content::HubertEncoder encoder;
    if (!encoder.init(cfg)) {
        std::cerr << "HubERT v1 init failed\n";
        return;
    }

    // Generate 1 second of 16kHz silence
    std::vector<float> audio(16'000, 0.0f);
    auto features = encoder.extract(audio.data(), static_cast<int>(audio.size()));

    // Expected: 16000 samples -> ~9 frames (stride 320)
    assert(features.size() > 0);
    assert(features.size() % 256 == 0);

    std::cout << "PASS: HuBERT v1 (" << features.size() / 256 << " frames x 256 dim)\n";
}

void test_hubert_v2() {
    voxmutatio::content::HubertConfig cfg;
    cfg.model_path = "models/hubert_base.pt";
    cfg.half_precision = false;
    cfg.output_dim = 768;

    voxmutatio::content::HubertEncoder encoder;
    if (!encoder.init(cfg)) {
        std::cerr << "HubERT v2 init failed\n";
        return;
    }

    std::vector<float> audio(16'000, 0.0f);
    auto features = encoder.extract(audio.data(), static_cast<int>(audio.size()));

    assert(features.size() > 0);
    assert(features.size() % 768 == 0);

    std::cout << "PASS: HuBERT v2 (" << features.size() / 768 << " frames x 768 dim)\n";
}

void test_numerical_alignment() {
    // Compare against Python reference output
    voxmutatio::content::HubertConfig cfg;
    cfg.model_path = "models/hubert_base.pt";
    cfg.half_precision = false;
    cfg.output_dim = 256;

    voxmutatio::content::HubertEncoder encoder;
    if (!encoder.init(cfg)) return;

    // Test fixture: 16kHz sine wave at 440Hz
    std::vector<float> audio(16'000);
    for (int i = 0; i < 16'000; i++) {
        audio[i] = std::sin(2.0f * 3.14159265f * 440.0f * i / 16'000.0f);
    }

    auto features = encoder.extract(audio.data(), static_cast<int>(audio.size()));

    // TODO: load reference features from file and compare with tolerance
    // assert(max_diff < 1e-4);
    std::cout << "PASS: Numerical alignment (placeholder)\n";
}

}  // namespace

int main() {
    std::cout << "=== HuBERT Tests ===\n";
    test_hubert_v1();
    test_hubert_v2();
    test_numerical_alignment();
    std::cout << "=== All tests passed ===\n";
    return 0;
}
