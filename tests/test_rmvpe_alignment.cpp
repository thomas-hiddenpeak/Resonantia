/**
 * @file test_rmvpe_alignment.cpp
 * @brief RMVPE F0 extraction numerical alignment vs Python reference.
 *
 * SC-002: F0 error < 0.5 Hz (voiced frames) vs RVC RMVPE reference.
 */
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "voxmutatio/f0/rmvpe.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/io/safetensors.h"
#include "f0_checkpoint_fixture.h"
#include "test_artifacts.h"

namespace {

struct RefTensor {
    std::vector<int> shape;
    std::vector<float> data;
    bool ok = false;
};

RefTensor load_reference(const std::string& path) {
    RefTensor ref;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return ref;
    int32_t ndim = 0;
    ifs.read(reinterpret_cast<char*>(&ndim), 4);
    if (ndim <= 0 || ndim > 8) return ref;
    int64_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        int32_t s = 0;
        ifs.read(reinterpret_cast<char*>(&s), 4);
        ref.shape.push_back(s);
        total *= s;
    }
    ref.data.resize(total);
    ifs.read(reinterpret_cast<char*>(ref.data.data()), total * sizeof(float));
    ref.ok = ifs.good() || ifs.eof();
    return ref;
}

}  // namespace

TEST_CASE("RMVPE salience alignment", "[rmvpe][alignment]") {
    using namespace voxmutatio;

    std::string model_path = "../models/rmvpe.safetensors";
    std::string audio_path = "../tests/fixtures/speech_librispeech.wav";
    std::string ref_path = "../tests/fixtures/rmvpe_ref_salience.bin";

    auto audio = io::read_audio(audio_path);
    REQUIRE(audio.has_value());

    auto ref = load_reference(ref_path);
    REQUIRE(ref.ok);
    REQUIRE(ref.shape.size() == 2);
    int ref_T = ref.shape[0];
    int ref_C = ref.shape[1];
    std::cout << "Salience reference: [" << ref_T << ", " << ref_C << "]" << std::endl;
    REQUIRE(ref_C == 360);

    f0::RmvpeConfig config;
    config.model_path = model_path;

    f0::RmvpeExtractor extractor;
    REQUIRE(extractor.init(config));

    int T = 0;
    auto salience = extractor.infer_salience(audio->data.data(),
                                             static_cast<int>(audio->data.size()), T);
    std::cout << "C++ salience: [" << T << ", 360]" << std::endl;
    REQUIRE(T == ref_T);
    REQUIRE(salience.size() == ref.data.size());

    // RMS error over salience
    double sum_sq = 0.0, max_diff = 0.0;
    for (size_t i = 0; i < salience.size(); ++i) {
        double d = static_cast<double>(salience[i]) - ref.data[i];
        sum_sq += d * d;
        max_diff = std::max(max_diff, std::abs(d));
    }
    double rms = std::sqrt(sum_sq / salience.size());
    std::cout << "Salience RMS error: " << rms << ", max diff: " << max_diff << std::endl;

    CHECK(rms < 1e-3);
}

TEST_CASE("RMVPE F0 alignment (SC-002)", "[rmvpe][alignment][f0]") {
    using namespace voxmutatio;

    std::string model_path = "../models/rmvpe.safetensors";
    std::string audio_path = "../tests/fixtures/speech_librispeech.wav";
    std::string ref_path = "../tests/fixtures/rmvpe_ref_f0.bin";

    auto audio = io::read_audio(audio_path);
    REQUIRE(audio.has_value());

    auto ref = load_reference(ref_path);
    REQUIRE(ref.ok);
    int ref_T = ref.shape[0];
    std::cout << "F0 reference: " << ref_T << " frames" << std::endl;

    f0::RmvpeConfig config;
    config.model_path = model_path;

    f0::RmvpeExtractor extractor;
    REQUIRE(extractor.init(config));

    auto f0v = extractor.infer(audio->data.data(),
                               static_cast<int>(audio->data.size()));
    std::cout << "C++ F0: " << f0v.size() << " frames" << std::endl;
    REQUIRE(static_cast<int>(f0v.size()) == ref_T);

    // Compare voiced frames (where reference F0 > 0)
    int voiced = 0, matched = 0;
    double max_err = 0.0, sum_err = 0.0;
    for (int t = 0; t < ref_T; ++t) {
        float ref_f0 = ref.data[t];
        if (ref_f0 > 0) {
            voiced++;
            double err = std::abs(f0v[t] - ref_f0);
            sum_err += err;
            max_err = std::max(max_err, err);
            if (err < 0.5) matched++;
        }
    }
    double mean_err = voiced > 0 ? sum_err / voiced : 0.0;
    std::cout << "Voiced frames: " << voiced << std::endl;
    std::cout << "Matched (<0.5Hz): " << matched << "/" << voiced << std::endl;
    std::cout << "Mean F0 error: " << mean_err << " Hz, max: " << max_err << " Hz" << std::endl;

    // SC-002: mean F0 error < 0.5 Hz on voiced frames
    CHECK(mean_err < 0.5);
    // Most voiced frames should match tightly
    CHECK(matched >= voiced * 0.95);
}

TEST_CASE("RMVPE initialization and inference failure boundaries",
          "[rmvpe][integration][contract]") {
    using namespace voxmutatio;

    f0::RmvpeConfig config;
    config.model_path = "../models/rmvpe.safetensors";
    f0::RmvpeExtractor extractor;
    REQUIRE(extractor.init(config));

    float finite_sample = 0.0f;
    float nonfinite_sample = std::numeric_limits<float>::quiet_NaN();
    int frames = 19;
    CHECK(extractor.infer(nullptr, 1).empty());
    CHECK(extractor.infer(&finite_sample, 0).empty());
    CHECK(extractor.infer(&finite_sample, std::numeric_limits<int>::max()).empty());
    CHECK(extractor.infer(&nonfinite_sample, 1).empty());
    CHECK(extractor.infer_salience(&nonfinite_sample, 1, frames).empty());
    CHECK(frames == 0);

    resonantia_test::ScopedArtifact sparse_checkpoint(
        resonantia_test::unique_artifact_path("rmvpe-complete-sparse", ".safetensors"));
    REQUIRE(resonantia_test::f0_fixture::create_zero_payload_sparse_clone(
        "../models/rmvpe.safetensors", sparse_checkpoint.path()));
    {
        f0::RmvpeExtractor manifest_probe;
        f0::RmvpeConfig sparse_config;
        sparse_config.model_path = sparse_checkpoint.path().string();
        REQUIRE(manifest_probe.init(sparse_config));
    }
    REQUIRE(resonantia_test::f0_fixture::write_tensor_value(
        sparse_checkpoint.path(), "cnn.weight", 0,
        std::numeric_limits<float>::quiet_NaN()));
    {
        io::SafetensorsLoader generic_loader;
        REQUIRE(generic_loader.load(sparse_checkpoint.path().string()));
    }

    config.model_path = sparse_checkpoint.path().string();
    CHECK_FALSE(extractor.init(config));
    CHECK(extractor.infer(&finite_sample, 1).empty());
    frames = 19;
    CHECK(extractor.infer_salience(&finite_sample, 1, frames).empty());
    CHECK(frames == 0);
}
