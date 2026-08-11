// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/catch_approx.hpp>

#include "voxmutatio/pipeline/pipeline.h"
#include "voxmutatio/core/types.h"
#include "voxmutatio/io/audio_io.h"
#include "test_artifacts.h"
#include "vits_checkpoint_fixture.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <limits>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

TEST_CASE("Pipeline initialization", "[pipeline][init]") {
    voxmutatio::VCConfig config;
    config.device = "cpu";  // Use CPU for testing
    
    voxmutatio::pipeline::VoiceConversionPipeline pipeline;
    
    // Init should fail without model paths (expected)
    bool result = pipeline.init(config);
    
    REQUIRE_FALSE(result);
}

namespace {

voxmutatio::VCConfig real_config(voxmutatio::ModelVersion version) {
    voxmutatio::VCConfig config;
    config.hubert_model_path = "../models/hubert_base/model.safetensors";
    config.rmvpe_model_path = "../models/rmvpe.safetensors";
    config.synthesizer_model_path = version == voxmutatio::ModelVersion::kV1
        ? "../models/pretrained/pretrained/f0G40k.safetensors"
        : "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    config.version = version;
    config.has_f0 = true;
    config.model_sample_rate = 40'000;
    return config;
}

bool write_constant_index(const fs::path& path, std::int64_t rows,
                          std::int64_t dimensions, float value = 0.0f) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return false;
    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    const std::vector<float> values(
        static_cast<std::size_t>(rows * dimensions), value);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    return static_cast<bool>(output);
}

class ScopedCurrentPath {
public:
    explicit ScopedCurrentPath(const fs::path& replacement)
        : original_(fs::current_path()) {
        fs::current_path(replacement);
    }
    ScopedCurrentPath(const ScopedCurrentPath&) = delete;
    ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;
    ~ScopedCurrentPath() {
        std::error_code ignored;
        fs::current_path(original_, ignored);
    }

private:
    fs::path original_;
};

}  // namespace

TEST_CASE("Pipeline derives v1/v2 feature dimensions from the model version",
          "[pipeline][init][contract]") {
    voxmutatio::pipeline::VoiceConversionPipeline v1;
    REQUIRE(v1.init(real_config(voxmutatio::ModelVersion::kV1)));
    CHECK(v1.feature_dim() == 256);
    CHECK(v1.num_speakers() == 109);

    voxmutatio::pipeline::VoiceConversionPipeline v2;
    REQUIRE(v2.init(real_config(voxmutatio::ModelVersion::kV2)));
    CHECK(v2.feature_dim() == 768);
    CHECK(v2.num_speakers() == 109);
}

TEST_CASE("Pipeline rejects unsupported or mismatched inference contracts",
          "[pipeline][init][contract]") {
    SECTION("non-F0 models are rejected at init") {
        auto config = real_config(voxmutatio::ModelVersion::kV2);
        config.has_f0 = false;
        voxmutatio::pipeline::VoiceConversionPipeline pipeline;
        REQUIRE_FALSE(pipeline.init(config));
    }

    SECTION("checkpoint version must match configured version") {
        auto config = real_config(voxmutatio::ModelVersion::kV1);
        config.synthesizer_model_path =
            "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
        voxmutatio::pipeline::VoiceConversionPipeline pipeline;
        REQUIRE_FALSE(pipeline.init(config));
    }

    SECTION("failed reinitialization invalidates the previous pipeline") {
        voxmutatio::pipeline::VoiceConversionPipeline pipeline;
        REQUIRE(pipeline.init(real_config(voxmutatio::ModelVersion::kV2)));
        auto invalid = real_config(voxmutatio::ModelVersion::kV2);
        invalid.max_chunk_sec = 0.0;
        REQUIRE_FALSE(pipeline.init(invalid));
        voxmutatio::AudioBuffer input;
        input.sample_rate = 16'000;
        input.data.assign(400, 0.0f);
        const auto result = pipeline.convert_buffer(input);
        CHECK_FALSE(result.success);
        CHECK(result.error_message == "pipeline not initialized");
    }

    SECTION("unimplemented runtime options are rejected") {
        const auto require_rejected = [](voxmutatio::VCConfig config) {
            voxmutatio::pipeline::VoiceConversionPipeline pipeline;
            REQUIRE_FALSE(pipeline.init(config));
        };

        auto config = real_config(voxmutatio::ModelVersion::kV2);
        config.device = "cpu";
        require_rejected(config);
        config = real_config(voxmutatio::ModelVersion::kV2);
        config.use_half_precision = true;
        require_rejected(config);
        config = real_config(voxmutatio::ModelVersion::kV2);
        config.formant_shift = 1.0;
        require_rejected(config);
        config = real_config(voxmutatio::ModelVersion::kV2);
        config.target_sample_rate = 48'000;
        require_rejected(config);
        config = real_config(voxmutatio::ModelVersion::kV2);
        config.f0_method = voxmutatio::F0Method::kPm;
        require_rejected(config);
    }

    SECTION("out-of-domain enum values are rejected") {
        const auto require_rejected = [](voxmutatio::VCConfig config) {
            voxmutatio::pipeline::VoiceConversionPipeline pipeline;
            REQUIRE_FALSE(pipeline.init(config));
        };

        auto config = real_config(voxmutatio::ModelVersion::kV2);
        config.version = static_cast<voxmutatio::ModelVersion>(0xff);
        require_rejected(config);
        config = real_config(voxmutatio::ModelVersion::kV2);
        config.content_encoder = static_cast<voxmutatio::ContentEncoder>(0xff);
        require_rejected(config);
        config = real_config(voxmutatio::ModelVersion::kV2);
        config.f0_method = static_cast<voxmutatio::F0Method>(0xff);
        require_rejected(config);
    }

    SECTION("unbounded, non-finite, and unsupported I/O controls are rejected") {
        const auto require_rejected = [](voxmutatio::VCConfig config) {
            voxmutatio::pipeline::VoiceConversionPipeline pipeline;
            REQUIRE_FALSE(pipeline.init(config));
        };

        auto config = real_config(voxmutatio::ModelVersion::kV2);
        config.max_chunk_sec = 0.0;
        require_rejected(config);
        config = real_config(voxmutatio::ModelVersion::kV2);
        config.index_rate = std::numeric_limits<double>::quiet_NaN();
        require_rejected(config);
        config = real_config(voxmutatio::ModelVersion::kV2);
        config.output_format = "flac";
        require_rejected(config);
        config = real_config(voxmutatio::ModelVersion::kV2);
        config.filter_radius = 4;
        require_rejected(config);
        config = real_config(voxmutatio::ModelVersion::kV2);
        config.model_sample_rate = 0;
        require_rejected(config);
    }
}

TEST_CASE("Pipeline rejects an explicit index with the wrong feature width",
          "[pipeline][index][contract]") {
    const auto require_rejected = [](voxmutatio::ModelVersion version,
                                     int index_dimensions) {
        resonantia_test::ScopedArtifact wrong_index(
            resonantia_test::unique_artifact_path("wrong-width-index", ".index"));
        REQUIRE(write_constant_index(wrong_index.path(), 1, index_dimensions));

        auto config = real_config(version);
        config.index_path = wrong_index.path().string();
        config.index_rate = 1.0;

        voxmutatio::pipeline::VoiceConversionPipeline pipeline;
        CHECK_FALSE(pipeline.init(config));
    };

    SECTION("v2 rejects a v1-width index") {
        require_rejected(voxmutatio::ModelVersion::kV2, 256);
    }
    SECTION("v1 rejects a v2-width index") {
        require_rejected(voxmutatio::ModelVersion::kV1, 768);
    }
}

TEST_CASE("Pipeline reports a runtime index lookup failure instead of bypassing it",
          "[pipeline][index][contract]") {
    resonantia_test::ScopedArtifact overflowing_index(
        resonantia_test::unique_artifact_path("overflowing-index", ".index"));
    REQUIRE(write_constant_index(overflowing_index.path(), 8, 768,
                                 std::numeric_limits<float>::max()));

    auto config = real_config(voxmutatio::ModelVersion::kV2);
    config.index_path = overflowing_index.path().string();
    config.index_rate = 1.0;

    voxmutatio::pipeline::VoiceConversionPipeline pipeline;
    REQUIRE(pipeline.init(config));
    auto input = voxmutatio::io::read_audio(
        "../tests/fixtures/speech_librispeech.wav", 16'000);
    REQUIRE(input.has_value());
    input->data.resize(std::min<std::size_t>(input->data.size(), 16'000));

    const auto result = pipeline.convert_buffer(*input, 0);
    CHECK_FALSE(result.success);
    CHECK(result.error_message == "index retrieval failed");
    CHECK(result.index_ms > 0.0);
}

TEST_CASE("Pipeline rejects a nonzero index rate without a usable index",
          "[pipeline][index][contract]") {
    auto config = real_config(voxmutatio::ModelVersion::kV2);
    config.index_rate = 1.0;

    voxmutatio::pipeline::VoiceConversionPipeline pipeline;
    REQUIRE_FALSE(pipeline.init(config));

    config.index_rate = 0.0;
    REQUIRE(pipeline.init(config));
    CHECK_FALSE(pipeline.set_runtime_params(0, 0.0, 0.5, 0.5, 0.5, 3));
    CHECK(pipeline.config().index_rate == Catch::Approx(0.0));
}

TEST_CASE("Synthesizer validates speaker and pitch bounds before inference",
          "[synthesizer][contract]") {
    using voxmutatio::ModelVersion;
    using voxmutatio::synthesizer::Synthesizer;
    using voxmutatio::synthesizer::SynthesizerConfig;

    SynthesizerConfig config;
    config.model_path = "../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    config.version = ModelVersion::kV2;
    REQUIRE(config.has_f0);

    Synthesizer synth;
    REQUIRE(synth.init(config));
    REQUIRE(synth.feature_dim() == 768);
    REQUIRE(synth.num_speakers() == 109);

    std::vector<float> features(768, 0.0f);
    float pitch = 1.0f;
    float f0 = 110.0f;
    CHECK(synth.infer(features.data(), 1, &pitch, &f0, -1).data.empty());
    CHECK(synth.infer(features.data(), 1, &pitch, &f0, synth.num_speakers()).data.empty());

    pitch = 256.0f;
    CHECK(synth.infer(features.data(), 1, &pitch, &f0, 0).data.empty());
    pitch = 1.0f;
    CHECK(synth.infer(features.data(), 1, nullptr, &f0, 0).data.empty());
    CHECK(synth.infer(features.data(), 1, &pitch, nullptr, 0).data.empty());

    CHECK(synth.infer_stream(features.data(), 1, &pitch, &f0, 0, -1, 1).data.empty());
    CHECK(synth.infer_stream(features.data(), 1, &pitch, &f0, 0, 0, 0).data.empty());
    auto one_frame = synth.infer_stream(features.data(), 1, &pitch, &f0, 0, 0, 1);
    CHECK(one_frame.sample_rate == 40'000);
    CHECK(one_frame.data.size() == 400);

    config.version = static_cast<ModelVersion>(0xff);
    Synthesizer invalid_version;
    CHECK_FALSE(invalid_version.init(config));
}

TEST_CASE("Synthesizer fail-closes checkpoint and public numerical boundaries",
          "[synthesizer][boundary]") {
    using voxmutatio::ModelVersion;
    using voxmutatio::synthesizer::Synthesizer;
    using voxmutatio::synthesizer::SynthesizerConfig;

    resonantia_test::ScopedArtifact checkpoint_path(
        resonantia_test::unique_artifact_path("vits-sparse", ".safetensors"));
    resonantia_test::vits_fixture::SparseCheckpoint checkpoint;
    REQUIRE(resonantia_test::vits_fixture::create_sparse_checkpoint(
        checkpoint_path.path(), checkpoint));

    SynthesizerConfig config;
    config.model_path = fs::relative(
        checkpoint_path.path(), fs::current_path()).string();
    REQUIRE_FALSE(config.model_path.empty());
    config.version = ModelVersion::kV2;
    Synthesizer synth;
    REQUIRE(synth.init(config));
    REQUIRE(synth.initialized());
    REQUIRE(synth.num_speakers() == 1);

    const auto require_config_rejected = [&](SynthesizerConfig rejected) {
        Synthesizer invalid;
        CHECK_FALSE(invalid.init(rejected));
        CHECK_FALSE(invalid.initialized());
    };
    auto rejected = config;
    rejected.half_precision = true;
    require_config_rejected(rejected);
    rejected = config;
    rejected.spec_channels = 513;
    require_config_rejected(rejected);
    rejected = config;
    rejected.segment_size = 0;
    require_config_rejected(rejected);
    rejected = config;
    rejected.p_dropout = 1;
    require_config_rejected(rejected);
    // Every runtime-used tensor is part of init validation, including tensors
    // at the beginning and end of the decoder/encoder manifest.
    REQUIRE(checkpoint.write_value(
        checkpoint_path.path(), "model.enc_p.emb_phone.bias", 0,
        std::numeric_limits<float>::quiet_NaN()));
    CHECK_FALSE(synth.init(config));
    CHECK_FALSE(synth.initialized());
    CHECK(synth.num_speakers() == 0);
    REQUIRE(checkpoint.write_value(
        checkpoint_path.path(), "model.enc_p.emb_phone.bias", 0, 0.0f));
    REQUIRE(synth.init(config));

    REQUIRE(checkpoint.write_value(
        checkpoint_path.path(),
        "model.flow.flows.0.enc.in_layers.0.weight_v", 0, 0.0f));
    CHECK_FALSE(synth.init(config));
    CHECK_FALSE(synth.initialized());
    REQUIRE(checkpoint.write_value(
        checkpoint_path.path(),
        "model.flow.flows.0.enc.in_layers.0.weight_v", 0, 1.0f));
    REQUIRE(synth.init(config));

    REQUIRE(checkpoint.write_value(
        checkpoint_path.path(), "model.dec.conv_post.weight", 0,
        std::numeric_limits<float>::infinity()));
    CHECK_FALSE(synth.init(config));
    CHECK_FALSE(synth.initialized());
    REQUIRE(checkpoint.write_value(
        checkpoint_path.path(), "model.dec.conv_post.weight", 0, 0.0f));
    REQUIRE(synth.init(config));

    std::vector<float> features(static_cast<std::size_t>(synth.feature_dim()), 0.0f);
    float pitch = 1.0f;
    float f0 = 110.0f;
    const auto valid = synth.infer(features.data(), 1, &pitch, &f0, 0);
    REQUIRE(valid.data.size() == 400);
    CHECK(std::all_of(valid.data.begin(), valid.data.end(),
                      [](float value) { return std::isfinite(value); }));

    voxmutatio::AudioBuffer after_cwd_change;
    {
        ScopedCurrentPath changed_cwd(fs::current_path() / "fixtures");
        after_cwd_change = synth.infer(features.data(), 1, &pitch, &f0, 0);
    }
    REQUIRE(after_cwd_change.data.size() == 400);

    Synthesizer copied = synth;
    REQUIRE(copied.initialized());
    REQUIRE(copied.infer(features.data(), 1, &pitch, &f0, 0).data.size() == 400);
    Synthesizer moved = std::move(copied);
    CHECK_FALSE(copied.initialized());
    CHECK(copied.infer(features.data(), 1, &pitch, &f0, 0).data.empty());
    REQUIRE(moved.initialized());
    REQUIRE(moved.infer(features.data(), 1, &pitch, &f0, 0).data.size() == 400);

    // The frame-arithmetic gate runs before scanning any caller buffer. A
    // single-value sentinel is therefore safe for a rejected oversized frame
    // count in both the full and legacy windowed public entry points.
    float sentinel = 0.0f;
    constexpr int kOverflowingAttentionFrames = 40'000;
    CHECK(synth.infer(&sentinel, kOverflowingAttentionFrames, &sentinel,
                      &sentinel, 0).data.empty());
    CHECK(synth.infer_stream(&sentinel, kOverflowingAttentionFrames, &sentinel,
                             &sentinel, 0, 0, 1).data.empty());

    int pitch_sentinel = 1;
    CHECK(synth.debug_text_encoder(
        &sentinel, kOverflowingAttentionFrames, &pitch_sentinel).empty());
    CHECK(synth.debug_flow(
        &sentinel, kOverflowingAttentionFrames, &pitch_sentinel, 0).empty());
    CHECK(synth.debug_encp(
        &sentinel, kOverflowingAttentionFrames, &pitch_sentinel, false).empty());

    constexpr int kOverflowingHarFrames =
        std::numeric_limits<int>::max() / 400 + 1;
    constexpr int kOverflowingConvPreFrames =
        std::numeric_limits<int>::max() / 512 + 1;
    constexpr int kOverflowingStageFrames =
        std::numeric_limits<int>::max() / 2'560 + 1;
    constexpr int kOverflowingGeneratorFrames =
        std::numeric_limits<int>::max() / 12'800 + 1;
    constexpr int kOverflowingFlowFrames =
        std::numeric_limits<int>::max() / 768 + 1;
    CHECK(synth.debug_har(&sentinel, kOverflowingHarFrames).empty());
    CHECK(synth.debug_convpre(&sentinel, kOverflowingConvPreFrames, 0).empty());
    CHECK(synth.debug_gen_stage0(&sentinel, &sentinel,
                                 kOverflowingStageFrames, 0, 0).empty());
    CHECK(synth.debug_decode(&sentinel, &sentinel,
                             kOverflowingGeneratorFrames, 0).empty());
    CHECK(synth.debug_flow_reverse(
        &sentinel, kOverflowingFlowFrames, 0).empty());

    features[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(synth.debug_text_encoder(features.data(), 1, &pitch_sentinel).empty());
    CHECK(synth.debug_flow(features.data(), 1, &pitch_sentinel, 0).empty());
    CHECK(synth.debug_encp(features.data(), 1, &pitch_sentinel, false).empty());
    features[0] = 0.0f;
    std::vector<float> z(192, 0.0f);
    z[0] = std::numeric_limits<float>::infinity();
    CHECK(synth.debug_convpre(z.data(), 1, 0).empty());
    CHECK(synth.debug_gen_stage0(z.data(), &f0, 1, 0, 0).empty());
    CHECK(synth.debug_decode(z.data(), &f0, 1, 0).empty());
    CHECK(synth.debug_flow_reverse(z.data(), 1, 0).empty());
    const float nonfinite_f0 = std::numeric_limits<float>::quiet_NaN();
    CHECK(synth.debug_har(&nonfinite_f0, 1).empty());

    for (const float nonfinite : {
             std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::infinity()}) {
        features[0] = nonfinite;
        CHECK(synth.infer(features.data(), 1, &pitch, &f0, 0).data.empty());
        features[0] = 0.0f;

        pitch = nonfinite;
        CHECK(synth.infer(features.data(), 1, &pitch, &f0, 0).data.empty());
        pitch = 1.0f;

        f0 = nonfinite;
        CHECK(synth.infer(features.data(), 1, &pitch, &f0, 0).data.empty());
        f0 = 110.0f;
    }

    // Finite caller data and weights can still overflow intermediate FP32.
    // The public boundary must not return the resulting NaN/Inf waveform.
    REQUIRE(checkpoint.write_value(
        checkpoint_path.path(), "model.enc_p.emb_phone.weight", 0,
        std::numeric_limits<float>::max()));
    REQUIRE(synth.init(config));
    features[0] = std::numeric_limits<float>::max();
    CHECK(synth.infer(features.data(), 1, &pitch, &f0, 0).data.empty());

    // Reconstruction uses double g*v/norm. The mathematically valid result is
    // finite even though the old float multiplication order overflowed first.
    REQUIRE(checkpoint.write_value(
        checkpoint_path.path(), "model.enc_p.emb_phone.weight", 0, 0.0f));
    REQUIRE(checkpoint.write_value(
        checkpoint_path.path(), "model.dec.conv_pre.bias", 0, 1.0f));
    REQUIRE(checkpoint.write_value(
        checkpoint_path.path(), "model.dec.ups.0.weight_g", 0,
        std::numeric_limits<float>::max() / 4.0f));
    REQUIRE(checkpoint.write_value(
        checkpoint_path.path(), "model.dec.ups.0.weight_v", 0,
        std::numeric_limits<float>::max()));
    REQUIRE(synth.init(config));
    features[0] = 0.0f;
    const auto high_but_finite = synth.infer(
        features.data(), 1, &pitch, &f0, 0);
    REQUIRE(high_but_finite.data.size() == 400);
    CHECK(std::all_of(high_but_finite.data.begin(), high_but_finite.data.end(),
                      [](float value) { return std::isfinite(value); }));

    // A checkpoint changed after init is rejected before the persisted mapping
    // can materialize the now-degenerate row for CUDA.
    REQUIRE(checkpoint.write_value(
        checkpoint_path.path(), "model.dec.ups.0.weight_v", 0, 0.0f));
    CHECK(synth.infer(features.data(), 1, &pitch, &f0, 0).data.empty());

    // Runtime uses the instance-owned validated mapping, but also fingerprints
    // the path so replacing it with an incompatible checkpoint fails closed.
    REQUIRE(checkpoint.write_value(
        checkpoint_path.path(), "model.dec.ups.0.weight_v", 0,
        std::numeric_limits<float>::max()));
    REQUIRE(synth.init(config));
    resonantia_test::ScopedArtifact incompatible_path(
        resonantia_test::unique_artifact_path("vits-v1-replacement", ".safetensors"));
    resonantia_test::vits_fixture::SparseCheckpoint incompatible;
    REQUIRE(resonantia_test::vits_fixture::create_sparse_checkpoint(
        incompatible_path.path(), incompatible, 256));
    std::filesystem::rename(incompatible_path.path(), checkpoint_path.path());
    CHECK(synth.infer(features.data(), 1, &pitch, &f0, 0).data.empty());
    CHECK_FALSE(synth.init(config));
    CHECK_FALSE(synth.initialized());
}

TEST_CASE("Pipeline rejects malformed direct audio buffers", "[pipeline][contract]") {
    voxmutatio::pipeline::VoiceConversionPipeline pipeline;
    REQUIRE(pipeline.init(real_config(voxmutatio::ModelVersion::kV2)));

    voxmutatio::AudioBuffer input;
    input.sample_rate = 0;
    input.data = {0.0f};
    CHECK_FALSE(pipeline.convert_buffer(input).success);

    input.sample_rate = 16'000;
    input.data.clear();
    CHECK_FALSE(pipeline.convert_buffer(input).success);

    input.data = {std::numeric_limits<float>::quiet_NaN()};
    CHECK_FALSE(pipeline.convert_buffer(input).success);
}

TEST_CASE("Pipeline config defaults", "[pipeline][config]") {
    voxmutatio::VCConfig config;
    
    REQUIRE(config.f0_up_key == 0);
    REQUIRE(config.formant_shift == Catch::Approx(0.0));
    REQUIRE(config.index_rate == Catch::Approx(0.0));
    REQUIRE(config.rms_mix_rate == Catch::Approx(0.5));
    REQUIRE(config.protect == Catch::Approx(0.5));
    REQUIRE(config.device == "cuda");
    REQUIRE(config.use_half_precision == false);
    REQUIRE(config.version == voxmutatio::ModelVersion::kV1);
    REQUIRE(config.has_f0 == true);
}

TEST_CASE("Pipeline runtime parameter updates are transactional", "[pipeline][config]") {
    voxmutatio::pipeline::VoiceConversionPipeline pipeline;
    REQUIRE(pipeline.set_runtime_params(12, 0.0, 0.25, 0.75, 0.4, 5));
    const auto accepted = pipeline.config();
    CHECK(accepted.f0_up_key == 12);
    CHECK(accepted.index_rate == Catch::Approx(0.25));
    CHECK(accepted.filter_radius == 5);

    CHECK_FALSE(pipeline.set_runtime_params(49, 0.0, 0.25, 0.75, 0.4, 5));
    CHECK_FALSE(pipeline.set_runtime_params(12, 0.0,
        std::numeric_limits<double>::quiet_NaN(), 0.75, 0.4, 5));
    CHECK_FALSE(pipeline.set_runtime_params(12, 0.0, 0.25, 0.75, 0.4,
                                             std::numeric_limits<int>::max()));
    const auto rejected = pipeline.config();
    CHECK(rejected.f0_up_key == accepted.f0_up_key);
    CHECK(rejected.index_rate == Catch::Approx(accepted.index_rate));
    CHECK(rejected.filter_radius == accepted.filter_radius);
}

TEST_CASE("VCResult default values", "[pipeline][result]") {
    voxmutatio::VCResult result;
    
    REQUIRE(result.success == false);
    REQUIRE(result.hubert_ms == Catch::Approx(0.0));
    REQUIRE(result.f0_ms == Catch::Approx(0.0));
    REQUIRE(result.index_ms == Catch::Approx(0.0));
    REQUIRE(result.synth_ms == Catch::Approx(0.0));
    REQUIRE(result.total_ms == Catch::Approx(0.0));
    REQUIRE(result.error_message == "");
}

TEST_CASE("Edge case: empty audio buffer", "[pipeline][edge]") {
    voxmutatio::AudioBuffer empty;
    empty.data = {};
    empty.sample_rate = 16000;
    
    REQUIRE(empty.num_samples() == 0);
    REQUIRE(empty.duration_s() == Catch::Approx(0.0));
}

TEST_CASE("Edge case: very short audio", "[pipeline][edge]") {
    // 0.05s at 16kHz = 800 samples
    voxmutatio::AudioBuffer short_audio;
    short_audio.data.resize(800, 0.0f);
    short_audio.sample_rate = 16000;
    
    REQUIRE(short_audio.num_samples() == 800);
    REQUIRE(short_audio.duration_s() == Catch::Approx(0.05));
}

TEST_CASE("Edge case: long audio (5min)", "[pipeline][edge]") {
    // 5min at 16kHz = 4800000 samples
    voxmutatio::AudioBuffer long_audio;
    long_audio.data.resize(4800000, 0.0f);
    long_audio.sample_rate = 16000;
    
    REQUIRE(long_audio.num_samples() == 4800000);
    REQUIRE(long_audio.duration_s() == Catch::Approx(300.0));  // 5 minutes
}

TEST_CASE("Stereo to mono downmix", "[pipeline][edge]") {
    // Simulate stereo downmix
    std::vector<float> stereo = {
        0.1f, 0.2f,  // frame 0: L, R
        0.3f, 0.4f,  // frame 1: L, R
        0.5f, 0.6f   // frame 2: L, R
    };
    
    std::vector<float> mono(stereo.size() / 2);
    for (size_t i = 0; i < mono.size(); ++i) {
        mono[i] = (stereo[i * 2] + stereo[i * 2 + 1]) / 2.0f;
    }
    
    REQUIRE(mono.size() == 3);
    REQUIRE(mono[0] == Catch::Approx(0.15f));
    REQUIRE(mono[1] == Catch::Approx(0.35f));
    REQUIRE(mono[2] == Catch::Approx(0.55f));
}
