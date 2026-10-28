// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>

#include "cuda_failure_injection.h"
#include "test_artifacts.h"
#include "vits_checkpoint_fixture.h"
#include "voxmutatio/synthesizer/synthesizer.h"

#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>
#include <vector>

namespace allocation_failure {

enum class Kind {
    kNone,
    kStandard,
    kUnknown,
};

struct UnknownException {};

struct State {
    Kind failure = Kind::kNone;
    std::size_t fail_at = 0;
    std::size_t allocations = 0;
};

thread_local State state;

void reset() noexcept {
    state = {};
}

void arm(Kind kind) noexcept {
    state.failure = kind;
    state.fail_at = state.allocations + 1;
}

void arm_nth(Kind kind, std::size_t invocation) noexcept {
    state.failure = kind;
    state.fail_at = invocation;
}

bool armed() noexcept {
    return state.failure != Kind::kNone;
}

std::size_t allocation_count() noexcept {
    return state.allocations;
}

void fail_if_armed() {
    ++state.allocations;
    if (state.fail_at == 0 || state.allocations != state.fail_at) return;
    const Kind kind = state.failure;
    state.failure = Kind::kNone;
    state.fail_at = 0;
    if (kind == Kind::kStandard) throw std::bad_alloc{};
    if (kind == Kind::kUnknown) throw UnknownException{};
}

}  // namespace allocation_failure

// Private executable-wide host-allocation seam. It makes both std::exception
// and non-std exceptions deterministic without adding a production hook.
void* operator new(std::size_t bytes) {
    allocation_failure::fail_if_armed();
    if (void* memory = std::malloc(bytes == 0 ? 1 : bytes)) return memory;
    throw std::bad_alloc{};
}

void* operator new[](std::size_t bytes) {
    return ::operator new(bytes);
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

void operator delete[](void* memory) noexcept {
    ::operator delete(memory);
}

void operator delete[](void* memory, std::size_t) noexcept {
    ::operator delete(memory);
}

namespace {

using resonantia_test::cuda_failure::Operation;
using voxmutatio::ModelVersion;
using voxmutatio::synthesizer::Synthesizer;
using voxmutatio::synthesizer::SynthesizerConfig;

struct ExpectedCudaObservations {
    std::size_t allocations;
    std::size_t copies;
    std::size_t launches;
    std::size_t synchronizations;
};

SynthesizerConfig sparse_config(const std::filesystem::path& checkpoint) {
    SynthesizerConfig config;
    config.model_path = checkpoint.string();
    config.version = ModelVersion::kV2;
    return config;
}

template <typename Call>
void require_unknown_host_failure_contained(Synthesizer& synthesizer,
                                            Call&& call) {
    allocation_failure::arm(allocation_failure::Kind::kUnknown);
    bool escaped = false;
    std::vector<float> output;
    try {
        output = std::forward<Call>(call)();
    } catch (...) {
        escaped = true;
    }

    CHECK_FALSE(allocation_failure::armed());
    CHECK_FALSE(escaped);
    CHECK(output.empty());
    CHECK(synthesizer.initialized());
}

void require_debug_cuda_failure_contained(
    Synthesizer& synthesizer, const std::vector<float>& latent,
    Operation operation, std::size_t invocation,
    const ExpectedCudaObservations& expected) {
    namespace failure = resonantia_test::cuda_failure;
    REQUIRE(failure::live_allocation_count() == 0);
    failure::reset();
    failure::fail_nth(operation, invocation);

    bool escaped = false;
    std::vector<float> output;
    try {
        output = synthesizer.debug_convpre(latent.data(), 1, 0);
    } catch (...) {
        escaped = true;
    }

    CHECK_FALSE(escaped);
    CHECK(output.empty());
    CHECK(synthesizer.initialized());
    CHECK(failure::call_count(Operation::kAllocation) == expected.allocations);
    CHECK(failure::call_count(Operation::kCopy) == expected.copies);
    CHECK(failure::call_count(Operation::kLaunch) == expected.launches);
    CHECK(failure::call_count(Operation::kSynchronize) ==
          expected.synchronizations);
    CHECK(failure::live_allocation_count() == 0);
    failure::reset();
}

}  // namespace

TEST_CASE("Synthesizer contains host and unknown exceptions at every public entry",
          "[synthesizer][failure][public-boundary]") {
    resonantia_test::ScopedArtifact checkpoint_path(
        resonantia_test::unique_artifact_path("vits-public-failure", ".safetensors"));
    resonantia_test::vits_fixture::SparseCheckpoint checkpoint;
    REQUIRE(resonantia_test::vits_fixture::create_sparse_checkpoint(
        checkpoint_path.path(), checkpoint));

    const SynthesizerConfig config = sparse_config(checkpoint_path.path());
    Synthesizer synthesizer;
    REQUIRE(synthesizer.init(config));

    // Replacement initialization is invalidating even when a host allocation
    // throws: neither std::bad_alloc nor a non-std exception may cross bool init().
    for (const auto kind : {allocation_failure::Kind::kStandard,
                            allocation_failure::Kind::kUnknown}) {
        allocation_failure::arm(kind);
        bool escaped = false;
        bool accepted = true;
        try {
            accepted = synthesizer.init(config);
        } catch (...) {
            escaped = true;
        }
        CHECK_FALSE(allocation_failure::armed());
        CHECK_FALSE(escaped);
        CHECK_FALSE(accepted);
        CHECK_FALSE(synthesizer.initialized());
        CHECK(synthesizer.num_speakers() == 0);
        CHECK_FALSE(synthesizer.has_f0());
        REQUIRE(synthesizer.init(config));
    }

    std::vector<float> features(768, 0.0f);
    std::vector<float> latent(192, 0.0f);
    int pitch_coarse = 1;
    float pitch = 1.0f;
    float f0 = 110.0f;

    // Each public debug hook is independently responsible for converting an
    // internal non-std host failure into its documented empty-vector result.
    require_unknown_host_failure_contained(synthesizer, [&] {
        return synthesizer.debug_text_encoder(
            features.data(), 1, &pitch_coarse);
    });
    require_unknown_host_failure_contained(synthesizer, [&] {
        return synthesizer.debug_flow(features.data(), 1, &pitch_coarse, 0);
    });
    require_unknown_host_failure_contained(synthesizer, [&] {
        return synthesizer.debug_har(&f0, 1);
    });
    require_unknown_host_failure_contained(synthesizer, [&] {
        return synthesizer.debug_convpre(latent.data(), 1, 0);
    });
    require_unknown_host_failure_contained(synthesizer, [&] {
        return synthesizer.debug_gen_stage0(latent.data(), &f0, 1, 0, 0);
    });
    require_unknown_host_failure_contained(synthesizer, [&] {
        return synthesizer.debug_decode(latent.data(), &f0, 1, 0);
    });
    require_unknown_host_failure_contained(synthesizer, [&] {
        return synthesizer.debug_encp(features.data(), 1, &pitch_coarse, false);
    });
    require_unknown_host_failure_contained(synthesizer, [&] {
        return synthesizer.debug_flow_reverse(latent.data(), 1, 0);
    });

    allocation_failure::arm(allocation_failure::Kind::kUnknown);
    bool infer_escaped = false;
    voxmutatio::AudioBuffer inferred;
    try {
        inferred = synthesizer.infer(
            features.data(), 1, &pitch, &f0, 0);
    } catch (...) {
        infer_escaped = true;
    }
    CHECK_FALSE(allocation_failure::armed());
    CHECK_FALSE(infer_escaped);
    CHECK(inferred.data.empty());
    CHECK(inferred.sample_rate == 40'000);
    CHECK(synthesizer.initialized());

    // First characterize allocations made by the delegated full inference on
    // this thread.  Failing the immediately following allocation then targets
    // infer_stream's own crop assignment rather than infer()'s catch boundary.
    const auto warmed_infer = synthesizer.infer(
        features.data(), 1, &pitch, &f0, 0);
    REQUIRE(warmed_infer.data.size() == 400);
    allocation_failure::reset();
    const auto full_for_stream = synthesizer.infer(
        features.data(), 1, &pitch, &f0, 0);
    const std::size_t full_infer_allocations =
        allocation_failure::allocation_count();
    REQUIRE(full_for_stream.data.size() == 400);
    REQUIRE(full_infer_allocations > 0);

    allocation_failure::reset();
    const auto successful_stream = synthesizer.infer_stream(
        features.data(), 1, &pitch, &f0, 0, 0, 1);
    const std::size_t successful_stream_allocations =
        allocation_failure::allocation_count();
    REQUIRE(successful_stream.data.size() == 400);
    REQUIRE(successful_stream_allocations == full_infer_allocations + 1);

    allocation_failure::reset();
    allocation_failure::arm_nth(
        allocation_failure::Kind::kUnknown, successful_stream_allocations);
    bool stream_escaped = false;
    voxmutatio::AudioBuffer streamed;
    try {
        streamed = synthesizer.infer_stream(
            features.data(), 1, &pitch, &f0, 0, 0, 1);
    } catch (...) {
        stream_escaped = true;
    }
    const std::size_t stream_allocations =
        allocation_failure::allocation_count();
    CHECK_FALSE(allocation_failure::armed());
    CHECK_FALSE(stream_escaped);
    CHECK(streamed.data.empty());
    CHECK(streamed.sample_rate == 40'000);
    CHECK(synthesizer.initialized());
    CHECK(stream_allocations == successful_stream_allocations);
    allocation_failure::reset();
}

TEST_CASE("Synthesizer contains representative VITS CUDA failures",
          "[synthesizer][cuda][failure][public-boundary]") {
    namespace failure = resonantia_test::cuda_failure;

    resonantia_test::ScopedArtifact checkpoint_path(
        resonantia_test::unique_artifact_path("vits-cuda-public-failure",
                                               ".safetensors"));
    resonantia_test::vits_fixture::SparseCheckpoint checkpoint;
    REQUIRE(resonantia_test::vits_fixture::create_sparse_checkpoint(
        checkpoint_path.path(), checkpoint));

    Synthesizer synthesizer;
    REQUIRE(synthesizer.init(sparse_config(checkpoint_path.path())));
    REQUIRE(synthesizer.initialized());

    const std::vector<float> latent(192, 0.0f);
    require_debug_cuda_failure_contained(
        synthesizer, latent, Operation::kAllocation, 1, {1, 0, 0, 0});
    // conv_pre owns four tensor buffers plus bounded im2col scratch. Its three
    // launches are im2col, SGEMM, and bias; readback remains the fourth copy.
    require_debug_cuda_failure_contained(
        synthesizer, latent, Operation::kAllocation, 5, {5, 3, 0, 0});
    require_debug_cuda_failure_contained(
        synthesizer, latent, Operation::kCopy, 4, {5, 4, 3, 3});
    require_debug_cuda_failure_contained(
        synthesizer, latent, Operation::kLaunch, 1, {5, 3, 1, 0});
    require_debug_cuda_failure_contained(
        synthesizer, latent, Operation::kSynchronize, 1, {5, 3, 1, 1});
    require_debug_cuda_failure_contained(
        synthesizer, latent, Operation::kLaunch, 2, {5, 3, 2, 1});
    require_debug_cuda_failure_contained(
        synthesizer, latent, Operation::kSynchronize, 2, {5, 3, 2, 2});

    const auto recovered = synthesizer.debug_convpre(latent.data(), 1, 0);
    REQUIRE(recovered.size() == 512);
    CHECK(synthesizer.initialized());
    CHECK(failure::live_allocation_count() == 0);

    // The full infer path enters the content kernels before VITS. Recompiling
    // that source with the private CudaBuffer seam covers allocation only; its
    // cudaMemcpy and launch sites are deliberately not claimed as injected here.
    std::vector<float> features(768, 0.0f);
    float pitch = 1.0f;
    float f0 = 110.0f;
    failure::reset();
    failure::fail_nth(Operation::kAllocation, 1);
    bool escaped = false;
    voxmutatio::AudioBuffer result;
    try {
        result = synthesizer.infer(features.data(), 1, &pitch, &f0, 0);
    } catch (...) {
        escaped = true;
    }
    CHECK_FALSE(escaped);
    CHECK(result.data.empty());
    CHECK(result.sample_rate == 40'000);
    CHECK(synthesizer.initialized());
    CHECK(failure::call_count(Operation::kAllocation) == 1);
    CHECK(failure::live_allocation_count() == 0);
    failure::reset();
}
