// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_test_macros.hpp>

#include "cuda_failure_injection.h"
#include "voxmutatio/f0/rmvpe_ops.h"
#include "voxmutatio/synthesizer/vits_ops.h"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace {

using resonantia_test::cuda_failure::Operation;

struct ExpectedObservations {
    std::size_t allocations;
    std::size_t copies;
    std::size_t launches;
    std::size_t synchronizations;
    std::size_t peak_live_allocations;
};

void require_no_live_allocations() {
    REQUIRE(resonantia_test::cuda_failure::live_allocation_count() == 0);
}

template <typename Call>
void require_injected_failure(Call&& call, Operation operation,
                              std::size_t invocation,
                              const ExpectedObservations& expected) {
    namespace failure = resonantia_test::cuda_failure;
    require_no_live_allocations();
    failure::reset();
    failure::fail_nth(operation, invocation);

    REQUIRE_THROWS_AS(call(), std::runtime_error);

    CHECK(failure::call_count(Operation::kAllocation) == expected.allocations);
    CHECK(failure::call_count(Operation::kCopy) == expected.copies);
    CHECK(failure::call_count(Operation::kLaunch) == expected.launches);
    CHECK(failure::call_count(Operation::kSynchronize) ==
          expected.synchronizations);
    CHECK(failure::peak_live_allocation_count() ==
          expected.peak_live_allocations);
    CHECK(failure::live_allocation_count() == 0);
    failure::reset();
}

template <typename Call>
void exercise_failure_boundaries(Call&& call) {
    SECTION("the fourth allocation fails after three buffers are owned") {
        require_injected_failure(call, Operation::kAllocation, 4,
                                 {4, 2, 0, 0, 3});
    }
    SECTION("the third copy fails before launch") {
        require_injected_failure(call, Operation::kCopy, 3,
                                 {4, 3, 0, 0, 4});
    }
    SECTION("the first launch fails before synchronization and output copy") {
        require_injected_failure(call, Operation::kLaunch, 1,
                                 {4, 3, 1, 0, 4});
    }
    SECTION("the first synchronization fails before the output copy") {
        require_injected_failure(call, Operation::kSynchronize, 1,
                                 {4, 3, 1, 1, 4});
    }
    SECTION("the observable D2H output copy fails after synchronization") {
        require_injected_failure(call, Operation::kCopy, 4,
                                 {4, 4, 1, 1, 4});
    }
}

}  // namespace

TEST_CASE("RMVPE CUDA temporary resources unwind at injected failures",
          "[cuda][failure][rmvpe]") {
    const std::array<float, 1> input{2.0f};
    const std::array<float, 1> weight{3.0f};
    const std::array<float, 1> bias{1.0f};
    const auto call = [&] {
        return voxmutatio::f0::ops::conv2d_1x1(
            input.data(), 1, 1, 1, weight.data(), bias.data(), 1);
    };
    exercise_failure_boundaries(call);
}

TEST_CASE("VITS CUDA temporary resources unwind at injected failures",
          "[cuda][failure][vits]") {
    const std::array<float, 1> input{2.0f};
    const std::array<float, 1> weight{3.0f};
    const std::array<float, 1> bias{1.0f};
    const auto call = [&] {
        return voxmutatio::synthesizer::ops::conv1d(
            input.data(), 1, 1, weight.data(), bias.data(), 1, 1, 1, 0, 1, 1);
    };
    exercise_failure_boundaries(call);
}

TEST_CASE("VITS direct ops reject invalid structural parameters before CUDA",
          "[cuda][boundary][vits]") {
    namespace failure = resonantia_test::cuda_failure;
    std::array<float, 3> input{1.0f, 2.0f, 3.0f};
    const std::array<float, 3> weight{1.0f, 1.0f, 1.0f};

    failure::reset();
    SECTION("null borrowed input") {
        REQUIRE_THROWS_AS(voxmutatio::synthesizer::ops::conv1d(
                              nullptr, 1, 1, weight.data(), nullptr,
                              1, 1, 1, 0, 1, 1),
                          std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::conv1d(
                            input.data(), 1, 1, nullptr, nullptr,
                            1, 1, 1, 0, 1, 1),
                        std::invalid_argument);
    }
    SECTION("groups must divide both channel counts") {
        REQUIRE_THROWS_AS(voxmutatio::synthesizer::ops::conv1d(
                              input.data(), 3, 1, weight.data(), nullptr,
                              2, 1, 1, 0, 1, 2),
                          std::invalid_argument);
    }
    SECTION("zero divisors and negative dimensions") {
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::conv1d(
                            input.data(), 1, 1, weight.data(), nullptr,
                            1, 1, 0, 0, 1, 1),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::conv1d(
                            input.data(), 1, 1, weight.data(), nullptr,
                            1, 1, 1, 0, 1, 0),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::conv1d(
                            input.data(), 1, -1, weight.data(), nullptr,
                            1, 1, 1, 0, 1, 1),
                        std::invalid_argument);
    }
    SECTION("convolution output must be positive and representable") {
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::conv1d(
                            input.data(), 1, 1, weight.data(), nullptr,
                            1, 2, 1, 0, 1, 1),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::conv1d(
                            input.data(), 1, 1, weight.data(), nullptr,
                            1, 1, 1, std::numeric_limits<int>::max(), 1, 1),
                        std::length_error);
    }
    SECTION("convolution tensor products and grid dimensions are bounded") {
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::conv1d(
                            input.data(), std::numeric_limits<int>::max(), 2,
                            weight.data(), nullptr, 1, 1, 1, 0, 1, 1),
                        std::length_error);
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::conv1d(
                            input.data(), 1, 1, weight.data(), nullptr,
                            65'536, 1, 1, 0, 1, 1),
                        std::length_error);
    }
    SECTION("transpose convolution validates divisors, output, and products") {
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::conv_transpose1d(
                            input.data(), 1, 1, weight.data(), nullptr,
                            1, 1, 0, 0),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::conv_transpose1d(
                            input.data(), 1, 1, weight.data(), nullptr,
                            1, 1, 1, 1),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::conv_transpose1d(
                            input.data(), 1, std::numeric_limits<int>::max(),
                            weight.data(), nullptr, 1, 1, 2, 0),
                        std::length_error);
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::conv_transpose1d(
                            input.data(), std::numeric_limits<int>::max(), 2,
                            weight.data(), nullptr, 1, 1, 1, 0),
                        std::length_error);
    }
    SECTION("in-place activations validate their borrowed buffer and scalars") {
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::tanh_inplace(nullptr, 1),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::tanh_inplace(
                            input.data(), 0),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::leaky_relu_inplace(
                            input.data(), -1, 0.1f),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::synthesizer::ops::leaky_relu_inplace(
                            input.data(), 1,
                            std::numeric_limits<float>::quiet_NaN()),
                        std::invalid_argument);
    }
    CHECK(failure::call_count(Operation::kAllocation) == 0);
    CHECK(failure::call_count(Operation::kCopy) == 0);
    CHECK(failure::call_count(Operation::kLaunch) == 0);
    CHECK(failure::call_count(Operation::kSynchronize) == 0);
    CHECK(failure::live_allocation_count() == 0);
}

TEST_CASE("RMVPE direct ops reject invalid structural parameters before CUDA",
          "[cuda][boundary][rmvpe]") {
    namespace failure = resonantia_test::cuda_failure;
    std::array<float, 1> input{1.0f};
    const std::array<float, 1> weight{1.0f};

    failure::reset();
    SECTION("null borrowed weight") {
        REQUIRE_THROWS_AS(voxmutatio::f0::ops::conv2d_1x1(
                              input.data(), 1, 1, 1, nullptr, nullptr, 1),
                          std::invalid_argument);
    }
    SECTION("pooling kernel cannot exceed the input") {
        REQUIRE_THROWS_AS(voxmutatio::f0::ops::avgpool2d(
                              input.data(), 1, 1, 1, 2, 1),
                          std::invalid_argument);
    }
    SECTION("convolution dimensions, products, and grid extents are bounded") {
        CHECK_THROWS_AS(voxmutatio::f0::ops::conv2d_1x1(
                            input.data(), 0, 1, 1, weight.data(), nullptr, 1),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::f0::ops::conv2d_1x1(
                            input.data(), std::numeric_limits<int>::max(), 2, 1,
                            weight.data(), nullptr, 1),
                        std::length_error);
        CHECK_THROWS_AS(voxmutatio::f0::ops::conv2d_1x1(
                            input.data(), 1, 1, 1, weight.data(), nullptr, 65'536),
                        std::length_error);
        CHECK_THROWS_AS(voxmutatio::f0::ops::conv2d_1x1(
                            input.data(), 1, 1'048'561, 1,
                            weight.data(), nullptr, 1),
                        std::length_error);
    }
    SECTION("batch normalization validates all pointers, shape, and epsilon") {
        CHECK_THROWS_AS(voxmutatio::f0::ops::batchnorm2d(
                            input.data(), 1, 1, 1,
                            nullptr, input.data(), input.data(), input.data()),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::f0::ops::batchnorm2d(
                            input.data(), 1, 1, 1,
                            input.data(), input.data(), input.data(), input.data(),
                            std::numeric_limits<float>::infinity()),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::f0::ops::batchnorm2d(
                            input.data(),
                            std::numeric_limits<int>::max(), 2, 1,
                            input.data(), input.data(), input.data(), input.data()),
                        std::length_error);
    }
    SECTION("pooling and transpose reject invalid and overflowing extents") {
        CHECK_THROWS_AS(voxmutatio::f0::ops::avgpool2d(
                            input.data(), 1, 1, 1, 0, 1),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::f0::ops::avgpool2d(
                            input.data(), 1, 1, 1, 1, 0),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::f0::ops::avgpool2d(
                            input.data(), std::numeric_limits<int>::max(), 2, 1, 1, 1),
                        std::length_error);
        CHECK_THROWS_AS(voxmutatio::f0::ops::conv_transpose2d_s2(
                            input.data(), 1, std::numeric_limits<int>::max(), 1,
                            weight.data(), 1),
                        std::length_error);
    }
    SECTION("in-place and host-only ops reject invalid borrowed ranges") {
        CHECK_THROWS_AS(voxmutatio::f0::ops::relu_inplace(nullptr, 1),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::f0::ops::relu_inplace(input.data(), 0),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::f0::ops::sigmoid_inplace(
                            input.data(), -1),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::f0::ops::concat_channels(
                            nullptr, 1, input.data(), 1, 1, 1),
                        std::invalid_argument);
        CHECK_THROWS_AS(voxmutatio::f0::ops::concat_channels(
                            input.data(), std::numeric_limits<int>::max(),
                            input.data(), 1, 1, 1),
                        std::length_error);
    }
    CHECK(failure::call_count(Operation::kAllocation) == 0);
    CHECK(failure::call_count(Operation::kCopy) == 0);
    CHECK(failure::call_count(Operation::kLaunch) == 0);
    CHECK(failure::call_count(Operation::kSynchronize) == 0);
    CHECK(failure::live_allocation_count() == 0);
}

TEST_CASE("CUDA op validation preserves the valid numerical path",
          "[cuda][boundary][numeric]") {
    namespace failure = resonantia_test::cuda_failure;
    const std::array<float, 1> input{2.0f};
    const std::array<float, 1> weight{3.0f};
    const std::array<float, 1> bias{1.0f};

    failure::reset();
    SECTION("RMVPE 1x1 convolution") {
        const auto output = voxmutatio::f0::ops::conv2d_1x1(
            input.data(), 1, 1, 1, weight.data(), bias.data(), 1);
        REQUIRE(output.size() == 1);
        CHECK(output.front() == 7.0f);
    }
    SECTION("VITS 1D convolution") {
        const auto output = voxmutatio::synthesizer::ops::conv1d(
            input.data(), 1, 1, weight.data(), bias.data(),
            1, 1, 1, 0, 1, 1);
        REQUIRE(output.size() == 1);
        CHECK(output.front() == 7.0f);
    }
    CHECK(failure::call_count(Operation::kAllocation) == 4);
    CHECK(failure::call_count(Operation::kCopy) == 4);
    CHECK(failure::call_count(Operation::kLaunch) == 1);
    CHECK(failure::call_count(Operation::kSynchronize) == 1);
    CHECK(failure::peak_live_allocation_count() == 4);
    CHECK(failure::live_allocation_count() == 0);
}
