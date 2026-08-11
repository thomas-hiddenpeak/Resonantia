// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "voxmutatio/content/cuda/kernels.h"

namespace content_cuda = voxmutatio::content::cuda;

TEST_CASE("CUDA content STFT returns a bounded real half-spectrum", "[content][cuda]") {
  std::vector<float> audio(64);
  for (std::size_t i = 0; i < audio.size(); ++i)
    audio[i] = std::sin(2.0 * M_PI * static_cast<double>(i) / 16.0);

  auto fbank = content_cuda::compute_fbank(audio.data(), static_cast<int>(audio.size()),
                                            4, 16, 8, 16'000);
  REQUIRE(fbank.size() == 7 * 4);
  for (float value : fbank) REQUIRE(std::isfinite(value));

  REQUIRE_THROWS_AS(content_cuda::compute_fbank(nullptr, 64, 4, 16, 8, 16'000),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(content_cuda::compute_fbank(audio.data(), 8, 4, 16, 8, 16'000),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      content_cuda::compute_fbank(audio.data(), 64, std::numeric_limits<int>::max(),
                                  16, 8, 16'000),
      std::invalid_argument);
}

TEST_CASE("CUDA row-major matmul honors declared dimensions", "[content][cuda]") {
  const float a[] = {1, 2, 3, 4, 5, 6};
  const float b[] = {7, 8, 9, 10, 11, 12};
  float c[4]{};
  content_cuda::matmul(a, 2, 3, b, 2, c);
  CHECK(c[0] == Catch::Approx(58.0f));
  CHECK(c[1] == Catch::Approx(64.0f));
  CHECK(c[2] == Catch::Approx(139.0f));
  CHECK(c[3] == Catch::Approx(154.0f));
}

TEST_CASE("CUDA content kernels reject zero divisors before arithmetic",
          "[content][cuda][contract]") {
  const float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float weight[] = {1.0f, 1.0f};

  REQUIRE_THROWS_AS(
      content_cuda::conv1d_strided(input, 1, 4, weight, nullptr, 1, 2, 0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      content_cuda::conv1d_grouped(input, 1, 4, weight, nullptr, 1, 2, 0, 0, 1),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      content_cuda::conv1d_grouped(input, 1, 4, weight, nullptr, 1, 2, 1, 0, 0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      content_cuda::multihead_attention_split(input, input, input, 1, 4, 0),
      std::invalid_argument);
}

TEST_CASE("CUDA content kernels reject dimensions beyond int indexing",
          "[content][cuda][contract]") {
  constexpr int too_many_rows = std::numeric_limits<int>::max();
  const float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float output[4]{};

  REQUIRE_THROWS_AS(content_cuda::layer_norm(input, too_many_rows, 2),
                    std::length_error);
  REQUIRE_THROWS_AS(
      content_cuda::layer_norm_affine(input, too_many_rows, 2, weight, weight),
      std::length_error);
  REQUIRE_THROWS_AS(
      content_cuda::multihead_attention(input, input, input, too_many_rows, 1, 2),
      std::length_error);
  REQUIRE_THROWS_AS(
      content_cuda::multihead_attention_split(input, input, input, too_many_rows, 4, 1),
      std::length_error);
  REQUIRE_THROWS_AS(
      content_cuda::linear(input, too_many_rows, 2, weight, nullptr, 2),
      std::length_error);
  REQUIRE_THROWS_AS(
      content_cuda::group_norm(input, output, too_many_rows, 2, 1, weight, weight),
      std::length_error);
  REQUIRE_THROWS_AS(
      content_cuda::conv1d_grouped(input, 1, 1, weight, nullptr, 1, 1, 1,
                                   std::numeric_limits<int>::max(), 1),
      std::length_error);
}
