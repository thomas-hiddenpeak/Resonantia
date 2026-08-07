// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Gradient check for the autograd engine: analytic gradients (backward) vs
// central finite differences. This is the training-side analog of the
// inference numerical-alignment gate (spec 002 SC-001).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "voxmutatio/autograd/tensor.h"

using voxmutatio::autograd::Tensor;
namespace ag = voxmutatio::autograd;

namespace {

// f(A,B,C) = sum( (A @ B) elementwise* C ), A[M,K] B[K,N] C[M,N]
float eval_f(const std::vector<float>& a, const std::vector<float>& b,
             const std::vector<float>& c, int M, int K, int N) {
  auto A = Tensor::from_host(a, {M, K}, true);
  auto B = Tensor::from_host(b, {K, N}, true);
  auto C = Tensor::from_host(c, {M, N}, true);
  auto loss = ag::sum(ag::mul(ag::matmul(A, B), C));
  return loss.to_host()[0];
}

double max_rel_err(const std::vector<float>& analytic,
                   const std::vector<float>& numeric) {
  double m = 0.0;
  for (size_t i = 0; i < analytic.size(); ++i) {
    double diff = std::abs((double)analytic[i] - numeric[i]);
    double denom = 1.0 + std::abs((double)numeric[i]);
    m = std::max(m, diff / denom);
  }
  return m;
}

}  // namespace

TEST_CASE("Autograd gradient check: matmul + mul + sum", "[autograd][gradcheck]") {
  const int M = 3, K = 4, N = 2;
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> a(M * K), b(K * N), c(M * N);
  for (auto& x : a) x = dist(rng);
  for (auto& x : b) x = dist(rng);
  for (auto& x : c) x = dist(rng);

  // Analytic gradients via backward.
  auto A = Tensor::from_host(a, {M, K}, true);
  auto B = Tensor::from_host(b, {K, N}, true);
  auto C = Tensor::from_host(c, {M, N}, true);
  auto loss = ag::sum(ag::mul(ag::matmul(A, B), C));
  ag::backward(loss);
  auto gA = A.grad_to_host();
  auto gB = B.grad_to_host();
  auto gC = C.grad_to_host();

  // Numeric gradients via central finite differences.
  const float eps = 1e-2f;
  auto numeric_grad = [&](std::vector<float>& p) {
    std::vector<float> g(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
      float orig = p[i];
      p[i] = orig + eps;
      float fp = eval_f(a, b, c, M, K, N);
      p[i] = orig - eps;
      float fm = eval_f(a, b, c, M, K, N);
      p[i] = orig;
      g[i] = (fp - fm) / (2 * eps);
    }
    return g;
  };
  auto nA = numeric_grad(a);
  auto nB = numeric_grad(b);
  auto nC = numeric_grad(c);

  double eA = max_rel_err(gA, nA);
  double eB = max_rel_err(gB, nB);
  double eC = max_rel_err(gC, nC);
  std::printf("grad rel err: A=%.2e B=%.2e C=%.2e\n", eA, eB, eC);

  CHECK(eA < 1e-2);
  CHECK(eB < 1e-2);
  CHECK(eC < 1e-2);
}

TEST_CASE("Autograd accumulation: reused tensor", "[autograd][gradcheck]") {
  // g = sum(x * x) -> dg/dx = 2x. x reused by mul twice via same node.
  const int n = 5;
  std::vector<float> x(n);
  for (int i = 0; i < n; ++i) x[i] = 0.5f * (i + 1);

  auto X = Tensor::from_host(x, {1, n}, true);
  auto loss = ag::sum(ag::mul(X, X));
  ag::backward(loss);
  auto g = X.grad_to_host();

  double m = 0.0;
  for (int i = 0; i < n; ++i) m = std::max(m, std::abs((double)g[i] - 2.0 * x[i]));
  std::printf("reuse grad max abs err: %.2e\n", m);
  CHECK(m < 1e-4);
}
