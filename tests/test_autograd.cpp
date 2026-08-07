// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Gradient check for the autograd engine: analytic gradients (backward) vs
// central finite differences. This is the training-side analog of the
// inference numerical-alignment gate (spec 002 SC-001).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <functional>
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

// Generic finite-difference gradient check.
// `build` maps a list of param tensors to a scalar loss.
double grad_check(const std::function<Tensor(std::vector<Tensor>&)>& build,
                  std::vector<std::vector<float>> params,
                  const std::vector<std::vector<int>>& shapes) {
  std::vector<Tensor> ts;
  for (size_t i = 0; i < params.size(); ++i)
    ts.push_back(Tensor::from_host(params[i], shapes[i], true));
  Tensor loss = build(ts);
  ag::backward(loss);
  std::vector<std::vector<float>> analytic;
  for (auto& t : ts) analytic.push_back(t.grad_to_host());

  const float eps = 1e-2f;
  double maxerr = 0.0;
  for (size_t p = 0; p < params.size(); ++p) {
    for (size_t i = 0; i < params[p].size(); ++i) {
      float orig = params[p][i];
      params[p][i] = orig + eps;
      std::vector<Tensor> tp;
      for (size_t j = 0; j < params.size(); ++j)
        tp.push_back(Tensor::from_host(params[j], shapes[j], false));
      float fp = build(tp).to_host()[0];
      params[p][i] = orig - eps;
      std::vector<Tensor> tm;
      for (size_t j = 0; j < params.size(); ++j)
        tm.push_back(Tensor::from_host(params[j], shapes[j], false));
      float fm = build(tm).to_host()[0];
      params[p][i] = orig;
      double num = (fp - fm) / (2 * eps);
      double denom = 1.0 + std::abs(num);
      maxerr = std::max(maxerr, std::abs((double)analytic[p][i] - num) / denom);
    }
  }
  return maxerr;
}

std::vector<float> rand_vec(int n, std::mt19937& rng, float lo = -1.0f, float hi = 1.0f) {
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
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

TEST_CASE("Autograd A1: linear + add_bias", "[autograd][gradcheck][a1]") {
  std::mt19937 rng(7);
  const int M = 4, K = 3, N = 2;
  // loss = sum(gelu(linear(x, W, b)))
  double e = grad_check(
      [M, N](std::vector<Tensor>& t) {
        return ag::sum(ag::gelu(ag::linear(t[0], t[1], t[2])));
      },
      {rand_vec(M * K, rng), rand_vec(N * K, rng), rand_vec(N, rng)},
      {{M, K}, {N, K}, {N}});
  std::printf("[linear] %.2e\n", e);
  CHECK(e < 2e-2);

  double eb = grad_check(
      [](std::vector<Tensor>& t) { return ag::sum(ag::gelu(ag::add_bias(t[0], t[1]))); },
      {rand_vec(4 * 3, rng), rand_vec(3, rng)}, {{4, 3}, {3}});
  std::printf("[add_bias] %.2e\n", eb);
  CHECK(eb < 2e-2);
}

TEST_CASE("Autograd A1: activations", "[autograd][gradcheck][a1]") {
  // Values bounded away from 0 (relu kink).
  std::vector<float> x = {-0.8f, -0.4f, 0.3f, 0.7f, -0.6f, 0.5f, 0.9f, -0.2f, 0.6f};
  std::vector<int> sh = {3, 3};
  struct { const char* name; std::function<Tensor(std::vector<Tensor>&)> f; } acts[] = {
      {"relu", [](std::vector<Tensor>& t) { return ag::sum(ag::relu(t[0])); }},
      {"leaky", [](std::vector<Tensor>& t) { return ag::sum(ag::leaky_relu(t[0], 0.1f)); }},
      {"gelu", [](std::vector<Tensor>& t) { return ag::sum(ag::gelu(t[0])); }},
      {"tanh", [](std::vector<Tensor>& t) { return ag::sum(ag::tanh_op(t[0])); }},
      {"sigmoid", [](std::vector<Tensor>& t) { return ag::sum(ag::sigmoid(t[0])); }},
  };
  for (auto& a : acts) {
    double e = grad_check(a.f, {x}, {sh});
    std::printf("[%s] %.2e\n", a.name, e);
    CHECK(e < 2e-2);
  }
}

TEST_CASE("Autograd A1: softmax", "[autograd][gradcheck][a1]") {
  std::mt19937 rng(11);
  const int R = 3, C = 4;
  // loss = sum(softmax(x) * target)
  double e = grad_check(
      [R, C](std::vector<Tensor>& t) {
        return ag::sum(ag::mul(ag::softmax_rows(t[0], R, C), t[1]));
      },
      {rand_vec(R * C, rng), rand_vec(R * C, rng)}, {{R, C}, {R, C}});
  std::printf("[softmax] %.2e\n", e);
  CHECK(e < 2e-2);
}

TEST_CASE("Autograd A1: layer_norm", "[autograd][gradcheck][a1]") {
  std::mt19937 rng(13);
  const int R = 3, D = 5;
  // loss = sum(layer_norm(x, w, b) * target)
  double e = grad_check(
      [R, D](std::vector<Tensor>& t) {
        return ag::sum(ag::mul(ag::layer_norm(t[0], t[1], t[2], R, D), t[3]));
      },
      {rand_vec(R * D, rng), rand_vec(D, rng, 0.5f, 1.5f), rand_vec(D, rng),
       rand_vec(R * D, rng)},
      {{R, D}, {D}, {D}, {R, D}});
  std::printf("[layer_norm] %.2e\n", e);
  CHECK(e < 2e-2);
}

TEST_CASE("Autograd A1: conv1d", "[autograd][gradcheck][a1][conv]") {
  std::mt19937 rng(17);
  auto run = [&](int Cin, int L, int Cout, int K, int stride, int pad, int dil,
                 int groups, const char* label) {
    int ipg = Cin / groups;
    double e = grad_check(
        [=](std::vector<Tensor>& t) {
          return ag::sum(ag::gelu(ag::conv1d(t[0], t[1], t[2], Cin, L, Cout, K,
                                             stride, pad, dil, groups)));
        },
        {rand_vec(Cin * L, rng), rand_vec(Cout * ipg * K, rng), rand_vec(Cout, rng)},
        {{Cin, L}, {Cout, ipg, K}, {Cout}});
    std::printf("[conv1d %s] %.2e\n", label, e);
    CHECK(e < 3e-2);
  };
  run(2, 6, 3, 3, 1, 1, 1, 1, "basic");
  run(2, 8, 2, 3, 2, 1, 1, 1, "stride2");
  run(2, 8, 2, 3, 1, 2, 2, 1, "dilated");
  run(4, 6, 4, 3, 1, 1, 1, 2, "grouped");
}

TEST_CASE("Autograd A1: conv_transpose1d", "[autograd][gradcheck][a1][conv]") {
  std::mt19937 rng(19);
  auto run = [&](int Cin, int L, int Cout, int K, int stride, int pad,
                 const char* label) {
    double e = grad_check(
        [=](std::vector<Tensor>& t) {
          return ag::sum(ag::gelu(
              ag::conv_transpose1d(t[0], t[1], t[2], Cin, L, Cout, K, stride, pad)));
        },
        {rand_vec(Cin * L, rng), rand_vec(Cin * Cout * K, rng), rand_vec(Cout, rng)},
        {{Cin, L}, {Cin, Cout, K}, {Cout}});
    std::printf("[convT %s] %.2e\n", label, e);
    CHECK(e < 3e-2);
  };
  run(3, 4, 2, 4, 2, 1, "up2_k4");   // generator-style upsampling
  run(2, 5, 3, 3, 1, 1, "same");
}
