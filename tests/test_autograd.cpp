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

TEST_CASE("Autograd GAN: exp + conv2d", "[autograd][gradcheck][gan]") {
  std::mt19937 rng(23);
  SECTION("exp") {
    double e = grad_check(
        [](std::vector<Tensor>& t) { return ag::sum(ag::exp_op(t[0])); },
        {rand_vec(12, rng, -1.0f, 1.0f)}, {{3, 4}});
    std::printf("[exp] %.2e\n", e);
    CHECK(e < 3e-2);
  }
  SECTION("conv2d") {
    auto run = [&](int Cin, int H, int W, int Cout, int kh, int kw, int sh, int sw,
                   int ph, int pw, const char* label) {
      double e = grad_check(
          [=](std::vector<Tensor>& t) {
            return ag::sum(ag::gelu(ag::conv2d(t[0], t[1], t[2], Cin, H, W, Cout,
                                               kh, kw, sh, sw, ph, pw)));
          },
          {rand_vec(Cin * H * W, rng), rand_vec(Cout * Cin * kh * kw, rng), rand_vec(Cout, rng)},
          {{Cin, H, W}, {Cout, Cin, kh, kw}, {Cout}});
      std::printf("[conv2d %s] %.2e\n", label, e);
      CHECK(e < 3e-2);
    };
    run(1, 8, 3, 4, 5, 1, 3, 1, 2, 0, "mpd");     // MPD period-reshape conv (kw=1)
    run(2, 5, 4, 3, 3, 3, 1, 1, 1, 1, "square");
    run(2, 6, 2, 2, 3, 1, 2, 1, 1, 0, "stride");
  }
}

TEST_CASE("Autograd A3: primitives", "[autograd][gradcheck][a3]") {
  std::mt19937 rng(29);
  {
    const int R = 3, C = 4;
    double e = grad_check(
        [R, C](std::vector<Tensor>& t) { return ag::sum(ag::mul(ag::transpose2d(t[0], R, C), t[1])); },
        {rand_vec(R * C, rng), rand_vec(C * R, rng)}, {{R, C}, {C, R}});
    std::printf("[transpose2d] %.2e\n", e); CHECK(e < 2e-2);
  }
  {
    const int Ra = 2, Rb = 3, C = 4;
    double e = grad_check(
        [Ra, Rb, C](std::vector<Tensor>& t) { return ag::sum(ag::mul(ag::concat_rows(t[0], t[1], C), t[2])); },
        {rand_vec(Ra * C, rng), rand_vec(Rb * C, rng), rand_vec((Ra + Rb) * C, rng)},
        {{Ra, C}, {Rb, C}, {Ra + Rb, C}});
    std::printf("[concat_rows] %.2e\n", e); CHECK(e < 2e-2);
  }
  {
    const int R = 5, C = 3, start = 1, count = 3;
    double e = grad_check(
        [=](std::vector<Tensor>& t) { return ag::sum(ag::mul(ag::slice_rows(t[0], start, count, C), t[1])); },
        {rand_vec(R * C, rng), rand_vec(count * C, rng)}, {{R, C}, {count, C}});
    std::printf("[slice_rows] %.2e\n", e); CHECK(e < 2e-2);
  }
  {
    const int R = 3, C = 4;
    double e = grad_check(
        [R, C](std::vector<Tensor>& t) { return ag::sum(ag::mul(ag::scale(t[0], 2.5f), t[1])); },
        {rand_vec(R * C, rng), rand_vec(R * C, rng)}, {{R, C}, {R, C}});
    std::printf("[scale] %.2e\n", e); CHECK(e < 2e-2);
  }
}

TEST_CASE("Autograd A3: embedding", "[autograd][gradcheck][a3]") {
  std::mt19937 rng(31);
  const int V = 6, D = 4;
  std::vector<int> idx = {0, 3, 1, 5, 3};  // repeat (3) tests accumulation
  auto table_host = rand_vec(V * D, rng);
  auto table = Tensor::from_host(table_host, {V, D}, true);
  ag::backward(ag::sum(ag::gelu(ag::embedding(table, idx, D))));
  auto analytic = table.grad_to_host();

  const float eps = 1e-2f;
  double maxerr = 0.0;
  for (int i = 0; i < V * D; ++i) {
    auto h = table_host; h[i] += eps;
    float fp = ag::sum(ag::gelu(ag::embedding(Tensor::from_host(h, {V, D}, false), idx, D))).to_host()[0];
    h[i] -= 2 * eps;
    float fm = ag::sum(ag::gelu(ag::embedding(Tensor::from_host(h, {V, D}, false), idx, D))).to_host()[0];
    double num = (fp - fm) / (2 * eps);
    maxerr = std::max(maxerr, std::abs((double)analytic[i] - num) / (1.0 + std::abs(num)));
  }
  std::printf("[embedding] %.2e\n", maxerr); CHECK(maxerr < 2e-2);
}

TEST_CASE("Autograd A4: mel-loss primitives", "[autograd][gradcheck][a4]") {
  std::mt19937 rng(37);
  {
    auto xp = rand_vec(12, rng, 0.2f, 2.0f);
    double es = grad_check([](std::vector<Tensor>& t) { return ag::sum(ag::sqrt_op(t[0])); }, {xp}, {{3, 4}});
    double el = grad_check([](std::vector<Tensor>& t) { return ag::sum(ag::log_op(t[0])); }, {xp}, {{3, 4}});
    std::printf("[sqrt] %.2e [log] %.2e\n", es, el);
    CHECK(es < 2e-2); CHECK(el < 2e-2);
  }
  {
    std::vector<float> x = {-0.8f, 0.5f, -0.3f, 0.9f, -0.6f, 0.4f};
    double e = grad_check([](std::vector<Tensor>& t) { return ag::sum(ag::abs_op(t[0])); }, {x}, {{2, 3}});
    std::printf("[abs] %.2e\n", e); CHECK(e < 2e-2);
  }
  {
    const int T = 3, nfft = 4, hop = 2;
    const int L = (T - 1) * hop + nfft;
    double e = grad_check(
        [=](std::vector<Tensor>& t) { return ag::sum(ag::mul(ag::frame(t[0], T, nfft, hop), t[1])); },
        {rand_vec(L, rng), rand_vec(T * nfft, rng)}, {{1, L}, {T, nfft}});
    std::printf("[frame] %.2e\n", e); CHECK(e < 2e-2);
  }
}

TEST_CASE("Autograd A2: AdamW convex convergence", "[autograd][a2][optim]") {
  // Minimize f(x) = sum((x - target)^2). Start x = 0, expect x -> target.
  const int n = 8;
  std::mt19937 rng(23);
  std::vector<float> target = rand_vec(n, rng, -2.0f, 2.0f);
  std::vector<float> neg_target(n);
  for (int i = 0; i < n; ++i) neg_target[i] = -target[i];

  auto x = Tensor::from_host(std::vector<float>(n, 0.0f), {1, n}, true);
  auto negt = Tensor::from_host(neg_target, {1, n}, false);

  ag::AdamW opt({x}, /*lr=*/0.1f);
  for (int it = 0; it < 400; ++it) {
    auto diff = ag::add(x, negt);          // x - target
    auto loss = ag::sum(ag::mul(diff, diff));
    ag::backward(loss);
    opt.step();
  }

  auto xh = x.to_host();
  double max_err = 0.0;
  for (int i = 0; i < n; ++i)
    max_err = std::max(max_err, std::abs((double)xh[i] - target[i]));
  std::printf("[AdamW] max |x-target| after 400 steps: %.2e\n", max_err);
  CHECK(max_err < 1e-2);
}
