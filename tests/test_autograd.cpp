// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Gradient check for the autograd engine: analytic gradients (backward) vs
// central finite differences. This is the training-side analog of the
// inference numerical-alignment gate (spec 002 SC-001).

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "voxmutatio/autograd/tensor.h"

using voxmutatio::autograd::Tensor;
namespace ag = voxmutatio::autograd;

TEST_CASE("Loss and activation match reference subgradients at zero", "[autograd][joint][boundary]") {
  auto x = Tensor::from_host({-2.f, 0.f, 3.f}, {3}, true);
  ag::backward(ag::sum(ag::abs_op(x)));
  CHECK(x.grad_to_host() == std::vector<float>{-1.f, 0.f, 1.f});
  ag::backward(ag::sum(ag::leaky_relu(x, 0.1f)));
  CHECK(x.grad_to_host() == std::vector<float>{0.1f, 0.1f, 1.f});
}

TEST_CASE("Joint training reshape and clamp preserve boundary gradients", "[autograd][joint]") {
  auto x = Tensor::from_host({-1, 2, 3, 4}, {2, 2}, true);
  auto y = ag::clamp_min(ag::reshape(x, {4}), 2.0f);
  CHECK(y.to_host() == std::vector<float>{2, 2, 3, 4});
  ag::backward(ag::sum(y));
  CHECK(x.grad_to_host() == std::vector<float>{0, 1, 1, 1});
  CHECK_THROWS_AS(ag::reshape(x, {3}), std::invalid_argument);
  CHECK_THROWS_AS(ag::reshape(x, {0, 4}), std::invalid_argument);
  CHECK_THROWS_AS(ag::clamp_min(x, std::numeric_limits<float>::infinity()), std::invalid_argument);
}

TEST_CASE("Joint weight norm differentiates original g and v parameters", "[autograd][joint]") {
  auto v = Tensor::from_host({3, 4, 0, 5}, {2, 2}, true);
  auto g = Tensor::from_host({2, 3}, {2, 1}, true);
  auto w = ag::weight_norm(v, g);
  auto expected = std::vector<float>{1.2f, 1.6f, 0, 3};
  auto actual = w.to_host();
  for (int i = 0; i < 4; ++i) CHECK(std::abs(actual[i]-expected[i]) < 1e-6f);
  ag::backward(ag::sum(w));
  auto dv = v.grad_to_host(), dg = g.grad_to_host();
  expected = {0.064f, -0.048f, 0.6f, 0};
  for (int i = 0; i < 4; ++i) CHECK(std::abs(dv[i]-expected[i]) < 1e-6f);
  CHECK(std::abs(dg[0]-1.4f) < 1e-6f);
  CHECK(std::abs(dg[1]-1.0f) < 1e-6f);
  CHECK_THROWS_AS(ag::weight_norm(Tensor::zeros({2, 2}), g), std::invalid_argument);
  CHECK_THROWS_AS(ag::weight_norm(v, Tensor::zeros({3})), std::invalid_argument);
  ag::AdamW opt({g}, 1e-4f, .8f, .99f, 1e-9f, .01f);
  opt.step();
  CHECK(std::abs(opt.first_moment(0)[0] - .28f) < 1e-6f);
  CHECK(std::abs(opt.second_moment(0)[0] - .0196f) < 1e-6f);
  CHECK_THROWS_AS(opt.first_moment(1), std::out_of_range);
}

TEST_CASE("Weight norm preserves the frozen FP32 reduction order", "[autograd][joint]") {
  // Independent torch 2.13 CPU/aarch64 _weight_norm_interface, dim=0.
  // Dyadic inputs cover differing magnitudes and partial vector tails.
  const std::vector<int> lengths{3,4,17,1025,4097};
  const std::vector<float> scales{
      0x1.92df28p-2f,0x1.918676p-2f,0x1.5a299ap-2f,
      0x1.049cf8p-4f,0x1.050908p-5f};
  for (std::size_t row=0;row<lengths.size();++row) {
    const int n=lengths[row];std::vector<float> values(n);
    for(int j=0;j<n;++j) values[j]=std::ldexp(float((j*37)%61-30),-(j%12+2));
    auto result=ag::weight_norm(Tensor::from_host(values,{1,n}),Tensor::from_host({3},{1,1})).to_host();
    INFO("reduction length="<<n);
    for(int j=0;j<n;++j) CHECK(result[j]==values[j]*scales[row]);
  }
}

TEST_CASE("Clamped log and sqrt have zero plateau gradients", "[autograd][bounds][gradcheck]") {
  auto x = Tensor::from_host({-4.0f, 0.0f, 4.0f}, {3}, true);
  auto root = ag::sqrt_op(x);
  REQUIRE(root.to_host() == std::vector<float>{0, 0, 2});
  ag::backward(ag::sum(root));
  CHECK(x.grad_to_host() == std::vector<float>{0, 0, 0.25f});

  auto small = Tensor::from_host({-1.0f, 0.0f, 1e-13f, 1e-12f, 1.0f}, {5}, true);
  auto logarithm = ag::log_op(small);
  for (auto value : logarithm.to_host()) REQUIRE(std::isfinite(value));
  ag::backward(ag::sum(logarithm));
  CHECK(small.grad_to_host() == std::vector<float>{0, 0, 0, 0, 1});
  const auto left = ag::log_op(Tensor::from_host({0.9e-13f}, {1})).to_host()[0];
  const auto right = ag::log_op(Tensor::from_host({1.1e-13f}, {1})).to_host()[0];
  CHECK((right - left) / 2e-14f == small.grad_to_host()[2]);
}

TEST_CASE("Softmax backward preserves FP32 cancellation across vector tails", "[autograd][joint]") {
  // Independently evaluated with the frozen torch CPU/aarch64 softmax kernel.
  const std::vector<int> lengths{3,4,17,128,129};
  const std::vector<float> dot_products{
      -0x1.48p+1f,-0x1.c4p+0f,-0x1.e910fp-2f,-0x1.a9aep-5f,-0x1.a5d254p-5f};
  for (std::size_t i=0;i<lengths.size();++i) {
    const int n=lengths[i];
    std::vector<float> dy(n);
    for (int j=0;j<n;++j) dy[j]=std::ldexp(float((j*37)%61-30),-(j%12+2));
    auto x=Tensor::zeros({1,n},true);
    auto y=ag::softmax_rows(x,1,n);
    ag::backward(ag::sum(ag::mul(y,Tensor::from_host(dy,{1,n}))));
    const auto grad=x.grad_to_host();
    const float probability=1.f/n;
    INFO("softmax row length="<<n);
    for(int j=0;j<n;++j) CHECK(grad[j]==(dy[j]-dot_products[i])*probability);
  }
}

TEST_CASE("Single-row conditioning preserves reference FP32 dot products", "[autograd][joint]") {
  // Independently evaluated CPU/aarch64 Conv1d(g, W, bias), as used by RVC.
  // A canonical 2D torch linear input can select a different BLAS transpose path.
  const std::vector<int> sizes{256};
  const std::vector<std::vector<float>> expected{
      {-0x1.a29b5p-6f,-0x1.96816cp-3f,0x1.4b2fe4p-4f,0x1.039cd8p-3f,
       0x1.20f1ecp-4f,-0x1.493a08p-3f,0x1.6879bp-2f}};
  for(std::size_t i=0;i<sizes.size();++i) {
    const int n=sizes[i];std::vector<float> x(n),w(7*n),bias(7);
    for(int j=0;j<n;++j)x[j]=float((j*13)%29-14)/37.f;
    for(int j=0;j<7*n;++j)w[j]=float((j*37)%61-30)/113.f;
    for(int j=0;j<7;++j)bias[j]=float(j-3)/17.f;
    const auto out=ag::linear(Tensor::from_host(x,{1,n}),
        Tensor::from_host(w,{7,n}),Tensor::from_host(bias,{7})).to_host();
    for(int j=0;j<7;++j)CHECK(out[j]==expected[i][j]);
  }
}

TEST_CASE("AdamW rejects non-finite betas and invalid learning-rate updates", "[autograd][bounds]") {
  auto parameter = Tensor::from_host({1.0f}, {1}, true);
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  const auto inf = std::numeric_limits<float>::infinity();
  for (float beta : {nan, inf, -inf, -0.1f, 1.0f}) {
    CHECK_THROWS_AS(ag::AdamW({parameter}, 0.01f, beta, 0.99f), std::invalid_argument);
    CHECK_THROWS_AS(ag::AdamW({parameter}, 0.01f, 0.9f, beta), std::invalid_argument);
  }
  ag::AdamW optimizer({parameter}, 0.01f);
  for (float lr : {nan, inf, -inf, 0.0f, -1.0f}) {
    CHECK_THROWS_AS(optimizer.set_lr(lr), std::invalid_argument);
    CHECK(optimizer.lr() == 0.01f);
  }
  optimizer.set_lr(0.02f);
  ag::backward(ag::sum(parameter));
  optimizer.step();
  CHECK(std::abs(parameter.to_host()[0] - 0.98f) < 1e-6f);
}

TEST_CASE("AdamW gradient norm retains contributions beyond float unit precision",
          "[autograd][optim][reduction]") {
  // A single float atomic accumulator loses every +1 after 2^24. This size
  // also exercises a reduction tail; the total is exactly representable.
  constexpr int n = (1 << 24) + 4096;
  auto parameter = Tensor::zeros({n}, true);
  parameter.n->ensure_grad();
  std::vector<float> gradient(n, 1.0f);
  parameter.n->grad.copy_from_host(gradient.data(), gradient.size());
  ag::AdamW optimizer({parameter}, 0.01f);
  const float expected = std::sqrt(static_cast<float>(n));
  const float norm = optimizer.clip_grad_norm(expected * 0.5f);
  CHECK(std::abs(norm / expected - 1.0f) < 1e-6f);
  const auto clipped = parameter.grad_to_host();
  REQUIRE(clipped.size() == static_cast<std::size_t>(n));
  CHECK(std::all_of(clipped.begin(), clipped.end(),
                    [](float value) { return std::abs(value - 0.5f) < 1e-6f; }));
}

TEST_CASE("AdamW gradient norm rejects non-finite values and handles empty gradients",
          "[autograd][optim][reduction]") {
  auto parameter = Tensor::zeros({257}, true);
  ag::AdamW optimizer({parameter}, 0.01f);
  CHECK(optimizer.clip_grad_norm(1.0f) == 0.0f);
  parameter.n->ensure_grad();
  CHECK(optimizer.clip_grad_norm(1.0f) == 0.0f);
  for (float value : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity()}) {
    std::vector<float> gradient(257, 0.0f);
    gradient.back() = value;
    parameter.n->grad.copy_from_host(gradient.data(), gradient.size());
    CHECK_THROWS_AS(optimizer.clip_grad_norm(1.0f), std::runtime_error);
  }
}

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

TEST_CASE("Convolutions add bias after the product reduction", "[autograd][conv]") {
  const float large=0x1p24f;
  auto x=Tensor::from_host({1,1},{2,1});
  auto w=Tensor::from_host({large,-large},{2,1,1});
  auto b=Tensor::from_host({1},{1});
  CHECK(ag::conv_transpose1d(x,w,b,2,1,1,1,1,0).to_host()==std::vector<float>{1});
  x=Tensor::from_host({1,1,1,1},{2,2});
  w=Tensor::from_host({large,-large,large,-large},{2,1,2});
  b=Tensor::from_host({1,-1},{2});
  CHECK(ag::conv1d(x,w,b,2,2,2,2,1,0,1,2).to_host()==std::vector<float>{1,-1});
  // Individually small tensors must not overflow the columns workspace index.
  CHECK_THROWS_AS(ag::conv_transpose1d(Tensor::zeros({1,50000}),
      Tensor::zeros({1,1,50000}),{},1,50000,1,50000,1,0),std::length_error);
}

TEST_CASE("Small-input convolution preserves blocked FP32 accumulation", "[autograd][joint][conv]") {
  // Frozen CPU/aarch64 Conv1d: reductions span two and three 640-element blocks.
  const std::vector<int> channels{96,192},indices{0,1,15,31,32,64,127,383};
  const std::vector<std::vector<float>> expected{
      {-0x1.584f9p-2f,-0x1.7625ccp-3f,0x1.936c28p-2f,-0x1.0e19c8p-2f,
       -0x1.2ba464p-1f,0x1.81c27p-2f,-0x1.de919p-3f,0x1.d2c15p-1f},
      {-0x1.86fd3cp-1f,0x1.e8af2p-8f,0x1.bf81eep-2f,0x1.97b73ep-4f,
       -0x1.e4e0f8p-3f,-0x1.3b706ep-2f,-0x1.54eceap-3f,-0x1.b9c578p-5f}};
  for(std::size_t i=0;i<channels.size();++i) {
    const int cin=channels[i];std::vector<float> x(cin*32),w(12*cin*7),b(12);
    for(int j=0;j<int(x.size());++j)x[j]=float((j*13)%29-14)/37.f;
    for(int j=0;j<int(w.size());++j)w[j]=float((j*37)%61-30)/113.f;
    for(int j=0;j<12;++j)b[j]=float(j-3)/17.f;
    const auto out=ag::conv1d(Tensor::from_host(x,{cin,32}),
        Tensor::from_host(w,{12,cin,7}),Tensor::from_host(b,{12}),
        cin,32,12,7,1,3,1,1).to_host();
    for(std::size_t j=0;j<indices.size();++j)CHECK(out[indices[j]]==expected[i][j]);
  }
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

TEST_CASE("Transposed convolution preserves reference channel accumulation", "[autograd][joint][conv]") {
  // Frozen torch CPU/aarch64 conv_transpose1d; includes overlapping output taps.
  std::vector<float> x(17*3),w(17*2*4);
  for(int j=0;j<int(x.size());++j)x[j]=std::ldexp(float((j*13)%29-14),-(j%7+2));
  for(int j=0;j<int(w.size());++j)w[j]=std::ldexp(float((j*37)%61-30),-(j%12+2));
  const std::vector<float> expected{
      -0x1.b19a1p+1f,0x1.54b198p+2f,-0x1.917bap-1f,-0x1.fd4c5p+2f,
      0x1.7f5358p+1f,-0x1.1aab8p+1f,0x1.1e69p+1f,-0x1.6ee7e4p+4f,
      0x1.1e488p-2f,-0x1.05bc3ap+3f,0x1.6477ap+1f,0x1.4b02ep+1f};
  auto actual=ag::conv_transpose1d(Tensor::from_host(x,{17,3}),
      Tensor::from_host(w,{17,2,4}),Tensor::from_host({.75f,-.125f},{2}),
      17,3,2,4,2,1).to_host();
  REQUIRE(actual.size()==expected.size());
  for(std::size_t j=0;j<expected.size();++j)CHECK(actual[j]==expected[j]);
  x.resize(512*32);w.resize(512*16*4);
  for(int j=0;j<int(x.size());++j)x[j]=float((j*13)%29-14)/37.f;
  for(int j=0;j<int(w.size());++j)w[j]=float((j*37)%61-30)/113.f;
  actual=ag::conv_transpose1d(Tensor::from_host(x,{512,32}),
      Tensor::from_host(w,{512,16,4}),{},512,32,16,4,2,1).to_host();
  const std::vector<int> indices{0,1,2,15,31,32,63,64,127,255,511,1023};
  const std::vector<float> wide_expected{
      -0x1.4406a6p-2f,-0x1.02a208p-1f,-0x1.27dc44p-2f,-0x1.3de71cp-1f,
      0x1.6189d4p-1f,0x1.0dd674p+0f,-0x1.767a8ap-1f,0x1.9941c2p-2f,
      0x1.a7d468p-1f,0x1.17f0d4p-1f,0x1.83b458p-2f,-0x1.c86734p-1f};
  REQUIRE(actual.size()==1024);
  for(std::size_t j=0;j<indices.size();++j)CHECK(actual[indices[j]]==wide_expected[j]);
}

TEST_CASE("Period convolution preserves reference FP32 accumulation", "[autograd][joint][conv]") {
  // Independent CPU/aarch64 Conv2d; includes both sides of the backend cutoff,
  // non-aligned heights and two period widths. Values are exact FP32 outputs.
  struct Case {int channels,height,period; std::vector<float> expected;};
  const std::vector<Case> cases{
      {1,256,5,{-0x1.3968700000000p-3f,-0x1.3fc69a0000000p-3f,-0x1.6773d20000000p-3f,-0x1.d79fe80000000p-3f,-0x1.e266a80000000p-3f,-0x1.1512de0000000p-4f,-0x1.2cac180000000p-3f,0x1.794d760000000p-2f}},
      {32,256,5,{-0x1.4c8e080000000p-2f,0x1.f62ace0000000p-3f,-0x1.f96c5e0000000p-3f,-0x1.201fb20000000p-1f,0x1.66a5400000000p-3f,0x1.e711700000000p-4f,0x1.da817a0000000p-5f,0x1.318aac0000000p-2f}},
      {128,64,5,{0x1.16662e0000000p-4f,0x1.4a8de20000000p-7f,-0x1.bf21fe0000000p-3f,-0x1.26cb660000000p-3f,-0x1.d4dce20000000p-7f,-0x1.1e497c0000000p-1f,0x1.8add5a0000000p-3f,0x1.65d6dc0000000p-3f}},
      {32,257,3,{-0x1.c195020000000p-3f,-0x1.5bb24e0000000p-3f,-0x1.b9beb00000000p-3f,-0x1.4579480000000p-1f,-0x1.752ae20000000p-3f,-0x1.b9beb00000000p-3f,-0x1.7720820000000p-3f,0x1.6c13b40000000p-2f}}};
  for(const auto& c:cases) {
    std::vector<float> x(c.channels*c.height*c.period),w(12*c.channels*5),b(12);
    for(int j=0;j<int(x.size());++j)x[j]=float((j*13)%29-14)/37.f;
    for(int j=0;j<int(w.size());++j)w[j]=float((j*37)%61-30)/113.f;
    for(int j=0;j<12;++j)b[j]=float(j-3)/17.f;
    auto out=ag::conv2d(Tensor::from_host(x,{c.channels,c.height,c.period}),
        Tensor::from_host(w,{12,c.channels,5,1}),Tensor::from_host(b,{12}),
        c.channels,c.height,c.period,12,5,1,3,1,2,0).to_host();
    const std::vector<int> indices{0,1,15,31,32,64,127,int(out.size())-1};
    INFO("channels="<<c.channels<<" height="<<c.height<<" period="<<c.period);
    for(std::size_t j=0;j<indices.size();++j)CHECK(out[indices[j]]==c.expected[j]);
  }
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
  SECTION("flip_rows") {
    double e = grad_check(
        [](std::vector<Tensor>& t) {
          return ag::sum(ag::gelu(ag::flip_rows(t[0], 4, 3)));
        },
        {rand_vec(12, rng)}, {{4, 3}});
    std::printf("[flip_rows] %.2e\n", e);
    CHECK(e < 3e-2);
  }
  SECTION("reflect_pad_end") {
    const std::vector<float> input = {1, 2, 3, 4, 5};
    auto padded = ag::reflect_pad_end(Tensor::from_host(input, {1, 5}, false), 7).to_host();
    const std::vector<float> expected{1, 2, 3, 4, 5, 4, 3};
    REQUIRE(padded == expected);
    double e = grad_check(
        [](std::vector<Tensor>& t) {
          return ag::sum(ag::gelu(ag::reflect_pad_end(t[0], 7)));
        },
        {input}, {{1, 5}});
    std::printf("[reflect_pad_end] %.2e\n", e);
    CHECK(e < 3e-2);
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

TEST_CASE("Autograd rejects invalid public shapes and indices", "[autograd][bounds]") {
  REQUIRE_THROWS_AS(Tensor::from_host({1.0f}, {2}, false), std::invalid_argument);

  auto a = Tensor::from_host({1.0f, 2.0f}, {1, 2}, false);
  auto b = Tensor::from_host({1.0f}, {1, 1}, false);
  REQUIRE_THROWS_AS(ag::add(a, b), std::invalid_argument);
  REQUIRE_THROWS_AS(ag::matmul(a, a), std::invalid_argument);
  REQUIRE_THROWS_AS(ag::slice_rows(a, 1, 1, 2), std::out_of_range);
  REQUIRE_THROWS_AS(ag::frame(a, 2, 2, 1), std::out_of_range);

  auto table = Tensor::from_host(std::vector<float>(6, 0.0f), {3, 2}, false);
  REQUIRE_THROWS_AS(ag::embedding(table, {-1}, 2), std::out_of_range);
  REQUIRE_THROWS_AS(ag::embedding(table, {3}, 2), std::out_of_range);
}

TEST_CASE("Recycled graph storage cannot retain values gradients or parents", "[autograd][reuse]") {
  std::weak_ptr<ag::Node> previous;
  for (int step=0;step<12;++step) {
    CHECK(previous.expired());
    const int size=step%2 ? 257 : 31;
    const float value=0.25f*(step+1);
    auto x=Tensor::from_host(std::vector<float>(size,value),{size},true);
    previous=x.n;
    CHECK(x.n->parents.empty());
    CHECK_FALSE(static_cast<bool>(x.n->backward_fn));
    for(float gradient:x.grad_to_host()) CHECK(gradient==0.0f);
    auto square=ag::mul(x,x);
    auto zero=Tensor::zeros({size});
    for(float item:zero.to_host()) CHECK(item==0.0f);
    ag::backward(ag::sum(square));
    for(float gradient:x.grad_to_host()) CHECK(gradient==2*value);
    for(float item:x.to_host()) CHECK(item==value);
  }
  CHECK(previous.expired());
}

TEST_CASE("Recycled parameters still have no gradient before backward", "[autograd][reuse][missing-gradient]") {
  {
    auto old=Tensor::from_host(std::vector<float>(113,1.0f),{113},true);
    ag::backward(ag::sum(ag::mul(old,old)));
  }
  auto parameter=Tensor::from_host(std::vector<float>(113,2.0f),{113},true);
  CHECK(parameter.n->grad.empty());
  ag::AdamW optimizer({parameter},0.1f,0.9f,0.999f,1e-8f,0.1f);
  optimizer.step();
  for(float value:parameter.to_host()) CHECK(value==2.0f);
}

TEST_CASE("AdamW restore preserves the next update and rejects partial invalid state", "[autograd][optim][resume]") {
  auto x=Tensor::from_host({1.f,-2.f,3.f},{3},true);
  ag::AdamW original({x},1e-4f,.8f,.99f,1e-9f,.01f);
  ag::backward(ag::sum(ag::mul(x,x)));original.step();original.set_lr(9e-5f);
  const auto first=original.first_moment(0),second=original.second_moment(0);
  auto y=Tensor::from_host(x.to_host(),{3},true);
  ag::AdamW restored({y},1e-4f,.8f,.99f,1e-9f,.01f);
  restored.restore_state(original.step_count(),original.lr(),[&](std::size_t i,bool v){
    REQUIRE(i==0);return v?second:first;
  });
  REQUIRE(restored.step_count()==original.step_count());REQUIRE(restored.lr()==original.lr());
  for(int mode=0;mode<5;++mode) {
    CHECK_THROWS(restored.restore_state(mode==0?-1:1,mode==1?0.f:1e-4f,
      [&](std::size_t,bool v){
        auto values=v?second:first;
        if(mode==2 && v)values.pop_back();
        if(mode==3 && v)values[0]=-1;
        if(mode==4 && v)values[0]=std::numeric_limits<float>::infinity();
        return values;
      }));
    CHECK(restored.step_count()==1);CHECK(restored.lr()==original.lr());
    CHECK(restored.first_moment(0)==first);CHECK(restored.second_moment(0)==second);
  }
  ag::backward(ag::sum(ag::mul(x,x)));original.step();
  ag::backward(ag::sum(ag::mul(y,y)));restored.step();
  CHECK(y.to_host()==x.to_host());
  CHECK(restored.first_moment(0)==original.first_moment(0));
  CHECK(restored.second_moment(0)==original.second_moment(0));
}
