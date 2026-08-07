// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/training/discriminator.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "voxmutatio/io/safetensors.h"

namespace voxmutatio::training {

namespace ag = voxmutatio::autograd;

namespace {

std::vector<float> reconstruct_wn(const io::SafetensorsLoader& L, const std::string& name,
                                  int out_dim, int inner) {
  const auto* g = reinterpret_cast<const float*>(L.data(name + ".weight_g"));
  const auto* v = reinterpret_cast<const float*>(L.data(name + ".weight_v"));
  std::vector<float> out(static_cast<std::size_t>(out_dim) * inner);
  for (int o = 0; o < out_dim; ++o) {
    double n2 = 0.0;
    for (int i = 0; i < inner; ++i) { float x = v[o * inner + i]; n2 += (double)x * x; }
    float inv = static_cast<float>(1.0 / std::sqrt(n2));
    for (int i = 0; i < inner; ++i) out[o * inner + i] = g[o] * v[o * inner + i] * inv;
  }
  return out;
}

std::vector<float> load_plain(const io::SafetensorsLoader& L, const std::string& name, int count) {
  const auto* p = reinterpret_cast<const float*>(L.data(name));
  return std::vector<float>(p, p + count);
}

int conv1d_out(int L, int k, int stride, int pad, int dil = 1) {
  return (L + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

}  // namespace

bool Discriminator::init(const std::string& path) {
  io::SafetensorsLoader L;
  if (!L.load(path)) { fprintf(stderr, "Discriminator: cannot load %s\n", path.c_str()); return false; }
  const std::string P = "model.discriminators.";

  auto param = [&](std::vector<float> host, std::vector<int> shape) {
    auto t = ag::Tensor::from_host(host, std::move(shape), true);
    params_.push_back(t);
    return t;
  };

  // ---- DiscriminatorS (index 0): 1D grouped convs ----
  {
    struct Spec { int cin, cout, k, stride, pad, groups; };
    const Spec specs[6] = {
        {1, 16, 15, 1, 7, 1}, {16, 64, 41, 4, 20, 4}, {64, 256, 41, 4, 20, 16},
        {256, 1024, 41, 4, 20, 64}, {1024, 1024, 41, 4, 20, 256}, {1024, 1024, 5, 1, 2, 1}};
    for (int i = 0; i < 6; ++i) {
      const Spec& s = specs[i];
      int inner = (s.cin / s.groups) * s.k;
      std::string n = P + "0.convs." + std::to_string(i);
      Conv1dW c;
      c.w = param(reconstruct_wn(L, n, s.cout, inner), {s.cout, s.cin / s.groups, s.k});
      c.b = param(load_plain(L, n + ".bias", s.cout), {s.cout});
      c.cin = s.cin; c.cout = s.cout; c.k = s.k; c.stride = s.stride; c.pad = s.pad; c.groups = s.groups;
      s_convs_.push_back(c);
    }
    s_post_w_ = param(reconstruct_wn(L, P + "0.conv_post", 1, 1024 * 3), {1, 1024, 3});
    s_post_b_ = param(load_plain(L, P + "0.conv_post.bias", 1), {1});
  }

  // ---- DiscriminatorP (indices 1..8): 2D period convs ----
  const int periods[8] = {2, 3, 5, 7, 11, 17, 23, 37};
  const int pc_cin[5] = {1, 32, 128, 512, 1024};
  const int pc_cout[5] = {32, 128, 512, 1024, 1024};
  for (int d = 0; d < 8; ++d) {
    PDisc pd;
    pd.period = periods[d];
    std::string base = P + std::to_string(d + 1) + ".convs.";
    for (int i = 0; i < 5; ++i) {
      int cin = pc_cin[i], cout = pc_cout[i];
      int sh = (i < 4) ? 3 : 1;
      Conv2dW c;
      c.w = param(reconstruct_wn(L, base + std::to_string(i), cout, cin * 5 * 1),
                  {cout, cin, 5, 1});
      c.b = param(load_plain(L, base + std::to_string(i) + ".bias", cout), {cout});
      c.cin = cin; c.cout = cout; c.kh = 5; c.kw = 1; c.sh = sh; c.sw = 1; c.ph = 2; c.pw = 0;
      pd.convs.push_back(c);
    }
    std::string pp = P + std::to_string(d + 1) + ".conv_post";
    pd.post_w = param(reconstruct_wn(L, pp, 1, 1024 * 3 * 1), {1, 1024, 3, 1});
    pd.post_b = param(load_plain(L, pp + ".bias", 1), {1});
    p_.push_back(std::move(pd));
  }

  return true;
}

std::vector<SubDiscResult> Discriminator::forward(const ag::Tensor& audio, int L) {
  std::vector<SubDiscResult> results;

  // DiscriminatorS
  {
    SubDiscResult r;
    ag::Tensor h = audio;
    int C = 1, Lc = L;
    for (const auto& c : s_convs_) {
      h = ag::conv1d(h, c.w, c.b, C, Lc, c.cout, c.k, c.stride, c.pad, 1, c.groups);
      Lc = conv1d_out(Lc, c.k, c.stride, c.pad);
      C = c.cout;
      h = ag::leaky_relu(h, 0.1f);
      r.fmaps.push_back(h);
    }
    h = ag::conv1d(h, s_post_w_, s_post_b_, 1024, Lc, 1, 3, 1, 1, 1, 1);
    r.fmaps.push_back(h);
    r.score = h;
    results.push_back(std::move(r));
  }

  // DiscriminatorP
  for (const auto& pd : p_) {
    SubDiscResult r;
    int p = pd.period, H = L / p, C = 1;
    ag::Tensor h = audio;  // conv2d reads first H*p samples as [1,H,p]
    for (const auto& c : pd.convs) {
      h = ag::conv2d(h, c.w, c.b, C, H, p, c.cout, c.kh, c.kw, c.sh, c.sw, c.ph, c.pw);
      H = (H + 2 * c.ph - c.kh) / c.sh + 1;
      C = c.cout;
      h = ag::leaky_relu(h, 0.1f);
      r.fmaps.push_back(h);
    }
    h = ag::conv2d(h, pd.post_w, pd.post_b, 1024, H, p, 1, 3, 1, 1, 1, 1, 0);
    r.fmaps.push_back(h);
    r.score = h;
    results.push_back(std::move(r));
  }

  return results;
}

}  // namespace voxmutatio::training
