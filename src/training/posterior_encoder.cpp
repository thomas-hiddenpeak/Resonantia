// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/training/posterior_encoder.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "voxmutatio/io/safetensors.h"
#include "voxmutatio/separation/stft.h"

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

}  // namespace

std::vector<float> compute_spec_host(const float* audio, int L, int n_fft, int hop, int& out_T) {
  const int pad = (n_fft - hop) / 2;
  const int nfreq = n_fft / 2 + 1;
  const int T = L / hop;
  out_T = T;

  std::vector<float> y(L + 2 * pad);
  for (int i = 0; i < L; ++i) y[pad + i] = audio[i];
  for (int j = 0; j < pad; ++j) {
    y[pad - 1 - j] = audio[std::min(j + 1, L - 1)];        // reflect left
    y[pad + L + j] = audio[std::max(L - 2 - j, 0)];        // reflect right
  }

  std::vector<float> win(n_fft);
  for (int n = 0; n < n_fft; ++n)
    win[n] = 0.5f - 0.5f * std::cos(2.0 * M_PI * n / n_fft);

  // Precompute DFT bases.
  std::vector<float> cosb(static_cast<std::size_t>(nfreq) * n_fft);
  std::vector<float> sinb(static_cast<std::size_t>(nfreq) * n_fft);
  for (int k = 0; k < nfreq; ++k)
    for (int n = 0; n < n_fft; ++n) {
      double a = 2.0 * M_PI * k * n / n_fft;
      cosb[k * n_fft + n] = static_cast<float>(std::cos(a)) * win[n];
      sinb[k * n_fft + n] = static_cast<float>(std::sin(a)) * win[n];
    }

  std::vector<float> spec(static_cast<std::size_t>(nfreq) * T);
  for (int t = 0; t < T; ++t) {
    const float* frame = &y[t * hop];
    for (int k = 0; k < nfreq; ++k) {
      const float* c = &cosb[k * n_fft];
      const float* s = &sinb[k * n_fft];
      double re = 0.0, im = 0.0;
      for (int n = 0; n < n_fft; ++n) { re += frame[n] * c[n]; im -= frame[n] * s[n]; }
      spec[k * T + t] = static_cast<float>(std::sqrt(re * re + im * im));
    }
  }
  return spec;
}

std::vector<float> compute_spec(const float* audio, int L, int n_fft, int hop, int& out_T) {
  // GPU cuFFT STFT with VITS reflect pad = (n_fft - hop)/2 and periodic Hann.
  // Numerically matches compute_spec_host to ~1e-7 (see test_separation [spec]).
  separation::Stft stft(n_fft, hop, n_fft, true, (n_fft - hop) / 2);
  return stft.magnitude(audio, L, out_T);
}

void WaveNet::load(const io::SafetensorsLoader& L, const std::string& prefix,
                   int hidden, int n_layers, int kernel, int speaker_id,
                   std::vector<ag::Tensor>& params_out) {
  hidden_ = hidden;
  kernel_ = kernel;
  const int two_h = 2 * hidden;

  // cond(g) folded per layer: cond_layer Conv1d(gin, 2*hidden*n_layers, 1).
  const int gin = 256;
  auto cond_fused = reconstruct_wn(L, prefix + ".cond_layer", two_h * n_layers, gin);
  auto cond_b = load_plain(L, prefix + ".cond_layer.bias", two_h * n_layers);
  const auto* emb_g = reinterpret_cast<const float*>(L.data("model.emb_g.weight"));
  std::vector<float> cond(two_h * n_layers);
  for (int o = 0; o < two_h * n_layers; ++o) {
    float s = cond_b[o];
    for (int ci = 0; ci < gin; ++ci) s += cond_fused[o * gin + ci] * emb_g[speaker_id * gin + ci];
    cond[o] = s;
  }

  auto param = [&](std::vector<float> host, std::vector<int> shape) {
    auto t = ag::Tensor::from_host(host, std::move(shape), true);
    params_out.push_back(t);
    return t;
  };

  for (int i = 0; i < n_layers; ++i) {
    WNLayer ly;
    std::string in = prefix + ".in_layers." + std::to_string(i);
    ly.in_w = param(reconstruct_wn(L, in, two_h, hidden * kernel), {two_h, hidden, kernel});
    auto in_b = load_plain(L, in + ".bias", two_h);
    for (int c = 0; c < two_h; ++c) in_b[c] += cond[i * two_h + c];  // fold cond
    ly.in_b = param(in_b, {two_h});

    int rs_ch = (i < n_layers - 1) ? two_h : hidden;
    std::string rs = prefix + ".res_skip_layers." + std::to_string(i);
    ly.rs_w = param(reconstruct_wn(L, rs, rs_ch, hidden), {rs_ch, hidden, 1});
    ly.rs_b = param(load_plain(L, rs + ".bias", rs_ch), {rs_ch});
    ly.rs_ch = rs_ch;
    layers_.push_back(ly);
  }
}

ag::Tensor WaveNet::forward(const ag::Tensor& x_in, int T) const {
  const int h = hidden_, two_h = 2 * hidden_, pad = (kernel_ - 1) / 2;
  ag::Tensor x = x_in;
  ag::Tensor output;
  const int n = static_cast<int>(layers_.size());
  for (int i = 0; i < n; ++i) {
    const WNLayer& ly = layers_[i];
    auto xin = ag::conv1d(x, ly.in_w, ly.in_b, h, T, two_h, kernel_, 1, pad, 1, 1);
    auto tt = ag::tanh_op(ag::slice_rows(xin, 0, h, T));
    auto ss = ag::sigmoid(ag::slice_rows(xin, h, h, T));
    auto acts = ag::mul(tt, ss);
    auto rs = ag::conv1d(acts, ly.rs_w, ly.rs_b, h, T, ly.rs_ch, 1, 1, 0, 1, 1);
    if (i < n - 1) {
      x = ag::add(x, ag::slice_rows(rs, 0, h, T));
      auto skip = ag::slice_rows(rs, h, h, T);
      output = output.n ? ag::add(output, skip) : skip;
    } else {
      output = output.n ? ag::add(output, rs) : rs;
    }
  }
  return output;
}

bool PosteriorEncoder::init(const std::string& path, int speaker_id) {
  io::SafetensorsLoader L;
  if (!L.load(path)) { fprintf(stderr, "PosteriorEncoder: cannot load %s\n", path.c_str()); return false; }
  const std::string P = "model.enc_q.";

  auto param = [&](std::vector<float> host, std::vector<int> shape) {
    auto t = ag::Tensor::from_host(host, std::move(shape), true);
    params_.push_back(t);
    return t;
  };
  pre_w_ = param(load_plain(L, P + "pre.weight", kHidden * kSpec), {kHidden, kSpec, 1});
  pre_b_ = param(load_plain(L, P + "pre.bias", kHidden), {kHidden});
  wn_.load(L, P + "enc", kHidden, 16, 5, speaker_id, params_);
  proj_w_ = param(load_plain(L, P + "proj.weight", 2 * kInter * kHidden), {2 * kInter, kHidden, 1});
  proj_b_ = param(load_plain(L, P + "proj.bias", 2 * kInter), {2 * kInter});
  return true;
}

ag::Tensor PosteriorEncoder::forward(const ag::Tensor& spec, int T, bool sample,
                                     ag::Tensor& m_q, ag::Tensor& logs_q) {
  auto x = ag::conv1d(spec, pre_w_, pre_b_, kSpec, T, kHidden, 1, 1, 0, 1, 1);
  x = wn_.forward(x, T);
  auto stats = ag::conv1d(x, proj_w_, proj_b_, kHidden, T, 2 * kInter, 1, 1, 0, 1, 1);
  m_q = ag::slice_rows(stats, 0, kInter, T);
  logs_q = ag::slice_rows(stats, kInter, kInter, T);
  if (!sample) return m_q;

  static std::mt19937 rng(12345);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> eps(static_cast<std::size_t>(kInter) * T);
  for (auto& e : eps) e = nd(rng);
  auto eps_t = ag::Tensor::from_host(eps, {kInter, T}, false);
  return ag::add(m_q, ag::mul(ag::exp_op(logs_q), eps_t));
}

bool Flow::init(const std::string& path, int speaker_id) {
  io::SafetensorsLoader L;
  if (!L.load(path)) { fprintf(stderr, "Flow: cannot load %s\n", path.c_str()); return false; }
  const int hidden = 192, half = 96;
  auto param = [&](std::vector<float> host, std::vector<int> shape) {
    auto t = ag::Tensor::from_host(host, std::move(shape), true);
    params_.push_back(t);
    return t;
  };
  for (int f = 0; f < 4; ++f) {
    std::string p = "model.flow.flows." + std::to_string(f * 2) + ".";
    Coupling& c = couplings_[f];
    c.pre_w = param(load_plain(L, p + "pre.weight", hidden * half), {hidden, half, 1});
    c.pre_b = param(load_plain(L, p + "pre.bias", hidden), {hidden});
    c.wn.load(L, p + "enc", hidden, 3, 5, speaker_id, params_);
    c.post_w = param(load_plain(L, p + "post.weight", half * hidden), {half, hidden, 1});
    c.post_b = param(load_plain(L, p + "post.bias", half), {half});
  }
  return true;
}

ag::Tensor Flow::forward(const ag::Tensor& z, int T) const {
  const int inter = 192, half = 96, hidden = 192;
  ag::Tensor x = z;
  for (int f = 0; f < 4; ++f) {
    const Coupling& c = couplings_[f];
    auto x0 = ag::slice_rows(x, 0, half, T);
    auto x1 = ag::slice_rows(x, half, half, T);
    auto h = ag::conv1d(x0, c.pre_w, c.pre_b, half, T, hidden, 1, 1, 0, 1, 1);
    h = c.wn.forward(h, T);
    auto m = ag::conv1d(h, c.post_w, c.post_b, hidden, T, half, 1, 1, 0, 1, 1);
    auto x1n = ag::add(x1, m);
    x = ag::concat_rows(x0, x1n, T);
    x = ag::flip_rows(x, inter, T);
  }
  return x;
}

}  // namespace voxmutatio::training
