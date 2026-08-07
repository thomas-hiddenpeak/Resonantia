// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/training/mel_loss.h"

#include <cmath>
#include <vector>

namespace voxmutatio::training {

namespace ag = voxmutatio::autograd;

namespace {

float hz_to_mel(float hz) {
  const float f_sp = 200.0f / 3.0f;
  float mel = hz / f_sp;
  const float min_log_hz = 1000.0f, min_log_mel = 1000.0f / f_sp;
  const float logstep = std::log(6.4f) / 27.0f;
  if (hz >= min_log_hz) mel = min_log_mel + std::log(hz / min_log_hz) / logstep;
  return mel;
}
float mel_to_hz(float mel) {
  const float f_sp = 200.0f / 3.0f;
  float hz = f_sp * mel;
  const float min_log_hz = 1000.0f, min_log_mel = 1000.0f / f_sp;
  const float logstep = std::log(6.4f) / 27.0f;
  if (mel >= min_log_mel) hz = min_log_hz * std::exp(logstep * (mel - min_log_mel));
  return hz;
}

}  // namespace

MelLoss::MelLoss(const MelSpecConfig& cfg) : cfg_(cfg) {
  n_freq_ = cfg_.n_fft / 2 + 1;
  int N = cfg_.n_fft, F = n_freq_;
  float fmax = cfg_.fmax > 0 ? cfg_.fmax : cfg_.sample_rate * 0.5f;

  // Hann (periodic) window folded into DFT matrices.
  std::vector<float> window(N);
  for (int n = 0; n < N; ++n)
    window[n] = 0.5f - 0.5f * std::cos(2.0f * M_PI * n / N);

  std::vector<float> cosw(N * F), sinw(N * F);
  for (int n = 0; n < N; ++n) {
    for (int f = 0; f < F; ++f) {
      float ang = 2.0f * M_PI * f * n / N;
      cosw[n * F + f] = window[n] * std::cos(ang);
      sinw[n * F + f] = -window[n] * std::sin(ang);
    }
  }
  cosw_ = ag::Tensor::from_host(cosw, {N, F}, false);
  sinw_ = ag::Tensor::from_host(sinw, {N, F}, false);

  // Slaney mel filterbank [n_mels, F], stored transposed [F, n_mels].
  std::vector<float> fftfreqs(F);
  for (int i = 0; i < F; ++i) fftfreqs[i] = 0.5f * cfg_.sample_rate * i / (F - 1);
  float mel_min = hz_to_mel(cfg_.fmin), mel_max = hz_to_mel(fmax);
  int M = cfg_.n_mels;
  std::vector<float> freqs(M + 2);
  for (int i = 0; i < M + 2; ++i)
    freqs[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (M + 1));

  std::vector<float> mel_t(F * M, 0.0f);
  for (int m = 0; m < M; ++m) {
    float lo = freqs[m + 1] - freqs[m];
    float hi = freqs[m + 2] - freqs[m + 1];
    float enorm = 2.0f / (freqs[m + 2] - freqs[m]);
    for (int f = 0; f < F; ++f) {
      float l = (fftfreqs[f] - freqs[m]) / lo;
      float r = (freqs[m + 2] - fftfreqs[f]) / hi;
      float w = std::max(0.0f, std::min(l, r));
      mel_t[f * M + m] = w * enorm;
    }
  }
  mel_basis_t_ = ag::Tensor::from_host(mel_t, {F, M}, false);
}

int MelLoss::num_frames(int L) const {
  if (L < cfg_.n_fft) return 0;
  return (L - cfg_.n_fft) / cfg_.hop + 1;
}

ag::Tensor MelLoss::log_mel(const ag::Tensor& audio, int L) const {
  int T = num_frames(L);
  auto frames = ag::frame(audio, T, cfg_.n_fft, cfg_.hop);   // [T, n_fft]
  auto re = ag::matmul(frames, cosw_);                       // [T, F]
  auto im = ag::matmul(frames, sinw_);                       // [T, F]
  auto power = ag::add(ag::mul(re, re), ag::mul(im, im));    // [T, F]
  auto mag = ag::sqrt_op(power);                             // [T, F]
  auto mel = ag::matmul(mag, mel_basis_t_);                  // [T, n_mels]
  return ag::log_op(mel);                                    // [T, n_mels]
}

std::vector<float> MelLoss::target_log_mel(const float* audio, int L,
                                           int& out_T) const {
  out_T = num_frames(L);
  std::vector<float> host(audio, audio + L);
  auto a = ag::Tensor::from_host(host, {1, L}, false);
  return log_mel(a, L).to_host();
}

ag::Tensor MelLoss::l1(const ag::Tensor& gen_logmel, const ag::Tensor& target_const,
                       int T) const {
  auto neg = ag::scale(target_const, -1.0f);
  auto diff = ag::add(gen_logmel, neg);
  auto s = ag::sum(ag::abs_op(diff));
  return ag::scale(s, 1.0f / (T * cfg_.n_mels));
}

}  // namespace voxmutatio::training
