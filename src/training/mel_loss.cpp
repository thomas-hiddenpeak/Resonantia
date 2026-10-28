// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/training/mel_loss.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
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
  if (cfg.rvc_profile && (cfg.n_fft != 2048 || cfg.hop != 400 || cfg.n_mels != 125 ||
      cfg.sample_rate != 40000 || cfg.fmin != 0 || (cfg.fmax != 0 && cfg.fmax != 20000)))
    throw std::invalid_argument("RVC mel profile requires 40k/2048/400/125 and the full frequency band");
  const double nyquist = static_cast<double>(cfg.sample_rate) / 2;
  const double upper = cfg.fmax == 0 ? nyquist : cfg.fmax;
  if (cfg.n_fft <= 1 || cfg.hop <= 0 || cfg.n_mels <= 0 || cfg.sample_rate <= 0 ||
      cfg.n_mels > std::numeric_limits<int>::max() - 2 ||
      !std::isfinite(cfg.fmin) || !std::isfinite(cfg.fmax) ||
      cfg.fmin < 0 || cfg.fmax < 0 || cfg.fmin >= upper || upper > nyquist)
    throw std::invalid_argument("invalid mel configuration or frequency range");
  n_freq_ = cfg_.n_fft / 2 + 1;
  if (static_cast<std::int64_t>(cfg.n_fft) * n_freq_ > std::numeric_limits<int>::max() ||
      static_cast<std::int64_t>(cfg.n_mels) * n_freq_ > std::numeric_limits<int>::max())
    throw std::invalid_argument("mel matrix dimensions exceed kernel limits");
  int N = cfg_.n_fft, F = n_freq_;
  float fmax = cfg_.fmax > 0 ? cfg_.fmax : cfg_.sample_rate * 0.5f;

  // Hann (periodic) window folded into DFT matrices.
  std::vector<float> window(N);
  for (int n = 0; n < N; ++n)
    window[n] = 0.5f - 0.5f * std::cos(2.0f * M_PI * n / N);

  std::vector<float> cosw(N * F), sinw(N * F);
  for (int n = 0; n < N; ++n) {
    for (int f = 0; f < F; ++f) {
      if (cfg.rvc_profile) {
        const double ang = 2.0 * M_PI * f * n / N;
        cosw[n * F + f] = window[n] * std::cos(ang);
        sinw[n * F + f] = -window[n] * std::sin(ang);
      } else {
        float ang = 2.0f * M_PI * f * n / N;
        cosw[n * F + f] = window[n] * std::cos(ang);
        sinw[n * F + f] = -window[n] * std::sin(ang);
      }
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
    if (!(lo > 0) || !(hi > 0))
      throw std::invalid_argument("mel frequency bands are not representable");
    float enorm = 2.0f / (freqs[m + 2] - freqs[m]);
    if (!std::isfinite(enorm))
      throw std::invalid_argument("mel filter normalization is not finite");
    for (int f = 0; f < F; ++f) {
      float l = (fftfreqs[f] - freqs[m]) / lo;
      float r = (freqs[m + 2] - fftfreqs[f]) / hi;
      float w = std::max(0.0f, std::min(l, r));
      mel_t[f * M + m] = w * enorm;
    }
  }
  mel_basis_t_ = ag::Tensor::from_host(mel_t, {F, M}, false);
  if (cfg.rvc_profile) {
    // librosa's Slaney bank computes the frequency geometry in float64.
    const auto hz = [](double m) {
      return m < 15.0 ? m*(200.0/3.0) : 1000.0*std::exp(std::log(6.4)/27.0*(m-15.0));
    };
    const double max_mel = 15.0+std::log(20000.0/1000.0)/(std::log(6.4)/27.0);
    std::vector<double> bands(M+2);
    for (int i=0;i<M+2;++i) bands[i]=hz(max_mel*i/(M+1));
    for (int f=0;f<F;++f) for (int m=0;m<M;++m) {
      const double freq=40000.0*f/N;
      const double tri=std::max(0.0,std::min((freq-bands[m])/(bands[m+1]-bands[m]),
                                          (bands[m+2]-freq)/(bands[m+2]-bands[m+1])));
      // librosa casts triangles to F32 before applying area normalization.
      mel_t[f*M+m]=static_cast<float>(tri)*(2.0/(bands[m+2]-bands[m]));
    }
    mel_basis_t_=ag::Tensor::from_host(mel_t,{F,M},false);
  }
}

int MelLoss::num_frames(int L) const {
  if (L < 0) throw std::invalid_argument("mel input length must be nonnegative");
  if (cfg_.rvc_profile) {
    const int pad=(cfg_.n_fft-cfg_.hop)/2;
    if (L <= pad || L > std::numeric_limits<int>::max()-2*pad)
      throw std::invalid_argument("RVC mel input must exceed reflect padding and fit kernel limits");
    return 1+(L+2*pad-cfg_.n_fft)/cfg_.hop;
  }
  if (L < cfg_.n_fft) return 0;
  return (L - cfg_.n_fft) / cfg_.hop + 1;
}

ag::Tensor MelLoss::log_mel(const ag::Tensor& audio, int L) const {
  int T = num_frames(L);
  if (T <= 0 || T > std::numeric_limits<int>::max() / cfg_.n_fft ||
      T > std::numeric_limits<int>::max() / cfg_.n_mels ||
      !audio.n || audio.shape() != std::vector<int>{1, L})
    throw std::invalid_argument("mel audio must have shape [1, L] and at least one frame");
  auto padded = audio;
  if (cfg_.rvc_profile) {
    const int pad=(cfg_.n_fft-cfg_.hop)/2;
    std::vector<int> indices(L+2*pad);
    for (int i=0;i<L+2*pad;++i) {
      const int pos=i-pad;
      indices[i]=pos<0 ? -pos : (pos>=L ? 2*L-2-pos : pos);
    }
    padded=ag::reshape(ag::embedding(ag::reshape(audio,{L,1}),indices,1),{1,L+2*pad});
  }
  auto frames = ag::frame(padded, T, cfg_.n_fft, cfg_.hop);   // [T, n_fft]
  auto re = ag::matmul(frames, cosw_);                       // [T, F]
  auto im = ag::matmul(frames, sinw_);                       // [T, F]
  auto power = ag::add(ag::mul(re, re), ag::mul(im, im));    // [T, F]
  if (cfg_.rvc_profile)
    power=ag::add(power,ag::Tensor::from_host(std::vector<float>(power.numel(),2e-7f),power.shape()));
  auto mag = ag::sqrt_op(power);                             // [T, F]
  auto mel = ag::matmul(mag, mel_basis_t_);                  // [T, n_mels]
  return ag::log_op(cfg_.rvc_profile ? ag::clamp_min(mel,2e-6f) : mel);
}

std::vector<float> MelLoss::target_log_mel(const float* audio, int L,
                                           int& out_T) const {
  const int frames = num_frames(L);
  if (!audio || frames <= 0 || frames > std::numeric_limits<int>::max() / cfg_.n_fft ||
      frames > std::numeric_limits<int>::max() / cfg_.n_mels)
    throw std::invalid_argument("mel target requires audio and at least one frame");
  std::vector<float> host(audio, audio + L);
  auto a = ag::Tensor::from_host(host, {1, L}, false);
  auto result = log_mel(a, L).to_host();
  out_T = frames;
  return result;
}

ag::Tensor MelLoss::l1(const ag::Tensor& gen_logmel, const ag::Tensor& target_const,
                       int T) const {
  if (T <= 0 || !gen_logmel.n || !target_const.n ||
      gen_logmel.shape() != std::vector<int>{T, cfg_.n_mels} ||
      target_const.shape() != gen_logmel.shape() ||
      static_cast<std::int64_t>(T) * cfg_.n_mels > std::numeric_limits<int>::max())
    throw std::invalid_argument("mel loss tensors must have matching [T, n_mels] shapes");
  auto neg = ag::scale(target_const, -1.0f);
  auto diff = ag::add(gen_logmel, neg);
  auto s = ag::sum(ag::abs_op(diff));
  return ag::scale(s, 1.0f / (T * cfg_.n_mels));
}

}  // namespace voxmutatio::training
