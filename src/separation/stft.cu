// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/separation/stft.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "voxmutatio/core/cuda_buffer.h"
#include "validation.h"

namespace voxmutatio::separation {

namespace {

void ck_cufft(cufftResult status, const char* operation) {
  if (status != CUFFT_SUCCESS) {
    throw std::runtime_error(std::string("STFT cuFFT failure in ") + operation +
                             " (status " + std::to_string(static_cast<int>(status)) + ")");
  }
}

inline int grid(int n) { return n > 0 ? 1 + (n - 1) / 256 : 0; }

// framed[t, n] = padded[t*hop + n] * window[n]
__global__ void k_frame_window(const float* padded, const float* win, float* framed,
                               int T, int n_fft, int hop) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * n_fft) return;
  int t = i / n_fft, n = i % n_fft;
  framed[i] = padded[t * hop + n] * win[n];
}

// complex [T, n_freq] (cufft layout) -> re/im [n_freq, T] (freq-major)
__global__ void k_split(const cufftComplex* c, float* re, float* im, int T, int nf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * nf) return;
  int t = i / nf, f = i % nf;
  re[f * T + t] = c[i].x;
  im[f * T + t] = c[i].y;
}

// re/im [n_freq, T] -> complex [T, n_freq]
__global__ void k_merge(const float* re, const float* im, cufftComplex* c, int T, int nf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * nf) return;
  int t = i / nf, f = i % nf;
  c[i].x = re[f * T + t];
  c[i].y = im[f * T + t];
}

// Overlap-add: y[t*hop+n] += (framed[t,n]/n_fft) * win[n]; wsum[t*hop+n] += win[n]^2
__global__ void k_ola(const float* framed, const float* win, float* y, float* wsum,
                      int T, int n_fft, int hop, float inv_nfft) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * n_fft) return;
  int t = i / n_fft, n = i % n_fft;
  atomicAdd(&y[t * hop + n], framed[i] * inv_nfft * win[n]);
  atomicAdd(&wsum[t * hop + n], win[n] * win[n]);
}

__global__ void k_norm(float* y, const float* wsum, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] /= fmaxf(wsum[i], 1e-8f);
}

}  // namespace

Stft::Stft(int n_fft, int hop, int win_length, bool center, int pad_override)
    : n_fft_(n_fft), hop_(hop), win_(win_length > 0 ? win_length : n_fft),
      pad_(pad_override >= 0 ? pad_override : (center ? n_fft / 2 : 0)),
      center_(center) {
  if (n_fft_ <= 1 || hop_ <= 0 || win_ <= 0 || win_ > n_fft_ || pad_ < 0) {
    throw std::invalid_argument("STFT requires n_fft > 1, hop > 0, and 0 < win <= n_fft");
  }
  window_.resize(n_fft_, 0.0f);
  // Periodic Hann of length win_, centered in n_fft.
  int off = (n_fft_ - win_) / 2;
  for (int n = 0; n < win_; ++n)
    window_[off + n] = 0.5f - 0.5f * std::cos(2.0 * M_PI * n / win_);
}

int Stft::num_frames(int L) const {
  const std::int64_t padded = static_cast<std::int64_t>(L) + 2LL * pad_;
  if (L <= 0 || padded > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("STFT input length is invalid");
  }
  int Lp = static_cast<int>(padded);
  if (Lp < n_fft_) throw std::invalid_argument("STFT input is shorter than one frame");
  return 1 + (Lp - n_fft_) / hop_;
}

void Stft::forward(const float* audio, int L, std::vector<float>& re,
                   std::vector<float>& im, int& T) const {
  const int nf = n_freq();
  if (!audio) throw std::invalid_argument("STFT input pointer is null");
  T = num_frames(L);
  const int Lp = static_cast<int>(static_cast<std::int64_t>(L) + 2LL * pad_);
  const int frame_elements =
      detail::checked_int_product({T, n_fft_}, "STFT frame tensor");
  const int spectrum_elements =
      detail::checked_int_product({T, nf}, "STFT spectrum tensor");
  const std::size_t complex_storage =
      detail::checked_float_count({T, nf, 2}, "STFT complex tensor");
  detail::checked_float_count({Lp}, "STFT padded input");

  std::vector<float> padded(Lp, 0.0f);
  for (int i = 0; i < L; ++i) padded[pad_ + i] = audio[i];
  for (int j = 0; j < pad_; ++j) {  // reflect
    padded[pad_ - 1 - j] = audio[std::min(j + 1, L - 1)];
    padded[pad_ + L + j] = audio[std::max(L - 2 - j, 0)];
  }

  core::CudaBuffer d_pad(static_cast<std::size_t>(Lp));
  core::CudaBuffer d_win(static_cast<std::size_t>(n_fft_));
  core::CudaBuffer d_framed(static_cast<std::size_t>(frame_elements));
  d_pad.copy_from_host(padded.data(), padded.size());
  d_win.copy_from_host(window_.data(), window_.size());
  k_frame_window<<<grid(frame_elements), 256>>>(d_pad.data(), d_win.data(),
                                            d_framed.data(), T, n_fft_, hop_);

  core::CudaBuffer d_c(complex_storage);
  cufftHandle plan{};
  ck_cufft(cufftPlan1d(&plan, n_fft_, CUFFT_R2C, T), "cufftPlan1d(R2C)");
  cufftResult exec_status = cufftExecR2C(
      plan, d_framed.data(), reinterpret_cast<cufftComplex*>(d_c.data()));
  cufftResult destroy_status = cufftDestroy(plan);
  ck_cufft(exec_status, "cufftExecR2C");
  ck_cufft(destroy_status, "cufftDestroy");

  core::CudaBuffer d_re(static_cast<std::size_t>(spectrum_elements));
  core::CudaBuffer d_im(static_cast<std::size_t>(spectrum_elements));
  k_split<<<grid(spectrum_elements), 256>>>(reinterpret_cast<cufftComplex*>(d_c.data()),
                                 d_re.data(), d_im.data(), T, nf);

  re.resize(static_cast<std::size_t>(spectrum_elements));
  im.resize(static_cast<std::size_t>(spectrum_elements));
  d_re.copy_to_host(re.data(), re.size());
  d_im.copy_to_host(im.data(), im.size());
}

std::vector<float> Stft::inverse(const std::vector<float>& re, const std::vector<float>& im,
                                 int T, int L_out) const {
  const int nf = n_freq();
  if (T <= 0 || L_out <= 0 || T != num_frames(L_out)) {
    throw std::invalid_argument("STFT inverse shape does not match T and output length");
  }
  const int frame_elements =
      detail::checked_int_product({T, n_fft_}, "STFT inverse frame tensor");
  const int spectrum_elements =
      detail::checked_int_product({T, nf}, "STFT inverse spectrum tensor");
  const std::size_t complex_storage =
      detail::checked_float_count({T, nf, 2}, "STFT inverse complex tensor");
  const int Lp =
      static_cast<int>(static_cast<std::int64_t>(L_out) + 2LL * pad_);
  detail::checked_float_count({Lp}, "STFT inverse padded output");
  if (re.size() != static_cast<std::size_t>(spectrum_elements) ||
      im.size() != re.size()) {
    throw std::invalid_argument("STFT inverse shape does not match T and output length");
  }

  core::CudaBuffer d_re(re.size()), d_im(im.size());
  core::CudaBuffer d_c(complex_storage);
  core::CudaBuffer d_win(static_cast<std::size_t>(n_fft_));
  d_re.copy_from_host(re.data(), re.size());
  d_im.copy_from_host(im.data(), im.size());
  d_win.copy_from_host(window_.data(), window_.size());
  k_merge<<<grid(spectrum_elements), 256>>>(d_re.data(), d_im.data(),
                                 reinterpret_cast<cufftComplex*>(d_c.data()), T, nf);

  core::CudaBuffer d_framed(static_cast<std::size_t>(frame_elements));
  cufftHandle plan{};
  ck_cufft(cufftPlan1d(&plan, n_fft_, CUFFT_C2R, T), "cufftPlan1d(C2R)");
  cufftResult exec_status = cufftExecC2R(
      plan, reinterpret_cast<cufftComplex*>(d_c.data()), d_framed.data());
  cufftResult destroy_status = cufftDestroy(plan);
  ck_cufft(exec_status, "cufftExecC2R");
  ck_cufft(destroy_status, "cufftDestroy");

  core::CudaBuffer d_y(static_cast<std::size_t>(Lp));
  core::CudaBuffer d_w(static_cast<std::size_t>(Lp));
  d_y.zero();
  d_w.zero();
  k_ola<<<grid(frame_elements), 256>>>(d_framed.data(), d_win.data(), d_y.data(),
                                   d_w.data(), T, n_fft_, hop_, 1.0f / n_fft_);
  k_norm<<<grid(Lp), 256>>>(d_y.data(), d_w.data(), Lp);

  std::vector<float> y(Lp);
  d_y.copy_to_host(y.data(), y.size());

  return std::vector<float>(y.begin() + pad_, y.begin() + pad_ + L_out);
}

std::vector<float> Stft::magnitude(const float* audio, int L, int& T) const {
  std::vector<float> re, im;
  forward(audio, L, re, im, T);
  std::vector<float> mag(re.size());
  for (size_t i = 0; i < re.size(); ++i) mag[i] = std::sqrt(re[i] * re[i] + im[i] * im[i]);
  return mag;
}

}  // namespace voxmutatio::separation
