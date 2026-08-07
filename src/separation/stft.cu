// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/separation/stft.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace voxmutatio::separation {

namespace {

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); } } while (0)

inline int grid(int n) { return (n + 255) / 256; }

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

Stft::Stft(int n_fft, int hop, int win_length, bool center)
    : n_fft_(n_fft), hop_(hop), win_(win_length > 0 ? win_length : n_fft),
      pad_(center ? n_fft / 2 : 0), center_(center) {
  window_.resize(n_fft_, 0.0f);
  // Periodic Hann of length win_, centered in n_fft.
  int off = (n_fft_ - win_) / 2;
  for (int n = 0; n < win_; ++n)
    window_[off + n] = 0.5f - 0.5f * std::cos(2.0 * M_PI * n / win_);
}

int Stft::num_frames(int L) const {
  int Lp = L + 2 * pad_;
  return 1 + (Lp - n_fft_) / hop_;
}

void Stft::forward(const float* audio, int L, std::vector<float>& re,
                   std::vector<float>& im, int& T) const {
  const int nf = n_freq();
  T = num_frames(L);
  int Lp = L + 2 * pad_;

  std::vector<float> padded(Lp, 0.0f);
  for (int i = 0; i < L; ++i) padded[pad_ + i] = audio[i];
  for (int j = 0; j < pad_; ++j) {  // reflect
    padded[pad_ - 1 - j] = audio[std::min(j + 1, L - 1)];
    padded[pad_ + L + j] = audio[std::max(L - 2 - j, 0)];
  }

  float *d_pad, *d_win, *d_framed;
  CK(cudaMalloc(&d_pad, Lp * sizeof(float)));
  CK(cudaMalloc(&d_win, n_fft_ * sizeof(float)));
  CK(cudaMalloc(&d_framed, static_cast<size_t>(T) * n_fft_ * sizeof(float)));
  CK(cudaMemcpy(d_pad, padded.data(), Lp * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_win, window_.data(), n_fft_ * sizeof(float), cudaMemcpyHostToDevice));
  k_frame_window<<<grid(T * n_fft_), 256>>>(d_pad, d_win, d_framed, T, n_fft_, hop_);

  cufftComplex* d_c;
  CK(cudaMalloc(&d_c, static_cast<size_t>(T) * nf * sizeof(cufftComplex)));
  cufftHandle plan;
  cufftPlan1d(&plan, n_fft_, CUFFT_R2C, T);
  cufftExecR2C(plan, d_framed, d_c);
  cufftDestroy(plan);

  float *d_re, *d_im;
  CK(cudaMalloc(&d_re, static_cast<size_t>(nf) * T * sizeof(float)));
  CK(cudaMalloc(&d_im, static_cast<size_t>(nf) * T * sizeof(float)));
  k_split<<<grid(T * nf), 256>>>(d_c, d_re, d_im, T, nf);

  re.resize(static_cast<size_t>(nf) * T);
  im.resize(static_cast<size_t>(nf) * T);
  CK(cudaMemcpy(re.data(), d_re, re.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(im.data(), d_im, im.size() * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(d_pad); cudaFree(d_win); cudaFree(d_framed);
  cudaFree(d_c); cudaFree(d_re); cudaFree(d_im);
}

std::vector<float> Stft::inverse(const std::vector<float>& re, const std::vector<float>& im,
                                 int T, int L_out) const {
  const int nf = n_freq();
  int Lp = L_out + 2 * pad_;

  float *d_re, *d_im, *d_win;
  cufftComplex* d_c;
  CK(cudaMalloc(&d_re, re.size() * sizeof(float)));
  CK(cudaMalloc(&d_im, im.size() * sizeof(float)));
  CK(cudaMalloc(&d_c, static_cast<size_t>(T) * nf * sizeof(cufftComplex)));
  CK(cudaMalloc(&d_win, n_fft_ * sizeof(float)));
  CK(cudaMemcpy(d_re, re.data(), re.size() * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_im, im.data(), im.size() * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_win, window_.data(), n_fft_ * sizeof(float), cudaMemcpyHostToDevice));
  k_merge<<<grid(T * nf), 256>>>(d_re, d_im, d_c, T, nf);

  float* d_framed;
  CK(cudaMalloc(&d_framed, static_cast<size_t>(T) * n_fft_ * sizeof(float)));
  cufftHandle plan;
  cufftPlan1d(&plan, n_fft_, CUFFT_C2R, T);
  cufftExecC2R(plan, d_c, d_framed);
  cufftDestroy(plan);

  float *d_y, *d_w;
  CK(cudaMalloc(&d_y, Lp * sizeof(float)));
  CK(cudaMalloc(&d_w, Lp * sizeof(float)));
  CK(cudaMemset(d_y, 0, Lp * sizeof(float)));
  CK(cudaMemset(d_w, 0, Lp * sizeof(float)));
  k_ola<<<grid(T * n_fft_), 256>>>(d_framed, d_win, d_y, d_w, T, n_fft_, hop_, 1.0f / n_fft_);
  k_norm<<<grid(Lp), 256>>>(d_y, d_w, Lp);

  std::vector<float> y(Lp);
  CK(cudaMemcpy(y.data(), d_y, Lp * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(d_re); cudaFree(d_im); cudaFree(d_c); cudaFree(d_win);
  cudaFree(d_framed); cudaFree(d_y); cudaFree(d_w);

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
