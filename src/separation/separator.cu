// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Open-Unmix umxhq vocals — pure C++/CUDA forward runner (spec 004 S2).

#include "voxmutatio/separation/separator.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "voxmutatio/core/cuda_buffer.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/io/safetensors.h"
#include "voxmutatio/separation/stft.h"
#include "validation.h"

namespace voxmutatio::separation {

namespace {

using core::CudaBuffer;

[[noreturn]] void throw_cuda(const char* operation, cudaError_t error) {
  throw std::runtime_error(std::string("Separator CUDA failure in ") + operation +
                           ": " + cudaGetErrorString(error));
}
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) \
  throw_cuda(#call, e_); } while (0)

void ck_blas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("Separator cuBLAS failure in ") + operation +
                             " (status " + std::to_string(static_cast<int>(status)) + ")");
  }
}

inline int grid(int n) { return n > 0 ? 1 + (n - 1) / 256 : 0; }

cublasHandle_t cublas() {
  static cublasHandle_t h = [] {
    cublasHandle_t x = nullptr;
    ck_blas(cublasCreate(&x), "cublasCreate");
    return x;
  }();
  return h;
}

// Row-major C[M,N] = alpha*op(A)[M,K] @ op(B)[K,N] + beta*C.
void gemm_rm(bool transA, bool transB, int M, int N, int K, float alpha,
             const float* A, const float* B, float beta, float* C) {
  int lda = transA ? M : K;
  int ldb = transB ? K : N;
  ck_blas(cublasSgemm(cublas(), transB ? CUBLAS_OP_T : CUBLAS_OP_N,
              transA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha,
              B, ldb, A, lda, &beta, C, N), "cublasSgemm");
}

// Build fc1 input: fc_in[f, c*nb_bins + b] = (mix[c,b,f] + mean[b]) * scale[b].
// mix layout [nb_channels, 2049, nb_frames] (channel-major); crop to nb_bins.
__global__ void k_standardize(const float* mix, const float* mean, const float* scale,
                              float* fc_in, int nb_bins, int out_bins, int nb_frames) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int cols = 2 * nb_bins;
  if (idx >= nb_frames * cols) return;
  int f = idx / cols, r = idx % cols;
  int c = r / nb_bins, b = r % nb_bins;
  float v = mix[(c * out_bins + b) * nb_frames + f];
  fc_in[idx] = (v + mean[b]) * scale[b];
}

// BatchNorm (eval) over [N, C]: y = (x-rm)/sqrt(rv+eps)*w + b.
__global__ void k_bn(float* x, const float* w, const float* bias, const float* rm,
                     const float* rv, float eps, int N, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * C) return;
  int c = idx % C;
  x[idx] = (x[idx] - rm[c]) * rsqrtf(rv[c] + eps) * w[c] + bias[c];
}

__global__ void k_tanh(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = tanhf(x[i]);
}
__global__ void k_relu(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = fmaxf(x[i], 0.0f);
}
// x[n, c] += b[c]
__global__ void k_add_rowvec(float* x, const float* b, int N, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * C) return;
  x[idx] += b[idx % C];
}

// One LSTM timestep for hidden units [0,H): gates = xproj_t + hproj (order i,f,g,o).
__global__ void k_lstm_cell(const float* xproj_t, const float* hproj, const float* c_prev,
                            float* h_out, float* c_out, int H) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) return;
  float i = 1.0f / (1.0f + expf(-(xproj_t[j] + hproj[j])));
  float f = 1.0f / (1.0f + expf(-(xproj_t[H + j] + hproj[H + j])));
  float g = tanhf(xproj_t[2 * H + j] + hproj[2 * H + j]);
  float o = 1.0f / (1.0f + expf(-(xproj_t[3 * H + j] + hproj[3 * H + j])));
  float c = f * c_prev[j] + i * g;
  c_out[j] = c;
  h_out[j] = o * tanhf(c);
}

// out[n, 0:H) = a[n], out[n, H:2H) = b[n]
__global__ void k_concat(const float* a, const float* b, float* out, int N, int H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * H) return;
  int n = idx / H, j = idx % H;
  out[n * 2 * H + j] = a[idx];
  out[n * 2 * H + H + j] = b[idx];
}

// Final mask + reconstruct: est[c,b,f] = relu(net[f, c*obins+b]*oscale[b]+omean[b]) * mix[c,b,f].
__global__ void k_mask(const float* net, const float* oscale, const float* omean,
                       const float* mix, float* est, int obins, int nb_frames) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 2 * obins * nb_frames) return;
  int f = idx % nb_frames, rest = idx / nb_frames;  // est layout [c,b,f]
  int b = rest % obins, c = rest / obins;
  float v = net[f * 2 * obins + c * obins + b] * oscale[b] + omean[b];
  v = fmaxf(v, 0.0f);
  est[idx] = v * mix[idx];
}

}  // namespace

// Model dimensions (umxhq vocals; verified via tools/inspect_umx.py).
constexpr int kNbBins = 1487;      // input bandwidth crop
constexpr int kObins = 2049;       // n_fft/2 + 1
constexpr int kHidden = 512;       // FC hidden
constexpr int kH = 256;            // LSTM hidden per direction
constexpr int kLayers = 3;
constexpr int kNfft = 4096;
constexpr int kHop = 1024;
constexpr int kSr = 44100;
constexpr float kBnEps = 1e-5f;

void require_supported_frames(int frames) {
  if (frames <= 0) throw std::invalid_argument("Separator frame count must be positive");
  // This is the widest frame-indexed tensor and kernel launch in UMX. Every
  // other frame product (2974, 1024, 512, or 256 values/frame) is smaller.
  detail::checked_int_product({frames, 2, kObins}, "Separator frame tensor");
}

struct LstmDir {
  CudaBuffer ih_w;   // [1024, 512]
  CudaBuffer hh_w;   // [1024, 256]
  CudaBuffer bsum;   // [1024] = bias_ih + bias_hh
};

struct Separator::Impl {
  bool ok = false;
  Stft stft{kNfft, kHop};
  CudaBuffer in_mean, in_scale, out_scale, out_mean;
  CudaBuffer fc1_w, bn1_w, bn1_b, bn1_rm, bn1_rv;
  LstmDir lstm[kLayers][2];  // [layer][0=fwd,1=rev]
  CudaBuffer fc2_w, bn2_w, bn2_b, bn2_rm, bn2_rv;
  CudaBuffer fc3_w, bn3_w, bn3_b, bn3_rm, bn3_rv;

  static bool up(const io::SafetensorsLoader& L, const std::string& name,
                 CudaBuffer& buf, std::initializer_list<int64_t> shape,
                 detail::TensorValueConstraint constraint =
                     detail::TensorValueConstraint::kFinite) {
    return detail::upload_checked_f32(L, name, shape, buf, "separator", constraint);
  }

  explicit Impl(const std::string& path) {
    io::SafetensorsLoader L;
    if (!L.load(path)) { fprintf(stderr, "separator: cannot load %s\n", path.c_str()); return; }
    bool g = true;
    g &= up(L, "input_mean", in_mean, {kNbBins});
    g &= up(L, "input_scale", in_scale, {kNbBins});
    g &= up(L, "output_scale", out_scale, {kObins});
    g &= up(L, "output_mean", out_mean, {kObins});
    g &= up(L, "fc1.weight", fc1_w, {kHidden, 2 * kNbBins});
    g &= up(L, "bn1.weight", bn1_w, {kHidden});
    g &= up(L, "bn1.bias", bn1_b, {kHidden});
    g &= up(L, "bn1.running_mean", bn1_rm, {kHidden});
    g &= up(L, "bn1.running_var", bn1_rv, {kHidden},
            detail::TensorValueConstraint::kFiniteNonnegative);
    for (int l = 0; l < kLayers; ++l) {
      for (int d = 0; d < 2; ++d) {
        std::string sfx = "_l" + std::to_string(l) + (d ? "_reverse" : "");
        g &= up(L, "lstm.weight_ih" + sfx, lstm[l][d].ih_w, {4 * kH, kHidden});
        g &= up(L, "lstm.weight_hh" + sfx, lstm[l][d].hh_w, {4 * kH, kH});
        // bsum = bias_ih + bias_hh
        const float* bi = nullptr;
        const float* bh = nullptr;
        std::size_t bi_count = 0;
        std::size_t bh_count = 0;
        if (!detail::checked_f32_tensor(L, "lstm.bias_ih" + sfx, {4 * kH},
                                        "separator", bi, bi_count) ||
            !detail::checked_f32_tensor(L, "lstm.bias_hh" + sfx, {4 * kH},
                                        "separator", bh, bh_count)) {
          g = false;
          continue;
        }
        std::vector<float> bs(4 * kH);
        for (int i = 0; i < 4 * kH; ++i) {
          bs[i] = bi[i] + bh[i];
          if (!std::isfinite(bs[i])) g = false;
        }
        if (!g) continue;
        lstm[l][d].bsum.copy_from_host(bs.data(), 4 * kH);
      }
    }
    g &= up(L, "fc2.weight", fc2_w, {kHidden, 2 * kH + kHidden});
    g &= up(L, "bn2.weight", bn2_w, {kHidden});
    g &= up(L, "bn2.bias", bn2_b, {kHidden});
    g &= up(L, "bn2.running_mean", bn2_rm, {kHidden});
    g &= up(L, "bn2.running_var", bn2_rv, {kHidden},
            detail::TensorValueConstraint::kFiniteNonnegative);
    g &= up(L, "fc3.weight", fc3_w, {2 * kObins, kHidden});
    g &= up(L, "bn3.weight", bn3_w, {2 * kObins});
    g &= up(L, "bn3.bias", bn3_b, {2 * kObins});
    g &= up(L, "bn3.running_mean", bn3_rm, {2 * kObins});
    g &= up(L, "bn3.running_var", bn3_rv, {2 * kObins},
            detail::TensorValueConstraint::kFiniteNonnegative);
    ok = g;
  }

  // mix_mag [2, kObins, F] (channel-major) -> est vocal magnitude, same layout.
  std::vector<float> run_model(const std::vector<float>& mix_mag, int F) const {
    require_supported_frames(F);
    const int standardized_elements =
        detail::checked_int_product({F, 2, kNbBins}, "Separator standardized input");
    const int hidden_elements =
        detail::checked_int_product({F, kHidden}, "Separator hidden tensor");
    const int gate_elements =
        detail::checked_int_product({F, 4, kH}, "Separator LSTM gates");
    const int recurrent_elements =
        detail::checked_int_product({F, kH}, "Separator recurrent tensor");
    const int output_elements =
        detail::checked_int_product({F, 2, kObins}, "Separator output tensor");
    CudaBuffer d_mix; d_mix.copy_from_host(mix_mag.data(), mix_mag.size());
    CudaBuffer fc_in; fc_in.allocate(static_cast<size_t>(F) * 2 * kNbBins);
    k_standardize<<<grid(standardized_elements), 256>>>(d_mix.data(), in_mean.data(),
        in_scale.data(), fc_in.data(), kNbBins, kObins, F);

    CudaBuffer cur; cur.allocate(static_cast<size_t>(F) * kHidden);
    gemm_rm(false, true, F, kHidden, 2 * kNbBins, 1.0f, fc_in.data(), fc1_w.data(), 0.0f, cur.data());
    k_bn<<<grid(hidden_elements), 256>>>(cur.data(), bn1_w.data(), bn1_b.data(),
        bn1_rm.data(), bn1_rv.data(), kBnEps, F, kHidden);
    k_tanh<<<grid(hidden_elements), 256>>>(cur.data(), hidden_elements);

    CudaBuffer skip; skip.copy_from_device(cur.data(), static_cast<size_t>(F) * kHidden);

    // 3-layer bidirectional LSTM. Input size is kHidden (512) for all layers.
    CudaBuffer lstm_in; lstm_in.copy_from_device(cur.data(), static_cast<size_t>(F) * kHidden);
    CudaBuffer Xf, Xr, hseq_f, hseq_r, hproj, c_buf, d_zero, lstm_out;
    Xf.allocate(static_cast<size_t>(F) * 4 * kH);
    Xr.allocate(static_cast<size_t>(F) * 4 * kH);
    hseq_f.allocate(static_cast<size_t>(F) * kH);
    hseq_r.allocate(static_cast<size_t>(F) * kH);
    hproj.allocate(4 * kH);
    c_buf.allocate(kH);
    d_zero.allocate(kH); d_zero.zero();
    lstm_out.allocate(static_cast<size_t>(F) * kHidden);

    for (int l = 0; l < kLayers; ++l) {
      const LstmDir& fwd = lstm[l][0];
      const LstmDir& rev = lstm[l][1];
      gemm_rm(false, true, F, 4 * kH, kHidden, 1.0f, lstm_in.data(), fwd.ih_w.data(), 0.0f, Xf.data());
      k_add_rowvec<<<grid(gate_elements), 256>>>(Xf.data(), fwd.bsum.data(), F,
                                                 4 * kH);
      gemm_rm(false, true, F, 4 * kH, kHidden, 1.0f, lstm_in.data(), rev.ih_w.data(), 0.0f, Xr.data());
      k_add_rowvec<<<grid(gate_elements), 256>>>(Xr.data(), rev.bsum.data(), F,
                                                 4 * kH);

      c_buf.zero();
      for (int t = 0; t < F; ++t) {
        const float* hprev = t ? hseq_f.data() + (t - 1) * kH : d_zero.data();
        gemm_rm(false, true, 1, 4 * kH, kH, 1.0f, hprev, fwd.hh_w.data(), 0.0f, hproj.data());
        k_lstm_cell<<<grid(kH), 256>>>(Xf.data() + static_cast<size_t>(t) * 4 * kH,
            hproj.data(), c_buf.data(), hseq_f.data() + static_cast<size_t>(t) * kH, c_buf.data(), kH);
      }
      c_buf.zero();
      for (int t = F - 1; t >= 0; --t) {
        const float* hprev = (t < F - 1) ? hseq_r.data() + (t + 1) * kH : d_zero.data();
        gemm_rm(false, true, 1, 4 * kH, kH, 1.0f, hprev, rev.hh_w.data(), 0.0f, hproj.data());
        k_lstm_cell<<<grid(kH), 256>>>(Xr.data() + static_cast<size_t>(t) * 4 * kH,
            hproj.data(), c_buf.data(), hseq_r.data() + static_cast<size_t>(t) * kH, c_buf.data(), kH);
      }
      k_concat<<<grid(recurrent_elements), 256>>>(hseq_f.data(), hseq_r.data(),
                                                  lstm_out.data(), F, kH);
      lstm_in.copy_from_device(lstm_out.data(), static_cast<size_t>(F) * kHidden);
    }

    // skip connection + fc2/bn2/relu + fc3/bn3.
    CudaBuffer cat; cat.allocate(static_cast<size_t>(F) * 2 * kHidden);
    k_concat<<<grid(hidden_elements), 256>>>(skip.data(), lstm_out.data(), cat.data(),
                                             F, kHidden);
    CudaBuffer h2; h2.allocate(static_cast<size_t>(F) * kHidden);
    gemm_rm(false, true, F, kHidden, 2 * kHidden, 1.0f, cat.data(), fc2_w.data(), 0.0f, h2.data());
    k_bn<<<grid(hidden_elements), 256>>>(h2.data(), bn2_w.data(), bn2_b.data(),
        bn2_rm.data(), bn2_rv.data(), kBnEps, F, kHidden);
    k_relu<<<grid(hidden_elements), 256>>>(h2.data(), hidden_elements);
    CudaBuffer h3; h3.allocate(static_cast<size_t>(F) * 2 * kObins);
    gemm_rm(false, true, F, 2 * kObins, kHidden, 1.0f, h2.data(), fc3_w.data(), 0.0f, h3.data());
    k_bn<<<grid(output_elements), 256>>>(h3.data(), bn3_w.data(), bn3_b.data(),
        bn3_rm.data(), bn3_rv.data(), kBnEps, F, 2 * kObins);

    CudaBuffer est; est.allocate(static_cast<size_t>(2) * kObins * F);
    k_mask<<<grid(output_elements), 256>>>(h3.data(), out_scale.data(), out_mean.data(),
        d_mix.data(), est.data(), kObins, F);
    CK(cudaDeviceSynchronize());

    std::vector<float> out(static_cast<size_t>(2) * kObins * F);
    est.copy_to_host(out.data(), out.size());
    detail::require_finite_output(out, "Separator model output");
    return out;
  }
};

Separator::Separator(const std::string& weights_path)
    : p_(std::make_unique<Impl>(weights_path)) {}
Separator::~Separator() = default;

bool Separator::valid() const { return p_ && p_->ok; }
int Separator::n_fft() const { return kNfft; }
int Separator::hop() const { return kHop; }
int Separator::sample_rate() const { return kSr; }
int Separator::nb_output_bins() const { return kObins; }

std::vector<float> Separator::run_model(const std::vector<float>& mix_mag, int nb_frames) const {
  if (!valid()) throw std::runtime_error("Separator is not initialized");
  require_supported_frames(nb_frames);
  const std::size_t expected =
      detail::checked_size_product({2, kObins, nb_frames}, "Separator magnitude input");
  if (mix_mag.size() != expected) {
    throw std::invalid_argument("Separator magnitude input must have shape [2, bins, frames]");
  }
  detail::require_finite_input(mix_mag, "Separator magnitude input");
  return p_->run_model(mix_mag, nb_frames);
}

std::vector<float> Separator::separate_stereo(const std::vector<float>& stereo, int Lch) const {
  if (!valid()) throw std::runtime_error("Separator is not initialized");
  if (Lch <= 0 || stereo.size() != static_cast<std::size_t>(2) * Lch) {
    throw std::invalid_argument("Separator stereo input must have shape [2, L]");
  }
  require_supported_frames(p_->stft.num_frames(Lch));
  detail::require_finite_input(stereo, "Separator stereo input");
  const int nf = kObins;
  int T = 0;
  std::vector<float> re[2], im[2];
  for (int c = 0; c < 2; ++c)
    p_->stft.forward(stereo.data() + static_cast<size_t>(c) * Lch, Lch, re[c], im[c], T);

  // mix magnitude [2, nf, T].
  std::vector<float> mix(static_cast<size_t>(2) * nf * T);
  for (int c = 0; c < 2; ++c)
    for (size_t i = 0; i < static_cast<size_t>(nf) * T; ++i)
      mix[static_cast<size_t>(c) * nf * T + i] =
          std::sqrt(re[c][i] * re[c][i] + im[c][i] * im[c][i]);
  detail::require_finite_output(mix, "Separator STFT magnitude");

  auto est = p_->run_model(mix, T);  // [2, nf, T]

  // Reconstruct each channel with mixture phase.
  std::vector<float> out(static_cast<size_t>(2) * Lch, 0.0f);
  for (int c = 0; c < 2; ++c) {
    std::vector<float> vr(static_cast<size_t>(nf) * T), vi(static_cast<size_t>(nf) * T);
    for (size_t i = 0; i < static_cast<size_t>(nf) * T; ++i) {
      float m = std::sqrt(re[c][i] * re[c][i] + im[c][i] * im[c][i]) + 1e-10f;
      float e = est[static_cast<size_t>(c) * nf * T + i];
      vr[i] = e * re[c][i] / m;
      vi[i] = e * im[c][i] / m;
    }
    auto ch = p_->stft.inverse(vr, vi, T, Lch);
    for (int i = 0; i < Lch; ++i) out[static_cast<size_t>(c) * Lch + i] = ch[i];
  }
  detail::require_finite_output(out, "Separator waveform output");
  return out;
}

std::vector<float> Separator::separate_mono(const float* audio, int L, int sr) const {
  if (!valid()) throw std::runtime_error("Separator is not initialized");
  if (!audio || L <= 0 || sr <= 0) {
    throw std::invalid_argument("Separator mono input requires audio, L > 0, and sr > 0");
  }
  const int model_length =
      sr == kSr ? L : detail::checked_resampled_length(L, sr, kSr, "Separator input");
  require_supported_frames(p_->stft.num_frames(model_length));
  detail::require_finite_input(audio, static_cast<std::size_t>(L),
                               "Separator mono input");
  auto r = (sr == kSr) ? std::vector<float>(audio, audio + L)
                       : io::resample_linear(audio, L, sr, kSr);
  if (r.empty()) throw std::runtime_error("Separator resampling failed");
  detail::require_finite_output(r, "Separator resampled input");
  int L44 = static_cast<int>(r.size());
  std::vector<float> stereo(static_cast<size_t>(2) * L44);
  for (int i = 0; i < L44; ++i) { stereo[i] = r[i]; stereo[L44 + i] = r[i]; }
  auto voc = separate_stereo(stereo, L44);
  std::vector<float> mono(L44);
  for (int i = 0; i < L44; ++i) mono[i] = 0.5f * (voc[i] + voc[L44 + i]);
  if (sr != kSr) mono = io::resample_linear(mono.data(), L44, kSr, sr);
  if (mono.empty()) throw std::runtime_error("Separator output resampling failed");
  detail::require_finite_output(mono, "Separator mono output");
  return mono;
}

}  // namespace voxmutatio::separation
