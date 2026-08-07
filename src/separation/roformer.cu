// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// MelBand-RoFormer forward runner (spec 004 S6). Implemented + aligned stage by
// stage: band-split -> transformer blocks -> mask estimator -> apply/iSTFT.

#include "voxmutatio/separation/roformer.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "voxmutatio/core/cuda_buffer.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/io/safetensors.h"
#include "voxmutatio/separation/stft.h"

namespace voxmutatio::separation {
namespace {

using core::CudaBuffer;

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) \
  fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); } while (0)

inline int grid(int n) { return (n + 255) / 256; }

cublasHandle_t cublas() {
  static cublasHandle_t h = [] { cublasHandle_t x; cublasCreate(&x); return x; }();
  return h;
}

// Row-major C[M,N] = alpha*op(A)[M,K] @ op(B)[K,N] + beta*C.
void gemm_rm(bool tA, bool tB, int M, int N, int K, float alpha,
             const float* A, const float* B, float beta, float* C) {
  int lda = tA ? M : K, ldb = tB ? K : N;
  cublasSgemm(cublas(), tB ? CUBLAS_OP_T : CUBLAS_OP_N, tA ? CUBLAS_OP_T : CUBLAS_OP_N,
              N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
}

// Model constants (anvuew dereverb 8_256_6; verified via convert_roformer_weights.py).
constexpr int kNfft = 2048, kHop = 512, kNfreq = 1025, kMerged = 2050;
constexpr int kFiLen = 3958, kBandIn = 7916, kNbands = 60;
constexpr int kDim = 256, kHeads = 8, kDhead = 64, kInner = 512, kDepth = 6;
constexpr int kSr = 44100;
constexpr float kNormEps = 1e-12f;

// out[n,d] = x[n,d] / max(||x_n||_2, eps) * sqrt(D) * gamma[d]   (lucidrains RMSNorm)
__global__ void k_rmsnorm(const float* x, const float* gamma, float* out, int N, int D) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float* xn = x + static_cast<size_t>(n) * D;
  float ss = 0.0f;
  for (int d = 0; d < D; ++d) ss += xn[d] * xn[d];
  float inv = rsqrtf(fmaxf(ss, kNormEps * kNormEps)) * sqrtf((float)D);
  float* on = out + static_cast<size_t>(n) * D;
  for (int d = 0; d < D; ++d) on[d] = xn[d] * inv * gamma[d];
}

// RMSNorm on a strided band slice in[:, off:off+D] (row stride IN) -> contiguous out[T,D].
__global__ void k_rmsnorm_slice(const float* in, int IN, int off, const float* gamma,
                                float* out, int T, int D) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* r = in + static_cast<size_t>(t) * IN + off;
  float ss = 0.0f;
  for (int d = 0; d < D; ++d) ss += r[d] * r[d];
  float inv = rsqrtf(fmaxf(ss, kNormEps * kNormEps)) * sqrtf((float)D);
  float* o = out + static_cast<size_t>(t) * D;
  for (int d = 0; d < D; ++d) o[d] = r[d] * inv * gamma[d];
}

// Gather + complex fold: out[t, k*2+c] = merged[freq_indices[k], t, c],
// merged[m,t,c] with m=f*2+s -> re/im[s][f*T+t]. re/im layout [2, kNfreq, T].
__global__ void k_gather_fold(const float* re, const float* im, const int64_t* fi,
                              float* out, int T) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * kBandIn) return;
  int t = idx / kBandIn, rest = idx % kBandIn, k = rest >> 1, c = rest & 1;
  int m = (int)fi[k], s = m & 1, f = m >> 1;
  const float* src = (c == 0) ? re : im;
  out[idx] = src[(static_cast<size_t>(s) * kNfreq + f) * T + t];
}

// out[(t*NB + b)*Dim + d] = lin[t*Dim + d] + bias[d]
__global__ void k_bias_scatter_band(const float* lin, const float* bias, float* out,
                                    int b, int NB, int T, int Dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * Dim) return;
  int t = idx / Dim, d = idx % Dim;
  out[(static_cast<size_t>(t) * NB + b) * Dim + d] = lin[idx] + bias[d];
}

std::vector<int64_t> read_i64(const std::string& path, int expect) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.good()) return {};
  auto n = static_cast<size_t>(f.tellg()) / sizeof(int64_t);
  std::vector<int64_t> v(n);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(int64_t));
  if (expect && (int)n != expect) fprintf(stderr, "roformer: %s has %zu, want %d\n", path.c_str(), n, expect);
  return v;
}

}  // namespace

struct Attn {
  CudaBuffer norm_gamma, qkv_w, gates_w, gates_b, out_w, rope_freqs;
};
struct Ffn {
  CudaBuffer n0_gamma, w1, b1, w4, b4;
};
struct Block {  // one transformer (time or freq)
  Attn attn;
  Ffn ffn;
  CudaBuffer out_gamma;
};
struct MaskBand {
  CudaBuffer w0, b0, w2, b2, w4, b4;  // MLP 256->1024->1024->din*2
  int din = 0;
};

struct Roformer::Impl {
  bool ok = false;
  Stft stft{kNfft, kHop};
  // band split
  std::vector<CudaBuffer> bs_gamma, bs_w, bs_b;
  std::vector<int> dim_in, off_in;   // per band input dim + offset into kBandIn
  // transformer blocks [depth][0=time,1=freq]
  Block blocks[kDepth][2];
  // mask estimators (num_stems=1)
  std::vector<MaskBand> mask;
  // band maps
  CudaBuffer d_freq_indices;         // int64 as raw
  CudaBuffer d_num_bands_per_freq;   // float
  std::vector<int64_t> freq_indices, num_bands_per_freq;

  static bool up(const io::SafetensorsLoader& L, const std::string& n, CudaBuffer& b, int expect = 0) {
    const auto* t = L.get_tensor(n);
    if (!t) { fprintf(stderr, "roformer: missing %s\n", n.c_str()); return false; }
    int cnt = (int)(t->data_nbytes / sizeof(float));
    if (expect && cnt != expect) fprintf(stderr, "roformer: %s cnt %d want %d\n", n.c_str(), cnt, expect);
    b.copy_from_host(reinterpret_cast<const float*>(L.data(n)), cnt);
    return true;
  }

  explicit Impl(const std::string& dir, const std::string& model) {
    io::SafetensorsLoader L;
    if (!L.load(dir + "/" + model + ".safetensors")) {
      fprintf(stderr, "roformer: cannot load %s\n", (dir + "/" + model).c_str());
      return;
    }
    auto di = read_i64(dir + "/roformer_dim_inputs.i64", kNbands);
    freq_indices = read_i64(dir + "/roformer_freq_indices.i64", kFiLen);
    num_bands_per_freq = read_i64(dir + "/roformer_num_bands_per_freq.i64", kMerged);
    if (di.empty() || freq_indices.empty() || num_bands_per_freq.empty()) return;

    bool g = true;
    int off = 0;
    for (int b = 0; b < kNbands; ++b) {
      int din = (int)di[b];
      dim_in.push_back(din);
      off_in.push_back(off);
      off += din;
      bs_gamma.emplace_back(); bs_w.emplace_back(); bs_b.emplace_back();
      std::string p = "band_split.to_features." + std::to_string(b) + ".";
      g &= up(L, p + "0.gamma", bs_gamma[b], din);
      g &= up(L, p + "1.weight", bs_w[b], kDim * din);
      g &= up(L, p + "1.bias", bs_b[b], kDim);
    }
    for (int i = 0; i < kDepth; ++i) {
      for (int d = 0; d < 2; ++d) {
        std::string p = "layers." + std::to_string(i) + "." + std::to_string(d) + ".";
        Block& bl = blocks[i][d];
        g &= up(L, p + "layers.0.0.norm.gamma", bl.attn.norm_gamma, kDim);
        g &= up(L, p + "layers.0.0.rotary_embed.freqs", bl.attn.rope_freqs, kDhead / 2);
        g &= up(L, p + "layers.0.0.to_qkv.weight", bl.attn.qkv_w, 3 * kInner * kDim);
        g &= up(L, p + "layers.0.0.to_gates.weight", bl.attn.gates_w, kHeads * kDim);
        g &= up(L, p + "layers.0.0.to_gates.bias", bl.attn.gates_b, kHeads);
        g &= up(L, p + "layers.0.0.to_out.0.weight", bl.attn.out_w, kDim * kInner);
        g &= up(L, p + "layers.0.1.net.0.gamma", bl.ffn.n0_gamma, kDim);
        g &= up(L, p + "layers.0.1.net.1.weight", bl.ffn.w1, 1024 * kDim);
        g &= up(L, p + "layers.0.1.net.1.bias", bl.ffn.b1, 1024);
        g &= up(L, p + "layers.0.1.net.4.weight", bl.ffn.w4, kDim * 1024);
        g &= up(L, p + "layers.0.1.net.4.bias", bl.ffn.b4, kDim);
        g &= up(L, p + "norm.gamma", bl.out_gamma, kDim);
      }
    }
    mask.resize(kNbands);
    for (int b = 0; b < kNbands; ++b) {
      int din = dim_in[b];
      mask[b].din = din;
      std::string p = "mask_estimators.0.to_freqs." + std::to_string(b) + ".0.";
      g &= up(L, p + "0.weight", mask[b].w0, 1024 * kDim);
      g &= up(L, p + "0.bias", mask[b].b0, 1024);
      g &= up(L, p + "2.weight", mask[b].w2, 1024 * 1024);
      g &= up(L, p + "2.bias", mask[b].b2, 1024);
      g &= up(L, p + "4.weight", mask[b].w4, (din * 2) * 1024);
      g &= up(L, p + "4.bias", mask[b].b4, din * 2);
    }
    std::vector<float> nbpf(num_bands_per_freq.begin(), num_bands_per_freq.end());
    d_num_bands_per_freq.copy_from_host(nbpf.data(), nbpf.size());
    ok = g;
  }

  int frames(int L) const { return stft.num_frames(L); }

  // STFT both channels -> stacked re/im [2, kNfreq, T] on device.
  void stft_stereo(const std::vector<float>& stereo, int L, CudaBuffer& d_re,
                   CudaBuffer& d_im, int& T) const {
    std::vector<float> re[2], im[2];
    for (int c = 0; c < 2; ++c)
      stft.forward(stereo.data() + static_cast<size_t>(c) * L, L, re[c], im[c], T);
    std::vector<float> hre(static_cast<size_t>(2) * kNfreq * T), him(hre.size());
    for (int c = 0; c < 2; ++c)
      for (size_t i = 0; i < static_cast<size_t>(kNfreq) * T; ++i) {
        hre[static_cast<size_t>(c) * kNfreq * T + i] = re[c][i];
        him[static_cast<size_t>(c) * kNfreq * T + i] = im[c][i];
      }
    d_re.copy_from_host(hre.data(), hre.size());
    d_im.copy_from_host(him.data(), him.size());
  }

  // STFT + gather + fold -> band split input [T, kBandIn] on device.
  void band_split_in(const std::vector<float>& stereo, int L, CudaBuffer& bin, int& T) const {
    CudaBuffer d_re, d_im;
    stft_stereo(stereo, L, d_re, d_im, T);
    CudaBuffer d_fi;
    d_fi.copy_from_host(reinterpret_cast<const float*>(freq_indices.data()), freq_indices.size() * 2);
    bin.allocate(static_cast<size_t>(T) * kBandIn);
    k_gather_fold<<<grid(T * kBandIn), 256>>>(d_re.data(), d_im.data(),
        reinterpret_cast<const int64_t*>(d_fi.data()), bin.data(), T);
    CK(cudaDeviceSynchronize());
  }

  // STFT + gather + band split -> bandsplit_out [T, kNbands, kDim] on device.
  void band_split(const std::vector<float>& stereo, int L, CudaBuffer& out, int& T) const {
    CudaBuffer d_re, d_im;
    stft_stereo(stereo, L, d_re, d_im, T);
    CudaBuffer d_fi;
    d_fi.copy_from_host(reinterpret_cast<const float*>(freq_indices.data()),
                        freq_indices.size() * 2);  // int64 -> 2 float slots each
    CudaBuffer bin;
    bin.allocate(static_cast<size_t>(T) * kBandIn);
    k_gather_fold<<<grid(T * kBandIn), 256>>>(d_re.data(), d_im.data(),
        reinterpret_cast<const int64_t*>(d_fi.data()), bin.data(), T);

    out.allocate(static_cast<size_t>(T) * kNbands * kDim);
    CudaBuffer tmp, lin;
    for (int b = 0; b < kNbands; ++b) {
      int din = dim_in[b];
      tmp.allocate(static_cast<size_t>(T) * din);
      k_rmsnorm_slice<<<grid(T), 256>>>(bin.data(), kBandIn, off_in[b], bs_gamma[b].data(),
                                        tmp.data(), T, din);
      lin.allocate(static_cast<size_t>(T) * kDim);
      gemm_rm(false, true, T, kDim, din, 1.0f, tmp.data(), bs_w[b].data(), 0.0f, lin.data());
      k_bias_scatter_band<<<grid(T * kDim), 256>>>(lin.data(), bs_b[b].data(), out.data(), b, kNbands, T, kDim);
    }
    CK(cudaDeviceSynchronize());
  }
};

Roformer::Roformer(const std::string& dir, const std::string& model)
    : p_(std::make_unique<Impl>(dir, model)) {}
Roformer::~Roformer() = default;
bool Roformer::valid() const { return p_ && p_->ok; }
int Roformer::n_fft() const { return kNfft; }
int Roformer::hop() const { return kHop; }
int Roformer::sample_rate() const { return kSr; }

std::vector<float> Roformer::debug_bandsplit(const std::vector<float>& stereo, int L, int& T) const {
  CudaBuffer out;
  p_->band_split(stereo, L, out, T);
  std::vector<float> h(static_cast<size_t>(T) * kNbands * kDim);
  out.copy_to_host(h.data(), h.size());
  return h;
}

std::vector<float> Roformer::debug_bandsplit_in(const std::vector<float>& stereo, int L, int& T) const {
  CudaBuffer bin;
  p_->band_split_in(stereo, L, bin, T);
  std::vector<float> h(static_cast<size_t>(T) * kBandIn);
  bin.copy_to_host(h.data(), h.size());
  return h;
}

std::vector<int> Roformer::debug_offsets() const {
  std::vector<int> v(p_->off_in);
  v.insert(v.end(), p_->dim_in.begin(), p_->dim_in.end());
  return v;
}

std::vector<float> Roformer::debug_blocks(const std::vector<float>&, int, int& T) const { T = 0; return {}; }
std::vector<float> Roformer::debug_mask(const std::vector<float>&, int, int& T) const { T = 0; return {}; }
std::vector<float> Roformer::separate_stereo(const std::vector<float>&, int) const { return {}; }
std::vector<float> Roformer::separate_mono(const float*, int, int) const { return {}; }

}  // namespace voxmutatio::separation
