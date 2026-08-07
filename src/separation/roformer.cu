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
#include <utility>
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

// x[n,d] += b[d]
__global__ void k_add_rowbias(float* x, const float* b, int N, int D) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N * D) x[i] += b[i % D];
}
// a[i] += b[i]
__global__ void k_add(float* a, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}
__global__ void k_gelu(float* x, int n) {  // exact GELU (torch default erf form)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = x[i]; x[i] = 0.5f * v * (1.0f + erff(v * 0.70710678f)); }
}
__global__ void k_tanh(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = tanhf(x[i]);
}

// qkv[B*S, 3*H*DH] -> q,k,v [B,H,S,DH]  (rearrange 'bs (qkv h d) -> qkv b h s d')
__global__ void k_split_heads(const float* qkv, float* q, float* k, float* v,
                              int B, int S, int H, int DH) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int inner = H * DH, tot = B * S * inner;
  if (idx >= tot) return;
  int d = idx % DH, h = (idx / DH) % H, s = (idx / inner) % S, b = idx / (S * inner);
  size_t src = (static_cast<size_t>(b) * S + s) * (3 * inner) + h * DH + d;
  size_t dst = ((static_cast<size_t>(b) * H + h) * S + s) * DH + d;
  q[dst] = qkv[src];
  k[dst] = qkv[src + inner];
  v[dst] = qkv[src + 2 * inner];
}
// RoPE (lucidrains interleaved pairs) on x[B,H,S,DH] in place; freqs[DH/2].
__global__ void k_rope(float* x, const float* freqs, int B, int H, int S, int DH) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int half = DH / 2, tot = B * H * S * half;
  if (idx >= tot) return;
  int p = idx % half, s = (idx / half) % S;
  size_t base = (static_cast<size_t>(idx / half)) * DH + 2 * p;
  float ang = s * freqs[p], c = cosf(ang), sn = sinf(ang);
  float x0 = x[base], x1 = x[base + 1];
  x[base] = x0 * c - x1 * sn;
  x[base + 1] = x1 * c + x0 * sn;
}
// Attention (online softmax) q,k,v [B,H,S,DH] -> out [B,H,S,DH]; scale = 1/sqrt(DH).
__global__ void k_attention(const float* q, const float* k, const float* v, float* out,
                            int B, int H, int S, int DH, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // (b,h,i)
  if (idx >= B * H * S) return;
  const float* qi = q + static_cast<size_t>(idx) * DH;
  size_t bh = idx / S;  // (b*H+h)
  const float* kb = k + bh * S * DH;
  const float* vb = v + bh * S * DH;
  float acc[64];
  for (int d = 0; d < DH; ++d) acc[d] = 0.0f;
  float m = -1e30f, l = 0.0f;
  for (int j = 0; j < S; ++j) {
    const float* kj = kb + static_cast<size_t>(j) * DH;
    float s = 0.0f;
    for (int d = 0; d < DH; ++d) s += qi[d] * kj[d];
    s *= scale;
    float mn = fmaxf(m, s), corr = __expf(m - mn), p = __expf(s - mn);
    l = l * corr + p;
    const float* vj = vb + static_cast<size_t>(j) * DH;
    for (int d = 0; d < DH; ++d) acc[d] = acc[d] * corr + p * vj[d];
    m = mn;
  }
  float* o = out + static_cast<size_t>(idx) * DH;
  float inv = 1.0f / l;
  for (int d = 0; d < DH; ++d) o[d] = acc[d] * inv;
}
// merge heads [B,H,S,DH] -> [B*S, H*DH]
__global__ void k_merge_heads(const float* in, float* out, int B, int S, int H, int DH) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int inner = H * DH;
  if (idx >= B * S * inner) return;
  int d = idx % DH, h = (idx / DH) % H, s = (idx / inner) % S, b = idx / (S * inner);
  out[(static_cast<size_t>(b) * S + s) * inner + h * DH + d] =
      in[((static_cast<size_t>(b) * H + h) * S + s) * DH + d];
}
// mo[bs, h*DH+d] *= sigmoid(gates[bs, h])
__global__ void k_apply_gates(float* mo, const float* gates, int BS, int H, int DH) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * H * DH) return;
  int h = (idx / DH) % H, bs = idx / (H * DH);
  mo[idx] *= 1.0f / (1.0f + __expf(-gates[bs * H + h]));
}
// transpose [A,B,D] <-> [B,A,D] (swap first two dims, keep feature D)
__global__ void k_transpose12(const float* in, float* out, int A, int Bn, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= A * Bn * D) return;
  int d = idx % D, b = (idx / D) % Bn, a = idx / (Bn * D);
  out[(static_cast<size_t>(b) * A + a) * D + d] = in[idx];
}

// extract band b: out[t, d] = blk[(t*NB + b)*D + d]
__global__ void k_extract_band(const float* blk, float* out, int b, int NB, int T, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * D) return;
  int t = idx / D, d = idx % D;
  out[idx] = blk[(static_cast<size_t>(t) * NB + b) * D + d];
}
// GLU + scatter: mask[t, off+d] = m2[t,d] * sigmoid(m2[t, din+d])  (band chunk in kBandIn)
__global__ void k_glu_scatter(const float* m2, float* mask, int off, int din, int T) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * din) return;
  int t = idx / din, d = idx % din;
  float a = m2[static_cast<size_t>(t) * (2 * din) + d];
  float g = m2[static_cast<size_t>(t) * (2 * din) + din + d];
  mask[static_cast<size_t>(t) * kBandIn + off + d] = a * (1.0f / (1.0f + __expf(-g)));
}
// Scatter-add complex masks into merged freq (bands overlap): mask[t, k, c] with
// feature k*2+c maps to merged freq fi[k]. summed[m, t] (re/im) accumulated.
__global__ void k_scatter_mask(const float* mask, const int64_t* fi, float* sre,
                               float* sim, int T) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * kFiLen) return;
  int t = idx / kFiLen, k = idx % kFiLen;
  int m = (int)fi[k];
  float re = mask[static_cast<size_t>(t) * kBandIn + k * 2];
  float im = mask[static_cast<size_t>(t) * kBandIn + k * 2 + 1];
  atomicAdd(&sre[static_cast<size_t>(m) * T + t], re);
  atomicAdd(&sim[static_cast<size_t>(m) * T + t], im);
}
// Average mask (/num_bands_per_freq), complex-multiply with stft, zero DC, split to
// channel re/im. merged m=f*2+s, nbpf indexed by f. stft re/im [2,kNfreq,T].
__global__ void k_apply_mask(const float* re, const float* im, const float* sre,
                             const float* sim, const float* nbpf, float* ore,
                             float* oim, int T) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= kMerged * T) return;
  int t = idx % T, m = idx / T;
  int s = m & 1, f = m >> 1;
  float denom = fmaxf(nbpf[f], 1e-8f);
  float mr = sre[idx] / denom, mi = sim[idx] / denom;
  if (f == 0) { mr = 0.0f; mi = 0.0f; }  // zero DC
  size_t src = (static_cast<size_t>(s) * kNfreq + f) * T + t;
  float xr = re[src], xi = im[src];
  ore[src] = xr * mr - xi * mi;
  oim[src] = xr * mi + xi * mr;
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

  // One transformer (depth 1): attn residual + ffn residual + output RMSNorm.
  // x is [B*S, kDim] on device (B sequences of length S); modified in place.
  void transformer(CudaBuffer& x, int B, int S, const Block& bl) const {
    const int BS = B * S, D = kDim, H = kHeads, DH = kDhead, IN = kInner;
    CudaBuffer xn; xn.allocate(static_cast<size_t>(BS) * D);
    k_rmsnorm<<<grid(BS), 256>>>(x.data(), bl.attn.norm_gamma.data(), xn.data(), BS, D);
    CudaBuffer qkv; qkv.allocate(static_cast<size_t>(BS) * 3 * IN);
    gemm_rm(false, true, BS, 3 * IN, D, 1.0f, xn.data(), bl.attn.qkv_w.data(), 0.0f, qkv.data());
    CudaBuffer q, k, v;
    q.allocate(static_cast<size_t>(BS) * IN); k.allocate(q.size()); v.allocate(q.size());
    k_split_heads<<<grid(BS * IN), 256>>>(qkv.data(), q.data(), k.data(), v.data(), B, S, H, DH);
    int rope_n = B * H * S * (DH / 2);
    k_rope<<<grid(rope_n), 256>>>(q.data(), bl.attn.rope_freqs.data(), B, H, S, DH);
    k_rope<<<grid(rope_n), 256>>>(k.data(), bl.attn.rope_freqs.data(), B, H, S, DH);
    CudaBuffer ao; ao.allocate(q.size());
    k_attention<<<grid(B * H * S), 256>>>(q.data(), k.data(), v.data(), ao.data(),
        B, H, S, DH, 1.0f / sqrtf((float)DH));
    CudaBuffer mo; mo.allocate(q.size());
    k_merge_heads<<<grid(BS * IN), 256>>>(ao.data(), mo.data(), B, S, H, DH);
    CudaBuffer gates; gates.allocate(static_cast<size_t>(BS) * H);
    gemm_rm(false, true, BS, H, D, 1.0f, xn.data(), bl.attn.gates_w.data(), 0.0f, gates.data());
    k_add_rowbias<<<grid(BS * H), 256>>>(gates.data(), bl.attn.gates_b.data(), BS, H);
    k_apply_gates<<<grid(BS * IN), 256>>>(mo.data(), gates.data(), BS, H, DH);
    CudaBuffer att; att.allocate(static_cast<size_t>(BS) * D);
    gemm_rm(false, true, BS, D, IN, 1.0f, mo.data(), bl.attn.out_w.data(), 0.0f, att.data());
    k_add<<<grid(BS * D), 256>>>(x.data(), att.data(), BS * D);

    CudaBuffer fn; fn.allocate(static_cast<size_t>(BS) * D);
    k_rmsnorm<<<grid(BS), 256>>>(x.data(), bl.ffn.n0_gamma.data(), fn.data(), BS, D);
    CudaBuffer h1; h1.allocate(static_cast<size_t>(BS) * 1024);
    gemm_rm(false, true, BS, 1024, D, 1.0f, fn.data(), bl.ffn.w1.data(), 0.0f, h1.data());
    k_add_rowbias<<<grid(BS * 1024), 256>>>(h1.data(), bl.ffn.b1.data(), BS, 1024);
    k_gelu<<<grid(BS * 1024), 256>>>(h1.data(), BS * 1024);
    CudaBuffer h2; h2.allocate(static_cast<size_t>(BS) * D);
    gemm_rm(false, true, BS, D, 1024, 1.0f, h1.data(), bl.ffn.w4.data(), 0.0f, h2.data());
    k_add_rowbias<<<grid(BS * D), 256>>>(h2.data(), bl.ffn.b4.data(), BS, D);
    k_add<<<grid(BS * D), 256>>>(x.data(), h2.data(), BS * D);

    CudaBuffer xo; xo.allocate(static_cast<size_t>(BS) * D);
    k_rmsnorm<<<grid(BS), 256>>>(x.data(), bl.out_gamma.data(), xo.data(), BS, D);
    x = std::move(xo);
  }

  // 6 blocks: each = time transformer (over T per band) + freq transformer (over bands per t).
  // x is [T, kNbands, kDim] in place.
  void blocks_forward(CudaBuffer& x, int T) const {
    const size_t n = static_cast<size_t>(T) * kNbands * kDim;
    for (int i = 0; i < kDepth; ++i) {
      CudaBuffer xt; xt.allocate(n);
      k_transpose12<<<grid(n), 256>>>(x.data(), xt.data(), T, kNbands, kDim);  // [T,60,D]->[60,T,D]
      transformer(xt, kNbands, T, blocks[i][0]);                              // time
      k_transpose12<<<grid(n), 256>>>(xt.data(), x.data(), kNbands, T, kDim);  // [60,T,D]->[T,60,D]
      transformer(x, T, kNbands, blocks[i][1]);                               // freq
    }
    CK(cudaDeviceSynchronize());
  }

  // Mask estimator: per band MLP(256->1024->1024->din*2, Tanh) + GLU -> mask [T, kBandIn].
  void mask_estimator(const CudaBuffer& blk, CudaBuffer& mask_out, int T) const {
    mask_out.allocate(static_cast<size_t>(T) * kBandIn);
    CudaBuffer bandin, h1, h2, m2;
    for (int b = 0; b < kNbands; ++b) {
      int din = dim_in[b];
      bandin.allocate(static_cast<size_t>(T) * kDim);
      k_extract_band<<<grid(T * kDim), 256>>>(blk.data(), bandin.data(), b, kNbands, T, kDim);
      h1.allocate(static_cast<size_t>(T) * 1024);
      gemm_rm(false, true, T, 1024, kDim, 1.0f, bandin.data(), mask[b].w0.data(), 0.0f, h1.data());
      k_add_rowbias<<<grid(T * 1024), 256>>>(h1.data(), mask[b].b0.data(), T, 1024);
      k_tanh<<<grid(T * 1024), 256>>>(h1.data(), T * 1024);
      h2.allocate(static_cast<size_t>(T) * 1024);
      gemm_rm(false, true, T, 1024, 1024, 1.0f, h1.data(), mask[b].w2.data(), 0.0f, h2.data());
      k_add_rowbias<<<grid(T * 1024), 256>>>(h2.data(), mask[b].b2.data(), T, 1024);
      k_tanh<<<grid(T * 1024), 256>>>(h2.data(), T * 1024);
      m2.allocate(static_cast<size_t>(T) * din * 2);
      gemm_rm(false, true, T, din * 2, 1024, 1.0f, h2.data(), mask[b].w4.data(), 0.0f, m2.data());
      k_add_rowbias<<<grid(T * din * 2), 256>>>(m2.data(), mask[b].b4.data(), T, din * 2);
      k_glu_scatter<<<grid(T * din), 256>>>(m2.data(), mask_out.data(), off_in[b], din, T);
    }
    CK(cudaDeviceSynchronize());
  }

  // Full forward: stereo planar [2,L] -> stem planar [2,L].
  std::vector<float> forward(const std::vector<float>& stereo, int L) const {
    int T = 0;
    CudaBuffer d_re, d_im;
    stft_stereo(stereo, L, d_re, d_im, T);
    CudaBuffer x;
    band_split(stereo, L, x, T);
    blocks_forward(x, T);
    CudaBuffer mask_out;
    mask_estimator(x, mask_out, T);
    return apply_mask(mask_out, d_re, d_im, T, L);
  }

  // Stage 4: scatter-average complex mask, multiply STFT, zero DC, iSTFT per channel.
  std::vector<float> apply_mask(const CudaBuffer& mask_out, const CudaBuffer& d_re,
                                const CudaBuffer& d_im, int T, int L) const {
    CudaBuffer d_fi;
    d_fi.copy_from_host(reinterpret_cast<const float*>(freq_indices.data()), freq_indices.size() * 2);
    CudaBuffer sre, sim, ore, oim;
    sre.allocate(static_cast<size_t>(kMerged) * T); sre.zero();
    sim.allocate(static_cast<size_t>(kMerged) * T); sim.zero();
    ore.allocate(static_cast<size_t>(2) * kNfreq * T);
    oim.allocate(static_cast<size_t>(2) * kNfreq * T);
    k_scatter_mask<<<grid(T * kFiLen), 256>>>(mask_out.data(),
        reinterpret_cast<const int64_t*>(d_fi.data()), sre.data(), sim.data(), T);
    k_apply_mask<<<grid(kMerged * T), 256>>>(d_re.data(), d_im.data(), sre.data(), sim.data(),
        d_num_bands_per_freq.data(), ore.data(), oim.data(), T);
    CK(cudaDeviceSynchronize());

    std::vector<float> hre(static_cast<size_t>(2) * kNfreq * T), him(hre.size());
    ore.copy_to_host(hre.data(), hre.size());
    oim.copy_to_host(him.data(), him.size());
    std::vector<float> out(static_cast<size_t>(2) * L, 0.0f);
    for (int c = 0; c < 2; ++c) {
      std::vector<float> cre(hre.begin() + static_cast<size_t>(c) * kNfreq * T,
                             hre.begin() + static_cast<size_t>(c + 1) * kNfreq * T);
      std::vector<float> cim(him.begin() + static_cast<size_t>(c) * kNfreq * T,
                             him.begin() + static_cast<size_t>(c + 1) * kNfreq * T);
      auto ch = stft.inverse(cre, cim, T, L);
      for (int i = 0; i < L && i < (int)ch.size(); ++i) out[static_cast<size_t>(c) * L + i] = ch[i];
    }
    return out;
  }

  std::vector<float> apply_from_host_mask(const std::vector<float>& mask_host,
                                          const std::vector<float>& stereo, int L, int T) const {
    CudaBuffer d_re, d_im; int Ts = 0;
    stft_stereo(stereo, L, d_re, d_im, Ts);
    CudaBuffer mask_out; mask_out.copy_from_host(mask_host.data(), mask_host.size());
    return apply_mask(mask_out, d_re, d_im, T, L);
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

std::vector<float> Roformer::debug_blocks(const std::vector<float>& stereo, int L, int& T) const {
  CudaBuffer x;
  p_->band_split(stereo, L, x, T);
  p_->blocks_forward(x, T);
  std::vector<float> h(static_cast<size_t>(T) * kNbands * kDim);
  x.copy_to_host(h.data(), h.size());
  return h;
}
std::vector<float> Roformer::debug_blocks_from(const std::vector<float>& bandsplit, int T) const {
  CudaBuffer x;
  x.copy_from_host(bandsplit.data(), bandsplit.size());
  p_->blocks_forward(x, T);
  std::vector<float> h(static_cast<size_t>(T) * kNbands * kDim);
  x.copy_to_host(h.data(), h.size());
  return h;
}
std::vector<float> Roformer::debug_mask(const std::vector<float>& blk_ref, int T) const {
  CudaBuffer x, mask_out;
  x.copy_from_host(blk_ref.data(), blk_ref.size());
  p_->mask_estimator(x, mask_out, T);
  std::vector<float> h(static_cast<size_t>(T) * kBandIn);
  mask_out.copy_to_host(h.data(), h.size());
  return h;
}

std::vector<float> Roformer::separate_stereo(const std::vector<float>& stereo, int L) const {
  return p_->forward(stereo, L);
}

std::vector<float> Roformer::debug_apply(const std::vector<float>& mask,
                                         const std::vector<float>& stereo, int L, int T) const {
  return p_->apply_from_host_mask(mask, stereo, L, T);
}

std::vector<float> Roformer::separate_mono(const float* audio, int L, int sr) const {
  auto r = (sr == kSr) ? std::vector<float>(audio, audio + L)
                       : io::resample_linear(audio, L, sr, kSr);
  int L44 = static_cast<int>(r.size());
  std::vector<float> stereo(static_cast<size_t>(2) * L44);
  for (int i = 0; i < L44; ++i) { stereo[i] = r[i]; stereo[L44 + i] = r[i]; }
  auto st = p_->forward(stereo, L44);
  std::vector<float> mono(L44);
  for (int i = 0; i < L44; ++i) mono[i] = 0.5f * (st[i] + st[L44 + i]);
  if (sr == kSr) return mono;
  return io::resample_linear(mono.data(), L44, kSr, sr);
}

}  // namespace voxmutatio::separation
