// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// MelBand-RoFormer forward runner (spec 004 S6). Implemented + aligned stage by
// stage: band-split -> transformer blocks -> mask estimator -> apply/iSTFT.

#include "voxmutatio/separation/roformer.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <array>
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

// MelBand-RoFormer dimensions are config-driven (read from roformer_config.i64):
// n_fft, hop, dim, depth, num_bands, heads, dim_head, mask_depth, ff_mult, sr.
// Attention head dim is capped at 64 (k_attention accumulator).
constexpr int kMaxDhead = 64;
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
                              float* out, int T, int band_in, int nfreq) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * band_in) return;
  int t = idx / band_in, rest = idx % band_in, k = rest >> 1, c = rest & 1;
  int m = (int)fi[k], s = m & 1, f = m >> 1;
  const float* src = (c == 0) ? re : im;
  out[idx] = src[(static_cast<size_t>(s) * nfreq + f) * T + t];
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
__global__ void k_glu_scatter(const float* m2, float* mask, int off, int din, int T, int band_in) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * din) return;
  int t = idx / din, d = idx % din;
  float a = m2[static_cast<size_t>(t) * (2 * din) + d];
  float g = m2[static_cast<size_t>(t) * (2 * din) + din + d];
  mask[static_cast<size_t>(t) * band_in + off + d] = a * (1.0f / (1.0f + __expf(-g)));
}
// Scatter-add complex masks into merged freq (bands overlap): mask[t, k, c] with
// feature k*2+c maps to merged freq fi[k]. summed[m, t] (re/im) accumulated.
__global__ void k_scatter_mask(const float* mask, const int64_t* fi, float* sre,
                               float* sim, int T, int fi_len, int band_in) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * fi_len) return;
  int t = idx / fi_len, k = idx % fi_len;
  int m = (int)fi[k];
  float re = mask[static_cast<size_t>(t) * band_in + k * 2];
  float im = mask[static_cast<size_t>(t) * band_in + k * 2 + 1];
  atomicAdd(&sre[static_cast<size_t>(m) * T + t], re);
  atomicAdd(&sim[static_cast<size_t>(m) * T + t], im);
}
// Average mask (/num_bands_per_freq), complex-multiply with stft, zero DC, split to
// channel re/im. merged m=f*2+s, nbpf indexed by f. stft re/im [2,kNfreq,T].
__global__ void k_apply_mask(const float* re, const float* im, const float* sre,
                             const float* sim, const float* nbpf, float* ore,
                             float* oim, int T, int merged, int nfreq) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= merged * T) return;
  int t = idx % T, m = idx / T;
  int s = m & 1, f = m >> 1;
  float denom = fmaxf(nbpf[f], 1e-8f);
  float mr = sre[idx] / denom, mi = sim[idx] / denom;
  if (f == 0) { mr = 0.0f; mi = 0.0f; }  // zero DC
  size_t src = (static_cast<size_t>(s) * nfreq + f) * T + t;
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
  std::vector<CudaBuffer> w, b;  // MLP linears (mask_depth+1); Tanh between; GLU after
  int din = 0;
};

struct Roformer::Impl {
  bool ok = false;
  // runtime config (roformer_config.i64)
  int nfft = 2048, hop = 512, nfreq = 1025, merged = 2050;
  int dim = 256, depth = 6, num_bands = 60, heads = 8, dim_head = 64, inner = 512;
  int ff_hidden = 1024, mask_depth = 2, sr = 44100, fi_len = 0, band_in = 0;
  std::unique_ptr<Stft> stft;
  // band split
  std::vector<CudaBuffer> bs_gamma, bs_w, bs_b;
  std::vector<int> dim_in, off_in;   // per band input dim + offset into band_in
  // transformer blocks [depth][0=time,1=freq]
  std::vector<std::array<Block, 2>> blocks;
  // mask estimators (num_stems=1)
  std::vector<MaskBand> mask;
  // band maps
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
    // Config: [nfft, hop, dim, depth, num_bands, heads, dim_head, mask_depth, ff_mult, sr]
    auto cfg = read_i64(dir + "/" + model + "_config.i64", 0);
    if (cfg.size() < 9) { fprintf(stderr, "roformer: missing/short config for %s\n", model.c_str()); return; }
    nfft = (int)cfg[0]; hop = (int)cfg[1]; dim = (int)cfg[2]; depth = (int)cfg[3];
    num_bands = (int)cfg[4]; heads = (int)cfg[5]; dim_head = (int)cfg[6];
    mask_depth = (int)cfg[7]; ff_hidden = dim * (int)cfg[8];
    sr = cfg.size() > 9 ? (int)cfg[9] : 44100;
    nfreq = nfft / 2 + 1; merged = nfreq * 2; inner = heads * dim_head;
    if (dim_head > kMaxDhead) { fprintf(stderr, "roformer: dim_head %d > %d unsupported\n", dim_head, kMaxDhead); return; }
    stft = std::make_unique<Stft>(nfft, hop);

    io::SafetensorsLoader L;
    if (!L.load(dir + "/" + model + ".safetensors")) {
      fprintf(stderr, "roformer: cannot load %s\n", (dir + "/" + model).c_str());
      return;
    }
    auto di = read_i64(dir + "/" + model + "_dim_inputs.i64", num_bands);
    freq_indices = read_i64(dir + "/" + model + "_freq_indices.i64", 0);
    num_bands_per_freq = read_i64(dir + "/" + model + "_num_bands_per_freq.i64", nfreq);
    if (di.empty() || freq_indices.empty() || num_bands_per_freq.empty()) return;
    fi_len = (int)freq_indices.size();

    bool g = true;
    int off = 0;
    for (int b = 0; b < num_bands; ++b) {
      int din = (int)di[b];
      dim_in.push_back(din);
      off_in.push_back(off);
      off += din;
      bs_gamma.emplace_back(); bs_w.emplace_back(); bs_b.emplace_back();
      std::string p = "band_split.to_features." + std::to_string(b) + ".";
      g &= up(L, p + "0.gamma", bs_gamma[b], din);
      g &= up(L, p + "1.weight", bs_w[b], dim * din);
      g &= up(L, p + "1.bias", bs_b[b], dim);
    }
    band_in = off;
    blocks.resize(depth);
    for (int i = 0; i < depth; ++i) {
      for (int d = 0; d < 2; ++d) {
        std::string p = "layers." + std::to_string(i) + "." + std::to_string(d) + ".";
        Block& bl = blocks[i][d];
        g &= up(L, p + "layers.0.0.norm.gamma", bl.attn.norm_gamma, dim);
        g &= up(L, p + "layers.0.0.rotary_embed.freqs", bl.attn.rope_freqs, dim_head / 2);
        g &= up(L, p + "layers.0.0.to_qkv.weight", bl.attn.qkv_w, 3 * inner * dim);
        g &= up(L, p + "layers.0.0.to_gates.weight", bl.attn.gates_w, heads * dim);
        g &= up(L, p + "layers.0.0.to_gates.bias", bl.attn.gates_b, heads);
        g &= up(L, p + "layers.0.0.to_out.0.weight", bl.attn.out_w, dim * inner);
        g &= up(L, p + "layers.0.1.net.0.gamma", bl.ffn.n0_gamma, dim);
        g &= up(L, p + "layers.0.1.net.1.weight", bl.ffn.w1, ff_hidden * dim);
        g &= up(L, p + "layers.0.1.net.1.bias", bl.ffn.b1, ff_hidden);
        g &= up(L, p + "layers.0.1.net.4.weight", bl.ffn.w4, dim * ff_hidden);
        g &= up(L, p + "layers.0.1.net.4.bias", bl.ffn.b4, dim);
        g &= up(L, p + "norm.gamma", bl.out_gamma, dim);
      }
    }
    mask.resize(num_bands);
    for (int b = 0; b < num_bands; ++b) {
      int din = dim_in[b];
      mask[b].din = din;
      // MLP: (mask_depth+1) linears at keys .0, .2, .4, ...; dims dim -> ff_hidden(x mask_depth) -> din*2
      for (int l = 0; l <= mask_depth; ++l) {
        int in_d = (l == 0) ? dim : ff_hidden;
        int out_d = (l == mask_depth) ? din * 2 : ff_hidden;
        mask[b].w.emplace_back(); mask[b].b.emplace_back();
        std::string p = "mask_estimators.0.to_freqs." + std::to_string(b) + ".0." + std::to_string(2 * l) + ".";
        g &= up(L, p + "weight", mask[b].w[l], out_d * in_d);
        g &= up(L, p + "bias", mask[b].b[l], out_d);
      }
    }
    std::vector<float> nbpf(num_bands_per_freq.begin(), num_bands_per_freq.end());
    d_num_bands_per_freq.copy_from_host(nbpf.data(), nbpf.size());
    ok = g;
  }

  int frames(int L) const { return stft->num_frames(L); }

  // STFT both channels -> stacked re/im [2, nfreq, T] on device.
  void stft_stereo(const std::vector<float>& stereo, int L, CudaBuffer& d_re,
                   CudaBuffer& d_im, int& T) const {
    std::vector<float> re[2], im[2];
    for (int c = 0; c < 2; ++c)
      stft->forward(stereo.data() + static_cast<size_t>(c) * L, L, re[c], im[c], T);
    std::vector<float> hre(static_cast<size_t>(2) * nfreq * T), him(hre.size());
    for (int c = 0; c < 2; ++c)
      for (size_t i = 0; i < static_cast<size_t>(nfreq) * T; ++i) {
        hre[static_cast<size_t>(c) * nfreq * T + i] = re[c][i];
        him[static_cast<size_t>(c) * nfreq * T + i] = im[c][i];
      }
    d_re.copy_from_host(hre.data(), hre.size());
    d_im.copy_from_host(him.data(), him.size());
  }

  // STFT + gather + fold -> band split input [T, band_in] on device.
  void band_split_in(const std::vector<float>& stereo, int L, CudaBuffer& bin, int& T) const {
    CudaBuffer d_re, d_im;
    stft_stereo(stereo, L, d_re, d_im, T);
    CudaBuffer d_fi;
    d_fi.copy_from_host(reinterpret_cast<const float*>(freq_indices.data()), freq_indices.size() * 2);
    bin.allocate(static_cast<size_t>(T) * band_in);
    k_gather_fold<<<grid(T * band_in), 256>>>(d_re.data(), d_im.data(),
        reinterpret_cast<const int64_t*>(d_fi.data()), bin.data(), T, band_in, nfreq);
    CK(cudaDeviceSynchronize());
  }

  // STFT + gather + band split -> bandsplit_out [T, num_bands, dim] on device.
  void band_split(const std::vector<float>& stereo, int L, CudaBuffer& out, int& T) const {
    CudaBuffer d_re, d_im;
    stft_stereo(stereo, L, d_re, d_im, T);
    CudaBuffer d_fi;
    d_fi.copy_from_host(reinterpret_cast<const float*>(freq_indices.data()),
                        freq_indices.size() * 2);  // int64 -> 2 float slots each
    CudaBuffer bin;
    bin.allocate(static_cast<size_t>(T) * band_in);
    k_gather_fold<<<grid(T * band_in), 256>>>(d_re.data(), d_im.data(),
        reinterpret_cast<const int64_t*>(d_fi.data()), bin.data(), T, band_in, nfreq);

    out.allocate(static_cast<size_t>(T) * num_bands * dim);
    CudaBuffer tmp, lin;
    for (int b = 0; b < num_bands; ++b) {
      int din = dim_in[b];
      tmp.allocate(static_cast<size_t>(T) * din);
      k_rmsnorm_slice<<<grid(T), 256>>>(bin.data(), band_in, off_in[b], bs_gamma[b].data(),
                                        tmp.data(), T, din);
      lin.allocate(static_cast<size_t>(T) * dim);
      gemm_rm(false, true, T, dim, din, 1.0f, tmp.data(), bs_w[b].data(), 0.0f, lin.data());
      k_bias_scatter_band<<<grid(T * dim), 256>>>(lin.data(), bs_b[b].data(), out.data(), b, num_bands, T, dim);
    }
    CK(cudaDeviceSynchronize());
  }

  // One transformer (depth 1): attn residual + ffn residual + output RMSNorm.
  // x is [B*S, kDim] on device (B sequences of length S); modified in place.
  void transformer(CudaBuffer& x, int B, int S, const Block& bl) const {
    const int BS = B * S, D = dim, H = heads, DH = dim_head, IN = inner, FF = ff_hidden;
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
    CudaBuffer h1; h1.allocate(static_cast<size_t>(BS) * FF);
    gemm_rm(false, true, BS, FF, D, 1.0f, fn.data(), bl.ffn.w1.data(), 0.0f, h1.data());
    k_add_rowbias<<<grid(BS * FF), 256>>>(h1.data(), bl.ffn.b1.data(), BS, FF);
    k_gelu<<<grid(BS * FF), 256>>>(h1.data(), BS * FF);
    CudaBuffer h2; h2.allocate(static_cast<size_t>(BS) * D);
    gemm_rm(false, true, BS, D, FF, 1.0f, h1.data(), bl.ffn.w4.data(), 0.0f, h2.data());
    k_add_rowbias<<<grid(BS * D), 256>>>(h2.data(), bl.ffn.b4.data(), BS, D);
    k_add<<<grid(BS * D), 256>>>(x.data(), h2.data(), BS * D);

    CudaBuffer xo; xo.allocate(static_cast<size_t>(BS) * D);
    k_rmsnorm<<<grid(BS), 256>>>(x.data(), bl.out_gamma.data(), xo.data(), BS, D);
    x = std::move(xo);
  }

  // 6 blocks: each = time transformer (over T per band) + freq transformer (over bands per t).
  // x is [T, num_bands, dim] in place.
  void blocks_forward(CudaBuffer& x, int T) const {
    const size_t n = static_cast<size_t>(T) * num_bands * dim;
    for (int i = 0; i < depth; ++i) {
      CudaBuffer xt; xt.allocate(n);
      k_transpose12<<<grid(n), 256>>>(x.data(), xt.data(), T, num_bands, dim);  // [T,NB,D]->[NB,T,D]
      transformer(xt, num_bands, T, blocks[i][0]);                             // time
      k_transpose12<<<grid(n), 256>>>(xt.data(), x.data(), num_bands, T, dim);  // [NB,T,D]->[T,NB,D]
      transformer(x, T, num_bands, blocks[i][1]);                              // freq
    }
    CK(cudaDeviceSynchronize());
  }

  // Mask estimator: per band MLP(dim->ff_hidden x mask_depth ->din*2, Tanh) + GLU -> mask [T, band_in].
  void mask_estimator(const CudaBuffer& blk, CudaBuffer& mask_out, int T) const {
    mask_out.allocate(static_cast<size_t>(T) * band_in);
    CudaBuffer bandin, cur, nxt;
    for (int b = 0; b < num_bands; ++b) {
      int din = dim_in[b];
      bandin.allocate(static_cast<size_t>(T) * dim);
      k_extract_band<<<grid(T * dim), 256>>>(blk.data(), bandin.data(), b, num_bands, T, dim);
      const float* in_ptr = bandin.data();
      int in_d = dim;
      for (int l = 0; l <= mask_depth; ++l) {
        int out_d = (l == mask_depth) ? din * 2 : ff_hidden;
        nxt.allocate(static_cast<size_t>(T) * out_d);
        gemm_rm(false, true, T, out_d, in_d, 1.0f, in_ptr, mask[b].w[l].data(), 0.0f, nxt.data());
        k_add_rowbias<<<grid(T * out_d), 256>>>(nxt.data(), mask[b].b[l].data(), T, out_d);
        if (l != mask_depth) k_tanh<<<grid(T * out_d), 256>>>(nxt.data(), T * out_d);
        cur = std::move(nxt);
        in_ptr = cur.data();
        in_d = out_d;
      }
      k_glu_scatter<<<grid(T * din), 256>>>(cur.data(), mask_out.data(), off_in[b], din, T, band_in);
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
    sre.allocate(static_cast<size_t>(merged) * T); sre.zero();
    sim.allocate(static_cast<size_t>(merged) * T); sim.zero();
    ore.allocate(static_cast<size_t>(2) * nfreq * T);
    oim.allocate(static_cast<size_t>(2) * nfreq * T);
    k_scatter_mask<<<grid(T * fi_len), 256>>>(mask_out.data(),
        reinterpret_cast<const int64_t*>(d_fi.data()), sre.data(), sim.data(), T, fi_len, band_in);
    k_apply_mask<<<grid(merged * T), 256>>>(d_re.data(), d_im.data(), sre.data(), sim.data(),
        d_num_bands_per_freq.data(), ore.data(), oim.data(), T, merged, nfreq);
    CK(cudaDeviceSynchronize());

    std::vector<float> hre(static_cast<size_t>(2) * nfreq * T), him(hre.size());
    ore.copy_to_host(hre.data(), hre.size());
    oim.copy_to_host(him.data(), him.size());
    std::vector<float> out(static_cast<size_t>(2) * L, 0.0f);
    for (int c = 0; c < 2; ++c) {
      std::vector<float> cre(hre.begin() + static_cast<size_t>(c) * nfreq * T,
                             hre.begin() + static_cast<size_t>(c + 1) * nfreq * T);
      std::vector<float> cim(him.begin() + static_cast<size_t>(c) * nfreq * T,
                             him.begin() + static_cast<size_t>(c + 1) * nfreq * T);
      auto ch = stft->inverse(cre, cim, T, L);
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
int Roformer::n_fft() const { return p_->nfft; }
int Roformer::hop() const { return p_->hop; }
int Roformer::sample_rate() const { return p_->sr; }

std::vector<float> Roformer::debug_bandsplit(const std::vector<float>& stereo, int L, int& T) const {
  CudaBuffer out;
  p_->band_split(stereo, L, out, T);
  std::vector<float> h(static_cast<size_t>(T) * p_->num_bands * p_->dim);
  out.copy_to_host(h.data(), h.size());
  return h;
}

std::vector<float> Roformer::debug_bandsplit_in(const std::vector<float>& stereo, int L, int& T) const {
  CudaBuffer bin;
  p_->band_split_in(stereo, L, bin, T);
  std::vector<float> h(static_cast<size_t>(T) * p_->band_in);
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
  std::vector<float> h(static_cast<size_t>(T) * p_->num_bands * p_->dim);
  x.copy_to_host(h.data(), h.size());
  return h;
}
std::vector<float> Roformer::debug_blocks_from(const std::vector<float>& bandsplit, int T) const {
  CudaBuffer x;
  x.copy_from_host(bandsplit.data(), bandsplit.size());
  p_->blocks_forward(x, T);
  std::vector<float> h(static_cast<size_t>(T) * p_->num_bands * p_->dim);
  x.copy_to_host(h.data(), h.size());
  return h;
}
std::vector<float> Roformer::debug_mask(const std::vector<float>& blk_ref, int T) const {
  CudaBuffer x, mask_out;
  x.copy_from_host(blk_ref.data(), blk_ref.size());
  p_->mask_estimator(x, mask_out, T);
  std::vector<float> h(static_cast<size_t>(T) * p_->band_in);
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
  int msr = p_->sr;
  auto r = (sr == msr) ? std::vector<float>(audio, audio + L)
                       : io::resample_linear(audio, L, sr, msr);
  int L44 = static_cast<int>(r.size());
  std::vector<float> stereo(static_cast<size_t>(2) * L44);
  for (int i = 0; i < L44; ++i) { stereo[i] = r[i]; stereo[L44 + i] = r[i]; }
  auto st = p_->forward(stereo, L44);
  std::vector<float> mono(L44);
  for (int i = 0; i < L44; ++i) mono[i] = 0.5f * (st[i] + st[L44 + i]);
  if (sr == msr) return mono;
  return io::resample_linear(mono.data(), L44, msr, sr);
}

}  // namespace voxmutatio::separation
