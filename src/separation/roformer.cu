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
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "voxmutatio/core/cuda_buffer.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/io/safetensors.h"
#include "voxmutatio/separation/stft.h"
#include "validation.h"

namespace voxmutatio::separation {
namespace {

using core::CudaBuffer;

[[noreturn]] void throw_cuda(const char* expr, cudaError_t error) {
  throw std::runtime_error(std::string("Roformer CUDA failure in ") + expr + ": " +
                           cudaGetErrorString(error));
}

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) \
  throw_cuda(#call, e_); } while (0)

void ck_blas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("Roformer cuBLAS failure in ") + operation +
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
void gemm_rm(bool tA, bool tB, int M, int N, int K, float alpha,
             const float* A, const float* B, float beta, float* C) {
  int lda = tA ? M : K, ldb = tB ? K : N;
  ck_blas(cublasSgemm(cublas(), tB ? CUBLAS_OP_T : CUBLAS_OP_N,
                      tA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, ldb,
                      A, lda, &beta, C, N),
          "cublasSgemm");
}

// Strided-batched row-major C[b][M,N] = alpha*op(A)[M,K] @ op(B)[K,N].
void gemm_batched(bool tA, bool tB, int M, int N, int K, float alpha,
                  const float* A, long sA, const float* B, long sB,
                  float beta, float* C, long sC, int batch) {
  int lda = tA ? M : K, ldb = tB ? K : N;
  ck_blas(cublasSgemmStridedBatched(cublas(), tB ? CUBLAS_OP_T : CUBLAS_OP_N,
      tA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, ldb, sB, A, lda, sA,
      &beta, C, N, sC, batch), "cublasSgemmStridedBatched");
}

// Softmax over the last dim (width W) of rows x W matrix, in place.
__global__ void k_softmax_rows(float* x, int rows, int W) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float* row = x + static_cast<size_t>(r) * W;
  float m = -1e30f;
  for (int j = 0; j < W; ++j) m = fmaxf(m, row[j]);
  float s = 0.0f;
  for (int j = 0; j < W; ++j) { float e = __expf(row[j] - m); row[j] = e; s += e; }
  float inv = 1.0f / s;
  for (int j = 0; j < W; ++j) row[j] *= inv;
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
// (Replaced by cuBLAS strided-batched GEMM + k_softmax_rows in transformer().)
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

std::vector<int64_t> read_i64(const std::string& path, int expect,
                              std::size_t max_values = 10'000'000) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.good()) return {};
  const std::streamoff bytes = f.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(int64_t)) != 0) return {};
  auto n = static_cast<size_t>(bytes) / sizeof(int64_t);
  if (n > max_values || (expect && static_cast<int>(n) != expect)) return {};
  std::vector<int64_t> v(n);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(int64_t));
  if (!f) {
    fprintf(stderr, "roformer: %s has %zu values, want %d\n", path.c_str(), n, expect);
    return {};
  }
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
  int chunk_size = 0;  // long-audio chunk (samples); 0 -> process whole input
  int target_stem = 0;  // which mask_estimator (stem) to output (multi-stem models)
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

  // Reusable device scratch (avoids per-band/per-transformer cudaMalloc churn,
  // the same fix that gave the training path ~40x). Grow-only via ensure().
  struct Scratch {
    CudaBuffer d_fi;                                      // freq_indices (int64->float), built once
    CudaBuffer bin, bs_tmp, bs_lin;                       // band_split
    CudaBuffer xn, qkv, q, k, v, scores, ao, mo, gates, att, fn, h1, h2, xo;  // transformer
    CudaBuffer xt;                                        // blocks_forward transpose
    CudaBuffer bandin, mlp_a, mlp_b;                      // mask_estimator MLP ping-pong
    CudaBuffer sre, sim, ore, oim;                        // apply_mask
  };
  mutable Scratch sc;
  mutable std::mutex scratch_mu;
  static void ensure(CudaBuffer& b, std::size_t n) { if (b.size() < n) b.allocate(n); }
  void release_scratch() const { sc = Scratch{}; }
  struct ScratchGuard {
    const Impl* owner;
    ~ScratchGuard() { owner->release_scratch(); }
  };

  template <typename Fn>
  decltype(auto) with_scratch_release(Fn&& fn) const {
    // All public forwards on one instance share `sc`; serialize them so one
    // call cannot resize or release another call's in-flight device buffers.
    std::unique_lock<std::mutex> lock(scratch_mu);
    ScratchGuard guard{this};
    return std::forward<Fn>(fn)();
  }

  static bool up(const io::SafetensorsLoader& L, const std::string& n, CudaBuffer& b,
                 std::initializer_list<int64_t> shape) {
    return detail::upload_checked_f32(L, n, shape, b, "roformer");
  }

  explicit Impl(const std::string& dir, const std::string& model) {
    // Config: [nfft, hop, dim, depth, num_bands, heads, dim_head, mask_depth, ff_mult, sr]
    auto cfg = read_i64(dir + "/" + model + "_config.i64", 0, 12);
    if (cfg.size() < 9 || cfg.size() > 12 || cfg[0] < 16 || cfg[0] > 32'768 ||
        cfg[1] < 1 || cfg[1] > cfg[0] || cfg[2] < 1 || cfg[2] > 2'048 ||
        cfg[3] < 1 || cfg[3] > 64 || cfg[4] < 1 || cfg[4] > 512 ||
        cfg[5] < 1 || cfg[5] > 64 || cfg[6] < 2 || cfg[6] > kMaxDhead ||
        cfg[6] % 2 != 0 || cfg[7] < 0 || cfg[7] > 8 || cfg[8] < 1 || cfg[8] > 32 ||
        (cfg.size() > 9 && (cfg[9] < 8'000 || cfg[9] > 384'000)) ||
        (cfg.size() > 10 && (cfg[10] < 0 || cfg[10] > 100'000'000)) ||
        (cfg.size() > 11 && (cfg[11] < 0 || cfg[11] > 64))) {
      fprintf(stderr, "roformer: invalid config for %s\n", model.c_str());
      return;
    }
    nfft = (int)cfg[0]; hop = (int)cfg[1]; dim = (int)cfg[2]; depth = (int)cfg[3];
    num_bands = (int)cfg[4]; heads = (int)cfg[5]; dim_head = (int)cfg[6];
    mask_depth = (int)cfg[7]; ff_hidden = dim * (int)cfg[8];
    sr = cfg.size() > 9 ? (int)cfg[9] : 44100;
    chunk_size = cfg.size() > 10 ? (int)cfg[10] : 0;
    target_stem = cfg.size() > 11 ? (int)cfg[11] : 0;
    nfreq = nfft / 2 + 1; merged = nfreq * 2; inner = heads * dim_head;
    if (ff_hidden <= 0 || inner <= 0) {
      fprintf(stderr, "roformer: invalid model configuration for %s\n", model.c_str());
      return;
    }
    // Full time-attention is quadratic in the frame count. Legacy sidecars use
    // 8-11 second chunks, which require more than 1-2 GiB for a single score
    // tensor. Bound every model, including sidecars with chunk_size=0, to four
    // seconds so valid input cannot unexpectedly consume all device memory.
    const int safe_chunk = std::max(nfft, sr * 4);
    if (chunk_size < nfft || chunk_size > safe_chunk) chunk_size = safe_chunk;
    stft = std::make_unique<Stft>(nfft, hop);

    io::SafetensorsLoader L;
    if (!L.load(dir + "/" + model + ".safetensors")) {
      fprintf(stderr, "roformer: cannot load %s\n", (dir + "/" + model).c_str());
      return;
    }
    auto di = read_i64(dir + "/" + model + "_dim_inputs.i64", num_bands);
    freq_indices = read_i64(dir + "/" + model + "_freq_indices.i64", 0, 500'000);
    num_bands_per_freq = read_i64(dir + "/" + model + "_num_bands_per_freq.i64", nfreq);
    if (di.empty() || freq_indices.empty() || num_bands_per_freq.empty()) return;
    fi_len = (int)freq_indices.size();

    bool g = true;
    int64_t off = 0;
    for (int b = 0; b < num_bands; ++b) {
      if (di[b] <= 0 || di[b] > merged || di[b] % 2 != 0 ||
          off + di[b] > 1'000'000) {
        fprintf(stderr, "roformer: invalid band dimensions\n");
        return;
      }
      int din = static_cast<int>(di[b]);
      dim_in.push_back(din);
      off_in.push_back(static_cast<int>(off));
      off += din;
      bs_gamma.emplace_back(); bs_w.emplace_back(); bs_b.emplace_back();
      std::string p = "band_split.to_features." + std::to_string(b) + ".";
      g &= up(L, p + "0.gamma", bs_gamma[b], {din});
      g &= up(L, p + "1.weight", bs_w[b], {dim, din});
      g &= up(L, p + "1.bias", bs_b[b], {dim});
    }
    band_in = static_cast<int>(off);
    if (freq_indices.size() * 2 != static_cast<std::size_t>(band_in)) {
      fprintf(stderr, "roformer: band map size does not match band dimensions\n");
      return;
    }
    for (int64_t index : freq_indices) {
      if (index < 0 || index >= merged) {
        fprintf(stderr, "roformer: frequency index is out of range\n");
        return;
      }
    }
    for (int64_t count : num_bands_per_freq) {
      if (count <= 0 || count > num_bands) {
        fprintf(stderr, "roformer: invalid bands-per-frequency map\n");
        return;
      }
    }
    std::vector<int> occurrences(static_cast<std::size_t>(merged), 0);
    for (int64_t index : freq_indices) {
      ++occurrences[static_cast<std::size_t>(index)];
    }
    for (int f = 0; f < nfreq; ++f) {
      const int expected = static_cast<int>(num_bands_per_freq[f]);
      if (occurrences[static_cast<std::size_t>(2 * f)] != expected ||
          occurrences[static_cast<std::size_t>(2 * f + 1)] != expected) {
        fprintf(stderr, "roformer: bands-per-frequency does not match frequency map\n");
        return;
      }
    }
    try {
      require_supported_frames(stft->num_frames(chunk_size));
    } catch (const std::invalid_argument& error) {
      fprintf(stderr, "roformer: unsafe configured tensor sizes: %s\n", error.what());
      return;
    }
    blocks.resize(depth);
    for (int i = 0; i < depth; ++i) {
      for (int d = 0; d < 2; ++d) {
        std::string p = "layers." + std::to_string(i) + "." + std::to_string(d) + ".";
        Block& bl = blocks[i][d];
        g &= up(L, p + "layers.0.0.norm.gamma", bl.attn.norm_gamma, {dim});
        g &= up(L, p + "layers.0.0.rotary_embed.freqs", bl.attn.rope_freqs,
                {dim_head / 2});
        g &= up(L, p + "layers.0.0.to_qkv.weight", bl.attn.qkv_w,
                {3 * inner, dim});
        g &= up(L, p + "layers.0.0.to_gates.weight", bl.attn.gates_w,
                {heads, dim});
        g &= up(L, p + "layers.0.0.to_gates.bias", bl.attn.gates_b, {heads});
        g &= up(L, p + "layers.0.0.to_out.0.weight", bl.attn.out_w,
                {dim, inner});
        g &= up(L, p + "layers.0.1.net.0.gamma", bl.ffn.n0_gamma, {dim});
        g &= up(L, p + "layers.0.1.net.1.weight", bl.ffn.w1,
                {ff_hidden, dim});
        g &= up(L, p + "layers.0.1.net.1.bias", bl.ffn.b1, {ff_hidden});
        g &= up(L, p + "layers.0.1.net.4.weight", bl.ffn.w4,
                {dim, ff_hidden});
        g &= up(L, p + "layers.0.1.net.4.bias", bl.ffn.b4, {dim});
        g &= up(L, p + "norm.gamma", bl.out_gamma, {dim});
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
        std::string p = "mask_estimators." + std::to_string(target_stem) + ".to_freqs." + std::to_string(b) + ".0." + std::to_string(2 * l) + ".";
        g &= up(L, p + "weight", mask[b].w[l], {out_d, in_d});
        g &= up(L, p + "bias", mask[b].b[l], {out_d});
      }
    }
    std::vector<float> nbpf(num_bands_per_freq.begin(), num_bands_per_freq.end());
    d_num_bands_per_freq.copy_from_host(nbpf.data(), nbpf.size());
    ok = g;
  }

  int frames(int L) const { return stft->num_frames(L); }

  void require_supported_frames(int T) const {
    if (T <= 0) throw std::invalid_argument("Roformer frame count must be positive");
    detail::checked_int_product({T, band_in}, "Roformer band input");
    detail::checked_int_product({T, num_bands, dim}, "Roformer block tensor");
    detail::checked_int_product({T, num_bands, inner}, "Roformer attention tensor");
    detail::checked_int_product({T, num_bands, heads, dim_head / 2},
                                "Roformer rotary tensor");
    detail::checked_int_product({T, num_bands, ff_hidden}, "Roformer FFN tensor");
    detail::checked_int_product({T, fi_len}, "Roformer frequency map launch");
    detail::checked_int_product({T, merged}, "Roformer merged spectrum");
    detail::checked_int_product({2, T, nfreq}, "Roformer spectrum tensor");
    for (int din : dim_in) {
      detail::checked_int_product({T, std::max(ff_hidden, 2 * din)},
                                  "Roformer mask tensor");
    }
    detail::checked_float_count({num_bands, heads, T, T},
                                "Roformer time-attention scores");
    detail::checked_float_count({T, heads, num_bands, num_bands},
                                "Roformer frequency-attention scores");
  }

  int require_supported_length(int L, bool chunked) const {
    if (L <= 0) throw std::invalid_argument("Roformer audio length must be positive");
    const int work_length =
        chunked && chunk_size > 0 ? std::min(L, chunk_size) : L;
    const int T = frames(work_length);
    require_supported_frames(T);
    return T;
  }

  // STFT both channels -> stacked re/im [2, nfreq, T] on device.
  void stft_stereo(const std::vector<float>& stereo, int L, CudaBuffer& d_re,
                   CudaBuffer& d_im, int& T) const {
    T = require_supported_length(L, false);
    std::vector<float> re[2], im[2];
    for (int c = 0; c < 2; ++c) {
      int channel_frames = 0;
      stft->forward(stereo.data() + static_cast<size_t>(c) * L, L, re[c], im[c],
                    channel_frames);
      if (channel_frames != T) {
        throw std::runtime_error("Roformer channel STFT frame counts disagree");
      }
    }
    std::vector<float> hre(static_cast<size_t>(2) * nfreq * T), him(hre.size());
    for (int c = 0; c < 2; ++c)
      for (size_t i = 0; i < static_cast<size_t>(nfreq) * T; ++i) {
        hre[static_cast<size_t>(c) * nfreq * T + i] = re[c][i];
        him[static_cast<size_t>(c) * nfreq * T + i] = im[c][i];
      }
    detail::require_finite_output(hre, "Roformer STFT real component");
    detail::require_finite_output(him, "Roformer STFT imaginary component");
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
    const int band_elements =
        detail::checked_int_product({T, band_in}, "Roformer band input");
    k_gather_fold<<<grid(band_elements), 256>>>(d_re.data(), d_im.data(),
        reinterpret_cast<const int64_t*>(d_fi.data()), bin.data(), T, band_in, nfreq);
    CK(cudaDeviceSynchronize());
  }

  // Gather + fold PRECOMPUTED spectra -> band-split features out [T, num_bands, dim].
  void band_split_from_spec(const CudaBuffer& d_re, const CudaBuffer& d_im, CudaBuffer& out, int T) const {
    require_supported_frames(T);
    if (sc.d_fi.empty())
      sc.d_fi.copy_from_host(reinterpret_cast<const float*>(freq_indices.data()), freq_indices.size() * 2);
    ensure(sc.bin, static_cast<size_t>(T) * band_in);
    const int band_elements =
        detail::checked_int_product({T, band_in}, "Roformer band input");
    k_gather_fold<<<grid(band_elements), 256>>>(d_re.data(), d_im.data(),
        reinterpret_cast<const int64_t*>(sc.d_fi.data()), sc.bin.data(), T, band_in, nfreq);
    out.allocate(static_cast<size_t>(T) * num_bands * dim);
    for (int b = 0; b < num_bands; ++b) {
      int din = dim_in[b];
      ensure(sc.bs_tmp, static_cast<size_t>(T) * din);
      k_rmsnorm_slice<<<grid(T), 256>>>(sc.bin.data(), band_in, off_in[b], bs_gamma[b].data(),
                                        sc.bs_tmp.data(), T, din);
      ensure(sc.bs_lin, static_cast<size_t>(T) * dim);
      gemm_rm(false, true, T, dim, din, 1.0f, sc.bs_tmp.data(), bs_w[b].data(), 0.0f, sc.bs_lin.data());
      const int projected_elements =
          detail::checked_int_product({T, dim}, "Roformer projected band");
      k_bias_scatter_band<<<grid(projected_elements), 256>>>(
          sc.bs_lin.data(), bs_b[b].data(), out.data(), b, num_bands, T, dim);
    }
    CK(cudaDeviceSynchronize());
  }

  // STFT + gather + band split -> bandsplit_out [T, num_bands, dim] on device (debug/standalone).
  void band_split(const std::vector<float>& stereo, int L, CudaBuffer& out, int& T) const {
    CudaBuffer d_re, d_im;
    stft_stereo(stereo, L, d_re, d_im, T);
    band_split_from_spec(d_re, d_im, out, T);
  }

  // One transformer (depth 1): attn residual + ffn residual + output RMSNorm.
  // x is [B*S, kDim] on device (B sequences of length S); modified in place.
  void transformer(CudaBuffer& x, int B, int S, const Block& bl) const {
    const int BS = detail::checked_int_product({B, S}, "Roformer attention batch");
    const int D = dim, H = heads, DH = dim_head, IN = inner, FF = ff_hidden;
    const int bs_inner =
        detail::checked_int_product({BS, IN}, "Roformer attention tensor");
    const int rope_n =
        detail::checked_int_product({B, H, S, DH / 2}, "Roformer rotary tensor");
    const int attention_rows =
        detail::checked_int_product({B, H, S}, "Roformer attention rows");
    const int bs_heads =
        detail::checked_int_product({BS, H}, "Roformer attention gates");
    const int bs_dim = detail::checked_int_product({BS, D}, "Roformer block tensor");
    const int bs_ff = detail::checked_int_product({BS, FF}, "Roformer FFN tensor");
    ensure(sc.xn, static_cast<size_t>(BS) * D);
    k_rmsnorm<<<grid(BS), 256>>>(x.data(), bl.attn.norm_gamma.data(), sc.xn.data(), BS, D);
    ensure(sc.qkv, static_cast<size_t>(BS) * 3 * IN);
    gemm_rm(false, true, BS, 3 * IN, D, 1.0f, sc.xn.data(), bl.attn.qkv_w.data(), 0.0f, sc.qkv.data());
    ensure(sc.q, static_cast<size_t>(BS) * IN); ensure(sc.k, static_cast<size_t>(BS) * IN); ensure(sc.v, static_cast<size_t>(BS) * IN);
    k_split_heads<<<grid(bs_inner), 256>>>(sc.qkv.data(), sc.q.data(), sc.k.data(),
                                           sc.v.data(), B, S, H, DH);
    k_rope<<<grid(rope_n), 256>>>(sc.q.data(), bl.attn.rope_freqs.data(), B, H, S, DH);
    k_rope<<<grid(rope_n), 256>>>(sc.k.data(), bl.attn.rope_freqs.data(), B, H, S, DH);
    // Attention via batched GEMM: scores = softmax(Q K^T / sqrt(DH)); out = scores V.
    ensure(sc.scores,
           detail::checked_float_count({B, H, S, S}, "Roformer attention scores"));
    gemm_batched(false, true, S, S, DH, 1.0f / sqrtf((float)DH),
                 sc.q.data(), (long)S * DH, sc.k.data(), (long)S * DH, 0.0f,
                 sc.scores.data(), (long)S * S, B * H);
    k_softmax_rows<<<grid(attention_rows), 256>>>(sc.scores.data(), attention_rows, S);
    ensure(sc.ao, static_cast<size_t>(BS) * IN);
    gemm_batched(false, false, S, DH, S, 1.0f, sc.scores.data(), (long)S * S,
                 sc.v.data(), (long)S * DH, 0.0f, sc.ao.data(), (long)S * DH, B * H);
    ensure(sc.mo, static_cast<size_t>(BS) * IN);
    k_merge_heads<<<grid(bs_inner), 256>>>(sc.ao.data(), sc.mo.data(), B, S, H, DH);
    ensure(sc.gates, static_cast<size_t>(BS) * H);
    gemm_rm(false, true, BS, H, D, 1.0f, sc.xn.data(), bl.attn.gates_w.data(), 0.0f, sc.gates.data());
    k_add_rowbias<<<grid(bs_heads), 256>>>(sc.gates.data(), bl.attn.gates_b.data(), BS, H);
    k_apply_gates<<<grid(bs_inner), 256>>>(sc.mo.data(), sc.gates.data(), BS, H, DH);
    ensure(sc.att, static_cast<size_t>(BS) * D);
    gemm_rm(false, true, BS, D, IN, 1.0f, sc.mo.data(), bl.attn.out_w.data(), 0.0f, sc.att.data());
    k_add<<<grid(bs_dim), 256>>>(x.data(), sc.att.data(), bs_dim);

    ensure(sc.fn, static_cast<size_t>(BS) * D);
    k_rmsnorm<<<grid(BS), 256>>>(x.data(), bl.ffn.n0_gamma.data(), sc.fn.data(), BS, D);
    ensure(sc.h1, static_cast<size_t>(BS) * FF);
    gemm_rm(false, true, BS, FF, D, 1.0f, sc.fn.data(), bl.ffn.w1.data(), 0.0f, sc.h1.data());
    k_add_rowbias<<<grid(bs_ff), 256>>>(sc.h1.data(), bl.ffn.b1.data(), BS, FF);
    k_gelu<<<grid(bs_ff), 256>>>(sc.h1.data(), bs_ff);
    ensure(sc.h2, static_cast<size_t>(BS) * D);
    gemm_rm(false, true, BS, D, FF, 1.0f, sc.h1.data(), bl.ffn.w4.data(), 0.0f, sc.h2.data());
    k_add_rowbias<<<grid(bs_dim), 256>>>(sc.h2.data(), bl.ffn.b4.data(), BS, D);
    k_add<<<grid(bs_dim), 256>>>(x.data(), sc.h2.data(), bs_dim);

    ensure(sc.xo, static_cast<size_t>(BS) * D);
    k_rmsnorm<<<grid(BS), 256>>>(x.data(), bl.out_gamma.data(), sc.xo.data(), BS, D);
    x.copy_from_device(sc.xo.data(), static_cast<size_t>(BS) * D);
  }

  // 6 blocks: each = time transformer (over T per band) + freq transformer (over bands per t).
  // x is [T, num_bands, dim] in place.
  void blocks_forward(CudaBuffer& x, int T) const {
    require_supported_frames(T);
    const int n = detail::checked_int_product({T, num_bands, dim},
                                              "Roformer block tensor");
    ensure(sc.xt, n);
    for (int i = 0; i < depth; ++i) {
      k_transpose12<<<grid(n), 256>>>(x.data(), sc.xt.data(), T, num_bands, dim);  // [T,NB,D]->[NB,T,D]
      transformer(sc.xt, num_bands, T, blocks[i][0]);                              // time
      k_transpose12<<<grid(n), 256>>>(sc.xt.data(), x.data(), num_bands, T, dim);  // [NB,T,D]->[T,NB,D]
      transformer(x, T, num_bands, blocks[i][1]);                                  // freq
    }
    CK(cudaDeviceSynchronize());
  }

  // Mask estimator: per band MLP(dim->ff_hidden x mask_depth ->din*2, Tanh) + GLU -> mask [T, band_in].
  void mask_estimator(const CudaBuffer& blk, CudaBuffer& mask_out, int T) const {
    require_supported_frames(T);
    mask_out.allocate(static_cast<size_t>(T) * band_in);
    for (int b = 0; b < num_bands; ++b) {
      int din = dim_in[b];
      ensure(sc.bandin, static_cast<size_t>(T) * dim);
      const int band_elements =
          detail::checked_int_product({T, dim}, "Roformer mask band");
      k_extract_band<<<grid(band_elements), 256>>>(blk.data(), sc.bandin.data(), b,
                                                   num_bands, T, dim);
      const float* in_ptr = sc.bandin.data();
      int in_d = dim;
      for (int l = 0; l <= mask_depth; ++l) {
        int out_d = (l == mask_depth) ? din * 2 : ff_hidden;
        CudaBuffer& dst = (l % 2 == 0) ? sc.mlp_a : sc.mlp_b;
        ensure(dst, static_cast<size_t>(T) * out_d);
        gemm_rm(false, true, T, out_d, in_d, 1.0f, in_ptr, mask[b].w[l].data(), 0.0f, dst.data());
        const int layer_elements =
            detail::checked_int_product({T, out_d}, "Roformer mask layer");
        k_add_rowbias<<<grid(layer_elements), 256>>>(dst.data(), mask[b].b[l].data(),
                                                     T, out_d);
        if (l != mask_depth)
          k_tanh<<<grid(layer_elements), 256>>>(dst.data(), layer_elements);
        in_ptr = dst.data();
        in_d = out_d;
      }
      const float* last = (mask_depth % 2 == 0) ? sc.mlp_a.data() : sc.mlp_b.data();
      const int mask_elements =
          detail::checked_int_product({T, din}, "Roformer mask output");
      k_glu_scatter<<<grid(mask_elements), 256>>>(last, mask_out.data(), off_in[b],
                                                  din, T, band_in);
    }
    CK(cudaDeviceSynchronize());
  }

  // Full forward: stereo planar [2,L] -> stem planar [2,L].
  std::vector<float> forward(const std::vector<float>& stereo, int L) const {
    int T = 0;
    CudaBuffer d_re, d_im;
    stft_stereo(stereo, L, d_re, d_im, T);
    CudaBuffer x;
    band_split_from_spec(d_re, d_im, x, T);  // reuse spectra (no second STFT)
    blocks_forward(x, T);
    CudaBuffer mask_out;
    mask_estimator(x, mask_out, T);
    return apply_mask(mask_out, d_re, d_im, T, L);
  }

  // Chunked overlap-add for long audio (time attention is O(T^2)): 50% Hann
  // overlap, normalize by window sum so single-cover regions recover exactly.
  std::vector<float> forward_chunked(const std::vector<float>& stereo, int L) const {
    if (chunk_size <= 0 || L <= chunk_size) return forward(stereo, L);
    const int cs = chunk_size, hop = cs / 2;
    std::vector<float> win(cs);
    for (int i = 0; i < cs; ++i) win[i] = 0.5f - 0.5f * std::cos(2.0 * M_PI * i / cs);
    std::vector<float> out(static_cast<size_t>(2) * L, 0.0f), wsum(L, 0.0f);
    std::vector<float> chunk(static_cast<size_t>(2) * cs);
    for (std::int64_t start = 0; start < L; start += hop) {
      const std::int64_t remaining = static_cast<std::int64_t>(L) - start;
      int len = static_cast<int>(std::min<std::int64_t>(cs, remaining));
      const bool first = start == 0;
      const bool last = remaining <= cs;
      std::fill(chunk.begin(), chunk.end(), 0.0f);
      for (int c = 0; c < 2; ++c)
        for (int i = 0; i < len; ++i)
          chunk[static_cast<size_t>(c) * cs + i] =
              stereo[static_cast<size_t>(c) * L + static_cast<size_t>(start) + i];
      auto y = forward(chunk, cs);
      // There is no preceding/following chunk at the signal boundaries, so do
      // not fade the uncovered outer half to zero. Interior overlaps retain the
      // complementary periodic-Hann pair whose weights sum to one.
      for (int i = 0; i < len; ++i) {
        const float weight = ((first && i < hop) || (last && i >= hop))
                                 ? 1.0f
                                 : win[i];
        for (int c = 0; c < 2; ++c)
          out[static_cast<size_t>(c) * L + static_cast<size_t>(start) + i] +=
              y[static_cast<size_t>(c) * cs + i] * weight;
        wsum[static_cast<size_t>(start) + i] += weight;
      }
      if (last) break;
    }
    for (int c = 0; c < 2; ++c)
      for (int i = 0; i < L; ++i) out[static_cast<size_t>(c) * L + i] /= std::max(wsum[i], 1e-6f);
    return out;
  }

  // Stage 4: scatter-average complex mask, multiply STFT, zero DC, iSTFT per channel.
  std::vector<float> apply_mask(const CudaBuffer& mask_out, const CudaBuffer& d_re,
                                const CudaBuffer& d_im, int T, int L) const {
    require_supported_frames(T);
    if (sc.d_fi.empty())
      sc.d_fi.copy_from_host(reinterpret_cast<const float*>(freq_indices.data()), freq_indices.size() * 2);
    ensure(sc.sre, static_cast<size_t>(merged) * T); sc.sre.zero();
    ensure(sc.sim, static_cast<size_t>(merged) * T); sc.sim.zero();
    ensure(sc.ore, static_cast<size_t>(2) * nfreq * T);
    ensure(sc.oim, static_cast<size_t>(2) * nfreq * T);
    const int scatter_elements =
        detail::checked_int_product({T, fi_len}, "Roformer frequency map launch");
    const int spectrum_elements =
        detail::checked_int_product({T, merged}, "Roformer merged spectrum");
    k_scatter_mask<<<grid(scatter_elements), 256>>>(mask_out.data(),
        reinterpret_cast<const int64_t*>(sc.d_fi.data()), sc.sre.data(), sc.sim.data(), T, fi_len, band_in);
    k_apply_mask<<<grid(spectrum_elements), 256>>>(d_re.data(), d_im.data(), sc.sre.data(), sc.sim.data(),
        d_num_bands_per_freq.data(), sc.ore.data(), sc.oim.data(), T, merged, nfreq);
    CK(cudaDeviceSynchronize());

    std::vector<float> hre(static_cast<size_t>(2) * nfreq * T), him(hre.size());
    sc.ore.copy_to_host(hre.data(), hre.size());
    sc.oim.copy_to_host(him.data(), him.size());
    detail::require_finite_output(hre, "Roformer masked spectrum real component");
    detail::require_finite_output(him, "Roformer masked spectrum imaginary component");
    std::vector<float> out(static_cast<size_t>(2) * L, 0.0f);
    for (int c = 0; c < 2; ++c) {
      std::vector<float> cre(hre.begin() + static_cast<size_t>(c) * nfreq * T,
                             hre.begin() + static_cast<size_t>(c + 1) * nfreq * T);
      std::vector<float> cim(him.begin() + static_cast<size_t>(c) * nfreq * T,
                             him.begin() + static_cast<size_t>(c + 1) * nfreq * T);
      auto ch = stft->inverse(cre, cim, T, L);
      for (int i = 0; i < L && i < (int)ch.size(); ++i) out[static_cast<size_t>(c) * L + i] = ch[i];
    }
    detail::require_finite_output(out, "Roformer waveform output");
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
  if (!valid()) throw std::runtime_error("Roformer is not initialized");
  p_->require_supported_length(L, false);
  if (L <= 0 || stereo.size() != static_cast<std::size_t>(2) * L) {
    throw std::invalid_argument("Roformer stereo input must have shape [2, L]");
  }
  detail::require_finite_input(stereo, "Roformer stereo input");
  return p_->with_scratch_release([&] {
    CudaBuffer out;
    p_->band_split(stereo, L, out, T);
    std::vector<float> h(static_cast<size_t>(T) * p_->num_bands * p_->dim);
    out.copy_to_host(h.data(), h.size());
    detail::require_finite_output(h, "Roformer band-split output");
    return h;
  });
}

std::vector<float> Roformer::debug_bandsplit_in(const std::vector<float>& stereo, int L, int& T) const {
  if (!valid()) throw std::runtime_error("Roformer is not initialized");
  p_->require_supported_length(L, false);
  if (L <= 0 || stereo.size() != static_cast<std::size_t>(2) * L) {
    throw std::invalid_argument("Roformer stereo input must have shape [2, L]");
  }
  detail::require_finite_input(stereo, "Roformer stereo input");
  return p_->with_scratch_release([&] {
    CudaBuffer bin;
    p_->band_split_in(stereo, L, bin, T);
    std::vector<float> h(static_cast<size_t>(T) * p_->band_in);
    bin.copy_to_host(h.data(), h.size());
    detail::require_finite_output(h, "Roformer band-split input tensor");
    return h;
  });
}

std::vector<int> Roformer::debug_offsets() const {
  if (!valid()) throw std::runtime_error("Roformer is not initialized");
  std::vector<int> v(p_->off_in);
  v.insert(v.end(), p_->dim_in.begin(), p_->dim_in.end());
  return v;
}

std::vector<float> Roformer::debug_blocks(const std::vector<float>& stereo, int L, int& T) const {
  if (!valid()) throw std::runtime_error("Roformer is not initialized");
  p_->require_supported_length(L, false);
  if (L <= 0 || stereo.size() != static_cast<std::size_t>(2) * L) {
    throw std::invalid_argument("Roformer stereo input must have shape [2, L]");
  }
  detail::require_finite_input(stereo, "Roformer stereo input");
  return p_->with_scratch_release([&] {
    CudaBuffer x;
    p_->band_split(stereo, L, x, T);
    p_->blocks_forward(x, T);
    std::vector<float> h(static_cast<size_t>(T) * p_->num_bands * p_->dim);
    x.copy_to_host(h.data(), h.size());
    detail::require_finite_output(h, "Roformer transformer output");
    return h;
  });
}
std::vector<float> Roformer::debug_blocks_from(const std::vector<float>& bandsplit, int T) const {
  if (!valid()) throw std::runtime_error("Roformer is not initialized");
  p_->require_supported_frames(T);
  if (T <= 0 || bandsplit.size() != static_cast<std::size_t>(T) * p_->num_bands * p_->dim) {
    throw std::invalid_argument("Roformer band-split tensor has the wrong shape");
  }
  detail::require_finite_input(bandsplit, "Roformer band-split tensor");
  return p_->with_scratch_release([&] {
    CudaBuffer x;
    x.copy_from_host(bandsplit.data(), bandsplit.size());
    p_->blocks_forward(x, T);
    std::vector<float> h(static_cast<size_t>(T) * p_->num_bands * p_->dim);
    x.copy_to_host(h.data(), h.size());
    detail::require_finite_output(h, "Roformer transformer output");
    return h;
  });
}
std::vector<float> Roformer::debug_mask(const std::vector<float>& blk_ref, int T) const {
  if (!valid()) throw std::runtime_error("Roformer is not initialized");
  p_->require_supported_frames(T);
  if (T <= 0 || blk_ref.size() != static_cast<std::size_t>(T) * p_->num_bands * p_->dim) {
    throw std::invalid_argument("Roformer transformer tensor has the wrong shape");
  }
  detail::require_finite_input(blk_ref, "Roformer transformer tensor");
  return p_->with_scratch_release([&] {
    CudaBuffer x, mask_out;
    x.copy_from_host(blk_ref.data(), blk_ref.size());
    p_->mask_estimator(x, mask_out, T);
    std::vector<float> h(static_cast<size_t>(T) * p_->band_in);
    mask_out.copy_to_host(h.data(), h.size());
    detail::require_finite_output(h, "Roformer mask output");
    return h;
  });
}

std::vector<float> Roformer::separate_stereo(const std::vector<float>& stereo, int L) const {
  if (!valid()) throw std::runtime_error("Roformer is not initialized");
  p_->require_supported_length(L, true);
  if (L <= 0 || stereo.size() != static_cast<size_t>(2) * L) {
    throw std::invalid_argument("Roformer stereo input must have shape [2, L]");
  }
  detail::require_finite_input(stereo, "Roformer stereo input");
  return p_->with_scratch_release([&] {
    auto out = p_->forward_chunked(stereo, L);
    detail::require_finite_output(out, "Roformer stereo output");
    return out;
  });
}

std::vector<float> Roformer::debug_apply(const std::vector<float>& mask,
                                         const std::vector<float>& stereo, int L, int T) const {
  if (!valid()) throw std::runtime_error("Roformer is not initialized");
  const int expected_frames = p_->require_supported_length(L, false);
  p_->require_supported_frames(T);
  if (T != expected_frames ||
      stereo.size() != static_cast<std::size_t>(2) * L ||
      mask.size() != static_cast<std::size_t>(T) * p_->band_in) {
    throw std::invalid_argument("Roformer mask/apply inputs have incompatible shapes");
  }
  detail::require_finite_input(mask, "Roformer mask input");
  detail::require_finite_input(stereo, "Roformer stereo input");
  return p_->with_scratch_release(
      [&] { return p_->apply_from_host_mask(mask, stereo, L, T); });
}

std::vector<float> Roformer::separate_mono(const float* audio, int L, int sr) const {
  if (!valid()) throw std::runtime_error("Roformer is not initialized");
  if (!audio || L <= 0 || sr <= 0) {
    throw std::invalid_argument("Roformer mono input requires audio, L > 0, and sr > 0");
  }
  const int model_length =
      sr == p_->sr ? L
                   : detail::checked_resampled_length(L, sr, p_->sr, "Roformer input");
  p_->require_supported_length(model_length, true);
  detail::require_finite_input(audio, static_cast<std::size_t>(L),
                               "Roformer mono input");
  return p_->with_scratch_release([&] {
  int msr = p_->sr;
  auto r = (sr == msr) ? std::vector<float>(audio, audio + L)
                       : io::resample_linear(audio, L, sr, msr);
  if (r.empty()) throw std::runtime_error("Roformer input resampling failed");
  detail::require_finite_output(r, "Roformer resampled input");
  int L44 = static_cast<int>(r.size());
  std::vector<float> stereo(static_cast<size_t>(2) * L44);
  for (int i = 0; i < L44; ++i) { stereo[i] = r[i]; stereo[L44 + i] = r[i]; }
  auto st = p_->forward_chunked(stereo, L44);
  std::vector<float> mono(L44);
  for (int i = 0; i < L44; ++i) mono[i] = 0.5f * (st[i] + st[L44 + i]);
  if (sr != msr) mono = io::resample_linear(mono.data(), L44, msr, sr);
  if (mono.empty()) throw std::runtime_error("Roformer output resampling failed");
  detail::require_finite_output(mono, "Roformer mono output");
  return mono;
  });
}

}  // namespace voxmutatio::separation
