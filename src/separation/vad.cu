// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Silero VAD v5 pure C++/CUDA runner (reimplemented from official MIT model).

#include "voxmutatio/separation/vad.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "voxmutatio/core/cuda_buffer.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/io/safetensors.h"
#include "validation.h"

namespace voxmutatio::separation {
namespace {

using core::CudaBuffer;

[[noreturn]] void throw_cuda(const char* operation, cudaError_t error) {
  throw std::runtime_error(std::string("VAD CUDA failure in ") + operation + ": " +
                           cudaGetErrorString(error));
}
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) \
  throw_cuda(#call, e_); } while (0)

void ck_blas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("VAD cuBLAS failure in ") + operation +
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
void gemm_rm(bool tA, bool tB, int M, int N, int K, float alpha,
             const float* A, const float* B, float beta, float* C) {
  int lda = tA ? M : K, ldb = tB ? K : N;
  ck_blas(cublasSgemm(cublas(), tB ? CUBLAS_OP_T : CUBLAS_OP_N,
                      tA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, ldb,
                      A, lda, &beta, C, N),
          "cublasSgemm");
}

constexpr int kChunk = 512, kCtx = 64, kWin = 640, kNfft = 256, kHop = 128;
constexpr int kFreq = 129, kFrames = 4, kHid = 128;

int checked_chunk_count(int samples) {
  if (samples <= 0) throw std::invalid_argument("VAD sample count must be positive");
  const int chunks = samples / kChunk;
  if (chunks > 0) {
    // C*129*4 is the widest int-sized launch. Encoder and recurrent launches
    // use at most C*512 elements.
    detail::checked_int_product({chunks, kFreq, kFrames}, "VAD chunk tensor");
  }
  return chunks;
}

// STFT magnitude: mag[c,b,f] = sqrt(re^2+im^2), re/im from conv of window with basis.
// win [C, kWin]; basis [258, kNfft]; out [C, kFreq, kFrames].
__global__ void k_stft(const float* win, const float* basis, float* mag, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= C * kFreq * kFrames) return;
  int f = idx % kFrames, b = (idx / kFrames) % kFreq, c = idx / (kFrames * kFreq);
  const float* w = win + static_cast<size_t>(c) * kWin + f * kHop;
  const float* br = basis + static_cast<size_t>(b) * kNfft;
  const float* bi = basis + static_cast<size_t>(kFreq + b) * kNfft;
  float re = 0.0f, im = 0.0f;
  for (int k = 0; k < kNfft; ++k) { re += w[k] * br[k]; im += w[k] * bi[k]; }
  mag[(static_cast<size_t>(c) * kFreq + b) * kFrames + f] = sqrtf(re * re + im * im);
}

// Conv1d(Cin->Cout, k=3, pad=1, stride) + ReLU, batched over C chunks.
// in [C, Cin, Fin]; w [Cout,Cin,3]; b [Cout]; out [C, Cout, Fout].
__global__ void k_conv_relu(const float* in, const float* w, const float* bias, float* out,
                            int C, int Cin, int Fin, int Cout, int Fout, int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= C * Cout * Fout) return;
  int of = idx % Fout, oc = (idx / Fout) % Cout, c = idx / (Fout * Cout);
  float acc = bias[oc];
  const float* inc = in + static_cast<size_t>(c) * Cin * Fin;
  const float* wc = w + static_cast<size_t>(oc) * Cin * 3;
  for (int ic = 0; ic < Cin; ++ic)
    for (int k = 0; k < 3; ++k) {
      int inf = of * stride - 1 + k;
      if (inf >= 0 && inf < Fin) acc += wc[ic * 3 + k] * inc[ic * Fin + inf];
    }
  out[(static_cast<size_t>(c) * Cout + oc) * Fout + of] = fmaxf(acc, 0.0f);
}

// One LSTM timestep (hidden kHid): gates = xproj + Whh@h_prev (order i,f,g,o).
__global__ void k_lstm_cell(const float* xproj, const float* hproj, const float* c_prev,
                            float* h_out, float* c_out, int H) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) return;
  float i = 1.0f / (1.0f + expf(-(xproj[j] + hproj[j])));
  float f = 1.0f / (1.0f + expf(-(xproj[H + j] + hproj[H + j])));
  float g = tanhf(xproj[2 * H + j] + hproj[2 * H + j]);
  float o = 1.0f / (1.0f + expf(-(xproj[3 * H + j] + hproj[3 * H + j])));
  float cc = f * c_prev[j] + i * g;
  c_out[j] = cc;
  h_out[j] = o * tanhf(cc);
}
__global__ void k_add_rowbias(float* x, const float* b, int N, int D) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N * D) x[i] += b[i % D];
}
// prob[c] = sigmoid(bias + sum_i relu(h[c,i]) * w[i])
__global__ void k_classify(const float* h, const float* w, float bias, float* prob, int C, int H) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const float* hc = h + static_cast<size_t>(c) * H;
  float acc = bias;
  for (int i = 0; i < H; ++i) acc += fmaxf(hc[i], 0.0f) * w[i];
  prob[c] = 1.0f / (1.0f + expf(-acc));
}

}  // namespace

struct Vad::Impl {
  bool ok = false;
  CudaBuffer basis;                                   // [258, 256]
  CudaBuffer ew[4], eb[4];                             // encoder conv weights/bias
  int e_cin[4] = {kFreq, 128, 64, 64}, e_cout[4] = {128, 64, 64, 128}, e_stride[4] = {1, 2, 2, 1};
  CudaBuffer wih, whh, bsum;                           // LSTM (bsum = bias_ih + bias_hh)
  CudaBuffer cls_w; float cls_b = 0.0f;               // classifier

  static bool up(const io::SafetensorsLoader& L, const std::string& n, CudaBuffer& b,
                 std::initializer_list<int64_t> shape) {
    return detail::upload_checked_f32(L, n, shape, b, "vad");
  }

  explicit Impl(const std::string& dir, const std::string& model) {
    io::SafetensorsLoader L;
    if (!L.load(dir + "/" + model + ".safetensors")) { fprintf(stderr, "vad: cannot load %s\n", model.c_str()); return; }
    bool g = true;
    g &= up(L, "stft.forward_basis_buffer", basis, {258, 1, kNfft});
    for (int i = 0; i < 4; ++i) {
      std::string p = "encoder." + std::to_string(i) + ".reparam_conv.";
      g &= up(L, p + "weight", ew[i], {e_cout[i], e_cin[i], 3});
      g &= up(L, p + "bias", eb[i], {e_cout[i]});
    }
    g &= up(L, "decoder.rnn.weight_ih", wih, {4 * kHid, kHid});
    g &= up(L, "decoder.rnn.weight_hh", whh, {4 * kHid, kHid});
    const float* bi = nullptr;
    const float* bh = nullptr;
    std::size_t bi_count = 0;
    std::size_t bh_count = 0;
    if (detail::checked_f32_tensor(L, "decoder.rnn.bias_ih", {4 * kHid}, "vad",
                                   bi, bi_count) &&
        detail::checked_f32_tensor(L, "decoder.rnn.bias_hh", {4 * kHid}, "vad",
                                   bh, bh_count)) {
      std::vector<float> bs(4 * kHid);
      for (int i = 0; i < 4 * kHid; ++i) {
        bs[i] = bi[i] + bh[i];
        if (!std::isfinite(bs[i])) g = false;
      }
      if (g) bsum.copy_from_host(bs.data(), bs.size());
    } else {
      g = false;
    }
    g &= up(L, "decoder.decoder.2.weight", cls_w, {1, kHid, 1});
    const float* cls_bias = nullptr;
    std::size_t cls_bias_count = 0;
    if (detail::checked_f32_tensor(L, "decoder.decoder.2.bias", {1}, "vad",
                                   cls_bias, cls_bias_count)) {
      cls_b = cls_bias[0];
    } else {
      g = false;
    }
    ok = g;
  }

  std::vector<float> probs(const float* audio, int n) const {
    int C = checked_chunk_count(n);
    if (C <= 0) return {};
    const int stft_elements =
        detail::checked_int_product({C, kFreq, kFrames}, "VAD STFT tensor");
    const int recurrent_elements =
        detail::checked_int_product({C, 4, kHid}, "VAD recurrent tensor");
    // Build padded windows [C, kWin] on host (context + chunk + reflect pad).
    std::vector<float> win(static_cast<size_t>(C) * kWin, 0.0f);
    for (int c = 0; c < C; ++c) {
      float* w = win.data() + static_cast<size_t>(c) * kWin;
      for (int i = 0; i < kCtx; ++i) { int s = c * kChunk - kCtx + i; w[i] = (s >= 0) ? audio[s] : 0.0f; }
      for (int i = 0; i < kChunk; ++i) w[kCtx + i] = audio[c * kChunk + i];
      for (int j = 0; j < kCtx; ++j) w[576 + j] = w[574 - j];  // ReflectionPad1d[0,64]
    }
    CudaBuffer d_win; d_win.copy_from_host(win.data(), win.size());

    CudaBuffer mag; mag.allocate(static_cast<size_t>(C) * kFreq * kFrames);
    k_stft<<<grid(stft_elements), 256>>>(d_win.data(), basis.data(), mag.data(), C);

    CudaBuffer cur = std::move(mag);
    int Fin = kFrames;
    for (int i = 0; i < 4; ++i) {
      int Fout = (Fin + 2 - 3) / e_stride[i] + 1;
      CudaBuffer nxt; nxt.allocate(static_cast<size_t>(C) * e_cout[i] * Fout);
      const int conv_elements = detail::checked_int_product(
          {C, e_cout[i], Fout}, "VAD encoder tensor");
      k_conv_relu<<<grid(conv_elements), 256>>>(cur.data(), ew[i].data(), eb[i].data(),
          nxt.data(), C, e_cin[i], Fin, e_cout[i], Fout, e_stride[i]);
      cur = std::move(nxt); Fin = Fout;
    }
    // cur is [C, 128, 1] = encoder features [C, 128].
    CudaBuffer xproj; xproj.allocate(static_cast<size_t>(C) * 4 * kHid);
    gemm_rm(false, true, C, 4 * kHid, kHid, 1.0f, cur.data(), wih.data(), 0.0f, xproj.data());
    k_add_rowbias<<<grid(recurrent_elements), 256>>>(xproj.data(), bsum.data(), C,
                                                     4 * kHid);

    CudaBuffer hseq; hseq.allocate(static_cast<size_t>(C) * kHid);
    CudaBuffer cbuf, hproj, zero;
    cbuf.allocate(kHid); cbuf.zero();
    hproj.allocate(4 * kHid);
    zero.allocate(kHid); zero.zero();
    for (int c = 0; c < C; ++c) {
      const float* hprev = c ? hseq.data() + (c - 1) * kHid : zero.data();
      gemm_rm(false, true, 1, 4 * kHid, kHid, 1.0f, hprev, whh.data(), 0.0f, hproj.data());
      k_lstm_cell<<<grid(kHid), 256>>>(xproj.data() + static_cast<size_t>(c) * 4 * kHid,
          hproj.data(), cbuf.data(), hseq.data() + static_cast<size_t>(c) * kHid, cbuf.data(), kHid);
    }
    CudaBuffer dprob; dprob.allocate(C);
    k_classify<<<grid(C), 256>>>(hseq.data(), cls_w.data(), cls_b, dprob.data(), C, kHid);
    CK(cudaDeviceSynchronize());
    std::vector<float> out(C);
    dprob.copy_to_host(out.data(), C);
    detail::require_finite_output(out, "VAD probability output");
    for (float probability : out) {
      if (probability < 0.0f || probability > 1.0f) {
        throw std::runtime_error("VAD produced a probability outside [0, 1]");
      }
    }
    return out;
  }
};

Vad::Vad(const std::string& dir, const std::string& model)
    : p_(std::make_unique<Impl>(dir, model)) {}
Vad::~Vad() = default;
bool Vad::valid() const { return p_ && p_->ok; }
int Vad::sample_rate() const { return 16000; }

std::vector<float> Vad::probs(const float* audio16k, int n) const {
  if (!valid()) throw std::runtime_error("VAD is not initialized");
  if (!audio16k || n <= 0) throw std::invalid_argument("VAD input is invalid");
  checked_chunk_count(n);
  detail::require_finite_input(audio16k, static_cast<std::size_t>(n), "VAD input");
  return p_->probs(audio16k, n);
}

std::vector<std::pair<int, int>> Vad::segments(const float* audio, int n, int sr, float thresh,
                                               int min_speech_ms, int min_silence_ms, int pad_ms) const {
  if (!valid()) throw std::runtime_error("VAD is not initialized");
  if (!audio || n <= 0 || sr <= 0 || !std::isfinite(thresh) || thresh < 0.0f ||
      thresh > 1.0f || min_speech_ms < 0 || min_silence_ms < 0 || pad_ms < 0 ||
      min_speech_ms > 60'000 || min_silence_ms > 60'000 || pad_ms > 60'000) {
    throw std::invalid_argument("VAD segmentation parameters are invalid");
  }
  checked_chunk_count(n);
  const int model_length =
      sr == 16000 ? n : detail::checked_resampled_length(n, sr, 16000, "VAD input");
  checked_chunk_count(model_length);
  detail::require_finite_input(audio, static_cast<std::size_t>(n),
                               "VAD segmentation input");
  std::vector<float> a16 = (sr == 16000) ? std::vector<float>(audio, audio + n)
                                         : io::resample_linear(audio, n, sr, 16000);
  if (a16.empty()) throw std::runtime_error("VAD resampling failed");
  detail::require_finite_output(a16, "VAD resampled input");
  auto pr = probs(a16.data(), static_cast<int>(a16.size()));
  // Hysteresis over 32 ms chunks (512 samples @ 16k), then map to input sr.
  const float lo = std::max(0.0f, thresh - 0.15f);
  const int min_speech = min_speech_ms * 16, min_sil = min_silence_ms * 16, pad = pad_ms * 16;
  std::vector<std::pair<int, int>> segs;  // in 16k samples
  bool on = false; int start = 0, last_speech = 0;
  for (int c = 0; c < (int)pr.size(); ++c) {
    int s = c * kChunk;
    if (!on && pr[c] >= thresh) { on = true; start = s; last_speech = s + kChunk; }
    else if (on) {
      if (pr[c] >= lo) last_speech = s + kChunk;
      if (s - last_speech >= min_sil) {
        if (last_speech - start >= min_speech) segs.push_back({start, last_speech});
        on = false;
      }
    }
  }
  if (on && last_speech - start >= min_speech) segs.push_back({start, last_speech});
  // pad + clamp + map 16k -> input sr.
  double r = static_cast<double>(sr) / 16000.0;
  std::vector<std::pair<int, int>> out;
  for (auto& s : segs) {
    int a = std::max(0, s.first - pad), b = std::min((int)a16.size(), s.second + pad);
    out.push_back({(int)(a * r), std::min(n, (int)(b * r))});
  }
  return out;
}

}  // namespace voxmutatio::separation
