// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/autograd/tensor.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <unordered_set>
#include <vector>

namespace voxmutatio::autograd {

namespace {

inline void ck(cudaError_t e, const char* f, int l) {
  if (e != cudaSuccess)
    fprintf(stderr, "CUDA %s:%d: %s\n", f, l, cudaGetErrorString(e));
}
#define CK(e) ck((e), __FILE__, __LINE__)

cublasHandle_t cublas() {
  static cublasHandle_t h = [] {
    cublasHandle_t x;
    cublasCreate(&x);
    return x;
  }();
  return h;
}

// Check a kernel launch.
inline void check_launch(const char* name) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess)
    fprintf(stderr, "kernel %s launch: %s\n", name, cudaGetErrorString(e));
}

__global__ void k_add(const float* a, const float* b, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = a[i] + b[i];
}
__global__ void k_mul(const float* a, const float* b, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = a[i] * b[i];
}
// dst += src
__global__ void k_acc(float* dst, const float* src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}
// dst += g * x
__global__ void k_acc_mul(float* dst, const float* g, const float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += g[i] * x[i];
}
// dst[i] += s[0]  (broadcast scalar)
__global__ void k_acc_bcast(float* dst, const float* s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += s[0];
}
// out[0] = sum(a)  (out preset to 0)
__global__ void k_sum(const float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, a[i]);
}

inline int grid(int n) { return (n + 255) / 256; }

// Row-major C[M,N] += alpha * op(A) @ op(B).  (beta applied to C)
// op(A) is [M,K], op(B) is [K,N]. Physical leading dims from row-major storage.
void gemm_rm(bool transA, bool transB, int M, int N, int K, float alpha,
             const float* A, const float* B, float beta, float* C) {
  int lda = transA ? M : K;  // physical ld of A's row-major storage
  int ldb = transB ? K : N;  // physical ld of B's row-major storage
  cublasSgemm(cublas(),
              transB ? CUBLAS_OP_T : CUBLAS_OP_N,
              transA ? CUBLAS_OP_T : CUBLAS_OP_N,
              N, M, K, &alpha,
              B, ldb, A, lda, &beta,
              C, N);
}

std::shared_ptr<Node> make_node(std::vector<int> shape, bool requires_grad) {
  auto nd = std::make_shared<Node>();
  nd->shape = std::move(shape);
  nd->requires_grad = requires_grad;
  nd->data.allocate(static_cast<std::size_t>(nd->numel()));
  return nd;
}

bool any_requires_grad(std::initializer_list<const Tensor*> ts) {
  for (auto* t : ts)
    if (t->n->requires_grad || !t->n->parents.empty()) return true;
  return false;
}

}  // namespace

Tensor Tensor::from_host(const std::vector<float>& v, std::vector<int> shape,
                         bool requires_grad) {
  auto nd = make_node(std::move(shape), requires_grad);
  nd->data.copy_from_host(v.data(), v.size());
  return Tensor(nd);
}

Tensor Tensor::zeros(std::vector<int> shape, bool requires_grad) {
  auto nd = make_node(std::move(shape), requires_grad);
  nd->data.zero();
  return Tensor(nd);
}

std::vector<float> Tensor::to_host() const {
  std::vector<float> out(static_cast<std::size_t>(numel()));
  n->data.copy_to_host(out.data(), out.size());
  return out;
}

std::vector<float> Tensor::grad_to_host() const {
  std::vector<float> out(static_cast<std::size_t>(numel()), 0.0f);
  if (n->grad.size() >= static_cast<std::size_t>(numel()))
    n->grad.copy_to_host(out.data(), out.size());
  return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
  int n = static_cast<int>(a.numel());
  auto out = make_node(a.shape(), any_requires_grad({&a, &b}));
  k_add<<<grid(n), 256>>>(a.n->data.data(), b.n->data.data(), out->data.data(), n);
  check_launch("add");
  out->parents = {a.n, b.n};
  Node* o = out.get();
  Node* pa = a.n.get();
  Node* pb = b.n.get();
  out->backward_fn = [o, pa, pb, n] {
    k_acc<<<grid(n), 256>>>(pa->grad.data(), o->grad.data(), n);
    k_acc<<<grid(n), 256>>>(pb->grad.data(), o->grad.data(), n);
  };
  return Tensor(out);
}

Tensor mul(const Tensor& a, const Tensor& b) {
  int n = static_cast<int>(a.numel());
  auto out = make_node(a.shape(), any_requires_grad({&a, &b}));
  k_mul<<<grid(n), 256>>>(a.n->data.data(), b.n->data.data(), out->data.data(), n);
  check_launch("mul");
  out->parents = {a.n, b.n};
  Node* o = out.get();
  Node* pa = a.n.get();
  Node* pb = b.n.get();
  out->backward_fn = [o, pa, pb, n] {
    // da += grad * b ; db += grad * a
    k_acc_mul<<<grid(n), 256>>>(pa->grad.data(), o->grad.data(), pb->data.data(), n);
    k_acc_mul<<<grid(n), 256>>>(pb->grad.data(), o->grad.data(), pa->data.data(), n);
  };
  return Tensor(out);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  int M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
  auto out = make_node({M, N}, any_requires_grad({&a, &b}));
  // C = A @ B
  gemm_rm(false, false, M, N, K, 1.0f, a.n->data.data(), b.n->data.data(), 0.0f,
          out->data.data());
  out->parents = {a.n, b.n};
  Node* o = out.get();
  Node* pa = a.n.get();
  Node* pb = b.n.get();
  out->backward_fn = [o, pa, pb, M, N, K] {
    // dA[M,K] += dO[M,N] @ B[K,N]^T
    gemm_rm(false, true, M, K, N, 1.0f, o->grad.data(), pb->data.data(), 1.0f,
            pa->grad.data());
    // dB[K,N] += A[M,K]^T @ dO[M,N]
    gemm_rm(true, false, K, N, M, 1.0f, pa->data.data(), o->grad.data(), 1.0f,
            pb->grad.data());
  };
  return Tensor(out);
}

Tensor sum(const Tensor& a) {
  int n = static_cast<int>(a.numel());
  auto out = make_node({1}, any_requires_grad({&a}));
  out->data.zero();
  k_sum<<<grid(n), 256>>>(a.n->data.data(), out->data.data(), n);
  check_launch("sum");
  out->parents = {a.n};
  Node* o = out.get();
  Node* pa = a.n.get();
  out->backward_fn = [o, pa, n] {
    k_acc_bcast<<<grid(n), 256>>>(pa->grad.data(), o->grad.data(), n);
  };
  return Tensor(out);
}

// ==== Phase A1 kernels ====

__global__ void k_add_bias(const float* x, const float* b, float* o, int R, int C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < R * C) o[i] = x[i] + b[i % C];
}
__global__ void k_colsum_acc(float* db, const float* dy, int R, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) {
    float s = 0.0f;
    for (int r = 0; r < R; ++r) s += dy[r * C + c];
    db[c] += s;
  }
}
__global__ void k_relu_f(const float* x, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = fmaxf(0.0f, x[i]);
}
__global__ void k_relu_b(float* dx, const float* dy, const float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dx[i] += (x[i] > 0.0f) ? dy[i] : 0.0f;
}
__global__ void k_leaky_f(const float* x, float* o, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = (x[i] >= 0.0f) ? x[i] : s * x[i];
}
__global__ void k_leaky_b(float* dx, const float* dy, const float* x, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dx[i] += dy[i] * ((x[i] >= 0.0f) ? 1.0f : s);
}
__global__ void k_gelu_f(const float* x, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = x[i] * 0.5f * (1.0f + erff(x[i] * 0.7071067811865476f));
}
__global__ void k_gelu_b(float* dx, const float* dy, const float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    float cdf = 0.5f * (1.0f + erff(v * 0.7071067811865476f));
    float pdf = 0.3989422804014327f * expf(-0.5f * v * v);
    dx[i] += dy[i] * (cdf + v * pdf);
  }
}
__global__ void k_tanh_f(const float* x, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = tanhf(x[i]);
}
__global__ void k_tanh_b(float* dx, const float* dy, const float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dx[i] += dy[i] * (1.0f - y[i] * y[i]);
}
__global__ void k_sig_f(const float* x, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = 1.0f / (1.0f + expf(-x[i]));
}
__global__ void k_sig_b(float* dx, const float* dy, const float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dx[i] += dy[i] * y[i] * (1.0f - y[i]);
}
__global__ void k_softmax_f(const float* x, float* o, int R, int C) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;
  const float* xr = x + r * C;
  float* orow = o + r * C;
  float mx = xr[0];
  for (int c = 1; c < C; ++c) mx = fmaxf(mx, xr[c]);
  float sum = 0.0f;
  for (int c = 0; c < C; ++c) { orow[c] = expf(xr[c] - mx); sum += orow[c]; }
  for (int c = 0; c < C; ++c) orow[c] /= sum;
}
__global__ void k_softmax_b(float* dx, const float* dy, const float* y, int R, int C) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;
  const float* dyr = dy + r * C;
  const float* yr = y + r * C;
  float* dxr = dx + r * C;
  float s = 0.0f;
  for (int c = 0; c < C; ++c) s += dyr[c] * yr[c];
  for (int c = 0; c < C; ++c) dxr[c] += yr[c] * (dyr[c] - s);
}
__global__ void k_ln_f(const float* x, const float* w, const float* b, float* o,
                       float* mean, float* rstd, int R, int D, float eps) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;
  const float* xr = x + r * D;
  float m = 0.0f;
  for (int d = 0; d < D; ++d) m += xr[d];
  m /= D;
  float v = 0.0f;
  for (int d = 0; d < D; ++d) { float t = xr[d] - m; v += t * t; }
  v /= D;
  float rs = rsqrtf(v + eps);
  mean[r] = m;
  rstd[r] = rs;
  float* orow = o + r * D;
  for (int d = 0; d < D; ++d) orow[d] = (xr[d] - m) * rs * w[d] + b[d];
}
__global__ void k_ln_b(float* dx, float* dw, float* db, const float* dy,
                       const float* x, const float* w, const float* mean,
                       const float* rstd, int R, int D) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;
  const float* xr = x + r * D;
  const float* dyr = dy + r * D;
  float* dxr = dx + r * D;
  float m = mean[r], rs = rstd[r];
  float sum1 = 0.0f, sum2 = 0.0f;
  for (int d = 0; d < D; ++d) {
    float xhat = (xr[d] - m) * rs;
    float dxhat = dyr[d] * w[d];
    sum1 += dxhat;
    sum2 += dxhat * xhat;
  }
  for (int d = 0; d < D; ++d) {
    float xhat = (xr[d] - m) * rs;
    float dxhat = dyr[d] * w[d];
    dxr[d] += rs * (dxhat - sum1 / D - xhat * sum2 / D);
    atomicAdd(&dw[d], dyr[d] * xhat);
    atomicAdd(&db[d], dyr[d]);
  }
}

Tensor add_bias(const Tensor& x, const Tensor& b) {
  int C = x.shape().back();
  int R = static_cast<int>(x.numel()) / C;
  int n = R * C;
  auto out = make_node(x.shape(), any_requires_grad({&x, &b}));
  k_add_bias<<<grid(n), 256>>>(x.n->data.data(), b.n->data.data(), out->data.data(), R, C);
  check_launch("add_bias");
  out->parents = {x.n, b.n};
  Node* o = out.get(); Node* px = x.n.get(); Node* pb = b.n.get();
  out->backward_fn = [o, px, pb, R, C, n] {
    k_acc<<<grid(n), 256>>>(px->grad.data(), o->grad.data(), n);
    k_colsum_acc<<<grid(C), 256>>>(pb->grad.data(), o->grad.data(), R, C);
  };
  return Tensor(out);
}

Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b) {
  int M = x.shape()[0], K = x.shape()[1], N = w.shape()[0];
  bool has_b = static_cast<bool>(b.n);
  bool rg = any_requires_grad({&x, &w}) ||
            (has_b && (b.n->requires_grad || !b.n->parents.empty()));
  auto out = make_node({M, N}, rg);
  gemm_rm(false, true, M, N, K, 1.0f, x.n->data.data(), w.n->data.data(), 0.0f, out->data.data());
  if (has_b) {
    k_add_bias<<<grid(M * N), 256>>>(out->data.data(), b.n->data.data(), out->data.data(), M, N);
    check_launch("linear_bias");
    out->parents = {x.n, w.n, b.n};
  } else {
    out->parents = {x.n, w.n};
  }
  Node* o = out.get(); Node* px = x.n.get(); Node* pw = w.n.get();
  Node* pb = has_b ? b.n.get() : nullptr;
  out->backward_fn = [o, px, pw, pb, M, N, K] {
    gemm_rm(false, false, M, K, N, 1.0f, o->grad.data(), pw->data.data(), 1.0f, px->grad.data());
    gemm_rm(true, false, N, K, M, 1.0f, o->grad.data(), px->data.data(), 1.0f, pw->grad.data());
    if (pb) k_colsum_acc<<<grid(N), 256>>>(pb->grad.data(), o->grad.data(), M, N);
  };
  return Tensor(out);
}

template <typename Fwd, typename Bwd>
static Tensor act_op(const Tensor& x, Fwd fwd, Bwd bwd, bool use_output) {
  int n = static_cast<int>(x.numel());
  auto out = make_node(x.shape(), any_requires_grad({&x}));
  fwd(x.n->data.data(), out->data.data(), n);
  out->parents = {x.n};
  Node* o = out.get(); Node* px = x.n.get();
  out->backward_fn = [o, px, n, bwd, use_output] {
    const float* ref = use_output ? o->data.data() : px->data.data();
    bwd(px->grad.data(), o->grad.data(), ref, n);
  };
  return Tensor(out);
}

Tensor relu(const Tensor& x) {
  return act_op(x,
      [](const float* a, float* o, int n) { k_relu_f<<<grid(n), 256>>>(a, o, n); },
      [](float* dx, const float* dy, const float* r, int n) { k_relu_b<<<grid(n), 256>>>(dx, dy, r, n); },
      false);
}
Tensor leaky_relu(const Tensor& x, float slope) {
  return act_op(x,
      [slope](const float* a, float* o, int n) { k_leaky_f<<<grid(n), 256>>>(a, o, slope, n); },
      [slope](float* dx, const float* dy, const float* r, int n) { k_leaky_b<<<grid(n), 256>>>(dx, dy, r, slope, n); },
      false);
}
Tensor gelu(const Tensor& x) {
  return act_op(x,
      [](const float* a, float* o, int n) { k_gelu_f<<<grid(n), 256>>>(a, o, n); },
      [](float* dx, const float* dy, const float* r, int n) { k_gelu_b<<<grid(n), 256>>>(dx, dy, r, n); },
      false);
}
Tensor tanh_op(const Tensor& x) {
  return act_op(x,
      [](const float* a, float* o, int n) { k_tanh_f<<<grid(n), 256>>>(a, o, n); },
      [](float* dx, const float* dy, const float* y, int n) { k_tanh_b<<<grid(n), 256>>>(dx, dy, y, n); },
      true);
}
Tensor sigmoid(const Tensor& x) {
  return act_op(x,
      [](const float* a, float* o, int n) { k_sig_f<<<grid(n), 256>>>(a, o, n); },
      [](float* dx, const float* dy, const float* y, int n) { k_sig_b<<<grid(n), 256>>>(dx, dy, y, n); },
      true);
}

Tensor softmax_rows(const Tensor& x, int rows, int cols) {
  auto out = make_node(x.shape(), any_requires_grad({&x}));
  k_softmax_f<<<grid(rows), 256>>>(x.n->data.data(), out->data.data(), rows, cols);
  check_launch("softmax");
  out->parents = {x.n};
  Node* o = out.get(); Node* px = x.n.get();
  out->backward_fn = [o, px, rows, cols] {
    k_softmax_b<<<grid(rows), 256>>>(px->grad.data(), o->grad.data(), o->data.data(), rows, cols);
  };
  return Tensor(out);
}

Tensor layer_norm(const Tensor& x, const Tensor& w, const Tensor& b,
                  int rows, int dim, float eps) {
  auto out = make_node(x.shape(), any_requires_grad({&x, &w, &b}));
  auto mean = std::make_shared<core::CudaBuffer>(rows);
  auto rstd = std::make_shared<core::CudaBuffer>(rows);
  k_ln_f<<<grid(rows), 256>>>(x.n->data.data(), w.n->data.data(), b.n->data.data(),
                              out->data.data(), mean->data(), rstd->data(), rows, dim, eps);
  check_launch("layer_norm");
  out->parents = {x.n, w.n, b.n};
  Node* o = out.get(); Node* px = x.n.get(); Node* pw = w.n.get(); Node* pb = b.n.get();
  out->backward_fn = [o, px, pw, pb, mean, rstd, rows, dim] {
    k_ln_b<<<grid(rows), 256>>>(px->grad.data(), pw->grad.data(), pb->grad.data(),
                                o->grad.data(), px->data.data(), pw->data.data(),
                                mean->data(), rstd->data(), rows, dim);
  };
  return Tensor(out);
}

// ==== Phase A1 convolutions ====

__global__ void k_conv1d_f(const float* x, const float* w, const float* b, float* o,
    int Cin, int L, int Cout, int K, int stride, int pad, int dil, int groups, int Lout) {
  int ot = blockIdx.x * blockDim.x + threadIdx.x, co = blockIdx.y;
  if (ot >= Lout || co >= Cout) return;
  int ipg = Cin / groups, opg = Cout / groups, g = co / opg, cin0 = g * ipg;
  float s = b ? b[co] : 0.0f;
  for (int cl = 0; cl < ipg; ++cl) {
    const float* xc = x + (cin0 + cl) * L;
    const float* wc = w + (co * ipg + cl) * K;
    for (int k = 0; k < K; ++k) {
      int t = ot * stride - pad + k * dil;
      if (t >= 0 && t < L) s += xc[t] * wc[k];
    }
  }
  o[co * Lout + ot] = s;
}
__global__ void k_conv1d_dx(float* dx, const float* dout, const float* w,
    int Cin, int L, int Cout, int K, int stride, int pad, int dil, int groups, int Lout) {
  int ot = blockIdx.x * blockDim.x + threadIdx.x, co = blockIdx.y;
  if (ot >= Lout || co >= Cout) return;
  int ipg = Cin / groups, opg = Cout / groups, g = co / opg, cin0 = g * ipg;
  float go = dout[co * Lout + ot];
  const float* wc = w + co * ipg * K;
  for (int cl = 0; cl < ipg; ++cl) {
    for (int k = 0; k < K; ++k) {
      int t = ot * stride - pad + k * dil;
      if (t >= 0 && t < L) atomicAdd(&dx[(cin0 + cl) * L + t], go * wc[cl * K + k]);
    }
  }
}
__global__ void k_conv1d_dw(float* dw, const float* dout, const float* x,
    int Cin, int L, int Cout, int K, int stride, int pad, int dil, int groups, int Lout) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int ipg = Cin / groups, total = Cout * ipg * K;
  if (idx >= total) return;
  int k = idx % K, cl = (idx / K) % ipg, co = idx / (K * ipg);
  int opg = Cout / groups, g = co / opg, cin0 = g * ipg;
  const float* xc = x + (cin0 + cl) * L;
  float s = 0.0f;
  for (int ot = 0; ot < Lout; ++ot) {
    int t = ot * stride - pad + k * dil;
    if (t >= 0 && t < L) s += dout[co * Lout + ot] * xc[t];
  }
  dw[idx] += s;
}
__global__ void k_conv_db(float* db, const float* dout, int Cout, int Lout) {
  int co = blockIdx.x * blockDim.x + threadIdx.x;
  if (co >= Cout) return;
  float s = 0.0f;
  for (int ot = 0; ot < Lout; ++ot) s += dout[co * Lout + ot];
  db[co] += s;
}

__global__ void k_convt_f(const float* x, const float* w, const float* b, float* o,
    int Cin, int L, int Cout, int K, int stride, int pad, int Lout) {
  int ot = blockIdx.x * blockDim.x + threadIdx.x, co = blockIdx.y;
  if (ot >= Lout || co >= Cout) return;
  float s = b ? b[co] : 0.0f;
  for (int k = 0; k < K; ++k) {
    int num = ot + pad - k;
    if (num % stride != 0) continue;
    int t = num / stride;
    if (t < 0 || t >= L) continue;
    for (int ci = 0; ci < Cin; ++ci) s += x[ci * L + t] * w[(ci * Cout + co) * K + k];
  }
  o[co * Lout + ot] = s;
}
__global__ void k_convt_dx(float* dx, const float* dout, const float* w,
    int Cin, int L, int Cout, int K, int stride, int pad, int Lout) {
  int t = blockIdx.x * blockDim.x + threadIdx.x, ci = blockIdx.y;
  if (t >= L || ci >= Cin) return;
  float s = 0.0f;
  for (int k = 0; k < K; ++k) {
    int ot = t * stride - pad + k;
    if (ot < 0 || ot >= Lout) continue;
    for (int co = 0; co < Cout; ++co) s += dout[co * Lout + ot] * w[(ci * Cout + co) * K + k];
  }
  dx[ci * L + t] += s;
}
__global__ void k_convt_dw(float* dw, const float* dout, const float* x,
    int Cin, int L, int Cout, int K, int stride, int pad, int Lout) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x, total = Cin * Cout * K;
  if (idx >= total) return;
  int k = idx % K, co = (idx / K) % Cout, ci = idx / (K * Cout);
  float s = 0.0f;
  for (int t = 0; t < L; ++t) {
    int ot = t * stride - pad + k;
    if (ot >= 0 && ot < Lout) s += dout[co * Lout + ot] * x[ci * L + t];
  }
  dw[idx] += s;
}

Tensor conv1d(const Tensor& x, const Tensor& w, const Tensor& b,
              int Cin, int L, int Cout, int K, int stride, int pad,
              int dilation, int groups) {
  int Lout = (L + 2 * pad - dilation * (K - 1) - 1) / stride + 1;
  bool has_b = static_cast<bool>(b.n);
  bool rg = any_requires_grad({&x, &w}) ||
            (has_b && (b.n->requires_grad || !b.n->parents.empty()));
  auto out = make_node({Cout, Lout}, rg);
  dim3 gf(grid(Lout), Cout);
  k_conv1d_f<<<gf, 256>>>(x.n->data.data(), w.n->data.data(),
                          has_b ? b.n->data.data() : nullptr, out->data.data(),
                          Cin, L, Cout, K, stride, pad, dilation, groups, Lout);
  check_launch("conv1d");
  if (has_b) out->parents = {x.n, w.n, b.n}; else out->parents = {x.n, w.n};
  Node* o = out.get(); Node* px = x.n.get(); Node* pw = w.n.get();
  Node* pb = has_b ? b.n.get() : nullptr;
  int ipg = Cin / groups;
  out->backward_fn = [o, px, pw, pb, Cin, L, Cout, K, stride, pad, dilation, groups, Lout, ipg] {
    dim3 gd(grid(Lout), Cout);
    k_conv1d_dx<<<gd, 256>>>(px->grad.data(), o->grad.data(), pw->data.data(),
                             Cin, L, Cout, K, stride, pad, dilation, groups, Lout);
    k_conv1d_dw<<<grid(Cout * ipg * K), 256>>>(pw->grad.data(), o->grad.data(),
                             px->data.data(), Cin, L, Cout, K, stride, pad, dilation, groups, Lout);
    if (pb) k_conv_db<<<grid(Cout), 256>>>(pb->grad.data(), o->grad.data(), Cout, Lout);
  };
  return Tensor(out);
}

Tensor conv_transpose1d(const Tensor& x, const Tensor& w, const Tensor& b,
                        int Cin, int L, int Cout, int K, int stride, int pad) {
  int Lout = (L - 1) * stride - 2 * pad + K;
  bool has_b = static_cast<bool>(b.n);
  bool rg = any_requires_grad({&x, &w}) ||
            (has_b && (b.n->requires_grad || !b.n->parents.empty()));
  auto out = make_node({Cout, Lout}, rg);
  dim3 gf(grid(Lout), Cout);
  k_convt_f<<<gf, 256>>>(x.n->data.data(), w.n->data.data(),
                         has_b ? b.n->data.data() : nullptr, out->data.data(),
                         Cin, L, Cout, K, stride, pad, Lout);
  check_launch("conv_transpose1d");
  if (has_b) out->parents = {x.n, w.n, b.n}; else out->parents = {x.n, w.n};
  Node* o = out.get(); Node* px = x.n.get(); Node* pw = w.n.get();
  Node* pb = has_b ? b.n.get() : nullptr;
  out->backward_fn = [o, px, pw, pb, Cin, L, Cout, K, stride, pad, Lout] {
    dim3 gd(grid(L), Cin);
    k_convt_dx<<<gd, 256>>>(px->grad.data(), o->grad.data(), pw->data.data(),
                            Cin, L, Cout, K, stride, pad, Lout);
    k_convt_dw<<<grid(Cin * Cout * K), 256>>>(pw->grad.data(), o->grad.data(),
                            px->data.data(), Cin, L, Cout, K, stride, pad, Lout);
    if (pb) k_conv_db<<<grid(Cout), 256>>>(pb->grad.data(), o->grad.data(), Cout, Lout);
  };
  return Tensor(out);
}

// ==== Phase A3 shape/gather primitives ====

__global__ void k_transpose(const float* x, float* y, int R, int C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < R * C) { int r = i / C, c = i % C; y[c * R + r] = x[i]; }
}
__global__ void k_transpose_acc(float* dx, const float* dy, int R, int C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < R * C) { int r = i / C, c = i % C; dx[i] += dy[c * R + r]; }
}
__global__ void k_embed_f(const float* table, const int* idx, float* o, int T, int D) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < T * D) { int t = i / D, d = i % D; o[i] = table[idx[t] * D + d]; }
}
__global__ void k_embed_b(float* dtable, const int* idx, const float* dy, int T, int D) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < T * D) { int t = i / D, d = i % D; atomicAdd(&dtable[idx[t] * D + d], dy[i]); }
}
__global__ void k_scale(const float* x, float* o, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = x[i] * s;
}
__global__ void k_acc_scaled(float* dst, const float* src, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i] * s;
}

Tensor transpose2d(const Tensor& x, int rows, int cols) {
  auto out = make_node({cols, rows}, any_requires_grad({&x}));
  k_transpose<<<grid(rows * cols), 256>>>(x.n->data.data(), out->data.data(), rows, cols);
  check_launch("transpose2d");
  out->parents = {x.n};
  Node* o = out.get(); Node* px = x.n.get();
  out->backward_fn = [o, px, rows, cols] {
    k_transpose_acc<<<grid(rows * cols), 256>>>(px->grad.data(), o->grad.data(), rows, cols);
  };
  return Tensor(out);
}

Tensor concat_rows(const Tensor& a, const Tensor& b, int cols) {
  int Ra = static_cast<int>(a.numel()) / cols;
  int Rb = static_cast<int>(b.numel()) / cols;
  auto out = make_node({Ra + Rb, cols}, any_requires_grad({&a, &b}));
  cudaMemcpy(out->data.data(), a.n->data.data(), Ra * cols * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(out->data.data() + Ra * cols, b.n->data.data(), Rb * cols * sizeof(float), cudaMemcpyDeviceToDevice);
  out->parents = {a.n, b.n};
  Node* o = out.get(); Node* pa = a.n.get(); Node* pb = b.n.get();
  int na = Ra * cols, nb = Rb * cols;
  out->backward_fn = [o, pa, pb, na, nb] {
    k_acc<<<grid(na), 256>>>(pa->grad.data(), o->grad.data(), na);
    k_acc<<<grid(nb), 256>>>(pb->grad.data(), o->grad.data() + na, nb);
  };
  return Tensor(out);
}

Tensor slice_rows(const Tensor& x, int start, int count, int cols) {
  auto out = make_node({count, cols}, any_requires_grad({&x}));
  cudaMemcpy(out->data.data(), x.n->data.data() + start * cols,
             count * cols * sizeof(float), cudaMemcpyDeviceToDevice);
  out->parents = {x.n};
  Node* o = out.get(); Node* px = x.n.get();
  int n = count * cols, off = start * cols;
  out->backward_fn = [o, px, n, off] {
    k_acc<<<grid(n), 256>>>(px->grad.data() + off, o->grad.data(), n);
  };
  return Tensor(out);
}

Tensor embedding(const Tensor& table, const std::vector<int>& idx, int dim) {
  int T = static_cast<int>(idx.size());
  auto out = make_node({T, dim}, any_requires_grad({&table}));
  int* d_idx_raw = nullptr;
  cudaMalloc(&d_idx_raw, T * sizeof(int));
  cudaMemcpy(d_idx_raw, idx.data(), T * sizeof(int), cudaMemcpyHostToDevice);
  std::shared_ptr<int> d_idx(d_idx_raw, [](int* p) { if (p) cudaFree(p); });
  k_embed_f<<<grid(T * dim), 256>>>(table.n->data.data(), d_idx.get(), out->data.data(), T, dim);
  check_launch("embedding");
  out->parents = {table.n};
  Node* o = out.get(); Node* pt = table.n.get();
  out->backward_fn = [o, pt, d_idx, T, dim] {
    k_embed_b<<<grid(T * dim), 256>>>(pt->grad.data(), d_idx.get(), o->grad.data(), T, dim);
  };
  return Tensor(out);
}

Tensor scale(const Tensor& x, float s) {
  int n = static_cast<int>(x.numel());
  auto out = make_node(x.shape(), any_requires_grad({&x}));
  k_scale<<<grid(n), 256>>>(x.n->data.data(), out->data.data(), s, n);
  check_launch("scale");
  out->parents = {x.n};
  Node* o = out.get(); Node* px = x.n.get();
  out->backward_fn = [o, px, s, n] {
    k_acc_scaled<<<grid(n), 256>>>(px->grad.data(), o->grad.data(), s, n);
  };
  return Tensor(out);
}

// ==== Elementwise math for mel loss ====
__global__ void k_sqrt_f(const float* x, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = sqrtf(fmaxf(x[i], 0.0f));
}
__global__ void k_sqrt_b(float* dx, const float* dy, const float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dx[i] += dy[i] * 0.5f / fmaxf(y[i], 1e-12f);
}
__global__ void k_log_f(const float* x, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = logf(fmaxf(x[i], 1e-12f));
}
__global__ void k_log_b(float* dx, const float* dy, const float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dx[i] += dy[i] / fmaxf(x[i], 1e-12f);
}
__global__ void k_abs_f(const float* x, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = fabsf(x[i]);
}
__global__ void k_abs_b(float* dx, const float* dy, const float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dx[i] += dy[i] * ((x[i] >= 0.0f) ? 1.0f : -1.0f);
}
__global__ void k_frame_f(const float* x, float* o, int T, int nfft, int hop) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < T * nfft) { int t = i / nfft, n = i % nfft; o[i] = x[t * hop + n]; }
}
__global__ void k_frame_b(float* dx, const float* dout, int T, int nfft, int hop) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < T * nfft) { int t = i / nfft, n = i % nfft; atomicAdd(&dx[t * hop + n], dout[i]); }
}

Tensor sqrt_op(const Tensor& x) {
  return act_op(x,
      [](const float* a, float* o, int n) { k_sqrt_f<<<grid(n), 256>>>(a, o, n); },
      [](float* dx, const float* dy, const float* y, int n) { k_sqrt_b<<<grid(n), 256>>>(dx, dy, y, n); },
      true);
}
Tensor log_op(const Tensor& x) {
  return act_op(x,
      [](const float* a, float* o, int n) { k_log_f<<<grid(n), 256>>>(a, o, n); },
      [](float* dx, const float* dy, const float* r, int n) { k_log_b<<<grid(n), 256>>>(dx, dy, r, n); },
      false);
}
Tensor abs_op(const Tensor& x) {
  return act_op(x,
      [](const float* a, float* o, int n) { k_abs_f<<<grid(n), 256>>>(a, o, n); },
      [](float* dx, const float* dy, const float* r, int n) { k_abs_b<<<grid(n), 256>>>(dx, dy, r, n); },
      false);
}

Tensor frame(const Tensor& x, int T, int n_fft, int hop) {
  auto out = make_node({T, n_fft}, any_requires_grad({&x}));
  k_frame_f<<<grid(T * n_fft), 256>>>(x.n->data.data(), out->data.data(), T, n_fft, hop);
  check_launch("frame");
  out->parents = {x.n};
  Node* o = out.get(); Node* px = x.n.get();
  out->backward_fn = [o, px, T, n_fft, hop] {
    k_frame_b<<<grid(T * n_fft), 256>>>(px->grad.data(), o->grad.data(), T, n_fft, hop);
  };
  return Tensor(out);
}

void backward(const Tensor& loss) {
  // Topological order via DFS over parents.
  std::vector<Node*> topo;
  std::unordered_set<Node*> visited;
  std::function<void(const std::shared_ptr<Node>&)> dfs =
      [&](const std::shared_ptr<Node>& node) {
        if (!node || visited.count(node.get())) return;
        visited.insert(node.get());
        for (auto& p : node->parents) dfs(p);
        topo.push_back(node.get());
      };
  dfs(loss.n);

  for (Node* nd : topo) nd->ensure_grad();

  // Seed loss grad = 1.
  std::vector<float> ones(static_cast<std::size_t>(loss.numel()), 1.0f);
  loss.n->grad.copy_from_host(ones.data(), ones.size());

  for (auto it = topo.rbegin(); it != topo.rend(); ++it)
    if ((*it)->backward_fn) (*it)->backward_fn();
  CK(cudaDeviceSynchronize());
}

// ==== AdamW ====

__global__ void k_adamw(float* p, const float* g, float* m, float* v, float lr,
                        float b1, float b2, float eps, float wd, float bc1,
                        float bc2, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float grad = g[i];
  float mi = b1 * m[i] + (1.0f - b1) * grad;
  float vi = b2 * v[i] + (1.0f - b2) * grad * grad;
  m[i] = mi;
  v[i] = vi;
  float mh = mi / bc1;
  float vh = vi / bc2;
  p[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * p[i]);
}

AdamW::AdamW(std::vector<Tensor> params, float lr, float beta1, float beta2,
             float eps, float weight_decay)
    : params_(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), wd_(weight_decay) {
  m_.resize(params_.size());
  v_.resize(params_.size());
  for (std::size_t i = 0; i < params_.size(); ++i) {
    std::size_t n = static_cast<std::size_t>(params_[i].numel());
    m_[i].allocate(n);
    m_[i].zero();
    v_[i].allocate(n);
    v_[i].zero();
  }
}

void AdamW::zero_grad() {
  for (auto& p : params_) p.n->ensure_grad();
}

void AdamW::step() {
  ++t_;
  float bc1 = 1.0f - std::pow(beta1_, t_);
  float bc2 = 1.0f - std::pow(beta2_, t_);
  for (std::size_t i = 0; i < params_.size(); ++i) {
    int n = static_cast<int>(params_[i].numel());
    if (params_[i].n->grad.size() < static_cast<std::size_t>(n)) continue;
    k_adamw<<<grid(n), 256>>>(params_[i].n->data.data(), params_[i].n->grad.data(),
                              m_[i].data(), v_[i].data(), lr_, beta1_, beta2_,
                              eps_, wd_, bc1, bc2, n);
  }
  CK(cudaDeviceSynchronize());
}

}  // namespace voxmutatio::autograd
