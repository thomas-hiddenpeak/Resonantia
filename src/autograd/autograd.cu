// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/autograd/tensor.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace voxmutatio::autograd {

namespace {

inline void ck(cudaError_t e, const char* f, int l) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA ") + f + ":" + std::to_string(l) +
                             ": " + cudaGetErrorString(e));
  }
}
#define CK(e) ck((e), __FILE__, __LINE__)

inline void ck_blas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

cublasHandle_t cublas() {
  static cublasHandle_t h = [] {
    cublasHandle_t x = nullptr;
    ck_blas(cublasCreate(&x), "cublasCreate");
    return x;
  }();
  return h;
}

// Check a kernel launch.
inline void check_launch(const char* name) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("kernel ") + name + " launch failed: " +
                             cudaGetErrorString(e));
  }
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
  ck_blas(cublasSgemm(cublas(),
                      transB ? CUBLAS_OP_T : CUBLAS_OP_N,
                      transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                      N, M, K, &alpha,
                      B, ldb, A, lda, &beta,
                      C, N),
          "cublasSgemm");
}

// Reuses device scratch buffers across ops/steps so im2col does not cudaMalloc/
// cudaFree per conv (cudaFree syncs the device). Buffers return to the free list
// via the shared_ptr deleter; capacity grows monotonically to steady state.
class ScratchPool {
 public:
  static ScratchPool& get() { static ScratchPool p; return p; }
  std::shared_ptr<core::CudaBuffer> acquire(std::size_t n) {
    std::unique_ptr<core::CudaBuffer> buf;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!free_.empty()) {
        auto best = free_.end();
        for (auto it = free_.begin(); it != free_.end(); ++it) {
          if ((*it)->size() >= n &&
              (best == free_.end() || (*it)->size() < (*best)->size()))
            best = it;
        }
        if (best == free_.end())
          best = std::max_element(free_.begin(), free_.end(),
              [](const auto& a, const auto& b) { return a->size() < b->size(); });
        buf = std::move(*best);
        free_.erase(best);
      }
    }
    if (!buf) buf = std::make_unique<core::CudaBuffer>();
    if (buf->size() < n) buf->allocate(n);
    core::CudaBuffer* raw = buf.release();
    return std::shared_ptr<core::CudaBuffer>(raw, [this](core::CudaBuffer* p) {
      std::lock_guard<std::mutex> lock(mu_);
      free_.emplace_back(p);
    });
  }
 private:
  std::mutex mu_;
  std::vector<std::unique_ptr<core::CudaBuffer>> free_;
};

void require_tensor(const Tensor& t, const char* name) {
  if (!t.n) throw std::invalid_argument(std::string(name) + " is an empty Tensor");
}

int checked_numel_int(const Tensor& t, const char* name) {
  require_tensor(t, name);
  const int64_t n = t.numel();
  if (n > std::numeric_limits<int>::max())
    throw std::length_error(std::string(name) + " exceeds CUDA kernel index range");
  return static_cast<int>(n);
}

void require_rank(const Tensor& t, std::size_t rank, const char* name) {
  require_tensor(t, name);
  if (t.shape().size() != rank)
    throw std::invalid_argument(std::string(name) + " has the wrong rank");
}

void require_same_shape(const Tensor& a, const Tensor& b, const char* operation) {
  require_tensor(a, "lhs");
  require_tensor(b, "rhs");
  if (a.shape() != b.shape())
    throw std::invalid_argument(std::string(operation) + " requires equal shapes");
}

void require_flat_size(const Tensor& t, int64_t expected, const char* name) {
  require_tensor(t, name);
  if (expected <= 0 || t.numel() != expected)
    throw std::invalid_argument(std::string(name) + " size does not match declared dimensions");
}

// Cache only dead graph buffers, never graph edges or closures. Operations use
// the legacy default CUDA stream, so reuse on that stream preserves queued work
// order without cudaFree's device synchronization. Each host thread/device cache
// retains at most 256 MiB and 4096 buffers; live tensors are not included in this cap.
class NodeBufferCache {
 public:
  explicit NodeBufferCache(int device) : device_(device) {}
  ~NodeBufferCache() {
    int previous = device_;
    const bool restore = cleanup_status(cudaGetDevice(&previous));
    cleanup_status(cudaSetDevice(device_));
    for (auto& entry : entries_) entry = {};
    if (restore) cleanup_status(cudaSetDevice(previous));
  }
  core::CudaBuffer acquire(std::size_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& entry : entries_) {
      if (entry.size() == n) {
        bytes_ -= entry.size() * sizeof(float);
        return std::move(entry);
      }
    }
    return {};
  }
  void release(core::CudaBuffer buffer) noexcept {
    {
      std::lock_guard<std::mutex> lock(mu_);
      const auto size = buffer.size() * sizeof(float);
      if (size && size <= kLimit - bytes_) {
        for (auto& entry : entries_) {
          if (entry.empty()) {
            entry = std::move(buffer);
            bytes_ += size;
            return;
          }
        }
      }
    }
    // A tensor can be destroyed on another host thread/device. Free on the
    // owning device and restore the caller's selection when the cache is full.
    int previous = device_;
    const bool restore = cleanup_status(cudaGetDevice(&previous));
    cleanup_status(cudaSetDevice(device_));
    buffer = {};
    if (restore) cleanup_status(cudaSetDevice(previous));
  }
 private:
  static bool cleanup_status(cudaError_t status) noexcept {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "Autograd cache cleanup: %s\n", cudaGetErrorString(status));
    return false;
  }
  static constexpr std::size_t kLimit = 256 * 1024 * 1024;
  int device_;
  std::size_t bytes_ = 0;
  std::mutex mu_;
  std::array<core::CudaBuffer, 4096> entries_;
};

std::shared_ptr<NodeBufferCache> node_cache() {
  int device = 0;
  CK(cudaGetDevice(&device));
  static thread_local std::unordered_map<int, std::shared_ptr<NodeBufferCache>> caches;
  auto& cache = caches[device];
  if (!cache) cache = std::make_shared<NodeBufferCache>(device);
  return cache;
}

std::shared_ptr<Node> make_node(std::vector<int> shape, bool requires_grad) {
  if (shape.empty()) throw std::invalid_argument("tensor shape cannot be empty");
  Node staged;
  staged.shape = std::move(shape);
  const int64_t n = staged.numel();
  if (n > std::numeric_limits<int>::max())
    throw std::length_error("tensor exceeds CUDA kernel index range");
  auto cache = node_cache();
  auto nd = std::shared_ptr<Node>(new Node, [owner = cache](Node* node) noexcept {
    std::unique_ptr<Node> owned(node);
    // Drop graph ownership outside the cache lock: parents may release buffers.
    owned->backward_fn = {};
    owned->parents.clear();
    owner->release(std::move(owned->data));
    owner->release(std::move(owned->grad));
  });
  nd->shape = std::move(staged.shape);
  nd->requires_grad = requires_grad;
  nd->data = cache->acquire(static_cast<std::size_t>(n));
  nd->data.resize(static_cast<std::size_t>(n));
  // Keep grad absent until backward: optimizers distinguish absent from zero.
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
  if (static_cast<int64_t>(v.size()) != nd->numel())
    throw std::invalid_argument("Tensor::from_host data size does not match shape");
  nd->data.copy_from_host(v.data(), v.size());
  return Tensor(nd);
}

Tensor Tensor::zeros(std::vector<int> shape, bool requires_grad) {
  auto nd = make_node(std::move(shape), requires_grad);
  nd->data.zero();
  return Tensor(nd);
}

std::vector<float> Tensor::to_host() const {
  require_tensor(*this, "tensor");
  std::vector<float> out(static_cast<std::size_t>(numel()));
  n->data.copy_to_host(out.data(), out.size());
  return out;
}

std::vector<float> Tensor::grad_to_host() const {
  require_tensor(*this, "tensor");
  std::vector<float> out(static_cast<std::size_t>(numel()), 0.0f);
  if (n->grad.size() >= static_cast<std::size_t>(numel()))
    n->grad.copy_to_host(out.data(), out.size());
  return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
  require_same_shape(a, b, "add");
  int n = checked_numel_int(a, "add input");
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
  require_same_shape(a, b, "mul");
  int n = checked_numel_int(a, "mul input");
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
  require_rank(a, 2, "matmul lhs");
  require_rank(b, 2, "matmul rhs");
  if (a.shape()[1] != b.shape()[0])
    throw std::invalid_argument("matmul inner dimensions do not match");
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
  int n = checked_numel_int(a, "sum input");
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
__global__ void k_linear_single_row(const float* x, const float* w, float* out,
                                    int outputs, int inner) {
  const int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=outputs)return;
  const float* weight=w+row*inner;
  float a=0.f,b=0.f,c=0.f,d=0.f;
  for(int j=0;j<inner;j+=4) {
    a=__fmaf_rn(x[j],weight[j],a);
    b=__fmaf_rn(x[j+1],weight[j+1],b);
    c=__fmaf_rn(x[j+2],weight[j+2],c);
    d=__fmaf_rn(x[j+3],weight[j+3],d);
  }
  out[row]=__fadd_rn(__fadd_rn(a,b),__fadd_rn(c,d));
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
  if (i < n) dx[i] += dy[i] * ((x[i] > 0.0f) ? 1.0f : s);
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
__device__ float finish_fp32_lanes(const float* lanes, int count) {
  if (count < 4) {
    float sum = lanes[0];
    for (int i = 1; i < count; ++i) sum = __fadd_rn(sum, lanes[i]);
    return sum;
  }
  return __fadd_rn(__fadd_rn(lanes[0], lanes[2]),
                   __fadd_rn(lanes[1], lanes[3]));
}
__global__ void k_softmax_f(const float* x, float* o, int R, int C) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;
  const float* xr = x + r * C;
  float* orow = o + r * C;
  float mx = xr[0];
  for (int c = 1; c < C; ++c) mx = fmaxf(mx, xr[c]);
  float lanes[4]{};
  #pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    for (int c = lane; c < C; c += 4) {
      orow[c] = expf(xr[c] - mx);
      lanes[lane] = __fadd_rn(lanes[lane], orow[c]);
    }
  }
  const float reciprocal = 1.0f / finish_fp32_lanes(lanes, C);
  for (int c = 0; c < C; ++c) orow[c] *= reciprocal;
}
__global__ void k_softmax_b(float* dx, const float* dy, const float* y, int R, int C) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;
  const float* dyr = dy + r * C;
  const float* yr = y + r * C;
  float* dxr = dx + r * C;
  float lanes[4]{};
  #pragma unroll
  for (int lane = 0; lane < 4; ++lane)
    for (int c = lane; c < C; c += 4)
      lanes[lane] = __fadd_rn(lanes[lane], __fmul_rn(dyr[c], yr[c]));
  const float s = finish_fp32_lanes(lanes, C);
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
  checked_numel_int(x, "add_bias input");
  require_rank(b, 1, "add_bias bias");
  int C = x.shape().back();
  if (b.numel() != C) throw std::invalid_argument("add_bias bias size mismatch");
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
  require_rank(x, 2, "linear input");
  require_rank(w, 2, "linear weight");
  if (x.shape()[1] != w.shape()[1])
    throw std::invalid_argument("linear input/weight dimensions do not match");
  int M = x.shape()[0], K = x.shape()[1], N = w.shape()[0];
  bool has_b = static_cast<bool>(b.n);
  if (has_b) {
    require_rank(b, 1, "linear bias");
    if (b.numel() != N) throw std::invalid_argument("linear bias size mismatch");
  }
  bool rg = any_requires_grad({&x, &w}) ||
            (has_b && (b.n->requires_grad || !b.n->parents.empty()));
  auto out = make_node({M, N}, rg);
  if (M==1 && K>=16 && K%4==0) {
    // Match the frozen single-vector conditioning reduction, including FMA.
    k_linear_single_row<<<grid(N),256>>>(x.n->data.data(),w.n->data.data(),out->data.data(),N,K);
    check_launch("linear single-row reduction");
  } else {
    gemm_rm(false, true, M, N, K, 1.0f, x.n->data.data(), w.n->data.data(), 0.0f, out->data.data());
  }
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
  int n = checked_numel_int(x, "activation input");
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
  if (rows <= 0 || cols <= 0 || x.numel() != static_cast<int64_t>(rows) * cols)
    throw std::invalid_argument("softmax_rows dimensions do not match input");
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
  if (rows <= 0 || dim <= 0 || !std::isfinite(eps) || eps <= 0.0f)
    throw std::invalid_argument("invalid layer_norm parameters");
  require_flat_size(x, static_cast<int64_t>(rows) * dim, "layer_norm input");
  require_flat_size(w, dim, "layer_norm weight");
  require_flat_size(b, dim, "layer_norm bias");
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
  float s = 0.0f;
  for (int cl = 0; cl < ipg; ++cl) {
    const float* xc = x + (cin0 + cl) * L;
    const float* wc = w + (co * ipg + cl) * K;
    for (int k = 0; k < K; ++k) {
      int t = ot * stride - pad + k * dil;
      if (t >= 0 && t < L) s += xc[t] * wc[k];
    }
  }
  o[co * Lout + ot] = b ? s + b[co] : s;
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

// im2col for conv1d (groups==1): col[(c*K+k), ot] = x[c, ot*stride - pad + k*dil].
__global__ void k_im2col_1d(const float* x, float* col, int Cin, int L, int K,
    int stride, int pad, int dil, int Lout) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= Cin * K * Lout) return;
  int ot = i % Lout, r = i / Lout, k = r % K, c = r / K;
  int t = ot * stride - pad + k * dil;
  col[r * Lout + ot] = (t >= 0 && t < L) ? x[c * L + t] : 0.0f;
}
// col2im (scatter dcol back into dx).
__global__ void k_col2im_1d(float* dx, const float* dcol, int Cin, int L, int K,
    int stride, int pad, int dil, int Lout) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= Cin * K * Lout) return;
  int ot = i % Lout, r = i / Lout, k = r % K, c = r / K;
  int t = ot * stride - pad + k * dil;
  if (t >= 0 && t < L) atomicAdd(&dx[c * L + t], dcol[r * Lout + ot]);
}
// Broadcast add bias over columns: o[c, n] += b[c].
__global__ void k_bias_rows(float* o, const float* b, int C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < C * N) o[i] += b[i / N];
}

// Accumulate input channels in ascending order with one FP32 FMA per channel.
// A GEMM-selected reduction tree changes near-zero decoder activations enough to
// choose different Leaky-ReLU derivatives. Tiles reuse weights/input on device.
__global__ void k_convt_columns(const float* x, const float* w, float* columns,
                                int Cin, int L, int rows, int time_tiles) {
  __shared__ float weights[16][17];
  __shared__ float input[16][16];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row_start = (blockIdx.x / time_tiles) * 16;
  const int time_start = (blockIdx.x % time_tiles) * 16;
  const int row = row_start + ty, t = time_start + tx;
  float sum = 0.0f;
  for (int64_t base = 0; base < Cin; base += 16) {
    const int64_t channel = base + ty;
    const int weight_row = row_start + tx;
    weights[tx][ty] = channel < Cin && weight_row < rows
        ? w[channel * rows + weight_row] : 0.0f;
    input[ty][tx] = channel < Cin && t < L ? x[channel * L + t] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 16; ++k)
      sum = __fmaf_rn(weights[ty][k], input[k][tx], sum);
    __syncthreads();
  }
  if (row < rows && t < L) columns[row * L + t] = sum;
}

__global__ void k_convt_from_columns(const float* columns, const float* b, float* o,
    int L, int Cout, int K, int stride, int pad, int Lout) {
  int ot = blockIdx.x * blockDim.x + threadIdx.x, co = blockIdx.y;
  if (ot >= Lout || co >= Cout) return;
  float s = 0.0f;
  for (int k = 0; k < K; ++k) {
    int num = ot + pad - k;
    if (num % stride != 0) continue;
    int t = num / stride;
    if (t < 0 || t >= L) continue;
    s += columns[(co*K+k)*L+t];
  }
  o[co * Lout + ot] = b ? s + b[co] : s;
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

// The frozen two-thread CPU reference selects its channels-last convolution
// path above 20480 input elements (batch=1). Its dot product visits kernel taps
// before input channels and starts from bias. Reproduce that arithmetic on CUDA
// so backend-dependent reduction trees cannot change near-zero activation signs.
// Transposed convolutions visit reversed taps, with one output phase per block;
// this avoids multiplying the inserted stride zeros while preserving FMA order.
template <bool Transposed>
__global__ void k_conv1d_channels_last(const float* x, const float* w,
    const float* b, float* out, int Cin, int L, int Cout, int K, int stride,
    int pad, int dilation, int Lout, int time_tiles, int phases) {
  __shared__ float weights[16][16];
  __shared__ float input[16][16];
  const int tx=threadIdx.x, ty=threadIdx.y;
  const int tile=blockIdx.x % time_tiles;
  const int phase=(blockIdx.x / time_tiles) % phases;
  const int co=(blockIdx.x / (time_tiles*phases))*16+ty;
  const int64_t ot=Transposed ? (static_cast<int64_t>(tile)*16+tx)*stride+phase
                              : (static_cast<int64_t>(tile)*16+tx)*phases+phase;
  float acc=co<Cout && b ? b[co] : 0.f;
  const int residue=Transposed ? (static_cast<int64_t>(phase)+pad)%stride : 0;
  const int taps=Transposed ? (residue<K ? (K-1-residue)/stride+1 : 0) : K;
  for(int tap=0;tap<taps;++tap) {
    const int k=Transposed ? residue+(taps-1-tap)*stride : tap;
    const int64_t source=Transposed ? (ot+pad-k)/stride :
        ((ot/phases)*stride-pad+static_cast<int64_t>(k)*dilation)*phases+phase;
    for(int64_t base=0;base<Cin;base+=16) {
      const int64_t wc=base+tx, xc=base+ty;
      weights[ty][tx]=co<Cout && wc<Cin
          ? w[Transposed ? (wc*Cout+co)*K+k : (static_cast<int64_t>(co)*Cin+wc)*K+k] : 0.f;
      input[ty][tx]=xc<Cin && ot<Lout && source>=0 && source<L ? x[xc*L+source] : 0.f;
      __syncthreads();
      #pragma unroll
      for(int c=0;c<16;++c)
        if(base+c<Cin)acc=__fmaf_rn(weights[ty][c],input[c][tx],acc);
      __syncthreads();
    }
  }
  if(co<Cout && ot<Lout)out[static_cast<int64_t>(co)*Lout+ot]=acc;
}

template <bool Transposed>
void conv1d_channels_last(const float* x, const float* w, const float* b,
    float* out, int Cin, int L, int Cout, int K, int stride, int pad,
    int dilation, int Lout, int plane_width=1) {
  const int phases=Transposed ? std::min(stride,Lout) : plane_width;
  const int time_tiles=Transposed ? ((Lout-1)/stride)/16+1 : (Lout/plane_width-1)/16+1;
  const int rows=(Cout-1)/16+1;
  k_conv1d_channels_last<Transposed><<<rows*time_tiles*phases,dim3(16,16)>>>(
      x,w,b,out,Cin,L,Cout,K,stride,pad,dilation,Lout,time_tiles,phases);
  check_launch("conv1d channels-last accumulation");
}

// The frozen aarch64 small-convolution reference combines consecutive FP32
// dot products in blocks of 640. Keep both the within-block FMA order and the
// separately rounded block sum; bias is the initial accumulated value. This
// tiled implementation reuses the existing im2col workspace on the GPU.
__global__ void k_conv_columns_blocked(const float* w, const float* columns,
    const float* bias, float* out, int M, int N, int K, int column_tiles) {
  __shared__ float weights[16][16];
  __shared__ float input[16][16];
  const int tx=threadIdx.x, ty=threadIdx.y;
  const int row=(blockIdx.x/column_tiles)*16+ty;
  const int col=(blockIdx.x%column_tiles)*16+tx;
  float total=row<M && bias ? bias[row] : 0.f;
  for(int64_t chunk=0;chunk<K;chunk+=640) {
    float partial=0.f;
    const int64_t end=min(chunk+640,static_cast<int64_t>(K));
    for(int64_t base=chunk;base<end;base+=16) {
      weights[ty][tx]=row<M && base+tx<K ? w[static_cast<int64_t>(row)*K+base+tx] : 0.f;
      input[ty][tx]=col<N && base+ty<K ? columns[(base+ty)*N+col] : 0.f;
      __syncthreads();
      #pragma unroll
      for(int k=0;k<16;++k)
        if(base+k<end)partial=__fmaf_rn(weights[ty][k],input[k][tx],partial);
      __syncthreads();
    }
    total=__fadd_rn(total,partial);
  }
  if(row<M && col<N)out[static_cast<int64_t>(row)*N+col]=total;
}

void conv_columns_blocked(const float* w, const float* columns,
    const float* bias, float* out, int M, int N, int K) {
  const int column_tiles=(N-1)/16+1;
  k_conv_columns_blocked<<<((M-1)/16+1)*column_tiles,dim3(16,16)>>>(
      w,columns,bias,out,M,N,K,column_tiles);
  check_launch("convolution blocked accumulation");
}

Tensor conv1d(const Tensor& x, const Tensor& w, const Tensor& b,
              int Cin, int L, int Cout, int K, int stride, int pad,
              int dilation, int groups) {
  if (Cin <= 0 || L <= 0 || Cout <= 0 || K <= 0 || stride <= 0 || pad < 0 ||
      dilation <= 0 || groups <= 0 || Cin % groups != 0 || Cout % groups != 0)
    throw std::invalid_argument("invalid conv1d parameters");
  require_flat_size(x, static_cast<int64_t>(Cin) * L, "conv1d input");
  require_flat_size(w, static_cast<int64_t>(Cout) * (Cin / groups) * K, "conv1d weight");
  bool has_b = static_cast<bool>(b.n);
  if (has_b) require_flat_size(b, Cout, "conv1d bias");
  const int64_t numerator = static_cast<int64_t>(L) + 2LL * pad -
                            static_cast<int64_t>(dilation) * (K - 1) - 1;
  if (numerator < 0) throw std::invalid_argument("conv1d kernel exceeds padded input");
  const int64_t lout64 = numerator / stride + 1;
  if (lout64 <= 0 || lout64 > std::numeric_limits<int>::max())
    throw std::length_error("conv1d output length is invalid");
  int Lout = static_cast<int>(lout64);
  bool rg = any_requires_grad({&x, &w}) ||
            (has_b && (b.n->requires_grad || !b.n->parents.empty()));
  auto out = make_node({Cout, Lout}, rg);
  Node* o = out.get(); Node* px = x.n.get(); Node* pw = w.n.get();
  Node* pb = has_b ? b.n.get() : nullptr;

  if (groups == 1) {
    // Reference-ordered FP32 convolution; backward retains cuBLAS GEMM.
    const int rows = Cin * K;
    if (static_cast<int64_t>(Cin)*L > 20480) {
      conv1d_channels_last<false>(px->data.data(),pw->data.data(),
          has_b?pb->data.data():nullptr,o->data.data(),Cin,L,Cout,K,stride,pad,dilation,Lout);
    } else {
      auto col = ScratchPool::get().acquire(static_cast<std::size_t>(rows) * Lout);
      k_im2col_1d<<<grid(rows * Lout), 256>>>(px->data.data(), col->data(), Cin, L, K,
                                              stride, pad, dilation, Lout);
      conv_columns_blocked(pw->data.data(),col->data(),has_b?pb->data.data():nullptr,
          o->data.data(),Cout,Lout,rows);
      check_launch("conv1d_im2col");
    }
    if (has_b) out->parents = {x.n, w.n, b.n}; else out->parents = {x.n, w.n};
    out->backward_fn = [o, px, pw, pb, Cin, L, Cout, K, stride, pad, dilation, Lout, rows] {
      // Recompute im2col during backward rather than retaining every convolution's
      // workspace for the lifetime of the graph. This bounds forward memory by the
      // largest convolution workspace instead of their sum.
      auto col = ScratchPool::get().acquire(static_cast<std::size_t>(rows) * Lout);
      k_im2col_1d<<<grid(rows * Lout), 256>>>(px->data.data(), col->data(), Cin, L, K,
                                              stride, pad, dilation, Lout);
      check_launch("conv1d_backward_im2col");
      // dW += dout[Cout,Lout] @ col^T[Lout, Cin*K]
      gemm_rm(false, true, Cout, rows, Lout, 1.0f, o->grad.data(), col->data(), 1.0f, pw->grad.data());
      // dcol = W^T[Cin*K,Cout] @ dout[Cout,Lout]; scatter into dx
      auto dcol = ScratchPool::get().acquire(static_cast<std::size_t>(rows) * Lout);
      gemm_rm(true, false, rows, Lout, Cout, 1.0f, pw->data.data(), o->grad.data(), 0.0f, dcol->data());
      k_col2im_1d<<<grid(rows * Lout), 256>>>(px->grad.data(), dcol->data(), Cin, L, K,
                                              stride, pad, dilation, Lout);
      if (pb) k_conv_db<<<grid(Cout), 256>>>(pb->grad.data(), o->grad.data(), Cout, Lout);
    };
    return Tensor(out);
  }

  // Grouped convolution: naive kernels.
  dim3 gf(grid(Lout), Cout);
  k_conv1d_f<<<gf, 256>>>(px->data.data(), pw->data.data(),
                          has_b ? pb->data.data() : nullptr, o->data.data(),
                          Cin, L, Cout, K, stride, pad, dilation, groups, Lout);
  check_launch("conv1d");
  if (has_b) out->parents = {x.n, w.n, b.n}; else out->parents = {x.n, w.n};
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
  if (Cin <= 0 || L <= 0 || Cout <= 0 || K <= 0 || stride <= 0 || pad < 0)
    throw std::invalid_argument("invalid conv_transpose1d parameters");
  require_flat_size(x, static_cast<int64_t>(Cin) * L, "conv_transpose1d input");
  require_flat_size(w, static_cast<int64_t>(Cin) * Cout * K, "conv_transpose1d weight");
  bool has_b = static_cast<bool>(b.n);
  if (has_b) require_flat_size(b, Cout, "conv_transpose1d bias");
  const int64_t lout64 = static_cast<int64_t>(L - 1) * stride - 2LL * pad + K;
  if (lout64 <= 0 || lout64 > std::numeric_limits<int>::max())
    throw std::length_error("conv_transpose1d output length is invalid");
  int Lout = static_cast<int>(lout64);
  const int64_t column_count = static_cast<int64_t>(Cout) * K * L;
  if (column_count > std::numeric_limits<int>::max())
    throw std::length_error("conv_transpose1d workspace exceeds CUDA index range");
  bool rg = any_requires_grad({&x, &w}) ||
            (has_b && (b.n->requires_grad || !b.n->parents.empty()));
  auto out = make_node({Cout, Lout}, rg);
  dim3 gf(grid(Lout), Cout);
  if (static_cast<int64_t>(Cin)*L > 20480) {
    conv1d_channels_last<true>(x.n->data.data(),w.n->data.data(),
        has_b?b.n->data.data():nullptr,out->data.data(),Cin,L,Cout,K,stride,pad,1,Lout);
  } else {
    // The small-input reference uses channel reduction, overlap-add, then bias.
    // The workspace is returned immediately; backward does not retain it.
    auto columns=ScratchPool::get().acquire(static_cast<std::size_t>(column_count));
    const int time_tiles = (L - 1) / 16 + 1;
    const int row_tiles = (Cout * K - 1) / 16 + 1;
    k_convt_columns<<<time_tiles * row_tiles, dim3(16, 16)>>>(
        x.n->data.data(), w.n->data.data(), columns->data(), Cin, L, Cout*K, time_tiles);
    check_launch("conv_transpose1d channel reduction");
    k_convt_from_columns<<<gf,256>>>(columns->data(),
        has_b?b.n->data.data():nullptr,out->data.data(),L,Cout,K,stride,pad,Lout);
  }
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
  if (rows <= 0 || cols <= 0)
    throw std::invalid_argument("transpose2d dimensions must be positive");
  require_flat_size(x, static_cast<int64_t>(rows) * cols, "transpose2d input");
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
  if (cols <= 0 || a.numel() % cols != 0 || b.numel() % cols != 0)
    throw std::invalid_argument("concat_rows inputs do not match column count");
  int Ra = static_cast<int>(a.numel()) / cols;
  int Rb = static_cast<int>(b.numel()) / cols;
  auto out = make_node({Ra + Rb, cols}, any_requires_grad({&a, &b}));
  CK(cudaMemcpy(out->data.data(), a.n->data.data(), Ra * cols * sizeof(float),
                cudaMemcpyDeviceToDevice));
  CK(cudaMemcpy(out->data.data() + Ra * cols, b.n->data.data(), Rb * cols * sizeof(float),
                cudaMemcpyDeviceToDevice));
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
  if (cols <= 0 || start < 0 || count <= 0 || x.numel() % cols != 0)
    throw std::invalid_argument("invalid slice_rows parameters");
  const int64_t rows = x.numel() / cols;
  if (static_cast<int64_t>(start) + count > rows)
    throw std::out_of_range("slice_rows exceeds input");
  auto out = make_node({count, cols}, any_requires_grad({&x}));
  CK(cudaMemcpy(out->data.data(), x.n->data.data() + start * cols,
                count * cols * sizeof(float), cudaMemcpyDeviceToDevice));
  out->parents = {x.n};
  Node* o = out.get(); Node* px = x.n.get();
  int n = count * cols, off = start * cols;
  out->backward_fn = [o, px, n, off] {
    k_acc<<<grid(n), 256>>>(px->grad.data() + off, o->grad.data(), n);
  };
  return Tensor(out);
}

Tensor embedding(const Tensor& table, const std::vector<int>& idx, int dim) {
  require_rank(table, 2, "embedding table");
  if (dim <= 0 || table.shape()[1] != dim || idx.empty())
    throw std::invalid_argument("invalid embedding dimensions");
  const int vocab = table.shape()[0];
  for (int value : idx)
    if (value < 0 || value >= vocab) throw std::out_of_range("embedding index out of range");
  int T = static_cast<int>(idx.size());
  auto out = make_node({T, dim}, any_requires_grad({&table}));
  int* d_idx_raw = nullptr;
  CK(cudaMalloc(&d_idx_raw, T * sizeof(int)));
  CK(cudaMemcpy(d_idx_raw, idx.data(), T * sizeof(int), cudaMemcpyHostToDevice));
  std::shared_ptr<int> d_idx(d_idx_raw, [](int* p) {
    if (!p) return;
    cudaError_t e = cudaFree(p);
    if (e != cudaSuccess)
      fprintf(stderr, "autograd embedding cudaFree failed: %s\n", cudaGetErrorString(e));
  });
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
  if (!std::isfinite(s)) throw std::invalid_argument("scale must be finite");
  int n = checked_numel_int(x, "scale input");
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
  // The clamped branch (including the singular endpoint) uses zero subgradient.
  if (i < n && y[i] > 0.0f) dx[i] += dy[i] * 0.5f / y[i];
}
__global__ void k_log_f(const float* x, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = logf(fmaxf(x[i], 1e-12f));
}
__global__ void k_log_b(float* dx, const float* dy, const float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && x[i] > 1e-12f) dx[i] += dy[i] / x[i];
}
__global__ void k_abs_f(const float* x, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = fabsf(x[i]);
}
__global__ void k_abs_b(float* dx, const float* dy, const float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dx[i] += dy[i] * ((x[i] > 0.0f) ? 1.0f : (x[i] < 0.0f ? -1.0f : 0.0f));
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
  if (T <= 0 || n_fft <= 0 || hop <= 0)
    throw std::invalid_argument("invalid frame parameters");
  require_tensor(x, "frame input");
  const int64_t needed = static_cast<int64_t>(T - 1) * hop + n_fft;
  if (needed > x.numel()) throw std::out_of_range("frame window exceeds input");
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

// ==== GAN ops: exp + conv2d ====
__global__ void k_exp_f(const float* x, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = expf(x[i]);
}
__global__ void k_exp_b(float* dx, const float* dy, const float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dx[i] += dy[i] * y[i];  // d/dx exp(x) = exp(x) = output
}

Tensor exp_op(const Tensor& x) {
  return act_op(x,
      [](const float* a, float* o, int n) { k_exp_f<<<grid(n), 256>>>(a, o, n); },
      [](float* dx, const float* dy, const float* y, int n) { k_exp_b<<<grid(n), 256>>>(dx, dy, y, n); },
      true);
}

__global__ void k_flip_rows_f(const float* x, float* o, int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * cols) { int r = i / cols, c = i % cols; o[r * cols + c] = x[(rows - 1 - r) * cols + c]; }
}
__global__ void k_flip_rows_b(float* dx, const float* dy, int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * cols) { int r = i / cols, c = i % cols; atomicAdd(&dx[r * cols + c], dy[(rows - 1 - r) * cols + c]); }
}
Tensor flip_rows(const Tensor& x, int rows, int cols) {
  if (rows <= 0 || cols <= 0)
    throw std::invalid_argument("flip_rows dimensions must be positive");
  require_flat_size(x, static_cast<int64_t>(rows) * cols, "flip_rows input");
  auto out = make_node(x.shape(), any_requires_grad({&x}));
  k_flip_rows_f<<<grid(rows * cols), 256>>>(x.n->data.data(), out->data.data(), rows, cols);
  check_launch("flip_rows");
  out->parents = {x.n};
  Node* o = out.get(); Node* px = x.n.get();
  out->backward_fn = [o, px, rows, cols] {
    k_flip_rows_b<<<grid(rows * cols), 256>>>(px->grad.data(), o->grad.data(), rows, cols);
  };
  return Tensor(out);
}

__global__ void k_reflect_pad_end_f(const float* x, float* out, int length,
                                    int target_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= target_length) return;
  const int src = (i < length) ? i : (2 * length - 2 - i);
  out[i] = x[src];
}

__global__ void k_reflect_pad_end_b(float* dx, const float* dout, int length,
                                    int target_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= target_length) return;
  const int src = (i < length) ? i : (2 * length - 2 - i);
  atomicAdd(&dx[src], dout[i]);
}

Tensor reflect_pad_end(const Tensor& x, int target_len) {
  require_rank(x, 2, "reflect_pad_end input");
  if (x.shape()[0] != 1)
    throw std::invalid_argument("reflect_pad_end expects shape [1, length]");
  const int length = x.shape()[1];
  if (target_len < length || target_len >= 2 * length)
    throw std::invalid_argument("reflect_pad_end target is outside reflect range");
  if (target_len == length) return x;
  auto out = make_node({1, target_len}, any_requires_grad({&x}));
  k_reflect_pad_end_f<<<grid(target_len), 256>>>(x.n->data.data(), out->data.data(),
                                                length, target_len);
  check_launch("reflect_pad_end");
  out->parents = {x.n};
  Node* o = out.get();
  Node* px = x.n.get();
  out->backward_fn = [o, px, length, target_len] {
    k_reflect_pad_end_b<<<grid(target_len), 256>>>(px->grad.data(), o->grad.data(),
                                                  length, target_len);
  };
  return Tensor(out);
}

__global__ void k_conv2d_f(const float* x, const float* w, const float* b, float* o,
    int Cin, int H, int W, int Cout, int kh, int kw, int sh, int sw, int ph, int pw,
    int Hout, int Wout) {
  int ow = blockIdx.x * blockDim.x + threadIdx.x, oh = blockIdx.y, co = blockIdx.z;
  if (ow >= Wout || oh >= Hout || co >= Cout) return;
  float s = b ? b[co] : 0.0f;
  for (int ci = 0; ci < Cin; ++ci) {
    const float* xc = x + ci * H * W;
    const float* wc = w + (co * Cin + ci) * kh * kw;
    for (int i = 0; i < kh; ++i) {
      int h = oh * sh - ph + i;
      if (h < 0 || h >= H) continue;
      for (int j = 0; j < kw; ++j) {
        int wcol = ow * sw - pw + j;
        if (wcol >= 0 && wcol < W) s += xc[h * W + wcol] * wc[i * kw + j];
      }
    }
  }
  o[(co * Hout + oh) * Wout + ow] = s;
}
__global__ void k_conv2d_dx(float* dx, const float* dout, const float* w,
    int Cin, int H, int W, int Cout, int kh, int kw, int sh, int sw, int ph, int pw,
    int Hout, int Wout) {
  int ow = blockIdx.x * blockDim.x + threadIdx.x, oh = blockIdx.y, co = blockIdx.z;
  if (ow >= Wout || oh >= Hout || co >= Cout) return;
  float go = dout[(co * Hout + oh) * Wout + ow];
  for (int ci = 0; ci < Cin; ++ci) {
    const float* wc = w + (co * Cin + ci) * kh * kw;
    for (int i = 0; i < kh; ++i) {
      int h = oh * sh - ph + i;
      if (h < 0 || h >= H) continue;
      for (int j = 0; j < kw; ++j) {
        int wcol = ow * sw - pw + j;
        if (wcol >= 0 && wcol < W) atomicAdd(&dx[ci * H * W + h * W + wcol], go * wc[i * kw + j]);
      }
    }
  }
}
__global__ void k_conv2d_dw(float* dw, const float* dout, const float* x,
    int Cin, int H, int W, int Cout, int kh, int kw, int sh, int sw, int ph, int pw,
    int Hout, int Wout) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x, total = Cout * Cin * kh * kw;
  if (idx >= total) return;
  int j = idx % kw, i = (idx / kw) % kh, ci = (idx / (kw * kh)) % Cin, co = idx / (kw * kh * Cin);
  const float* xc = x + ci * H * W;
  float s = 0.0f;
  for (int oh = 0; oh < Hout; ++oh) {
    int h = oh * sh - ph + i;
    if (h < 0 || h >= H) continue;
    for (int ow = 0; ow < Wout; ++ow) {
      int wcol = ow * sw - pw + j;
      if (wcol >= 0 && wcol < W) s += dout[(co * Hout + oh) * Wout + ow] * xc[h * W + wcol];
    }
  }
  dw[idx] += s;
}
__global__ void k_conv2d_db(float* db, const float* dout, int Cout, int Hout, int Wout) {
  int co = blockIdx.x * blockDim.x + threadIdx.x;
  if (co >= Cout) return;
  float s = 0.0f;
  for (int i = 0; i < Hout * Wout; ++i) s += dout[co * Hout * Wout + i];
  db[co] += s;
}

// im2col for 2D conv: col[(ci*kh*kw + i*kw + j), (oh*Wout+ow)] = x[ci, h, wcol].
__global__ void k_im2col_2d(const float* x, float* col, int Cin, int H, int W,
    int kh, int kw, int sh, int sw, int ph, int pw, int Hout, int Wout) {
  int cols = Hout * Wout, rows = Cin * kh * kw;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;
  int c = idx % cols, r = idx / cols;
  int j = r % kw, i = (r / kw) % kh, ci = r / (kw * kh);
  int ow = c % Wout, oh = c / Wout;
  int h = oh * sh - ph + i, wcol = ow * sw - pw + j;
  col[idx] = (h >= 0 && h < H && wcol >= 0 && wcol < W) ? x[(ci * H + h) * W + wcol] : 0.0f;
}
__global__ void k_col2im_2d(float* dx, const float* dcol, int Cin, int H, int W,
    int kh, int kw, int sh, int sw, int ph, int pw, int Hout, int Wout) {
  int cols = Hout * Wout, rows = Cin * kh * kw;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;
  int c = idx % cols, r = idx / cols;
  int j = r % kw, i = (r / kw) % kh, ci = r / (kw * kh);
  int ow = c % Wout, oh = c / Wout;
  int h = oh * sh - ph + i, wcol = ow * sw - pw + j;
  if (h >= 0 && h < H && wcol >= 0 && wcol < W)
    atomicAdd(&dx[(ci * H + h) * W + wcol], dcol[idx]);
}

Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b,
              int Cin, int H, int W, int Cout, int kh, int kw,
              int sh, int sw, int ph, int pw) {
  if (Cin <= 0 || H <= 0 || W <= 0 || Cout <= 0 || kh <= 0 || kw <= 0 ||
      sh <= 0 || sw <= 0 || ph < 0 || pw < 0)
    throw std::invalid_argument("invalid conv2d parameters");
  require_flat_size(x, static_cast<int64_t>(Cin) * H * W, "conv2d input");
  require_flat_size(w, static_cast<int64_t>(Cout) * Cin * kh * kw, "conv2d weight");
  bool has_b = static_cast<bool>(b.n);
  if (has_b) require_flat_size(b, Cout, "conv2d bias");
  const int64_t hn = static_cast<int64_t>(H) + 2LL * ph - kh;
  const int64_t wn = static_cast<int64_t>(W) + 2LL * pw - kw;
  if (hn < 0 || wn < 0) throw std::invalid_argument("conv2d kernel exceeds padded input");
  const int64_t hout64 = hn / sh + 1;
  const int64_t wout64 = wn / sw + 1;
  if (hout64 <= 0 || wout64 <= 0 || hout64 > std::numeric_limits<int>::max() ||
      wout64 > std::numeric_limits<int>::max())
    throw std::length_error("conv2d output dimensions are invalid");
  int Hout = static_cast<int>(hout64);
  int Wout = static_cast<int>(wout64);
  bool rg = any_requires_grad({&x, &w}) ||
            (has_b && (b.n->requires_grad || !b.n->parents.empty()));
  auto out = make_node({Cout, Hout, Wout}, rg);
  Node* o = out.get(); Node* pxn = x.n.get(); Node* pwn = w.n.get();
  Node* pbn = has_b ? b.n.get() : nullptr;

  // Period discriminators convolve height independently at each period column.
  // Preserve the same reference accumulation as 1D without flattening columns
  // into adjacent time samples. Other 2D shapes retain the general GEMM path.
  const int rows = Cin * kh * kw;
  const int cols = Hout * Wout;
  const bool period_conv=kw==1 && sw==1 && pw==0;
  if(period_conv && static_cast<int64_t>(Cin)*H*W>20480) {
    conv1d_channels_last<false>(pxn->data.data(),pwn->data.data(),
        has_b?pbn->data.data():nullptr,o->data.data(),Cin,H*W,Cout,kh,sh,ph,1,cols,W);
  } else {
    auto col = ScratchPool::get().acquire(static_cast<std::size_t>(rows) * cols);
    k_im2col_2d<<<grid(rows * cols), 256>>>(pxn->data.data(), col->data(), Cin, H, W,
                                            kh, kw, sh, sw, ph, pw, Hout, Wout);
    if(period_conv) {
      conv_columns_blocked(pwn->data.data(),col->data(),has_b?pbn->data.data():nullptr,
          o->data.data(),Cout,cols,rows);
    } else {
      gemm_rm(false, false, Cout, cols, rows, 1.0f, pwn->data.data(), col->data(), 0.0f, o->data.data());
      if (has_b) k_bias_rows<<<grid(Cout * cols), 256>>>(o->data.data(), pbn->data.data(), Cout, cols);
    }
    check_launch("conv2d_im2col");
  }
  if (has_b) out->parents = {x.n, w.n, b.n}; else out->parents = {x.n, w.n};
  out->backward_fn = [o, pxn, pwn, pbn, Cin, H, W, Cout, kh, kw, sh, sw, ph, pw, Hout, Wout, rows, cols] {
    auto col = ScratchPool::get().acquire(static_cast<std::size_t>(rows) * cols);
    k_im2col_2d<<<grid(rows * cols), 256>>>(pxn->data.data(), col->data(), Cin, H, W,
                                           kh, kw, sh, sw, ph, pw, Hout, Wout);
    check_launch("conv2d_backward_im2col");
    // dW += dout[Cout,cols] @ col^T[cols,rows]
    gemm_rm(false, true, Cout, rows, cols, 1.0f, o->grad.data(), col->data(), 1.0f, pwn->grad.data());
    // dcol = W^T[rows,Cout] @ dout[Cout,cols]; scatter into dx
    auto dcol = ScratchPool::get().acquire(static_cast<std::size_t>(rows) * cols);
    gemm_rm(true, false, rows, cols, Cout, 1.0f, pwn->data.data(), o->grad.data(), 0.0f, dcol->data());
    k_col2im_2d<<<grid(rows * cols), 256>>>(pxn->grad.data(), dcol->data(), Cin, H, W,
                                            kh, kw, sh, sw, ph, pw, Hout, Wout);
    if (pbn) k_conv2d_db<<<grid(Cout), 256>>>(pbn->grad.data(), o->grad.data(), Cout, Hout, Wout);
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

  auto cache = node_cache();
  for (Node* nd : topo) {
    if (nd->grad.empty()) nd->grad = cache->acquire(static_cast<std::size_t>(nd->numel()));
    nd->ensure_grad();
  }

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

__global__ void k_grad_sq_acc(const float* grad, float* total, int n) {
  // Reduce locally before contending on the scalar. Besides eliminating one
  // global atomic per element, this keeps small contributions from being lost
  // after the running FP32 total grows beyond their unit precision.
  __shared__ float partial[256];
  float sum = 0.0f;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < static_cast<std::size_t>(n); i += stride) {
    const float value = grad[i];
    sum += value * value;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = 128; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(total, partial[0]);
}

__global__ void k_scale_grad(float* grad, float scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) grad[i] *= scale;
}

AdamW::AdamW(std::vector<Tensor> params, float lr, float beta1, float beta2,
             float eps, float weight_decay)
    : params_(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), wd_(weight_decay) {
  if (params_.empty() || !std::isfinite(lr_) || lr_ <= 0.0f ||
      !std::isfinite(beta1_) || beta1_ < 0.0f || beta1_ >= 1.0f ||
      !std::isfinite(beta2_) || beta2_ < 0.0f || beta2_ >= 1.0f ||
      !std::isfinite(eps_) || eps_ <= 0.0f || !std::isfinite(wd_) || wd_ < 0.0f)
    throw std::invalid_argument("invalid AdamW parameters");
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

void AdamW::set_lr(float lr) {
  if (!std::isfinite(lr) || lr <= 0.0f)
    throw std::invalid_argument("AdamW learning rate must be finite and positive");
  lr_ = lr;
}

void AdamW::restore_state(int steps, float lr,
    const std::function<std::vector<float>(std::size_t, bool)>& read_moment) {
  if(steps<0 || steps==std::numeric_limits<int>::max() ||
      !std::isfinite(lr) || lr<=0 || !read_moment)
    throw std::invalid_argument("invalid AdamW restore metadata");
  std::vector<core::CudaBuffer> first(params_.size()),second(params_.size());
  for(std::size_t i=0;i<params_.size();++i) {
    for(bool v:{false,true}) {
      auto values=read_moment(i,v);
      if(values.size()!=static_cast<std::size_t>(params_[i].numel()))
        throw std::invalid_argument("AdamW restore moment shape mismatch");
      for(float value:values)
        if(!std::isfinite(value) || (v && value<0))
          throw std::invalid_argument("invalid AdamW restore moment");
      (v?second[i]:first[i]).copy_from_host(values.data(),values.size());
    }
  }
  m_.swap(first);v_.swap(second);t_=steps;lr_=lr;
}

float AdamW::clip_grad_norm(float max_norm) {
  if (!std::isfinite(max_norm) || max_norm <= 0.0f)
    throw std::invalid_argument("gradient max_norm must be finite and positive");
  core::CudaBuffer sum_sq(1);
  sum_sq.zero();
  for (auto& param : params_) {
    const int n = checked_numel_int(param, "optimizer parameter");
    if (param.n->grad.size() < static_cast<std::size_t>(n)) continue;
    const int blocks = std::min(4096, 1 + (n - 1) / 256);
    k_grad_sq_acc<<<blocks, 256>>>(param.n->grad.data(), sum_sq.data(), n);
    check_launch("gradient norm");
  }
  float host_sum = 0.0f;
  sum_sq.copy_to_host(&host_sum, 1);
  const float norm = std::sqrt(host_sum);
  if (!std::isfinite(norm)) throw std::runtime_error("non-finite gradient norm");
  if (norm > max_norm) {
    const float factor = max_norm / (norm + 1e-12f);
    for (auto& param : params_) {
      const int n = checked_numel_int(param, "optimizer parameter");
      if (param.n->grad.size() < static_cast<std::size_t>(n)) continue;
      k_scale_grad<<<grid(n), 256>>>(param.n->grad.data(), factor, n);
      check_launch("gradient clipping");
    }
  }
  return norm;
}

void AdamW::step() {
  if(t_==std::numeric_limits<int>::max())
    throw std::overflow_error("AdamW step counter overflow");
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

// Joint training keeps the upstream weight-norm parameterization differentiable.
__global__ void k_clamp_lower(const float* x, float* y, int n, float lower) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] < lower ? lower : x[i];
}
__global__ void k_clamp_lower_backward(const float* x, const float* dy, float* dx,
                                      int n, float lower) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && x[i] >= lower) dx[i] += dy[i];
}
// Keep the frozen FP32 reference's four-lane reduction, including scalar tails.
// Mixed double/FP32 reductions can move near-zero downstream activations across
// a Leaky-ReLU branch even when normalized weights differ by only a few ULPs.
__device__ float row_dot_fp32(const float* a, const float* b, int inner,
                              float* partial) {
  const int tid=threadIdx.x;
  if(tid<4) {
    float value=0;
    for(int j=tid;j<inner;j+=4)
      value=__fadd_rn(value,__fmul_rn(a[j],b[j]));
    partial[tid]=value;
  }
  __syncthreads();
  return finish_fp32_lanes(partial, inner);
}
__global__ void k_weight_norm_forward(const float* v, const float* g, float* w,
                                      float* norms, int inner) {
  __shared__ float partial[4];
  const int row = blockIdx.x, tid = threadIdx.x;
  const float norm = sqrtf(row_dot_fp32(v+row*inner,v+row*inner,inner,partial));
  if (tid == 0) norms[row] = isfinite(g[row]) ? norm : NAN;
  const float scale = g[row]/norm;
  for (int j = tid; j < inner; j += 256) w[row*inner+j] = v[row*inner+j]*scale;
}
__global__ void k_weight_norm_backward(const float* v, const float* g,
    const float* norm, const float* dw, float* dv, float* dg, int inner) {
  __shared__ float partial[4];
  const int row = blockIdx.x, tid = threadIdx.x;
  const float n = norm[row], d = row_dot_fp32(dw+row*inner,v+row*inner,inner,partial);
  if (tid == 0) dg[row] += d/n;
  const float a=g[row]/n;
  const float b=__fmul_rn(a,d/n)/n;
  for (int j = tid; j < inner; j += 256) {
    const int i = row*inner+j;
    dv[i] += __fsub_rn(__fmul_rn(a,dw[i]),__fmul_rn(b,v[i]));
  }
}

Tensor reshape(const Tensor& x, std::vector<int> shape) {
  const int n = checked_numel_int(x, "reshape input");
  int64_t count = 1;
  if (shape.empty()) throw std::invalid_argument("reshape needs positive dimensions");
  for (int d : shape) {
    if (d <= 0 || count > std::numeric_limits<int>::max()/d)
      throw std::invalid_argument("invalid reshape dimensions");
    count *= d;
  }
  if (count != n) throw std::invalid_argument("reshape changes element count");
  auto out = make_node(std::move(shape), x.requires_grad());
  CK(cudaMemcpy(out->data.data(), x.n->data.data(), n*sizeof(float), cudaMemcpyDeviceToDevice));
  out->parents = {x.n}; Node* o = out.get(); Node* px = x.n.get();
  out->backward_fn = [o, px, n] {
    k_acc<<<grid(n),256>>>(px->grad.data(), o->grad.data(), n); check_launch("reshape backward");
  };
  return Tensor(out);
}
Tensor clamp_min(const Tensor& x, float lower) {
  const int n = checked_numel_int(x, "clamp input");
  if (!std::isfinite(lower)) throw std::invalid_argument("clamp lower must be finite");
  auto out = make_node(x.shape(), x.requires_grad());
  k_clamp_lower<<<grid(n),256>>>(x.n->data.data(), out->data.data(), n, lower);
  check_launch("clamp_min");
  out->parents = {x.n}; Node* o = out.get(); Node* px = x.n.get();
  out->backward_fn = [o,px,n,lower] {
    k_clamp_lower_backward<<<grid(n),256>>>(px->data.data(), o->grad.data(), px->grad.data(), n, lower);
    check_launch("clamp_min backward");
  };
  return Tensor(out);
}
Tensor weight_norm(const Tensor& v, const Tensor& g) {
  const int count = checked_numel_int(v, "weight_norm value");
  if (v.shape().size() < 2 || checked_numel_int(g, "weight_norm scale") != v.shape()[0])
    throw std::invalid_argument("weight_norm needs one scale per first-axis row");
  const int rows = v.shape()[0], inner = count/rows;
  auto norms = std::make_shared<core::CudaBuffer>(rows);
  auto out = make_node(v.shape(), any_requires_grad({&v,&g}));
  k_weight_norm_forward<<<rows,256>>>(v.n->data.data(), g.n->data.data(), out->data.data(), norms->data(), inner);
  check_launch("weight_norm");
  std::vector<float> host(rows); norms->copy_to_host(host.data(), rows);
  for (float norm : host) if (!(norm > 0) || !std::isfinite(norm))
    throw std::invalid_argument("weight_norm has a zero/nonfinite norm or scale");
  out->parents = {v.n,g.n}; Node* o = out.get(); Node* pv = v.n.get(); Node* pg = g.n.get();
  out->backward_fn = [o,pv,pg,norms,rows,inner] {
    k_weight_norm_backward<<<rows,256>>>(pv->data.data(),pg->data.data(),norms->data(),o->grad.data(),pv->grad.data(),pg->grad.data(),inner);
    check_launch("weight_norm backward");
  };
  return Tensor(out);
}
std::vector<float> AdamW::first_moment(std::size_t index) const {
  const auto& buffer = m_.at(index); std::vector<float> values(buffer.size());
  buffer.copy_to_host(values.data(), values.size()); return values;
}
std::vector<float> AdamW::second_moment(std::size_t index) const {
  const auto& buffer = v_.at(index); std::vector<float> values(buffer.size());
  buffer.copy_to_host(values.data(), values.size()); return values;
}

}  // namespace voxmutatio::autograd
