// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/autograd/tensor.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

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

}  // namespace voxmutatio::autograd
