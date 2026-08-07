// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Minimal reverse-mode automatic differentiation over device float tensors.
// Foundation for pure C++/CUDA training (spec 002). Each op records a backward
// closure; backward() does a topological reverse traversal accumulating grads.

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "voxmutatio/core/cuda_buffer.h"

namespace voxmutatio::autograd {

/// A node in the autograd graph: owns data and (optionally) gradient.
struct Node {
  core::CudaBuffer data;
  core::CudaBuffer grad;               // accumulated during backward
  std::vector<int> shape;
  bool requires_grad = false;
  std::vector<std::shared_ptr<Node>> parents;
  std::function<void()> backward_fn;   // reads this->grad, adds to parents' grad

  [[nodiscard]] int64_t numel() const {
    int64_t n = 1;
    for (int s : shape) n *= s;
    return n;
  }
  void ensure_grad() {
    grad.resize(static_cast<std::size_t>(numel()));
    grad.zero();
  }
};

/// Handle to a Node. Copyable (shared ownership of the graph node).
class Tensor {
 public:
  std::shared_ptr<Node> n;

  Tensor() = default;
  explicit Tensor(std::shared_ptr<Node> node) : n(std::move(node)) {}

  static Tensor from_host(const std::vector<float>& v, std::vector<int> shape,
                          bool requires_grad = false);
  static Tensor zeros(std::vector<int> shape, bool requires_grad = false);

  [[nodiscard]] const std::vector<int>& shape() const { return n->shape; }
  [[nodiscard]] int64_t numel() const { return n->numel(); }
  [[nodiscard]] bool requires_grad() const { return n->requires_grad; }

  [[nodiscard]] std::vector<float> to_host() const;
  [[nodiscard]] std::vector<float> grad_to_host() const;
};

// ---- Elementwise ops (equal shape) ----
Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);

// ---- Linear algebra ----
/// 2D matmul: a[M,K] @ b[K,N] -> [M,N].
Tensor matmul(const Tensor& a, const Tensor& b);

// ---- Reductions ----
/// Sum all elements -> scalar [1].
Tensor sum(const Tensor& a);

// ---- Bias / Linear (Phase A1) ----
/// Broadcast add: x[R,C] + b[C] -> [R,C].
Tensor add_bias(const Tensor& x, const Tensor& b);
/// Linear: x[M,K] @ w[N,K]^T (+ b[N]) -> [M,N]. Pass Tensor{} for no bias.
Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b);

// ---- Activations (elementwise) ----
Tensor relu(const Tensor& x);
Tensor leaky_relu(const Tensor& x, float slope);
Tensor gelu(const Tensor& x);        // exact erf
Tensor tanh_op(const Tensor& x);
Tensor sigmoid(const Tensor& x);

// ---- Softmax over last dim (x is [rows, cols]) ----
Tensor softmax_rows(const Tensor& x, int rows, int cols);

// ---- LayerNorm over last dim with affine (x is [rows, dim]) ----
Tensor layer_norm(const Tensor& x, const Tensor& w, const Tensor& b,
                  int rows, int dim, float eps = 1e-5f);

// ---- Convolutions (channels-first) ----
/// conv1d: x[Cin,L], w[Cout, Cin/groups, K], b[Cout] (or Tensor{}) -> [Cout,Lout].
Tensor conv1d(const Tensor& x, const Tensor& w, const Tensor& b,
              int Cin, int L, int Cout, int K, int stride, int pad,
              int dilation, int groups);
/// conv_transpose1d: x[Cin,L], w[Cin,Cout,K], b[Cout] (or Tensor{}) -> [Cout,Lout].
Tensor conv_transpose1d(const Tensor& x, const Tensor& w, const Tensor& b,
                        int Cin, int L, int Cout, int K, int stride, int pad);
/// conv2d (no dilation/groups): x[Cin,H,W], w[Cout,Cin,kh,kw], b[Cout] (or Tensor{})
/// -> [Cout,Hout,Wout]. For MultiPeriodDiscriminator (kw=1 typical).
Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& b,
              int Cin, int H, int W, int Cout, int kh, int kw,
              int sh, int sw, int ph, int pw);

// ---- Shape / gather ops (Phase A3 primitives) ----
/// Transpose a 2D tensor: x[rows,cols] -> [cols,rows].
Tensor transpose2d(const Tensor& x, int rows, int cols);
/// Concatenate along dim 0 (rows): a[Ra,cols], b[Rb,cols] -> [Ra+Rb, cols].
Tensor concat_rows(const Tensor& a, const Tensor& b, int cols);
/// Slice rows [start, start+count) of x[rows,cols] -> [count, cols].
Tensor slice_rows(const Tensor& x, int start, int count, int cols);
/// Embedding lookup: table[vocab,dim] gathered by idx[T] -> [T,dim].
Tensor embedding(const Tensor& table, const std::vector<int>& idx, int dim);
/// Scale by a constant.
Tensor scale(const Tensor& x, float s);

// ---- Elementwise math (for differentiable mel loss) ----
Tensor sqrt_op(const Tensor& x);
Tensor log_op(const Tensor& x);   // natural log
Tensor abs_op(const Tensor& x);
Tensor exp_op(const Tensor& x);
/// Frame a 1D signal [L] into overlapping frames [T, n_fft]: f[t,n]=x[t*hop+n].
Tensor frame(const Tensor& x, int T, int n_fft, int hop);

/// Reverse-mode backprop from a scalar loss.
void backward(const Tensor& loss);

/// AdamW optimizer over a fixed set of parameter tensors.
class AdamW {
 public:
  AdamW(std::vector<Tensor> params, float lr, float beta1 = 0.9f,
        float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f);

  /// Apply one update step using each parameter's accumulated .grad.
  void step();
  /// Zero all parameter gradients.
  void zero_grad();
  void set_lr(float lr) { lr_ = lr; }
  [[nodiscard]] float lr() const { return lr_; }
  [[nodiscard]] int step_count() const { return t_; }

 private:
  std::vector<Tensor> params_;
  std::vector<core::CudaBuffer> m_, v_;
  float lr_, beta1_, beta2_, eps_, wd_;
  int t_ = 0;
};

}  // namespace voxmutatio::autograd
