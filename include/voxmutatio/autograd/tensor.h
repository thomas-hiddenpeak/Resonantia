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

/// Reverse-mode backprop from a scalar loss.
void backward(const Tensor& loss);

}  // namespace voxmutatio::autograd
