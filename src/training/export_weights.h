// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "voxmutatio/autograd/tensor.h"

namespace voxmutatio::training::detail {

// Private staging representation. Decoder and Flow are published by one writer.
struct ExportTensor {
  std::string name;
  std::vector<int> shape;
  std::vector<float> data;
};

inline void derive_wn(const std::vector<float>& fused, int rows, int inner,
                      std::vector<float>& g, std::vector<float>& v) {
  if (rows <= 0 || inner <= 0 ||
      fused.size() != static_cast<std::size_t>(rows) * inner)
    throw std::invalid_argument("invalid exported weight-norm shape");
  g.assign(rows, 0.0f);
  v = fused;
  for (int row = 0; row < rows; ++row) {
    const auto start = static_cast<std::size_t>(row) * inner;
    double norm_squared = 0;
    for (int i = 0; i < inner; ++i) {
      const float value = fused[start + i];
      if (!std::isfinite(value)) throw std::invalid_argument("non-finite exported weight");
      norm_squared += static_cast<double>(value) * value;
    }
    const double norm = std::sqrt(norm_squared);
    if (norm > std::numeric_limits<float>::max())
      throw std::invalid_argument("exported weight norm overflow");
    g[row] = static_cast<float>(norm);
    // A zero trained row is valid, but v must remain nondegenerate for inference.
    if (norm == 0) v[start] = 1.0f;
  }
}

inline void append_plain(std::vector<ExportTensor>& entries, const std::string& name,
                          const autograd::Tensor& tensor) {
  entries.push_back({name, tensor.shape(), tensor.to_host()});
}

inline void append_weight_norm(std::vector<ExportTensor>& entries,
                               const std::string& name, const autograd::Tensor& tensor) {
  const auto& shape = tensor.shape();
  const int rows = shape.at(0);
  std::vector<float> g, v;
  derive_wn(tensor.to_host(), rows, static_cast<int>(tensor.numel() / rows), g, v);
  std::vector<int> g_shape(shape.size(), 1);
  g_shape[0] = rows;
  entries.push_back({name + ".weight_g", std::move(g_shape), std::move(g)});
  entries.push_back({name + ".weight_v", shape, std::move(v)});
}

}  // namespace voxmutatio::training::detail
