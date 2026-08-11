// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "voxmutatio/io/safetensors.h"

namespace voxmutatio::training::detail {

inline const float* require_f32(const io::SafetensorsLoader& loader,
                                const std::string& name, std::size_t count) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::length_error("training tensor size overflow: " + name);
  }
  const auto* tensor = loader.get_tensor(name);
  if (!tensor || tensor->dtype != "F32" ||
      tensor->data_nbytes != count * sizeof(float)) {
    throw std::runtime_error("missing or incompatible training tensor: " + name);
  }
  const auto* data = reinterpret_cast<const float*>(loader.data(name));
  if (!data) throw std::runtime_error("unreadable training tensor: " + name);
  return data;
}

inline std::vector<float> load_f32(const io::SafetensorsLoader& loader,
                                   const std::string& name, std::size_t count) {
  const float* data = require_f32(loader, name, count);
  std::vector<float> result(data, data + count);
  for (float value : result) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("non-finite training tensor: " + name);
    }
  }
  return result;
}

inline int embedding_rows(const io::SafetensorsLoader& loader,
                          const std::string& name, int width) {
  const auto* tensor = loader.get_tensor(name);
  if (!tensor || tensor->dtype != "F32" || tensor->shape.size() != 2 ||
      tensor->shape[0] <= 0 ||
      tensor->shape[0] > std::numeric_limits<int>::max() ||
      tensor->shape[1] != width) {
    throw std::runtime_error("invalid speaker embedding tensor: " + name);
  }
  require_f32(loader, name, static_cast<std::size_t>(tensor->shape[0]) * width);
  return static_cast<int>(tensor->shape[0]);
}

inline std::vector<float> reconstruct_weight_norm(const io::SafetensorsLoader& loader,
                                                  const std::string& name,
                                                  int out_dim, int inner) {
  if (out_dim <= 0 || inner <= 0) {
    throw std::invalid_argument("weight-norm dimensions must be positive");
  }
  const float* g = require_f32(loader, name + ".weight_g", out_dim);
  const float* v = require_f32(loader, name + ".weight_v",
                               static_cast<std::size_t>(out_dim) * inner);
  std::vector<float> result(static_cast<std::size_t>(out_dim) * inner);
  for (int o = 0; o < out_dim; ++o) {
    if (!std::isfinite(g[o])) {
      throw std::runtime_error("non-finite weight norm scale: " + name);
    }
    double norm_squared = 0.0;
    for (int i = 0; i < inner; ++i) {
      const float value = v[static_cast<std::size_t>(o) * inner + i];
      if (!std::isfinite(value)) {
        throw std::runtime_error("non-finite weight norm vector: " + name);
      }
      norm_squared += static_cast<double>(value) * value;
    }
    if (!(norm_squared > 0.0) || !std::isfinite(norm_squared)) {
      throw std::runtime_error("degenerate weight norm vector: " + name);
    }
    const double norm = std::sqrt(norm_squared);
    for (int i = 0; i < inner; ++i) {
      const std::size_t index = static_cast<std::size_t>(o) * inner + i;
      const double reconstructed =
          static_cast<double>(g[o]) * static_cast<double>(v[index]) / norm;
      if (!std::isfinite(reconstructed) ||
          std::abs(reconstructed) > std::numeric_limits<float>::max()) {
        throw std::runtime_error("weight norm reconstruction overflow: " + name);
      }
      result[index] = static_cast<float>(reconstructed);
    }
  }
  return result;
}

inline const float* require_speaker_embedding(const io::SafetensorsLoader& loader,
                                              int speaker_id, int width) {
  const int rows = embedding_rows(loader, "model.emb_g.weight", width);
  if (speaker_id < 0 || speaker_id >= rows) {
    throw std::out_of_range("speaker id is outside model.emb_g.weight");
  }
  return require_f32(loader, "model.emb_g.weight",
                     static_cast<std::size_t>(rows) * width);
}

}  // namespace voxmutatio::training::detail
