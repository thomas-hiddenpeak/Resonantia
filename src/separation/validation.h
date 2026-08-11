// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "voxmutatio/core/cuda_buffer.h"
#include "voxmutatio/io/safetensors.h"

namespace voxmutatio::separation::detail {

enum class TensorValueConstraint {
  kFinite,
  kFiniteNonnegative,
};

inline bool all_finite(const float* values, std::size_t count) {
  if (!values && count != 0) return false;
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) return false;
  }
  return true;
}

inline bool checked_element_count(std::initializer_list<int64_t> shape,
                                  std::size_t& count) {
  count = 1;
  for (const int64_t dim : shape) {
    if (dim <= 0 || static_cast<std::uint64_t>(dim) >
                        std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    const auto size = static_cast<std::size_t>(dim);
    if (count > std::numeric_limits<std::size_t>::max() / size) return false;
    count *= size;
  }
  return true;
}

inline std::size_t checked_size_product(std::initializer_list<int64_t> factors,
                                        std::string_view context) {
  std::size_t result = 0;
  if (!checked_element_count(factors, result)) {
    throw std::invalid_argument(std::string(context) + " size is not representable");
  }
  return result;
}

inline int checked_int_product(std::initializer_list<int64_t> factors,
                               std::string_view context) {
  const std::size_t result = checked_size_product(factors, context);
  if (result > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(std::string(context) + " exceeds the CUDA int limit");
  }
  return static_cast<int>(result);
}

inline std::size_t checked_float_count(std::initializer_list<int64_t> factors,
                                       std::string_view context) {
  const std::size_t result = checked_size_product(factors, context);
  if (result > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      result > std::vector<float>().max_size()) {
    throw std::invalid_argument(std::string(context) +
                                " float storage is not representable");
  }
  return result;
}

inline int checked_resampled_length(int input_length, int source_rate,
                                    int destination_rate,
                                    std::string_view context) {
  if (input_length <= 0 || source_rate <= 0 || destination_rate <= 0) {
    throw std::invalid_argument(std::string(context) + " resampling parameters are invalid");
  }
  const auto scaled = static_cast<std::uint64_t>(input_length) *
                      static_cast<std::uint64_t>(destination_rate);
  const auto output = scaled / static_cast<std::uint64_t>(source_rate);
  if (output == 0 ||
      output > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(std::string(context) +
                                " resampled length is not representable");
  }
  return static_cast<int>(output);
}

inline bool checked_f32_tensor(const io::SafetensorsLoader& loader,
                               const std::string& name,
                               std::initializer_list<int64_t> expected_shape,
                               std::string_view component,
                               const float*& data,
                               std::size_t& count,
                               TensorValueConstraint constraint =
                                   TensorValueConstraint::kFinite) {
  data = nullptr;
  count = 0;
  const auto* tensor = loader.get_tensor(name);
  if (!tensor) {
    std::fprintf(stderr, "%.*s: missing %s\n", static_cast<int>(component.size()),
                 component.data(), name.c_str());
    return false;
  }

  if (tensor->dtype != "F32" ||
      tensor->shape.size() != expected_shape.size() ||
      !std::equal(tensor->shape.begin(), tensor->shape.end(),
                  expected_shape.begin()) ||
      !checked_element_count(expected_shape, count) ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      tensor->data_nbytes != count * sizeof(float)) {
    std::fprintf(stderr, "%.*s: invalid dtype/shape/size for %s\n",
                 static_cast<int>(component.size()), component.data(), name.c_str());
    return false;
  }

  const auto* raw = loader.data(name);
  if (!raw || reinterpret_cast<std::uintptr_t>(raw) % alignof(float) != 0) {
    std::fprintf(stderr, "%.*s: invalid data range/alignment for %s\n",
                 static_cast<int>(component.size()), component.data(), name.c_str());
    return false;
  }
  data = reinterpret_cast<const float*>(raw);
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(data[i]) ||
        (constraint == TensorValueConstraint::kFiniteNonnegative && data[i] < 0.0f)) {
      std::fprintf(stderr, "%.*s: invalid numeric value in %s\n",
                   static_cast<int>(component.size()), component.data(), name.c_str());
      data = nullptr;
      count = 0;
      return false;
    }
  }
  return true;
}

inline bool upload_checked_f32(const io::SafetensorsLoader& loader,
                               const std::string& name,
                               std::initializer_list<int64_t> expected_shape,
                               core::CudaBuffer& buffer,
                               std::string_view component,
                               TensorValueConstraint constraint =
                                   TensorValueConstraint::kFinite) {
  const float* data = nullptr;
  std::size_t count = 0;
  if (!checked_f32_tensor(loader, name, expected_shape, component, data, count,
                          constraint)) {
    return false;
  }
  buffer.copy_from_host(data, count);
  return true;
}

inline void require_finite_input(const float* values, std::size_t count,
                                 std::string_view context) {
  if (!all_finite(values, count)) {
    throw std::invalid_argument(std::string(context) + " contains NaN or infinity");
  }
}

inline void require_finite_input(const std::vector<float>& values,
                                 std::string_view context) {
  require_finite_input(values.data(), values.size(), context);
}

inline void require_finite_output(const float* values, std::size_t count,
                                  std::string_view context) {
  if (!all_finite(values, count)) {
    throw std::runtime_error(std::string(context) + " produced NaN or infinity");
  }
}

inline void require_finite_output(const std::vector<float>& values,
                                  std::string_view context) {
  require_finite_output(values.data(), values.size(), context);
}

}  // namespace voxmutatio::separation::detail
