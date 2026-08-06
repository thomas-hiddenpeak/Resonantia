// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// 1D operations for VITS synthesizer (channels-first [C, L]).

#pragma once

#include <vector>

namespace voxmutatio::synthesizer::ops {

/// General 1D convolution (channels-first).
/// input: [Cin, L], weight: [Cout, Cin/groups, K], bias: [Cout] or nullptr.
/// Returns [Cout, out_len], out_len = (L + 2*pad - dilation*(K-1) - 1)/stride + 1.
std::vector<float> conv1d(const float* input, int Cin, int L,
                          const float* weight, const float* bias,
                          int Cout, int K, int stride, int pad,
                          int dilation, int groups);

/// ConvTranspose1d (channels-first).
/// input: [Cin, L], weight: [Cin, Cout, K] (PyTorch layout), bias: [Cout] or nullptr.
/// Returns [Cout, out_len], out_len = (L-1)*stride - 2*pad + K.
std::vector<float> conv_transpose1d(const float* input, int Cin, int L,
                                    const float* weight, const float* bias,
                                    int Cout, int K, int stride, int pad);

/// LeakyReLU in place.
void leaky_relu_inplace(float* data, int size, float slope);

/// tanh in place.
void tanh_inplace(float* data, int size);

}  // namespace voxmutatio::synthesizer::ops
