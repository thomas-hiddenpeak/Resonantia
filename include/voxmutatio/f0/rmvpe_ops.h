// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// 2D CNN operations for RMVPE (channels-first NCHW, batch=1 => [C,H,W]).

#pragma once

#include <vector>

namespace voxmutatio::f0::ops {

/// Conv2d with 3x3 kernel, stride 1, padding 1.
/// input: [Cin, H, W], weight: [Cout, Cin, 3, 3], bias: [Cout] or nullptr.
/// Returns [Cout, H, W].
std::vector<float> conv2d_3x3(const float* input, int Cin, int H, int W,
                              const float* weight, const float* bias, int Cout);

/// Conv2d with 1x1 kernel (shortcut). weight: [Cout, Cin, 1, 1].
std::vector<float> conv2d_1x1(const float* input, int Cin, int H, int W,
                              const float* weight, const float* bias, int Cout);

/// BatchNorm2d inference (in place): y = (x-mean)/sqrt(var+eps)*gamma + beta.
void batchnorm2d(float* data, int C, int H, int W,
                 const float* gamma, const float* beta,
                 const float* mean, const float* var, float eps = 1e-5f);

/// ReLU in place.
void relu_inplace(float* data, int size);

/// Sigmoid in place.
void sigmoid_inplace(float* data, int size);

/// AvgPool2d with kernel (kh, kw), stride == kernel. Returns [C, H/kh, W/kw].
std::vector<float> avgpool2d(const float* input, int C, int H, int W,
                             int kh, int kw);

/// ConvTranspose2d, 3x3, stride 2, padding 1, output_padding 1 (doubles H,W).
/// weight: [Cin, Cout, 3, 3] (PyTorch ConvTranspose layout). No bias.
/// Returns [Cout, 2H, 2W].
std::vector<float> conv_transpose2d_s2(const float* input, int Cin, int H, int W,
                                       const float* weight, int Cout);

/// Concatenate two [C,H,W] tensors along channel dim.
std::vector<float> concat_channels(const float* a, int Ca,
                                    const float* b, int Cb, int H, int W);

}  // namespace voxmutatio::f0::ops
