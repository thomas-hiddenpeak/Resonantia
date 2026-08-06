// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace voxmutatio::content::cuda {

// ============================================================================
// Fbank Feature Extraction
// ============================================================================

/// Compute log-Mel filterbank features from PCM audio
/// audio: [num_samples] 16kHz mono PCM
/// Returns: [num_frames, num_mels] filterbank features
std::vector<float> compute_fbank(const float* audio, int num_samples,
                                  int num_mels = 80,
                                  int frame_size = 1600,
                                  int hop_length = 320,
                                  int sample_rate = 16000);

// ============================================================================
// Linear Algebra (cuBLAS wrappers)
// ============================================================================

/// Matrix multiplication: C = A @ B
/// A: [M, K] row-major, B: [K, N] row-major, C: [M, N] row-major
void matmul(const float* A, int M, int K,
            const float* B, int N,
            float* C);

/// Batched matrix multiplication on GPU
void batched_matmul(const float** d_A, int M, int K,
                    const float** d_B, int N,
                    float** d_C, int batch_size);

// ============================================================================
// Layer Normalization
// ============================================================================

/// Layer normalization: normalize along last dimension
/// input: [..., dim] row-major
/// Returns: normalized output
std::vector<float> layer_norm(const float* input, int seq_len, int dim);

// ============================================================================
// Activation Functions
// ============================================================================

/// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu_forward(const float* input, float* output, int size);

/// ReLU: max(0, x)
void relu_forward(const float* input, float* output, int size);

// ============================================================================
// Transformer Attention
// ============================================================================

/// Multi-head self-attention forward pass
/// q, k, v: [seq_len, num_heads, head_dim]
/// Returns: [seq_len, num_heads * head_dim]
std::vector<float> multihead_attention(const float* q, const float* k,
                                        const float* v,
                                        int seq_len, int num_heads,
                                        int head_dim);

// ============================================================================
// Convolution (1D)
// ============================================================================

/// 1D convolution with padding
/// input: [seq_len, channels]
/// weight: [out_channels, channels, kernel_size]
/// Returns: [seq_len, out_channels] (same padding)
std::vector<float> conv1d(const float* input, int seq_len, int channels,
                          const float* weight, int out_channels,
                          int kernel_size);

// ============================================================================
// HuBERT-specific operations (numerically aligned with transformers)
// ============================================================================

/// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
void gelu_exact(const float* input, float* output, int size);

/// 1D convolution, strided, no padding (channels-first layout)
/// input: [in_channels, in_len]
/// weight: [out_channels, in_channels, kernel]
/// bias: [out_channels] or nullptr
/// Returns: [out_channels, out_len], out_len = (in_len - kernel) / stride + 1
std::vector<float> conv1d_strided(const float* input, int in_channels, int in_len,
                                   const float* weight, const float* bias,
                                   int out_channels, int kernel, int stride);

/// Grouped 1D convolution with padding (channels-first layout)
/// input: [in_channels, in_len]
/// weight: [out_channels, in_channels/groups, kernel]
/// bias: [out_channels] or nullptr
/// Returns: [out_channels, out_len]
std::vector<float> conv1d_grouped(const float* input, int in_channels, int in_len,
                                   const float* weight, const float* bias,
                                   int out_channels, int kernel, int stride,
                                   int padding, int groups);

/// Group normalization (channels-first: [channels, length])
/// Normalizes over (channels/num_groups, length) per group.
/// weight, bias: [channels]
void group_norm(const float* input, float* output, int channels, int length,
                int num_groups, const float* weight, const float* bias,
                float eps = 1e-5f);

/// Layer normalization with affine transform (row-major [seq_len, dim])
/// Normalizes over last dim. weight, bias: [dim]
std::vector<float> layer_norm_affine(const float* input, int seq_len, int dim,
                                      const float* weight, const float* bias,
                                      float eps = 1e-5f);

/// Multi-head attention with proper head splitting.
/// q, k, v: [seq_len, dim] where dim = num_heads * head_dim
/// Applies scaling 1/sqrt(head_dim). Returns [seq_len, dim].
std::vector<float> multihead_attention_split(const float* q, const float* k,
                                              const float* v,
                                              int seq_len, int dim,
                                              int num_heads);

/// Linear layer: output = input @ weight^T + bias
/// input: [M, K], weight: [N, K] (PyTorch layout), bias: [N] or nullptr
/// Returns: [M, N]
std::vector<float> linear(const float* input, int M, int K,
                          const float* weight, const float* bias, int N);

}  // namespace voxmutatio::content::cuda
