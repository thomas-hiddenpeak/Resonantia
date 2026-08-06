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

}  // namespace voxmutatio::content::cuda
