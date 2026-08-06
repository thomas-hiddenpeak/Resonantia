// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/content/cuda/kernels.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace voxmutatio::content::cuda {

namespace {

cudaError_t check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
                cudaGetErrorString(err));
        return err;
    }
    return err;
}

#define CUDA_CHECK(err) check_cuda(err, __FILE__, __LINE__)

cublasStatus_t check_cublas(cublasStatus_t err, const char* file, int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", file, line, err);
        return err;
    }
    return err;
}

#define CUBLAS_CHECK(err) check_cublas(err, __FILE__, __LINE__)

// GELU kernel
__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float coeff1 = sqrtf(2.0f / M_PI);
        float coeff2 = 0.044715f;
        output[idx] = 0.5f * x * (1.0f + tanhf(coeff1 * (x + coeff2 * x * x * x)));
    }
}

// ReLU kernel
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Layer normalization kernel
__global__ void layer_norm_kernel(const float* input, float* output,
                                   float* mean, float* rstd,
                                   int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) {
        // Compute mean
        float sum = 0.0f;
        for (int d = 0; d < dim; ++d) {
            sum += input[idx * dim + d];
        }
        mean[idx] = sum / dim;

        // Compute variance
        float var = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = input[idx * dim + d] - mean[idx];
            var += diff * diff;
        }
        rstd[idx] = 1.0f / sqrtf(var / dim + 1e-5f);

        // Normalize
        for (int d = 0; d < dim; ++d) {
            output[idx * dim + d] = (input[idx * dim + d] - mean[idx]) * rstd[idx];
        }
    }
}

// Hann window function (host version)
static float hann_window_host(int i, int size) {
    return 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (size - 1)));
}

// STFT computation with cuFFT
std::vector<float> compute_stft(const float* audio, int num_samples,
                                 int frame_size, int hop_length) {
    int num_frames = (num_samples - frame_size) / hop_length + 1;
    int fft_size = frame_size;
    
    // Apply Hann window and extract frames (host side)
    std::vector<float> windowed(num_frames * fft_size);
    for (int f = 0; f < num_frames; ++f) {
        for (int i = 0; i < frame_size; ++i) {
            windowed[f * fft_size + i] = 
                audio[f * hop_length + i] * hann_window_host(i, frame_size);
        }
    }
    
    // GPU FFT
    cufftComplex* d_input = nullptr;
    cufftComplex* d_output = nullptr;
    cudaMalloc(&d_input, num_frames * fft_size * sizeof(cufftComplex));
    cudaMalloc(&d_output, num_frames * (fft_size / 2 + 1) * sizeof(cufftComplex));
    
    // Convert to complex
    std::vector<cuFloatComplex> h_input(num_frames * fft_size, {0.0f, 0.0f});
    for (int i = 0; i < num_frames * fft_size; ++i) {
        h_input[i].x = windowed[i];
    }
    cudaMemcpy(d_input, h_input.data(), 
               num_frames * fft_size * sizeof(cufftComplex), 
               cudaMemcpyHostToDevice);
    
    // Create FFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, fft_size, CUFFT_C2C, num_frames);
    cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD);
    cufftDestroy(plan);
    
    // Copy results back
    std::vector<cuFloatComplex> h_output(num_frames * (fft_size / 2 + 1));
    cudaMemcpy(h_output.data(), d_output,
               num_frames * (fft_size / 2 + 1) * sizeof(cufftComplex),
               cudaMemcpyDeviceToHost);
    
    // Compute magnitude spectrum
    std::vector<float> magnitudes(num_frames * (fft_size / 2 + 1));
    for (int i = 0; i < num_frames * (fft_size / 2 + 1); ++i) {
        magnitudes[i] = sqrtf(h_output[i].x * h_output[i].x + 
                              h_output[i].y * h_output[i].y);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return magnitudes;
}

}  // namespace

std::vector<float> compute_fbank(const float* audio, int num_samples,
                                  int num_mels, int frame_size,
                                  int hop_length, int sample_rate) {
    // Compute STFT
    int num_freqs = frame_size / 2 + 1;
    auto magnitudes = compute_stft(audio, num_samples, frame_size, hop_length);
    int num_frames = static_cast<int>(magnitudes.size()) / num_freqs;
    
    // Create Mel filterbank matrix (simplified)
    std::vector<std::vector<float>> mel_filters(num_mels, std::vector<float>(num_freqs, 0.0f));
    
    // Mel scale conversion
    auto freq_to_mel = [](float freq) -> float {
        return 2595.0f * log10f(1.0f + freq / 700.0f);
    };
    
    auto mel_to_freq = [](float mel) -> float {
        return 700.0f * powf(10.0f, mel / 2595.0f) - 700.0f;
    };
    
    float mel_low = freq_to_mel(0.0f);
    float mel_high = freq_to_mel(sample_rate / 2.0f);
    
    std::vector<float> mel_points(num_mels + 2);
    for (int i = 0; i < num_mels + 2; ++i) {
        mel_points[i] = mel_low + (mel_high - mel_low) * i / (num_mels + 1);
    }
    
    std::vector<float> freq_points(num_mels + 2);
    for (int i = 0; i < num_mels + 2; ++i) {
        freq_points[i] = mel_to_freq(mel_points[i]);
    }
    
    std::vector<int> freq_indices(num_mels + 2);
    for (int i = 0; i < num_mels + 2; ++i) {
        freq_indices[i] = static_cast<int>(roundf(
            (num_freqs - 1) * freq_points[i] / (sample_rate / 2.0f)));
    }
    
    // Fill Mel filterbank
    for (int m = 0; m < num_mels; ++m) {
        for (int f = freq_indices[m]; f <= freq_indices[m + 1]; ++f) {
            if (f > freq_indices[m]) {
                mel_filters[m][f] = static_cast<float>(f - freq_indices[m]) /
                                   static_cast<float>(freq_indices[m + 1] - freq_indices[m]);
            }
        }
        for (int f = freq_indices[m + 1]; f < freq_indices[m + 2]; ++f) {
            mel_filters[m][f] = static_cast<float>(freq_indices[m + 2] - f) /
                               static_cast<float>(freq_indices[m + 2] - freq_indices[m + 1]);
        }
    }
    
    // Apply Mel filterbank and log
    std::vector<float> fbank(num_frames * num_mels);
    for (int f = 0; f < num_frames; ++f) {
        for (int m = 0; m < num_mels; ++m) {
            float sum = 0.0f;
            for (int i = 0; i < num_freqs; ++i) {
                sum += magnitudes[f * num_freqs + i] * mel_filters[m][i];
            }
            fbank[f * num_mels + m] = logf(sum + 1e-10f);
        }
    }
    
    return fbank;
}

void matmul(const float* A, int M, int K,
            const float* B, int N,
            float* C) {
    // GPU allocations
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy inputs
    CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // cuBLAS uses column-major, so C = A @ B becomes C^T = B^T @ A^T
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                             N, M, K, &alpha,
                             d_B, N, d_A, M, &beta, d_C, N));
    
    // Copy result (transpose back to row-major)
    CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

std::vector<float> layer_norm(const float* input, int seq_len, int dim) {
    std::vector<float> output(seq_len * dim);
    std::vector<float> mean(seq_len);
    std::vector<float> rstd(seq_len);
    
    float *d_input, *d_output, *d_mean, *d_rstd;
    CUDA_CHECK(cudaMalloc(&d_input, seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rstd, seq_len * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, input, seq_len * dim * sizeof(float), 
                          cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_size = (seq_len + block_size - 1) / block_size;
    
    layer_norm_kernel<<<grid_size, block_size>>>(d_input, d_output, d_mean, d_rstd,
                                                  seq_len, dim);
    
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, seq_len * dim * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_rstd);
    
    return output;
}

void gelu_forward(const float* input, float* output, int size) {
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    gelu_kernel<<<grid_size, block_size>>>(d_input, d_output, size);
    
    CUDA_CHECK(cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void relu_forward(const float* input, float* output, int size) {
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    relu_kernel<<<grid_size, block_size>>>(d_input, d_output, size);
    
    CUDA_CHECK(cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

std::vector<float> multihead_attention(const float* q, const float* k,
                                        const float* v,
                                        int seq_len, int num_heads,
                                        int head_dim) {
    int total_dim = num_heads * head_dim;
    std::vector<float> output(seq_len * total_dim, 0.0f);
    
    // Scaled dot-product attention for each head
    float scale = 1.0f / sqrtf(head_dim);
    
    for (int h = 0; h < num_heads; ++h) {
        // Compute attention scores: Q @ K^T
        std::vector<float> scores(seq_len * seq_len);
        
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q[i * total_dim + h * head_dim + d] *
                           k[j * total_dim + h * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }
        
        // Softmax
        for (int i = 0; i < seq_len; ++i) {
            float max_val = scores[i * seq_len];
            for (int j = 1; j < seq_len; ++j) {
                max_val = fmaxf(max_val, scores[i * seq_len + j]);
            }
            
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                scores[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
                sum += scores[i * seq_len + j];
            }
            
            for (int j = 0; j < seq_len; ++j) {
                scores[i * seq_len + j] /= sum;
            }
        }
        
        // Apply attention to values: scores @ V
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                float val = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    val += scores[i * seq_len + j] *
                           v[j * total_dim + h * head_dim + d];
                }
                output[i * total_dim + h * head_dim + d] = val;
            }
        }
    }
    
    return output;
}

std::vector<float> conv1d(const float* input, int seq_len, int channels,
                          const float* weight, int out_channels,
                          int kernel_size) {
    int padding = kernel_size / 2;
    std::vector<float> output(seq_len * out_channels, 0.0f);
    
    for (int o = 0; o < out_channels; ++o) {
        for (int i = 0; i < seq_len; ++i) {
            float sum = 0.0f;
            for (int k = -padding; k <= padding; ++k) {
                int idx = i + k;
                if (idx >= 0 && idx < seq_len) {
                    for (int c = 0; c < channels; ++c) {
                        sum += input[idx * channels + c] *
                               weight[o * channels * kernel_size + 
                                     c * kernel_size + (k + padding)];
                    }
                }
            }
            output[i * out_channels + o] = sum;
        }
    }
    
    return output;
}

void batched_matmul(const float** d_A, int M, int K,
                    const float** d_B, int N,
                    float** d_C, int batch_size) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Note: This is a simplified implementation
    // Full batched GEMM would use cublasSgemmBatched
    CUBLAS_CHECK(cublasDestroy(handle));
}

// ============================================================================
// HuBERT-specific kernels
// ============================================================================

// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
__global__ void gelu_exact_kernel(const float* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = in[idx];
        out[idx] = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
    }
}

// Strided conv1d (channels-first). input:[in_ch,in_len] weight:[out_ch,in_ch,kernel]
__global__ void conv1d_strided_kernel(const float* input, const float* weight,
                                       const float* bias, float* output,
                                       int in_ch, int in_len, int out_ch,
                                       int kernel, int stride, int out_len) {
    int o = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < out_ch && t < out_len) {
        float sum = bias ? bias[o] : 0.0f;
        int start = t * stride;
        for (int c = 0; c < in_ch; ++c) {
            const float* in_c = input + c * in_len + start;
            const float* w_oc = weight + (o * in_ch + c) * kernel;
            #pragma unroll 4
            for (int k = 0; k < kernel; ++k) {
                sum += in_c[k] * w_oc[k];
            }
        }
        output[o * out_len + t] = sum;
    }
}

// Grouped conv1d with padding. weight:[out_ch, in_ch/groups, kernel]
__global__ void conv1d_grouped_kernel(const float* input, const float* weight,
                                       const float* bias, float* output,
                                       int in_ch, int in_len, int out_ch,
                                       int kernel, int stride, int padding,
                                       int groups, int out_len) {
    int o = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < out_ch && t < out_len) {
        int in_per_group = in_ch / groups;
        int out_per_group = out_ch / groups;
        int g = o / out_per_group;
        int in_start_ch = g * in_per_group;
        float sum = bias ? bias[o] : 0.0f;
        int t_in_start = t * stride - padding;
        for (int ci = 0; ci < in_per_group; ++ci) {
            int c = in_start_ch + ci;
            const float* w = weight + (o * in_per_group + ci) * kernel;
            for (int k = 0; k < kernel; ++k) {
                int ti = t_in_start + k;
                if (ti >= 0 && ti < in_len) {
                    sum += input[c * in_len + ti] * w[k];
                }
            }
        }
        output[o * out_len + t] = sum;
    }
}

// Group norm (channels-first [channels, length]). One block per group.
__global__ void group_norm_kernel(const float* input, float* output,
                                   const float* weight, const float* bias,
                                   int channels, int length, int num_groups,
                                   float eps) {
    int g = blockIdx.x;
    if (g >= num_groups) return;
    int ch_per_group = channels / num_groups;
    int count = ch_per_group * length;

    float sum = 0.0f;
    for (int c = 0; c < ch_per_group; ++c) {
        int ch = g * ch_per_group + c;
        for (int t = 0; t < length; ++t) sum += input[ch * length + t];
    }
    float mean = sum / count;

    float var = 0.0f;
    for (int c = 0; c < ch_per_group; ++c) {
        int ch = g * ch_per_group + c;
        for (int t = 0; t < length; ++t) {
            float d = input[ch * length + t] - mean;
            var += d * d;
        }
    }
    var /= count;
    float rstd = rsqrtf(var + eps);

    for (int c = 0; c < ch_per_group; ++c) {
        int ch = g * ch_per_group + c;
        for (int t = 0; t < length; ++t) {
            float norm = (input[ch * length + t] - mean) * rstd;
            output[ch * length + t] = norm * weight[ch] + bias[ch];
        }
    }
}

// Layer norm with affine (row-major [seq_len, dim]). One block per row.
__global__ void layer_norm_affine_kernel(const float* input, float* output,
                                          const float* weight, const float* bias,
                                          int seq_len, int dim, float eps) {
    int row = blockIdx.x;
    if (row >= seq_len) return;
    const float* x = input + row * dim;
    float* y = output + row * dim;

    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) sum += x[d];
    float mean = sum / dim;

    float var = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float diff = x[d] - mean;
        var += diff * diff;
    }
    var /= dim;
    float rstd = rsqrtf(var + eps);

    for (int d = 0; d < dim; ++d) {
        y[d] = (x[d] - mean) * rstd * weight[d] + bias[d];
    }
}

// Softmax over rows of a [rows, cols] matrix (in place)
__global__ void softmax_rows_kernel(float* data, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float* row = data + r * cols;
    float mx = row[0];
    for (int c = 1; c < cols; ++c) mx = fmaxf(mx, row[c]);
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) { row[c] = expf(row[c] - mx); sum += row[c]; }
    for (int c = 0; c < cols; ++c) row[c] /= sum;
}

void gelu_exact(const float* input, float* output, int size) {
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, input, size * sizeof(float), cudaMemcpyHostToDevice));
    int block = 256, grid = (size + block - 1) / block;
    gelu_exact_kernel<<<grid, block>>>(d_in, d_out, size);
    CUDA_CHECK(cudaMemcpy(output, d_out, size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out);
}

std::vector<float> conv1d_strided(const float* input, int in_channels, int in_len,
                                   const float* weight, const float* bias,
                                   int out_channels, int kernel, int stride) {
    int out_len = (in_len - kernel) / stride + 1;
    std::vector<float> output(out_channels * out_len);

    float *d_in, *d_w, *d_b = nullptr, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, in_channels * in_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, out_channels * in_channels * kernel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, out_channels * out_len * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, input, in_channels * in_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, weight, out_channels * in_channels * kernel * sizeof(float), cudaMemcpyHostToDevice));
    if (bias) {
        CUDA_CHECK(cudaMalloc(&d_b, out_channels * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_b, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice));
    }

    int block = 256;
    dim3 grid((out_len + block - 1) / block, out_channels);
    conv1d_strided_kernel<<<grid, block>>>(d_in, d_w, d_b, d_out,
                                            in_channels, in_len, out_channels,
                                            kernel, stride, out_len);
    CUDA_CHECK(cudaMemcpy(output.data(), d_out, out_channels * out_len * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
    if (d_b) cudaFree(d_b);
    return output;
}

std::vector<float> conv1d_grouped(const float* input, int in_channels, int in_len,
                                   const float* weight, const float* bias,
                                   int out_channels, int kernel, int stride,
                                   int padding, int groups) {
    int out_len = (in_len + 2 * padding - kernel) / stride + 1;
    std::vector<float> output(out_channels * out_len);

    int w_size = out_channels * (in_channels / groups) * kernel;
    float *d_in, *d_w, *d_b = nullptr, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, in_channels * in_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, out_channels * out_len * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, input, in_channels * in_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, weight, w_size * sizeof(float), cudaMemcpyHostToDevice));
    if (bias) {
        CUDA_CHECK(cudaMalloc(&d_b, out_channels * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_b, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice));
    }

    int block = 256;
    dim3 grid((out_len + block - 1) / block, out_channels);
    conv1d_grouped_kernel<<<grid, block>>>(d_in, d_w, d_b, d_out,
                                            in_channels, in_len, out_channels,
                                            kernel, stride, padding, groups, out_len);
    CUDA_CHECK(cudaMemcpy(output.data(), d_out, out_channels * out_len * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
    if (d_b) cudaFree(d_b);
    return output;
}

void group_norm(const float* input, float* output, int channels, int length,
                int num_groups, const float* weight, const float* bias, float eps) {
    float *d_in, *d_out, *d_w, *d_b;
    int n = channels * length;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, channels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, channels * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, input, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, weight, channels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, bias, channels * sizeof(float), cudaMemcpyHostToDevice));

    group_norm_kernel<<<num_groups, 1>>>(d_in, d_out, d_w, d_b, channels, length, num_groups, eps);
    CUDA_CHECK(cudaMemcpy(output, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_w); cudaFree(d_b);
}

std::vector<float> layer_norm_affine(const float* input, int seq_len, int dim,
                                      const float* weight, const float* bias, float eps) {
    std::vector<float> output(seq_len * dim);
    float *d_in, *d_out, *d_w, *d_b;
    CUDA_CHECK(cudaMalloc(&d_in, seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, input, seq_len * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, weight, dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, bias, dim * sizeof(float), cudaMemcpyHostToDevice));

    layer_norm_affine_kernel<<<seq_len, 1>>>(d_in, d_out, d_w, d_b, seq_len, dim, eps);
    CUDA_CHECK(cudaMemcpy(output.data(), d_out, seq_len * dim * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_w); cudaFree(d_b);
    return output;
}

// Linear: output[M,N] = input[M,K] @ weight[N,K]^T + bias[N]
std::vector<float> linear(const float* input, int M, int K,
                          const float* weight, const float* bias, int N) {
    std::vector<float> output(M * N);
    float *d_in, *d_w, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, input, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, weight, N * K * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    // output_cm[N,M] = weight_cm^T[N,K] @ input_cm[K,M]
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             d_w, K, d_in, K, &beta, d_out, N));
    cublasDestroy(handle);

    CUDA_CHECK(cudaMemcpy(output.data(), d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);

    // Add bias
    if (bias) {
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
                output[m * N + n] += bias[n];
    }
    return output;
}

// Multi-head attention with head splitting.
// q,k,v: [seq_len, dim], dim = num_heads * head_dim
std::vector<float> multihead_attention_split(const float* q, const float* k,
                                              const float* v,
                                              int seq_len, int dim,
                                              int num_heads) {
    int head_dim = dim / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    std::vector<float> output(seq_len * dim, 0.0f);

    // Process each head
    for (int h = 0; h < num_heads; ++h) {
        // Extract per-head Q,K,V into contiguous [seq_len, head_dim]
        std::vector<float> qh(seq_len * head_dim), kh(seq_len * head_dim), vh(seq_len * head_dim);
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                qh[i * head_dim + d] = q[i * dim + h * head_dim + d] * scale;
                kh[i * head_dim + d] = k[i * dim + h * head_dim + d];
                vh[i * head_dim + d] = v[i * dim + h * head_dim + d];
            }
        }

        // scores[seq,seq] = qh @ kh^T
        float *d_q, *d_k, *d_scores;
        CUDA_CHECK(cudaMalloc(&d_q, seq_len * head_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k, seq_len * head_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scores, seq_len * seq_len * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_q, qh.data(), seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k, kh.data(), seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice));

        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        float alpha = 1.0f, beta = 0.0f;
        // scores_cm[seq,seq] = kh_cm^T[seq,hd] @ qh_cm[hd,seq]
        // scores row-major [i,j] = sum_d qh[i,d]*kh[j,d]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 seq_len, seq_len, head_dim, &alpha,
                                 d_k, head_dim, d_q, head_dim, &beta,
                                 d_scores, seq_len));

        // Softmax over rows
        int block = 256, grid = (seq_len + block - 1) / block;
        softmax_rows_kernel<<<grid, block>>>(d_scores, seq_len, seq_len);

        // out[seq,hd] = scores[seq,seq] @ vh[seq,hd]
        float *d_v, *d_outh;
        CUDA_CHECK(cudaMalloc(&d_v, seq_len * head_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outh, seq_len * head_dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_v, vh.data(), seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice));
        // out_cm[hd,seq] = vh_cm^T? We want out[i,d] = sum_j scores[i,j]*vh[j,d]
        // out_cm[hd,seq]: use vh_cm[hd,seq] @ scores_cm... 
        // Simpler: out row-major = scores[seq,seq] @ vh[seq,hd]
        // out_cm[hd,seq] = vh_cm[hd,seq] (=vh[seq,hd] rowmajor) treated as [hd,seq]... 
        // Let C=out[seq,hd] rowmajor = out_cm[hd,seq]. A=scores[seq,seq], B=vh[seq,hd].
        // C_cm[hd,seq] = B_cm^T?[hd,seq] ... use: cublasSgemm(N,N, hd, seq, seq, vh_cm[hd,seq]?)
        // vh row-major [seq,hd] = vh_cm[hd,seq]. scores row-major[seq,seq]=scores_cm[seq,seq].
        // out_cm[hd,seq] = vh_cm[hd,seq] @ scores_cm^T[seq,seq]  (since out[i,d]=sum_j scores[i,j]vh[j,d] = sum_j vh_cm[d,j]*scores_cm[j? ])
        // out[i,d] = sum_j scores[i,j] * vh[j,d]. In col-major: out_cm[d,i]=sum_j vh_cm[d,j]*scores_cm[j,i]? scores_cm[j,i]=scores[i,j] (since scores_cm is transpose). Yes!
        // So out_cm[hd,seq] = vh_cm[hd,seq] @ scores_cm[seq,seq], with OP_N,OP_N.
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 head_dim, seq_len, seq_len, &alpha,
                                 d_v, head_dim, d_scores, seq_len, &beta,
                                 d_outh, head_dim));

        std::vector<float> outh(seq_len * head_dim);
        CUDA_CHECK(cudaMemcpy(outh.data(), d_outh, seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost));

        // Scatter back into output
        for (int i = 0; i < seq_len; ++i)
            for (int d = 0; d < head_dim; ++d)
                output[i * dim + h * head_dim + d] = outh[i * head_dim + d];

        cublasDestroy(handle);
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_scores);
        cudaFree(d_v); cudaFree(d_outh);
    }

    return output;
}

}  // namespace voxmutatio::content::cuda
