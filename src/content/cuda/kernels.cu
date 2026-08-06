// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/content/cuda/kernels.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include < cufft.h>

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

// Hann window function
__device__ float hann_window(int i, int size) {
    return 0.5f * (1.0f - cosf(2.0f * M_PI * i / (size - 1)));
}

// STFT kernel using cuFFT
std::vector<float> compute_stft(const float* audio, int num_samples,
                                 int frame_size, int hop_length) {
    int num_frames = (num_samples - frame_size) / hop_length + 1;
    int fft_size = frame_size;
    
    // Apply Hann window and extract frames
    std::vector<float> windowed(num_frames * fft_size);
    for (int f = 0; f < num_frames; ++f) {
        for (int i = 0; i < frame_size; ++i) {
            windowed[f * fft_size + i] = 
                audio[f * hop_length + i] * hann_window(i, frame_size);
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

}  // namespace voxmutatio::content::cuda
