// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/content/cuda/kernels.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "voxmutatio/core/cuda_buffer.h"

namespace voxmutatio::content::cuda {

namespace {

void check_cublas(cublasStatus_t err, const char* file, int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error at ") + file + ":" +
                                 std::to_string(line) + " (status " +
                                 std::to_string(static_cast<int>(err)) + ")");
    }
}

#define CUBLAS_CHECK(err) check_cublas(err, __FILE__, __LINE__)

class CublasHandle final {
 public:
    CublasHandle() {
        const cublasStatus_t status = cublasCreate(&handle_);
        if (status != CUBLAS_STATUS_SUCCESS && handle_) {
            (void)cublasDestroy(handle_);
            handle_ = nullptr;
        }
        CUBLAS_CHECK(status);
    }

    ~CublasHandle() noexcept {
        if (handle_) (void)cublasDestroy(handle_);
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }

    void close() {
        if (!handle_) return;
        const cublasStatus_t status = cublasDestroy(handle_);
        if (status == CUBLAS_STATUS_SUCCESS) handle_ = nullptr;
        CUBLAS_CHECK(status);
    }

 private:
    cublasHandle_t handle_{};
};

void check_cufft(cufftResult err, const char* file, int line) {
    if (err != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string("cuFFT error at ") + file + ":" +
                                 std::to_string(line) + " (status " +
                                 std::to_string(static_cast<int>(err)) + ")");
    }
}

#define CUFFT_CHECK(err) check_cufft(err, __FILE__, __LINE__)

std::size_t checked_element_count(std::initializer_list<int> dimensions,
                                  const char* operation) {
    std::size_t count = 1;
    constexpr std::size_t kMaxElements =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    for (int dimension : dimensions) {
        if (dimension <= 0) {
            throw std::invalid_argument(std::string(operation) +
                                        " requires positive dimensions");
        }
        if (count > kMaxElements / static_cast<std::size_t>(dimension)) {
            throw std::length_error(std::string(operation) +
                                    " tensor exceeds INT_MAX elements");
        }
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

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
    if (!audio || num_samples <= 0 || frame_size <= 1 || hop_length <= 0 ||
        num_samples < frame_size) {
        throw std::invalid_argument(
            "compute_fbank requires audio, frame_size > 1, hop_length > 0, and at least one frame");
    }
    int num_frames = (num_samples - frame_size) / hop_length + 1;
    int fft_size = frame_size;
    int num_freqs = fft_size / 2 + 1;
    const std::size_t framed_count = static_cast<std::size_t>(num_frames) * fft_size;
    if (framed_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("compute_fbank input is too large");
    }
    
    // Apply Hann window and extract frames (host side)
    std::vector<float> windowed(framed_count);
    for (int f = 0; f < num_frames; ++f) {
        for (int i = 0; i < frame_size; ++i) {
            windowed[f * fft_size + i] = 
                audio[f * hop_length + i] * hann_window_host(i, frame_size);
        }
    }
    
    // GPU FFT
    core::CudaBuffer d_input(framed_count);
    // CudaBuffer stores floats; an R2C output contains two floats per bin.
    core::CudaBuffer d_output(static_cast<std::size_t>(num_frames) * num_freqs * 2);
    d_input.copy_from_host(windowed.data(), windowed.size());
    
    // Create FFT plan
    cufftHandle plan{};
    CUFFT_CHECK(cufftPlan1d(&plan, fft_size, CUFFT_R2C, num_frames));
    cufftResult exec_status = cufftExecR2C(
        plan, d_input.data(), reinterpret_cast<cufftComplex*>(d_output.data()));
    cufftResult destroy_status = cufftDestroy(plan);
    CUFFT_CHECK(exec_status);
    CUFFT_CHECK(destroy_status);
    
    // Copy results back
    std::vector<cuFloatComplex> h_output(static_cast<std::size_t>(num_frames) * num_freqs);
    d_output.copy_to_host(reinterpret_cast<float*>(h_output.data()), h_output.size() * 2);
    
    // Compute magnitude spectrum
    std::vector<float> magnitudes(static_cast<std::size_t>(num_frames) * num_freqs);
    for (std::size_t i = 0; i < magnitudes.size(); ++i) {
        magnitudes[i] = sqrtf(h_output[i].x * h_output[i].x + 
                              h_output[i].y * h_output[i].y);
    }
    
    return magnitudes;
}

}  // namespace

std::vector<float> compute_fbank(const float* audio, int num_samples,
                                  int num_mels, int frame_size,
                                  int hop_length, int sample_rate) {
    constexpr int kMaximumMelBins = 4096;
    if (num_mels <= 0 || num_mels > kMaximumMelBins || sample_rate <= 0) {
        throw std::invalid_argument(
            "compute_fbank requires sample_rate > 0 and 1..4096 mel bins");
    }
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
        freq_indices[i] = std::clamp(static_cast<int>(roundf(
            (num_freqs - 1) * freq_points[i] / (sample_rate / 2.0f))), 0,
            num_freqs - 1);
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
    if (!A || !B || !C || M <= 0 || K <= 0 || N <= 0) {
        throw std::invalid_argument("matmul requires non-null buffers and positive dimensions");
    }
    const std::size_t a_count = checked_element_count({M, K}, "matmul A");
    const std::size_t b_count = checked_element_count({K, N}, "matmul B");
    const std::size_t c_count = checked_element_count({M, N}, "matmul C");
    core::CudaBuffer d_A(a_count);
    core::CudaBuffer d_B(b_count);
    core::CudaBuffer d_C(c_count);
    d_A.copy_from_host(A, a_count);
    d_B.copy_from_host(B, b_count);
    
    // cuBLAS handle
    CublasHandle handle;
    
    // cuBLAS uses column-major, so C = A @ B becomes C^T = B^T @ A^T
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t gemm_status = cublasSgemm(
        handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B.data(), N,
        d_A.data(), K, &beta, d_C.data(), N);
    CUBLAS_CHECK(gemm_status);
    handle.close();
    
    // Copy result (transpose back to row-major)
    d_C.copy_to_host(C, c_count);
}

std::vector<float> layer_norm(const float* input, int seq_len, int dim) {
    if (!input || seq_len <= 0 || dim <= 0) {
        throw std::invalid_argument(
            "layer_norm requires a non-null input and positive dimensions");
    }
    const std::size_t element_count =
        checked_element_count({seq_len, dim}, "layer_norm");
    std::vector<float> output(element_count);
    std::vector<float> mean(seq_len);
    std::vector<float> rstd(seq_len);
    
    core::CudaBuffer d_input(element_count);
    core::CudaBuffer d_output(element_count);
    core::CudaBuffer d_mean(seq_len);
    core::CudaBuffer d_rstd(seq_len);

    d_input.copy_from_host(input, element_count);
    
    int block_size = 256;
    int grid_size = (seq_len - 1) / block_size + 1;
    
    layer_norm_kernel<<<grid_size, block_size>>>(d_input.data(), d_output.data(),
                                                  d_mean.data(), d_rstd.data(),
                                                  seq_len, dim);

    d_output.copy_to_host(output.data(), element_count);
    
    return output;
}

void gelu_forward(const float* input, float* output, int size) {
    if (!input || !output || size <= 0) {
        throw std::invalid_argument(
            "gelu_forward requires non-null buffers and a positive size");
    }
    core::CudaBuffer d_input(size);
    core::CudaBuffer d_output(size);

    d_input.copy_from_host(input, size);
    
    int block_size = 256;
    int grid_size = (size - 1) / block_size + 1;
    
    gelu_kernel<<<grid_size, block_size>>>(d_input.data(), d_output.data(), size);

    d_output.copy_to_host(output, size);
}

void relu_forward(const float* input, float* output, int size) {
    if (!input || !output || size <= 0) {
        throw std::invalid_argument(
            "relu_forward requires non-null buffers and a positive size");
    }
    core::CudaBuffer d_input(size);
    core::CudaBuffer d_output(size);

    d_input.copy_from_host(input, size);
    
    int block_size = 256;
    int grid_size = (size - 1) / block_size + 1;
    
    relu_kernel<<<grid_size, block_size>>>(d_input.data(), d_output.data(), size);

    d_output.copy_to_host(output, size);
}

std::vector<float> multihead_attention(const float* q, const float* k,
                                        const float* v,
                                        int seq_len, int num_heads,
                                        int head_dim) {
    if (!q || !k || !v || seq_len <= 0 || num_heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument(
            "multihead_attention requires non-null inputs and positive dimensions");
    }
    const int total_dim = static_cast<int>(
        checked_element_count({num_heads, head_dim}, "multihead_attention head dimension"));
    const std::size_t output_count =
        checked_element_count({seq_len, total_dim}, "multihead_attention output");
    const std::size_t score_count =
        checked_element_count({seq_len, seq_len}, "multihead_attention scores");
    std::vector<float> output(output_count, 0.0f);
    
    // Scaled dot-product attention for each head
    float scale = 1.0f / sqrtf(head_dim);
    
    for (int h = 0; h < num_heads; ++h) {
        // Compute attention scores: Q @ K^T
        std::vector<float> scores(score_count);
        
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
    if (!input || !weight || seq_len <= 0 || channels <= 0 || out_channels <= 0 ||
        kernel_size <= 0 || kernel_size % 2 == 0) {
        throw std::invalid_argument(
            "conv1d requires non-null inputs, positive dimensions, and an odd kernel");
    }
    int padding = kernel_size / 2;
    const std::size_t output_count =
        checked_element_count({seq_len, out_channels}, "conv1d output");
    (void)checked_element_count(
        {out_channels, channels, kernel_size}, "conv1d weights");
    std::vector<float> output(output_count, 0.0f);
    
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
    if (!d_A || !d_B || !d_C || M <= 0 || K <= 0 || N <= 0 || batch_size <= 0) {
        throw std::invalid_argument(
            "batched_matmul requires device pointer arrays and positive dimensions");
    }
    CublasHandle handle;
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasStatus_t gemm_status = cublasSgemmBatched(
        handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K,
        &beta, d_C, N, batch_size);
    CUBLAS_CHECK(gemm_status);
    handle.close();
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
    if (!input || !output || size <= 0) {
        throw std::invalid_argument(
            "gelu_exact requires non-null buffers and a positive size");
    }
    core::CudaBuffer d_in(size);
    core::CudaBuffer d_out(size);
    d_in.copy_from_host(input, size);
    int block = 256, grid = (size - 1) / block + 1;
    gelu_exact_kernel<<<grid, block>>>(d_in.data(), d_out.data(), size);
    d_out.copy_to_host(output, size);
}

std::vector<float> conv1d_strided(const float* input, int in_channels, int in_len,
                                   const float* weight, const float* bias,
                                   int out_channels, int kernel, int stride) {
    if (!input || !weight || in_channels <= 0 || in_len <= 0 || out_channels <= 0 ||
        kernel <= 0 || stride <= 0 || in_len < kernel) {
        throw std::invalid_argument(
            "conv1d_strided requires non-null inputs, positive dimensions and stride, "
            "and in_len >= kernel");
    }
    int out_len = (in_len - kernel) / stride + 1;
    const std::size_t input_count =
        checked_element_count({in_channels, in_len}, "conv1d_strided input");
    const std::size_t weight_count = checked_element_count(
        {out_channels, in_channels, kernel}, "conv1d_strided weights");
    const std::size_t output_count =
        checked_element_count({out_channels, out_len}, "conv1d_strided output");
    std::vector<float> output(output_count);

    core::CudaBuffer d_in(input_count);
    core::CudaBuffer d_w(weight_count);
    core::CudaBuffer d_b;
    core::CudaBuffer d_out(output_count);
    d_in.copy_from_host(input, input_count);
    d_w.copy_from_host(weight, weight_count);
    if (bias) {
        d_b.copy_from_host(bias, out_channels);
    }

    int block = 256;
    dim3 grid((out_len - 1) / block + 1, out_channels);
    conv1d_strided_kernel<<<grid, block>>>(d_in.data(), d_w.data(),
                                            bias ? d_b.data() : nullptr, d_out.data(),
                                            in_channels, in_len, out_channels,
                                            kernel, stride, out_len);
    d_out.copy_to_host(output.data(), output_count);
    return output;
}

std::vector<float> conv1d_grouped(const float* input, int in_channels, int in_len,
                                   const float* weight, const float* bias,
                                   int out_channels, int kernel, int stride,
                                   int padding, int groups) {
    if (!input || !weight || in_channels <= 0 || in_len <= 0 || out_channels <= 0 ||
        kernel <= 0 || stride <= 0 || padding < 0 || groups <= 0 ||
        in_channels % groups != 0 || out_channels % groups != 0 ||
        static_cast<int64_t>(in_len) + 2LL * padding < kernel) {
        throw std::invalid_argument(
            "conv1d_grouped requires valid non-null inputs, positive dimensions/stride/groups, "
            "divisible channels, and a non-empty output");
    }
    const int64_t numerator =
        static_cast<int64_t>(in_len) + 2LL * padding - kernel;
    const int64_t out_len64 = numerator / stride + 1;
    if (out_len64 <= 0 || out_len64 > std::numeric_limits<int>::max()) {
        throw std::length_error("conv1d_grouped output length exceeds int range");
    }
    const int out_len = static_cast<int>(out_len64);
    const int channels_per_group = in_channels / groups;
    const std::size_t input_count =
        checked_element_count({in_channels, in_len}, "conv1d_grouped input");
    const std::size_t weight_count = checked_element_count(
        {out_channels, channels_per_group, kernel}, "conv1d_grouped weights");
    const std::size_t output_count =
        checked_element_count({out_channels, out_len}, "conv1d_grouped output");
    std::vector<float> output(output_count);

    core::CudaBuffer d_in(input_count);
    core::CudaBuffer d_w(weight_count);
    core::CudaBuffer d_b;
    core::CudaBuffer d_out(output_count);
    d_in.copy_from_host(input, input_count);
    d_w.copy_from_host(weight, weight_count);
    if (bias) {
        d_b.copy_from_host(bias, out_channels);
    }

    int block = 256;
    dim3 grid((out_len - 1) / block + 1, out_channels);
    conv1d_grouped_kernel<<<grid, block>>>(d_in.data(), d_w.data(),
                                            bias ? d_b.data() : nullptr, d_out.data(),
                                            in_channels, in_len, out_channels,
                                            kernel, stride, padding, groups, out_len);
    d_out.copy_to_host(output.data(), output_count);
    return output;
}

void group_norm(const float* input, float* output, int channels, int length,
                int num_groups, const float* weight, const float* bias, float eps) {
    if (!input || !output || !weight || !bias || channels <= 0 || length <= 0 ||
        num_groups <= 0 || channels % num_groups != 0 || !std::isfinite(eps) || eps <= 0.0f) {
        throw std::invalid_argument(
            "group_norm requires valid buffers, dimensions/groups, and positive finite eps");
    }
    const std::size_t element_count =
        checked_element_count({channels, length}, "group_norm");
    core::CudaBuffer d_in(element_count);
    core::CudaBuffer d_out(element_count);
    core::CudaBuffer d_w(channels);
    core::CudaBuffer d_b(channels);
    d_in.copy_from_host(input, element_count);
    d_w.copy_from_host(weight, channels);
    d_b.copy_from_host(bias, channels);

    group_norm_kernel<<<num_groups, 1>>>(d_in.data(), d_out.data(), d_w.data(),
                                         d_b.data(), channels, length, num_groups, eps);
    d_out.copy_to_host(output, element_count);
}

std::vector<float> layer_norm_affine(const float* input, int seq_len, int dim,
                                      const float* weight, const float* bias, float eps) {
    if (!input || !weight || !bias || seq_len <= 0 || dim <= 0 ||
        !std::isfinite(eps) || eps <= 0.0f) {
        throw std::invalid_argument(
            "layer_norm_affine requires valid buffers/dimensions and positive finite eps");
    }
    const std::size_t element_count =
        checked_element_count({seq_len, dim}, "layer_norm_affine");
    std::vector<float> output(element_count);
    core::CudaBuffer d_in(element_count);
    core::CudaBuffer d_out(element_count);
    core::CudaBuffer d_w(dim);
    core::CudaBuffer d_b(dim);
    d_in.copy_from_host(input, element_count);
    d_w.copy_from_host(weight, dim);
    d_b.copy_from_host(bias, dim);

    layer_norm_affine_kernel<<<seq_len, 1>>>(d_in.data(), d_out.data(), d_w.data(),
                                             d_b.data(), seq_len, dim, eps);
    d_out.copy_to_host(output.data(), element_count);
    return output;
}

// Linear: output[M,N] = input[M,K] @ weight[N,K]^T + bias[N]
std::vector<float> linear(const float* input, int M, int K,
                          const float* weight, const float* bias, int N) {
    if (!input || !weight || M <= 0 || K <= 0 || N <= 0) {
        throw std::invalid_argument(
            "linear requires non-null inputs and positive dimensions");
    }
    const std::size_t input_count = checked_element_count({M, K}, "linear input");
    const std::size_t weight_count = checked_element_count({N, K}, "linear weights");
    const std::size_t output_count = checked_element_count({M, N}, "linear output");
    std::vector<float> output(output_count);
    core::CudaBuffer d_in(input_count);
    core::CudaBuffer d_w(weight_count);
    core::CudaBuffer d_out(output_count);
    d_in.copy_from_host(input, input_count);
    d_w.copy_from_host(weight, weight_count);

    CublasHandle handle;
    float alpha = 1.0f, beta = 0.0f;
    // output_cm[N,M] = weight_cm^T[N,K] @ input_cm[K,M]
    CUBLAS_CHECK(cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             d_w.data(), K, d_in.data(), K, &beta, d_out.data(), N));
    handle.close();

    d_out.copy_to_host(output.data(), output_count);

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
    if (!q || !k || !v || seq_len <= 0 || dim <= 0 || num_heads <= 0 ||
        dim % num_heads != 0) {
        throw std::invalid_argument(
            "multihead_attention_split requires non-null inputs, positive dimensions/heads, "
            "and dim divisible by num_heads");
    }
    int head_dim = dim / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    const std::size_t tensor_count =
        checked_element_count({seq_len, dim}, "multihead_attention_split tensor");
    const std::size_t head_count = checked_element_count(
        {seq_len, head_dim}, "multihead_attention_split head");
    const std::size_t score_count = checked_element_count(
        {seq_len, seq_len}, "multihead_attention_split scores");
    std::vector<float> output(tensor_count, 0.0f);

    // Process each head
    for (int h = 0; h < num_heads; ++h) {
        // Extract per-head Q,K,V into contiguous [seq_len, head_dim]
        std::vector<float> qh(head_count), kh(head_count), vh(head_count);
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                qh[i * head_dim + d] = q[i * dim + h * head_dim + d] * scale;
                kh[i * head_dim + d] = k[i * dim + h * head_dim + d];
                vh[i * head_dim + d] = v[i * dim + h * head_dim + d];
            }
        }

        // scores[seq,seq] = qh @ kh^T
        core::CudaBuffer d_q(head_count);
        core::CudaBuffer d_k(head_count);
        core::CudaBuffer d_scores(score_count);
        d_q.copy_from_host(qh.data(), head_count);
        d_k.copy_from_host(kh.data(), head_count);

        CublasHandle handle;
        float alpha = 1.0f, beta = 0.0f;
        // scores_cm[seq,seq] = kh_cm^T[seq,hd] @ qh_cm[hd,seq]
        // scores row-major [i,j] = sum_d qh[i,d]*kh[j,d]
        CUBLAS_CHECK(cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                                 seq_len, seq_len, head_dim, &alpha,
                                 d_k.data(), head_dim, d_q.data(), head_dim, &beta,
                                 d_scores.data(), seq_len));

        // Softmax over rows
        int block = 256, grid = (seq_len - 1) / block + 1;
        softmax_rows_kernel<<<grid, block>>>(d_scores.data(), seq_len, seq_len);

        // out[seq,hd] = scores[seq,seq] @ vh[seq,hd]
        core::CudaBuffer d_v(head_count);
        core::CudaBuffer d_outh(head_count);
        d_v.copy_from_host(vh.data(), head_count);
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
        CUBLAS_CHECK(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                                 head_dim, seq_len, seq_len, &alpha,
                                 d_v.data(), head_dim, d_scores.data(), seq_len, &beta,
                                 d_outh.data(), head_dim));

        std::vector<float> outh(head_count);
        d_outh.copy_to_host(outh.data(), head_count);

        // Scatter back into output
        for (int i = 0; i < seq_len; ++i)
            for (int d = 0; d < head_dim; ++d)
                output[i * dim + h * head_dim + d] = outh[i * head_dim + d];

        handle.close();
    }

    return output;
}

}  // namespace voxmutatio::content::cuda
