// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/synthesizer/vits_ops.h"

#include "voxmutatio/core/cuda_buffer.h"
#include "relative_attention.h"

#if defined(VOXMUTATIO_TEST_CUDA_FAILURE_INJECTION)
#include "cuda_failure_injection.h"
#endif

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace voxmutatio::synthesizer::ops {

namespace {

inline void ck(cudaError_t e, const char* f, int l) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + f + ":" +
                                 std::to_string(l) + ": " + cudaGetErrorString(e));
    }
}
#define CK(e) ck((e), __FILE__, __LINE__)

void check_blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("VITS cuBLAS operation failed: " +
                                 std::to_string(static_cast<int>(status)));
}

class BlasHandle {
public:
    BlasHandle() {
        const auto status = cublasCreate(&handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            if (handle_) (void)cublasDestroy(handle_);
            handle_ = nullptr;
            check_blas(status);
        }
        const auto mode = cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH);
        if (mode != CUBLAS_STATUS_SUCCESS) {
            (void)cublasDestroy(handle_);
            handle_ = nullptr;
            check_blas(mode);
        }
    }
    ~BlasHandle() { if (handle_) (void)cublasDestroy(handle_); }
    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;
    cublasHandle_t get() const { return handle_; }
private:
    cublasHandle_t handle_ = nullptr;
};

// In production these expand directly to the CUDA runtime calls and add no
// fault-seam branch or synchronization. The private test build intercepts them
// and performs an explicit post-launch synchronization for independent fault
// injection; production keeps the existing blocking D2H copy boundary.
#if defined(VOXMUTATIO_TEST_CUDA_FAILURE_INJECTION)
#define CUDA_COPY(...) resonantia_test::cuda_failure::copy(__VA_ARGS__)
#define BEFORE_KERNEL_LAUNCH() CK(resonantia_test::cuda_failure::before_launch())
#define AFTER_KERNEL_LAUNCH()                                      \
    do {                                                           \
        CK(cudaGetLastError());                                    \
        CK(resonantia_test::cuda_failure::synchronize());          \
    } while (false)
#else
#define CUDA_COPY(...) cudaMemcpy(__VA_ARGS__)
#define BEFORE_KERNEL_LAUNCH() ((void)0)
#define AFTER_KERNEL_LAUNCH() CK(cudaGetLastError())
#endif

constexpr int kMaxGridY = 65'535;

void require_pointer(const void* pointer, const char* operation) {
    if (!pointer) {
        throw std::invalid_argument(std::string(operation) +
                                    " requires a non-null borrowed pointer");
    }
}

std::size_t checked_element_count(std::initializer_list<int> dimensions,
                                  const char* operation) {
    std::size_t count = 1;
    constexpr std::size_t kMaxElements =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    for (const int dimension : dimensions) {
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

void require_grid_channels(int channels, const char* operation) {
    if (channels > kMaxGridY) {
        throw std::length_error(std::string(operation) +
                                " channels exceed CUDA grid.y extent");
    }
}

int checked_conv1d_output_length(int input_length, int kernel, int stride,
                                 int padding, int dilation) {
    const std::int64_t receptive =
        static_cast<std::int64_t>(dilation) * (kernel - 1);
    if (receptive > std::numeric_limits<int>::max()) {
        throw std::length_error("VITS conv1d receptive field exceeds int range");
    }
    const std::int64_t numerator = static_cast<std::int64_t>(input_length) +
                                   2LL * padding - receptive - 1;
    if (numerator < 0) {
        throw std::invalid_argument("VITS conv1d kernel exceeds padded input");
    }
    const std::int64_t output_length = numerator / stride + 1;
    if (output_length <= 0 || output_length > std::numeric_limits<int>::max()) {
        throw std::length_error("VITS conv1d output length exceeds int range");
    }

    const std::int64_t last_origin = (output_length - 1) * stride;
    const std::int64_t last_access = last_origin - padding + receptive;
    if (last_origin > std::numeric_limits<int>::max() ||
        last_access > std::numeric_limits<int>::max()) {
        throw std::length_error("VITS conv1d device index exceeds int range");
    }
    return static_cast<int>(output_length);
}

int checked_conv_transpose1d_output_length(int input_length, int kernel,
                                           int stride, int padding) {
    const std::int64_t expanded =
        static_cast<std::int64_t>(input_length - 1) * stride;
    const std::int64_t output_length = expanded - 2LL * padding + kernel;
    if (output_length <= 0) {
        throw std::invalid_argument("VITS conv_transpose1d output is empty");
    }
    if (expanded > std::numeric_limits<int>::max() ||
        output_length > std::numeric_limits<int>::max() ||
        output_length - 1 + padding > std::numeric_limits<int>::max()) {
        throw std::length_error(
            "VITS conv_transpose1d output or device index exceeds int range");
    }
    return static_cast<int>(output_length);
}

// input [Cin,L], weight [Cout, Cin/groups, K], output [Cout, out_len]
__global__ void conv1d_kernel(const float* input, const float* weight,
                              const float* bias, float* output,
                              int Cin, int L, int Cout, int K,
                              int stride, int pad, int dilation, int groups,
                              int out_len) {
    int ot = blockIdx.x * blockDim.x + threadIdx.x;
    int co = blockIdx.y;
    if (ot >= out_len || co >= Cout) return;

    int in_per_group = Cin / groups;
    int out_per_group = Cout / groups;
    int g = co / out_per_group;
    int ci_start = g * in_per_group;

    float sum = bias ? bias[co] : 0.0f;
    int t0 = ot * stride - pad;
    for (int ci = 0; ci < in_per_group; ++ci) {
        const float* in_c = input + (ci_start + ci) * L;
        const float* w = weight + (co * in_per_group + ci) * K;
        for (int k = 0; k < K; ++k) {
            int t = t0 + k * dilation;
            if (t >= 0 && t < L) sum += in_c[t] * w[k];
        }
    }
    output[co * out_len + ot] = sum;
}

// input [Cin,L], weight [Cin, Cout, K], output [Cout, out_len]
__global__ void conv_transpose1d_kernel(const float* input, const float* weight,
                                        const float* bias, float* output,
                                        int Cin, int L, int Cout, int K,
                                        int stride, int pad, int out_len) {
    int ot = blockIdx.x * blockDim.x + threadIdx.x;
    int co = blockIdx.y;
    if (ot >= out_len || co >= Cout) return;

    float sum = bias ? bias[co] : 0.0f;
    for (int k = 0; k < K; ++k) {
        int num = ot + pad - k;
        if (num % stride != 0) continue;
        int t = num / stride;
        if (t < 0 || t >= L) continue;
        for (int ci = 0; ci < Cin; ++ci) {
            sum += input[ci * L + t] * weight[(ci * Cout + co) * K + k];
        }
    }
    output[co * out_len + ot] = sum;
}

__global__ void leaky_relu_kernel(float* data, int size, float slope) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) { float x = data[i]; data[i] = x >= 0 ? x : slope * x; }
}

// At most 16 MiB, reused across groups and temporal tiles in one operation.
constexpr int kColumnElements = 4 * 1024 * 1024;
constexpr int kTileFrames = 1024;

__global__ void conv_columns_kernel(const float* input, float* columns, int L,
                                    int K, int rows, int tile, int start,
                                    int stride, int pad, int dilation) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * tile) return;
    const int row = i / tile;
    const int t = (start + i % tile) * stride - pad + (row % K) * dilation;
    columns[i] = (t >= 0 && t < L) ? input[(row / K) * L + t] : 0.0f;
}

__global__ void conv_bias_kernel(float* output, const float* bias, int length,
                                 int channels, int start, int tile) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < channels * tile)
        output[(i / tile) * length + start + i % tile] += bias[i / tile];
}

__global__ void transpose_gather_kernel(const float* columns, const float* bias,
                                        float* output, int channels, int K,
                                        int stride, int pad, int length,
                                        int start, int tile, int first, int inputs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels * tile) return;
    const int co = i / tile, t = start + i % tile;
    float value = bias ? bias[co] : 0.0f;
    for (int k = 0; k < K; ++k) {
        const int num = t + pad - k;
        if (num % stride != 0) continue;
        const int it = num / stride - first;
        if (it >= 0 && it < inputs) value += columns[(co * K + k) * inputs + it];
    }
    output[co * length + t] = value;
}

__global__ void tanh_kernel(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) data[i] = tanhf(data[i]);
}

__global__ void relative_softmax_kernel(float* scores, const float* q,
                                        const float* relative_k, int T, int D,
                                        int window, float scale) {
    const int row = blockIdx.x;
    __shared__ float reduction[256];
    float maximum = -CUDART_INF_F;
    for (int col = threadIdx.x; col < T; col += blockDim.x) {
        const int relative = col - row;
        float extra = 0.0f;
        if (relative >= -window && relative <= window) {
            for (int d = 0; d < D; ++d)
                extra += (q[d * T + row] * scale) * relative_k[(relative + window) * D + d];
        }
        const float value = scores[row * T + col] + extra;
        scores[row * T + col] = value;
        maximum = fmaxf(maximum, value);
    }
    reduction[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    maximum = reduction[0];
    // All readers must finish before the shared maximum is reused for sums.
    __syncthreads();
    float sum = 0.0f;
    for (int col = threadIdx.x; col < T; col += blockDim.x) {
        const float value = expf(scores[row * T + col] - maximum);
        scores[row * T + col] = value;
        sum += value;
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    sum = reduction[0];
    for (int col = threadIdx.x; col < T; col += blockDim.x)
        scores[row * T + col] /= sum;
}

__global__ void relative_values_kernel(float* output, const float* scores,
                                       const float* relative_v, int T, int D, int window) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= T * D) return;
    const int row = index % T, d = index / T;
    float extra = 0.0f;
    const int first = max(0, row - window), last = min(T - 1, row + window);
    for (int col = first; col <= last; ++col)
        extra += scores[row * T + col] * relative_v[(col - row + window) * D + d];
    output[index] += extra;
}

}  // namespace

std::vector<float> relative_attention(const float* q, const float* k, const float* v,
                                     const float* relative_k, const float* relative_v,
                                     int frames, int heads, int head_dim, int window) {
    require_pointer(q, "VITS attention");
    require_pointer(k, "VITS attention");
    require_pointer(v, "VITS attention");
    require_pointer(relative_k, "VITS attention");
    require_pointer(relative_v, "VITS attention");
    if (window < 0 || window > (std::numeric_limits<int>::max() - 1) / 2)
        throw std::invalid_argument("invalid relative attention window");
    const auto count = checked_element_count({heads, head_dim, frames}, "VITS attention");
    const auto score_count = checked_element_count({frames, frames}, "VITS attention scores");
    const auto relative_count = checked_element_count({2 * window + 1, head_dim}, "VITS relative embeddings");
    const int head_count = static_cast<int>(count / heads);
    core::CudaBuffer dq(count), dk(count), dv(count), dout(count), ds(score_count);
    core::CudaBuffer drk(relative_count), drv(relative_count);
    CK(CUDA_COPY(dq.data(), q, count * sizeof(float), cudaMemcpyHostToDevice));
    CK(CUDA_COPY(dk.data(), k, count * sizeof(float), cudaMemcpyHostToDevice));
    CK(CUDA_COPY(dv.data(), v, count * sizeof(float), cudaMemcpyHostToDevice));
    CK(CUDA_COPY(drk.data(), relative_k, relative_count * sizeof(float), cudaMemcpyHostToDevice));
    CK(CUDA_COPY(drv.data(), relative_v, relative_count * sizeof(float), cudaMemcpyHostToDevice));
    BlasHandle handle;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const float one = 1.0f, zero = 0.0f;
    for (int head = 0; head < heads; ++head) {
        const int offset = head * head_count;
        BEFORE_KERNEL_LAUNCH();
        check_blas(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_T,
                              frames, frames, head_dim, &scale, dk.data() + offset, frames,
                              dq.data() + offset, frames, &zero, ds.data(), frames));
        AFTER_KERNEL_LAUNCH();
        BEFORE_KERNEL_LAUNCH();
        relative_softmax_kernel<<<frames, 256>>>(ds.data(), dq.data() + offset,
                                                drk.data(), frames, head_dim, window, scale);
        AFTER_KERNEL_LAUNCH();
        BEFORE_KERNEL_LAUNCH();
        check_blas(cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                              frames, head_dim, frames, &one, ds.data(), frames,
                              dv.data() + offset, frames, &zero, dout.data() + offset, frames));
        AFTER_KERNEL_LAUNCH();
        BEFORE_KERNEL_LAUNCH();
        relative_values_kernel<<<1 + (head_count - 1) / 256, 256>>>(
            dout.data() + offset, ds.data(), drv.data(), frames, head_dim, window);
        AFTER_KERNEL_LAUNCH();
    }
    std::vector<float> output(count);
    CK(CUDA_COPY(output.data(), dout.data(), count * sizeof(float), cudaMemcpyDeviceToHost));
    return output;
}

std::vector<float> conv1d(const float* input, int Cin, int L,
                          const float* weight, const float* bias,
                          int Cout, int K, int stride, int pad,
                          int dilation, int groups) {
    require_pointer(input, "VITS conv1d");
    require_pointer(weight, "VITS conv1d");
    if (Cin <= 0 || L <= 0 || Cout <= 0 || K <= 0 || stride <= 0 ||
        pad < 0 || dilation <= 0 || groups <= 0 || Cin % groups != 0 ||
        Cout % groups != 0) {
        throw std::invalid_argument(
            "VITS conv1d requires positive dimensions/stride/dilation/groups, "
            "nonnegative padding, and divisible channels");
    }
    require_grid_channels(Cout, "VITS conv1d");
    const int out_len =
        checked_conv1d_output_length(L, K, stride, pad, dilation);
    const std::size_t input_count =
        checked_element_count({Cin, L}, "VITS conv1d input");
    const std::size_t weight_count = checked_element_count(
        {Cout, Cin / groups, K}, "VITS conv1d weights");
    const std::size_t output_count =
        checked_element_count({Cout, out_len}, "VITS conv1d output");
    std::vector<float> output(output_count);

    core::CudaBuffer d_in(input_count);
    core::CudaBuffer d_w(weight_count);
    core::CudaBuffer d_b;
    core::CudaBuffer d_out(output_count);
    CK(CUDA_COPY(d_in.data(), input, input_count * sizeof(float), cudaMemcpyHostToDevice));
    CK(CUDA_COPY(d_w.data(), weight, weight_count * sizeof(float), cudaMemcpyHostToDevice));
    if (bias) {
        d_b.allocate(static_cast<std::size_t>(Cout));
        CK(CUDA_COPY(d_b.data(), bias, Cout * sizeof(float), cudaMemcpyHostToDevice));
    }

    const int rows = (Cin / groups) * K;
    if (Cin / groups >= 8 && Cout / groups >= 8 && rows <= kColumnElements) {
        const int capacity = std::min({out_len, kTileFrames, kColumnElements / rows});
        core::CudaBuffer columns(static_cast<std::size_t>(rows) * capacity);
        BlasHandle handle;
        const float one = 1.0f, zero = 0.0f;
        const int channels = Cout / groups;
        for (int group = 0; group < groups; ++group) {
            for (int start = 0; start < out_len;) {
                const int tile = std::min(capacity, out_len - start);
                BEFORE_KERNEL_LAUNCH();
                conv_columns_kernel<<<1 + (rows * tile - 1) / 256, 256>>>(
                    d_in.data() + group * (Cin / groups) * L, columns.data(),
                    L, K, rows, tile, start, stride, pad, dilation);
                AFTER_KERNEL_LAUNCH();
                float* output_group = d_out.data() + group * channels * out_len;
                BEFORE_KERNEL_LAUNCH();
                check_blas(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                    tile, channels, rows, &one, columns.data(), tile,
                    d_w.data() + group * channels * rows, rows, &zero,
                    output_group + start, out_len));
                AFTER_KERNEL_LAUNCH();
                if (bias) {
                    BEFORE_KERNEL_LAUNCH();
                    conv_bias_kernel<<<1 + (channels * tile - 1) / 256, 256>>>(
                        output_group, d_b.data() + group * channels, out_len, channels, start, tile);
                    AFTER_KERNEL_LAUNCH();
                }
                start += tile;
            }
        }
    } else {
        dim3 grid(1 + (out_len - 1) / 128, Cout);
        BEFORE_KERNEL_LAUNCH();
        conv1d_kernel<<<grid, 128>>>(d_in.data(), d_w.data(), d_b.data(), d_out.data(),
                                    Cin, L, Cout, K, stride, pad, dilation, groups, out_len);
        AFTER_KERNEL_LAUNCH();
    }
    CK(CUDA_COPY(output.data(), d_out.data(), output_count * sizeof(float),
                 cudaMemcpyDeviceToHost));
    return output;
}

std::vector<float> conv_transpose1d(const float* input, int Cin, int L,
                                    const float* weight, const float* bias,
                                    int Cout, int K, int stride, int pad) {
    require_pointer(input, "VITS conv_transpose1d");
    require_pointer(weight, "VITS conv_transpose1d");
    if (Cin <= 0 || L <= 0 || Cout <= 0 || K <= 0 || stride <= 0 || pad < 0) {
        throw std::invalid_argument(
            "VITS conv_transpose1d requires positive dimensions/stride and "
            "nonnegative padding");
    }
    require_grid_channels(Cout, "VITS conv_transpose1d");
    const int out_len =
        checked_conv_transpose1d_output_length(L, K, stride, pad);
    const std::size_t input_count =
        checked_element_count({Cin, L}, "VITS conv_transpose1d input");
    const std::size_t weight_count = checked_element_count(
        {Cin, Cout, K}, "VITS conv_transpose1d weights");
    const std::size_t output_count = checked_element_count(
        {Cout, out_len}, "VITS conv_transpose1d output");
    std::vector<float> output(output_count);

    core::CudaBuffer d_in(input_count);
    core::CudaBuffer d_w(weight_count);
    core::CudaBuffer d_b;
    core::CudaBuffer d_out(output_count);
    CK(CUDA_COPY(d_in.data(), input, input_count * sizeof(float), cudaMemcpyHostToDevice));
    CK(CUDA_COPY(d_w.data(), weight, weight_count * sizeof(float), cudaMemcpyHostToDevice));
    if (bias) {
        d_b.allocate(static_cast<std::size_t>(Cout));
        CK(CUDA_COPY(d_b.data(), bias, Cout * sizeof(float), cudaMemcpyHostToDevice));
    }

    const int rows = Cout * K;  // bounded by checked weight_count above
    const int capacity = std::min(kTileFrames, kColumnElements / rows);
    if (Cin >= 8 && Cout >= 8 && capacity > K) {
        core::CudaBuffer columns(static_cast<std::size_t>(rows) * capacity);
        BlasHandle handle;
        const float one = 1.0f, zero = 0.0f;
        const int tile_limit = static_cast<int>(std::min<std::int64_t>(
            out_len, static_cast<std::int64_t>(capacity - K) * stride));
        for (int start = 0; start < out_len;) {
            const int tile = std::min(tile_limit, out_len - start);
            const std::int64_t lower = static_cast<std::int64_t>(start) + pad - (K - 1);
            const int first = static_cast<int>(std::max<std::int64_t>(
                0, lower > 0 ? (lower + stride - 1) / stride : lower / stride));
            const int last = static_cast<int>(std::min<std::int64_t>(
                L - 1, (static_cast<std::int64_t>(start) + tile - 1 + pad) / stride));
            const int inputs = std::max(0, last - first + 1);
            if (inputs > capacity) throw std::logic_error("transpose convolution tile exceeds workspace");
            if (inputs) {
                BEFORE_KERNEL_LAUNCH();
                check_blas(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_T,
                    inputs, rows, Cin, &one, d_in.data() + first, L,
                    d_w.data(), rows, &zero, columns.data(), inputs));
                AFTER_KERNEL_LAUNCH();
            }
            BEFORE_KERNEL_LAUNCH();
            transpose_gather_kernel<<<1 + (Cout * tile - 1) / 256, 256>>>(
                columns.data(), d_b.data(), d_out.data(), Cout, K, stride, pad,
                out_len, start, tile, first, inputs);
            AFTER_KERNEL_LAUNCH();
            start += tile;
        }
    } else {
        dim3 grid(1 + (out_len - 1) / 128, Cout);
        BEFORE_KERNEL_LAUNCH();
        conv_transpose1d_kernel<<<grid, 128>>>(d_in.data(), d_w.data(), d_b.data(),
                                             d_out.data(), Cin, L, Cout, K, stride, pad, out_len);
        AFTER_KERNEL_LAUNCH();
    }
    CK(CUDA_COPY(output.data(), d_out.data(), output_count * sizeof(float),
                 cudaMemcpyDeviceToHost));
    return output;
}

void leaky_relu_inplace(float* data, int size, float slope) {
    require_pointer(data, "VITS leaky_relu");
    checked_element_count({size}, "VITS leaky_relu input");
    if (!std::isfinite(slope)) {
        throw std::invalid_argument("VITS leaky_relu requires a finite slope");
    }
    core::CudaBuffer d(static_cast<std::size_t>(size));
    CK(CUDA_COPY(d.data(), data, size * sizeof(float), cudaMemcpyHostToDevice));
    int block = 256, grid = 1 + (size - 1) / block;
    BEFORE_KERNEL_LAUNCH();
    leaky_relu_kernel<<<grid, block>>>(d.data(), size, slope);
    AFTER_KERNEL_LAUNCH();
    CK(CUDA_COPY(data, d.data(), size * sizeof(float), cudaMemcpyDeviceToHost));
}

void tanh_inplace(float* data, int size) {
    require_pointer(data, "VITS tanh");
    checked_element_count({size}, "VITS tanh input");
    core::CudaBuffer d(static_cast<std::size_t>(size));
    CK(CUDA_COPY(d.data(), data, size * sizeof(float), cudaMemcpyHostToDevice));
    int block = 256, grid = 1 + (size - 1) / block;
    BEFORE_KERNEL_LAUNCH();
    tanh_kernel<<<grid, block>>>(d.data(), size);
    AFTER_KERNEL_LAUNCH();
    CK(CUDA_COPY(data, d.data(), size * sizeof(float), cudaMemcpyDeviceToHost));
}

}  // namespace voxmutatio::synthesizer::ops

#undef AFTER_KERNEL_LAUNCH
#undef BEFORE_KERNEL_LAUNCH
#undef CUDA_COPY
