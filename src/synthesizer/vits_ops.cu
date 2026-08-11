// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/synthesizer/vits_ops.h"

#include "voxmutatio/core/cuda_buffer.h"

#if defined(VOXMUTATIO_TEST_CUDA_FAILURE_INJECTION)
#include "cuda_failure_injection.h"
#endif

#include <cuda_runtime.h>

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

__global__ void tanh_kernel(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) data[i] = tanhf(data[i]);
}

}  // namespace

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

    int block = 128;
    dim3 grid(1 + (out_len - 1) / block, Cout);
    BEFORE_KERNEL_LAUNCH();
    conv1d_kernel<<<grid, block>>>(d_in.data(), d_w.data(), d_b.data(), d_out.data(),
                                   Cin, L, Cout, K,
                                   stride, pad, dilation, groups, out_len);
    AFTER_KERNEL_LAUNCH();
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

    int block = 128;
    dim3 grid(1 + (out_len - 1) / block, Cout);
    BEFORE_KERNEL_LAUNCH();
    conv_transpose1d_kernel<<<grid, block>>>(d_in.data(), d_w.data(), d_b.data(),
                                             d_out.data(), Cin, L, Cout, K,
                                             stride, pad, out_len);
    AFTER_KERNEL_LAUNCH();
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
