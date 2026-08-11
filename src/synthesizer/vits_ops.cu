// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/synthesizer/vits_ops.h"

#include "voxmutatio/core/cuda_buffer.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
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
    int out_len = (L + 2 * pad - dilation * (K - 1) - 1) / stride + 1;
    std::vector<float> output(Cout * out_len);
    int w_size = Cout * (Cin / groups) * K;

    core::CudaBuffer d_in(static_cast<std::size_t>(Cin) * L);
    core::CudaBuffer d_w(static_cast<std::size_t>(w_size));
    core::CudaBuffer d_b;
    core::CudaBuffer d_out(static_cast<std::size_t>(Cout) * out_len);
    CK(cudaMemcpy(d_in.data(), input, Cin * L * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w.data(), weight, w_size * sizeof(float), cudaMemcpyHostToDevice));
    if (bias) {
        d_b.allocate(static_cast<std::size_t>(Cout));
        CK(cudaMemcpy(d_b.data(), bias, Cout * sizeof(float), cudaMemcpyHostToDevice));
    }

    int block = 128;
    dim3 grid((out_len + block - 1) / block, Cout);
    conv1d_kernel<<<grid, block>>>(d_in.data(), d_w.data(), d_b.data(), d_out.data(),
                                   Cin, L, Cout, K,
                                   stride, pad, dilation, groups, out_len);
    CK(cudaMemcpy(output.data(), d_out.data(), Cout * out_len * sizeof(float),
                  cudaMemcpyDeviceToHost));
    return output;
}

std::vector<float> conv_transpose1d(const float* input, int Cin, int L,
                                    const float* weight, const float* bias,
                                    int Cout, int K, int stride, int pad) {
    int out_len = (L - 1) * stride - 2 * pad + K;
    std::vector<float> output(Cout * out_len);
    int w_size = Cin * Cout * K;

    core::CudaBuffer d_in(static_cast<std::size_t>(Cin) * L);
    core::CudaBuffer d_w(static_cast<std::size_t>(w_size));
    core::CudaBuffer d_b;
    core::CudaBuffer d_out(static_cast<std::size_t>(Cout) * out_len);
    CK(cudaMemcpy(d_in.data(), input, Cin * L * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w.data(), weight, w_size * sizeof(float), cudaMemcpyHostToDevice));
    if (bias) {
        d_b.allocate(static_cast<std::size_t>(Cout));
        CK(cudaMemcpy(d_b.data(), bias, Cout * sizeof(float), cudaMemcpyHostToDevice));
    }

    int block = 128;
    dim3 grid((out_len + block - 1) / block, Cout);
    conv_transpose1d_kernel<<<grid, block>>>(d_in.data(), d_w.data(), d_b.data(),
                                             d_out.data(), Cin, L, Cout, K,
                                             stride, pad, out_len);
    CK(cudaMemcpy(output.data(), d_out.data(), Cout * out_len * sizeof(float),
                  cudaMemcpyDeviceToHost));
    return output;
}

void leaky_relu_inplace(float* data, int size, float slope) {
    core::CudaBuffer d(static_cast<std::size_t>(size));
    CK(cudaMemcpy(d.data(), data, size * sizeof(float), cudaMemcpyHostToDevice));
    int block = 256, grid = (size + block - 1) / block;
    leaky_relu_kernel<<<grid, block>>>(d.data(), size, slope);
    CK(cudaMemcpy(data, d.data(), size * sizeof(float), cudaMemcpyDeviceToHost));
}

void tanh_inplace(float* data, int size) {
    core::CudaBuffer d(static_cast<std::size_t>(size));
    CK(cudaMemcpy(d.data(), data, size * sizeof(float), cudaMemcpyHostToDevice));
    int block = 256, grid = (size + block - 1) / block;
    tanh_kernel<<<grid, block>>>(d.data(), size);
    CK(cudaMemcpy(data, d.data(), size * sizeof(float), cudaMemcpyDeviceToHost));
}

}  // namespace voxmutatio::synthesizer::ops
