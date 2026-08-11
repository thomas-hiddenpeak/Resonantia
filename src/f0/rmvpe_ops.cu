// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// 2D CNN operations for RMVPE, implemented with CUDA kernels.

#include "voxmutatio/f0/rmvpe_ops.h"

#include "voxmutatio/core/cuda_buffer.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace voxmutatio::f0::ops {

namespace {

inline void cuda_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                                 std::to_string(line) + ": " +
                                 cudaGetErrorString(e));
    }
}
#define CK(e) cuda_check((e), __FILE__, __LINE__)

// Conv2d NxN, stride 1, padding P. input [Cin,H,W], weight [Cout,Cin,K,K].
__global__ void conv2d_kernel(const float* input, const float* weight,
                              const float* bias, float* output,
                              int Cin, int H, int W, int Cout, int K, int P) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int co = blockIdx.z;
    if (ow >= W || oh >= H || co >= Cout) return;

    float sum = bias ? bias[co] : 0.0f;
    for (int ci = 0; ci < Cin; ++ci) {
        const float* in_c = input + ci * H * W;
        const float* w = weight + ((co * Cin + ci) * K) * K;
        for (int kh = 0; kh < K; ++kh) {
            int ih = oh + kh - P;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < K; ++kw) {
                int iw = ow + kw - P;
                if (iw < 0 || iw >= W) continue;
                sum += in_c[ih * W + iw] * w[kh * K + kw];
            }
        }
    }
    output[(co * H + oh) * W + ow] = sum;
}

__global__ void batchnorm_kernel(float* data, int C, int HW,
                                 const float* gamma, const float* beta,
                                 const float* mean, const float* var, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * HW;
    if (idx >= total) return;
    int c = idx / HW;
    float inv = rsqrtf(var[c] + eps);
    data[idx] = (data[idx] - mean[c]) * inv * gamma[c] + beta[c];
}

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = fmaxf(0.0f, data[idx]);
}

__global__ void sigmoid_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = 1.0f / (1.0f + expf(-data[idx]));
}

// AvgPool2d, kernel (kh,kw), stride == kernel.
__global__ void avgpool_kernel(const float* input, float* output,
                               int C, int H, int W, int kh, int kw,
                               int OH, int OW) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if (ow >= OW || oh >= OH || c >= C) return;
    const float* in_c = input + c * H * W;
    float sum = 0.0f;
    for (int i = 0; i < kh; ++i)
        for (int j = 0; j < kw; ++j)
            sum += in_c[(oh * kh + i) * W + (ow * kw + j)];
    output[(c * OH + oh) * OW + ow] = sum / (kh * kw);
}

// ConvTranspose2d, k=3, stride=2, pad=1, output_padding=1. weight [Cin,Cout,3,3].
// output size = 2*H, 2*W. For each output (co, oh, ow):
//   sum over ci, kh, kw where (oh + pad - kh) % stride == 0 etc.
__global__ void conv_transpose_kernel(const float* input, const float* weight,
                                      float* output, int Cin, int H, int W,
                                      int Cout, int OH, int OW,
                                      int K, int stride, int pad) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int co = blockIdx.z;
    if (ow >= OW || oh >= OH || co >= Cout) return;

    float sum = 0.0f;
    for (int kh = 0; kh < K; ++kh) {
        int ih_num = oh + pad - kh;
        if (ih_num % stride != 0) continue;
        int ih = ih_num / stride;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < K; ++kw) {
            int iw_num = ow + pad - kw;
            if (iw_num % stride != 0) continue;
            int iw = iw_num / stride;
            if (iw < 0 || iw >= W) continue;
            for (int ci = 0; ci < Cin; ++ci) {
                float in_v = input[(ci * H + ih) * W + iw];
                float w = weight[((ci * Cout + co) * K + kh) * K + kw];
                sum += in_v * w;
            }
        }
    }
    output[(co * OH + oh) * OW + ow] = sum;
}

std::vector<float> conv2d_generic(const float* input, int Cin, int H, int W,
                                  const float* weight, const float* bias,
                                  int Cout, int K, int P) {
    int in_size = Cin * H * W;
    int w_size = Cout * Cin * K * K;
    int out_size = Cout * H * W;
    std::vector<float> output(out_size);

    core::CudaBuffer d_in(static_cast<std::size_t>(in_size));
    core::CudaBuffer d_w(static_cast<std::size_t>(w_size));
    core::CudaBuffer d_b;
    core::CudaBuffer d_out(static_cast<std::size_t>(out_size));
    CK(cudaMemcpy(d_in.data(), input, in_size * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w.data(), weight, w_size * sizeof(float), cudaMemcpyHostToDevice));
    if (bias) {
        d_b.allocate(static_cast<std::size_t>(Cout));
        CK(cudaMemcpy(d_b.data(), bias, Cout * sizeof(float), cudaMemcpyHostToDevice));
    }

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16, Cout);
    conv2d_kernel<<<grid, block>>>(d_in.data(), d_w.data(), d_b.data(), d_out.data(),
                                   Cin, H, W, Cout, K, P);
    CK(cudaMemcpy(output.data(), d_out.data(), out_size * sizeof(float),
                  cudaMemcpyDeviceToHost));
    return output;
}

}  // namespace

std::vector<float> conv2d_3x3(const float* input, int Cin, int H, int W,
                              const float* weight, const float* bias, int Cout) {
    return conv2d_generic(input, Cin, H, W, weight, bias, Cout, 3, 1);
}

std::vector<float> conv2d_1x1(const float* input, int Cin, int H, int W,
                              const float* weight, const float* bias, int Cout) {
    return conv2d_generic(input, Cin, H, W, weight, bias, Cout, 1, 0);
}

void batchnorm2d(float* data, int C, int H, int W,
                 const float* gamma, const float* beta,
                 const float* mean, const float* var, float eps) {
    int HW = H * W;
    int total = C * HW;
    core::CudaBuffer d_data(static_cast<std::size_t>(total));
    core::CudaBuffer d_g(static_cast<std::size_t>(C));
    core::CudaBuffer d_be(static_cast<std::size_t>(C));
    core::CudaBuffer d_m(static_cast<std::size_t>(C));
    core::CudaBuffer d_v(static_cast<std::size_t>(C));
    CK(cudaMemcpy(d_data.data(), data, total * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_g.data(), gamma, C * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_be.data(), beta, C * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_m.data(), mean, C * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v.data(), var, C * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256, grid = (total + block - 1) / block;
    batchnorm_kernel<<<grid, block>>>(d_data.data(), C, HW, d_g.data(), d_be.data(),
                                      d_m.data(), d_v.data(), eps);
    CK(cudaMemcpy(data, d_data.data(), total * sizeof(float), cudaMemcpyDeviceToHost));
}

void relu_inplace(float* data, int size) {
    core::CudaBuffer d_data(static_cast<std::size_t>(size));
    CK(cudaMemcpy(d_data.data(), data, size * sizeof(float), cudaMemcpyHostToDevice));
    int block = 256, grid = (size + block - 1) / block;
    relu_kernel<<<grid, block>>>(d_data.data(), size);
    CK(cudaMemcpy(data, d_data.data(), size * sizeof(float), cudaMemcpyDeviceToHost));
}

void sigmoid_inplace(float* data, int size) {
    core::CudaBuffer d_data(static_cast<std::size_t>(size));
    CK(cudaMemcpy(d_data.data(), data, size * sizeof(float), cudaMemcpyHostToDevice));
    int block = 256, grid = (size + block - 1) / block;
    sigmoid_kernel<<<grid, block>>>(d_data.data(), size);
    CK(cudaMemcpy(data, d_data.data(), size * sizeof(float), cudaMemcpyDeviceToHost));
}

std::vector<float> avgpool2d(const float* input, int C, int H, int W,
                             int kh, int kw) {
    int OH = H / kh, OW = W / kw;
    std::vector<float> output(C * OH * OW);
    core::CudaBuffer d_in(static_cast<std::size_t>(C) * H * W);
    core::CudaBuffer d_out(static_cast<std::size_t>(C) * OH * OW);
    CK(cudaMemcpy(d_in.data(), input, C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((OW + 15) / 16, (OH + 15) / 16, C);
    avgpool_kernel<<<grid, block>>>(d_in.data(), d_out.data(), C, H, W, kh, kw, OH, OW);
    CK(cudaMemcpy(output.data(), d_out.data(), C * OH * OW * sizeof(float),
                  cudaMemcpyDeviceToHost));
    return output;
}

std::vector<float> conv_transpose2d_s2(const float* input, int Cin, int H, int W,
                                       const float* weight, int Cout) {
    int K = 3, stride = 2, pad = 1;
    int OH = 2 * H, OW = 2 * W;  // (H-1)*2 - 2 + 3 + 1 = 2H
    std::vector<float> output(Cout * OH * OW);
    int w_size = Cin * Cout * K * K;
    core::CudaBuffer d_in(static_cast<std::size_t>(Cin) * H * W);
    core::CudaBuffer d_w(static_cast<std::size_t>(w_size));
    core::CudaBuffer d_out(static_cast<std::size_t>(Cout) * OH * OW);
    CK(cudaMemcpy(d_in.data(), input, Cin * H * W * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w.data(), weight, w_size * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((OW + 15) / 16, (OH + 15) / 16, Cout);
    conv_transpose_kernel<<<grid, block>>>(d_in.data(), d_w.data(), d_out.data(), Cin, H, W,
                                           Cout, OH, OW, K, stride, pad);
    CK(cudaMemcpy(output.data(), d_out.data(), Cout * OH * OW * sizeof(float),
                  cudaMemcpyDeviceToHost));
    return output;
}

std::vector<float> concat_channels(const float* a, int Ca,
                                    const float* b, int Cb, int H, int W) {
    std::vector<float> out((Ca + Cb) * H * W);
    std::copy(a, a + Ca * H * W, out.begin());
    std::copy(b, b + Cb * H * W, out.begin() + Ca * H * W);
    return out;
}

}  // namespace voxmutatio::f0::ops
