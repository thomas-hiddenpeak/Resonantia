// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// 2D CNN operations for RMVPE, implemented with CUDA kernels.

#include "voxmutatio/f0/rmvpe_ops.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace voxmutatio::f0::ops {

namespace {

inline void cuda_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
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

    float *d_in, *d_w, *d_b = nullptr, *d_out;
    CK(cudaMalloc(&d_in, in_size * sizeof(float)));
    CK(cudaMalloc(&d_w, w_size * sizeof(float)));
    CK(cudaMalloc(&d_out, out_size * sizeof(float)));
    CK(cudaMemcpy(d_in, input, in_size * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w, weight, w_size * sizeof(float), cudaMemcpyHostToDevice));
    if (bias) {
        CK(cudaMalloc(&d_b, Cout * sizeof(float)));
        CK(cudaMemcpy(d_b, bias, Cout * sizeof(float), cudaMemcpyHostToDevice));
    }

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16, Cout);
    conv2d_kernel<<<grid, block>>>(d_in, d_w, d_b, d_out, Cin, H, W, Cout, K, P);
    CK(cudaMemcpy(output.data(), d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
    if (d_b) cudaFree(d_b);
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
    float *d_data, *d_g, *d_be, *d_m, *d_v;
    CK(cudaMalloc(&d_data, total * sizeof(float)));
    CK(cudaMalloc(&d_g, C * sizeof(float)));
    CK(cudaMalloc(&d_be, C * sizeof(float)));
    CK(cudaMalloc(&d_m, C * sizeof(float)));
    CK(cudaMalloc(&d_v, C * sizeof(float)));
    CK(cudaMemcpy(d_data, data, total * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_g, gamma, C * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_be, beta, C * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_m, mean, C * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v, var, C * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256, grid = (total + block - 1) / block;
    batchnorm_kernel<<<grid, block>>>(d_data, C, HW, d_g, d_be, d_m, d_v, eps);
    CK(cudaMemcpy(data, d_data, total * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_data); cudaFree(d_g); cudaFree(d_be); cudaFree(d_m); cudaFree(d_v);
}

void relu_inplace(float* data, int size) {
    float* d_data;
    CK(cudaMalloc(&d_data, size * sizeof(float)));
    CK(cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice));
    int block = 256, grid = (size + block - 1) / block;
    relu_kernel<<<grid, block>>>(d_data, size);
    CK(cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
}

void sigmoid_inplace(float* data, int size) {
    float* d_data;
    CK(cudaMalloc(&d_data, size * sizeof(float)));
    CK(cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice));
    int block = 256, grid = (size + block - 1) / block;
    sigmoid_kernel<<<grid, block>>>(d_data, size);
    CK(cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
}

std::vector<float> avgpool2d(const float* input, int C, int H, int W,
                             int kh, int kw) {
    int OH = H / kh, OW = W / kw;
    std::vector<float> output(C * OH * OW);
    float *d_in, *d_out;
    CK(cudaMalloc(&d_in, C * H * W * sizeof(float)));
    CK(cudaMalloc(&d_out, C * OH * OW * sizeof(float)));
    CK(cudaMemcpy(d_in, input, C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((OW + 15) / 16, (OH + 15) / 16, C);
    avgpool_kernel<<<grid, block>>>(d_in, d_out, C, H, W, kh, kw, OH, OW);
    CK(cudaMemcpy(output.data(), d_out, C * OH * OW * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out);
    return output;
}

std::vector<float> conv_transpose2d_s2(const float* input, int Cin, int H, int W,
                                       const float* weight, int Cout) {
    int K = 3, stride = 2, pad = 1;
    int OH = 2 * H, OW = 2 * W;  // (H-1)*2 - 2 + 3 + 1 = 2H
    std::vector<float> output(Cout * OH * OW);
    int w_size = Cin * Cout * K * K;
    float *d_in, *d_w, *d_out;
    CK(cudaMalloc(&d_in, Cin * H * W * sizeof(float)));
    CK(cudaMalloc(&d_w, w_size * sizeof(float)));
    CK(cudaMalloc(&d_out, Cout * OH * OW * sizeof(float)));
    CK(cudaMemcpy(d_in, input, Cin * H * W * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w, weight, w_size * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((OW + 15) / 16, (OH + 15) / 16, Cout);
    conv_transpose_kernel<<<grid, block>>>(d_in, d_w, d_out, Cin, H, W, Cout, OH, OW, K, stride, pad);
    CK(cudaMemcpy(output.data(), d_out, Cout * OH * OW * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
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
