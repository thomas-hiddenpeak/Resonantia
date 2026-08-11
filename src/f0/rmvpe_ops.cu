// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// 2D CNN operations for RMVPE, implemented with CUDA kernels.

#include "voxmutatio/f0/rmvpe_ops.h"

#include "voxmutatio/core/cuda_buffer.h"

#if defined(VOXMUTATIO_TEST_CUDA_FAILURE_INJECTION)
#include "cuda_failure_injection.h"
#endif

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
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

constexpr int kMaxGridYZ = 65'535;

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

void require_grid_yz_extent(int extent, int block, const char* operation) {
    const int blocks = 1 + (extent - 1) / block;
    if (blocks > kMaxGridYZ) {
        throw std::length_error(std::string(operation) +
                                " launch exceeds CUDA y/z grid extent");
    }
}

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
    require_pointer(input, "RMVPE conv2d");
    require_pointer(weight, "RMVPE conv2d");
    const std::size_t in_size =
        checked_element_count({Cin, H, W}, "RMVPE conv2d input");
    const std::size_t w_size =
        checked_element_count({Cout, Cin, K, K}, "RMVPE conv2d weights");
    const std::size_t out_size =
        checked_element_count({Cout, H, W}, "RMVPE conv2d output");
    require_grid_yz_extent(H, 16, "RMVPE conv2d");
    require_grid_yz_extent(Cout, 1, "RMVPE conv2d");
    std::vector<float> output(out_size);

    core::CudaBuffer d_in(in_size);
    core::CudaBuffer d_w(w_size);
    core::CudaBuffer d_b;
    core::CudaBuffer d_out(out_size);
    CK(CUDA_COPY(d_in.data(), input, in_size * sizeof(float), cudaMemcpyHostToDevice));
    CK(CUDA_COPY(d_w.data(), weight, w_size * sizeof(float), cudaMemcpyHostToDevice));
    if (bias) {
        d_b.allocate(static_cast<std::size_t>(Cout));
        CK(CUDA_COPY(d_b.data(), bias, Cout * sizeof(float), cudaMemcpyHostToDevice));
    }

    dim3 block(16, 16);
    dim3 grid(1 + (W - 1) / 16, 1 + (H - 1) / 16, Cout);
    BEFORE_KERNEL_LAUNCH();
    conv2d_kernel<<<grid, block>>>(d_in.data(), d_w.data(), d_b.data(), d_out.data(),
                                   Cin, H, W, Cout, K, P);
    AFTER_KERNEL_LAUNCH();
    CK(CUDA_COPY(output.data(), d_out.data(), out_size * sizeof(float),
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
    require_pointer(data, "RMVPE batchnorm2d");
    require_pointer(gamma, "RMVPE batchnorm2d");
    require_pointer(beta, "RMVPE batchnorm2d");
    require_pointer(mean, "RMVPE batchnorm2d");
    require_pointer(var, "RMVPE batchnorm2d");
    if (!std::isfinite(eps) || eps <= 0.0f) {
        throw std::invalid_argument("RMVPE batchnorm2d requires finite positive epsilon");
    }
    const std::size_t hw_count =
        checked_element_count({H, W}, "RMVPE batchnorm2d plane");
    const std::size_t total_count =
        checked_element_count({C, H, W}, "RMVPE batchnorm2d input");
    const int HW = static_cast<int>(hw_count);
    const int total = static_cast<int>(total_count);
    core::CudaBuffer d_data(total_count);
    core::CudaBuffer d_g(static_cast<std::size_t>(C));
    core::CudaBuffer d_be(static_cast<std::size_t>(C));
    core::CudaBuffer d_m(static_cast<std::size_t>(C));
    core::CudaBuffer d_v(static_cast<std::size_t>(C));
    CK(CUDA_COPY(d_data.data(), data, total * sizeof(float), cudaMemcpyHostToDevice));
    CK(CUDA_COPY(d_g.data(), gamma, C * sizeof(float), cudaMemcpyHostToDevice));
    CK(CUDA_COPY(d_be.data(), beta, C * sizeof(float), cudaMemcpyHostToDevice));
    CK(CUDA_COPY(d_m.data(), mean, C * sizeof(float), cudaMemcpyHostToDevice));
    CK(CUDA_COPY(d_v.data(), var, C * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256, grid = 1 + (total - 1) / block;
    BEFORE_KERNEL_LAUNCH();
    batchnorm_kernel<<<grid, block>>>(d_data.data(), C, HW, d_g.data(), d_be.data(),
                                      d_m.data(), d_v.data(), eps);
    AFTER_KERNEL_LAUNCH();
    CK(CUDA_COPY(data, d_data.data(), total * sizeof(float), cudaMemcpyDeviceToHost));
}

void relu_inplace(float* data, int size) {
    require_pointer(data, "RMVPE relu");
    checked_element_count({size}, "RMVPE relu input");
    core::CudaBuffer d_data(static_cast<std::size_t>(size));
    CK(CUDA_COPY(d_data.data(), data, size * sizeof(float), cudaMemcpyHostToDevice));
    int block = 256, grid = 1 + (size - 1) / block;
    BEFORE_KERNEL_LAUNCH();
    relu_kernel<<<grid, block>>>(d_data.data(), size);
    AFTER_KERNEL_LAUNCH();
    CK(CUDA_COPY(data, d_data.data(), size * sizeof(float), cudaMemcpyDeviceToHost));
}

void sigmoid_inplace(float* data, int size) {
    require_pointer(data, "RMVPE sigmoid");
    checked_element_count({size}, "RMVPE sigmoid input");
    core::CudaBuffer d_data(static_cast<std::size_t>(size));
    CK(CUDA_COPY(d_data.data(), data, size * sizeof(float), cudaMemcpyHostToDevice));
    int block = 256, grid = 1 + (size - 1) / block;
    BEFORE_KERNEL_LAUNCH();
    sigmoid_kernel<<<grid, block>>>(d_data.data(), size);
    AFTER_KERNEL_LAUNCH();
    CK(CUDA_COPY(data, d_data.data(), size * sizeof(float), cudaMemcpyDeviceToHost));
}

std::vector<float> avgpool2d(const float* input, int C, int H, int W,
                             int kh, int kw) {
    require_pointer(input, "RMVPE avgpool2d");
    const std::size_t input_count =
        checked_element_count({C, H, W}, "RMVPE avgpool2d input");
    if (kh <= 0 || kw <= 0 || kh > H || kw > W) {
        throw std::invalid_argument(
            "RMVPE avgpool2d requires a positive kernel within the input");
    }
    const int OH = H / kh;
    const int OW = W / kw;
    const std::size_t output_count =
        checked_element_count({C, OH, OW}, "RMVPE avgpool2d output");
    require_grid_yz_extent(OH, 16, "RMVPE avgpool2d");
    require_grid_yz_extent(C, 1, "RMVPE avgpool2d");
    std::vector<float> output(output_count);
    core::CudaBuffer d_in(input_count);
    core::CudaBuffer d_out(output_count);
    CK(CUDA_COPY(d_in.data(), input, input_count * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid(1 + (OW - 1) / 16, 1 + (OH - 1) / 16, C);
    BEFORE_KERNEL_LAUNCH();
    avgpool_kernel<<<grid, block>>>(d_in.data(), d_out.data(), C, H, W, kh, kw, OH, OW);
    AFTER_KERNEL_LAUNCH();
    CK(CUDA_COPY(output.data(), d_out.data(), output_count * sizeof(float),
                 cudaMemcpyDeviceToHost));
    return output;
}

std::vector<float> conv_transpose2d_s2(const float* input, int Cin, int H, int W,
                                       const float* weight, int Cout) {
    require_pointer(input, "RMVPE conv_transpose2d");
    require_pointer(weight, "RMVPE conv_transpose2d");
    constexpr int K = 3;
    constexpr int stride = 2;
    constexpr int pad = 1;
    const std::size_t input_count =
        checked_element_count({Cin, H, W}, "RMVPE conv_transpose2d input");
    if (H > std::numeric_limits<int>::max() / 2 ||
        W > std::numeric_limits<int>::max() / 2) {
        throw std::length_error("RMVPE conv_transpose2d output extent exceeds int range");
    }
    const int OH = 2 * H;
    const int OW = 2 * W;  // (H-1)*2 - 2 + 3 + 1 = 2H
    const std::size_t weight_count =
        checked_element_count({Cin, Cout, K, K}, "RMVPE conv_transpose2d weights");
    const std::size_t output_count =
        checked_element_count({Cout, OH, OW}, "RMVPE conv_transpose2d output");
    require_grid_yz_extent(OH, 16, "RMVPE conv_transpose2d");
    require_grid_yz_extent(Cout, 1, "RMVPE conv_transpose2d");
    std::vector<float> output(output_count);
    core::CudaBuffer d_in(input_count);
    core::CudaBuffer d_w(weight_count);
    core::CudaBuffer d_out(output_count);
    CK(CUDA_COPY(d_in.data(), input, input_count * sizeof(float), cudaMemcpyHostToDevice));
    CK(CUDA_COPY(d_w.data(), weight, weight_count * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid(1 + (OW - 1) / 16, 1 + (OH - 1) / 16, Cout);
    BEFORE_KERNEL_LAUNCH();
    conv_transpose_kernel<<<grid, block>>>(d_in.data(), d_w.data(), d_out.data(), Cin, H, W,
                                           Cout, OH, OW, K, stride, pad);
    AFTER_KERNEL_LAUNCH();
    CK(CUDA_COPY(output.data(), d_out.data(), output_count * sizeof(float),
                 cudaMemcpyDeviceToHost));
    return output;
}

std::vector<float> concat_channels(const float* a, int Ca,
                                    const float* b, int Cb, int H, int W) {
    require_pointer(a, "RMVPE concat_channels");
    require_pointer(b, "RMVPE concat_channels");
    const std::size_t a_count =
        checked_element_count({Ca, H, W}, "RMVPE concat first input");
    const std::size_t b_count =
        checked_element_count({Cb, H, W}, "RMVPE concat second input");
    if (Ca > std::numeric_limits<int>::max() - Cb) {
        throw std::length_error("RMVPE concat channel count exceeds int range");
    }
    const std::size_t output_count = checked_element_count(
        {Ca + Cb, H, W}, "RMVPE concat output");
    std::vector<float> out(output_count);
    std::copy(a, a + a_count, out.begin());
    std::copy(b, b + b_count, out.begin() + a_count);
    return out;
}

}  // namespace voxmutatio::f0::ops

#undef AFTER_KERNEL_LAUNCH
#undef BEFORE_KERNEL_LAUNCH
#undef CUDA_COPY
