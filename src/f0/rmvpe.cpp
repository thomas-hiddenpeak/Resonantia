// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// RMVPE F0 extractor — numerically aligned with the RVC RMVPE E2E model
// (DeepUnet + BiGRU). Processes 16kHz audio -> mel spectrogram -> salience
// [T, 360] -> F0 contour via local weighted-average decoding.

#include "voxmutatio/f0/rmvpe.h"
#include "voxmutatio/f0/rmvpe_ops.h"
#include "voxmutatio/content/cuda/kernels.h"
#include "voxmutatio/io/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace voxmutatio::f0 {

struct RmvpeExtractor::ModelState {
    io::SafetensorsLoader loader;
};

namespace {

namespace ops = voxmutatio::f0::ops;
namespace k = voxmutatio::content::cuda;

constexpr int kNMels = 128;
constexpr int kNFFT = 1024;
constexpr int kHop = 160;
constexpr int kWin = 1024;
constexpr float kFmin = 30.0f;
constexpr float kFmax = 8000.0f;
constexpr int kSampleRate = 16000;
constexpr float kMelClamp = 1e-5f;
constexpr int kNClass = 360;
constexpr int kEncDeLayers = 5;
constexpr int kInterLayers = 4;
constexpr int kNBlocks = 4;

// ---- Slaney mel scale (matches librosa) ----
float hz_to_mel_slaney(float hz) {
    const float f_sp = 200.0f / 3.0f;
    float mel = hz / f_sp;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;  // 15
    const float logstep = std::log(6.4f) / 27.0f;
    if (hz >= min_log_hz) mel = min_log_mel + std::log(hz / min_log_hz) / logstep;
    return mel;
}

float mel_to_hz_slaney(float mel) {
    const float f_sp = 200.0f / 3.0f;
    float hz = f_sp * mel;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;  // 15
    const float logstep = std::log(6.4f) / 27.0f;
    if (mel >= min_log_mel) hz = min_log_hz * std::exp(logstep * (mel - min_log_mel));
    return hz;
}

// Build librosa Slaney mel filterbank [n_mels, n_freqs]
std::vector<float> build_mel_basis() {
    int n_freqs = kNFFT / 2 + 1;  // 513
    std::vector<float> fftfreqs(n_freqs);
    for (int i = 0; i < n_freqs; ++i)
        fftfreqs[i] = 0.5f * kSampleRate * i / (n_freqs - 1);

    float mel_min = hz_to_mel_slaney(kFmin);
    float mel_max = hz_to_mel_slaney(kFmax);
    std::vector<float> freqs(kNMels + 2);
    for (int i = 0; i < kNMels + 2; ++i) {
        float mel = mel_min + (mel_max - mel_min) * i / (kNMels + 1);
        freqs[i] = mel_to_hz_slaney(mel);
    }

    std::vector<float> weights(kNMels * n_freqs, 0.0f);
    for (int m = 0; m < kNMels; ++m) {
        float fdiff_lower = freqs[m + 1] - freqs[m];
        float fdiff_upper = freqs[m + 2] - freqs[m + 1];
        float enorm = 2.0f / (freqs[m + 2] - freqs[m]);
        for (int f = 0; f < n_freqs; ++f) {
            float lower = (fftfreqs[f] - freqs[m]) / fdiff_lower;
            float upper = (freqs[m + 2] - fftfreqs[f]) / fdiff_upper;
            float w = std::max(0.0f, std::min(lower, upper));
            weights[m * n_freqs + f] = w * enorm;
        }
    }
    return weights;
}

// Periodic Hann window (matches torch.hann_window default)
std::vector<float> hann_window(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i)
        w[i] = 0.5f - 0.5f * std::cos(2.0f * M_PI * i / n);
    return w;
}

// Iterative radix-2 FFT (in place), complex arrays re/im of size n (power of 2)
void fft_radix2(std::vector<float>& re, std::vector<float>& im) {
    int n = static_cast<int>(re.size());
    // Bit reversal
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * M_PI / len;
        float wlen_re = std::cos(ang), wlen_im = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float w_re = 1.0f, w_im = 0.0f;
            for (int j = 0; j < len / 2; ++j) {
                float u_re = re[i + j], u_im = im[i + j];
                float v_re = re[i + j + len / 2] * w_re - im[i + j + len / 2] * w_im;
                float v_im = re[i + j + len / 2] * w_im + im[i + j + len / 2] * w_re;
                re[i + j] = u_re + v_re; im[i + j] = u_im + v_im;
                re[i + j + len / 2] = u_re - v_re; im[i + j + len / 2] = u_im - v_im;
                float nw_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
            }
        }
    }
}

// Mel spectrogram: audio -> [n_mels, T] log-mel (matches RMVPE MelSpectrogram)
std::vector<float> mel_spectrogram(const float* audio, int num_samples, int& out_T) {
    static std::vector<float> mel_basis = build_mel_basis();
    static std::vector<float> window = hann_window(kWin);

    // center=True: reflect-pad by n_fft/2 on both sides
    int pad = kNFFT / 2;  // 512
    std::vector<float> padded(num_samples + 2 * pad);
    for (int i = 0; i < pad; ++i)
        padded[i] = audio[std::min(pad - i, num_samples - 1)];  // reflect (excl. edge)
    std::memcpy(padded.data() + pad, audio, num_samples * sizeof(float));
    for (int i = 0; i < pad; ++i)
        padded[pad + num_samples + i] = audio[std::max(num_samples - 2 - i, 0)];

    int T = 1 + num_samples / kHop;  // torch center STFT frame count
    out_T = T;
    int n_freqs = kNFFT / 2 + 1;

    std::vector<float> mag(n_freqs * T);  // [freq, T]
    std::vector<float> re(kNFFT), im(kNFFT);
    for (int t = 0; t < T; ++t) {
        int start = t * kHop;  // into padded
        for (int i = 0; i < kNFFT; ++i) {
            float s = (start + i < (int)padded.size()) ? padded[start + i] : 0.0f;
            re[i] = s * window[i];
            im[i] = 0.0f;
        }
        fft_radix2(re, im);
        for (int f = 0; f < n_freqs; ++f)
            mag[f * T + t] = std::sqrt(re[f] * re[f] + im[f] * im[f]);
    }

    // mel = mel_basis [n_mels, n_freqs] @ mag [n_freqs, T] -> [n_mels, T]
    std::vector<float> logmel(kNMels * T);
    for (int m = 0; m < kNMels; ++m) {
        for (int t = 0; t < T; ++t) {
            float sum = 0.0f;
            for (int f = 0; f < n_freqs; ++f)
                sum += mel_basis[m * n_freqs + f] * mag[f * T + t];
            logmel[m * T + t] = std::log(std::max(sum, kMelClamp));
        }
    }
    return logmel;  // [n_mels, T]
}

bool valid_threshold(float threshold) {
    return std::isfinite(threshold) && threshold >= 0.0f && threshold <= 1.0f;
}

bool valid_audio_input(const float* audio, int num_samples) {
    if (!audio || num_samples <= 0) return false;

    if (num_samples > std::numeric_limits<int>::max() - kNFFT) return false;

    const int frames = 1 + num_samples / kHop;
    if (frames > std::numeric_limits<int>::max() - 31) return false;
    const int padded_frames = ((frames + 31) / 32) * 32;
    // The widest decoder activation is [16, padded_frames, 128], and the
    // current CUDA operator boundary expresses element counts as int.
    if (padded_frames > std::numeric_limits<int>::max() / (16 * kNMels)) {
        return false;
    }
    for (int i = 0; i < num_samples; ++i) {
        if (!std::isfinite(audio[i])) return false;
    }
    return true;
}

bool all_finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

bool expect_finite_tensor(const io::SafetensorsLoader& loader,
                          const std::string& name,
                          const std::vector<std::int64_t>& expected_shape,
                          bool require_nonnegative = false) {
    const io::Tensor* tensor = loader.get_tensor(name);
    if (!tensor || tensor->dtype != "F32" || tensor->shape != expected_shape) {
        return false;
    }

    std::size_t elements = 1;
    for (const std::int64_t dim : expected_shape) {
        if (dim <= 0 || static_cast<std::uint64_t>(dim) >
                            std::numeric_limits<std::size_t>::max()) {
            return false;
        }
        const std::size_t size = static_cast<std::size_t>(dim);
        if (elements > std::numeric_limits<std::size_t>::max() / size) return false;
        elements *= size;
    }
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        tensor->data_nbytes != elements * sizeof(float)) {
        return false;
    }

    const auto* bytes = loader.data(name);
    if (!bytes || reinterpret_cast<std::uintptr_t>(bytes) % alignof(float) != 0) {
        return false;
    }
    const auto* values = reinterpret_cast<const float*>(bytes);
    for (std::size_t i = 0; i < elements; ++i) {
        if (!std::isfinite(values[i]) || (require_nonnegative && values[i] < 0.0f)) {
            return false;
        }
    }
    return true;
}

bool expect_batch_norm(const io::SafetensorsLoader& loader,
                       const std::string& prefix, int channels) {
    const std::vector<std::int64_t> shape = {channels};
    return expect_finite_tensor(loader, prefix + "weight", shape) &&
           expect_finite_tensor(loader, prefix + "bias", shape) &&
           expect_finite_tensor(loader, prefix + "running_mean", shape) &&
           expect_finite_tensor(loader, prefix + "running_var", shape, true);
}

bool expect_conv_block(const io::SafetensorsLoader& loader,
                       const std::string& prefix, int in_channels,
                       int out_channels) {
    if (!expect_finite_tensor(loader, prefix + "conv.0.weight",
                              {out_channels, in_channels, 3, 3}) ||
        !expect_batch_norm(loader, prefix + "conv.1.", out_channels) ||
        !expect_finite_tensor(loader, prefix + "conv.3.weight",
                              {out_channels, out_channels, 3, 3}) ||
        !expect_batch_norm(loader, prefix + "conv.4.", out_channels)) {
        return false;
    }
    return in_channels == out_channels ||
           (expect_finite_tensor(loader, prefix + "shortcut.weight",
                                 {out_channels, in_channels, 1, 1}) &&
            expect_finite_tensor(loader, prefix + "shortcut.bias", {out_channels}));
}

bool expect_encoder_block(const io::SafetensorsLoader& loader,
                          const std::string& prefix, int in_channels,
                          int out_channels) {
    if (!expect_conv_block(loader, prefix + "conv.0.", in_channels, out_channels)) {
        return false;
    }
    for (int block = 1; block < kNBlocks; ++block) {
        if (!expect_conv_block(loader,
                               prefix + "conv." + std::to_string(block) + ".",
                               out_channels, out_channels)) {
            return false;
        }
    }
    return true;
}

bool expect_decoder_block(const io::SafetensorsLoader& loader,
                          const std::string& prefix, int in_channels,
                          int out_channels) {
    if (!expect_finite_tensor(loader, prefix + "conv1.0.weight",
                              {in_channels, out_channels, 3, 3}) ||
        !expect_batch_norm(loader, prefix + "conv1.1.", out_channels) ||
        !expect_conv_block(loader, prefix + "conv2.0.", 2 * out_channels,
                           out_channels)) {
        return false;
    }
    for (int block = 1; block < kNBlocks; ++block) {
        if (!expect_conv_block(loader,
                               prefix + "conv2." + std::to_string(block) + ".",
                               out_channels, out_channels)) {
            return false;
        }
    }
    return true;
}

bool validate_checkpoint(const io::SafetensorsLoader& loader) {
    if (!expect_finite_tensor(loader, "cnn.weight", {3, 16, 3, 3}) ||
        !expect_finite_tensor(loader, "cnn.bias", {3}) ||
        !expect_finite_tensor(loader, "fc.0.gru.weight_ih_l0", {768, 384}) ||
        !expect_finite_tensor(loader, "fc.0.gru.weight_hh_l0", {768, 256}) ||
        !expect_finite_tensor(loader, "fc.0.gru.bias_ih_l0", {768}) ||
        !expect_finite_tensor(loader, "fc.0.gru.bias_hh_l0", {768}) ||
        !expect_finite_tensor(loader, "fc.0.gru.weight_ih_l0_reverse", {768, 384}) ||
        !expect_finite_tensor(loader, "fc.0.gru.weight_hh_l0_reverse", {768, 256}) ||
        !expect_finite_tensor(loader, "fc.0.gru.bias_ih_l0_reverse", {768}) ||
        !expect_finite_tensor(loader, "fc.0.gru.bias_hh_l0_reverse", {768}) ||
        !expect_finite_tensor(loader, "fc.1.weight", {kNClass, 512}) ||
        !expect_finite_tensor(loader, "fc.1.bias", {kNClass}) ||
        !expect_batch_norm(loader, "unet.encoder.bn.", 1)) {
        return false;
    }

    constexpr int encoder_in[kEncDeLayers] = {1, 16, 32, 64, 128};
    constexpr int encoder_out[kEncDeLayers] = {16, 32, 64, 128, 256};
    for (int layer = 0; layer < kEncDeLayers; ++layer) {
        if (!expect_encoder_block(
                loader, "unet.encoder.layers." + std::to_string(layer) + ".",
                encoder_in[layer], encoder_out[layer])) {
            return false;
        }
    }

    for (int layer = 0; layer < kInterLayers; ++layer) {
        const int in_channels = layer == 0 ? 256 : 512;
        if (!expect_encoder_block(
                loader, "unet.intermediate.layers." + std::to_string(layer) + ".",
                in_channels, 512)) {
            return false;
        }
    }

    constexpr int decoder_in[kEncDeLayers] = {512, 256, 128, 64, 32};
    constexpr int decoder_out[kEncDeLayers] = {256, 128, 64, 32, 16};
    for (int layer = 0; layer < kEncDeLayers; ++layer) {
        if (!expect_decoder_block(
                loader, "unet.decoder.layers." + std::to_string(layer) + ".",
                decoder_in[layer], decoder_out[layer])) {
            return false;
        }
    }
    return true;
}

// ---- Weight access (zero-copy mmap) ----
struct Weights {
    const io::SafetensorsLoader* loader = nullptr;

    const float* get(const std::string& n) const {
        const std::uint8_t* p = loader ? loader->data(n) : nullptr;
        if (!p) throw std::logic_error("RMVPE validated tensor became unavailable");
        return reinterpret_cast<const float*>(p);
    }
};

// ConvBlockRes forward. x: [Cin,H,W] -> [Cout,H,W]
std::vector<float> conv_block_res(const Weights& w, const std::string& prefix,
                                  const std::vector<float>& x, int Cin, int Cout,
                                  int H, int W) {
    // conv.0 (Conv2d 3x3, no bias)
    auto h = ops::conv2d_3x3(x.data(), Cin, H, W, w.get(prefix + "conv.0.weight"),
                             nullptr, Cout);
    // conv.1 (BatchNorm)
    ops::batchnorm2d(h.data(), Cout, H, W, w.get(prefix + "conv.1.weight"),
                     w.get(prefix + "conv.1.bias"), w.get(prefix + "conv.1.running_mean"),
                     w.get(prefix + "conv.1.running_var"));
    // conv.2 (ReLU)
    ops::relu_inplace(h.data(), Cout * H * W);
    // conv.3 (Conv2d 3x3)
    h = ops::conv2d_3x3(h.data(), Cout, H, W, w.get(prefix + "conv.3.weight"),
                        nullptr, Cout);
    // conv.4 (BatchNorm)
    ops::batchnorm2d(h.data(), Cout, H, W, w.get(prefix + "conv.4.weight"),
                     w.get(prefix + "conv.4.bias"), w.get(prefix + "conv.4.running_mean"),
                     w.get(prefix + "conv.4.running_var"));
    // conv.5 (ReLU)
    ops::relu_inplace(h.data(), Cout * H * W);

    // Residual: shortcut if Cin != Cout
    if (Cin != Cout) {
        auto sc = ops::conv2d_1x1(x.data(), Cin, H, W, w.get(prefix + "shortcut.weight"),
                                  w.get(prefix + "shortcut.bias"), Cout);
        for (size_t i = 0; i < h.size(); ++i) h[i] += sc[i];
    } else {
        for (size_t i = 0; i < h.size(); ++i) h[i] += x[i];
    }
    return h;
}

// ResEncoderBlock: n_blocks ConvBlockRes, then optional avgpool.
// Returns pre-pool tensor in `pre`, and pooled result (or same) as return.
std::vector<float> res_encoder_block(const Weights& w, const std::string& prefix,
                                     std::vector<float> x, int Cin, int Cout,
                                     int& H, int& W, bool do_pool,
                                     std::vector<float>* pre) {
    x = conv_block_res(w, prefix + "conv.0.", x, Cin, Cout, H, W);
    for (int b = 1; b < kNBlocks; ++b)
        x = conv_block_res(w, prefix + "conv." + std::to_string(b) + ".", x, Cout, Cout, H, W);
    if (pre) *pre = x;
    if (do_pool) {
        auto pooled = ops::avgpool2d(x.data(), Cout, H, W, 2, 2);
        H /= 2; W /= 2;
        return pooled;
    }
    return x;
}

// ResDecoderBlock: convtranspose upsample -> concat skip -> n_blocks ConvBlockRes
std::vector<float> res_decoder_block(const Weights& w, const std::string& prefix,
                                     std::vector<float> x, int Cin, int Cout,
                                     int& H, int& W, const std::vector<float>& skip) {
    // conv1.0 ConvTranspose2d stride 2 (no bias)
    auto up = ops::conv_transpose2d_s2(x.data(), Cin, H, W, w.get(prefix + "conv1.0.weight"), Cout);
    int OH = 2 * H, OW = 2 * W;
    // conv1.1 BatchNorm
    ops::batchnorm2d(up.data(), Cout, OH, OW, w.get(prefix + "conv1.1.weight"),
                     w.get(prefix + "conv1.1.bias"), w.get(prefix + "conv1.1.running_mean"),
                     w.get(prefix + "conv1.1.running_var"));
    // conv1.2 ReLU
    ops::relu_inplace(up.data(), Cout * OH * OW);

    H = OH; W = OW;
    // concat with skip [Cout, H, W] -> [2*Cout, H, W]
    auto cat = ops::concat_channels(up.data(), Cout, skip.data(), Cout, H, W);

    // conv2.0 ConvBlockRes(2*Cout -> Cout), then conv2.1..3 (Cout->Cout)
    auto h = conv_block_res(w, prefix + "conv2.0.", cat, 2 * Cout, Cout, H, W);
    for (int b = 1; b < kNBlocks; ++b)
        h = conv_block_res(w, prefix + "conv2." + std::to_string(b) + ".", h, Cout, Cout, H, W);
    return h;
}

// Bidirectional GRU. input [T, in_dim] -> output [T, 2*hidden]
std::vector<float> bigru(const Weights& w, const std::string& prefix,
                         const std::vector<float>& input, int T, int in_dim, int hidden) {
    // Precompute input contributions gi = X @ W_ih^T + b_ih for both directions.
    auto run_direction = [&](bool reverse) {
        std::string suf = reverse ? "_reverse" : "";
        const float* w_ih = w.get(prefix + "weight_ih_l0" + suf);  // [3H, in]
        const float* w_hh = w.get(prefix + "weight_hh_l0" + suf);  // [3H, H]
        const float* b_ih = w.get(prefix + "bias_ih_l0" + suf);    // [3H]
        const float* b_hh = w.get(prefix + "bias_hh_l0" + suf);    // [3H]

        // gi = X @ W_ih^T + b_ih  -> [T, 3H]
        auto gi = k::linear(input.data(), T, in_dim, w_ih, b_ih, 3 * hidden);

        std::vector<float> out(T * hidden, 0.0f);
        std::vector<float> h_prev(hidden, 0.0f);
        for (int step = 0; step < T; ++step) {
            int t = reverse ? (T - 1 - step) : step;
            // gh = W_hh @ h_prev + b_hh  -> [3H]
            std::vector<float> gh(3 * hidden);
            for (int r = 0; r < 3 * hidden; ++r) {
                float s = b_hh[r];
                const float* wr = w_hh + r * hidden;
                for (int j = 0; j < hidden; ++j) s += wr[j] * h_prev[j];
                gh[r] = s;
            }
            const float* gi_t = gi.data() + t * 3 * hidden;
            std::vector<float> h_new(hidden);
            for (int j = 0; j < hidden; ++j) {
                float rr = 1.0f / (1.0f + std::exp(-(gi_t[j] + gh[j])));
                float zz = 1.0f / (1.0f + std::exp(-(gi_t[hidden + j] + gh[hidden + j])));
                float nn = std::tanh(gi_t[2 * hidden + j] + rr * gh[2 * hidden + j]);
                h_new[j] = (1.0f - zz) * nn + zz * h_prev[j];
            }
            std::memcpy(out.data() + t * hidden, h_new.data(), hidden * sizeof(float));
            h_prev = h_new;
        }
        return out;
    };

    auto fwd = run_direction(false);
    auto rev = run_direction(true);

    // Concatenate [T, hidden] + [T, hidden] -> [T, 2*hidden]
    std::vector<float> out(T * 2 * hidden);
    for (int t = 0; t < T; ++t) {
        std::memcpy(out.data() + t * 2 * hidden, fwd.data() + t * hidden, hidden * sizeof(float));
        std::memcpy(out.data() + t * 2 * hidden + hidden, rev.data() + t * hidden, hidden * sizeof(float));
    }
    return out;
}

}  // namespace

void RmvpeExtractor::reset() noexcept {
    state_.reset();
    threshold_ = 0.03f;
}

bool RmvpeExtractor::init(const RmvpeConfig& config) try {
    // Reinitialization is deliberately invalidating: no old mmap or threshold
    // remains observable if any part of the replacement checkpoint is invalid.
    reset();
    if (config.model_path.empty() || !valid_threshold(config.threshold)) return false;

    auto staged_state = std::make_shared<ModelState>();
    if (!staged_state->loader.load(config.model_path) ||
        !validate_checkpoint(staged_state->loader)) {
        return false;
    }

    state_ = std::move(staged_state);
    threshold_ = config.threshold;
    return true;
} catch (...) {
    reset();
    return false;
}

std::vector<float> RmvpeExtractor::infer_salience(const float* audio, int num_samples,
                                                  int& out_frames) {
    out_frames = 0;
    if (!state_ || !valid_audio_input(audio, num_samples)) return {};

    try {
    const auto state = state_;
    Weights w{&state->loader};

    // 1. Mel spectrogram [128, T]
    int T = 0;
    auto logmel = mel_spectrogram(audio, num_samples, T);

    // 2. Pad time to multiple of 32. mel is [128, T]; unet input is [1, H=Tpad, W=128].
    int Tpad = 32 * ((T - 1) / 32 + 1);
    // Build input [1, Tpad, 128]: transpose mel [128,T]->[T,128], pad time.
    std::vector<float> x(1 * Tpad * kNMels, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int m = 0; m < kNMels; ++m)
            x[t * kNMels + m] = logmel[m * T + t];

    int H = Tpad, W = kNMels;

    // 3. Encoder.bn (1 channel)
    ops::batchnorm2d(x.data(), 1, H, W, w.get("unet.encoder.bn.weight"),
                     w.get("unet.encoder.bn.bias"), w.get("unet.encoder.bn.running_mean"),
                     w.get("unet.encoder.bn.running_var"));

    // Encoder: 5 layers, channels 1->16->32->64->128->256
    std::vector<std::vector<float>> skips;
    int enc_in[kEncDeLayers] = {1, 16, 32, 64, 128};
    int enc_out[kEncDeLayers] = {16, 32, 64, 128, 256};
    for (int i = 0; i < kEncDeLayers; ++i) {
        std::vector<float> pre;
        x = res_encoder_block(w, "unet.encoder.layers." + std::to_string(i) + ".",
                              std::move(x), enc_in[i], enc_out[i], H, W, true, &pre);
        skips.push_back(std::move(pre));
    }
    // After encoder: C=256, H=Tpad/32, W=4

    // Intermediate: 4 layers at 256 channels (no pool)
    for (int i = 0; i < kInterLayers; ++i) {
        // Intermediate first block: in=256 (encoder.out_channel//2), out=512
        std::vector<float>* no_pre = nullptr;
        x = res_encoder_block(w, "unet.intermediate.layers." + std::to_string(i) + ".",
                              std::move(x), (i == 0 ? 256 : 512), 512, H, W, false, no_pre);
    }

    // Decoder: 5 layers, channels 512->256->128->64->32->16, concat skips reversed
    int dec_in[kEncDeLayers] = {512, 256, 128, 64, 32};
    int dec_out[kEncDeLayers] = {256, 128, 64, 32, 16};
    for (int i = 0; i < kEncDeLayers; ++i) {
        const auto& skip = skips[kEncDeLayers - 1 - i];
        x = res_decoder_block(w, "unet.decoder.layers." + std::to_string(i) + ".",
                              std::move(x), dec_in[i], dec_out[i], H, W, skip);
    }
    // After decoder: C=16, H=Tpad, W=128

    // cnn: Conv2d(16, 3, 3x3, pad 1) with bias
    auto cnn_out = ops::conv2d_3x3(x.data(), 16, H, W, w.get("cnn.weight"),
                                   w.get("cnn.bias"), 3);
    // cnn_out: [3, Tpad, 128]. Build GRU input [Tpad, 3*128=384]:
    //   transpose(1,2).flatten: out[t][c*128+wi] = cnn_out[c][t][wi]
    int flat = 3 * kNMels;  // 384
    std::vector<float> gru_in(Tpad * flat);
    for (int t = 0; t < Tpad; ++t)
        for (int c = 0; c < 3; ++c)
            for (int wi = 0; wi < kNMels; ++wi)
                gru_in[t * flat + c * kNMels + wi] = cnn_out[(c * H + t) * W + wi];

    // BiGRU(384 -> 256 bidir -> 512)
    auto gru_out = bigru(w, "fc.0.gru.", gru_in, Tpad, flat, 256);

    // Linear(512 -> 360) + sigmoid
    auto salience = k::linear(gru_out.data(), Tpad, 512, w.get("fc.1.weight"),
                              w.get("fc.1.bias"), kNClass);
    ops::sigmoid_inplace(salience.data(), Tpad * kNClass);

    // Trim to original T frames
    std::vector<float> result(T * kNClass);
    std::memcpy(result.data(), salience.data(), T * kNClass * sizeof(float));
    if (!all_finite(result)) return {};
    out_frames = T;
    return result;  // [T, 360]
    } catch (...) {
        out_frames = 0;
        return {};
    }
}

std::vector<float> RmvpeExtractor::infer(const float* audio, int num_samples) {
    int T = 0;
    auto salience = infer_salience(audio, num_samples, T);
    if (T <= 0 || salience.size() != static_cast<std::size_t>(T) * kNClass) return {};

    try {

    // Cents mapping: 20*i + 1997.3794084376191
    std::vector<float> cents_map(kNClass);
    for (int i = 0; i < kNClass; ++i)
        cents_map[i] = 20.0f * i + 1997.3794084376191f;

    std::vector<float> f0(T, 0.0f);
    for (int t = 0; t < T; ++t) {
        const float* sal = salience.data() + t * kNClass;
        // argmax
        int center = 0;
        float maxv = sal[0];
        for (int i = 1; i < kNClass; ++i)
            if (sal[i] > maxv) { maxv = sal[i]; center = i; }

        // Local weighted average over ±4 (with padding)
        int start = std::max(0, center - 4);
        int end = std::min(kNClass, center + 5);
        float prod = 0.0f, wsum = 0.0f;
        for (int i = start; i < end; ++i) {
            prod += sal[i] * cents_map[i];
            wsum += sal[i];
        }
        float cents = (wsum > 0) ? prod / wsum : 0.0f;
        if (maxv <= threshold_) cents = 0.0f;

        f0[t] = (cents > 0) ? 10.0f * std::pow(2.0f, cents / 1200.0f) : 0.0f;
    }
    return all_finite(f0) ? f0 : std::vector<float>{};
    } catch (...) {
        return {};
    }
}

std::vector<float> RmvpeExtractor::pitch_shift(const std::vector<float>& f0,
                                               int semitones) {
    if (semitones < -48 || semitones > 48 || !all_finite(f0)) return {};

    std::vector<float> out(f0.size());
    const float factor = std::pow(2.0f, semitones / 12.0f);
    for (size_t i = 0; i < f0.size(); ++i)
        out[i] = f0[i] > 0 ? f0[i] * factor : 0.0f;
    return all_finite(out) ? out : std::vector<float>{};
}

}  // namespace voxmutatio::f0
