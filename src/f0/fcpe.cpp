// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// FCPE F0 extractor — numerically aligned with torchfcpe's bundled
// CFNaiveMelPE (a fully-convolutional "conformer" pitch estimator).
// Audio -> log-mel(128) -> input_stack -> 6x conv-former -> weight-norm head
// -> sigmoid salience[T,360] -> local-argmax cent decode -> F0 (Hz).

#include "voxmutatio/f0/fcpe.h"
#include "voxmutatio/content/cuda/kernels.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace voxmutatio::f0 {

namespace {

namespace k = voxmutatio::content::cuda;

constexpr int kNMels = 128;
constexpr int kNFFT = 1024;
constexpr int kHop = 160;
constexpr int kWin = 1024;
constexpr int kFreqs = kNFFT / 2 + 1;  // 513
constexpr int kDim = 512;
constexpr int kFF = 2048;
constexpr int kGLU = 1024;
constexpr int kDWK = 31;      // depthwise kernel
constexpr int kNClass = 360;
constexpr int kNLayers = 6;
constexpr int kSampleRate = 16000;
constexpr float kFmin = 0.0f;
constexpr float kFmax = 8000.0f;
constexpr float kClipVal = 1e-5f;

// ---- librosa Slaney mel scale ----
float hz_to_mel_slaney(float hz) {
    const float f_sp = 200.0f / 3.0f;
    float mel = hz / f_sp;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;
    if (hz >= min_log_hz) mel = min_log_mel + std::log(hz / min_log_hz) / logstep;
    return mel;
}
float mel_to_hz_slaney(float mel) {
    const float f_sp = 200.0f / 3.0f;
    float hz = f_sp * mel;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;
    if (mel >= min_log_mel) hz = min_log_hz * std::exp(logstep * (mel - min_log_mel));
    return hz;
}

// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax) with slaney norm -> [n_mels, n_freqs].
std::vector<float> build_mel_basis() {
    std::vector<float> fftfreqs(kFreqs);
    for (int i = 0; i < kFreqs; ++i)
        fftfreqs[i] = 0.5f * kSampleRate * i / (kFreqs - 1);
    float mel_min = hz_to_mel_slaney(kFmin);
    float mel_max = hz_to_mel_slaney(kFmax);
    std::vector<float> freqs(kNMels + 2);
    for (int i = 0; i < kNMels + 2; ++i)
        freqs[i] = mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i / (kNMels + 1));
    std::vector<float> weights(kNMels * kFreqs, 0.0f);
    for (int m = 0; m < kNMels; ++m) {
        float fdiff_lower = freqs[m + 1] - freqs[m];
        float fdiff_upper = freqs[m + 2] - freqs[m + 1];
        float enorm = 2.0f / (freqs[m + 2] - freqs[m]);
        for (int f = 0; f < kFreqs; ++f) {
            float lower = (fftfreqs[f] - freqs[m]) / fdiff_lower;
            float upper = (freqs[m + 2] - fftfreqs[f]) / fdiff_upper;
            weights[m * kFreqs + f] = std::max(0.0f, std::min(lower, upper)) * enorm;
        }
    }
    return weights;
}

// In-place iterative radix-2 FFT.
void fft(std::vector<std::complex<float>>& a) {
    int n = static_cast<int>(a.size());
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * static_cast<float>(M_PI) / len;
        std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int m = 0; m < len / 2; ++m) {
                std::complex<float> u = a[i + m], v = a[i + m + len / 2] * w;
                a[i + m] = u + v;
                a[i + m + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// np.pad / torch reflect (no edge repeat).
std::vector<float> reflect_pad(const float* x, int n, int pad) {
    std::vector<float> out(n + 2 * pad);
    for (int i = 0; i < pad; ++i) out[i] = x[std::min(pad - i, n - 1)];
    std::memcpy(out.data() + pad, x, n * sizeof(float));
    for (int i = 0; i < pad; ++i) out[pad + n + i] = x[std::max(0, n - 2 - i)];
    return out;
}

void add_bias_tc(std::vector<float>& x, const float* bias, int T, int C) {
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < C; ++c) x[static_cast<size_t>(t) * C + c] += bias[c];
}

// [T, C] -> [C, T]
std::vector<float> transpose_tc(const std::vector<float>& x, int T, int C) {
    std::vector<float> o(static_cast<size_t>(T) * C);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < C; ++c)
            o[static_cast<size_t>(c) * T + t] = x[static_cast<size_t>(t) * C + c];
    return o;
}
// [C, T] -> [T, C]
std::vector<float> transpose_ct(const std::vector<float>& x, int C, int T) {
    std::vector<float> o(static_cast<size_t>(T) * C);
    for (int c = 0; c < C; ++c)
        for (int t = 0; t < T; ++t)
            o[static_cast<size_t>(t) * C + c] = x[static_cast<size_t>(c) * T + t];
    return o;
}

inline float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }

bool valid_threshold(float threshold) {
    return std::isfinite(threshold) && threshold >= 0.0f && threshold <= 1.0f;
}

bool valid_audio_input(const float* audio, int num_samples) {
    if (!audio || num_samples <= 0) return false;
    const int frames = num_samples / kHop + 1;
    // The widest frame-major activation is [frames, 2048], and CUDA helper
    // element counts are currently represented as int.
    if (frames > std::numeric_limits<int>::max() / kFF ||
        num_samples > std::numeric_limits<int>::max() - (kWin - kHop)) {
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
                          const std::vector<std::int64_t>& expected_shape) {
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
        if (!std::isfinite(values[i])) return false;
    }
    return true;
}

bool validate_checkpoint(const io::SafetensorsLoader& loader) {
    if (!expect_finite_tensor(loader, "input_stack.0.weight", {kDim, kNMels, 3}) ||
        !expect_finite_tensor(loader, "input_stack.0.bias", {kDim}) ||
        !expect_finite_tensor(loader, "input_stack.1.weight", {kDim}) ||
        !expect_finite_tensor(loader, "input_stack.1.bias", {kDim}) ||
        !expect_finite_tensor(loader, "input_stack.3.weight", {kDim, kDim, 3}) ||
        !expect_finite_tensor(loader, "input_stack.3.bias", {kDim}) ||
        !expect_finite_tensor(loader, "norm.weight", {kDim}) ||
        !expect_finite_tensor(loader, "norm.bias", {kDim}) ||
        !expect_finite_tensor(loader, "output_proj.parametrizations.weight.original0",
                              {kNClass, 1}) ||
        !expect_finite_tensor(loader, "output_proj.parametrizations.weight.original1",
                              {kNClass, kDim}) ||
        !expect_finite_tensor(loader, "output_proj.bias", {kNClass}) ||
        !expect_finite_tensor(loader, "cent_table", {kNClass})) {
        return false;
    }

    for (int layer = 0; layer < kNLayers; ++layer) {
        const std::string prefix =
            "net.encoder_layers." + std::to_string(layer) + ".conformer.net.";
        if (!expect_finite_tensor(loader, prefix + "0.weight", {kDim}) ||
            !expect_finite_tensor(loader, prefix + "0.bias", {kDim}) ||
            !expect_finite_tensor(loader, prefix + "2.weight", {kFF, kDim, 1}) ||
            !expect_finite_tensor(loader, prefix + "2.bias", {kFF}) ||
            !expect_finite_tensor(loader, prefix + "4.conv.weight", {kGLU, 1, kDWK}) ||
            !expect_finite_tensor(loader, prefix + "4.conv.bias", {kGLU}) ||
            !expect_finite_tensor(loader, prefix + "6.weight", {kDim, kGLU, 1}) ||
            !expect_finite_tensor(loader, prefix + "6.bias", {kDim})) {
            return false;
        }
    }

    const auto* cent_table = reinterpret_cast<const float*>(loader.data("cent_table"));
    const auto* projection = reinterpret_cast<const float*>(
        loader.data("output_proj.parametrizations.weight.original1"));
    if (!cent_table || !projection) return false;
    for (int i = 1; i < kNClass; ++i) {
        if (!(cent_table[i] > cent_table[i - 1])) return false;
    }
    for (int row = 0; row < kNClass; ++row) {
        double norm_squared = 0.0;
        for (int col = 0; col < kDim; ++col) {
            const double value = projection[static_cast<std::size_t>(row) * kDim + col];
            norm_squared += value * value;
        }
        if (!std::isfinite(norm_squared) || norm_squared <= 0.0) return false;
    }
    return true;
}

std::vector<float> compute_mel_impl(const std::vector<float>& mel_basis,
                                    const float* audio, int num_samples,
                                    int& out_frames) {
    // torchfcpe MelModule: center=False, manual reflect pad of (win-hop)/2 each side.
    const int pad = (kWin - kHop) / 2;  // 432
    auto y = reflect_pad(audio, num_samples, pad);
    const int ny = static_cast<int>(y.size());
    const int nf = (ny >= kNFFT) ? (ny - kNFFT) / kHop + 1 : 0;

    // Hann window (periodic: divide by N).
    std::vector<float> win(kWin);
    for (int i = 0; i < kWin; ++i) {
        win[i] = 0.5f - 0.5f * std::cos(
            2.0f * static_cast<float>(M_PI) * i / kWin);
    }

    // Target frame count = T // hop + 1 (wav2mel duplicates/trims the STFT to this).
    const int target = num_samples / kHop + 1;
    std::vector<float> mel(static_cast<std::size_t>(target) * kNMels, 0.0f);

    std::vector<std::complex<float>> buf(kNFFT);
    std::vector<float> mag(kFreqs);
    for (int frame = 0; frame < target; ++frame) {
        const int source_frame = std::min(frame, std::max(0, nf - 1));
        const int base = source_frame * kHop;
        for (int i = 0; i < kNFFT; ++i) {
            const float sample = base + i < ny ? y[base + i] : 0.0f;
            buf[i] = {sample * win[i], 0.0f};
        }
        fft(buf);
        for (int bin = 0; bin < kFreqs; ++bin) {
            mag[bin] = std::sqrt(buf[bin].real() * buf[bin].real() +
                                 buf[bin].imag() * buf[bin].imag() + 1e-9f);
        }
        float* row = &mel[static_cast<std::size_t>(frame) * kNMels];
        for (int mel_bin = 0; mel_bin < kNMels; ++mel_bin) {
            const float* weights =
                &mel_basis[static_cast<std::size_t>(mel_bin) * kFreqs];
            double acc = 0.0;
            for (int bin = 0; bin < kFreqs; ++bin) {
                acc += static_cast<double>(weights[bin]) * mag[bin];
            }
            row[mel_bin] = std::log(std::max(static_cast<float>(acc), kClipVal));
        }
    }
    out_frames = target;
    return mel;
}

}  // namespace

void FcpeExtractor::reset() noexcept {
    loaded_ = false;
    threshold_ = 0.006f;
    loader_ = io::SafetensorsLoader{};
    mel_basis_.clear();
    w_out_.clear();
    b_out_.clear();
    cent_table_.clear();
}

bool FcpeExtractor::init(const FcpeConfig& config) try {
    // Failed replacement initialization invalidates the old model instead of
    // leaving a mixture of the old mmap and newly materialised head weights.
    reset();
    if (config.model_path.empty() || !valid_threshold(config.threshold)) return false;

    io::SafetensorsLoader staged_loader;
    if (!staged_loader.load(config.model_path) ||
        !validate_checkpoint(staged_loader)) {
        return false;
    }

    auto staged_mel_basis = build_mel_basis();
    if (!all_finite(staged_mel_basis)) return false;

    // Materialise the weight-normed output projection: w[o,:] = g[o]*v[o,:]/||v[o,:]||.
    const float* g = reinterpret_cast<const float*>(
        staged_loader.data("output_proj.parametrizations.weight.original0"));
    const float* v = reinterpret_cast<const float*>(
        staged_loader.data("output_proj.parametrizations.weight.original1"));
    const float* bo = reinterpret_cast<const float*>(staged_loader.data("output_proj.bias"));
    const float* ct = reinterpret_cast<const float*>(staged_loader.data("cent_table"));
    if (!g || !v || !bo || !ct) return false;

    std::vector<float> staged_w_out(static_cast<std::size_t>(kNClass) * kDim, 0.0f);
    std::vector<float> staged_b_out(bo, bo + kNClass);
    std::vector<float> staged_cent_table(ct, ct + kNClass);
    for (int o = 0; o < kNClass; ++o) {
        double norm_squared = 0.0;
        for (int j = 0; j < kDim; ++j) {
            const double value = v[static_cast<std::size_t>(o) * kDim + j];
            norm_squared += value * value;
        }
        if (!std::isfinite(norm_squared) || norm_squared <= 0.0) return false;
        const double denominator = std::sqrt(norm_squared) + 1e-12;
        for (int j = 0; j < kDim; ++j) {
            const std::size_t index = static_cast<std::size_t>(o) * kDim + j;
            const double reconstructed =
                static_cast<double>(g[o]) * static_cast<double>(v[index]) / denominator;
            if (!std::isfinite(reconstructed) ||
                std::abs(reconstructed) > std::numeric_limits<float>::max()) {
                return false;
            }
            staged_w_out[index] = static_cast<float>(reconstructed);
        }
    }
    if (!all_finite(staged_w_out)) return false;

    loader_ = std::move(staged_loader);
    mel_basis_ = std::move(staged_mel_basis);
    w_out_ = std::move(staged_w_out);
    b_out_ = std::move(staged_b_out);
    cent_table_ = std::move(staged_cent_table);
    threshold_ = config.threshold;
    loaded_ = true;
    return true;
} catch (...) {
    reset();
    return false;
}

std::vector<float> FcpeExtractor::compute_mel(const float* audio, int num_samples,
                                              int& out_frames) {
    out_frames = 0;
    if (!loaded_ || !valid_audio_input(audio, num_samples)) return {};
    try {
        auto mel = compute_mel_impl(mel_basis_, audio, num_samples, out_frames);
        if (!all_finite(mel)) {
            out_frames = 0;
            return {};
        }
        return mel;
    } catch (...) {
        out_frames = 0;
        return {};
    }
}

std::vector<float> FcpeExtractor::infer(const float* audio, int num_samples) {
    if (!loaded_ || !valid_audio_input(audio, num_samples)) return {};

    try {
    const auto weight = [this](const std::string& name) -> const float* {
        const std::uint8_t* bytes = loader_.data(name);
        if (!bytes) throw std::logic_error("FCPE validated tensor became unavailable");
        return reinterpret_cast<const float*>(bytes);
    };

    int T = 0;
    auto x = compute_mel_impl(mel_basis_, audio, num_samples, T);  // [T, 128]
    if (T <= 0 || !all_finite(x)) return {};

    // input_stack: Conv1d(128,512,3,p1) -> GroupNorm(4,512) -> LeakyReLU(0.01) -> Conv1d(512,512,3,p1)
    auto h = k::conv1d(x.data(), T, kNMels,
                       weight("input_stack.0.weight"),
                       kDim, 3);  // [T,512]
    add_bias_tc(h, weight("input_stack.0.bias"), T, kDim);
    {
        auto hc = transpose_tc(h, T, kDim);  // [512,T]
        std::vector<float> gn(hc.size());
        k::group_norm(hc.data(), gn.data(), kDim, T, 4,
                      weight("input_stack.1.weight"),
                      weight("input_stack.1.bias"), 1e-5f);
        for (float& val : gn) if (val < 0.0f) val *= 0.01f;  // LeakyReLU(0.01)
        h = transpose_ct(gn, kDim, T);      // [T,512]
    }
    h = k::conv1d(h.data(), T, kDim,
                  weight("input_stack.3.weight"),
                  kDim, 3);
    add_bias_tc(h, weight("input_stack.3.bias"), T, kDim);

    // 6x CFNEncoderLayer: x = x + conformer(x)  (self-attention disabled in bundled model).
    for (int L = 0; L < kNLayers; ++L) {
        std::string p = "net.encoder_layers." + std::to_string(L) + ".conformer.net.";
        // 0: LayerNorm(512)
        auto y = k::layer_norm_affine(h.data(), T, kDim,
                    weight(p + "0.weight"), weight(p + "0.bias"));  // [T,512]
        // 2: pointwise Conv1d(512->2048, k1)
        y = k::conv1d(y.data(), T, kDim,
                      weight(p + "2.weight"), kFF, 1);
        add_bias_tc(y, weight(p + "2.bias"), T, kFF);
        // 3: GLU(dim=channel): out = a * sigmoid(b), split 2048 -> 1024
        std::vector<float> gl(static_cast<size_t>(T) * kGLU);
        for (int t = 0; t < T; ++t) {
            const float* a = &y[static_cast<size_t>(t) * kFF];
            const float* b = a + kGLU;
            float* o = &gl[static_cast<size_t>(t) * kGLU];
            for (int c = 0; c < kGLU; ++c) o[c] = a[c] * sigmoidf(b[c]);
        }
        // 4: DepthWiseConv1d(1024, k31, p15, groups=1024)  [channels-first]
        auto gc = transpose_tc(gl, T, kGLU);  // [1024,T]
        auto dw = k::conv1d_grouped(gc.data(), kGLU, T,
                      weight(p + "4.conv.weight"), weight(p + "4.conv.bias"),
                      kGLU, kDWK, 1, (kDWK - 1) / 2, kGLU);  // [1024,T]
        // 5: SiLU
        for (float& val : dw) val = val * sigmoidf(val);
        auto dt = transpose_ct(dw, kGLU, T);  // [T,1024]
        // 6: pointwise Conv1d(1024->512, k1)
        auto z = k::conv1d(dt.data(), T, kGLU,
                    weight(p + "6.weight"), kDim, 1);
        add_bias_tc(z, weight(p + "6.bias"), T, kDim);
        // residual
        for (size_t i = 0; i < h.size(); ++i) h[i] += z[i];
    }

    // Final LayerNorm(512) -> weight-norm Linear(512->360) -> sigmoid.
    auto nrm = k::layer_norm_affine(h.data(), T, kDim,
                weight("norm.weight"), weight("norm.bias"));  // [T,512]
    auto latent = k::linear(nrm.data(), T, kDim, w_out_.data(), b_out_.data(), kNClass);  // [T,360]
    for (float& val : latent) val = sigmoidf(val);

    // Decode: local-argmax (±4 around peak) weighted cent average -> Hz.
    std::vector<float> f0(T, 0.0f);
    for (int t = 0; t < T; ++t) {
        const float* row = &latent[static_cast<size_t>(t) * kNClass];
        int mi = 0;
        float mx = row[0];
        for (int c = 1; c < kNClass; ++c) if (row[c] > mx) { mx = row[c]; mi = c; }
        if (mx <= threshold_) { f0[t] = 0.0f; continue; }
        double num = 0.0, den = 0.0;
        for (int d = 0; d < 9; ++d) {
            int idx = std::clamp(mi - 4 + d, 0, kNClass - 1);
            num += static_cast<double>(cent_table_[idx]) * row[idx];
            den += row[idx];
        }
        float cent = (den > 0.0) ? static_cast<float>(num / den) : 0.0f;
        f0[t] = 10.0f * std::pow(2.0f, cent / 1200.0f);
    }
    return all_finite(f0) ? f0 : std::vector<float>{};
    } catch (...) {
        return {};
    }
}

std::vector<float> FcpeExtractor::pitch_shift(const std::vector<float>& f0, int semitones) {
    if (semitones < -48 || semitones > 48 || !all_finite(f0)) return {};

    std::vector<float> result(f0.size());
    const float multiplier = std::pow(2.0f, semitones / 12.0f);
    for (size_t i = 0; i < f0.size(); ++i)
        result[i] = (f0[i] > 0.0f) ? f0[i] * multiplier : 0.0f;
    return all_finite(result) ? result : std::vector<float>{};
}

}  // namespace voxmutatio::f0
