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
#include <cstring>
#include <string>
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

}  // namespace

bool FcpeExtractor::init(const FcpeConfig& config) {
    threshold_ = config.threshold;
    if (!loader_.load(config.model_path)) return false;

    mel_basis_ = build_mel_basis();

    // Materialise the weight-normed output projection: w[o,:] = g[o]*v[o,:]/||v[o,:]||.
    const float* g = reinterpret_cast<const float*>(
        loader_.data("output_proj.parametrizations.weight.original0"));  // [360,1]
    const float* v = reinterpret_cast<const float*>(
        loader_.data("output_proj.parametrizations.weight.original1"));  // [360,512]
    const float* bo = reinterpret_cast<const float*>(loader_.data("output_proj.bias"));
    const float* ct = reinterpret_cast<const float*>(loader_.data("cent_table"));
    if (!g || !v || !bo || !ct) return false;
    w_out_.assign(static_cast<size_t>(kNClass) * kDim, 0.0f);
    b_out_.assign(bo, bo + kNClass);
    cent_table_.assign(ct, ct + kNClass);
    for (int o = 0; o < kNClass; ++o) {
        double nrm = 0.0;
        for (int j = 0; j < kDim; ++j) { float vv = v[o * kDim + j]; nrm += static_cast<double>(vv) * vv; }
        float scale = g[o] / static_cast<float>(std::sqrt(nrm) + 1e-12);
        for (int j = 0; j < kDim; ++j) w_out_[o * kDim + j] = v[o * kDim + j] * scale;
    }

    loaded_ = true;
    return true;
}

std::vector<float> FcpeExtractor::compute_mel(const float* audio, int num_samples,
                                              int& out_frames) {
    // torchfcpe MelModule: center=False, manual reflect pad of (win-hop)/2 each side.
    const int pad = (kWin - kHop) / 2;  // 432
    auto y = reflect_pad(audio, num_samples, pad);
    int ny = static_cast<int>(y.size());
    int nf = (ny >= kNFFT) ? (ny - kNFFT) / kHop + 1 : 0;

    // Hann window (periodic: divide by N).
    std::vector<float> win(kWin);
    for (int i = 0; i < kWin; ++i)
        win[i] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * i / kWin);

    // Target frame count = T // hop + 1 (wav2mel duplicates/trims the STFT to this).
    int target = num_samples / kHop + 1;
    std::vector<float> mel(static_cast<size_t>(target) * kNMels, 0.0f);

    std::vector<std::complex<float>> buf(kNFFT);
    std::vector<float> mag(kFreqs);
    for (int f = 0; f < target; ++f) {
        int sf = std::min(f, std::max(0, nf - 1));  // duplicate last STFT frame if needed
        int base = sf * kHop;
        for (int i = 0; i < kNFFT; ++i) buf[i] = {y[base + i] * win[i], 0.0f};
        fft(buf);
        for (int b = 0; b < kFreqs; ++b)
            mag[b] = std::sqrt(buf[b].real() * buf[b].real() + buf[b].imag() * buf[b].imag() + 1e-9f);
        float* mrow = &mel[static_cast<size_t>(f) * kNMels];
        for (int m = 0; m < kNMels; ++m) {
            const float* wrow = &mel_basis_[static_cast<size_t>(m) * kFreqs];
            double acc = 0.0;
            for (int b = 0; b < kFreqs; ++b) acc += static_cast<double>(wrow[b]) * mag[b];
            mrow[m] = std::log(std::max(static_cast<float>(acc), kClipVal));  // range compression
        }
    }
    out_frames = target;
    return mel;
}

std::vector<float> FcpeExtractor::infer(const float* audio, int num_samples) {
    if (!loaded_) return {};
    int T = 0;
    auto x = compute_mel(audio, num_samples, T);  // [T, 128]
    if (T <= 0) return {};

    // input_stack: Conv1d(128,512,3,p1) -> GroupNorm(4,512) -> LeakyReLU(0.01) -> Conv1d(512,512,3,p1)
    auto h = k::conv1d(x.data(), T, kNMels,
                       reinterpret_cast<const float*>(loader_.data("input_stack.0.weight")),
                       kDim, 3);  // [T,512]
    add_bias_tc(h, reinterpret_cast<const float*>(loader_.data("input_stack.0.bias")), T, kDim);
    {
        auto hc = transpose_tc(h, T, kDim);  // [512,T]
        std::vector<float> gn(hc.size());
        k::group_norm(hc.data(), gn.data(), kDim, T, 4,
                      reinterpret_cast<const float*>(loader_.data("input_stack.1.weight")),
                      reinterpret_cast<const float*>(loader_.data("input_stack.1.bias")), 1e-5f);
        for (float& val : gn) if (val < 0.0f) val *= 0.01f;  // LeakyReLU(0.01)
        h = transpose_ct(gn, kDim, T);      // [T,512]
    }
    h = k::conv1d(h.data(), T, kDim,
                  reinterpret_cast<const float*>(loader_.data("input_stack.3.weight")),
                  kDim, 3);
    add_bias_tc(h, reinterpret_cast<const float*>(loader_.data("input_stack.3.bias")), T, kDim);

    // 6x CFNEncoderLayer: x = x + conformer(x)  (self-attention disabled in bundled model).
    for (int L = 0; L < kNLayers; ++L) {
        std::string p = "net.encoder_layers." + std::to_string(L) + ".conformer.net.";
        // 0: LayerNorm(512)
        auto y = k::layer_norm_affine(h.data(), T, kDim,
                    reinterpret_cast<const float*>(loader_.data(p + "0.weight")),
                    reinterpret_cast<const float*>(loader_.data(p + "0.bias")));  // [T,512]
        // 2: pointwise Conv1d(512->2048, k1)
        y = k::conv1d(y.data(), T, kDim,
                      reinterpret_cast<const float*>(loader_.data(p + "2.weight")), kFF, 1);
        add_bias_tc(y, reinterpret_cast<const float*>(loader_.data(p + "2.bias")), T, kFF);
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
                      reinterpret_cast<const float*>(loader_.data(p + "4.conv.weight")),
                      reinterpret_cast<const float*>(loader_.data(p + "4.conv.bias")),
                      kGLU, kDWK, 1, (kDWK - 1) / 2, kGLU);  // [1024,T]
        // 5: SiLU
        for (float& val : dw) val = val * sigmoidf(val);
        auto dt = transpose_ct(dw, kGLU, T);  // [T,1024]
        // 6: pointwise Conv1d(1024->512, k1)
        auto z = k::conv1d(dt.data(), T, kGLU,
                    reinterpret_cast<const float*>(loader_.data(p + "6.weight")), kDim, 1);
        add_bias_tc(z, reinterpret_cast<const float*>(loader_.data(p + "6.bias")), T, kDim);
        // residual
        for (size_t i = 0; i < h.size(); ++i) h[i] += z[i];
    }

    // Final LayerNorm(512) -> weight-norm Linear(512->360) -> sigmoid.
    auto nrm = k::layer_norm_affine(h.data(), T, kDim,
                reinterpret_cast<const float*>(loader_.data("norm.weight")),
                reinterpret_cast<const float*>(loader_.data("norm.bias")));  // [T,512]
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
    return f0;
}

std::vector<float> FcpeExtractor::pitch_shift(const std::vector<float>& f0, int semitones) {
    std::vector<float> result(f0.size());
    float multiplier = std::pow(2.0f, semitones / 12.0f);
    for (size_t i = 0; i < f0.size(); ++i)
        result[i] = (f0[i] > 0.0f) ? f0[i] * multiplier : 0.0f;
    return result;
}

}  // namespace voxmutatio::f0

