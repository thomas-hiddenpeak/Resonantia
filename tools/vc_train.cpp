/**
 * @file vc_train.cpp
 * @brief Decoder fine-tuning CLI: adapt a pretrained NSF-HiFiGAN decoder to a
 *        target voice from real audio, then export an inference-ready G.
 *
 * Real audio only. Per target clip the inference front-end produces the exact
 * latents the decoder sees at run time, so fine-tuning directly adapts the
 * vocoder to the target timbre:
 *   audio -> 16k -> HuBERT features -> 2x interp
 *   audio -> 16k -> RMVPE f0
 *   (features, coarse pitch) -> enc_p + flow -> z
 *   f0 -> sine source -> har
 *   decode(z, har) reconstructs the clip; mel-L1 vs the real 40k target.
 *
 * Usage:
 *   vc_train --hubert <dir> --rmvpe <path> --pretrained <G.safetensors>
 *            --target <wav|dir> --out <G_finetuned.safetensors>
 *            [--speaker 0] [--steps 300] [--lr 2e-4] [--seg 40] [--seed 0]
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "voxmutatio/autograd/tensor.h"
#include "voxmutatio/content/hubert_encoder.h"
#include "voxmutatio/core/device.h"
#include "voxmutatio/f0/rmvpe.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/synthesizer/synthesizer.h"
#include "voxmutatio/training/gan_trainer.h"
#include "voxmutatio/training/generator_trainer.h"
#include "voxmutatio/training/mel_loss.h"
#include "voxmutatio/training/posterior_encoder.h"

using namespace voxmutatio;
namespace ag = voxmutatio::autograd;
namespace fs = std::filesystem;

namespace {

constexpr int kUpp = 400;  // 40k hop

int f0_to_coarse(float f0) {
    const double mmin = 1127.0 * std::log(1.0 + 50.0 / 700.0);
    const double mmax = 1127.0 * std::log(1.0 + 1100.0 / 700.0);
    if (f0 <= 0.0f) return 1;
    double m = 1127.0 * std::log(1.0 + f0 / 700.0);
    m = (m - mmin) * 254.0 / (mmax - mmin) + 1.0;
    if (m <= 1.0) m = 1.0;
    if (m > 255.0) m = 255.0;
    return std::clamp(static_cast<int>(std::lround(m)), 1, 255);
}

std::vector<float> interpolate_2x(const std::vector<float>& feats, int T, int D) {
    std::vector<float> out(static_cast<size_t>(2) * T * D);
    for (int t = 0; t < T; ++t) {
        std::memcpy(&out[(2 * t) * D], &feats[t * D], D * sizeof(float));
        std::memcpy(&out[(2 * t + 1) * D], &feats[t * D], D * sizeof(float));
    }
    return out;
}

// One target clip reduced to (z, har, target-40k) with T frames.
struct Clip {
    std::vector<float> z;    // [192, T]
    std::vector<float> har;  // [T*400]
    std::vector<float> tgt;  // [T*400]
    std::vector<float> spec;    // [1025, T] (GAN: enc_q input)
    std::vector<float> m_p;     // [192, T]  (GAN: prior mean)
    std::vector<float> logs_p;  // [192, T]  (GAN: prior log-std)
    int T = 0;
};

void print_usage() {
    std::cout <<
        "Usage:\n"
        "  vc_train --hubert <dir> --rmvpe <path> --pretrained <G.safetensors>\n"
        "           --target <wav|dir> --out <G_finetuned.safetensors> [options]\n\n"
        "Options:\n"
        "  --gan            Full GAN fine-tune (enc_q+flow+dec vs discriminator)\n"
        "  --dmodel <path>  Discriminator safetensors (default: f0D40k next to G)\n"
        "  --speaker <id>   Source speaker embedding id (default: 0)\n"
        "  --steps <n>      Training steps (default: 300)\n"
        "  --epochs <n>     Passes over the dataset; overrides --steps (RVC-style, ~200)\n"
        "  --lr <f>         AdamW learning rate (default: 2e-4)\n"
        "  --seg <frames>   Training segment length in frames (default: 40)\n"
        "  --seed <n>       RNG seed (default: 0)\n"
        "  --help           Show this help\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string hubert_path, rmvpe_path, pretrained, target, out, dmodel;
    int speaker = 0, steps = 300, seg = 40, epochs = 0;
    unsigned seed = 0;
    float lr = 2e-4f;
    bool gan_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* n) { return (i + 1 < argc) ? argv[++i] : (std::cerr << "missing value for " << n << "\n", ""); };
        if (a == "--help" || a == "-h") { print_usage(); return 0; }
        else if (a == "--gan") { gan_mode = true; continue; }
        else if (a == "--dmodel") dmodel = next("--dmodel");
        else if (a == "--hubert") hubert_path = next("--hubert");
        else if (a == "--rmvpe") rmvpe_path = next("--rmvpe");
        else if (a == "--pretrained") pretrained = next("--pretrained");
        else if (a == "--target") target = next("--target");
        else if (a == "--out") out = next("--out");
        else if (a == "--speaker") speaker = std::atoi(next("--speaker"));
        else if (a == "--steps") steps = std::atoi(next("--steps"));
        else if (a == "--epochs") epochs = std::atoi(next("--epochs"));
        else if (a == "--lr") lr = std::atof(next("--lr"));
        else if (a == "--seg") seg = std::atoi(next("--seg"));
        else if (a == "--seed") seed = static_cast<unsigned>(std::atoi(next("--seed")));
        else { std::cerr << "Unknown argument: " << a << "\n"; return 1; }
    }

    if (hubert_path.empty() || rmvpe_path.empty() || pretrained.empty() ||
        target.empty() || out.empty()) {
        std::cerr << "error: --hubert, --rmvpe, --pretrained, --target, --out are required\n\n";
        print_usage();
        return 1;
    }

    Device device;
    if (auto err = device.init("cuda", 0)) { std::cerr << "CUDA init: " << *err << "\n"; return 1; }
    std::cout << "Device: " << device.name() << "\n";

    // Front-end models (inference components produce the run-time latents).
    content::HubertConfig hcfg; hcfg.model_path = hubert_path; hcfg.output_dim = 768; hcfg.num_layers = 12;
    content::HubertEncoder hubert;
    if (!hubert.init(hcfg)) { std::cerr << "HuBERT init failed\n"; return 1; }

    f0::RmvpeConfig rcfg; rcfg.model_path = rmvpe_path;
    f0::RmvpeExtractor rmvpe;
    if (!rmvpe.init(rcfg)) { std::cerr << "RMVPE init failed\n"; return 1; }

    synthesizer::SynthesizerConfig scfg;
    scfg.model_path = pretrained; scfg.version = ModelVersion::kV2;
    scfg.sample_rate = 40000; scfg.spk_embed_dim = 109;
    synthesizer::Synthesizer synth;
    if (!synth.init(scfg)) { std::cerr << "Synthesizer init failed\n"; return 1; }

    training::GeneratorTrainer trainer;
    if (!trainer.init(pretrained, speaker)) { std::cerr << "Trainer init failed\n"; return 1; }

    // Gather target audio files.
    std::vector<std::string> files;
    if (fs::is_directory(target)) {
        for (const auto& e : fs::directory_iterator(target)) {
            auto p = e.path();
            std::string ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".wav" || ext == ".flac") files.push_back(p.string());
        }
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(target);
    }
    if (files.empty()) { std::cerr << "error: no audio under " << target << "\n"; return 1; }
    std::cout << "Target clips: " << files.size() << "\n";

    // Extract per-clip latents (z, har) and the real 40k reconstruction target.
    std::vector<Clip> clips;
    for (const auto& f : files) {
        auto a16opt = io::read_audio(f, 16000);
        if (!a16opt) { std::cerr << "skip (read failed): " << f << "\n"; continue; }
        const auto& a16 = a16opt->data;
        int n16 = static_cast<int>(a16.size());

        auto feats = hubert.extract(a16.data(), n16);
        if (feats.empty()) { std::cerr << "skip (hubert): " << f << "\n"; continue; }
        int T = static_cast<int>(feats.size()) / 768;
        auto feats_up = interpolate_2x(feats, T, 768);
        int Tf = 2 * T;

        auto f0 = rmvpe.infer(a16.data(), n16);
        int p_len = std::min(Tf, static_cast<int>(f0.size()));
        if (p_len < seg) { std::cerr << "skip (short): " << f << "\n"; continue; }
        feats_up.resize(static_cast<size_t>(p_len) * 768);
        f0.resize(p_len);

        std::vector<int> coarse(p_len);
        for (int i = 0; i < p_len; ++i) coarse[i] = f0_to_coarse(f0[i]);

        Clip c;
        c.T = p_len;
        c.z = synth.debug_flow(feats_up.data(), p_len, coarse.data(), speaker);   // [192, p_len]
        c.har = synth.debug_har(f0.data(), p_len);                                 // [p_len*400]
        if (static_cast<int>(c.z.size()) != 192 * p_len || static_cast<int>(c.har.size()) != p_len * kUpp) {
            std::cerr << "skip (latent size): " << f << "\n"; continue;
        }

        // Real 40k target read natively (avoid the 16k bottleneck losing highs).
        std::vector<float> t40;
        if (auto a40 = io::read_audio(f, 40000)) t40 = std::move(a40->data);
        else t40 = io::resample_linear(a16.data(), n16, 16000, 40000);
        c.tgt.assign(static_cast<size_t>(p_len) * kUpp, 0.0f);
        int copyn = std::min<int>(t40.size(), p_len * kUpp);
        std::copy(t40.begin(), t40.begin() + copyn, c.tgt.begin());

        if (gan_mode) {
            int Ts = 0;
            c.spec = training::compute_spec(c.tgt.data(), p_len * kUpp, 2048, kUpp, Ts);
            auto mp_lsp = synth.debug_encp(feats_up.data(), p_len, coarse.data(), true);
            if (Ts != p_len || static_cast<int>(mp_lsp.size()) != 2 * 192 * p_len) {
                std::cerr << "skip (gan feats): " << f << "\n"; continue;
            }
            c.m_p.assign(mp_lsp.begin(), mp_lsp.begin() + 192 * p_len);
            c.logs_p.assign(mp_lsp.begin() + 192 * p_len, mp_lsp.end());
        }

        clips.push_back(std::move(c));
        std::cout << "  " << fs::path(f).filename().string() << ": " << p_len << " frames\n";
    }
    if (clips.empty()) { std::cerr << "error: no usable clips\n"; return 1; }

    // Epoch = one pass over all segments; steps scale with the dataset size (RVC-style).
    if (epochs > 0) {
        long long total_frames = 0;
        for (const auto& c : clips) total_frames += c.T;
        int per_epoch = std::max<int>(1, static_cast<int>(total_frames / seg));
        steps = std::max(1, epochs * per_epoch);
        std::cout << "Epochs " << epochs << " x " << per_epoch << " seg/epoch ("
                  << total_frames << " frames) -> " << steps << " steps\n";
    }

    // Differentiable 40k log-mel loss.
    training::MelSpecConfig mcfg;
    mcfg.n_fft = 1024; mcfg.hop = 256; mcfg.n_mels = 80; mcfg.sample_rate = 40000;
    training::MelLoss mel(mcfg);

    std::mt19937 rng(seed);
    const int Lseg = seg * kUpp;

    if (gan_mode) {
        if (dmodel.empty())
            dmodel = (fs::path(pretrained).parent_path() / "f0D40k.safetensors").string();
        training::GANTrainer gan;
        if (!gan.init(pretrained, dmodel, speaker, mcfg, lr, lr)) {
            std::cerr << "error: GAN init failed (need " << dmodel << ")\n"; return 1;
        }
        std::cout << "GAN fine-tune: steps=" << steps << " seg=" << seg << " lr=" << lr << "\n";
        float first = -1.0f, last = 0.0f;
        for (int it = 0; it < steps; ++it) {
            const Clip& c = clips[rng() % clips.size()];
            int maxs = c.T - seg;
            int s = (maxs > 0) ? static_cast<int>(rng() % (maxs + 1)) : 0;
            std::vector<float> spec_seg(1025 * seg), mp_seg(192 * seg), lsp_seg(192 * seg),
                har_seg(Lseg), tgt_seg(Lseg);
            for (int k = 0; k < 1025; ++k)
                for (int t = 0; t < seg; ++t) spec_seg[k * seg + t] = c.spec[k * c.T + (s + t)];
            for (int k = 0; k < 192; ++k)
                for (int t = 0; t < seg; ++t) {
                    mp_seg[k * seg + t] = c.m_p[k * c.T + (s + t)];
                    lsp_seg[k * seg + t] = c.logs_p[k * c.T + (s + t)];
                }
            std::copy(c.har.begin() + static_cast<size_t>(s) * kUpp,
                      c.har.begin() + static_cast<size_t>(s + seg) * kUpp, har_seg.begin());
            std::copy(c.tgt.begin() + static_cast<size_t>(s) * kUpp,
                      c.tgt.begin() + static_cast<size_t>(s + seg) * kUpp, tgt_seg.begin());
            auto ls = gan.train_step_full(spec_seg, har_seg, tgt_seg, mp_seg, lsp_seg, seg, Lseg);
            if (first < 0.0f) first = ls.mel;
            last = ls.mel;
            if ((it + 1) % 10 == 0 || it == 0)
                std::printf("  step %4d/%d  D=%.3f G=%.3f (mel=%.4f kl=%.4f fm=%.4f adv=%.4f)\n",
                            it + 1, steps, ls.d, ls.g, ls.mel, ls.kl, ls.fm, ls.adv);
        }
        std::printf("GAN fine-tune done: mel %.4f -> %.4f\n", first, last);
        if (!gan.export_model(pretrained, out)) { std::cerr << "error: export failed\n"; return 1; }
        std::cout << "Exported fine-tuned model: " << out << "\n";
        return 0;
    }

    ag::AdamW opt(trainer.params(), lr);
    std::cout << "Fine-tuning decoder: steps=" << steps << " seg=" << seg
              << " lr=" << lr << "\n";
    float running = 0.0f; int rn = 0; float first = -1.0f, last = 0.0f;
    for (int it = 0; it < steps; ++it) {
        const Clip& c = clips[rng() % clips.size()];
        int maxs = c.T - seg;
        int s = (maxs > 0) ? static_cast<int>(rng() % (maxs + 1)) : 0;

        std::vector<float> z_seg(192 * seg), har_seg(Lseg), tgt_seg(Lseg);
        for (int ch = 0; ch < 192; ++ch)
            for (int t = 0; t < seg; ++t)
                z_seg[ch * seg + t] = c.z[ch * c.T + (s + t)];
        std::copy(c.har.begin() + static_cast<size_t>(s) * kUpp,
                  c.har.begin() + static_cast<size_t>(s + seg) * kUpp, har_seg.begin());
        std::copy(c.tgt.begin() + static_cast<size_t>(s) * kUpp,
                  c.tgt.begin() + static_cast<size_t>(s + seg) * kUpp, tgt_seg.begin());

        int Tm = 0;
        auto tgt_host = mel.target_log_mel(tgt_seg.data(), Lseg, Tm);
        auto tgt = ag::Tensor::from_host(tgt_host, {Tm, mcfg.n_mels}, false);

        auto zc = ag::Tensor::from_host(z_seg, {192, seg}, false);
        auto hc = ag::Tensor::from_host(har_seg, {1, Lseg}, false);
        auto gen = trainer.decode(zc, hc, seg);
        auto gm = mel.log_mel(gen, Lseg);
        auto loss = mel.l1(gm, tgt, Tm);

        float lv = loss.to_host()[0];
        if (first < 0.0f) first = lv;
        last = lv;
        running += lv; ++rn;

        ag::backward(loss);
        opt.step();

        if ((it + 1) % 20 == 0 || it == 0) {
            std::printf("  step %4d/%d  mel-L1 %.4f (avg %.4f)\n",
                        it + 1, steps, lv, running / rn);
            running = 0.0f; rn = 0;
        }
    }
    std::printf("Fine-tune done: mel-L1 %.4f -> %.4f\n", first, last);

    // Export an inference-ready G.
    if (!trainer.export_model(pretrained, out)) {
        std::cerr << "error: export failed\n";
        return 1;
    }
    std::cout << "Exported fine-tuned model: " << out << "\n";
    return 0;
}
