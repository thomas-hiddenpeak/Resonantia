/**
 * @file vc_preprocess.cpp
 * @brief Slice raw target-voice recordings into fixed-length training clips.
 *
 * Real recordings are usually minutes long; the training front-end runs
 * O(T^2) attention and keeps per-clip latents resident, so long files must be
 * cut into short segments. This tool resamples to the target rate, trims
 * leading/trailing silence, drops silent slices, and writes numbered WAVs
 * ready for vc_train / build_index. Pure C++ — no Python at runtime.
 *
 * Usage:
 *   vc_preprocess --input <wav|dir> --output-dir <dir>
 *                 [--sr 40000] [--seg-sec 3.0] [--hop-sec 0]
 *                 [--min-rms 0.010] [--trim] [--trim-thresh 0.020]
 *                 [--separate] [--sep-model <umxhq_vocals.safetensors>]
 */
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/separation/roformer.h"

namespace fs = std::filesystem;
namespace io = voxmutatio::io;

namespace {

double rms(const float* x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += static_cast<double>(x[i]) * x[i];
    return std::sqrt(s / std::max(1, n));
}

// Trim leading/trailing samples below `thresh`, keeping `pad` samples of guard.
std::pair<int, int> voiced_span(const std::vector<float>& a, float thresh, int pad) {
    int n = static_cast<int>(a.size());
    int lo = 0, hi = n - 1;
    while (lo < n && std::fabs(a[lo]) < thresh) ++lo;
    while (hi > lo && std::fabs(a[hi]) < thresh) --hi;
    if (lo >= hi) return {0, 0};
    lo = std::max(0, lo - pad);
    hi = std::min(n - 1, hi + pad);
    return {lo, hi + 1};
}

void print_usage() {
    std::cout <<
        "Usage:\n"
        "  vc_preprocess --input <wav|dir> --output-dir <dir> [options]\n\n"
        "Options:\n"
        "  --sr <hz>          Output sample rate (default: 40000)\n"
        "  --seg-sec <f>      Segment length in seconds (default: 3.0)\n"
        "  --hop-sec <f>      Hop between segments; 0 = seg-sec, no overlap\n"
        "  --min-rms <f>      Drop slices quieter than this RMS (default: 0.010)\n"
        "  --trim             Trim leading/trailing silence before slicing\n"
        "  --trim-thresh <f>  Amplitude threshold for --trim (default: 0.020)\n"
        "  --separate         Extract vocals (MelBand-RoFormer) before slicing\n"
        "  --dereverb         Remove reverb (MelBand-RoFormer) before slicing\n"
        "  --denoise          Remove noise (MelBand-RoFormer) before slicing\n"
        "  --sep-dir <path>   Separation model dir (default: models/separation)\n"
        "  --help             Show this help\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string input, output_dir;
    int sr = 40000;
    double seg_sec = 3.0, hop_sec = 0.0, min_rms = 0.010, trim_thresh = 0.020;
    bool trim = false, separate = false, dereverb = false, denoise = false;
    std::string sep_dir = "models/separation";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--help" || a == "-h") { print_usage(); return 0; }
        else if (a == "--input") input = next();
        else if (a == "--output-dir") output_dir = next();
        else if (a == "--sr") sr = std::atoi(next());
        else if (a == "--seg-sec") seg_sec = std::atof(next());
        else if (a == "--hop-sec") hop_sec = std::atof(next());
        else if (a == "--min-rms") min_rms = std::atof(next());
        else if (a == "--trim") trim = true;
        else if (a == "--trim-thresh") trim_thresh = std::atof(next());
        else if (a == "--separate") separate = true;
        else if (a == "--dereverb") dereverb = true;
        else if (a == "--denoise") denoise = true;
        else if (a == "--sep-dir") sep_dir = next();
        else { std::cerr << "Unknown argument: " << a << "\n"; return 1; }
    }
    if (input.empty() || output_dir.empty()) {
        std::cerr << "error: --input and --output-dir are required\n\n";
        print_usage();
        return 1;
    }

    std::vector<std::string> files;
    if (fs::is_directory(input)) {
        for (const auto& e : fs::directory_iterator(input)) {
            std::string ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".wav" || ext == ".flac") files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(input);
    }
    if (files.empty()) { std::cerr << "error: no audio under " << input << "\n"; return 1; }

    std::unique_ptr<voxmutatio::separation::Roformer> voc;
    if (separate) {
        voc = std::make_unique<voxmutatio::separation::Roformer>(sep_dir, "vocal_roformer");
        if (!voc->valid()) {
            std::cerr << "error: could not load vocal model in: " << sep_dir << "\n";
            return 1;
        }
        std::cout << "Vocal separation enabled (MelBand-RoFormer)\n";
    }
    std::unique_ptr<voxmutatio::separation::Roformer> derev;
    if (dereverb) {
        derev = std::make_unique<voxmutatio::separation::Roformer>(sep_dir, "dereverb_roformer");
        if (!derev->valid()) {
            std::cerr << "error: could not load de-reverb model in: " << sep_dir << "\n";
            return 1;
        }
        std::cout << "De-reverb enabled (MelBand-RoFormer)\n";
    }
    std::unique_ptr<voxmutatio::separation::Roformer> denoiser;
    if (denoise) {
        denoiser = std::make_unique<voxmutatio::separation::Roformer>(sep_dir, "denoise_roformer");
        if (!denoiser->valid()) {
            std::cerr << "error: could not load de-noise model in: " << sep_dir << "\n";
            return 1;
        }
        std::cout << "De-noise enabled (MelBand-RoFormer)\n";
    }

    fs::create_directories(output_dir);
    const int seg = std::max(1, static_cast<int>(seg_sec * sr));
    const int hop = (hop_sec > 0.0) ? std::max(1, static_cast<int>(hop_sec * sr)) : seg;
    const int min_keep = std::max(1, static_cast<int>(1.0 * sr));  // >=1s tail kept

    int total = 0, dropped = 0;
    for (const auto& f : files) {
        auto opt = io::read_audio(f, sr);
        if (!opt) { std::cerr << "skip (read failed): " << f << "\n"; continue; }
        std::vector<float> a = std::move(opt->data);
        if (voc) {
            a = voc->separate_mono(a.data(), static_cast<int>(a.size()), sr);
        }
        if (derev) {
            a = derev->separate_mono(a.data(), static_cast<int>(a.size()), sr);
        }
        if (denoiser) {
            a = denoiser->separate_mono(a.data(), static_cast<int>(a.size()), sr);
        }
        if (trim) {
            auto [lo, hi] = voiced_span(a, static_cast<float>(trim_thresh), sr / 10);
            if (hi > lo) a = std::vector<float>(a.begin() + lo, a.begin() + hi);
        }
        int n = static_cast<int>(a.size());
        std::string stem = fs::path(f).stem().string();
        int idx = 0;
        for (int start = 0; start < n; start += hop) {
            int len = std::min(seg, n - start);
            if (len < min_keep && start != 0) break;  // drop tiny trailing scrap
            if (rms(a.data() + start, len) < min_rms) { ++dropped; continue; }
            char name[512];
            std::snprintf(name, sizeof(name), "%s/%s_%04d.wav",
                          output_dir.c_str(), stem.c_str(), idx++);
            if (!io::write_audio(name, a.data() + start, len, sr)) {
                std::cerr << "error: write failed: " << name << "\n";
                return 1;
            }
            ++total;
            if (len < seg) break;  // last (short) segment written
        }
        std::cout << "  " << fs::path(f).filename().string()
                  << " -> " << idx << " clips\n";
    }
    std::cout << "Wrote " << total << " clips to " << output_dir
              << " (dropped " << dropped << " silent)\n";
    return 0;
}
