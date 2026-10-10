// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "tools/app_support/vc_train_app.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <ostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "tools/app_support/cli_parse.h"
#include "tools/app_support/training_cancel.h"
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

namespace voxmutatio::app_support {
namespace {

namespace ag = voxmutatio::autograd;
namespace fs = std::filesystem;

constexpr int kUpp = 400;

int f0_to_coarse(float f0) {
    const double minimum = 1127.0 * std::log(1.0 + 50.0 / 700.0);
    const double maximum = 1127.0 * std::log(1.0 + 1100.0 / 700.0);
    if (f0 <= 0.0f) return 1;
    double mel = 1127.0 * std::log(1.0 + f0 / 700.0);
    mel = (mel - minimum) * 254.0 / (maximum - minimum) + 1.0;
    if (mel <= 1.0) mel = 1.0;
    if (mel > 255.0) mel = 255.0;
    return std::clamp(static_cast<int>(std::lround(mel)), 1, 255);
}

std::vector<float> interpolate_2x(const std::vector<float>& features,
                                  int frames, int dimensions) {
    std::vector<float> output(static_cast<std::size_t>(2) * frames * dimensions);
    for (int frame = 0; frame < frames; ++frame) {
        std::memcpy(&output[(2 * frame) * dimensions],
                    &features[frame * dimensions], dimensions * sizeof(float));
        std::memcpy(&output[(2 * frame + 1) * dimensions],
                    &features[frame * dimensions], dimensions * sizeof(float));
    }
    return output;
}

struct Clip {
    std::vector<float> z;
    std::vector<float> har;
    std::vector<float> target;
    std::vector<float> spec;
    std::vector<float> prior_mean;
    std::vector<float> prior_log_std;
    int frames = 0;
};

void print_usage(std::ostream& output) {
    output <<
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
        "  --lr-decay <f>   Per-epoch exponential LR decay (default: 0.999875)\n"
        "  --seg <frames>   Training segment length in frames (default: 40)\n"
        "  --seed <n>       RNG seed (default: 0)\n"
        "  --cancel-file <path> Optional job channel (0=run, 1=cancel at safe point)\n"
        "  --help           Show this help\n"
        "Exit status: 0=completed, 1=error, 2=cooperatively cancelled (no export)\n";
}

}  // namespace

ParseStatus parse_vc_train_options(int argc, char* const argv[],
                                   VcTrainOptions& options,
                                   std::ostream& output,
                                   std::ostream& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&](std::string_view option) {
            return cli::option_value(argc, argv, i, option, error);
        };
        if (argument == "--help" || argument == "-h") {
            print_usage(output);
            return ParseStatus::kExitSuccess;
        } else if (argument == "--gan") {
            options.gan_mode = true;
        } else if (argument == "--dmodel") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.discriminator_path = value;
        } else if (argument == "--hubert") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.hubert_path = value;
        } else if (argument == "--rmvpe") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.rmvpe_path = value;
        } else if (argument == "--pretrained") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.pretrained_path = value;
        } else if (argument == "--target") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.target_path = value;
        } else if (argument == "--out") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.output_path = value;
        } else if (argument == "--cancel-file") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            if (!options.cancel_file.empty() || !*value) {
                error << "error: --cancel-file requires one nonempty path and cannot be repeated\n";
                return ParseStatus::kExitFailure;
            }
            options.cancel_file = value;
        } else if (argument == "--speaker") {
            if (!cli::parse_integer_option(argc, argv, i, argument, 0, 108,
                                           options.speaker, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--steps") {
            if (!cli::parse_integer_option(argc, argv, i, argument, 1, 100'000,
                                           options.steps, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--epochs") {
            if (!cli::parse_integer_option(argc, argv, i, argument, 0, 2'000,
                                           options.epochs, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--lr") {
            if (!cli::parse_real_option(argc, argv, i, argument, 0.0f, 1.0f,
                                        options.learning_rate, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--lr-decay") {
            if (!cli::parse_real_option(argc, argv, i, argument, 0.0f, 1.0f,
                                        options.learning_rate_decay, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--seg") {
            if (!cli::parse_integer_option(argc, argv, i, argument, 8, 200,
                                           options.segment_frames, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--seed") {
            if (!cli::parse_integer_option(argc, argv, i, argument, 0U,
                                           std::numeric_limits<unsigned>::max(),
                                           options.seed, error)) {
                return ParseStatus::kExitFailure;
            }
        } else {
            error << "Unknown argument: " << argument << '\n';
            return ParseStatus::kExitFailure;
        }
    }

    if (options.hubert_path.empty() || options.rmvpe_path.empty() ||
        options.pretrained_path.empty() || options.target_path.empty() ||
        options.output_path.empty()) {
        error << "error: --hubert, --rmvpe, --pretrained, --target, --out are required\n\n";
        print_usage(output);
        return ParseStatus::kExitFailure;
    }
    if (options.speaker < 0 || options.speaker >= 109 || options.steps <= 0 ||
        options.steps > 100'000 || options.epochs < 0 || options.epochs > 2'000 ||
        options.segment_frames < 8 || options.segment_frames > 200 ||
        !std::isfinite(options.learning_rate) || options.learning_rate <= 0.0f ||
        options.learning_rate > 1.0f ||
        !std::isfinite(options.learning_rate_decay) ||
        options.learning_rate_decay <= 0.0f || options.learning_rate_decay > 1.0f) {
        error << "error: invalid training parameters (speaker 0..108, steps 1..100000, "
                 "epochs 0..2000, seg 8..200, lr 0..1, lr-decay 0..1)\n";
        return ParseStatus::kExitFailure;
    }
    return ParseStatus::kRun;
}

int run_vc_train(VcTrainOptions options) {
    using namespace resonantia::training_control;
    TrainingCancelReader cancellation(options.cancel_file);
    const auto report = [&](TrainingResult result) {
        if (result == TrainingResult::kCancelled)
            std::cout << "Training cancelled at safe point; no model exported\n";
        else if (result == TrainingResult::kError && cancellation.error_code())
            std::cerr << "error: training cancellation channel: "
                      << std::strerror(cancellation.error_code()) << "\n";
        return static_cast<int>(result);
    };
    const auto checkpoint = [&] { return report(cancellation.poll()); };
    if (const int stop = checkpoint()) return stop;
    Device device;
    if (auto error = device.init("cuda", 0)) {
        std::cerr << "CUDA init: " << *error << "\n";
        return 1;
    }
    std::cout << "Device: " << device.name() << "\n";
    if (const int stop = checkpoint()) return stop;

    content::HubertConfig hubert_config;
    hubert_config.model_path = options.hubert_path;
    hubert_config.output_dim = 768;
    hubert_config.num_layers = 12;
    hubert_config.use_final_proj = false;
    content::HubertEncoder hubert;
    if (!hubert.init(hubert_config)) {
        std::cerr << "HuBERT init failed\n";
        return 1;
    }

    if (const int stop = checkpoint()) return stop;

    f0::RmvpeConfig rmvpe_config;
    rmvpe_config.model_path = options.rmvpe_path;
    f0::RmvpeExtractor rmvpe;
    if (!rmvpe.init(rmvpe_config)) {
        std::cerr << "RMVPE init failed\n";
        return 1;
    }

    if (const int stop = checkpoint()) return stop;

    synthesizer::SynthesizerConfig synthesizer_config;
    synthesizer_config.model_path = options.pretrained_path;
    synthesizer_config.version = ModelVersion::kV2;
    synthesizer_config.sample_rate = 40'000;
    synthesizer_config.spk_embed_dim = 109;
    synthesizer::Synthesizer synthesizer;
    if (!synthesizer.init(synthesizer_config)) {
        std::cerr << "Synthesizer init failed\n";
        return 1;
    }

    if (const int stop = checkpoint()) return stop;

    training::GeneratorTrainer trainer;
    if (!trainer.init(options.pretrained_path, options.speaker)) {
        std::cerr << "Trainer init failed\n";
        return 1;
    }

    if (const int stop = checkpoint()) return stop;

    std::vector<std::string> files;
    if (fs::is_directory(options.target_path)) {
        for (const auto& entry : fs::directory_iterator(options.target_path)) {
            const auto path = entry.path();
            std::string extension = path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });
            if (extension == ".wav") files.push_back(path.string());
        }
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(options.target_path);
    }
    if (files.empty()) {
        std::cerr << "error: no audio under " << options.target_path << "\n";
        return 1;
    }
    std::cout << "Target clips: " << files.size() << "\n";

    std::vector<Clip> clips;
    for (const auto& file : files) {
        if (const int stop = checkpoint()) return stop;
        auto audio_16k = io::read_audio(file, 16'000);
        if (!audio_16k) {
            std::cerr << "skip (read failed): " << file << "\n";
            continue;
        }
        const auto& samples_16k = audio_16k->data;
        const int sample_count_16k = static_cast<int>(samples_16k.size());

        auto features = hubert.extract(samples_16k.data(), sample_count_16k);
        if (features.empty()) {
            std::cerr << "skip (hubert): " << file << "\n";
            continue;
        }
        const int feature_frames = static_cast<int>(features.size()) / 768;
        auto upsampled_features = interpolate_2x(features, feature_frames, 768);
        const int upsampled_frames = 2 * feature_frames;

        auto pitch = rmvpe.infer(samples_16k.data(), sample_count_16k);
        const int pitch_frames = std::min(upsampled_frames, static_cast<int>(pitch.size()));
        if (pitch_frames < options.segment_frames) {
            std::cerr << "skip (short): " << file << "\n";
            continue;
        }
        upsampled_features.resize(static_cast<std::size_t>(pitch_frames) * 768);
        pitch.resize(pitch_frames);

        std::vector<int> coarse_pitch(pitch_frames);
        for (int frame = 0; frame < pitch_frames; ++frame) {
            coarse_pitch[frame] = f0_to_coarse(pitch[frame]);
        }

        Clip clip;
        clip.frames = pitch_frames;
        clip.z = synthesizer.debug_flow(upsampled_features.data(), pitch_frames,
                                        coarse_pitch.data(), options.speaker);
        clip.har = synthesizer.debug_har(pitch.data(), pitch_frames);
        if (static_cast<int>(clip.z.size()) != 192 * pitch_frames ||
            static_cast<int>(clip.har.size()) != pitch_frames * kUpp) {
            std::cerr << "skip (latent size): " << file << "\n";
            continue;
        }

        std::vector<float> samples_40k;
        if (auto audio_40k = io::read_audio(file, 40'000)) {
            samples_40k = std::move(audio_40k->data);
        } else {
            samples_40k = io::resample_linear(samples_16k.data(), sample_count_16k,
                                              16'000, 40'000);
        }
        clip.target.assign(static_cast<std::size_t>(pitch_frames) * kUpp, 0.0f);
        const int copy_count = std::min<int>(samples_40k.size(), pitch_frames * kUpp);
        std::copy(samples_40k.begin(), samples_40k.begin() + copy_count,
                  clip.target.begin());

        if (options.gan_mode) {
            int spectrogram_frames = 0;
            clip.spec = training::compute_spec(clip.target.data(), pitch_frames * kUpp,
                                               2048, kUpp, spectrogram_frames);
            auto prior = synthesizer.debug_encp(upsampled_features.data(), pitch_frames,
                                                coarse_pitch.data(), true);
            if (spectrogram_frames != pitch_frames ||
                static_cast<int>(prior.size()) != 2 * 192 * pitch_frames) {
                std::cerr << "skip (gan feats): " << file << "\n";
                continue;
            }
            clip.prior_mean.assign(prior.begin(), prior.begin() + 192 * pitch_frames);
            clip.prior_log_std.assign(prior.begin() + 192 * pitch_frames, prior.end());
        }

        clips.push_back(std::move(clip));
        std::cout << "  " << fs::path(file).filename().string()
                  << ": " << pitch_frames << " frames\n";
    }
    if (clips.empty()) {
        std::cerr << "error: no usable clips\n";
        return 1;
    }

    if (const int stop = checkpoint()) return stop;

    long long total_frames = 0;
    for (const auto& clip : clips) total_frames += clip.frames;
    const int per_epoch = std::max<int>(
        1, static_cast<int>(total_frames / options.segment_frames));
    if (options.epochs > 0) {
        options.steps = std::max(1, options.epochs * per_epoch);
        std::cout << "Epochs " << options.epochs << " x " << per_epoch
                  << " seg/epoch (" << total_frames << " frames) -> "
                  << options.steps << " steps\n";
    }

    training::MelSpecConfig mel_config;
    mel_config.n_fft = 1024;
    mel_config.hop = 256;
    mel_config.n_mels = 80;
    mel_config.sample_rate = 40'000;
    training::MelLoss mel(mel_config);

    std::mt19937 random(options.seed);
    const int segment_samples = options.segment_frames * kUpp;

    if (options.gan_mode) {
        if (options.discriminator_path.empty()) {
            options.discriminator_path =
                (fs::path(options.pretrained_path).parent_path() /
                 "f0D40k.safetensors").string();
        }
        training::GANTrainer gan;
        if (!gan.init(options.pretrained_path, options.discriminator_path,
                      options.speaker, mel_config, options.learning_rate,
                      options.learning_rate)) {
            std::cerr << "error: GAN init failed (need "
                      << options.discriminator_path << ")\n";
            return 1;
        }
        std::cout << "GAN fine-tune: steps=" << options.steps
                  << " seg=" << options.segment_frames
                  << " lr=" << options.learning_rate << "\n";
        float first = -1.0f;
        float last = 0.0f;
        // One body is a complete D/G pair, including existing LR bookkeeping.
        const auto result = run_training_iterations(options.steps, cancellation, [&](int iteration) {
            const Clip& clip = clips[random() % clips.size()];
            const int maximum_start = clip.frames - options.segment_frames;
            const int start = maximum_start > 0
                ? static_cast<int>(random() % (maximum_start + 1)) : 0;
            std::vector<float> spec_segment(1025 * options.segment_frames);
            std::vector<float> mean_segment(192 * options.segment_frames);
            std::vector<float> log_std_segment(192 * options.segment_frames);
            std::vector<float> har_segment(segment_samples);
            std::vector<float> target_segment(segment_samples);
            for (int channel = 0; channel < 1025; ++channel) {
                for (int frame = 0; frame < options.segment_frames; ++frame) {
                    spec_segment[channel * options.segment_frames + frame] =
                        clip.spec[channel * clip.frames + (start + frame)];
                }
            }
            for (int channel = 0; channel < 192; ++channel) {
                for (int frame = 0; frame < options.segment_frames; ++frame) {
                    mean_segment[channel * options.segment_frames + frame] =
                        clip.prior_mean[channel * clip.frames + (start + frame)];
                    log_std_segment[channel * options.segment_frames + frame] =
                        clip.prior_log_std[channel * clip.frames + (start + frame)];
                }
            }
            std::copy(clip.har.begin() + static_cast<std::size_t>(start) * kUpp,
                      clip.har.begin() + static_cast<std::size_t>(
                          start + options.segment_frames) * kUpp,
                      har_segment.begin());
            std::copy(clip.target.begin() + static_cast<std::size_t>(start) * kUpp,
                      clip.target.begin() + static_cast<std::size_t>(
                          start + options.segment_frames) * kUpp,
                      target_segment.begin());
            const auto losses = gan.train_step_full(
                spec_segment, har_segment, target_segment, mean_segment,
                log_std_segment, options.segment_frames, segment_samples);
            if (first < 0.0f) first = losses.mel;
            last = losses.mel;
            if ((iteration + 1) % 10 == 0 || iteration == 0) {
                std::printf(
                    "  step %4d/%d  D=%.3f G=%.3f (mel=%.4f kl=%.4f fm=%.4f adv=%.4f)\n",
                    iteration + 1, options.steps, losses.d, losses.g, losses.mel,
                    losses.kl, losses.fm, losses.adv);
            }
            if (options.learning_rate_decay < 1.0f &&
                (iteration + 1) % per_epoch == 0) {
                gan.decay_lr(options.learning_rate_decay);
            }
            return true;
        });
        if (result != TrainingResult::kCompleted) return report(result);
        std::printf("GAN fine-tune done: mel %.4f -> %.4f (final lr=%.2e)\n",
                    first, last, gan.g_lr());
        if (const int stop = checkpoint()) return stop;
        if (!gan.export_model(options.pretrained_path, options.output_path)) {
            std::cerr << "error: export failed\n";
            return 1;
        }
        std::cout << "Exported fine-tuned model: " << options.output_path << "\n";
        return 0;
    }

    ag::AdamW optimizer(trainer.params(), options.learning_rate);
    std::cout << "Fine-tuning decoder: steps=" << options.steps
              << " seg=" << options.segment_frames
              << " lr=" << options.learning_rate << "\n";
    float running = 0.0f;
    int running_count = 0;
    float first = -1.0f;
    float last = 0.0f;
    // Keep each complete decoder forward/backward/update indivisible.
    const auto result = run_training_iterations(options.steps, cancellation, [&](int iteration) {
        const Clip& clip = clips[random() % clips.size()];
        const int maximum_start = clip.frames - options.segment_frames;
        const int start = maximum_start > 0
            ? static_cast<int>(random() % (maximum_start + 1)) : 0;

        std::vector<float> z_segment(192 * options.segment_frames);
        std::vector<float> har_segment(segment_samples);
        std::vector<float> target_segment(segment_samples);
        for (int channel = 0; channel < 192; ++channel) {
            for (int frame = 0; frame < options.segment_frames; ++frame) {
                z_segment[channel * options.segment_frames + frame] =
                    clip.z[channel * clip.frames + (start + frame)];
            }
        }
        std::copy(clip.har.begin() + static_cast<std::size_t>(start) * kUpp,
                  clip.har.begin() + static_cast<std::size_t>(
                      start + options.segment_frames) * kUpp,
                  har_segment.begin());
        std::copy(clip.target.begin() + static_cast<std::size_t>(start) * kUpp,
                  clip.target.begin() + static_cast<std::size_t>(
                      start + options.segment_frames) * kUpp,
                  target_segment.begin());

        int mel_frames = 0;
        auto target_host = mel.target_log_mel(target_segment.data(), segment_samples,
                                              mel_frames);
        auto target = ag::Tensor::from_host(target_host,
                                            {mel_frames, mel_config.n_mels}, false);
        auto z = ag::Tensor::from_host(
            z_segment, {192, options.segment_frames}, false);
        auto har = ag::Tensor::from_host(har_segment, {1, segment_samples}, false);
        auto generated = trainer.decode(z, har, options.segment_frames);
        auto generated_mel = mel.log_mel(generated, segment_samples);
        auto loss = mel.l1(generated_mel, target, mel_frames);

        const float loss_value = loss.to_host()[0];
        if (!std::isfinite(loss_value)) {
            std::cerr << "error: non-finite decoder loss; update skipped\n";
            return false;
        }
        if (first < 0.0f) first = loss_value;
        last = loss_value;
        running += loss_value;
        ++running_count;

        ag::backward(loss);
        optimizer.clip_grad_norm(100.0f);
        optimizer.step();

        if ((iteration + 1) % 20 == 0 || iteration == 0) {
            std::printf("  step %4d/%d  mel-L1 %.4f (avg %.4f)\n",
                        iteration + 1, options.steps, loss_value,
                        running / running_count);
            running = 0.0f;
            running_count = 0;
        }
        return true;
    });
    if (result != TrainingResult::kCompleted) return report(result);
    std::printf("Fine-tune done: mel-L1 %.4f -> %.4f\n", first, last);

    if (const int stop = checkpoint()) return stop;
    if (!trainer.export_model(options.pretrained_path, options.output_path)) {
        std::cerr << "error: export failed\n";
        return 1;
    }
    std::cout << "Exported fine-tuned model: " << options.output_path << "\n";
    return 0;
}

}  // namespace voxmutatio::app_support
