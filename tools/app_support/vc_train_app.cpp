// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "tools/app_support/vc_train_app.h"
#include "tools/app_support/progress_protocol.h"
#include "tools/app_support/joint_full_training.h"

#include <algorithm>
#include <cctype>
#include <chrono>
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
#include "voxmutatio/training/joint_trainer.h"
#include "src/training/joint_features.h"
#include "src/io/bounded_audio_io.h"

namespace voxmutatio::app_support {
namespace {

namespace ag = voxmutatio::autograd;
namespace fs = std::filesystem;

constexpr int kUpp = 400;

template<class Function>
auto timed(double& milliseconds, Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    auto result = function();
    milliseconds += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

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

void validate_joint_audio(const io::detail::WavMetadata& m) {
    // Preprocessed clips only: reject before payload allocation or model loading.
    if(m.sample_rate!=40000 || m.frame_count<32*400 || m.frame_count>600*400)
        throw std::invalid_argument("joint input requires 40 kHz WAV clips between 0.32 and 6 seconds; run vc_preprocess first");
}

int run_joint_training(const VcTrainOptions& options,
                       resonantia::training_control::TrainingCancelReader& cancellation) {
    using namespace resonantia::training_control;
    const auto stop=[&]() {return static_cast<int>(cancellation.poll());};
    if(!options.gan_mode || options.steps<1 || options.steps>50 || options.epochs!=0 ||
       options.segment_frames!=32 || options.synthesis_mode!=SynthesisMode::kSampled ||
       options.learning_rate_decay!=1.f || !std::isfinite(options.learning_rate) ||
       options.learning_rate<=0 || options.learning_rate>1)
        throw std::invalid_argument("unsupported bounded joint training options");
    std::vector<fs::path> files;
    if(fs::is_directory(options.target_path)) {
        for(const auto& entry:fs::directory_iterator(options.target_path)) {
            auto extension=entry.path().extension().string();
            std::transform(extension.begin(),extension.end(),extension.begin(),
                [](unsigned char c){return static_cast<char>(std::tolower(c));});
            if(extension==".wav")files.push_back(entry.path());
        }
        std::sort(files.begin(),files.end());
    } else files.emplace_back(options.target_path);
    if(files.empty())throw std::invalid_argument("no joint training WAV clips");
    for(const auto& file:files) {
        if(const int result=stop())return result;
        io::detail::WavChunkReader reader;
        if(!reader.open(file))throw std::invalid_argument("cannot read joint training WAV: "+file.string());
        validate_joint_audio(reader.metadata());
    }
    if(const int result=stop())return result;
    Device device;
    if(auto error=device.init("cuda",0))throw std::runtime_error("CUDA init: "+*error);
    std::cout<<"Device: "<<device.name()<<"\n";
    content::HubertConfig hc;hc.model_path=options.hubert_path;hc.output_dim=768;
    hc.num_layers=12;hc.use_final_proj=false;
    content::HubertEncoder hubert;
    if(!hubert.init(hc))throw std::runtime_error("joint HuBERT init failed");
    if(const int result=stop())return result;
    f0::RmvpeConfig fc;fc.model_path=options.rmvpe_path;f0::RmvpeExtractor rmvpe;
    if(!rmvpe.init(fc))throw std::runtime_error("joint RMVPE init failed");
    if(const int result=stop())return result;
    const auto discriminator=options.discriminator_path.empty()?
        (fs::path(options.pretrained_path).parent_path()/"f0D40k.safetensors").string():options.discriminator_path;
    training::JointTrainer trainer;
    if(!trainer.init(options.pretrained_path,discriminator,options.speaker,options.learning_rate,options.seed))
        throw std::runtime_error("joint trainer init failed");
    std::mt19937 random(options.seed);
    std::cout<<"Joint GAN: clips="<<files.size()<<" steps="<<options.steps
             <<" crop=32 max_frames=600 lr="<<options.learning_rate
             <<" lr_decay=1 seed="<<options.seed<<" fresh posterior/NSF noise per step\n";
    const auto start=std::chrono::steady_clock::now();
    const auto result=run_training_iterations(options.steps,cancellation,[&](int iteration) {
        const auto slot=static_cast<std::size_t>(iteration)%files.size();
        if(slot==0)std::shuffle(files.begin(),files.end(),random);
        // Only the current clip and its features exist on host. Never cache
        // the dataset's spectra, priors or sampled excitation across steps.
        training::JointTrainingClip clip;
        {
            io::detail::WavChunkReader reader;
            if(!reader.open(files[slot]))throw std::runtime_error("joint input changed or cannot be opened");
            validate_joint_audio(reader.metadata());
            if(!reader.read_mono_frames(0,reader.metadata().frame_count,clip.audio) || !reader.verify_unchanged())
                throw std::runtime_error("joint input changed or contains invalid audio");
        }
        const auto audio16=io::resample_linear(clip.audio.data(),static_cast<int>(clip.audio.size()),40000,16000);
        if(audio16.empty())throw std::runtime_error("joint audio resampling failed");
        const auto features=hubert.extract(audio16.data(),static_cast<int>(audio16.size()));
        if(features.empty() || features.size()%768)throw std::runtime_error("joint HuBERT feature layout failed");
        clip.phone=interpolate_2x(features,static_cast<int>(features.size()/768),768);
        clip.f0=rmvpe.infer(audio16.data(),static_cast<int>(audio16.size()));
        clip.frames=static_cast<int>(std::min({clip.phone.size()/768,clip.f0.size(),clip.audio.size()/400}));
        if(clip.frames<32 || clip.frames>600)throw std::runtime_error("joint clip has insufficient or excessive aligned frames");
        // Match the frontend's common valid extent; never shorten an oversized
        // input to pass the six-second contract (metadata was checked first).
        clip.phone.resize(clip.frames*768);clip.f0.resize(clip.frames);clip.audio.resize(clip.frames*400);
        clip.pitch.resize(clip.frames);
        for(int t=0;t<clip.frames;++t)clip.pitch[t]=f0_to_coarse(clip.f0[t]);
        clip.spectrum=training::detail::joint_spectrum(clip.audio);
        training::JointStepOptions step;step.crop_frames=32;
        step.crop_start=std::uniform_int_distribution<int>(0,clip.frames-32)(random);
        const auto loss=trainer.step(clip,step);
        std::printf("  joint step %d/%d frames=%d crop=%d D=%.6f G=%.6f mel=%.6f kl=%.6f fm=%.6f adv=%.6f\n",
            iteration+1,options.steps,clip.frames,step.crop_start,loss.discriminator,loss.generator,
            loss.mel,loss.kl,loss.feature,loss.adversarial);
        std::fflush(stdout);
        if(options.progress)write_progress_message(std::cout,{"training",std::uint64_t(iteration+1),std::uint64_t(options.steps),"steps"});
        return true;
    });
    if(result!=TrainingResult::kCompleted) {
        if(result==TrainingResult::kCancelled)std::cout<<"Training cancelled at safe point; no model exported\n";
        return static_cast<int>(result);
    }
    if(!trainer.export_model(options.output_path))throw std::runtime_error("joint model export failed");
    std::cout<<"Exported "<<options.output_path<<"\n";
    const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"Joint training done: "<<seconds<<" seconds; "<<seconds/options.steps<<" seconds/step including frontends\n";
    return 0;
}

void print_usage(std::ostream& output) {
    output <<
        "Usage:\n"
        "  vc_train --hubert <dir> --rmvpe <path> --pretrained <G.safetensors>\n"
        "           --target <wav|dir> --out <G_finetuned.safetensors> [options]\n\n"
        "Options:\n"
        "  --gan            Full GAN fine-tune (enc_q+flow+dec vs discriminator)\n"
        "  --gan-profile <legacy|joint|joint-full>  GAN recipe (default: legacy)\n"
        "                   joint: 40 kHz WAV <=6 s, 1..50 steps, crop 32, epochs 0, sampled\n"
        "                   joint defaults: 50 steps, lr 1e-4, no lr decay\n"
        "                   joint-full: 200 epochs, batch 1, crop 32, lr 1e-4\n"
        "  --cache-dir <dir> Frozen features on disk (required for joint-full)\n"
        "  --checkpoint-dir <dir> Full state every 5 epochs; keep latest 2\n"
        "                         Candidates at epochs 50/100/150/200\n"
        "  --resume <file>  Resume joint-full state with matching recipe/dataset\n"
        "  --resume-initial Restart interrupted initialization with no published state\n"
        "  --dmodel <path>  Discriminator safetensors (default: f0D40k next to G)\n"
        "  --speaker <id>   Source speaker embedding id (default: 0)\n"
        "  --steps <n>      Training steps (default: 300)\n"
        "  --epochs <n>     Passes over the dataset; overrides --steps (RVC-style, ~200)\n"
        "  --lr <f>         AdamW learning rate (default: 2e-4)\n"
        "  --lr-decay <f>   Per-epoch exponential LR decay (default: 0.999875)\n"
        "  --seg <frames>   Training segment length in frames (default: 40)\n"
        "  --seed <n>       Dataset and synthesis RNG seed (default: 0)\n"
        "  --synthesis-mode <m> sampled (default) or legacy mean\n"
        "  --cancel-file <path> Optional job channel (0=run, 1=cancel at safe point)\n"
        "  --progress          Emit structured completed-boundary progress\n"
        "  --help           Show this help\n"
        "Exit status: 0=completed, 1=error, 2=cooperatively cancelled (no export)\n";
}

}  // namespace

ParseStatus parse_vc_train_options(int argc, char* const argv[],
                                   VcTrainOptions& options,
                                   std::ostream& output,
                                   std::ostream& error) {
    bool profile_seen=false,steps_seen=false,segment_seen=false,lr_seen=false,decay_seen=false,epochs_seen=false;
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
        } else if (argument == "--gan-profile") {
            const char* value=next(argument);if(!value)return ParseStatus::kExitFailure;
            if(profile_seen || (std::string_view(value)!="legacy" && std::string_view(value)!="joint" && std::string_view(value)!="joint-full")) {
                error<<"error: --gan-profile must be legacy, joint or joint-full and cannot be repeated\n";
                return ParseStatus::kExitFailure;
            }
            profile_seen=true;options.gan_profile=value;
        } else if(argument=="--cache-dir" || argument=="--checkpoint-dir" || argument=="--resume") {
            const char* value=next(argument);if(!value)return ParseStatus::kExitFailure;
            auto& dest=argument=="--cache-dir"?options.cache_dir:argument=="--checkpoint-dir"?options.checkpoint_dir:options.resume_path;
            if(!*value || !dest.empty()){error<<"error: "<<argument<<" requires a nonempty path and cannot be repeated\n";return ParseStatus::kExitFailure;}
            dest=value;
        } else if (argument == "--resume-initial") {
            if(options.resume_initial){error<<"error: --resume-initial cannot be repeated\n";return ParseStatus::kExitFailure;}
            options.resume_initial=true;
        } else if (argument == "--dmodel") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.discriminator_path = value;
        } else if (argument == "--progress") {options.progress=true;
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
            steps_seen=true;
            if (!cli::parse_integer_option(argc, argv, i, argument, 1, 100'000,
                                           options.steps, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--epochs") {
            epochs_seen=true;
            if (!cli::parse_integer_option(argc, argv, i, argument, 0, 2'000,
                                           options.epochs, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--lr") {
            lr_seen=true;
            if (!cli::parse_real_option(argc, argv, i, argument, 0.0f, 1.0f,
                                        options.learning_rate, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--lr-decay") {
            decay_seen=true;
            if (!cli::parse_real_option(argc, argv, i, argument, 0.0f, 1.0f,
                                        options.learning_rate_decay, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--seg") {
            segment_seen=true;
            if (!cli::parse_integer_option(argc, argv, i, argument, 8, 200,
                                           options.segment_frames, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--synthesis-mode") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            if (!cli::parse_synthesis_mode(value, options.synthesis_mode)) {
                error << "error: --synthesis-mode must be sampled or mean\n";
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

    if(profile_seen && !options.gan_mode) {
        error<<"error: --gan-profile requires --gan\n";return ParseStatus::kExitFailure;
    }
    if(options.gan_profile=="joint") {
        if(!steps_seen)options.steps=50;
        if(!segment_seen)options.segment_frames=32;
        if(!lr_seen)options.learning_rate=1e-4f;
        if(!decay_seen)options.learning_rate_decay=1.f;
        if(options.steps>50 || options.epochs!=0 || options.segment_frames!=32 ||
           options.synthesis_mode!=SynthesisMode::kSampled || options.learning_rate_decay!=1.f) {
            error<<"error: joint profile requires steps 1..50, epochs 0, seg 32, sampled synthesis, lr-decay 1\n";
            return ParseStatus::kExitFailure;
        }
    }
    if(options.gan_profile=="joint-full") {
        if(options.resume_initial&&!options.resume_path.empty()) {
            error<<"error: --resume-initial and --resume are mutually exclusive\n";return ParseStatus::kExitFailure;
        }
        if(!epochs_seen)options.epochs=200;
        if(!segment_seen)options.segment_frames=32;
        if(!lr_seen)options.learning_rate=1e-4f;
        if(steps_seen || options.epochs<1 || options.segment_frames!=32 ||
           options.synthesis_mode!=SynthesisMode::kSampled || options.cache_dir.empty() || options.checkpoint_dir.empty()) {
            error<<"error: joint-full requires epochs 1..2000, seg 32, sampled, cache-dir and checkpoint-dir; steps is unsupported\n";
            return ParseStatus::kExitFailure;
        }
    } else if(!options.cache_dir.empty() || !options.checkpoint_dir.empty() || !options.resume_path.empty() || options.resume_initial) {
        error<<"error: cache-dir, checkpoint-dir, resume and resume-initial require joint-full\n";return ParseStatus::kExitFailure;
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
    if(options.gan_profile=="joint-full")return run_full_joint_training(options,cancellation);
    if(options.gan_profile=="joint")return run_joint_training(options,cancellation);
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
    synthesizer_config.synthesis_mode = options.synthesis_mode;
    synthesizer_config.synthesis_seed = options.seed;
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
    if (!options.gan_mode && !trainer.init(options.pretrained_path, options.speaker)) {
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
    double hubert_ms=0, f0_ms=0, latent_ms=0, harmonic_ms=0, spec_ms=0, prior_ms=0;
    for (const auto& file : files) {
        if (const int stop = checkpoint()) return stop;
        auto audio_16k = io::read_audio(file, 16'000);
        if (!audio_16k) {
            std::cerr << "skip (read failed): " << file << "\n";
            continue;
        }
        const auto& samples_16k = audio_16k->data;
        const int sample_count_16k = static_cast<int>(samples_16k.size());

        auto features = timed(hubert_ms, [&] { return hubert.extract(samples_16k.data(), sample_count_16k); });
        if (features.empty()) {
            std::cerr << "skip (hubert): " << file << "\n";
            continue;
        }
        const int feature_frames = static_cast<int>(features.size()) / 768;
        auto upsampled_features = interpolate_2x(features, feature_frames, 768);
        const int upsampled_frames = 2 * feature_frames;

        auto pitch = timed(f0_ms, [&] { return rmvpe.infer(samples_16k.data(), sample_count_16k); });
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
        // Decoder-only training consumes an inference latent. Full GAN training
        // samples enc_q instead, so it needs only the prior stats computed below.
        // Avoid a second enc_p pass, unused inverse Flow, and unused latent cache.
        if (!options.gan_mode) {
            clip.z = timed(latent_ms, [&] { return synthesizer.debug_flow(
                upsampled_features.data(), pitch_frames, coarse_pitch.data(), options.speaker); });
        }
        clip.har = timed(harmonic_ms, [&] { return synthesizer.debug_har(pitch.data(), pitch_frames); });
        if ((!options.gan_mode && static_cast<int>(clip.z.size()) != 192 * pitch_frames) ||
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
            clip.spec = timed(spec_ms, [&] { return training::compute_spec(
                clip.target.data(), pitch_frames * kUpp, 2048, kUpp, spectrogram_frames); });
            auto prior = timed(prior_ms, [&] { return synthesizer.debug_encp(
                upsampled_features.data(), pitch_frames, coarse_pitch.data(), true); });
            if (spectrogram_frames != pitch_frames ||
                static_cast<int>(prior.size()) != 2 * 192 * pitch_frames) {
                std::cerr << "skip (gan feats): " << file << "\n";
                continue;
            }
            clip.prior_mean.assign(prior.begin(), prior.begin() + 192 * pitch_frames);
            clip.prior_log_std.assign(prior.begin() + 192 * pitch_frames, prior.end());
        }

        clips.push_back(std::move(clip));
        if(options.progress)write_progress_message(std::cout,{"features",clips.size(),files.size(),"files"});
        std::cout << "  " << fs::path(file).filename().string()
                  << ": " << pitch_frames << " frames\n";
    }
    if (clips.empty()) {
        std::cerr << "error: no usable clips\n";
        return 1;
    }
    std::printf("Feature timings (ms): HuBERT=%.3f F0=%.3f latent=%.3f harmonic=%.3f spec=%.3f prior=%.3f\n",
                hubert_ms, f0_ms, latent_ms, harmonic_ms, spec_ms, prior_ms);

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
        std::vector<float> spec_segment(1025 * options.segment_frames);
        std::vector<float> mean_segment(192 * options.segment_frames);
        std::vector<float> log_std_segment(192 * options.segment_frames);
        std::vector<float> har_segment(segment_samples);
        std::vector<float> target_segment(segment_samples);
        const auto training_start = std::chrono::steady_clock::now();
        // One body is a complete D/G pair, including existing LR bookkeeping.
        const auto result = run_training_iterations(options.steps, cancellation, [&](int iteration) {
            const Clip& clip = clips[random() % clips.size()];
            const int maximum_start = clip.frames - options.segment_frames;
            const int start = maximum_start > 0
                ? static_cast<int>(random() % (maximum_start + 1)) : 0;
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
            if(options.progress)write_progress_message(std::cout,{"training",std::uint64_t(iteration+1),std::uint64_t(options.steps),"steps"});
            return true;
        });
        if (result != TrainingResult::kCompleted) return report(result);
        const double training_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - training_start).count();
        std::printf("Training timing: %.3f ms total, %.3f ms/step\n", training_ms, training_ms/options.steps);
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
    std::vector<float> z_segment(192 * options.segment_frames);
    std::vector<float> har_segment(segment_samples);
    std::vector<float> target_segment(segment_samples);
    // Keep each complete decoder forward/backward/update indivisible.
    const auto result = run_training_iterations(options.steps, cancellation, [&](int iteration) {
        const Clip& clip = clips[random() % clips.size()];
        const int maximum_start = clip.frames - options.segment_frames;
        const int start = maximum_start > 0
            ? static_cast<int>(random() % (maximum_start + 1)) : 0;

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
        if(options.progress)write_progress_message(std::cout,{"training",std::uint64_t(iteration+1),std::uint64_t(options.steps),"steps"});
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
