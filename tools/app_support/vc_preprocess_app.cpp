// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "tools/app_support/vc_preprocess_app.h"
#include "tools/app_support/progress_protocol.h"
#include "tools/app_support/operation_checkpoint.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tools/app_support/cli_parse.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/separation/roformer.h"
#include "voxmutatio/separation/sep_registry.h"
#include "voxmutatio/separation/vad.h"

namespace voxmutatio::app_support {
namespace {

namespace fs = std::filesystem;
namespace io = voxmutatio::io;

double rms(const float* samples, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        sum += static_cast<double>(samples[i]) * samples[i];
    }
    return std::sqrt(sum / std::max(1, count));
}

std::pair<int, int> voiced_span(const std::vector<float>& audio,
                                float threshold, int padding) {
    const int count = static_cast<int>(audio.size());
    int low = 0;
    int high = count - 1;
    while (low < count && std::fabs(audio[low]) < threshold) ++low;
    while (high > low && std::fabs(audio[high]) < threshold) --high;
    if (low >= high) return {0, 0};
    low = std::max(0, low - padding);
    high = std::min(count - 1, high + padding);
    return {low, high + 1};
}

void print_usage(std::ostream& output) {
    output <<
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
        "  --checkpoint-dir <dir> Save completed Roformer windows for restart\n"
        "  --resume           Reuse matching input/model/recipe checkpoints\n"
        "  --pause-file <p>   One-byte 0/1 control; safe pause exits 2, errors exit 1\n"
        "  --deharmony        Remove backing/harmony vocals (karaoke RoFormer)\n"
        "  --dereverb         Remove reverb (MelBand-RoFormer) before slicing\n"
        "  --deecho           Remove echo + reverb (MelBand-RoFormer) before slicing\n"
        "  --denoise          Remove noise (MelBand-RoFormer) before slicing\n"
        "  --sep-dir <path>   Separation model dir (default: models/separation)\n"
        "  --separate-model <n>  Vocal model name (default: vocal_roformer)\n"
        "  --deharmony-model <n> De-harmony model name (default: karaoke_roformer)\n"
        "  --dereverb-model <n>  De-reverb model name (default: dereverb_roformer)\n"
        "  --deecho-model <n>    De-echo model name (default: deecho_roformer)\n"
        "  --denoise-model <n>   De-noise model name (default: denoise_roformer)\n"
        "  --no-slice         Apply separation only; write full-length files (no slicing)\n"
        "  --preserve-stereo  Preserve L/R for cleanup (requires --no-slice --sr 44100)\n"
        "  --vad              Slice on Silero-VAD speech regions (smart slicing)\n"
        "  --vad-thresh <f>   VAD speech probability threshold (default: 0.5)\n"
        "  --vad-dir <path>   VAD model dir (default: models/vad)\n"
        "  --progress          Emit structured completed-boundary progress\n"
        "  --help             Show this help\n";
}

}  // namespace

ParseStatus parse_vc_preprocess_options(int argc, char* const argv[],
                                        VcPreprocessOptions& options,
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
        } else if (argument == "--progress") {options.progress=true;
        } else if (argument == "--checkpoint-dir") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.checkpoint_dir=value;
        } else if (argument == "--pause-file") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.pause_file=value;
        } else if (argument == "--resume") {options.resume=true;
        } else if (argument == "--input") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.input = value;
        } else if (argument == "--output-dir") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.output_dir = value;
        } else if (argument == "--sr") {
            if (!cli::parse_integer_option(argc, argv, i, argument, 8'000, 192'000,
                                           options.sample_rate, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--seg-sec") {
            if (!cli::parse_real_option(argc, argv, i, argument, 0.0,
                                        std::numeric_limits<double>::max(),
                                        options.segment_seconds, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--hop-sec") {
            if (!cli::parse_real_option(argc, argv, i, argument, 0.0,
                                        std::numeric_limits<double>::max(),
                                        options.hop_seconds, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--min-rms") {
            if (!cli::parse_real_option(argc, argv, i, argument, 0.0, 1.0,
                                        options.minimum_rms, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--trim") {
            options.trim = true;
        } else if (argument == "--trim-thresh") {
            if (!cli::parse_real_option(argc, argv, i, argument, 0.0, 1.0,
                                        options.trim_threshold, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--separate") {
            options.separate = true;
        } else if (argument == "--deharmony") {
            options.deharmony = true;
        } else if (argument == "--dereverb") {
            options.dereverb = true;
        } else if (argument == "--deecho") {
            options.deecho = true;
        } else if (argument == "--denoise") {
            options.denoise = true;
        } else if (argument == "--sep-dir") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.separation_dir = value;
        } else if (argument == "--separate-model") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.separate_model = value;
        } else if (argument == "--deharmony-model") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.deharmony_model = value;
        } else if (argument == "--dereverb-model") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.dereverb_model = value;
        } else if (argument == "--deecho-model") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.deecho_model = value;
        } else if (argument == "--denoise-model") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.denoise_model = value;
        } else if (argument == "--no-slice") {
            options.no_slice = true;
        } else if (argument == "--preserve-stereo") {
            options.preserve_stereo = true;
        } else if (argument == "--vad") {
            options.vad = true;
        } else if (argument == "--vad-thresh") {
            if (!cli::parse_real_option(argc, argv, i, argument, 0.0, 1.0,
                                        options.vad_threshold, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (argument == "--vad-dir") {
            const char* value = next(argument); if (!value) return ParseStatus::kExitFailure;
            options.vad_dir = value;
        } else {
            error << "Unknown argument: " << argument << '\n';
            return ParseStatus::kExitFailure;
        }
    }

    if (options.input.empty() || options.output_dir.empty()) {
        error << "error: --input and --output-dir are required\n\n";
        print_usage(output);
        return ParseStatus::kExitFailure;
    }

    const long double segment_samples =
        static_cast<long double>(options.segment_seconds) * options.sample_rate;
    const long double hop_samples =
        static_cast<long double>(options.hop_seconds) * options.sample_rate;
    if (segment_samples <= 0.0L ||
        segment_samples > static_cast<long double>(std::numeric_limits<int>::max())) {
        error << "error: invalid value for --seg-sec: must be positive and fit the "
                 "sample-count range\n";
        return ParseStatus::kExitFailure;
    }
    if (hop_samples > static_cast<long double>(std::numeric_limits<int>::max())) {
        error << "error: invalid value for --hop-sec: must fit the sample-count range\n";
        return ParseStatus::kExitFailure;
    }
    if((options.resume||!options.pause_file.empty())&&options.checkpoint_dir.empty()) {
        error << "error: --resume and --pause-file require --checkpoint-dir\n";
        return ParseStatus::kExitFailure;
    }
    return ParseStatus::kRun;
}

int run_vc_preprocess(const VcPreprocessOptions& options) try {
    if (options.preserve_stereo && (!options.no_slice || options.trim || options.vad ||
                                    options.sample_rate != 44100)) {
        std::cerr << "error: --preserve-stereo requires --no-slice --sr 44100 without trim/VAD\n";
        return 1;
    }
    std::vector<std::string> files;
    if (fs::is_directory(options.input)) {
        for (const auto& entry : fs::directory_iterator(options.input)) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });
            if (extension == ".wav") files.push_back(entry.path().string());
        }
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(options.input);
    }
    if (files.empty()) {
        std::cerr << "error: no audio under " << options.input << "\n";
        return 1;
    }

    namespace separation = voxmutatio::separation;
    auto resolve_model = [&](const std::string& task, const std::string& chosen,
                             const char* canonical) -> std::optional<std::string> {
        const std::string model = chosen.empty() ? canonical : chosen;
        if (!separation::sep_model_available(options.separation_dir, task, model)) {
            std::cerr << "error: model '" << model
                      << "' is not registered for task '" << task
                      << "' in: " << options.separation_dir << "\n";
            return std::nullopt;
        }
        return model;
    };

    const auto progress=[&](std::string_view phase,std::uint64_t done,std::uint64_t total,std::string_view unit) {
        if(options.progress)write_progress_message(std::cout,{phase,done,total,unit});
    };
    const auto observer=[&](std::string_view phase)->ProgressObserver {
        if(!options.progress)return {};
        return [&,phase](const OperationProgress& p){progress(phase,p.completed,p.total,p.unit);};
    };
    std::unique_ptr<OperationCheckpointSession> checkpoint_session;
    if(!options.checkpoint_dir.empty()) {
        if(files.size()>4096)throw std::invalid_argument("checkpoint preprocessing accepts at most 4096 files");
        progress("identity",0,files.size(),"files");
        OperationIdentity identity("preprocessing");
        std::ostringstream recipe;recipe.imbue(std::locale::classic());recipe<<std::setprecision(std::numeric_limits<double>::max_digits10);
        recipe<<options.sample_rate<<' '<<options.segment_seconds<<' '<<options.hop_seconds<<' '<<options.minimum_rms<<' '
              <<options.trim_threshold<<' '<<options.trim<<' '<<options.no_slice<<' '<<options.preserve_stereo<<' '
              <<options.vad<<' '<<options.vad_threshold;
        identity.field("recipe",recipe.str());
        for(std::size_t i=0;i<files.size();++i) {
            identity.field("filename",fs::path(files[i]).filename().string());identity.file("input",files[i]);
            progress("identity",i+1,files.size(),"files");
        }
        const auto add_model=[&](bool enabled,const std::string& chosen,const std::string& canonical) {
            identity.field(canonical,enabled?"enabled":"disabled");if(!enabled)return;
            const auto model=chosen.empty()?canonical:chosen;
            identity.field("model",model);
            for(const auto* suffix:{".safetensors",".json","_config.i64","_dim_inputs.i64","_freq_indices.i64","_num_bands_per_freq.i64"})
                identity.file(suffix,fs::path(options.separation_dir)/(model+suffix));
        };
        add_model(options.separate,options.separate_model,"vocal_roformer");
        add_model(options.deharmony,options.deharmony_model,"karaoke_roformer");
        add_model(options.dereverb,options.dereverb_model,"dereverb_roformer");
        add_model(options.deecho,options.deecho_model,"deecho_roformer");
        add_model(options.denoise,options.denoise_model,"denoise_roformer");
        if(options.vad)identity.file("vad",fs::path(options.vad_dir)/"silero_vad.safetensors");
        checkpoint_session=std::make_unique<OperationCheckpointSession>(options.checkpoint_dir,identity.finish(),options.resume,options.pause_file);
    }
    progress("loading-models",0,0,"unknown");
    std::unique_ptr<separation::Roformer> vocal_separator;
    if (options.separate) {
        auto model = resolve_model("separate", options.separate_model, "vocal_roformer");
        if (!model) return 1;
        vocal_separator = std::make_unique<separation::Roformer>(options.separation_dir, *model);
        if (!vocal_separator->valid()) {
            std::cerr << "error: could not load vocal model '" << *model
                      << "' in: " << options.separation_dir << "\n";
            return 1;
        }
        std::cout << "Vocal separation enabled (" << *model << ")\n";
    }

    std::unique_ptr<separation::Roformer> deharmonizer;
    if (options.deharmony) {
        auto model = resolve_model("deharmony", options.deharmony_model, "karaoke_roformer");
        if (!model) return 1;
        deharmonizer = std::make_unique<separation::Roformer>(options.separation_dir, *model);
        if (!deharmonizer->valid()) {
            std::cerr << "error: could not load de-harmony model '" << *model
                      << "' in: " << options.separation_dir << "\n";
            return 1;
        }
        std::cout << "De-harmony enabled (" << *model << ")\n";
    }

    std::unique_ptr<separation::Roformer> dereverberator;
    if (options.dereverb) {
        auto model = resolve_model("dereverb", options.dereverb_model, "dereverb_roformer");
        if (!model) return 1;
        dereverberator = std::make_unique<separation::Roformer>(options.separation_dir, *model);
        if (!dereverberator->valid()) {
            std::cerr << "error: could not load de-reverb model '" << *model
                      << "' in: " << options.separation_dir << "\n";
            return 1;
        }
        std::cout << "De-reverb enabled (" << *model << ")\n";
    }

    std::unique_ptr<separation::Roformer> deechoer;
    if (options.deecho) {
        auto model = resolve_model("deecho", options.deecho_model, "deecho_roformer");
        if (!model) return 1;
        deechoer = std::make_unique<separation::Roformer>(options.separation_dir, *model);
        if (!deechoer->valid()) {
            std::cerr << "error: could not load de-echo model '" << *model
                      << "' in: " << options.separation_dir << "\n";
            return 1;
        }
        std::cout << "De-echo enabled (" << *model << ")\n";
    }

    std::unique_ptr<separation::Roformer> denoiser;
    if (options.denoise) {
        auto model = resolve_model("denoise", options.denoise_model, "denoise_roformer");
        if (!model) return 1;
        denoiser = std::make_unique<separation::Roformer>(options.separation_dir, *model);
        if (!denoiser->valid()) {
            std::cerr << "error: could not load de-noise model '" << *model
                      << "' in: " << options.separation_dir << "\n";
            return 1;
        }
        std::cout << "De-noise enabled (" << *model << ")\n";
    }

    std::unique_ptr<separation::Vad> vad;
    if (options.vad) {
        vad = std::make_unique<separation::Vad>(options.vad_dir);
        if (!vad->valid()) {
            std::cerr << "error: could not load Silero VAD in: " << options.vad_dir << "\n";
            return 1;
        }
        std::cout << "Smart slicing enabled (Silero VAD, thresh "
                  << options.vad_threshold << ")\n";
    }

    fs::create_directories(options.output_dir);
    const long double segment_samples =
        static_cast<long double>(options.segment_seconds) * options.sample_rate;
    const long double hop_samples =
        static_cast<long double>(options.hop_seconds) * options.sample_rate;
    const int segment = std::max(1, static_cast<int>(segment_samples));
    const int hop = options.hop_seconds > 0.0
        ? std::max(1, static_cast<int>(hop_samples)) : segment;
    const int minimum_keep = std::max(1, options.sample_rate);

    int total = 0;
    int dropped = 0;
    int readable = 0;
    std::uint64_t handled=0;
    progress("files",0,files.size(),"files");
    for (const auto& file : files) {
        if(checkpoint_session)checkpoint_session->boundary();
        const auto file_done=[&]{progress("files",++handled,files.size(),"files");
            if(checkpoint_session)checkpoint_session->boundary();};
        const auto checkpoint_for=[&](std::string_view phase)->std::unique_ptr<io::BlockCheckpoint> {
            if(!checkpoint_session)return {};
            return checkpoint_session->stage("file_"+std::to_string(handled)+"_"+std::string(phase));
        };
        if (options.preserve_stereo) {
            auto loaded = io::read_stereo_audio(file);
            if (!loaded || loaded->frames() == 0) {
                std::cerr << "error: stereo input invalid or exceeds 600 seconds / 128 MiB\n";
                return 1;
            }
            const int count = static_cast<int>(loaded->frames());
            std::vector<float> planar(loaded->data.size());
            for (int i = 0; i < count; ++i) {
                planar[i] = loaded->data[2*i]; planar[count+i] = loaded->data[2*i+1];
            }
            for (const auto& [phase,stage] : std::vector<std::pair<std::string_view,separation::Roformer*>>{
                    {"separate",vocal_separator.get()},{"deharmony",deharmonizer.get()},
                    {"dereverb",dereverberator.get()},{"deecho",deechoer.get()},{"denoise",denoiser.get()}}) {
                if (!stage) continue;
                auto checkpoint=checkpoint_for(phase);
                planar = stage->separate_stereo(planar, count,observer(phase),checkpoint.get());
                if (planar.size() != loaded->data.size()) {
                    std::cerr << "error: stereo cleanup changed frame count\n"; return 1;
                }
            }
            for (int i = 0; i < count; ++i) {
                loaded->data[2*i] = planar[i]; loaded->data[2*i+1] = planar[count+i];
            }
            const auto output = fs::path(options.output_dir) / (fs::path(file).stem().string() + ".wav");
            if (!io::write_stereo_audio(output.string(), loaded->data.data(), count, 44100, true)) {
                std::cerr << "error: stereo write failed\n"; return 1;
            }
            ++readable; ++total;
            std::cout << "  " << fs::path(file).filename().string() << " -> stereo processed\n";
            file_done();continue;
        }
        auto loaded = io::read_audio(file, options.sample_rate);
        if (!loaded) {
            std::cerr << "skip (read failed): " << file << "\n";
            file_done();continue;
        }
        ++readable;
        std::vector<float> audio = std::move(loaded->data);
        if (vocal_separator) {
            auto checkpoint=checkpoint_for("separate");
            audio = vocal_separator->separate_mono(
                audio.data(), static_cast<int>(audio.size()), options.sample_rate,observer("separate"),checkpoint.get());
        }
        if (deharmonizer) {
            auto checkpoint=checkpoint_for("deharmony");
            audio = deharmonizer->separate_mono(
                audio.data(), static_cast<int>(audio.size()), options.sample_rate,observer("deharmony"),checkpoint.get());
        }
        if (dereverberator) {
            auto checkpoint=checkpoint_for("dereverb");
            audio = dereverberator->separate_mono(
                audio.data(), static_cast<int>(audio.size()), options.sample_rate,observer("dereverb"),checkpoint.get());
        }
        if (deechoer) {
            auto checkpoint=checkpoint_for("deecho");
            audio = deechoer->separate_mono(
                audio.data(), static_cast<int>(audio.size()), options.sample_rate,observer("deecho"),checkpoint.get());
        }
        if (denoiser) {
            auto checkpoint=checkpoint_for("denoise");
            audio = denoiser->separate_mono(
                audio.data(), static_cast<int>(audio.size()), options.sample_rate,observer("denoise"),checkpoint.get());
        }
        if (options.trim) {
            auto [low, high] = voiced_span(audio, static_cast<float>(options.trim_threshold),
                                           options.sample_rate / 10);
            if (high > low) {
                audio = std::vector<float>(audio.begin() + low, audio.begin() + high);
            }
        }

        const int count = static_cast<int>(audio.size());
        const std::string stem = fs::path(file).stem().string();
        if (options.no_slice) {
            char name[512];
            std::snprintf(name, sizeof(name), "%s/%s.wav",
                          options.output_dir.c_str(), stem.c_str());
            if (!io::write_audio(name, audio.data(), count, options.sample_rate)) {
                std::cerr << "error: write failed: " << name << "\n";
                return 1;
            }
            ++total;
            std::cout << "  " << fs::path(file).filename().string() << " -> processed\n";
            file_done();continue;
        }

        int index = 0;
        std::vector<std::pair<int, int>> regions;
        if (vad) {
            regions = vad->segments(audio.data(), count, options.sample_rate,
                                    static_cast<float>(options.vad_threshold));
            if (regions.empty()) {
                std::cout << "  " << fs::path(file).filename().string()
                          << " -> 0 clips (no speech detected)\n";
                file_done();continue;
            }
        } else {
            regions.emplace_back(0, count);
        }

        for (const auto& region : regions) {
            for (int start = region.first; start < region.second; start += hop) {
                const int length = std::min(segment, region.second - start);
                if (length < minimum_keep && start != region.first) break;
                if (!vad && rms(audio.data() + start, length) < options.minimum_rms) {
                    ++dropped;
                    continue;
                }
                char name[512];
                std::snprintf(name, sizeof(name), "%s/%s_%04d.wav",
                              options.output_dir.c_str(), stem.c_str(), index++);
                if (!io::write_audio(name, audio.data() + start, length,
                                     options.sample_rate)) {
                    std::cerr << "error: write failed: " << name << "\n";
                    return 1;
                }
                ++total;
                if (length < segment) break;
            }
        }
        std::cout << "  " << fs::path(file).filename().string()
                  << " -> " << index << " clips\n";
        file_done();
    }
    if (readable == 0) {
        std::cerr << "error: no input audio could be read from " << options.input << "\n";
        return 1;
    }
    std::cout << "Wrote " << total << " clips to " << options.output_dir
              << " (dropped " << dropped << " silent)\n";
    return 0;
} catch(const io::OperationPaused& e) {
    std::cout << "Paused: " << e.what() << std::endl;return 2;
} catch(const std::exception& e) {
    std::cerr << "Operation failed: " << e.what() << std::endl;return 1;
}

}  // namespace voxmutatio::app_support
