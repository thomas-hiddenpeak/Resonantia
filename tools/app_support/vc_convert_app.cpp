// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "tools/app_support/vc_convert_app.h"

#include <cstring>
#include <iostream>
#include <limits>
#include <ostream>
#include <string>
#include <string_view>

#include "tools/app_support/cli_parse.h"
#include "tools/app_support/file_conversion_policy.h"
#include "voxmutatio/core/device.h"
#include "voxmutatio/pipeline/pipeline.h"

namespace voxmutatio::app_support {
namespace {

void print_usage(std::ostream& output) {
    output
        << "Usage:\n"
        << "  vc_convert --model <path> --input <wav> --output <wav> [options]\n\n"
        << "Options:\n"
        << "  --hubert <path>       HuBERT safetensors (required when content-encoder=hubert)\n"
        << "  --model <path>        VITS synthesizer safetensors path\n"
        << "  --input <wav>         Input audio file path\n"
        << "  --output <wav>        Output audio file path\n"
        << "  --index <path>        Flat Index (required when index-rate > 0)\n"
        << "  --rmvpe <path>        RMVPE safetensors (required when f0-method=rmvpe)\n"
        << "  --fcpe <path>         FCPE safetensors (required when f0-method=fcpe)\n"
        << "  --f0-method <name>    F0 extractor: rmvpe (default) or fcpe\n"
        << "  --wavlm <path>        WavLM safetensors (required when content-encoder=wavlm)\n"
        << "  --content-encoder <n> Content encoder: hubert (default) or wavlm\n"
        << "  --speaker <id>        Speaker ID (default: 0)\n"
        << "  --pitch <semitones>   Pitch shift in semitones (default: 0)\n"
        << "  --index-rate <0-1>    Index blend rate (default: 0.0)\n"
        << "  --rms-mix <0-1>       RMS energy mix rate (default: 0.5)\n"
        << "  --protect <0-1>       Unvoiced protection (default: 0.5)\n"
        << "  --filter-radius <n>   F0 median filter: 0 or odd 3..31 (default: 0)\n"
        << "  --version <v1|v2>     Checkpoint architecture (default: v1)\n"
        << "  --speakers <n>        Legacy hint; checkpoint speaker count is authoritative\n"
        << "  --sr <hz>             Model rate; only 40 kHz is currently supported\n"
        << "  --device <cuda>       CUDA FP32 is the supported runtime\n"
        << "  --help                Show this help\n";
}

}  // namespace

ParseStatus parse_vc_convert_options(int argc, char* const argv[],
                                     VcConvertOptions& options,
                                     std::ostream& output,
                                     std::ostream& error) {
    if (argc < 2) {
        print_usage(output);
        return ParseStatus::kExitFailure;
    }

    // Keep the historical order-independent help behavior for this command.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(output);
            return ParseStatus::kExitSuccess;
        }
    }

    auto next = [&](int& index, std::string_view option) {
        return cli::option_value(argc, argv, index, option, error);
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--hubert") {
            const char* value = next(i, arg); if (!value) return ParseStatus::kExitFailure;
            options.config.hubert_model_path = value;
        } else if (arg == "--model") {
            const char* value = next(i, arg); if (!value) return ParseStatus::kExitFailure;
            options.config.synthesizer_model_path = value;
        } else if (arg == "--input") {
            const char* value = next(i, arg); if (!value) return ParseStatus::kExitFailure;
            options.input_path = value;
        } else if (arg == "--output") {
            const char* value = next(i, arg); if (!value) return ParseStatus::kExitFailure;
            options.output_path = value;
        } else if (arg == "--index") {
            const char* value = next(i, arg); if (!value) return ParseStatus::kExitFailure;
            options.config.index_path = value;
        } else if (arg == "--rmvpe") {
            const char* value = next(i, arg); if (!value) return ParseStatus::kExitFailure;
            options.config.rmvpe_model_path = value;
        } else if (arg == "--fcpe") {
            const char* value = next(i, arg); if (!value) return ParseStatus::kExitFailure;
            options.config.fcpe_model_path = value;
        } else if (arg == "--f0-method") {
            const char* value = next(i, arg); if (!value) return ParseStatus::kExitFailure;
            const std::string method = value;
            if (method == "fcpe") options.config.f0_method = F0Method::kFcpe;
            else if (method == "rmvpe") options.config.f0_method = F0Method::kRmvpe;
            else {
                error << "error: --f0-method must be rmvpe or fcpe\n";
                return ParseStatus::kExitFailure;
            }
        } else if (arg == "--wavlm") {
            const char* value = next(i, arg); if (!value) return ParseStatus::kExitFailure;
            options.config.wavlm_model_path = value;
        } else if (arg == "--content-encoder") {
            const char* value = next(i, arg); if (!value) return ParseStatus::kExitFailure;
            const std::string encoder = value;
            if (encoder == "wavlm") options.config.content_encoder = ContentEncoder::kWavlm;
            else if (encoder == "hubert") options.config.content_encoder = ContentEncoder::kHubert;
            else {
                error << "error: --content-encoder must be hubert or wavlm\n";
                return ParseStatus::kExitFailure;
            }
        } else if (arg == "--speaker") {
            if (!cli::parse_integer_option(argc, argv, i, arg, 0,
                                           std::numeric_limits<int>::max(),
                                           options.speaker_id, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (arg == "--pitch") {
            if (!cli::parse_integer_option(argc, argv, i, arg, -48, 48,
                                           options.config.f0_up_key, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (arg == "--index-rate") {
            if (!cli::parse_real_option(argc, argv, i, arg, 0.0, 1.0,
                                        options.config.index_rate, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (arg == "--rms-mix") {
            if (!cli::parse_real_option(argc, argv, i, arg, 0.0, 1.0,
                                        options.config.rms_mix_rate, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (arg == "--protect") {
            if (!cli::parse_real_option(argc, argv, i, arg, 0.0, 1.0,
                                        options.config.protect, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (arg == "--filter-radius") {
            if (!cli::parse_integer_option(argc, argv, i, arg, 0, 31,
                                           options.config.filter_radius, error)) {
                return ParseStatus::kExitFailure;
            }
            if (options.config.filter_radius != 0 &&
                (options.config.filter_radius < 3 || options.config.filter_radius % 2 == 0)) {
                error << "error: invalid value for --filter-radius: must be 0 or an odd "
                         "integer in 3..31\n";
                return ParseStatus::kExitFailure;
            }
        } else if (arg == "--half") {
            error << "error: FP16 inference is not implemented; use CUDA FP32\n";
            return ParseStatus::kExitFailure;
        } else if (arg == "--device") {
            const char* value = next(i, arg); if (!value) return ParseStatus::kExitFailure;
            options.config.device = value;
            if (options.config.device != "cuda") {
                error << "error: only the CUDA backend is implemented\n";
                return ParseStatus::kExitFailure;
            }
        } else if (arg == "--version") {
            const char* value = next(i, arg); if (!value) return ParseStatus::kExitFailure;
            const std::string version = value;
            if (version == "v2") options.config.version = ModelVersion::kV2;
            else if (version == "v1") options.config.version = ModelVersion::kV1;
            else {
                error << "error: --version must be v1 or v2\n";
                return ParseStatus::kExitFailure;
            }
        } else if (arg == "--speakers") {
            if (!cli::parse_integer_option(argc, argv, i, arg, 1,
                                           std::numeric_limits<int>::max(),
                                           options.config.num_speakers, error)) {
                return ParseStatus::kExitFailure;
            }
        } else if (arg == "--sr") {
            if (!cli::parse_integer_option(argc, argv, i, arg, 8'000, 192'000,
                                           options.config.model_sample_rate, error)) {
                return ParseStatus::kExitFailure;
            }
            if (options.config.model_sample_rate != 40'000) {
                error << "error: --sr must be 40000; other synthesizer rates are unsupported\n";
                return ParseStatus::kExitFailure;
            }
        } else {
            error << "Unknown argument: " << arg << '\n';
            return ParseStatus::kExitFailure;
        }
    }

    if (options.config.synthesizer_model_path.empty()) {
        error << "error: --model is required\n";
        return ParseStatus::kExitFailure;
    }
    if (options.input_path.empty()) {
        error << "error: --input is required\n";
        return ParseStatus::kExitFailure;
    }
    if (options.output_path.empty()) {
        error << "error: --output is required\n";
        return ParseStatus::kExitFailure;
    }
    if (options.config.content_encoder == ContentEncoder::kHubert &&
        options.config.hubert_model_path.empty()) {
        error << "error: --hubert is required when --content-encoder=hubert\n";
        return ParseStatus::kExitFailure;
    }
    if (options.config.content_encoder == ContentEncoder::kWavlm &&
        options.config.wavlm_model_path.empty()) {
        error << "error: --wavlm is required when --content-encoder=wavlm\n";
        return ParseStatus::kExitFailure;
    }
    if (options.config.f0_method == F0Method::kRmvpe &&
        options.config.rmvpe_model_path.empty()) {
        error << "error: --rmvpe is required when --f0-method=rmvpe\n";
        return ParseStatus::kExitFailure;
    }
    if (options.config.f0_method == F0Method::kFcpe &&
        options.config.fcpe_model_path.empty()) {
        error << "error: --fcpe is required when --f0-method=fcpe\n";
        return ParseStatus::kExitFailure;
    }
    if (options.config.index_rate > 0.0 && options.config.index_path.empty()) {
        error << "error: --index is required when --index-rate > 0\n";
        return ParseStatus::kExitFailure;
    }
    return ParseStatus::kRun;
}

int run_vc_convert(const VcConvertOptions& options) {
    Device device;
    if (auto err = device.init(options.config.device, options.config.gpu_device)) {
        std::cerr << "CUDA init error: " << *err << "\n";
        return 1;
    }
    std::cout << "Device: " << device.name() << "\n";

    pipeline::VoiceConversionPipeline pipeline;
    if (!pipeline.init(options.config)) {
        std::cerr << "Pipeline init failed\n";
        return 1;
    }

    std::cout << "Converting: " << options.input_path << " -> "
              << options.output_path << "\n";
    const bool use_bounded_file_path =
        file_conversion_path_for(options.config.content_encoder) ==
        FileConversionPath::kBounded;
    std::cout << "File path: "
              << (use_bounded_file_path ? "bounded HuBERT" : "legacy WavLM")
              << "\n";
    const auto result = convert_file_with_policy(
        pipeline, options.config, options.input_path, options.output_path,
        options.speaker_id);
    if (!result.success) {
        std::cerr << "Conversion failed: " << result.error_message << "\n";
        return 1;
    }

    std::cout << "Done (" << result.total_ms << " ms)\n"
              << "  HuBERT:  " << result.hubert_ms << " ms\n"
              << "  F0:      " << result.f0_ms << " ms\n"
              << "  Index:   " << result.index_ms << " ms\n"
              << "  Synth:   " << result.synth_ms << " ms\n";
    return 0;
}

}  // namespace voxmutatio::app_support
