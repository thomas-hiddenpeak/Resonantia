// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#pragma once

#include <iosfwd>
#include <string>

#include "tools/app_support/command.h"

namespace voxmutatio::app_support {

struct VcPreprocessOptions {
    std::string input;
    std::string output_dir;
    int sample_rate = 40'000;
    double segment_seconds = 3.0;
    double hop_seconds = 0.0;
    double minimum_rms = 0.010;
    double trim_threshold = 0.020;
    bool trim = false;
    bool separate = false;
    bool deharmony = false;
    bool dereverb = false;
    bool deecho = false;
    bool denoise = false;
    bool no_slice = false;
    bool vad = false;
    double vad_threshold = 0.5;
    std::string separation_dir = "models/separation";
    std::string vad_dir = "models/vad";
    std::string separate_model;
    std::string deharmony_model;
    std::string dereverb_model;
    std::string deecho_model;
    std::string denoise_model;
};

[[nodiscard]] ParseStatus parse_vc_preprocess_options(
    int argc, char* const argv[], VcPreprocessOptions& options,
    std::ostream& output, std::ostream& error);

[[nodiscard]] int run_vc_preprocess(const VcPreprocessOptions& options);

}  // namespace voxmutatio::app_support
