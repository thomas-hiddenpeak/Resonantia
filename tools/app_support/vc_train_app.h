// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#pragma once

#include <iosfwd>
#include <string>

#include "tools/app_support/command.h"
#include "voxmutatio/core/types.h"

namespace voxmutatio::app_support {

struct VcTrainOptions {
    std::string hubert_path;
    std::string rmvpe_path;
    std::string pretrained_path;
    std::string target_path;
    std::string output_path;
    std::string cancel_file; // Optional, one-byte job channel; empty preserves old behavior.
    std::string discriminator_path;
    int speaker = 0;
    int steps = 300;
    int segment_frames = 40;
    int epochs = 0;
    unsigned seed = 0;
    float learning_rate = 2e-4f;
    float learning_rate_decay = 0.999875f;
    bool gan_mode = false;
    std::string gan_profile = "legacy";
    std::string cache_dir;       // joint-full only; immutable frozen frontends.
    std::string checkpoint_dir;  // joint-full only; retained across cancellation.
    std::string resume_path;     // Full state, never an inference model.
    SynthesisMode synthesis_mode = SynthesisMode::kSampled;
};

[[nodiscard]] ParseStatus parse_vc_train_options(
    int argc, char* const argv[], VcTrainOptions& options,
    std::ostream& output, std::ostream& error);

[[nodiscard]] int run_vc_train(VcTrainOptions options);

}  // namespace voxmutatio::app_support
