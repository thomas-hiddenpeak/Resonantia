// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#pragma once

#include <iosfwd>
#include <string>

#include "tools/app_support/command.h"
#include "voxmutatio/core/types.h"

namespace voxmutatio::app_support {

struct VcBatchOptions {
    VCConfig config;
    std::string input_dir;
    std::string output_dir;
    int speaker_id = 0;
    bool recursive = false;
};

[[nodiscard]] ParseStatus parse_vc_batch_options(
    int argc, char* const argv[], VcBatchOptions& options,
    std::ostream& output, std::ostream& error);

[[nodiscard]] int run_vc_batch(const VcBatchOptions& options);

}  // namespace voxmutatio::app_support
