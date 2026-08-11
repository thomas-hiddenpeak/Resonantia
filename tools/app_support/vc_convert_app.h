// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#pragma once

#include <iosfwd>
#include <string>

#include "tools/app_support/command.h"
#include "voxmutatio/core/types.h"

namespace voxmutatio::app_support {

struct VcConvertOptions {
    VCConfig config;
    std::string input_path;
    std::string output_path;
    int speaker_id = 0;
};

[[nodiscard]] ParseStatus parse_vc_convert_options(
    int argc, char* const argv[], VcConvertOptions& options,
    std::ostream& output, std::ostream& error);

[[nodiscard]] int run_vc_convert(const VcConvertOptions& options);

}  // namespace voxmutatio::app_support
