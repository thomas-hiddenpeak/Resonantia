// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#pragma once

namespace voxmutatio::app_support {

// Parsing can either yield a runnable options object or request an immediate
// process exit (successful for --help, unsuccessful for invalid arguments).
enum class ParseStatus {
    kRun,
    kExitSuccess,
    kExitFailure,
};

[[nodiscard]] constexpr int parse_exit_code(ParseStatus status) noexcept {
    return status == ParseStatus::kExitSuccess ? 0 : 1;
}

}  // namespace voxmutatio::app_support
