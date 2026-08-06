// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/core/types.h"

namespace voxmutatio {

std::string_view error_code_string(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::kSuccess:
            return "Success";
        case ErrorCode::kInvalidInput:
            return "Invalid Input";
        case ErrorCode::kModelLoadFailed:
            return "Model Load Failed";
        case ErrorCode::kDeviceError:
            return "Device Error";
        case ErrorCode::kInferenceError:
            return "Inference Error";
        case ErrorCode::kIoError:
            return "I/O Error";
        case ErrorCode::kNotFound:
            return "Not Found";
    }
    return "Unknown Error";
}

}  // namespace voxmutatio
