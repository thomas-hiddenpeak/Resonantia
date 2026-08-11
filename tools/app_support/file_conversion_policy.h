// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "voxmutatio/core/types.h"

namespace voxmutatio::app_support {

enum class FileConversionPath {
    kBounded,
    kLegacy,
};

/// HuBERT-backed v1/v2 profiles have completed the bounded-file numerical and
/// resource gates. WavLM keeps the legacy file path until a matching
/// WavLM-trained synthesizer asset can qualify the bounded combination.
[[nodiscard]] constexpr FileConversionPath file_conversion_path_for(
    ContentEncoder encoder) noexcept {
    return encoder == ContentEncoder::kHubert ? FileConversionPath::kBounded
                                               : FileConversionPath::kLegacy;
}

template <typename Pipeline>
[[nodiscard]] auto convert_file_with_policy(
    Pipeline& pipeline, const VCConfig& config, const std::string& input_path,
    const std::string& output_path, int speaker_id) {
    if (file_conversion_path_for(config.content_encoder) ==
        FileConversionPath::kBounded) {
        return pipeline.convert_file_bounded(input_path, output_path, speaker_id);
    }
    return pipeline.convert_file(input_path, output_path, speaker_id);
}

}  // namespace voxmutatio::app_support
