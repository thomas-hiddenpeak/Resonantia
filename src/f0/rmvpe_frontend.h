// Copyright 2024 Resonantia Authors. Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace voxmutatio::f0::detail {

// Private fixed RMVPE frontend: librosa.filters.mel(htk=True, norm="slaney").
// HTK frequency spacing and Slaney area normalization are independent choices.
inline std::vector<float> rmvpe_mel_basis() {
    constexpr int mels = 128;
    constexpr int bins = 513;
    const double mel_min = 2595.0 * std::log10(1.0 + 30.0 / 700.0);
    const double mel_max = 2595.0 * std::log10(1.0 + 8000.0 / 700.0);
    double edges[mels + 2];
    for (int i = 0; i < mels + 2; ++i) {
        const double mel = mel_min + (mel_max - mel_min) * i / (mels + 1);
        edges[i] = 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    }
    std::vector<float> weights(mels * bins);
    for (int m = 0; m < mels; ++m) {
        const double lower_width = edges[m + 1] - edges[m];
        const double upper_width = edges[m + 2] - edges[m + 1];
        const double area = 2.0 / (edges[m + 2] - edges[m]);
        for (int f = 0; f < bins; ++f) {
            const double hz = 16000.0 * f / 1024.0;
            const double lower = (hz - edges[m]) / lower_width;
            const double upper = (edges[m + 2] - hz) / upper_width;
            // librosa stores the triangle in float32 before area normalization.
            const float triangle = static_cast<float>(std::max(0.0, std::min(lower, upper)));
            weights[m * bins + f] = static_cast<float>(triangle * area);
        }
    }
    return weights;
}

}  // namespace voxmutatio::f0::detail
