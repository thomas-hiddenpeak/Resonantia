#pragma once
#include "voxmutatio/core/pitch_edit.h"
#include <optional>
#include <cstdint>
#include <span>
#include <string>
namespace voxmutatio::f0 {
// Host-only, <=600s/61100 frames. Matches the Web bounded profile: 100Hz,
// no unvoiced interpolation. Invalid/UV-crossing edits throw; never partly apply.
std::vector<float> prepare_pitch_curve(std::span<const float> raw,int filter_radius,
    int shift,const std::vector<PitchEdit>& edits,double pad_seconds,double duration);
struct PitchIssue {
    double start=0,end=0;
    std::string reason;
    std::optional<PitchEdit> suggestion;
};
// Heuristic review candidates, not musical ground truth or calibrated confidence.
// At most 64 issues. UV, vibrato and continuous slides are not interpolated.
std::vector<PitchIssue> find_pitch_issues(std::span<const float> raw,std::span<const float> processed,
    int filter_radius,double pad_seconds,double duration);
}
