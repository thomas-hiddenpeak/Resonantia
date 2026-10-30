#pragma once
#include "voxmutatio/core/types.h"
#include "voxmutatio/f0/pitch_review.h"
#include <filesystem>
#include <functional>
namespace resonantia::web::pitch_review {
struct Evidence {
    std::filesystem::path input;
    std::string identity;
    std::uint64_t samples16k=0;
    double duration=0,pad=0;
    std::vector<float> raw;
};
Evidence read_evidence(const std::filesystem::path& work);
std::string operation_identity(const voxmutatio::VCConfig& config,const std::filesystem::path& input,
                               const std::filesystem::path& program);
// Explicitly rebind only dependency-equivalent blocks into a new private store.
// Full raw/synthesis coverage and every payload are verified; old work is read-only.
// Returns the count of synthesis cores that must be recomputed (max two).
std::size_t seed_preview(const std::filesystem::path& parent_work,const std::filesystem::path& destination,
    const std::filesystem::path& input,const std::filesystem::path& program,
    const voxmutatio::VCConfig& previous,const voxmutatio::VCConfig& next,
    double start,double end,const std::string& pause_file,const std::function<void()>& boundary);
void write_excerpt(const std::filesystem::path& input,const std::filesystem::path& output,double start,double end);
}
