#pragma once
#include <charconv>
#include <cmath>
#include <string_view>
#include <vector>
#include <utility>

namespace voxmutatio {
enum class PitchEditMode { kSmooth, kShift };
/// Input timeline seconds. Smooth preserves endpoints; shift uses [start,end).
struct PitchEdit {
    PitchEditMode mode = PitchEditMode::kSmooth;
    double start = 0, end = 0, semitones = 0;
};
inline bool valid_pitch_edits(const std::vector<PitchEdit>& edits) noexcept {
    if (edits.size() > 16) return false;
    double previous = -1;
    for (const auto& e : edits) {
        if (!std::isfinite(e.start) || !std::isfinite(e.end) || !std::isfinite(e.semitones) ||
            e.start < 0 || e.end > 600 || e.end - e.start < 0.02 - 1e-9 || e.end - e.start > 2 + 1e-9 ||
            e.start <= previous || std::abs(e.semitones) > 24 ||
            (e.mode != PitchEditMode::kSmooth && e.mode != PitchEditMode::kShift) ||
            (e.mode == PitchEditMode::kSmooth && e.semitones != 0)) return false;
        previous = e.end;
    }
    return true;
}
/// Strict, locale-independent CLI/HTTP text. Empty means no edits. Transactional.
/// Example: smooth:17.36:17.48;shift:23.05:23.11:12
inline bool parse_pitch_edits(std::string_view text, std::vector<PitchEdit>& output) {
    std::vector<PitchEdit> parsed;
    if (text.size() > 2048) return false;
    auto number = [](std::string_view s, double& value) {
        const auto r = std::from_chars(s.data(), s.data()+s.size(), value);
        return r.ec == std::errc{} && r.ptr == s.data()+s.size() && std::isfinite(value);
    };
    while (!text.empty()) {
        auto separator = text.find(';');
        auto row = text.substr(0, separator);
        std::vector<std::string_view> fields;
        for (;;) {
            const auto colon = row.find(':'); fields.push_back(row.substr(0, colon));
            if (colon == std::string_view::npos) break;
            row.remove_prefix(colon+1);
        }
        PitchEdit edit;
        if (fields.size() == 3 && fields[0] == "smooth") edit.mode = PitchEditMode::kSmooth;
        else if (fields.size() == 4 && fields[0] == "shift" && number(fields[3], edit.semitones))
            edit.mode = PitchEditMode::kShift;
        else return false;
        if (!number(fields[1], edit.start) || !number(fields[2], edit.end)) return false;
        parsed.push_back(edit);
        if (parsed.size() > 16) return false;
        if (separator == std::string_view::npos) break;
        text.remove_prefix(separator+1);
        if (text.empty()) return false;
    }
    if (!valid_pitch_edits(parsed)) return false;
    output = std::move(parsed); return true;
}
}  // namespace voxmutatio
