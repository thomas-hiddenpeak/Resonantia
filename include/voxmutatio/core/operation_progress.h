// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
#include <cstdint>
#include <functional>
#include <string_view>

namespace voxmutatio {
// A completed domain boundary, never a timer/estimated percentage. total==0
// means unknown and requires completed==0. Views are valid only in the call.
struct OperationProgress {
    std::string_view phase;
    std::uint64_t completed=0, total=0;
    std::string_view unit;
};
// Synchronous on the operation's caller thread. Must not reenter the instance.
// Exceptions follow the owning operation's ordinary failure convention. No
// observer means no progress work and preserves existing numerical behavior.
using ProgressObserver=std::function<void(const OperationProgress&)>;
inline void report_progress(const ProgressObserver& observer,std::string_view phase,
                            std::uint64_t completed,std::uint64_t total,std::string_view unit) {
    if(observer)observer({phase,completed,total,unit});
}
} // namespace voxmutatio
