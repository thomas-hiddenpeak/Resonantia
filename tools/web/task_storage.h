#pragma once
#include "tools/web/task_store.h"
#include <cstdint>
#include <filesystem>
#include <map>
#include <vector>

namespace resonantia::web {
struct TaskStorageLimits {
    std::uint64_t managed_bytes=64ULL*1024*1024*1024;
    std::uint64_t task_bytes=24ULL*1024*1024*1024;
    std::uint64_t reserve_bytes=8ULL*1024*1024*1024;
};
struct TaskStorageUsage {
    std::uint64_t managed_bytes=0, available_bytes=0;
    std::map<std::string,std::uint64_t> tasks;
};
// Bounded, no-follow inspection of the task store and explicitly owned full
// training directories. Published/unmanaged assets are never traversed.
TaskStorageUsage measure_task_storage(const std::filesystem::path& root,
    const std::filesystem::path& runs,const std::vector<TaskRecord>& records);
std::string storage_limit_reason(const TaskStorageUsage&,const TaskStorageLimits&,const std::string& id={});
// Removes only named redundant regular files of a successful task. Inputs,
// result, logs, full checkpoints, candidates and unknown files are retained.
// Repeating after a partial cleanup is safe. The caller commits retention state.
std::uint64_t prune_task_copies(const std::filesystem::path& root,
    const std::filesystem::path& runs,const TaskRecord& record);
} // namespace resonantia::web
