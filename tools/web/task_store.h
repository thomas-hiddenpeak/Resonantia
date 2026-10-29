#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <stdexcept>
#include <vector>

namespace resonantia::web {

// Admission limit, not corruption or loss of access to existing records.
class TaskCapacityError final : public std::runtime_error {
public:
    TaskCapacityError() : std::runtime_error("task store capacity reached") {}
};

// Persisted service-owned state. Model checkpoints and audio live separately;
// neither model weights nor uploaded bytes belong in these bounded records.
struct TaskRecord {
    std::string id, kind, name, state = "queued", stage = "queued", error, unit, result, phase;
    std::uint64_t revision = 0;
    std::uint64_t attempt = 1;
    std::uint64_t created_ms = 0, updated_ms = 0, started_ms = 0, finished_ms = 0;
    std::uint64_t progress_at_ms = 0, completed = 0, total = 0;
    std::map<std::string, std::string> parameters;
};

class TaskStore {
public:
    // One process owns a task directory. A second live owner fails immediately.
    // At capacity, creating another task fails without deleting old tasks.
    explicit TaskStore(std::filesystem::path root, std::size_t capacity = 256);
    ~TaskStore();
    TaskStore(const TaskStore&) = delete;
    TaskStore& operator=(const TaskStore&) = delete;

    // expected_revision==0 creates; otherwise compare-and-swap updates.
    // False means a stale revision/existing ID. Invalid records and I/O throw.
    // The caller's revision changes only after a committed update.
    bool save(TaskRecord& record, std::uint64_t expected_revision);
    [[nodiscard]] std::optional<TaskRecord> get(const std::string& id) const;
    [[nodiscard]] std::vector<TaskRecord> list(std::size_t offset = 0,
                                             std::size_t limit = 100) const;
    // Internal storage accounting; bounded by the absolute 4096-record limit.
    [[nodiscard]] std::vector<TaskRecord> snapshot() const;
    // Explicit startup operation: preserve counters/artifacts, retire former
    // active executions, and never automatically restart expensive work.
    std::size_t recover_interrupted(std::uint64_t now_ms);
    [[nodiscard]] std::filesystem::path task_directory(const std::string& id) const;
    [[nodiscard]] static bool valid_id(const std::string& id) noexcept;

private:
    [[nodiscard]] std::optional<TaskRecord> read_locked(const std::string& id) const;
    [[nodiscard]] std::vector<TaskRecord> list_locked() const;
    void write_locked(const TaskRecord& record);
    std::filesystem::path root_;
    std::size_t capacity_;
    int lock_fd_ = -1;
    mutable std::mutex mutex_;
};

} // namespace resonantia::web
