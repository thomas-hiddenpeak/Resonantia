// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace voxmutatio::io {
class OperationPaused final : public std::runtime_error {
public:
    OperationPaused() : std::runtime_error("operation paused at a durable boundary") {}
};

// Immutable, bounded float blocks. One owner per directory (process-wide flock),
// synchronous, not thread safe; no GPU or Web dependencies. The caller must bind
// identity to input/model bytes, numerical parameters and implementation before
// starting an operation. Parent directories must be trusted. Missing blocks are
// cache misses; corrupt or mismatched blocks throw and must never be recomputed
// silently. Destruction releases ownership, never removes committed work.
// Resume may finish an interrupted identity publication only when the existing
// directory contains no blocks; a missing identity beside blocks always fails.
class BlockCheckpoint {
public:
    enum class Mode { Create, Resume, ReadOnly };
    struct Entry { std::string key; std::uint64_t offset=0, count=0; };
    // Historical inspection only: this does not prove compatibility with a new
    // recipe. ReadOnly requires existing identity/lock, takes a shared lock,
    // never cleans pending files and rejects every save, including idempotent ones.
    static std::string stored_identity(const std::filesystem::path& directory);
    struct Limits {
        std::uint64_t block_bytes = 32ULL * 1024 * 1024;
        std::uint64_t total_bytes = 2ULL * 1024 * 1024 * 1024;
        std::uint32_t blocks = 4096;
    };
    BlockCheckpoint(const std::filesystem::path& directory, std::string identity,
                    Mode mode, std::function<bool()> pause_requested = {});
    BlockCheckpoint(const std::filesystem::path& directory, std::string identity,
                    Mode mode, std::function<bool()> pause_requested, Limits limits);
    ~BlockCheckpoint();
    BlockCheckpoint(const BlockCheckpoint&) = delete;
    BlockCheckpoint& operator=(const BlockCheckpoint&) = delete;
    // key: 1..64 ASCII lowercase letters/digits/_/-. offset and count are in
    // the owning algorithm's documented units, not a relative resume position.
    std::optional<std::vector<float>> load(const std::string& key,
                                         std::uint64_t offset, std::uint64_t count);
    // Atomic durable publication. Existing identical content is idempotent;
    // changing an existing block, non-finite values or exceeding limits fails.
    void save(const std::string& key, std::uint64_t offset, std::span<const float> data);
    // Call only at a safe boundary, after completed work is committed. A false
    // callback is a no-op; true throws OperationPaused. Callback errors propagate.
    void boundary() const;
    std::uint64_t restored_blocks() const noexcept;
    // Bounded metadata listing; load() must still verify each payload checksum.
    std::vector<Entry> entries() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace voxmutatio::io
