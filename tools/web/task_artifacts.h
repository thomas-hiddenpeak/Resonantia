#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <vector>
namespace resonantia::web {
// Used only while a task execution owns its directory. A completed stage is a
// checksummed artifact, never merely an existing file. No deletion of work.
class TaskArtifacts {
public:
    TaskArtifacts(std::filesystem::path root, std::string identity,
                  const std::vector<std::string>& compatible = {});
    const std::string& identity() const noexcept { return identity_; }
    std::optional<std::filesystem::path> completed(const std::string& stage) const;
    // Read-only access for completed-task playback. Does not establish that the
    // current recipe can reuse this stage. Listing may omit the content hash;
    // playback must verify it. No markers are created or rewritten.
    static std::optional<std::filesystem::path> retained_output(
        const std::filesystem::path& root, const std::string& stage, bool verify_checksum = true);
    void commit(const std::string& stage, const std::filesystem::path& output);
    void publish_result(const std::filesystem::path& input);
private:
    std::filesystem::path root_;
    std::string identity_;
};
} // namespace resonantia::web
