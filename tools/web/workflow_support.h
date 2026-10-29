#pragma once

#include <filesystem>
#include <functional>
#include <string_view>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "tools/web/http_transport.h"
#include "tools/web/task_storage.h"

namespace resonantia::web {

struct Paths {
    std::string repo;
    std::string models;
    std::string build;
    std::string runs;
    std::string webroot;
    // Empty disables persistence for embedding/legacy contract fixtures.
    std::string tasks;
    std::string hubert;
    std::string rmvpe;
    std::string gmodel;
    std::string dmodel;
    std::string ffmpeg = "/usr/bin/ffmpeg";
    // The standalone server enables this; embedded deterministic fixtures can
    // opt in with small limits without depending on their host disk capacity.
    bool enforce_task_storage = false;
    TaskStorageLimits storage_limits;
};

enum class ProcessLogMode {
    kAppend,
    kTruncate,
};

struct ProcessSpec {
    std::vector<std::string> argv;
    // Optional cross-process resource lease. A spawned child inherits the lock,
    // so a crashed runner cannot admit another GPU worker while it is still live.
    // Busy returns kLaunchFailed/EBUSY without spawning; callers may retry later.
    std::filesystem::path resource_lock;
    std::filesystem::path log_path;
    ProcessLogMode log_mode = ProcessLogMode::kAppend;
    std::vector<std::string> log_filter_terms;
    // Borrowed line, valid only during this synchronous call, before log filtering.
    // Failure is reported as io_error after the child has been drained/reaped.
    std::function<void(std::string_view)> observe_line;
};

enum class ProcessTermination {
    kExited,
    kSignaled,
    kLaunchFailed,
    kExecFailed,
    kWaitFailed,
};

struct ProcessResult {
    ProcessTermination termination = ProcessTermination::kLaunchFailed;
    int exit_code = -1;
    int signal_number = 0;
    int system_error = 0;
    int io_error = 0;

    [[nodiscard]] bool success() const noexcept {
        return termination == ProcessTermination::kExited && exit_code == 0 &&
               io_error == 0;
    }
};

class ProcessRunner {
public:
    virtual ~ProcessRunner() = default;
    [[nodiscard]] virtual ProcessResult run(const ProcessSpec& spec) const = 0;
};

class PosixProcessRunner final : public ProcessRunner {
public:
    [[nodiscard]] ProcessResult run(const ProcessSpec& spec) const override;
};

class TempDir {
public:
    TempDir(const std::filesystem::path& parent, const std::string& prefix);
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    TempDir(TempDir&& other) noexcept;
    TempDir& operator=(TempDir&&) = delete;
    ~TempDir();

    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    void release() noexcept { path_.clear(); }

private:
    std::string path_;
};

struct Upload {
    const Part* part = nullptr;
    std::string filename;
};

struct UploadResult {
    int status = 200;
    std::string error;
};

// Only writes to request-owned staging. Borrowed Parts must outlive this call.
// Compressed uploads share one nonblocking decoder slot per WebService instance.
[[nodiscard]] UploadResult normalize_uploads(
    const std::vector<Upload>& uploads, const std::filesystem::path& raw,
    const std::optional<std::filesystem::path>& work, const Paths& paths,
    const ProcessRunner& runner, std::mutex& decode_mutex, bool stereo = false);

[[nodiscard]] std::string lower(std::string value);
[[nodiscard]] bool safe_name(const std::string& value);
[[nodiscard]] std::string safe_filename(const std::string& value);
[[nodiscard]] std::string json_escape(const std::string& value);
[[nodiscard]] bool parse_int_field(const std::string& value, int minimum, int maximum,
                                   int& output);
[[nodiscard]] bool parse_double_field(const std::string& value, double minimum,
                                      double maximum, double& output);
[[nodiscard]] bool parse_bool_field(const std::string& value, bool& output);

[[nodiscard]] std::string read_file(const std::string& path,
                                    std::size_t limit = kMaxFileBytes);
[[nodiscard]] std::string log_tail(const std::string& path, std::size_t count = 600);
[[nodiscard]] bool collect_uploads(int fd, const std::vector<Part>& parts,
                                   std::vector<Upload>& uploads);
[[nodiscard]] bool write_uploads(const std::vector<Upload>& uploads,
                                 const std::filesystem::path& raw,
                                 const std::optional<std::filesystem::path>& work,
                                 std::string& error);
// Publishes raw/work and retires the previous regular steps.txt marker together.
// Recoverable failures restore the old generation; non-regular markers fail
// before live artifacts are moved. Unexpected exceptions retain backup bytes
// until commit or complete rollback. Callers must serialize material mutations.
[[nodiscard]] bool replace_material_dirs(const std::filesystem::path& directory,
                                         const std::filesystem::path& staged_raw,
                                         const std::filesystem::path& staged_work,
                                         std::string& error);
[[nodiscard]] bool replace_step_artifacts(
    const std::filesystem::path& directory,
    const std::filesystem::path& staged_work,
    const std::filesystem::path& staged_steps, std::string& error);
[[nodiscard]] bool replace_training_artifacts(
    const std::filesystem::path& directory,
    const std::filesystem::path& staged_clips,
    const std::filesystem::path& staged_model,
    const std::filesystem::path& staged_index, std::string& error);

void serve_static(int fd, const std::string& webroot, const std::string& request_path);
[[nodiscard]] std::string field(const std::vector<Part>& parts, const std::string& name,
                                const std::string& default_value);
[[nodiscard]] std::string query_get(const std::string& query, const std::string& key);

}  // namespace resonantia::web
