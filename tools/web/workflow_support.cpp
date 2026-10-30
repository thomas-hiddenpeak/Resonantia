#include "tools/web/workflow_support.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/wait.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unordered_set>
#include <unistd.h>
#include <utility>

extern char** environ;

namespace resonantia::web {

namespace fs = std::filesystem;

#ifdef RESONANTIA_TEST_MATERIAL_COMMIT_GATE
// Defined only by the private commit test executable, never a service target.
void test_material_raw_installed(const fs::path& directory);
#endif

#ifdef RESONANTIA_TEST_TRAINING_PUBLISH_GATE
// Private host test only: pause after the first of the three artifacts installs.
void test_training_clips_installed(const fs::path& directory);
#endif

namespace {

constexpr std::size_t kMaximumProcessLogLine = 64 * 1024;

// After live artifacts move, cleanup is unsafe until commit or full rollback.
// This guard must remain allocation-free even when diagnostic formatting throws.
class BackupRetentionGuard {
public:
    BackupRetentionGuard(TempDir& backup, const char* operation) noexcept
        : backup_(backup), operation_(operation) {}
    BackupRetentionGuard(const BackupRetentionGuard&) = delete;
    BackupRetentionGuard& operator=(const BackupRetentionGuard&) = delete;
    ~BackupRetentionGuard() noexcept {
        if (cleanup_allowed_) return;
        std::fprintf(stderr, "[vc_serve] %s transaction incomplete; backup retained at %s\n",
                     operation_, backup_.path().c_str());
        backup_.release();
    }
    void allow_cleanup() noexcept { cleanup_allowed_ = true; }

private:
    TempDir& backup_;
    const char* operation_;
    bool cleanup_allowed_ = false;
};

class UniqueFd {
public:
    explicit UniqueFd(int descriptor = -1) noexcept : descriptor_(descriptor) {}
    UniqueFd(const UniqueFd&) = delete;
    UniqueFd& operator=(const UniqueFd&) = delete;
    UniqueFd(UniqueFd&& other) noexcept : descriptor_(other.release()) {}
    UniqueFd& operator=(UniqueFd&&) = delete;
    ~UniqueFd() {
        if (descriptor_ >= 0) ::close(descriptor_);
    }

    [[nodiscard]] int get() const noexcept { return descriptor_; }
    [[nodiscard]] int release() noexcept {
        const int descriptor = descriptor_;
        descriptor_ = -1;
        return descriptor;
    }
    void reset(int descriptor = -1) noexcept {
        if (descriptor_ >= 0) ::close(descriptor_);
        descriptor_ = descriptor;
    }

private:
    int descriptor_ = -1;
};

class SpawnActions {
public:
    SpawnActions() noexcept : error_(posix_spawn_file_actions_init(&actions_)) {}
    SpawnActions(const SpawnActions&) = delete;
    SpawnActions& operator=(const SpawnActions&) = delete;
    ~SpawnActions() {
        if (error_ == 0) posix_spawn_file_actions_destroy(&actions_);
    }

    [[nodiscard]] int error() const noexcept { return error_; }
    [[nodiscard]] posix_spawn_file_actions_t* get() noexcept { return &actions_; }

private:
    posix_spawn_file_actions_t actions_{};
    int error_ = 0;
};

class ChildProcessGuard {
public:
    explicit ChildProcessGuard(pid_t process_id) noexcept : process_id_(process_id) {}
    ChildProcessGuard(const ChildProcessGuard&) = delete;
    ChildProcessGuard& operator=(const ChildProcessGuard&) = delete;
    ~ChildProcessGuard() {
        if (process_id_ <= 0) return;
        ::kill(process_id_, SIGKILL);
        int status = 0;
        while (::waitpid(process_id_, &status, 0) < 0 && errno == EINTR) {
        }
    }

    void release() noexcept { process_id_ = -1; }

private:
    pid_t process_id_ = -1;
};

bool move_above_standard_descriptors(UniqueFd& descriptor) noexcept {
    if (descriptor.get() > STDERR_FILENO) return true;
    const int duplicate = ::fcntl(descriptor.get(), F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
    if (duplicate < 0) return false;
    descriptor.reset(duplicate);
    return true;
}

void record_io_error(int& destination) noexcept {
    if (destination == 0) destination = errno == 0 ? EIO : errno;
}

bool write_all(int fd, std::string_view data, int& io_error) noexcept {
    std::size_t offset = 0;
    while (offset < data.size()) {
        const ssize_t written =
            ::write(fd, data.data() + offset, data.size() - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            record_io_error(io_error);
            return false;
        }
        offset += static_cast<std::size_t>(written);
    }
    return true;
}

bool matches_filter(const std::string& line,
                    const std::vector<std::string>& terms) {
    if (terms.empty()) return true;
    return std::any_of(terms.begin(), terms.end(), [&](const std::string& term) {
        return !term.empty() && line.find(term) != std::string::npos;
    });
}

void emit_log_line(int log_fd, const std::string& line, bool had_newline,
                   const std::vector<std::string>& terms, int& io_error) noexcept {
    if (io_error != 0 || !matches_filter(line, terms)) return;
    if (!write_all(log_fd, line, io_error)) return;
    if (had_newline) write_all(log_fd, "\n", io_error);
}

ProcessResult launch_failure(ProcessTermination termination, int error) noexcept {
    ProcessResult result;
    result.termination = termination;
    result.system_error = error == 0 ? EIO : error;
    return result;
}

}  // namespace

ProcessResult PosixProcessRunner::run(const ProcessSpec& spec) const {
    if (spec.argv.empty() || spec.argv.front().empty() || spec.log_path.empty()) {
        return launch_failure(ProcessTermination::kLaunchFailed, EINVAL);
    }
    for (const std::string& argument : spec.argv) {
        if (argument.find('\0') != std::string::npos) {
            return launch_failure(ProcessTermination::kLaunchFailed, EINVAL);
        }
    }

    std::vector<char*> argv;
    argv.reserve(spec.argv.size() + 1);
    for (const std::string& argument : spec.argv) {
        argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);
    std::string line;
    line.reserve(kMaximumProcessLogLine);

    const int log_flags = O_WRONLY | O_CREAT | O_CLOEXEC |
                          (spec.log_mode == ProcessLogMode::kAppend ? O_APPEND
                                                                    : O_TRUNC);
    UniqueFd log_fd(::open(spec.log_path.c_str(), log_flags, 0600));
    if (log_fd.get() < 0) {
        return launch_failure(ProcessTermination::kLaunchFailed, errno);
    }

    UniqueFd resource;
    if(!spec.resource_lock.empty()) {
        const auto rejected=[&](int error) {
            auto failure=launch_failure(ProcessTermination::kLaunchFailed,error);
            write_all(log_fd.get(),error==EBUSY?
                "[vc_serve] another GPU worker still owns the resource, possibly from a previous service; retry after it exits\n":
                "[vc_serve] cannot acquire the GPU process resource lease\n",failure.io_error);
            return failure;
        };
        if(spec.resource_lock.native().find('\0')!=std::string::npos)return rejected(EINVAL);
        resource.reset(::open(spec.resource_lock.c_str(),O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK,0600));
        if(resource.get()<0)return rejected(errno);
        struct stat st{};
        if(::fstat(resource.get(),&st))return rejected(errno);
        if(!S_ISREG(st.st_mode)||st.st_nlink!=1)return rejected(EINVAL);
        if(::flock(resource.get(),LOCK_EX|LOCK_NB))return rejected(errno==EWOULDBLOCK||errno==EAGAIN?EBUSY:errno);
        if(!move_above_standard_descriptors(resource))return rejected(errno);
    }

    int output_pipe[2] = {-1, -1};
    if (::pipe2(output_pipe, O_CLOEXEC) != 0) {
        return launch_failure(ProcessTermination::kLaunchFailed, errno);
    }
    UniqueFd output_read(output_pipe[0]);
    UniqueFd output_write(output_pipe[1]);
    if (!move_above_standard_descriptors(output_read) ||
        !move_above_standard_descriptors(output_write)) {
        return launch_failure(ProcessTermination::kLaunchFailed, errno);
    }

    SpawnActions actions;
    int action_error = actions.error();
    if (action_error != 0) {
        return launch_failure(ProcessTermination::kLaunchFailed, action_error);
    }
    action_error =
        posix_spawn_file_actions_adddup2(actions.get(), output_write.get(), STDOUT_FILENO);
    if (action_error == 0) {
        action_error =
            posix_spawn_file_actions_adddup2(actions.get(), output_write.get(), STDERR_FILENO);
    }
    if (action_error == 0) {
        action_error = posix_spawn_file_actions_addclose(actions.get(), output_read.get());
    }
    if (action_error == 0) {
        action_error = posix_spawn_file_actions_addclose(actions.get(), output_write.get());
    }
    if(action_error==0&&resource.get()>=0) {
        // posix_spawn dup2 actions clear FD_CLOEXEC even for identical source
        // and destination descriptors. The child retains this open description
        // and its flock if the server dies before wait/reap. It owns no stdin.
        action_error=posix_spawn_file_actions_adddup2(actions.get(),resource.get(),resource.get());
    }
    if (action_error != 0) {
        return launch_failure(ProcessTermination::kLaunchFailed, action_error);
    }

    pid_t child = -1;
    const int spawn_error =
        posix_spawn(&child, spec.argv.front().c_str(), actions.get(), nullptr,
                    argv.data(), environ);
    output_write.reset();

    if (spawn_error != 0) {
        int io_error = 0;
        const std::string message = "[vc_serve] process launch failed: " +
                                    std::string(std::strerror(spawn_error)) + "\n";
        write_all(log_fd.get(), message, io_error);
        ProcessResult result = launch_failure(ProcessTermination::kExecFailed,
                                              spawn_error);
        result.io_error = io_error;
        return result;
    }
    ChildProcessGuard child_guard(child);

    ProcessResult result;
    result.termination = ProcessTermination::kExited;
    bool observer_failed=false;
    const auto observe=[&] {
        if(!spec.observe_line||observer_failed)return;
        try {spec.observe_line(line);}
        catch(...) {observer_failed=true;}
    };
    std::array<char, 8192> buffer{};
    for (;;) {
        const ssize_t count = ::read(output_read.get(), buffer.data(), buffer.size());
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) {
            record_io_error(result.io_error);
            break;
        }
        if (count == 0) break;
        for (ssize_t index = 0; index < count; ++index) {
            const char character = buffer[static_cast<std::size_t>(index)];
            if (character == '\n') {
                observe();
                emit_log_line(log_fd.get(), line, true, spec.log_filter_terms,
                              result.io_error);
                line.clear();
            } else {
                line.push_back(character);
                if (line.size() == kMaximumProcessLogLine) {
                    observe();
                    emit_log_line(log_fd.get(), line, true, spec.log_filter_terms,
                                  result.io_error);
                    line.clear();
                }
            }
        }
    }
    if (!line.empty()) {
        observe();
        emit_log_line(log_fd.get(), line, false, spec.log_filter_terms, result.io_error);
    }
    output_read.reset();

    int status = 0;
    pid_t waited;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0) {
        result.termination = ProcessTermination::kWaitFailed;
        result.system_error = errno;
    } else if (WIFEXITED(status)) {
        child_guard.release();
        result.termination = ProcessTermination::kExited;
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        child_guard.release();
        result.termination = ProcessTermination::kSignaled;
        result.signal_number = WTERMSIG(status);
    } else {
        child_guard.release();
        result.termination = ProcessTermination::kWaitFailed;
        result.system_error = ECHILD;
    }
    if (::close(log_fd.release()) != 0) record_io_error(result.io_error);
    if(observer_failed&&!result.io_error)result.io_error=EIO;
    return result;
}

TempDir::TempDir(const fs::path& parent, const std::string& prefix) {
    std::error_code error;
    fs::create_directories(parent, error);
    if (error) {
        throw std::runtime_error("cannot create temporary parent: " + error.message());
    }
    const std::string pattern = (parent / (prefix + "XXXXXX")).string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* made = mkdtemp(writable.data());
    if (!made) {
        throw std::runtime_error("mkdtemp failed: " + std::string(std::strerror(errno)));
    }
    path_ = made;
}

TempDir::TempDir(TempDir&& other) noexcept : path_(std::move(other.path_)) {
    other.path_.clear();
}

TempDir::~TempDir() {
    if (path_.empty()) return;
    std::error_code ignored;
    fs::remove_all(path_, ignored);
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool safe_name(const std::string& value) {
    if (value.empty() || value.size() > 64) return false;
    for (unsigned char c : value) {
        if (!std::isalnum(c) && c != '_' && c != '-') return false;
    }
    return true;
}

std::string safe_filename(const std::string& value) {
    if (value.find('/') != std::string::npos || value.find('\\') != std::string::npos ||
        value.find('\0') != std::string::npos) {
        return "";
    }
    const std::string base = fs::path(value).filename().string();
    if (base.empty() || base == "." || base == ".." || base.size() > 255) return "";
    std::string output;
    output.reserve(base.size());
    for (std::size_t i = 0; i < base.size();) {
        const auto c = static_cast<unsigned char>(base[i]);
        if (c < 0x80) {
            if (std::isalnum(c) || c == '_' || c == '-' || c == '.') {
                output += static_cast<char>(c);
            }
            ++i;
            continue;
        }
        // Preserve complete UTF-8 characters: stripping them turns Chinese song
        // names into ".mp3" (no extension) and collapses distinct batch entries.
        const std::size_t width = c >= 0xc2 && c <= 0xdf ? 2 :
                                  c >= 0xe0 && c <= 0xef ? 3 :
                                  c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        if (width == 0 || width > base.size() - i) return "";
        unsigned codepoint = c & (0x7f >> width);
        for (std::size_t j = 1; j < width; ++j) {
            const auto next = static_cast<unsigned char>(base[i + j]);
            if ((next & 0xc0) != 0x80) return "";
            codepoint = (codepoint << 6) | (next & 0x3f);
        }
        if ((width == 3 && codepoint < 0x800) ||
            (width == 4 && codepoint < 0x10000) ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff) || codepoint > 0x10ffff) {
            return "";
        }
        output.append(base, i, width);
        i += width;
    }
    if (output.empty() || output == "." || output == ".." || output.size() > 255) return "";
    return output;
}

std::string json_escape(const std::string& value) {
    std::string output;
    for (char c : value) {
        switch (c) {
            case '"':
                output += "\\\"";
                break;
            case '\\':
                output += "\\\\";
                break;
            case '\n':
                output += "\\n";
                break;
            case '\r':
                break;
            case '\t':
                output += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char escaped[8];
                    std::snprintf(escaped, sizeof(escaped), "\\u%04x",
                                  static_cast<unsigned char>(c));
                    output += escaped;
                } else {
                    output += c;
                }
        }
    }
    return output;
}

bool parse_int_field(const std::string& value, int minimum, int maximum, int& output) {
    if (value.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno == ERANGE || end != value.c_str() + value.size() || parsed < minimum ||
        parsed > maximum) {
        return false;
    }
    output = static_cast<int>(parsed);
    return true;
}

bool parse_double_field(const std::string& value, double minimum, double maximum,
                        double& output) {
    if (value.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (errno == ERANGE || end != value.c_str() + value.size() || !std::isfinite(parsed) ||
        parsed < minimum || parsed > maximum) {
        return false;
    }
    output = parsed;
    return true;
}

bool parse_bool_field(const std::string& value, bool& output) {
    if (value == "0") {
        output = false;
        return true;
    }
    if (value == "1") {
        output = true;
        return true;
    }
    return false;
}

std::string read_file(const std::string& path, std::size_t limit) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return "";
    const std::streamoff length = input.tellg();
    if (length < 0 || static_cast<std::uintmax_t>(length) > limit) return "";
    std::string data(static_cast<std::size_t>(length), '\0');
    input.seekg(0);
    if (!data.empty() &&
        !input.read(data.data(), static_cast<std::streamsize>(data.size()))) {
        return "";
    }
    return data;
}

std::string log_tail(const std::string& path, std::size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return "";
    const std::streamoff length = input.tellg();
    if (length < 0) return "";
    const std::size_t take = std::min(count, static_cast<std::size_t>(length));
    std::string data(take, '\0');
    input.seekg(length - static_cast<std::streamoff>(take));
    if (take != 0 &&
        !input.read(data.data(), static_cast<std::streamsize>(take))) {
        return "";
    }
    return data;
}

namespace {

const char* audio_demuxer(const std::string& extension) {
    if (extension == ".mp3") return "mp3";
    if (extension == ".m4a") return "mov";
    if (extension == ".aac") return "aac";
    if (extension == ".flac") return "flac";
    if (extension == ".ogg" || extension == ".opus") return "ogg";
    if (extension == ".wma") return "asf";
    if (extension == ".aif" || extension == ".aiff") return "aiff";
    return nullptr;
}

bool valid_audio_upload(const std::string& filename, const std::string& data) {
    const std::string extension = lower(fs::path(filename).extension().string());
    if (extension != ".wav") return audio_demuxer(extension) && !data.empty();
    return extension == ".wav" && data.size() >= 12 && data.compare(0, 4, "RIFF") == 0 &&
           data.compare(8, 4, "WAVE") == 0;
}

}  // namespace

// Inspect bounded chunk headers, without loading the decoded payload into RAM.
// FFmpeg must finish a complete, nonempty PCM16/44.1 kHz RIFF file.
std::optional<std::uint64_t> decoded_wav_frames(const fs::path& path, std::uintmax_t size, bool stereo) {
    std::ifstream in(path, std::ios::binary);
    auto u32 = [](const unsigned char* p) {
        return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
               (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
    };
    unsigned char header[16];
    if (size < 44 || !in.read(reinterpret_cast<char*>(header), 12) ||
        std::memcmp(header, "RIFF", 4) || std::memcmp(header + 8, "WAVE", 4) ||
        u32(header + 4) != size - 8) return std::nullopt;
    bool format = false, data = false;
    std::uint64_t frames = 0;
    std::uintmax_t offset = 12;
    while (offset + 8 <= size) {
        in.seekg(static_cast<std::streamoff>(offset));
        if (!in.read(reinterpret_cast<char*>(header), 8)) return std::nullopt;
        const auto count = u32(header + 4);
        const auto padded = std::uintmax_t(count) + (count & 1u);
        if (padded > size - offset - 8) return std::nullopt;
        if (!std::memcmp(header, "fmt ", 4)) {
            if (format || count < 16 || !in.read(reinterpret_cast<char*>(header), 16)) return std::nullopt;
            const unsigned channels = stereo ? 2 : 1;
            format = header[0] == 1 && header[1] == 0 && header[2] == channels && header[3] == 0 &&
                     u32(header + 4) == 44100 && u32(header + 8) == 88200 * channels &&
                     header[12] == 2 * channels && header[13] == 0 && header[14] == 16 && header[15] == 0;
            if (!format) return std::nullopt;
        } else if (!std::memcmp(header, "data", 4)) {
            if (data || !count || count % (stereo ? 4 : 2)) return std::nullopt;
            data = true;
            frames = count / (stereo ? 4 : 2);
        }
        offset += 8 + padded;
    }
    if (!format || !data || offset != size) return std::nullopt;
    return frames;
}

namespace {

std::string mime_for(const std::string& path) {
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".html") {
        return "text/html; charset=utf-8";
    }
    if (path.size() >= 3 && path.substr(path.size() - 3) == ".js") {
        return "application/javascript";
    }
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".css") return "text/css";
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".wav") return "audio/wav";
    return "application/octet-stream";
}

}  // namespace

bool collect_uploads(int fd, const std::vector<Part>& parts, std::vector<Upload>& uploads) {
    std::unordered_set<std::string> seen;
    for (const Part& part : parts) {
        if (part.filename.empty()) continue;
        std::string filename = safe_filename(part.filename);
        if (filename.empty() || !valid_audio_upload(filename, part.data)) {
            send_response(fd, 400, "Bad Request", "application/json",
                          "{\"error\":\"upload must be non-empty WAV, MP3, M4A, AAC, FLAC, OGG, Opus, WMA or AIFF/AIF audio\"}");
            return false;
        }
        fs::path normalized(filename);
        normalized.replace_extension(".wav");
        filename = normalized.string();
        if (!seen.insert(filename).second) {
            send_response(fd, 400, "Bad Request", "application/json",
                          "{\"error\":\"duplicate upload filename\"}");
            return false;
        }
        uploads.push_back({&part, std::move(filename)});
    }
    return true;
}

bool write_uploads(const std::vector<Upload>& uploads, const fs::path& raw,
                   const std::optional<fs::path>& work, std::string& error) {
    std::error_code filesystem_error;
    fs::create_directories(raw, filesystem_error);
    if (filesystem_error) {
        error = filesystem_error.message();
        return false;
    }
    if (work) {
        fs::create_directories(*work, filesystem_error);
        if (filesystem_error) {
            error = filesystem_error.message();
            return false;
        }
    }
    for (const Upload& upload : uploads) {
        const fs::path destination = raw / upload.filename;
        std::ofstream output(destination, std::ios::binary | std::ios::trunc);
        if (!output) {
            error = "cannot create " + destination.string();
            return false;
        }
        output.write(upload.part->data.data(),
                     static_cast<std::streamsize>(upload.part->data.size()));
        output.close();
        if (!output) {
            error = "cannot write " + destination.string();
            return false;
        }
        if (work) {
            fs::copy_file(destination, *work / upload.filename,
                          fs::copy_options::overwrite_existing, filesystem_error);
            if (filesystem_error) {
                error = filesystem_error.message();
                return false;
            }
        }
    }
    return true;
}

UploadResult normalize_uploads(const std::vector<Upload>& uploads, const fs::path& raw,
                               const std::optional<fs::path>& work, const Paths& paths,
                               const ProcessRunner& runner, std::mutex& decode_mutex, bool stereo) {
    const std::uintmax_t file_limit = stereo ? kMaxStereoUploadBytes : 64u << 20;
    constexpr std::uintmax_t batch_limit = 128u << 20;
    const auto compressed = [](const Upload& upload) {
        return lower(fs::path(safe_filename(upload.part->filename)).extension().string()) != ".wav";
    };
    std::string error;
    if (std::none_of(uploads.begin(), uploads.end(), compressed)) {
        if (!write_uploads(uploads, raw, work, error)) return {500, "upload staging failed"};
        return {};
    }
    std::unique_lock decoder_lock(decode_mutex, std::try_to_lock);
    if (!decoder_lock.owns_lock()) return {409, "audio decoder busy; retry after the current upload"};
    for (const auto& executable : {paths.ffmpeg, std::string("/usr/bin/timeout"), std::string("/usr/bin/prlimit")}) {
        if (!fs::path(executable).is_absolute() || access(executable.c_str(), X_OK) != 0)
            return {503, "audio decoder unavailable: FFmpeg, timeout and prlimit are required"};
    }
    std::error_code ec;
    fs::create_directories(raw, ec);
    if (ec) return {500, "upload staging failed"};
    if (work) fs::create_directories(*work, ec);
    if (ec) return {500, "upload staging failed"};
    std::uintmax_t decoded_bytes = 0;
    for (const auto& upload : uploads) {
        if (!compressed(upload)) {
            if (!write_uploads({upload}, raw, work, error)) return {500, "upload staging failed"};
            continue;
        }
        const std::string extension = lower(fs::path(safe_filename(upload.part->filename)).extension().string());
        const char* demuxer = audio_demuxer(extension);
        if (!demuxer) return {400, "unsupported audio format"};
        TempDir temp(raw.parent_path(), ".decode_");
        const fs::path input = fs::path(temp.path()) / ("input" + extension);
        const fs::path output = raw / upload.filename;
        {
            std::ofstream stream(input, std::ios::binary);
            stream.write(upload.part->data.data(), static_cast<std::streamsize>(upload.part->data.size()));
            stream.close();
            if (!stream) return {500, "upload staging failed"};
        }
        ProcessSpec spec;
        // timeout owns the process group; prlimit is exec'd in that group. No
        // shell, network protocols, playlist probing or GPU decoding is used.
        spec.argv = {"/usr/bin/timeout", "--signal=TERM", "--kill-after=2s", "120s",
                     "/usr/bin/prlimit", "--as=2147483648", "--fsize=" + std::to_string(file_limit + (128u << 10)), "--",
                     paths.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
                     "-max_alloc", "67108864", "-threads", "2", "-filter_threads", "1",
                     "-protocol_whitelist", "file", "-format_whitelist", demuxer, "-f", demuxer};
        if (extension == ".m4a")
            spec.argv.insert(spec.argv.end(), {"-enable_drefs", "0", "-use_absolute_path", "0"});
        spec.argv.insert(spec.argv.end(), {"-i", input.string(), "-map", "0:a:0", "-vn", "-sn", "-dn",
                         "-map_metadata", "-1", "-threads", "2", "-ac", stereo ? "2" : "1", "-ar", "44100",
                         "-c:a", "pcm_s16le", "-f", "wav", "-fs", std::to_string(file_limit + 1), "-y", output.string()});
        spec.log_path = fs::path(temp.path()) / "decode.log";
        const auto result = runner.run(spec);
        const auto size = fs::file_size(output, ec);
        // FFmpeg's -fs may stop successfully and overshoot by one packet. The
        // sentinel lies ABOVE the accepted cap, so truncation can never pass.
        if (!ec && size > file_limit)
            return {413, stereo ? "解码后的音频超过 128 MiB（并非原文件大小）；翻唱音频请控制在 10 分钟以内"
                                : "解码后的音频超过 64 MiB（并非原文件大小）；请将录音分段上传"};
        if (!ec && size > batch_limit - decoded_bytes)
            return {413, "本批音频解码后合计超过 128 MiB（并非原文件大小）；请减少本次上传的文件"};
        if (!result.success()) {
            if (result.termination == ProcessTermination::kLaunchFailed ||
                result.termination == ProcessTermination::kExecFailed ||
                result.exit_code == 126 || result.exit_code == 127)
                return {503, "audio decoder could not start; check the FFmpeg installation"};
            if (result.exit_code == 124 || result.exit_code == 137)
                return {400, "audio decoding timed out (120 seconds); split the recording"};
            return {400, "audio decoding failed; file is damaged, has no audio, or its codec is unavailable"};
        }
        const auto frames = ec ? std::nullopt : decoded_wav_frames(output, size, stereo);
        if (!frames) return {400, "audio decoder produced incomplete or empty WAV"};
        if (stereo && *frames > kMaxStereoUploadFrames)
            return {413, "翻唱音频时长超过 10 分钟；请裁剪后上传，原文件大小不代表解码后的长度"};
        decoded_bytes += size;
        if (work) {
            fs::copy_file(output, *work / upload.filename, fs::copy_options::overwrite_existing, ec);
            if (ec) return {500, "upload staging failed"};
        }
    }
    return {};
}

bool replace_material_dirs(const fs::path& directory, const fs::path& staged_raw,
                           const fs::path& staged_work, std::string& error) {
    const fs::path steps = directory / "steps.txt";
    std::error_code filesystem_error;
    const fs::file_status marker_status = fs::symlink_status(steps, filesystem_error);
    if (filesystem_error && filesystem_error != std::errc::no_such_file_or_directory) {
        error = filesystem_error.message();
        return false;
    }
    if (fs::exists(marker_status) && !fs::is_regular_file(marker_status)) {
        error = "existing step marker must be a regular file";
        return false;
    }
    filesystem_error.clear();

    TempDir backup(directory, ".material_backup_");
    const fs::path raw = directory / "raw";
    const fs::path work = directory / "work";
    const fs::path old_raw = fs::path(backup.path()) / "raw";
    const fs::path old_work = fs::path(backup.path()) / "work";
    const fs::path old_steps = fs::path(backup.path()) / "steps.txt";
    bool moved_raw = false;
    bool moved_work = false;
    bool moved_steps = false;
    bool installed_raw = false;
    bool installed_work = false;
    BackupRetentionGuard retention(backup, "material");
    const auto move_old = [&](const fs::path& source, const fs::path& destination,
                              bool& moved) {
        const bool exists = fs::exists(source, filesystem_error);
        if (filesystem_error) return false;
        if (!exists) return true;
        fs::rename(source, destination, filesystem_error);
        if (!filesystem_error) moved = true;
        return !filesystem_error;
    };
    // Retire the previous generation's marker as part of the same transaction.
    // A later publish failure restores it together with the old raw/work pair.
    if (move_old(raw, old_raw, moved_raw) && move_old(work, old_work, moved_work) &&
        move_old(steps, old_steps, moved_steps)) {
        fs::rename(staged_raw, raw, filesystem_error);
        if (!filesystem_error) {
            installed_raw = true;
#ifdef RESONANTIA_TEST_MATERIAL_COMMIT_GATE
            test_material_raw_installed(directory);
#endif
            fs::rename(staged_work, work, filesystem_error);
            if (!filesystem_error) installed_work = true;
        }
        if (!filesystem_error) {
            retention.allow_cleanup();
            return true;
        }
    }

    bool rollback_ok = true;
    const auto remove_installed = [&](const fs::path& path, bool installed) {
        if (!installed) return;
        std::error_code rollback_error;
        fs::remove_all(path, rollback_error);
        if (rollback_error) rollback_ok = false;
    };
    const auto restore_old = [&](const fs::path& source, const fs::path& destination,
                                 bool moved) {
        if (!moved) return;
        std::error_code rollback_error;
        fs::rename(source, destination, rollback_error);
        if (rollback_error) rollback_ok = false;
    };
    remove_installed(raw, installed_raw);
    remove_installed(work, installed_work);
    restore_old(old_raw, raw, moved_raw);
    restore_old(old_work, work, moved_work);
    restore_old(old_steps, steps, moved_steps);
    if (rollback_ok) retention.allow_cleanup();
    error = filesystem_error.message();
    return false;
}

bool replace_step_artifacts(const fs::path& directory, const fs::path& staged_work,
                            const fs::path& staged_steps, std::string& error) {
    TempDir backup(directory, ".step_backup_");
    struct Artifact {
        fs::path staged;
        fs::path live;
        fs::path old;
        bool moved = false;
        bool installed = false;
    };
    std::array<Artifact, 2> artifacts{{
        {staged_work, directory / "work", fs::path(backup.path()) / "work"},
        {staged_steps, directory / "steps.txt",
         fs::path(backup.path()) / "steps.txt"},
    }};
    std::error_code filesystem_error;
    BackupRetentionGuard retention(backup, "step");

    for (Artifact& artifact : artifacts) {
        const bool exists = fs::exists(artifact.live, filesystem_error);
        if (filesystem_error) break;
        if (!exists) continue;
        fs::rename(artifact.live, artifact.old, filesystem_error);
        if (filesystem_error) break;
        artifact.moved = true;
    }
    if (!filesystem_error) {
        for (Artifact& artifact : artifacts) {
            fs::rename(artifact.staged, artifact.live, filesystem_error);
            if (filesystem_error) break;
            artifact.installed = true;
        }
    }
    if (!filesystem_error) {
        retention.allow_cleanup();
        return true;
    }

    bool rollback_ok = true;
    for (auto artifact = artifacts.rbegin(); artifact != artifacts.rend(); ++artifact) {
        if (!artifact->installed) continue;
        std::error_code rollback_error;
        fs::remove_all(artifact->live, rollback_error);
        if (rollback_error) rollback_ok = false;
    }
    for (auto artifact = artifacts.rbegin(); artifact != artifacts.rend(); ++artifact) {
        if (!artifact->moved) continue;
        std::error_code rollback_error;
        fs::rename(artifact->old, artifact->live, rollback_error);
        if (rollback_error) rollback_ok = false;
    }
    if (rollback_ok) retention.allow_cleanup();
    error = filesystem_error.message();
    return false;
}

bool replace_training_artifacts(const fs::path& directory,
                                const fs::path& staged_clips,
                                const fs::path& staged_model,
                                const fs::path& staged_index,
                                std::string& error) {
    TempDir backup(directory, ".train_backup_");
    struct Artifact {
        fs::path staged;
        fs::path live;
        fs::path old;
        bool moved = false;
        bool installed = false;
    };
    std::array<Artifact, 3> artifacts{{
        {staged_clips, directory / "clips", fs::path(backup.path()) / "clips"},
        {staged_model, directory / "model.safetensors",
         fs::path(backup.path()) / "model.safetensors"},
        {staged_index, directory / "model.index",
         fs::path(backup.path()) / "model.index"},
    }};

    std::error_code filesystem_error;
    BackupRetentionGuard retention(backup, "training");
    for (Artifact& artifact : artifacts) {
        const bool exists = fs::exists(artifact.live, filesystem_error);
        if (filesystem_error) break;
        if (!exists) continue;
        fs::rename(artifact.live, artifact.old, filesystem_error);
        if (filesystem_error) break;
        artifact.moved = true;
    }
    if (!filesystem_error) {
        for (Artifact& artifact : artifacts) {
            fs::rename(artifact.staged, artifact.live, filesystem_error);
            if (filesystem_error) break;
            artifact.installed = true;
#ifdef RESONANTIA_TEST_TRAINING_PUBLISH_GATE
            if (&artifact == &artifacts.front()) test_training_clips_installed(directory);
#endif
        }
    }
    if (!filesystem_error) {
        retention.allow_cleanup();
        return true;
    }

    bool rollback_ok = true;
    for (auto artifact = artifacts.rbegin(); artifact != artifacts.rend(); ++artifact) {
        if (!artifact->installed) continue;
        std::error_code rollback_error;
        fs::remove_all(artifact->live, rollback_error);
        if (rollback_error) rollback_ok = false;
    }
    for (auto artifact = artifacts.rbegin(); artifact != artifacts.rend(); ++artifact) {
        if (!artifact->moved) continue;
        std::error_code rollback_error;
        fs::rename(artifact->old, artifact->live, rollback_error);
        if (rollback_error) rollback_ok = false;
    }
    if (rollback_ok) retention.allow_cleanup();
    error = filesystem_error.message();
    return false;
}

void serve_static(int fd, const std::string& webroot, const std::string& request_path) {
    if (request_path.empty() || request_path.front() != '/' ||
        (request_path.size() > 1 && request_path[1] == '/') ||
        request_path.find('\0') != std::string::npos) {
        send_response(fd, 403, "Forbidden", "text/plain", "no");
        return;
    }
    const fs::path relative =
        request_path == "/" ? fs::path("index.html") : fs::path(request_path.substr(1));
    if (relative.is_absolute()) {
        send_response(fd, 403, "Forbidden", "text/plain", "no");
        return;
    }
    for (const fs::path& component : relative) {
        if (component == "." || component == "..") {
            send_response(fd, 403, "Forbidden", "text/plain", "no");
            return;
        }
    }
    std::error_code filesystem_error;
    const fs::path root = fs::canonical(webroot, filesystem_error);
    if (filesystem_error) {
        send_response(fd, 500, "Internal Server Error", "text/plain", "webroot unavailable");
        return;
    }
    const fs::path full_path = fs::canonical(root / relative, filesystem_error);
    if (filesystem_error || !fs::is_regular_file(full_path, filesystem_error) ||
        filesystem_error) {
        send_response(fd, 404, "Not Found", "text/plain", "not found");
        return;
    }
    auto root_component = root.begin();
    auto full_component = full_path.begin();
    while (root_component != root.end() && full_component != full_path.end() &&
           *root_component == *full_component) {
        ++root_component;
        ++full_component;
    }
    if (root_component != root.end()) {
        send_response(fd, 403, "Forbidden", "text/plain", "no");
        return;
    }
    const std::string full = full_path.string();
    const std::string content = read_file(full);
    if (content.empty()) {
        send_response(fd, 404, "Not Found", "text/plain", "not found");
        return;
    }
    send_response(fd, 200, "OK", mime_for(full), content);
}

std::string field(const std::vector<Part>& parts, const std::string& name,
                  const std::string& default_value) {
    for (const Part& part : parts) {
        if (part.name == name && part.filename.empty()) return part.data;
    }
    return default_value;
}

std::string query_get(const std::string& query, const std::string& key) {
    std::istringstream stream(query);
    std::string pair;
    while (std::getline(stream, pair, '&')) {
        const std::size_t equals = pair.find('=');
        if (equals != std::string::npos && pair.substr(0, equals) == key) {
            return pair.substr(equals + 1);
        }
    }
    return "";
}

}  // namespace resonantia::web
