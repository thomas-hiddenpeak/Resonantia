// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cerrno>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tools/web/workflow_support.h"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {

class TempTree {
public:
    TempTree() {
        std::string pattern = (fs::temp_directory_path() /
                               "resonantia-web-contract-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        char* made = mkdtemp(writable.data());
        if (!made) throw std::runtime_error("mkdtemp failed");
        path_ = made;
    }

    TempTree(const TempTree&) = delete;
    TempTree& operator=(const TempTree&) = delete;

    ~TempTree() {
        std::error_code ignored;
        fs::permissions(path_, fs::perms::owner_all, fs::perm_options::add, ignored);
        for (auto it = fs::recursive_directory_iterator(
                 path_, fs::directory_options::skip_permission_denied, ignored);
             !ignored && it != fs::recursive_directory_iterator(); it.increment(ignored)) {
            if (it->is_directory(ignored))
                fs::permissions(it->path(), fs::perms::owner_all,
                                fs::perm_options::add, ignored);
        }
        fs::remove_all(path_, ignored);
    }

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

void write_file(const fs::path& path, const std::string& data) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create " + path.string());
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
    output.close();
    if (!output) throw std::runtime_error("cannot write " + path.string());
}

std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

std::string minimal_wav(const std::string& payload = {}) {
    return std::string("RIFF", 4) + "xxxx" + "WAVE" + payload;
}

int reserve_loopback_port() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket failed while choosing port");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        close(fd);
        throw std::runtime_error("bind failed while choosing port");
    }
    socklen_t length = sizeof(address);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
        close(fd);
        throw std::runtime_error("getsockname failed while choosing port");
    }
    int port = ntohs(address.sin_port);
    close(fd);
    return port;
}

bool send_all(int fd, const std::string& data) {
    size_t offset = 0;
    while (offset < data.size()) {
        ssize_t written = send(fd, data.data() + offset, data.size() - offset,
                               MSG_NOSIGNAL);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        offset += static_cast<size_t>(written);
    }
    return true;
}

struct HttpResponse {
    int status = 0;
    std::string headers;
    std::string body;

    bool has_header(const std::string& header) const {
        return headers.find(header) != std::string::npos;
    }
};

HttpResponse request(int port, const std::string& method, const std::string& path,
                     const std::string& body = {}, const std::string& content_type = {},
                     const std::string& extra_headers = {}, int io_timeout_seconds = 0,
                     std::size_t* input_bytes = nullptr) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("request socket failed");
    if (io_timeout_seconds > 0) {
        const timeval timeout{io_timeout_seconds, 0};
        if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) ||
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout))) {
            close(fd);
            throw std::runtime_error("request socket timeout setup failed");
        }
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        int saved_errno = errno;
        close(fd);
        throw std::runtime_error("connect failed: " +
                                 std::string(std::strerror(saved_errno)));
    }

    std::string wire = method + " " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\n";
    if (method == "POST" || method == "PUT")
        wire += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    if (!content_type.empty()) wire += "Content-Type: " + content_type + "\r\n";
    wire += extra_headers;
    wire += "Connection: close\r\n\r\n" + body;
    if (input_bytes && *input_bytes + wire.size() > 4096) {
        close(fd);
        throw std::runtime_error("body deadline total input exceeded 4KiB");
    }
    if (!send_all(fd, wire)) {
        close(fd);
        throw std::runtime_error("request send failed");
    }
    if (input_bytes) *input_bytes += wire.size();
    shutdown(fd, SHUT_WR);

    std::string raw;
    char buffer[8192];
    for (;;) {
        ssize_t count = recv(fd, buffer, sizeof(buffer), 0);
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) {
            int saved_errno = errno;
            close(fd);
            throw std::runtime_error("response read failed: " +
                                     std::string(std::strerror(saved_errno)));
        }
        if (count == 0) break;
        raw.append(buffer, static_cast<size_t>(count));
    }
    close(fd);

    HttpResponse response;
    size_t line_end = raw.find("\r\n");
    size_t headers_end = raw.find("\r\n\r\n");
    if (line_end == std::string::npos || headers_end == std::string::npos)
        throw std::runtime_error("malformed HTTP response");
    size_t first_space = raw.find(' ');
    if (first_space == std::string::npos || first_space > line_end)
        throw std::runtime_error("malformed HTTP status");
    response.status = std::stoi(raw.substr(first_space + 1, 3));
    response.headers = raw.substr(0, headers_end + 2);
    response.body = raw.substr(headers_end + 4);
    return response;
}

std::string multipart_body(
    const std::vector<std::pair<std::string, std::string>>& fields,
    const std::vector<std::pair<std::string, std::string>>& files = {}) {
    constexpr char kBoundary[] = "resonantia-contract-boundary";
    std::string body;
    for (const auto& [name, value] : fields) {
        body += "--" + std::string(kBoundary) + "\r\n";
        body += "Content-Disposition: form-data; name=\"" + name + "\"\r\n\r\n";
        body += value + "\r\n";
    }
    for (const auto& [filename, value] : files) {
        body += "--" + std::string(kBoundary) + "\r\n";
        body += "Content-Disposition: form-data; name=\"audio\"; filename=\"" +
                filename + "\"\r\nContent-Type: audio/wav\r\n\r\n";
        body += value + "\r\n";
    }
    body += "--" + std::string(kBoundary) + "--\r\n";
    return body;
}

constexpr char kMultipartType[] =
    "multipart/form-data; boundary=resonantia-contract-boundary";

class ServerProcess {
public:
    ServerProcess(const fs::path& executable, const fs::path& root,
                  const fs::path& hold_trigger, const fs::path& hold_started,
                  const fs::path& hold_release, const fs::path& fail_trigger,
                  const std::string& token = {}, std::size_t* input_bytes = nullptr)
        : port_(reserve_loopback_port()), input_bytes_(input_bytes) {
        pid_ = fork();
        if (pid_ < 0) throw std::runtime_error("fork failed");
        if (pid_ == 0) {
            setenv("RESONANTIA_TEST_ROOT", root.c_str(), 1);
            setenv("RESONANTIA_TEST_HOLD_TRIGGER", hold_trigger.c_str(), 1);
            setenv("RESONANTIA_TEST_HOLD_STARTED", hold_started.c_str(), 1);
            setenv("RESONANTIA_TEST_HOLD_RELEASE", hold_release.c_str(), 1);
            setenv("RESONANTIA_TEST_FAIL_STEP_COMMIT", fail_trigger.c_str(), 1);
            const fs::path spawned_child_pid = root / "spawned-child.pid";
            setenv("RESONANTIA_TEST_CHILD_PID", spawned_child_pid.c_str(), 1);
            fs::path log = root / "server.log";
            int log_fd = open(log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0600);
            if (log_fd >= 0) {
                dup2(log_fd, STDOUT_FILENO);
                dup2(log_fd, STDERR_FILENO);
                close(log_fd);
            }
            std::string port = std::to_string(port_);
            std::vector<std::string> args{
                executable.string(), "--host", "127.0.0.1", "--port", port,
                "--webroot", (root / "webui").string(), "--repo", root.string()};
            if (!token.empty()) {
                args.push_back("--auth-token");
                args.push_back(token);
            }
            std::vector<char*> argv;
            for (auto& arg : args) argv.push_back(arg.data());
            argv.push_back(nullptr);
            execv(executable.c_str(), argv.data());
            _exit(127);
        }
        try {
            wait_until_ready();
        } catch (...) {
            kill(pid_, SIGKILL);
            waitpid(pid_, nullptr, 0);
            pid_ = -1;
            throw;
        }
    }

    ServerProcess(const ServerProcess&) = delete;
    ServerProcess& operator=(const ServerProcess&) = delete;

    ~ServerProcess() { stop(); }

    int port() const { return port_; }
    pid_t pid() const { return pid_; }

    void stop() {
        if (pid_ <= 0) return;
        kill(pid_, SIGTERM);
        for (int attempt = 0; attempt < 100; ++attempt) {
            int status = 0;
            pid_t result = waitpid(pid_, &status, WNOHANG);
            if (result == pid_) {
                pid_ = -1;
                return;
            }
            std::this_thread::sleep_for(10ms);
        }
        kill(pid_, SIGKILL);
        waitpid(pid_, nullptr, 0);
        pid_ = -1;
    }

private:
    void wait_until_ready() {
        for (int attempt = 0; attempt < 200; ++attempt) {
            int status = 0;
            pid_t result = waitpid(pid_, &status, WNOHANG);
            if (result == pid_) {
                pid_ = -1;
                throw std::runtime_error("vc_serve exited before accepting requests");
            }
            try {
                if (request(port_, "GET", "/", {}, {}, {}, input_bytes_ ? 1 : 0,
                            input_bytes_).status == 200) return;
            } catch (const std::exception&) {
            }
            std::this_thread::sleep_for(20ms);
        }
        throw std::runtime_error("vc_serve did not become ready");
    }

    pid_t pid_ = -1;
    int port_ = 0;
    std::size_t* input_bytes_ = nullptr;
};

bool wait_for(const std::function<bool()>& predicate,
              std::chrono::milliseconds timeout = 5s) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        if (predicate()) return true;
        std::this_thread::sleep_for(20ms);
    } while (std::chrono::steady_clock::now() < deadline);
    return predicate();
}

bool process_has_socket_fd(pid_t process_id) {
    std::error_code error;
    const fs::path directory = fs::path("/proc") / std::to_string(process_id) / "fd";
    for (fs::directory_iterator entry(directory, error), end; !error && entry != end;
         entry.increment(error)) {
        const fs::path target = fs::read_symlink(entry->path(), error);
        if (error) return true;
        if (target.string().rfind("socket:[", 0) == 0) return true;
    }
    return error.value() != 0;
}

class Checks {
public:
    void expect(bool condition, const std::string& message) {
        if (condition) return;
        ++failures_;
        std::cerr << "FAIL: " << message << '\n';
    }

    int failures() const { return failures_; }

private:
    int failures_ = 0;
};

bool has_hidden_training_directory(const fs::path& directory) {
    std::error_code error;
    for (fs::directory_iterator entry(directory, error), end; !error && entry != end;
         entry.increment(error)) {
        const std::string name = entry->path().filename().string();
        if (name.rfind(".train_stage_", 0) == 0 ||
            name.rfind(".train_backup_", 0) == 0) {
            return true;
        }
    }
    return error.value() != 0;
}

void check_training_artifact_transaction(Checks& checks, const fs::path& root) {
    using resonantia::web::replace_training_artifacts;

    const fs::path success = root / "transaction-success";
    write_file(success / "clips/old.wav", minimal_wav("old-success"));
    write_file(success / "model.safetensors", "old-success-model");
    write_file(success / "model.index", "old-success-index");
    const fs::path success_stage = success / "staged";
    write_file(success_stage / "clips/new.wav", minimal_wav("new-success"));
    write_file(success_stage / "model.safetensors", "new-success-model");
    write_file(success_stage / "model.index", "new-success-index");
    std::string error;
    const bool committed = replace_training_artifacts(
        success, success_stage / "clips", success_stage / "model.safetensors",
        success_stage / "model.index", error);
    checks.expect(committed && error.empty() &&
                      read_file(success / "clips/new.wav") ==
                          minimal_wav("new-success") &&
                      read_file(success / "model.safetensors") ==
                          "new-success-model" &&
                      read_file(success / "model.index") == "new-success-index" &&
                      !has_hidden_training_directory(success),
                  "training artifact transaction must replace an old triplet and clean backup");

    const fs::path rollback = root / "transaction-rollback";
    write_file(rollback / "clips/old.wav", minimal_wav("old-rollback"));
    write_file(rollback / "model.safetensors", "old-rollback-model");
    write_file(rollback / "model.index", "old-rollback-index");
    const fs::path rollback_stage = rollback / "staged";
    write_file(rollback_stage / "clips/new.wav", minimal_wav("new-rollback"));
    write_file(rollback_stage / "model.safetensors", "new-rollback-model");
    error.clear();
    const bool rejected = replace_training_artifacts(
        rollback, rollback_stage / "clips", rollback_stage / "model.safetensors",
        rollback_stage / "missing.index", error);
    checks.expect(!rejected && !error.empty() &&
                      read_file(rollback / "clips/old.wav") ==
                          minimal_wav("old-rollback") &&
                      read_file(rollback / "model.safetensors") ==
                          "old-rollback-model" &&
                      read_file(rollback / "model.index") == "old-rollback-index" &&
                      !fs::exists(rollback / "clips/new.wav") &&
                      !has_hidden_training_directory(rollback),
                  "partial training artifact publish must roll back the complete old triplet");
}

bool has_material_staging(const fs::path& directory) {
    for (const auto& entry : fs::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind(".material_backup_", 0) == 0 ||
            name.rfind(".material_upload_", 0) == 0 ||
            name.rfind(".train_upload_", 0) == 0) return true;
    }
    return false;
}

void check_material_transaction(Checks& checks, const fs::path& root) {
    using resonantia::web::replace_material_dirs;
    const std::string old_raw = minimal_wav("old-raw");
    const std::string old_work = minimal_wav("old-work");
    for (bool have_old : {false, true}) {
        for (bool fail_work : {false, true}) {
            const fs::path directory = root / (std::string("material-transaction-") +
                (have_old ? "old-" : "new-") + (fail_work ? "rollback" : "success"));
            if (have_old) {
                write_file(directory / "raw/old.wav", old_raw);
                write_file(directory / "work/old.wav", old_work);
                write_file(directory / "steps.txt", "denoise\n");
            }
            const fs::path staged = directory / "staged";
            write_file(staged / "raw/new.wav", minimal_wav("new-material"));
            if (!fail_work)
                write_file(staged / "work/new.wav", minimal_wav("new-material"));
            std::string error;
            const bool committed = replace_material_dirs(
                directory, staged / "raw", staged / "work", error);
            if (fail_work) {
                checks.expect(!committed && !error.empty() &&
                                  !fs::exists(directory / "raw/new.wav") &&
                                  !fs::exists(directory / "work/new.wav") &&
                                  (have_old
                                      ? read_file(directory / "raw/old.wav") == old_raw &&
                                        read_file(directory / "work/old.wav") == old_work &&
                                        read_file(directory / "steps.txt") == "denoise\n"
                                      : !fs::exists(directory / "raw") &&
                                        !fs::exists(directory / "work") &&
                                        !fs::exists(directory / "steps.txt")),
                              "second material rename failure must restore the previous generation");
            } else {
                checks.expect(committed && error.empty() &&
                                  read_file(directory / "raw/new.wav") == minimal_wav("new-material") &&
                                  read_file(directory / "work/new.wav") == minimal_wav("new-material") &&
                                  !fs::exists(directory / "steps.txt"),
                              "material transaction must publish raw/work and invalidate the old marker together");
            }
            checks.expect(!has_material_staging(directory),
                          "material success or complete rollback must remove its backup");
        }
    }
    for (bool dangling : {false, true}) {
        const fs::path directory = root / (dangling ? "material-dangling-marker"
                                                   : "material-symlink-marker");
        write_file(directory / "raw/old.wav", old_raw);
        write_file(directory / "work/old.wav", old_work);
        write_file(directory / "staged/raw/new.wav", minimal_wav("new"));
        write_file(directory / "staged/work/new.wav", minimal_wav("new"));
        if (!dangling) write_file(directory / "marker-target", "preserve\n");
        fs::create_symlink("marker-target", directory / "steps.txt");
        std::string error;
        checks.expect(!replace_material_dirs(directory, directory / "staged/raw",
                                              directory / "staged/work", error) &&
                          !error.empty() &&
                          fs::is_symlink(directory / "steps.txt") &&
                          read_file(directory / "raw/old.wav") == old_raw &&
                          read_file(directory / "work/old.wav") == old_work &&
                          (dangling ? !fs::exists(directory / "marker-target")
                                    : read_file(directory / "marker-target") == "preserve\n") &&
                          !has_material_staging(directory),
                      "symlink marker rejection must preserve material, link and target");
    }
}

void check_step_artifact_transaction(Checks& checks, const fs::path& root) {
    using resonantia::web::replace_step_artifacts;

    const fs::path success = root / "step-transaction-success";
    write_file(success / "work/old.wav", minimal_wav("old-step-success"));
    write_file(success / "steps.txt", "separate\n");
    const fs::path success_stage = success / "staged";
    write_file(success_stage / "work/new.wav", minimal_wav("new-step-success"));
    write_file(success_stage / "steps.txt", "separate\ndenoise\n");
    std::string error;
    const bool committed = replace_step_artifacts(
        success, success_stage / "work", success_stage / "steps.txt", error);
    checks.expect(committed && error.empty() &&
                      read_file(success / "work/new.wav") ==
                          minimal_wav("new-step-success") &&
                      read_file(success / "steps.txt") == "separate\ndenoise\n",
                  "step artifact transaction must publish work and marker together");

    const fs::path rollback = root / "step-transaction-rollback";
    write_file(rollback / "work/old.wav", minimal_wav("old-step-rollback"));
    write_file(rollback / "steps.txt", "separate\n");
    const fs::path rollback_stage = rollback / "staged";
    write_file(rollback_stage / "work/new.wav", minimal_wav("new-step-rollback"));
    error.clear();
    const bool rejected = replace_step_artifacts(
        rollback, rollback_stage / "work", rollback_stage / "missing-steps.txt",
        error);
    checks.expect(!rejected && !error.empty() &&
                      read_file(rollback / "work/old.wav") ==
                          minimal_wav("old-step-rollback") &&
                      read_file(rollback / "steps.txt") == "separate\n" &&
                      !fs::exists(rollback / "work/new.wav"),
                  "partial step publish must roll back both work and marker");
}

void create_fake_preprocess(const fs::path& root) {
    const fs::path executable = root / "build/vc_preprocess";
    write_file(executable, R"SH(#!/bin/sh
input=
output=
printf '%s\n' "$*" >> "$RESONANTIA_TEST_ROOT/preprocess-invocations"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --input) input=$2; shift 2 ;;
        --output-dir) output=$2; shift 2 ;;
        *) shift ;;
    esac
done
if [ -f "$RESONANTIA_TEST_HOLD_TRIGGER" ] && [ ! -d "$input" ]; then
    printf '%s' "$$" > "$RESONANTIA_TEST_CHILD_PID"
    : > "$RESONANTIA_TEST_HOLD_STARTED"
    while [ ! -f "$RESONANTIA_TEST_HOLD_RELEASE" ]; do
        sleep 0.02
    done
fi
mkdir -p "$output" || exit 1
source=$input
if [ -d "$input" ]; then
    source=
    for candidate in "$input"/*.wav; do
        if [ -f "$candidate" ]; then source=$candidate; break; fi
    done
fi
[ -n "$source" ] || exit 1
cp "$source" "$output/result.wav" || exit 1
if [ -f "$RESONANTIA_TEST_ROOT/preprocess-empty-output.trigger" ] && [ ! -d "$input" ]; then
    : > "$output/result.wav"
    exit 0
fi
if [ -f "$RESONANTIA_TEST_ROOT/preprocess-oversized-output.trigger" ] && [ ! -d "$input" ]; then
    truncate -s 67108865 "$output/result.wav"
    exit 0
fi
if [ -f "$RESONANTIA_TEST_ROOT/preprocess-cli-fail.trigger" ] && [ ! -d "$input" ]; then
    echo "error: requested preprocess failure"
    exit 31
fi
case "$input" in
    *"/runs/train-preprocess-failure/"*)
        echo "error: requested training preprocess failure"
        exit 32
        ;;
    *"/runs/step-cli-failure/"*)
        echo "processed with requested step failure"
        exit 33
        ;;
    *"/runs/step-nonwav-output/"*)
        rm -f "$output/result.wav"
        echo "not audio" > "$output/result.txt"
        exit 0
        ;;
    *"/runs/train-empty-clips/"*)
        : > "$output/result.wav"
        exit 0
        ;;
esac
if [ -f "$RESONANTIA_TEST_FAIL_STEP_COMMIT" ] && [ -d "$input" ]; then
    chmod 0555 "$(dirname "$input")" || exit 1
fi
echo "processed clip"
)SH");
    if (chmod(executable.c_str(), 0700) != 0)
        throw std::runtime_error("chmod fake vc_preprocess failed");
}

void create_fake_training_commands(const fs::path& root) {
    const fs::path train = root / "build/vc_train";
    write_file(train, R"SH(#!/bin/sh
target=
output=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --target) target=$2; shift 2 ;;
        --out) output=$2; shift 2 ;;
        *) shift ;;
    esac
done
[ -n "$target" ] && [ -n "$output" ] || exit 2
mkdir -p "$(dirname "$output")" || exit 3
case "$target" in
    *"/runs/train-empty-model/"*)
        rm -f "$output"
        echo "step done without model"
        exit 0
        ;;
esac
printf 'fake model' > "$output" || exit 4
case "$target" in
    *"/runs/train-cli-failure"*)
        echo "step error: requested training failure"
        exit 41
        ;;
esac
echo "step done"
)SH");
    if (chmod(train.c_str(), 0700) != 0)
        throw std::runtime_error("chmod fake vc_train failed");

    const fs::path build_index = root / "build/build_index";
    write_file(build_index, R"SH(#!/bin/sh
input=
output=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --input-dir) input=$2; shift 2 ;;
        --output) output=$2; shift 2 ;;
        *) shift ;;
    esac
done
[ -n "$input" ] && [ -n "$output" ] || exit 2
mkdir -p "$(dirname "$output")" || exit 3
case "$input" in
    *"/runs/index-empty-output/"*)
        rm -f "$output"
        echo "Wrote no index"
        exit 0
        ;;
esac
printf 'fake index' > "$output" || exit 4
case "$input" in
    *"/runs/index-cli-failure/"*)
        echo "index error: requested index failure"
        exit 51
        ;;
esac
echo "Wrote index"
)SH");
    if (chmod(build_index.c_str(), 0700) != 0)
        throw std::runtime_error("chmod fake build_index failed");
}

void create_fake_convert(const fs::path& root) {
    const fs::path convert = root / "build/vc_convert";
    write_file(convert, R"SH(#!/bin/sh
output=
printf '%s\n' "$*" > "$RESONANTIA_TEST_ROOT/convert-invoked"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output) output=$2; shift 2 ;;
        *) shift ;;
    esac
done
[ -n "$output" ] || exit 2
mkdir -p "$(dirname "$output")" || exit 3
if [ -f "$RESONANTIA_TEST_ROOT/convert-empty-output.trigger" ]; then
    : > "$output"
    exit 0
fi
if [ -f "$RESONANTIA_TEST_ROOT/convert-invalid-output.trigger" ]; then
    printf 'not a wav' > "$output"
    exit 0
fi
if [ -f "$RESONANTIA_TEST_ROOT/convert-oversized-output.trigger" ]; then
    printf 'RIFFxxxxWAVE' > "$output"
    truncate -s 67108865 "$output"
    exit 0
fi
printf 'RIFFxxxxWAVEconverted' > "$output" || exit 4
if [ -f "$RESONANTIA_TEST_ROOT/convert-cli-fail.trigger" ]; then
    echo "conversion failed after output"
    exit 61
fi
echo "conversion done"
echo "Done (12 ms)"
echo "  HuBERT:  3 ms"
echo "  F0:      4 ms"
echo "  Index:   2 ms"
echo "  Synth:   3 ms"
)SH");
    if (chmod(convert.c_str(), 0700) != 0)
        throw std::runtime_error("chmod fake vc_convert failed");
}

HttpResponse wait_for_job_completion(int port, const std::string& name) {
    HttpResponse status;
    if (!wait_for([&] {
            status = request(port, "GET", "/api/train/status?name=" + name);
            return status.body.find("\"name\":\"" + name + "\"") !=
                       std::string::npos &&
                   status.body.find("\"running\":false") != std::string::npos;
        })) {
        throw std::runtime_error("job did not finish: " + name);
    }
    return status;
}

void check_training_failure(Checks& checks, int port, const fs::path& root,
                            const std::string& name, const std::string& message) {
    const std::string old_model = "old-model:" + name;
    const std::string old_index = "old-index:" + name;
    const std::string old_clip = minimal_wav("old-clip:" + name);
    write_file(root / "runs" / name / "model.safetensors", old_model);
    write_file(root / "runs" / name / "model.index", old_index);
    write_file(root / "runs" / name / "clips/old.wav", old_clip);
    write_file(root / "runs" / name / "raw/input.wav", minimal_wav("raw:" + name));
    write_file(root / "runs" / name / "work/input.wav", minimal_wav("work:" + name));
    const std::string body = multipart_body({{"name", name}, {"steps", "1"}});
    const HttpResponse start =
        request(port, "POST", "/api/train", body, kMultipartType);
    if (start.status != 200) {
        checks.expect(false, message + " (job was not accepted)");
        return;
    }
    const HttpResponse status = wait_for_job_completion(port, name);
    bool hidden_transaction_dir = false;
    for (const auto& entry : fs::directory_iterator(root / "runs" / name)) {
        const std::string filename = entry.path().filename().string();
        if (filename.rfind(".train_stage_", 0) == 0 ||
            filename.rfind(".train_backup_", 0) == 0) {
            hidden_transaction_dir = true;
        }
    }
    checks.expect(
        status.body.find("\"stage\":\"error\"") != std::string::npos &&
            status.body.find("\"error\":true") != std::string::npos &&
            read_file(root / "runs" / name / "model.safetensors") == old_model &&
            read_file(root / "runs" / name / "model.index") == old_index &&
            read_file(root / "runs" / name / "clips/old.wav") == old_clip &&
            !fs::exists(root / "runs" / name / "clips/result.wav") &&
            !hidden_transaction_dir,
        message + " and must preserve the previous model/index/clips transaction");
}

void check_new_training_failure(Checks& checks, int port, const fs::path& root,
                                const std::string& name) {
    write_file(root / "runs" / name / "raw/input.wav", minimal_wav("raw:" + name));
    write_file(root / "runs" / name / "work/input.wav", minimal_wav("work:" + name));
    const std::string body = multipart_body({{"name", name}, {"steps", "1"}});
    const HttpResponse start =
        request(port, "POST", "/api/train", body, kMultipartType);
    if (start.status != 200) {
        checks.expect(false, "failed new training should be accepted before CLI exit");
        return;
    }
    const HttpResponse status = wait_for_job_completion(port, name);
    checks.expect(status.body.find("\"stage\":\"error\"") != std::string::npos &&
                      status.body.find("\"error\":true") != std::string::npos &&
                      !fs::exists(root / "runs" / name / "model.safetensors") &&
                      !fs::exists(root / "runs" / name / "model.index") &&
                      !fs::exists(root / "runs" / name / "clips"),
                  "failed training of a new voice must not expose partial artifacts");
}

void check_training_success(Checks& checks, int port, const fs::path& root,
                            const std::string& name) {
    write_file(root / "runs" / name / "raw/input.wav", minimal_wav("raw:" + name));
    write_file(root / "runs" / name / "work/input.wav", minimal_wav("work:" + name));
    const std::string body = multipart_body({{"name", name}, {"steps", "1"}});
    const HttpResponse start =
        request(port, "POST", "/api/train", body, kMultipartType);
    if (start.status != 200) {
        checks.expect(false, "successful training workflow should be accepted");
        return;
    }
    const HttpResponse status = wait_for_job_completion(port, name);
    std::error_code model_error, index_error;
    const auto model_size = fs::file_size(root / "runs" / name / "model.safetensors",
                                          model_error);
    const auto index_size = fs::file_size(root / "runs" / name / "model.index",
                                          index_error);
    checks.expect(status.body.find("\"stage\":\"done\"") != std::string::npos &&
                      status.body.find("\"error\":false") != std::string::npos &&
                      !model_error && model_size > 0 && !index_error && index_size > 0,
                  "successful training must require non-empty model and index outputs");
}

void check_training_preprocess_source(Checks& checks, int port,
                                      const fs::path& root,
                                      const std::string& name,
                                      bool has_step_marker) {
    const fs::path directory = root / "runs" / name;
    write_file(directory / "raw/input.wav", minimal_wav("raw:" + name));
    write_file(directory / "work/input.wav", minimal_wav("work:" + name));
    if (has_step_marker) write_file(directory / "steps.txt", "denoise\n");
    std::error_code ignored;
    fs::remove(root / "preprocess-invocations", ignored);
    const HttpResponse start = request(
        port, "POST", "/api/train",
        multipart_body({{"name", name}, {"steps", "1"}, {"denoise", "1"}}),
        kMultipartType);
    if (start.status != 200) {
        checks.expect(false, "training source-selection job should be accepted");
        return;
    }
    const HttpResponse status = wait_for_job_completion(port, name);
    const std::string invocation = read_file(root / "preprocess-invocations");
    const fs::path expected_input = directory / (has_step_marker ? "work" : "raw");
    checks.expect(
        status.body.find("\"stage\":\"done\"") != std::string::npos &&
            invocation.find("--input " + expected_input.string()) != std::string::npos &&
            (has_step_marker ? invocation.find("--denoise") == std::string::npos
                             : invocation.find("--denoise") != std::string::npos),
        has_step_marker
            ? "step-marked work must be sliced without repeating the selected stage"
            : "unprocessed material must use raw input and honor requested stages");
}

void check_training_upload(Checks& checks, int port, const fs::path& root) {
    const std::string name = "training-replaces-material";
    const fs::path directory = root / "runs" / name;
    const std::string fresh = minimal_wav("fresh-training-upload");
    write_file(directory / "raw/old.wav", minimal_wav("old-raw"));
    write_file(directory / "work/old.wav", minimal_wav("old-work"));
    write_file(directory / "steps.txt", "denoise\n");
    fs::remove(root / "preprocess-invocations");
    const HttpResponse start = request(
        port, "POST", "/api/train",
        multipart_body({{"name", name}, {"steps", "1"}, {"denoise", "1"}},
                       {{"fresh.wav", fresh}}), kMultipartType);
    if (start.status != 200) {
        checks.expect(false, "training with fresh material should be accepted");
        return;
    }
    const HttpResponse status = wait_for_job_completion(port, name);
    const std::string invocation = read_file(root / "preprocess-invocations");
    checks.expect(status.body.find("\"stage\":\"done\"") != std::string::npos &&
                      read_file(directory / "raw/fresh.wav") == fresh &&
                      read_file(directory / "work/fresh.wav") == fresh &&
                      !fs::exists(directory / "raw/old.wav") &&
                      !fs::exists(directory / "work/old.wav") &&
                      !fs::exists(directory / "steps.txt") &&
                      !has_material_staging(directory) &&
                      invocation.find("--input " + (directory / "raw").string()) != std::string::npos &&
                      invocation.find("--denoise") != std::string::npos,
                  "training upload must retire the old marker and apply requested cleanup to fresh raw");
}

void check_multipart_boundaries(Checks& checks, int port, const fs::path& root) {
    const std::string boundary = "resonantia-contract-boundary";
    const std::string name = "multipart-boundary";
    const fs::path material = root / "runs" / name;
    const std::string old_audio = minimal_wav("existing-material");
    const std::string audio = minimal_wav("new-material");
    const std::string body = multipart_body({{"name", name}}, {{"new.wav", audio}});
    const auto rejected = [&](const std::string& candidate, const std::string& type,
                              int status, const std::string& description) {
        fs::remove(material / "raw/new.wav");
        fs::remove(material / "work/new.wav");
        write_file(material / "raw/old.wav", old_audio);
        write_file(material / "work/old.wav", old_audio);
        write_file(material / "steps.txt", "denoise\n");
        const HttpResponse response = request(port, "POST", "/api/material", candidate, type);
        checks.expect(response.status == status, description);
        checks.expect(read_file(material / "raw/old.wav") == old_audio &&
                          read_file(material / "work/old.wav") == old_audio &&
                          read_file(material / "steps.txt") == "denoise\n" &&
                          !fs::exists(material / "raw/new.wav") &&
                          !fs::exists(material / "work/new.wav"),
                      description + " must preserve existing material and marker");
    };
    rejected(body, std::string(kMultipartType) + "; boundary=" + boundary, 400,
             "duplicate identical boundary parameters must fail");
    rejected(body, std::string(kMultipartType) + "; BOUNDARY=other", 400,
             "duplicate conflicting boundary parameters must fail case-insensitively");
    rejected(body, "multipart/form-data; note=\"ignored; boundary=" + boundary +
                       "; ignored\"", 400,
             "boundary text inside a quoted parameter must not select a delimiter");
    rejected(body, "multipart/form-data; xboundary=" + boundary, 400,
             "a boundary parameter name suffix must not select a delimiter");
    rejected(body, "multipart/form-data", 400, "missing boundary must fail");
    rejected(body, "multipart/form-data; boundary=\"\"", 400, "empty boundary must fail");
    rejected(body, "multipart/form-data; boundary=\"" + boundary, 400,
             "unterminated quoted boundary must fail");
    rejected(body, std::string(kMultipartType) + "; note=\"unterminated", 400,
             "unterminated trailing parameter must fail");
    rejected(body, std::string(kMultipartType) + "; note=value junk", 400,
             "trailing parameter junk must fail");
    rejected(body, std::string(kMultipartType) + "; note=\"trailing\\", 400,
             "unterminated quoted escape must fail");
    rejected(body, std::string(kMultipartType) + ";", 400,
             "an empty trailing parameter must fail");
    rejected(body, "multipart/form-data; boundary=other", 400,
             "mismatched body boundary must fail");
    rejected(body.substr(0, body.size() - (boundary.size() + 6)), kMultipartType, 400,
             "missing closing delimiter must fail");
    rejected(body + "junk", kMultipartType, 400,
             "bytes after the closing delimiter must fail");
    rejected(multipart_body({{"name", name}, {"note", std::string(16385, 'x')}},
                            {{"new.wav", audio}}), kMultipartType, 413,
             "multipart fields above 16 KiB must fail");
    std::vector<std::pair<std::string, std::string>> fields{{"name", name}};
    for (int i = 0; i < 127; ++i) fields.emplace_back("note", "x");
    rejected(multipart_body(fields, {{"new.wav", audio}}), kMultipartType, 413,
             "multipart bodies above 128 parts must fail");

    // A delimiter prefix in binary WAV data is payload, not a delimiter line.
    const std::string binary_audio = minimal_wav(
        std::string("\0\xff", 2) + "\r\n--" + boundary + "-suffix\r\n--" +
        boundary + "X\r\n--" + boundary + "\rX\r\n--" + boundary +
        "--suffix\r\n--" + boundary + "--\rX\r\nend");
    fields.pop_back();
    fields.back().second.assign(16384, 'x');
    const std::vector<std::string> accepted_types{
        kMultipartType,
        "multipart/form-data; boundary=\"" + boundary + "\"",
        "multipart/form-data; note=\"ignored; boundary=decoy\"; BOUNDARY=\"" +
            boundary + "\"; charset=utf-8",
        "multipart/form-data; note=\"escaped\\\"; boundary=decoy\"; boundary=" + boundary,
    };
    for (const std::string& type : accepted_types) {
        const HttpResponse accepted = request(
            port, "POST", "/api/material",
            multipart_body(fields, {{"new.wav", binary_audio}}), type);
        checks.expect(accepted.status == 200 &&
                          read_file(material / "raw/new.wav") == binary_audio &&
                          read_file(material / "work/new.wav") == binary_audio &&
                          !fs::exists(material / "steps.txt"),
                      "128 parts, 16 KiB field and binary delimiter prefixes must preserve "
                      "exact uploaded bytes with " + type);
    }
    checks.expect(!fs::exists(root / "preprocess-invocations") &&
                      !fs::exists(root / "convert-invoked"),
                  "multipart validation and material upload must not start model CLI jobs");
}

void check_multipart_part_headers(Checks& checks, int port, const fs::path& root) {
    const std::string boundary = "resonantia-contract-boundary";
    const std::string name_header = "Content-Disposition: form-data; name=\"name\"";
    const std::string file_header =
        "Content-Disposition: form-data; name=\"audio\"; filename=\"new.wav\"";
    const std::string fresh = minimal_wav("part-header-upload");
    const auto body = [&](const std::string& name, const std::string& field_headers,
                          const std::string& file_headers) {
        return "--" + boundary + "\r\n" + field_headers + "\r\n\r\n" + name +
               "\r\n--" + boundary + "\r\n" + file_headers + "\r\n\r\n" + fresh +
               "\r\n--" + boundary + "--\r\n";
    };
    struct RejectedHeader {
        const char* label;
        bool file_part;
        std::string headers;
    };
    const std::vector<RejectedHeader> rejected{
        {"missing disposition cannot borrow name from another header", false,
         "X-Metadata: name=\"name\""},
        {"wrong disposition type", false, "Content-Disposition: attachment; name=\"name\""},
        {"name suffix is not a name parameter", false,
         "Content-Disposition: form-data; xname=\"name\""},
        {"comment text is not a name parameter", false,
         "Content-Disposition: form-data (name=\"name\")"},
        {"duplicate disposition header is ambiguous", false, name_header + "\r\n" + name_header},
        {"duplicate identical name is ambiguous", false, name_header + "; name=\"name\""},
        {"duplicate conflicting name is ambiguous", false, name_header + "; NAME=\"other\""},
        {"duplicate filename is ambiguous", true, file_header + "; FILENAME=\"other.wav\""},
        {"filename cannot stand in for missing name", true,
         "Content-Disposition: form-data; filename=\"new.wav\""},
        {"unrelated header cannot supply a filename", true,
         "X-Metadata: filename=\"new.wav\"\r\nContent-Disposition: form-data; name=\"audio\""},
        {"header missing colon", false, "Content-Disposition form-data; name=\"name\""},
        {"whitespace inside header name", false, "Content-Disposition : form-data; name=\"name\""},
        {"orphan continuation line", false, " " + name_header},
        {"bare LF in header block", false, name_header + "\nX-Metadata: value"},
        {"bare CR in header block", false, name_header + "\rX-Metadata: value"},
        {"NUL in header block", false, name_header + "\r\nX-Metadata: " + std::string(1, '\0')},
        {"DEL in header block", false, name_header + "\r\nX-Metadata: " + std::string(1, '\x7f')},
        {"unterminated quoted parameter", false, name_header + "; note=\"unfinished"},
        {"unterminated quoted escape", false, name_header + "; note=\"unfinished\\"},
        {"trailing parameter junk", false, name_header + " junk"},
        {"empty trailing parameter", false, name_header + ";"},
        {"unterminated comment", false, name_header + " (unfinished"},
        {"filename star is prohibited for form-data", true,
         file_header + "; filename*=UTF-8''other.wav"},
    };
    const std::string cli_before = read_file(root / "preprocess-invocations");
    for (std::size_t i = 0; i < rejected.size(); ++i) {
        const auto& test = rejected[i];
        const std::string name = "part-header-reject-" + std::to_string(i);
        const fs::path directory = root / "runs" / name;
        const std::string old_raw = minimal_wav("old-raw"), old_work = minimal_wav("old-work");
        write_file(directory / "raw/old.wav", old_raw);
        write_file(directory / "work/old.wav", old_work);
        write_file(directory / "steps.txt", "denoise\n");
        const HttpResponse response = request(
            port, "POST", "/api/material",
            body(name, test.file_part ? name_header : test.headers,
                 test.file_part ? test.headers : file_header), kMultipartType);
        checks.expect(response.status == 400 &&
                          read_file(directory / "raw/old.wav") == old_raw &&
                          read_file(directory / "work/old.wav") == old_work &&
                          read_file(directory / "steps.txt") == "denoise\n" &&
                          !fs::exists(directory / "raw/new.wav") &&
                          !fs::exists(directory / "work/new.wav") &&
                          !has_material_staging(directory),
                      std::string(test.label) + " must return 400 and preserve raw/work/marker");
    }
    const std::vector<std::pair<std::string, std::string>> accepted{
        {name_header, file_header},
        {"cOnTeNt-DiSpOsItIoN: FORM-DATA; NAME = name",
         "Content-Disposition: form-data; name=audio; filename=new.wav"},
        {"Content-Disposition: form-data; name=\"na\\me\"",
         "Content-Disposition: form-data; filename=\"ne\\w.wav\"; name=\"audio\""},
        {"Content-Disposition: form-data;\r\n\tname = \"name\"", file_header},
        {"Content-Disposition: form-data (name=\"decoy\" (nested)); name=(ignored)\"name\"",
         file_header + " (escaped\\) comment)"},
        {"X-Metadata: name=\"decoy\"\r\n" + name_header,
         "X-Metadata: filename=\"decoy.wav\"\r\n" + file_header},
        {"Content-Disposition: form-data; xname=\"decoy\"; name=\"name\"",
         file_header + "; note=\"filename=\\\"decoy.wav\\\"\""},
        {name_header + "\r\nX-Trace: ordinary extension", file_header + "\r\nContent-Type: audio/wav"},
    };
    for (std::size_t i = 0; i < accepted.size(); ++i) {
        const std::string name = "part-header-accept-" + std::to_string(i);
        const HttpResponse response = request(
            port, "POST", "/api/material", body(name, accepted[i].first, accepted[i].second),
            kMultipartType);
        checks.expect(response.status == 200 &&
                          read_file(root / "runs" / name / "raw/new.wav") == fresh &&
                          read_file(root / "runs" / name / "work/new.wav") == fresh,
                      "valid disposition syntax and ignored metadata must preserve values and bytes, case " +
                          std::to_string(i));
    }
    const std::string repeated_name = "part-header-repeated-fields";
    const HttpResponse repeated = request(
        port, "POST", "/api/material",
        multipart_body({{"name", repeated_name}, {"", "empty field name"}, {"note", "one"}, {"note", "two"}},
                       {{"one.wav", fresh}, {"two.wav", fresh}}), kMultipartType);
    checks.expect(repeated.status == 200 &&
                      read_file(root / "runs" / repeated_name / "raw/one.wav") == fresh &&
                      read_file(root / "runs" / repeated_name / "raw/two.wav") == fresh,
                  "repeated names across parts and an explicit empty field name must remain valid");
    checks.expect(read_file(root / "preprocess-invocations") == cli_before,
                  "part header parsing must not start a model CLI");
}

void check_material_marker_failure(Checks& checks, int port, const fs::path& root) {
    for (const std::string route : {"material", "train"}) {
        const std::string name = "material-marker-" + route;
        const fs::path directory = root / "runs" / name;
        const std::string old_raw = minimal_wav("raw-before-failed-upload");
        const std::string old_work = minimal_wav("work-before-failed-upload");
        write_file(directory / "raw/old.wav", old_raw);
        write_file(directory / "work/old.wav", old_work);
        // An invalid marker must not be silently ignored, or recursively deleted.
        write_file(directory / "steps.txt/preserve.txt", "preserve-invalid-marker");
        const std::string invocations = read_file(root / "preprocess-invocations");
        const HttpResponse response = request(
            port, "POST", "/api/" + route,
            multipart_body({{"name", name}, {"steps", "1"}},
                           {{"new.wav", minimal_wav("new-upload")}}), kMultipartType);
        // Keep a failing pre-fix run from leaving its fake worker in flight.
        if (route == "train" && response.status == 200) wait_for_job_completion(port, name);
        checks.expect(response.status == 500,
                      route + " upload must fail when its existing step marker is not a regular file");
        checks.expect(read_file(directory / "raw/old.wav") == old_raw &&
                          read_file(directory / "work/old.wav") == old_work &&
                          read_file(directory / "steps.txt/preserve.txt") == "preserve-invalid-marker" &&
                          !fs::exists(directory / "raw/new.wav") &&
                          !fs::exists(directory / "work/new.wav") &&
                          !has_material_staging(directory),
                      route + " marker failure must preserve old material and clean only request staging");
        checks.expect(read_file(root / "preprocess-invocations") == invocations,
                      route + " marker failure must not launch the preprocessing CLI");
        if (route == "train") {
            const HttpResponse status = request(port, "GET", "/api/train/status?name=" + name);
            checks.expect(status.body.find("\"running\":false") != std::string::npos &&
                              status.body.find("\"error\":true") != std::string::npos,
                          "failed training upload must release its job reservation and report error");
        }
    }
}

void check_material_preview_race(Checks& checks, int port) {
    const std::string name = "material-preview-race";
    const std::string first = minimal_wav(std::string(64 * 1024, 'a'));
    const std::string second = minimal_wav(std::string(64 * 1024, 'b'));
    const auto upload = [&](int generation) {
        return request(port, "POST", "/api/material",
                       multipart_body({{"name", name}},
                           {{generation % 2 == 0 ? "first.wav" : "second.wav",
                             generation % 2 == 0 ? first : second}}), kMultipartType);
    };
    if (upload(0).status != 200) {
        checks.expect(false, "preview race fixture must publish its initial generation");
        return;
    }
    // Two clients only: bounded traffic, no model execution and no timing sleeps.
    auto publisher = std::async(std::launch::async, [&] {
        int failures = 0;
        for (int generation = 1; generation <= 256; ++generation)
            if (upload(generation).status != 200) ++failures;
        return failures;
    });
    int preview_failures = 0;
    for (int i = 0; i < 1024; ++i) {
        const HttpResponse preview = request(port, "GET", "/api/preview?name=" + name);
        if (preview.status != 200 || (preview.body != first && preview.body != second))
            ++preview_failures;
    }
    const int publish_failures = publisher.get();
    std::cout << "Material/preview concurrent check: publish failures=" << publish_failures
              << ", preview failures=" << preview_failures << '\n';
    checks.expect(publish_failures == 0 && preview_failures == 0,
                  "concurrent preview must return one complete generation while material is replaced");
}

void check_slow_preview_publication(Checks& checks, int port, const fs::path& root) {
    const std::string name = "slow-preview-publication";
    const fs::path work = root / "runs" / name / "work/large.wav";
    write_file(work, minimal_wav("slow-preview"));
    fs::resize_file(work, 8U * 1024U * 1024U);
    struct SocketGuard {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        ~SocketGuard() { if (fd >= 0) close(fd); }
    } client;
    if (client.fd < 0) throw std::runtime_error("slow preview socket failed");
    const int small_buffer = 4096;
    const timeval timeout{3, 0};
    if (setsockopt(client.fd, SOL_SOCKET, SO_RCVBUF, &small_buffer, sizeof(small_buffer)) != 0 ||
        setsockopt(client.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0)
        throw std::runtime_error("slow preview socket options failed");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    if (connect(client.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        !send_all(client.fd, "GET /api/preview?name=" + name +
                     " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"))
        throw std::runtime_error("slow preview request failed");
    std::string headers;
    while (headers.size() < 4096 && headers.find("\r\n\r\n") == std::string::npos) {
        char byte;
        if (recv(client.fd, &byte, 1, 0) != 1)
            throw std::runtime_error("slow preview response header failed");
        headers.push_back(byte);
    }
    checks.expect(headers.find("HTTP/1.1 200 ") == 0,
                  "slow preview fixture must begin a successful response");
    // Stop reading an 8 MiB body with a small receive window. Publication must
    // finish before this socket is closed, proving socket writes hold no material lock.
    auto publication = std::async(std::launch::async, [&] {
        return request(port, "POST", "/api/material",
                       multipart_body({{"name", name}}, {{"fresh.wav", minimal_wav("fresh")}}),
                       kMultipartType);
    });
    const bool ready = publication.wait_for(2s) == std::future_status::ready;
    shutdown(client.fd, SHUT_RDWR);
    close(client.fd);
    client.fd = -1;
    const HttpResponse published = publication.get();
    checks.expect(ready && published.status == 200,
                  "slow preview response must not retain the material publication lock");
}

// Bounded raw HTTP client for incomplete uploads. The Content-Length reservation
// can be large while the actual transmitted body remains only a few bytes.
class IncompleteUploadClient {
public:
    explicit IncompleteUploadClient(int port) {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) throw std::runtime_error("resource test socket failed");
        const timeval timeout{2, 0};
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port);
        if (setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) ||
            setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) ||
            connect(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address))) {
            close(fd_);
            throw std::runtime_error("resource test socket setup failed");
        }
    }
    IncompleteUploadClient(const IncompleteUploadClient&) = delete;
    IncompleteUploadClient& operator=(const IncompleteUploadClient&) = delete;
    ~IncompleteUploadClient() { if (fd_ >= 0) close(fd_); }

    void start(std::size_t declared_length, const std::string& body = {}) {
        const std::string wire =
            "POST /api/train HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: " +
            std::to_string(declared_length) + "\r\nContent-Type: " + kMultipartType +
            "\r\nConnection: close\r\n\r\n" + body;
        if (!send_all(fd_, wire)) throw std::runtime_error("resource test send failed");
    }
    void finish_write() {
        if (shutdown(fd_, SHUT_WR)) throw std::runtime_error("resource test EOF failed");
    }
    void reset() {
        const linger option{1, 0};
        if (setsockopt(fd_, SOL_SOCKET, SO_LINGER, &option, sizeof(option)))
            throw std::runtime_error("resource test reset setup failed");
        close(fd_);
        fd_ = -1;
    }
    int response_status() {
        std::string response;
        char buffer[4096];
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        for (;;) {
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("resource response deadline exceeded");
            const ssize_t size = recv(fd_, buffer, sizeof(buffer), 0);
            if (size < 0 && errno == EINTR) continue;
            if (size < 0) throw std::runtime_error("resource response read failed");
            if (size == 0) break;
            response.append(buffer, static_cast<std::size_t>(size));
            if (response.size() > 16 * 1024)
                throw std::runtime_error("resource test response exceeded bound");
        }
        if (response.rfind("HTTP/1.1 ", 0) || response.find("\r\n\r\n") == std::string::npos)
            throw std::runtime_error("resource test malformed response");
        return std::stoi(response.substr(9, 3));
    }
private:
    int fd_ = -1;
};

int server_socket_count(pid_t pid) {
    std::error_code error;
    int count = 0;
    const fs::path directory = fs::path("/proc") / std::to_string(pid) / "fd";
    for (fs::directory_iterator entry(directory, error), end; !error && entry != end;
         entry.increment(error)) {
        std::error_code link_error;
        const fs::path target = fs::read_symlink(entry->path(), link_error);
        if (link_error == std::errc::no_such_file_or_directory) continue;
        if (link_error) return -1;
        if (target.string().rfind("socket:[", 0) == 0) ++count;
    }
    return error ? -1 : count;
}

void check_header_deadline(Checks& checks, const ServerProcess& server,
                           const fs::path& root) {
    const auto case_start = std::chrono::steady_clock::now();
    const std::string initial_job = request(server.port(), "GET", "/api/train/status", {}, {}, {}, 1).body;
    const std::string cli_before = read_file(root / "preprocess-invocations");
    struct Socket {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        ~Socket() { if (fd >= 0) close(fd); }
    } client;
    if (client.fd < 0) throw std::runtime_error("header deadline socket failed");
    const timeval send_timeout{1, 0};
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(server.port());
    if (setsockopt(client.fd, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout)) ||
        connect(client.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)))
        throw std::runtime_error("header deadline connection failed");

    const auto start = std::chrono::steady_clock::now();
    const auto response_limit = start + 13s;
    const auto wait_readable_until = [&](std::chrono::steady_clock::time_point deadline) {
        for (;;) {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now());
            if (remaining <= 0ms) return false;
            pollfd event{client.fd, POLLIN, 0};
            const int ready = poll(&event, 1, static_cast<int>(remaining.count()) + 1);
            if (ready < 0 && errno == EINTR) continue;
            if (ready < 0) throw std::runtime_error("header deadline poll failed");
            if (ready > 0) return true;
        }
    };
    const std::string prefix = "POST /api/train HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Drip: ";
    if (!send_all(client.fd, prefix)) throw std::runtime_error("header prefix send failed");
    std::size_t sent = prefix.size();
    bool early_response = false;
    for (const auto offset : {3s, 6s}) {
        if (wait_readable_until(start + offset)) {
            early_response = true;
            break;
        }
        if (!send_all(client.fd, "x")) throw std::runtime_error("header drip send failed");
        ++sent;
    }
    // Never finish the headers, half-close, or reset: only the server's real
    // 10-second deadline may finish this request. Late bytes must not extend it.
    std::string wire;
    bool peer_closed = false;
    char buffer[1024];
    while (wait_readable_until(response_limit)) {
        const ssize_t count = recv(client.fd, buffer, sizeof(buffer), MSG_DONTWAIT);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (count < 0) throw std::runtime_error("header deadline receive failed");
        if (count == 0) { peer_closed = true; break; }
        wire.append(buffer, static_cast<std::size_t>(count));
        if (wire.size() > 4096) throw std::runtime_error("header response exceeded test bound");
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    checks.expect(!early_response && elapsed >= 9s && elapsed < 13s && peer_closed &&
                      wire.rfind("HTTP/1.1 400 ", 0) == 0 &&
                      wire.find("\r\n\r\n") != std::string::npos,
                  "header deadline must return 400 and close at 10s despite 3s/6s drips");
    if (!peer_closed) throw std::runtime_error("header deadline gate did not close by 13s");
    checks.expect(sent <= 4096 && wait_for([&] {
                      return server_socket_count(server.pid()) == 1;
                  }, 1s), "header timeout must close its accepted socket");
    checks.expect(request(server.port(), "GET", "/api/train/status", {}, {}, {}, 1).body == initial_job &&
                      read_file(root / "preprocess-invocations") == cli_before,
                  "header timeout must leave service usable, job unchanged and CLI unstarted");
    checks.expect(std::chrono::steady_clock::now() - case_start < 15s,
                  "header deadline case must finish within 15s");
    std::cout << "Header deadline: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
              << " ms, sent=" << sent << " bytes, peer_closed=" << peer_closed
              << ", max clients=2; unchanged 10s production deadline\n";
}

void check_incomplete_upload_resources(Checks& checks, const ServerProcess& server,
                                        const fs::path& root) {
    using namespace resonantia::web;
    const auto require = [&](bool condition, const std::string& message) {
        checks.expect(condition, message);
        if (!condition) throw std::runtime_error("incomplete upload resource gate failed");
    };
    const auto probe = [&] {
        IncompleteUploadClient client(server.port());
        client.start(1, "x"); // Complete but invalid multipart: 400 if admitted.
        client.finish_write();
        return client.response_status();
    };
    const auto idle_sockets = [&] {
        return wait_for([&] { return server_socket_count(server.pid()) == 1; }, 2s);
    };
    const fs::path directory = root / "runs/incomplete-upload";
    write_file(directory / "raw/old.wav", minimal_wav("raw-before-truncation"));
    write_file(directory / "work/old.wav", minimal_wav("work-before-truncation"));
    write_file(directory / "steps.txt", "denoise\n");
    // The MIME body itself is valid; only the declared HTTP length is incomplete.
    // Ignoring truncation would therefore start a real job against this fixture.
    const std::string partial_body = multipart_body({{"name", "incomplete-upload"}, {"steps", "1"}});
    const std::string cli_before = read_file(root / "preprocess-invocations");
    const std::string initial_job = request(server.port(), "GET", "/api/train/status").body;
    require(idle_sockets(), "only the listener should remain before resource checks");

    // Boundary rejection must happen from headers alone, without waiting for or
    // allocating a 128 MiB payload. Exact-limit admission is exercised below.
    {
        IncompleteUploadClient over_limit(server.port());
        over_limit.start(kMaxBodyBytes + 1);
        require(over_limit.response_status() == 413,
                "declared body above 128 MiB must fail before its body is sent");
    }
    for (const bool reset : {false, true}) {
        IncompleteUploadClient first(server.port()), second(server.port());
        first.start(kMaxBodyBytes, partial_body);
        second.start(kMaxBodyBytes, partial_body);
        require(wait_for([&] { return probe() == 503; }, 2s),
                "two 128 MiB declarations must exhaust the 256 MiB reservation budget");
        require(request(server.port(), "GET", "/api/train/status").body == initial_job,
                "incomplete uploads must not create or modify a job");
        if (reset) {
            first.reset();
        } else {
            first.finish_write();
            require(first.response_status() == 400, "truncated body EOF must return 400");
        }
        require(wait_for([&] { return probe() == 400; }, 2s),
                reset ? "RST must release the request body reservation"
                      : "EOF must release the request body reservation");
        {
            IncompleteUploadClient replacement(server.port());
            replacement.start(kMaxBodyBytes, partial_body);
            require(wait_for([&] { return probe() == 503; }, 2s),
                    "released capacity must admit another full 128 MiB reservation");
            replacement.finish_write();
            require(replacement.response_status() == 400,
                    "replacement must have been admitted, then fail only on truncation");
        }
        second.finish_write();
        require(second.response_status() == 400,
                "remaining exact-limit incomplete upload must return 400");
        require(idle_sockets(), "EOF/RST must close all accepted sockets");
    }

    // More than the connection cap, serially: a leaked connection counter would
    // eventually turn these 400s into 503s, without opening 32 sockets at once.
    for (int i = 0; i <= kMaxConnections; ++i) {
        IncompleteUploadClient client(server.port());
        client.start(1);
        client.finish_write();
        require(client.response_status() == 400,
                "serial truncated uploads must return their connection slots");
    }
    require(idle_sockets(), "serial truncation must leave only the listener socket");
    require(request(server.port(), "GET", "/api/train/status").body == initial_job,
            "all rejected/truncated uploads must leave job state unchanged");
    require(read_file(directory / "raw/old.wav") == minimal_wav("raw-before-truncation") &&
            read_file(directory / "work/old.wav") == minimal_wav("work-before-truncation") &&
            read_file(directory / "steps.txt") == "denoise\n" &&
            !has_hidden_training_directory(directory) && !has_material_staging(directory),
            "incomplete uploads must preserve old material and create no training staging");
    require(read_file(root / "preprocess-invocations") == cli_before,
            "incomplete upload resource checks must not start the training CLI chain");
    std::cout << "Incomplete upload resources: 128 MiB declaration +1 rejected; "
              << "256 MiB reservation full/recovered after EOF and RST; "
              << "33 serial truncated connections recovered; no job/material changes; "
              << "max actual body=" << partial_body.size() << " bytes, max clients=3\n";
}

void run_body_deadline_contract(Checks& checks, const fs::path& server,
                                const fs::path& root) {
    const auto case_start = std::chrono::steady_clock::now();
    const auto require = [&](bool ok, const std::string& message) {
        checks.expect(ok, message);
        if (!ok) throw std::runtime_error("body deadline gate: " + message);
    };
    std::size_t all_request_bytes = 0;
    ServerProcess process(server, root, root / "hold.trigger", root / "hold.started",
                          root / "hold.release", root / "fail.trigger", {},
                          &all_request_bytes);
    const fs::path material = root / "runs/body-deadline";
    write_file(material / "raw/old.wav", minimal_wav("raw-before-timeout"));
    write_file(material / "work/old.wav", minimal_wav("work-before-timeout"));
    write_file(material / "steps.txt", "denoise\n");
    const auto initial = request(process.port(), "GET", "/api/train/status",
                                 {}, {}, {}, 1, &all_request_bytes);
    require(initial.status == 200, "initial status must be available");
    const std::string cli_before = read_file(root / "preprocess-invocations");
    struct Socket {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        ~Socket() { if (fd >= 0) close(fd); }
    } client;
    require(client.fd >= 0, "socket creation failed");
    const timeval send_timeout{1, 0};
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(process.port());
    require(setsockopt(client.fd, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout)) == 0 &&
                connect(client.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
            "connection failed");
    constexpr std::size_t declared_body_bytes = 1024;
    // Complete MIME content makes accidental early dispatch observable. The
    // HTTP body remains incomplete throughout; no real material or CLI is used.
    const std::string body = multipart_body({{"name", "body-deadline"}, {"steps", "1"}});
    const std::string headers = "POST /api/train HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        "Content-Type: " + std::string(kMultipartType) + "\r\nContent-Length: " +
        std::to_string(declared_body_bytes) + "\r\nConnection: close\r\n\r\n";
    const auto send_bounded = [&](const std::string& bytes) {
        require(all_request_bytes + bytes.size() <= 4096, "total input exceeded 4KiB");
        require(send_all(client.fd, bytes), "body drip send failed");
        all_request_bytes += bytes.size();
    };
    require(body.size() + 5 < declared_body_bytes, "fixture would complete the body");
    const auto started = std::chrono::steady_clock::now();
    send_bounded(headers + body);
    std::size_t actual_body_bytes = body.size();
    const auto wait_readable_until = [&](std::chrono::steady_clock::time_point deadline) {
        for (;;) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now());
            if (left <= 0ms) return false;
            pollfd event{client.fd, POLLIN, 0};
            const int ready = poll(&event, 1, static_cast<int>(left.count()) + 1);
            if (ready < 0 && errno == EINTR) continue;
            require(ready >= 0, "response poll failed");
            if (ready > 0) return true;
        }
    };
    int drips = 0;
    for (const auto offset : {20s, 40s, 60s, 80s, 100s}) {
        require(!wait_readable_until(started + offset), "response arrived before the body deadline");
        send_bounded(" ");
        ++actual_body_bytes;
        ++drips;
        std::cout << "Body drip at " << offset.count() << "s; declared=" << declared_body_bytes
                  << ", actual body=" << actual_body_bytes << " bytes\n" << std::flush;
    }
    // Do not complete the body, half-close, or reset. Only the real server
    // deadline may produce the response and EOF. Stop observing by 124s.
    std::string wire;
    bool peer_closed = false;
    char buffer[1024];
    while (wait_readable_until(started + 124s)) {
        const ssize_t count = recv(client.fd, buffer, sizeof(buffer), MSG_DONTWAIT);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        require(count >= 0, "response read failed");
        if (count == 0) { peer_closed = true; break; }
        wire.append(buffer, static_cast<std::size_t>(count));
        require(wire.size() <= 4096, "response exceeded 4KiB");
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    require(peer_closed && elapsed >= 118s && elapsed < 124s && drips == 5 &&
                wire.rfind("HTTP/1.1 400 ", 0) == 0 && wire.find("\r\n\r\n") != std::string::npos,
            "120s absolute deadline must return 400/EOF despite periodic body bytes");
    require(wait_for([&] { return server_socket_count(process.pid()) == 1; }, 1s),
            "accepted socket must be released after timeout");
    const auto after = request(process.port(), "GET", "/api/train/status",
                               {}, {}, {}, 1, &all_request_bytes);
    require(after.status == 200 && after.body == initial.body &&
                read_file(root / "preprocess-invocations") == cli_before &&
                read_file(material / "raw/old.wav") == minimal_wav("raw-before-timeout") &&
                read_file(material / "work/old.wav") == minimal_wav("work-before-timeout") &&
                read_file(material / "steps.txt") == "denoise\n" &&
                !has_hidden_training_directory(material) && !has_material_staging(material),
            "service must remain usable; no job, material, CLI or staging changes");
    process.stop();  // Reap only our own temporary listener before reporting success.
    require(std::chrono::steady_clock::now() - case_start < 130s,
            "body deadline case exceeded 130s");
    std::cout << "Body deadline: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
              << " ms; declared body=" << declared_body_bytes
              << " bytes; actual body=" << actual_body_bytes
              << " bytes; all HTTP request bytes including readiness/status=" << all_request_bytes
              << "; 5 drips, 400/EOF, accepted socket closed, next status=200, no job/material changes; "
              << "max clients=2, own listener reaped\n";
}

void run_open_server_contracts(Checks& checks, const fs::path& server,
                               const fs::path& root) {
    const fs::path hold_trigger = root / "hold.trigger";
    const fs::path hold_started = root / "hold.started";
    const fs::path hold_release = root / "hold.release";
    const fs::path fail_trigger = root / "fail-step.trigger";
    ServerProcess process(server, root, hold_trigger, hold_started, hold_release,
                          fail_trigger);

    HttpResponse static_root = request(process.port(), "GET", "/");
    checks.expect(static_root.status == 200 && static_root.body == "contract-root",
                  "static root should be publicly readable");
    checks.expect(request(process.port(), "GET", "/api/voices").status == 200,
                  "default API mode should remain open");

    checks.expect(request(process.port(), "GET", "/api/tasks", {}, {},
                          "Range: bytes=1-2\r\nRange: bytes=3-4\r\n").status == 400,
                  "duplicate Range headers must fail before routing");
    const auto cancel_method = request(process.port(), "GET", "/api/train/cancel");
    checks.expect(cancel_method.status == 405 && cancel_method.has_header("Allow: POST\r\n"),
                  "cancel endpoint must reject GET with Allow POST");
    const auto initial_job = request(process.port(), "GET", "/api/train/status").body;
    const auto rename_method=request(process.port(),"GET","/api/task/rename");
    checks.expect(rename_method.status==405&&rename_method.has_header("Allow: POST\r\n"),"rename method contract");
    for(const auto& fields:std::vector<std::vector<std::pair<std::string,std::string>>>{
        {{"id","missing-task"},{"name","标题"},{"name","重复"}},
        {{"id","missing-task"},{"name","标题"}},
        {{"id","missing-task"},{"name","标题"},{"expected_name","旧"},{"extra","unsupported"}}})
        checks.expect(request(process.port(),"POST","/api/task/rename",multipart_body(fields),kMultipartType).status==400,"rename requires exactly three unique fields");
    checks.expect(request(process.port(), "POST", "/api/train/cancel",
                          multipart_body({{"job_id", "unknown"}}), kMultipartType).status == 404,
                  "cancel without an admitted job must return 404");
    for (const auto& fields : std::vector<std::vector<std::pair<std::string, std::string>>>{
             {}, {{"job_id", ""}}, {{"job_id", "bad/id"}},
             {{"job_id", "one"}, {"job_id", "two"}}, {{"name", "voice"}},
             {{"job_id", "one"}, {"extra", "value"}}}) {
        checks.expect(request(process.port(), "POST", "/api/train/cancel",
                              multipart_body(fields), kMultipartType).status == 400,
                      "cancel must accept exactly one valid job_id field");
    }
    checks.expect(request(process.port(), "POST", "/api/train/cancel",
                          multipart_body({}, {{"job.wav", minimal_wav("cancel-file")}}),
                          kMultipartType).status == 400,
                  "cancel endpoint must reject files");
    checks.expect(request(process.port(), "GET", "/api/train/status").body == initial_job,
                  "invalid cancellation must not alter job state");

    check_header_deadline(checks, process, root);
    check_incomplete_upload_resources(checks, process, root);

    check_multipart_boundaries(checks, process.port(), root);
    check_multipart_part_headers(checks, process.port(), root);
    check_material_marker_failure(checks, process.port(), root);
    check_material_preview_race(checks, process.port());
    check_slow_preview_publication(checks, process.port(), root);

    HttpResponse wrong_get = request(process.port(), "POST", "/api/voices");
    checks.expect(wrong_get.status == 405,
                  "known GET API called with POST should return 405");
    checks.expect(wrong_get.has_header("Allow: GET\r\n"),
                  "405 for GET API should advertise Allow: GET");
    HttpResponse wrong_post = request(process.port(), "GET", "/api/material");
    checks.expect(wrong_post.status == 405,
                  "known POST API called with GET should return 405");
    checks.expect(wrong_post.has_header("Allow: POST\r\n"),
                  "405 for POST API should advertise Allow: POST");
    checks.expect(request(process.port(), "GET", "/api/not-a-route").status == 404,
                  "unknown API path should remain 404");

    write_file(root / "runs/invalid-bool-train/work/input.wav",
               minimal_wav("invalid-bool-train"));
    const HttpResponse invalid_bool_train = request(
        process.port(), "POST", "/api/train",
        multipart_body({{"name", "invalid-bool-train"},
                        {"separate", "garbage"}}),
        kMultipartType);
    checks.expect(invalid_bool_train.status == 400,
                  "training boolean fields must reject values other than 0 or 1");
    if (invalid_bool_train.status != 400) {
        (void)wait_for_job_completion(process.port(), "invalid-bool-train");
    }
    checks.expect(
        request(process.port(), "POST", "/api/preprocess",
                multipart_body({{"pp_denoise", ""}},
                               {{"input.wav", minimal_wav("invalid-bool-preprocess")}}),
                kMultipartType)
                .status == 400,
        "preprocess boolean fields must reject an explicit empty value");
    fs::remove(root / "convert-invoked");
    checks.expect(
        request(process.port(), "POST", "/api/convert",
                multipart_body({{"pp_denoise", "garbage"}},
                               {{"input.wav", minimal_wav("invalid-bool-convert")}}),
                kMultipartType)
                .status == 400 &&
            !fs::exists(root / "convert-invoked"),
        "convert boolean fields must fail before launching a CLI");

    const fs::path busy_work = root / "runs/busy/work";
    write_file(busy_work / "original.wav", minimal_wav("busy-original"));
    write_file(hold_trigger, "1");
    HttpResponse preprocess_response;
    std::exception_ptr preprocess_error;
    std::thread preprocess_client([&] {
        try {
            std::string body = multipart_body({{"pp_denoise", "1"}},
                                              {{"input.wav", minimal_wav("busy")}});
            preprocess_response = request(process.port(), "POST", "/api/preprocess",
                                          body, kMultipartType);
        } catch (...) {
            preprocess_error = std::current_exception();
        }
    });
    bool preprocess_started = wait_for([&] { return fs::exists(hold_started); });
    checks.expect(preprocess_started,
                  "fake preprocess should acquire GPU before busy check");
    pid_t spawned_child = -1;
    bool child_pid_ready = wait_for([&] {
        std::ifstream input(root / "spawned-child.pid");
        return static_cast<bool>(input >> spawned_child) && spawned_child > 0;
    });
    checks.expect(child_pid_ready && !process_has_socket_fd(spawned_child),
                  "spawned CLI must not inherit listener or HTTP client sockets");
    std::string step_body = multipart_body({{"name", "busy"}, {"step", "denoise"}});
    HttpResponse busy_step = request(process.port(), "POST", "/api/step",
                                     step_body, kMultipartType);
    checks.expect(busy_step.status == 409,
                  "step should reject an already-held GPU immediately with 409");
    write_file(hold_release, "1");
    preprocess_client.join();
    checks.expect(!preprocess_error && preprocess_response.status == 200,
                  "the GPU holder should complete after release");
    wait_for([&] {
        HttpResponse status = request(process.port(), "GET", "/api/train/status");
        return status.body.find("\"running\":false") != std::string::npos;
    });

    const fs::path rollback_dir = root / "runs/rollback";
    const std::string original = minimal_wav("must-survive");
    write_file(rollback_dir / "work/original.wav", original);
    write_file(fail_trigger, "1");
    std::string rollback_body =
        multipart_body({{"name", "rollback"}, {"step", "denoise"}});
    HttpResponse rollback_start = request(process.port(), "POST", "/api/step",
                                          rollback_body, kMultipartType);
    checks.expect(rollback_start.status == 200,
                  "step worker should start before its publish failure");
    bool rollback_finished = wait_for([&] {
        HttpResponse status = request(process.port(), "GET",
                                      "/api/train/status?name=rollback");
        return status.body.find("\"running\":false") != std::string::npos &&
               status.body.find("\"error\":true") != std::string::npos;
    });
    checks.expect(rollback_finished,
                  "failed step publish should terminate and report an error");
    chmod(rollback_dir.c_str(), 0755);
    checks.expect(fs::exists(rollback_dir / "work/original.wav") &&
                      read_file(rollback_dir / "work/original.wav") == original,
                  "failed step publish must preserve the previous live work tree");
    std::error_code ignored;
    fs::remove(fail_trigger, ignored);

    const fs::path success_dir = root / "runs/success";
    write_file(success_dir / "work/original.wav", minimal_wav("replace-me"));
    std::string success_body =
        multipart_body({{"name", "success"}, {"step", "denoise"}});
    HttpResponse success_start = request(process.port(), "POST", "/api/step",
                                         success_body, kMultipartType);
    checks.expect(success_start.status == 200, "successful step should be accepted");
    bool success_finished = wait_for([&] {
        HttpResponse status = request(process.port(), "GET",
                                      "/api/train/status?name=success");
        return status.body.find("\"running\":false") != std::string::npos &&
               status.body.find("\"stage\":\"step-done\"") != std::string::npos &&
               status.body.find("\"error\":false") != std::string::npos;
    });
    checks.expect(success_finished, "successful step should finish cleanly");
    checks.expect(fs::exists(success_dir / "work/result.wav"),
                  "successful step should publish the staged work tree");
    bool hidden_transaction_dir = false;
    for (const auto& entry : fs::directory_iterator(success_dir)) {
        std::string name = entry.path().filename().string();
        if (name.rfind(".step_stage_", 0) == 0 ||
            name.rfind(".step_backup_", 0) == 0)
            hidden_transaction_dir = true;
    }
    checks.expect(!hidden_transaction_dir,
                  "successful step publish should clean staging and backup directories");
    checks.expect(read_file(success_dir / "steps.txt") == "denoise\n",
                  "successful step publish must commit its completion marker");

    const fs::path marker_failure_dir = root / "runs/marker-write-failure";
    const std::string marker_failure_original = minimal_wav("marker-must-survive");
    write_file(marker_failure_dir / "work/original.wav", marker_failure_original);
    fs::create_symlink("/dev/full", marker_failure_dir / "steps.txt");
    const HttpResponse marker_failure_start = request(
        process.port(), "POST", "/api/step",
        multipart_body({{"name", "marker-write-failure"}, {"step", "denoise"}}),
        kMultipartType);
    checks.expect(marker_failure_start.status == 200,
                  "step with an invalid marker target should start before staging fails");
    const HttpResponse marker_failure_status =
        wait_for_job_completion(process.port(), "marker-write-failure");
    checks.expect(
        marker_failure_status.body.find("\"stage\":\"error\"") !=
                std::string::npos &&
            marker_failure_status.body.find("\"error\":true") != std::string::npos &&
            read_file(marker_failure_dir / "work/original.wav") ==
                marker_failure_original &&
            fs::is_symlink(marker_failure_dir / "steps.txt"),
        "step marker failure must preserve the previous work and marker state");

    write_file(root / "preprocess-cli-fail.trigger", "1");
    const std::string preprocess_failure_body =
        multipart_body({{"pp_denoise", "1"}},
                       {{"input.wav", minimal_wav("preprocess-failure")}});
    const HttpResponse preprocess_failure =
        request(process.port(), "POST", "/api/preprocess",
                preprocess_failure_body, kMultipartType);
    checks.expect(preprocess_failure.status == 500,
                  "preprocess must propagate a non-zero CLI exit even when output exists");
    fs::remove(root / "preprocess-cli-fail.trigger", ignored);

    write_file(root / "preprocess-empty-output.trigger", "1");
    const HttpResponse preprocess_empty =
        request(process.port(), "POST", "/api/preprocess",
                multipart_body({{"pp_denoise", "1"}},
                               {{"input.wav", minimal_wav("empty-output")}}),
                kMultipartType);
    checks.expect(preprocess_empty.status == 500,
                  "preprocess success with an empty WAV must return an error");
    fs::remove(root / "preprocess-empty-output.trigger", ignored);

    write_file(root / "preprocess-oversized-output.trigger", "1");
    const HttpResponse preprocess_oversized = request(
        process.port(), "POST", "/api/preprocess",
        multipart_body({{"pp_denoise", "1"}},
                       {{"input.wav", minimal_wav("oversized-output")}}),
        kMultipartType);
    checks.expect(preprocess_oversized.status == 413,
                  "preprocess output above the buffered response cap must return 413");
    fs::remove(root / "preprocess-oversized-output.trigger", ignored);

    const fs::path oversized_preview = root / "runs/oversized-preview/work/large.wav";
    write_file(oversized_preview, minimal_wav("oversized-preview"));
    fs::resize_file(oversized_preview, 64U * 1024U * 1024U + 1U);
    checks.expect(request(process.port(), "GET",
                          "/api/preview?name=oversized-preview").status == 413,
                  "preview above the buffered response cap must return 413");

    const std::string unknown_voice_body = multipart_body(
        {{"voice", "not-installed"}},
        {{"input.wav", minimal_wav("unknown-voice")}});
    fs::remove(root / "convert-invoked", ignored);
    const HttpResponse unknown_voice =
        request(process.port(), "POST", "/api/convert", unknown_voice_body,
                kMultipartType);
    checks.expect(unknown_voice.status == 404 &&
                      !fs::exists(root / "convert-invoked"),
                  "an unknown custom voice must fail before launching conversion");

    fs::remove(root / "convert-invoked", ignored);
    const HttpResponse base_without_index = request(
        process.port(), "POST", "/api/convert",
        multipart_body({{"voice", "base"}, {"index_rate", "0.75"}},
                       {{"input.wav", minimal_wav("base-without-index")}}),
        kMultipartType);
    checks.expect(
        base_without_index.status == 400 &&
            base_without_index.body.find("index") != std::string::npos &&
            !fs::exists(root / "convert-invoked"),
        "a positive index rate for a base voice without an index must fail before "
        "launching conversion");

    write_file(root / "runs/no-index-voice/model.safetensors", "no-index-model");
    fs::remove(root / "convert-invoked", ignored);
    const HttpResponse custom_without_index = request(
        process.port(), "POST", "/api/convert",
        multipart_body({{"voice", "no-index-voice"}, {"index_rate", "0.25"}},
                       {{"input.wav", minimal_wav("custom-without-index")}}),
        kMultipartType);
    checks.expect(
        custom_without_index.status == 400 &&
            custom_without_index.body.find("index") != std::string::npos &&
            !fs::exists(root / "convert-invoked"),
        "a positive index rate for a custom voice without an index must fail "
        "before launching conversion");

    write_file(root / "runs/no-index-voice/model.index", "optional-index");
    const HttpResponse explicit_no_index = request(
        process.port(), "POST", "/api/convert",
        multipart_body({{"voice", "no-index-voice"}, {"index_rate", "0"}},
                       {{"input.wav", minimal_wav("explicit-no-index")}}),
        kMultipartType);
    const std::string no_index_arguments = read_file(root / "convert-invoked");
    checks.expect(
        explicit_no_index.status == 200 &&
            no_index_arguments.find(
                (root / "runs/no-index-voice/model.safetensors").string()) !=
                std::string::npos &&
            no_index_arguments.find("--index") == std::string::npos,
        "an explicit zero index rate must keep no-index conversion available");

    write_file(root / "runs/known-voice/model.safetensors", "known-model");
    write_file(root / "runs/known-voice/model.index", "known-index");
    const HttpResponse known_voice = request(
        process.port(), "POST", "/api/convert",
        multipart_body({{"voice", "known-voice"}, {"index_rate", "0.75"}},
                       {{"input.wav", minimal_wav("known-voice")}}),
        kMultipartType);
    const std::string known_arguments = read_file(root / "convert-invoked");
    checks.expect(
        known_voice.status == 200 &&
            known_arguments.find(
                (root / "runs/known-voice/model.safetensors").string()) !=
                std::string::npos &&
            known_arguments.find((root / "runs/known-voice/model.index").string()) !=
                std::string::npos &&
            known_arguments.find("--index-rate 0.750000") != std::string::npos,
        "an installed custom voice must pass its model, index and positive index "
        "rate to conversion");
    const std::string conversion_log = read_file(root / "server.log");
    checks.expect(conversion_log.find("HuBERT:  3 ms") != std::string::npos &&
                      conversion_log.find("Synth:   3 ms") != std::string::npos,
                  "successful conversion stage timings must survive temporary cleanup in the service log");

    write_file(root / "convert-cli-fail.trigger", "1");
    const HttpResponse convert_failure = request(
        process.port(), "POST", "/api/convert",
        multipart_body({}, {{"input.wav", minimal_wav("convert-failure")}}),
        kMultipartType);
    checks.expect(convert_failure.status == 500,
                  "convert must propagate a non-zero CLI exit even with output");
    fs::remove(root / "convert-cli-fail.trigger", ignored);

    write_file(root / "convert-empty-output.trigger", "1");
    const HttpResponse convert_empty = request(
        process.port(), "POST", "/api/convert",
        multipart_body({}, {{"input.wav", minimal_wav("convert-empty")}}),
        kMultipartType);
    checks.expect(convert_empty.status == 500,
                  "convert success with an empty WAV must return an error");
    fs::remove(root / "convert-empty-output.trigger", ignored);

    write_file(root / "convert-invalid-output.trigger", "1");
    const HttpResponse convert_invalid = request(
        process.port(), "POST", "/api/convert",
        multipart_body({}, {{"input.wav", minimal_wav("convert-invalid")}}),
        kMultipartType);
    checks.expect(convert_invalid.status == 500,
                  "convert success with a non-WAV artifact must return an error");
    fs::remove(root / "convert-invalid-output.trigger", ignored);

    write_file(root / "convert-oversized-output.trigger", "1");
    const HttpResponse convert_oversized = request(
        process.port(), "POST", "/api/convert",
        multipart_body({}, {{"input.wav", minimal_wav("convert-oversized")}}),
        kMultipartType);
    checks.expect(convert_oversized.status == 413,
                  "convert output above the buffered response cap must return 413");
    fs::remove(root / "convert-oversized-output.trigger", ignored);

    const fs::path failed_step_work = root / "runs/step-cli-failure/work";
    const std::string failed_step_original = minimal_wav("step-must-survive");
    write_file(failed_step_work / "original.wav", failed_step_original);
    const std::string failed_step_body = multipart_body(
        {{"name", "step-cli-failure"}, {"step", "denoise"}});
    const HttpResponse failed_step_start = request(
        process.port(), "POST", "/api/step", failed_step_body, kMultipartType);
    checks.expect(failed_step_start.status == 200,
                  "failing step worker should be accepted before its CLI exits");
    const HttpResponse failed_step_status =
        wait_for_job_completion(process.port(), "step-cli-failure");
    checks.expect(
        failed_step_status.body.find("\"stage\":\"error\"") !=
                std::string::npos &&
            failed_step_status.body.find("\"error\":true") != std::string::npos &&
            fs::exists(failed_step_work / "original.wav") &&
            read_file(failed_step_work / "original.wav") == failed_step_original,
        "step must not let grep mask a non-zero preprocess exit or publish its output");

    const fs::path nonwav_step_work = root / "runs/step-nonwav-output/work";
    const std::string nonwav_step_original = minimal_wav("nonwav-must-survive");
    write_file(nonwav_step_work / "original.wav", nonwav_step_original);
    const HttpResponse nonwav_step_start = request(
        process.port(), "POST", "/api/step",
        multipart_body({{"name", "step-nonwav-output"}, {"step", "denoise"}}),
        kMultipartType);
    checks.expect(nonwav_step_start.status == 200,
                  "step with a non-WAV staged result should start before validation");
    const HttpResponse nonwav_step_status =
        wait_for_job_completion(process.port(), "step-nonwav-output");
    checks.expect(
        nonwav_step_status.body.find("\"stage\":\"error\"") !=
                std::string::npos &&
            nonwav_step_status.body.find("\"error\":true") != std::string::npos &&
            read_file(nonwav_step_work / "original.wav") == nonwav_step_original &&
            !fs::exists(nonwav_step_work / "result.txt"),
        "step must reject a staged directory without a valid WAV and preserve live work");

    check_training_failure(
        checks, process.port(), root, "train-preprocess-failure",
        "training must not let grep mask a non-zero preprocessing exit");
    check_training_failure(checks, process.port(), root, "train-cli-failure",
                           "training must not let grep mask a non-zero vc_train exit");
    check_training_failure(checks, process.port(), root, "index-cli-failure",
                           "training must report a non-zero build_index exit as an error");
    check_training_failure(checks, process.port(), root, "train-empty-model",
                           "training success without a model artifact must report an error");
    check_training_failure(checks, process.port(), root, "index-empty-output",
                           "index success without an index artifact must report an error");
    check_training_failure(checks, process.port(), root, "train-empty-clips",
                           "training must reject preprocessing with an empty WAV clip");
    check_new_training_failure(checks, process.port(), root,
                               "train-cli-failure-new");
    check_training_preprocess_source(checks, process.port(), root,
                                     "train-flags-from-raw", false);
    check_training_preprocess_source(checks, process.port(), root,
                                     "train-flags-from-work", true);
    check_training_upload(checks, process.port(), root);
    check_training_success(checks, process.port(), root, "train-success");
}

void run_auth_contracts(Checks& checks, const fs::path& server,
                        const fs::path& root) {
    ServerProcess process(server, root, root / "auth-hold.trigger",
                          root / "auth-hold.started", root / "auth-hold.release",
                          root / "auth-fail.trigger", "contract-token");
    HttpResponse anonymous = request(process.port(), "GET", "/api/voices");
    checks.expect(anonymous.status == 401 &&
                      anonymous.has_header("WWW-Authenticate: Bearer\r\n"),
                  "token-enabled API should reject anonymous requests");
    checks.expect(request(process.port(), "GET", "/api/voices", {}, {},
                          "Authorization: Bearer wrong\r\n").status == 401,
                  "token-enabled API should reject a wrong bearer token");
    checks.expect(request(process.port(), "GET", "/api/voices", {}, {},
                          "Authorization: Bearer contract-token\r\n").status == 200,
                  "token-enabled API should accept the configured bearer token");
    const auto cancel_body = multipart_body({{"job_id", "unknown"}});
    checks.expect(request(process.port(), "POST", "/api/train/cancel", cancel_body,
                          kMultipartType).status == 401,
                  "cancellation requires the configured bearer token");
    checks.expect(request(process.port(), "POST", "/api/train/cancel", cancel_body,
                          kMultipartType, "Authorization: Bearer wrong\r\n").status == 401,
                  "wrong token must not cancel a job");
    checks.expect(request(process.port(), "POST", "/api/train/cancel", cancel_body,
                          kMultipartType, "Authorization: Bearer contract-token\r\n").status == 404,
                  "authorized cancellation must reach the job validation contract");
    checks.expect(request(process.port(), "GET", "/").status == 200,
                  "static assets should remain public when API auth is enabled");
}

}  // namespace

int main(int argc, char** argv) {
    const bool body_only = argc == 3 && std::string(argv[2]) == "--body-deadline-only";
    if (argc != 2 && !body_only) {
        std::cerr << "Usage: test_web_contract /absolute/path/to/vc_serve [--body-deadline-only]\n";
        return 2;
    }
    try {
        fs::path server = fs::canonical(argv[1]);
        TempTree tree;
        write_file(tree.path() / "webui/index.html", "contract-root");
        create_fake_preprocess(tree.path());
        create_fake_training_commands(tree.path());
        create_fake_convert(tree.path());

        Checks checks;
        if (body_only) {
            run_body_deadline_contract(checks, server, tree.path());
            return checks.failures() == 0 ? 0 : 1;
        }
        check_training_artifact_transaction(checks, tree.path());
        check_material_transaction(checks, tree.path());
        check_step_artifact_transaction(checks, tree.path());
        run_open_server_contracts(checks, server, tree.path());
        run_auth_contracts(checks, server, tree.path());
        if (checks.failures() != 0) {
            std::cerr << checks.failures() << " web contract check(s) failed\n";
            return 1;
        }
        std::cout << "All web contract checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Web contract test error: " << error.what() << '\n';
        return 1;
    }
}
