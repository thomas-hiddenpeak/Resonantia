// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
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
                     const std::string& extra_headers = {}) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("request socket failed");
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
    if (!send_all(fd, wire)) {
        close(fd);
        throw std::runtime_error("request send failed");
    }
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
                  const std::string& token = {})
        : port_(reserve_loopback_port()) {
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
                if (request(port_, "GET", "/").status == 200) return;
            } catch (const std::exception&) {
            }
            std::this_thread::sleep_for(20ms);
        }
        throw std::runtime_error("vc_serve did not become ready");
    }

    pid_t pid_ = -1;
    int port_ = 0;
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

    write_file(root / "runs/known-voice/model.safetensors", "known-model");
    write_file(root / "runs/known-voice/model.index", "known-index");
    const HttpResponse known_voice = request(
        process.port(), "POST", "/api/convert",
        multipart_body({{"voice", "known-voice"}},
                       {{"input.wav", minimal_wav("known-voice")}}),
        kMultipartType);
    const std::string known_arguments = read_file(root / "convert-invoked");
    checks.expect(
        known_voice.status == 200 &&
            known_arguments.find(
                (root / "runs/known-voice/model.safetensors").string()) !=
                std::string::npos &&
            known_arguments.find((root / "runs/known-voice/model.index").string()) !=
                std::string::npos,
        "an installed custom voice must pass its model and index to conversion");

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
    checks.expect(request(process.port(), "GET", "/").status == 200,
                  "static assets should remain public when API auth is enabled");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: test_web_contract /absolute/path/to/vc_serve\n";
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
        check_training_artifact_transaction(checks, tree.path());
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
