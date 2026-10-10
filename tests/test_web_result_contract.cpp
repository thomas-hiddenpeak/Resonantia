// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tools/web/web_service.h"

namespace {
namespace fs = std::filesystem;
using namespace resonantia::web;
using namespace std::chrono_literals;

const std::string kWav = "RIFFxxxxWAVEsynthetic-result";
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
void write(const fs::path& path, const std::string& bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << bytes;
    out.close();
    require(bool(out), "cannot write synthetic artifact");
}
std::string read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}
std::size_t fd_count() {
    std::size_t count = 0;
    for (const auto& entry : fs::directory_iterator("/proc/self/fd")) {
        (void)entry;
        ++count;
    }
    return count;
}
std::string argument(const ProcessSpec& spec, const std::string& flag) {
    for (std::size_t i = 1; i + 1 < spec.argv.size(); ++i)
        if (spec.argv[i] == flag) return spec.argv[i + 1];
    throw std::runtime_error("fake CLI missing " + flag);
}

// This supported dependency returns synthetic ProcessResult values. It never
// starts a process, raises a signal, calls waitpid, or causes an OS I/O failure.
class ResultRunner final : public ProcessRunner {
public:
    explicit ResultRunner(fs::path build) : build_(std::move(build)) {}
    void reset(std::string failing_program, ProcessResult result =
                   {ProcessTermination::kExited, 0, 0, 0, 0}) {
        std::lock_guard<std::mutex> lock(mu_);
        failing_program_ = std::move(failing_program);
        result_ = result;
        calls_.clear();
        outputs_.clear();
        sync_directories_.clear();
    }
    ProcessResult run(const ProcessSpec& spec) const override {
        std::lock_guard<std::mutex> lock(mu_);
        require(!spec.argv.empty() && fs::path(spec.argv.front()).parent_path() == build_,
                "unexpected fake executable path");
        const std::string program = fs::path(spec.argv.front()).filename().string();
        calls_.push_back(program);
        fs::path output;
        std::string bytes;
        if (program == "vc_preprocess") {
            output = fs::path(argument(spec, "--output-dir")) / "result.wav";
            bytes = kWav;
        } else if (program == "vc_convert") {
            output = argument(spec, "--output");
            bytes = kWav;
        } else if (program == "vc_train") {
            output = argument(spec, "--out");
            bytes = "new-model";
        } else if (program == "build_index") {
            output = argument(spec, "--output");
            bytes = "new-index";
        } else {
            throw std::runtime_error("unexpected fake program");
        }
        // Leave a valid-looking output even on failure: otherwise missing-output
        // validation could hide a bug that ignores the ProcessResult.
        write(output, bytes);
        outputs_.push_back(output);
        const auto log_parent = spec.log_path.parent_path();
        if (log_parent.parent_path() == fs::path("/tmp")) {
            const std::string leaf = log_parent.filename().string();
            require(leaf.rfind("resonantia-preprocess-", 0) == 0 ||
                        leaf.rfind("resonantia-convert-", 0) == 0,
                    "unexpected synchronous temporary directory");
            sync_directories_.push_back(log_parent);
        }
        std::ofstream log(spec.log_path, std::ios::app);
        log << "synthetic CLI result\n";
        log.close();
        require(bool(log), "cannot write fake log");
        return program == failing_program_ ? result_ :
            ProcessResult{ProcessTermination::kExited, 0, 0, 0, 0};
    }
    std::vector<std::string> calls() const {
        std::lock_guard<std::mutex> lock(mu_);
        return calls_;
    }
    void require_cleanup() const {
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto& path : outputs_)
            require(!fs::exists(path), "transient fake output still exists: " + path.string());
        for (const auto& path : sync_directories_)
            require(!fs::exists(path), "synchronous temporary directory still exists");
    }
private:
    fs::path build_;
    mutable std::mutex mu_;
    std::string failing_program_;
    ProcessResult result_;
    mutable std::vector<std::string> calls_;
    mutable std::vector<fs::path> outputs_, sync_directories_;
};

struct Fixture {
    fs::path directory, voice;
    std::shared_ptr<ResultRunner> runner;
    std::shared_ptr<WebService> service;
    const std::map<std::string, std::string> old{
        {"raw/old.wav", "RIFFxxxxWAVEold-raw"},
        {"work/old.wav", "RIFFxxxxWAVEold-work"},
        {"steps.txt", "dereverb\n"},
        {"clips/old.wav", "RIFFxxxxWAVEold-clips"},
        {"model.safetensors", "old-model"}, {"model.index", "old-index"}};
    explicit Fixture(fs::path root) : directory(std::move(root)), voice(directory / "runs/voice"),
        runner(std::make_shared<ResultRunner>(directory / "build")) {
        for (const auto& [path, bytes] : old) write(voice / path, bytes);
        Paths paths;
        paths.repo = directory.string();
        paths.runs = (directory / "runs").string();
        paths.build = (directory / "build").string();
        paths.models = (directory / "models").string();
        paths.gmodel = (directory / "models/base.safetensors").string();
        service = std::make_shared<WebService>(paths, runner);
    }
    bool settled() const {
        const auto limit = std::chrono::steady_clock::now() + 1s;
        while (service.use_count() != 1 && std::chrono::steady_clock::now() < limit)
            std::this_thread::sleep_for(1ms);
        return service.use_count() == 1;
    }
    ~Fixture() {
        // The existing service owns detached worker lifetimes through shared_ptr.
        // Never delete fixtures while a worker still owns the service.
        if (!settled()) std::abort();
    }
    void require_old() const {
        for (const auto& [path, bytes] : old)
            require(read(voice / path) == bytes, "old artifact changed: " + path);
        for (const char* child : {"raw", "work", "clips"}) {
            std::size_t count = 0;
            for (const auto& entry : fs::directory_iterator(voice / child)) {
                require(entry.path().filename() == "old.wav", "unexpected published file");
                ++count;
            }
            require(count == 1, "old artifact directory changed");
        }
    }
    void require_clean() const {
        runner->require_cleanup();
        for (const auto& entry : fs::recursive_directory_iterator(directory))
            require(entry.path().filename().string().front() != '.', "staging/backup still exists");
    }
};

struct Response { int status; std::string headers, body; };
Response request(Fixture& fixture, const Request& req) {
    require(req.body.size() <= 4096, "test request exceeded 4KiB");
    struct Pair {
        int fd[2]{-1, -1};
        ~Pair() { for (int value : fd) if (value >= 0) close(value); }
    } sockets;
    require(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets.fd) == 0,
            "socketpair failed");
    const timeval timeout{1, 0};
    for (int fd : sockets.fd)
        require(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout) == 0 &&
                    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof timeout) == 0,
                "socket timeout setup failed");
    // Responses are small; the socketpair observes the real HTTP response writer
    // without another listener, transport parser, thread, or external connection.
    fixture.service->handle_request(sockets.fd[0], req);
    close(sockets.fd[0]);
    sockets.fd[0] = -1;
    std::string wire;
    char buffer[1024];
    for (;;) {
        const auto n = recv(sockets.fd[1], buffer, sizeof buffer, 0);
        if (n < 0 && errno == EINTR) continue;
        require(n >= 0, "response read failed or timed out");
        if (n == 0) break;
        wire.append(buffer, static_cast<std::size_t>(n));
        require(wire.size() <= 8192, "test response exceeded 8KiB");
    }
    const auto headers_end = wire.find("\r\n\r\n");
    require(wire.rfind("HTTP/1.1 ", 0) == 0 && headers_end != std::string::npos,
            "malformed HTTP response");
    return {std::stoi(wire.substr(9, 3)), wire.substr(0, headers_end), wire.substr(headers_end + 4)};
}
void make_request(Request& req, const std::string& scenario) {
    const bool step = scenario == "step";
    const bool train = scenario.rfind("train-", 0) == 0;
    req.method = "POST";
    req.path = step ? "/api/step" : train ? "/api/train" :
               scenario == "preprocess" ? "/api/preprocess" : "/api/convert";
    req.content_type = "multipart/form-data; boundary=result-boundary";
    const auto field = [&](const std::string& name, const std::string& value) {
        req.body += "--result-boundary\r\nContent-Disposition: form-data; name=\"" +
                    name + "\"\r\n\r\n" + value + "\r\n";
    };
    if (step || train) {
        field("name", "voice");
        field("step", "denoise");
        field("steps", "1");
    } else {
        field("voice", "voice");
        field("pp_denoise", scenario == "convert-main" ? "0" : "1");
        req.body += "--result-boundary\r\nContent-Disposition: form-data; name=\"files\"; "
                    "filename=\"input.wav\"\r\nContent-Type: audio/wav\r\n\r\n" + kWav + "\r\n";
    }
    req.body += "--result-boundary--\r\n";
}
Response status(Fixture& fixture) {
    Request req;
    req.method = "GET";
    req.path = "/api/train/status";
    return request(fixture, req);
}
void require_terminal(Fixture& fixture, bool error, const std::string& stage) {
    require(fixture.settled(), "background worker did not finish within 1s");
    const auto response = status(fixture);
    require(response.status == 200 && response.body.find("\"running\":false") != std::string::npos &&
                response.body.find("\"done\":true") != std::string::npos &&
                response.body.find(error ? "\"error\":true" : "\"error\":false") != std::string::npos &&
                response.body.find("\"stage\":\"" + stage + "\"") != std::string::npos,
            "incorrect background terminal state");
}
struct Scenario { std::string name, failing_program; std::vector<std::string> failing_calls, retry_calls; };
}

int main() {
    try {
        const auto start = std::chrono::steady_clock::now();
        const auto descriptors = fd_count();
        fs::path fixture_path;
        int failures = 0, recoveries = 0;
        std::uintmax_t bytes = 0;
        {
            TempDir root(fs::temp_directory_path(), "resonantia-web-results-");
            fixture_path = root.path();
            const std::vector<Scenario> scenarios{
                {"preprocess", "vc_preprocess", {"vc_preprocess"}, {"vc_preprocess"}},
                {"convert-preprocess", "vc_preprocess", {"vc_preprocess"}, {"vc_preprocess", "vc_convert"}},
                {"convert-main", "vc_convert", {"vc_convert"}, {"vc_convert"}},
                {"step", "vc_preprocess", {"vc_preprocess"}, {"vc_preprocess"}},
                {"train-preprocess", "vc_preprocess", {"vc_preprocess"}, {"vc_preprocess", "vc_train", "build_index"}},
                {"train-model", "vc_train", {"vc_preprocess", "vc_train"}, {"vc_preprocess", "vc_train", "build_index"}},
                {"train-index", "build_index", {"vc_preprocess", "vc_train", "build_index"}, {"vc_preprocess", "vc_train", "build_index"}}};
            const std::vector<std::pair<std::string, ProcessResult>> results{
                {"signal-result", {ProcessTermination::kSignaled, -1, SIGTERM, 0, 0}},
                {"wait-result", {ProcessTermination::kWaitFailed, -1, 0, ECHILD, 0}},
                {"io-result", {ProcessTermination::kExited, 0, 0, 0, EIO}}};
            for (const auto& [label, result] : results) {
                require(!result.success(), "fixture result is not a failure");
                for (const auto& scenario : scenarios) {
                    Fixture fixture(fixture_path / (label + "-" + scenario.name));
                    Request req;
                    make_request(req, scenario.name);
                    const bool background = scenario.name == "step" || scenario.name.rfind("train-", 0) == 0;
                    const auto before = status(fixture).body;
                    fixture.runner->reset(scenario.failing_program, result);
                    const auto failed = request(fixture, req);
                    if (background) {
                        require(failed.status == 200, "background request not accepted");
                        require_terminal(fixture, true, "error");
                    } else {
                        require(failed.status == 500 && failed.headers.find("audio/wav") == std::string::npos &&
                                    failed.body.find(kWav) == std::string::npos,
                                "synchronous failure returned successful audio");
                        require(status(fixture).body == before, "synchronous failure changed job state");
                    }
                    require(fixture.runner->calls() == scenario.failing_calls, "wrong failure calls or downstream execution");
                    fixture.require_old();
                    fixture.require_clean();
                    require(fd_count() == descriptors, "failure leaked descriptors");
                    ++failures;

                    fixture.runner->reset("");
                    const auto recovered = request(fixture, req);
                    require(recovered.status == 200, "retry was rejected or failed");
                    if (background) {
                        require_terminal(fixture, false, scenario.name == "step" ? "step-done" : "done");
                        if (scenario.name == "step") {
                            require(read(fixture.voice / "work/result.wav") == kWav &&
                                        read(fixture.voice / "steps.txt") == "dereverb\ndenoise\n" &&
                                        !fs::exists(fixture.voice / "work/old.wav"), "step retry did not publish");
                        } else {
                            require(read(fixture.voice / "clips/result.wav") == kWav &&
                                        read(fixture.voice / "model.safetensors") == "new-model" &&
                                        read(fixture.voice / "model.index") == "new-index" &&
                                        !fs::exists(fixture.voice / "clips/old.wav"), "training retry did not publish");
                        }
                    } else {
                        require(recovered.body == kWav && recovered.headers.find("audio/wav") != std::string::npos,
                                "synchronous recovery missing synthetic audio");
                        fixture.require_old();
                    }
                    require(fixture.runner->calls() == scenario.retry_calls, "retry CLI sequence differs");
                    fixture.require_clean();
                    require(fd_count() == descriptors, "recovery leaked descriptors");
                    ++recoveries;
                    std::cout << label << " / " << scenario.name << ": failure, preservation, cleanup, retry passed\n";
                }
            }
            for (const auto& entry : fs::recursive_directory_iterator(fixture_path))
                if (entry.is_regular_file()) bytes += entry.file_size();
            require(bytes <= 1024 * 1024, "synthetic fixtures exceeded 1MiB");
        }
        require(!fs::exists(fixture_path), "unique fixture directory remained");
        require(failures == 21 && recoveries == 21 && fd_count() == descriptors,
                "incomplete result matrix or final FD leak");
        const auto elapsed = std::chrono::steady_clock::now() - start;
        require(elapsed < 5s, "consumer group exceeded 5s");
        rusage usage{};
        require(getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss <= 256 * 1024,
                "consumer test exceeded 256MiB RSS budget");
        std::cout << "Web result consumers: " << failures << "/21 failures, " << recoveries
                  << "/21 recoveries; elapsed=" << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
                  << " ms; fixture bytes=" << bytes << "; peak RSS=" << usage.ru_maxrss
                  << " KiB; no FD delta, workers settled, all owned temporaries removed; real child processes=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Web result consumer failure: " << error.what() << '\n';
        return 1;
    }
}
