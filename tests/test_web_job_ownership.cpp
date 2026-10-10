// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "tools/web/web_service.h"

namespace resonantia::web {
struct WebServiceJobOwnershipTestAccess {
    static void replay_training_finish(WebService& service, std::uint64_t generation) {
        service.finish_training(generation, WebService::TrainingOutcome::kError);
    }
    static void replay_step_finish(WebService& service, std::uint64_t generation) {
        service.finish(generation, "obsolete-step", true);
    }
};
}

namespace {
namespace fs = std::filesystem;
using namespace resonantia::web;
using namespace std::chrono_literals;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
const std::string kOld = "RIFFxxxxWAVEold";

class Gate {
public:
    bool enter() {
        std::unique_lock<std::mutex> lock(mu_);
        entered_ = true;
        changed_.notify_all();
        if (!changed_.wait_for(lock, 2s, [&] { return released_; })) {
            timed_out_ = true;
            return false;
        }
        return true;
    }
    bool await() {
        std::unique_lock<std::mutex> lock(mu_);
        return changed_.wait_for(lock, 2s, [&] { return entered_; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mu_);
        released_ = true;
        changed_.notify_all();
    }
    bool timed_out() const {
        std::lock_guard<std::mutex> lock(mu_);
        return timed_out_;
    }
private:
    mutable std::mutex mu_;
    std::condition_variable changed_;
    bool entered_ = false, released_ = false, timed_out_ = false;
};

struct ResponseFault final : std::exception {
    const char* what() const noexcept override { return "explicit response failure"; }
};
struct ResponseBoundary {
    Gate gate;
    bool throw_after = false;
    std::uint64_t generation = 0;
    bool accepted = false;
};
thread_local ResponseBoundary* response_boundary = nullptr;

// Only voice B waits. Returning failure at preprocessing prevents every real
// training/index stage. No subprocess, signal, injected allocator or syscall.
class FakeRunner final : public ProcessRunner {
public:
    mutable Gate b;
    mutable std::atomic<int> calls{0};
    ProcessResult run(const ProcessSpec& spec) const override {
        ++calls;
        require(fs::path(spec.argv.at(0)).filename() == "vc_preprocess",
                "unexpected training/index stage in fake worker");
        if (spec.log_path.parent_path().filename() == "B") (void)b.enter();
        return {ProcessTermination::kExited, 1, 0, 0, 0};
    }
};

struct Fixture {
    TempDir root{fs::temp_directory_path(), "resonantia-job-ownership-"};
    fs::path runs = fs::path(root.path()) / "runs";
    std::shared_ptr<FakeRunner> runner = std::make_shared<FakeRunner>();
    std::shared_ptr<WebService> service;
    explicit Fixture(bool failed_upload) {
        for (const char* voice : {"A", "B", "C"}) {
            for (const char* sub : {"raw", "work"}) {
                fs::create_directories(runs / voice / sub);
                std::ofstream(runs / voice / sub / "old.wav") << kOld;
            }
            std::ofstream(runs / voice / "model.safetensors") << "old-model";
            std::ofstream(runs / voice / "model.index") << "old-index";
        }
        // The normal validation rejects a directory as a preprocessing marker
        // before any live material moves; this is not an OS failure injection.
        if (failed_upload) fs::create_directory(runs / "A/steps.txt");
        Paths paths;
        paths.runs = runs.string();
        paths.build = (fs::path(root.path()) / "fake-build").string();
        paths.models = (fs::path(root.path()) / "absent-models").string();
        service = std::make_shared<WebService>(paths, runner);
    }
    bool settled() const {
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while (service.use_count() != 1 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(1ms);
        return service.use_count() == 1;
    }
    ~Fixture() {
        runner->b.release();
        // Never delete a fixture still owned by a detached worker.
        if (!settled()) std::_Exit(2);
    }
    void unchanged_and_clean() const {
        for (const char* voice : {"A", "B", "C"}) {
            for (const char* sub : {"raw", "work"}) {
                std::ifstream in(runs / voice / sub / "old.wav");
                require(std::string(std::istreambuf_iterator<char>(in), {}) == kOld,
                        "old material changed");
                std::size_t count = 0;
                for (const auto& entry : fs::directory_iterator(runs / voice / sub)) {
                    require(entry.path().filename() == "old.wav", "extra material was published");
                    ++count;
                }
                require(count == 1, "old material directory changed");
            }
            for (const auto& pair : {std::pair{"model.safetensors", "old-model"},
                                    std::pair{"model.index", "old-index"}}) {
                std::ifstream in(runs / voice / pair.first);
                require(std::string(std::istreambuf_iterator<char>(in), {}) == pair.second,
                        "published artifact changed");
            }
        }
        for (const auto& entry : fs::recursive_directory_iterator(runs))
            require(entry.path().filename().string().front() != '.', "staging remains");
    }
};

struct Response { int code = 0; std::string body; };
Response request(Fixture& fixture, const Request& req) {
    require(req.body.size() <= 4096, "oversized test request");
    struct Pair {
        int fd[2]{-1, -1};
        ~Pair() { for (int f : fd) if (f >= 0) close(f); }
    } sockets;
    require(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets.fd) == 0,
            "socketpair failed");
    const timeval timeout{1, 0};
    for (int fd : sockets.fd)
        require(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout) == 0 &&
                    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof timeout) == 0,
                "socket timeout setup failed");
    fixture.service->handle_request(sockets.fd[0], req);
    close(sockets.fd[0]); sockets.fd[0] = -1;
    std::string wire;
    char buffer[1024];
    for (;;) {
        const auto count = recv(sockets.fd[1], buffer, sizeof buffer, 0);
        if (count < 0 && errno == EINTR) continue;
        require(count >= 0, "response read failed");
        if (!count) break;
        wire.append(buffer, static_cast<std::size_t>(count));
        require(wire.size() <= 8192, "oversized test response");
    }
    const auto split = wire.find("\r\n\r\n");
    require(wire.rfind("HTTP/1.1 ", 0) == 0 && split != std::string::npos,
            "invalid HTTP response");
    return {std::stoi(wire.substr(9, 3)), wire.substr(split + 4)};
}
void train_request(Request& req, const char* name, bool upload = false) {
    req.method = "POST"; req.path = "/api/train";
    req.content_type = "multipart/form-data; boundary=ownership";
    req.body = "--ownership\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\n" +
               std::string(name) + "\r\n";
    if (upload)
        req.body += "--ownership\r\nContent-Disposition: form-data; name=\"files\"; "
                    "filename=\"new.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFxxxxWAVEnew\r\n";
    req.body += "--ownership--\r\n";
}
std::string status(Fixture& f) {
    Request req; req.method = "GET"; req.path = "/api/train/status";
    return request(f, req).body;
}
std::string public_id(const std::string& body) {
    const auto start = body.find("\"job_id\":\"");
    require(start != std::string::npos, "missing public job id");
    const auto from = start + 10;
    return body.substr(from, body.find('"', from) - from);
}
void busy(Fixture& f, const char* name) {
    Request req; train_request(req, name);
    require(request(f, req).code == 409, "C/B admitted while another job owns the slot");
}

struct PendingRequest {
    ResponseBoundary boundary;
    std::optional<Response> response;
    bool response_fault = false;
    std::exception_ptr error;
    std::jthread thread;
    PendingRequest(Fixture& f, const Request& req, bool fail) {
        boundary.throw_after = fail;
        thread = std::jthread([&, this] {
            response_boundary = &boundary;
            try { response = request(f, req); }
            catch (const ResponseFault&) { response_fault = true; }
            catch (...) { error = std::current_exception(); }
            response_boundary = nullptr;
        });
    }
    void complete() {
        boundary.gate.release(); thread.join();
        if (error) std::rethrow_exception(error);
        require(!boundary.gate.timed_out(), "response barrier timed out");
    }
    ~PendingRequest() {
        boundary.gate.release();
        if (thread.joinable()) thread.join();
    }
};

void scenario(bool failed_upload, bool response_fault) {
    Fixture f(failed_upload);
    Request a; train_request(a, "A", failed_upload);
    PendingRequest pending(f, a, response_fault);
    require(pending.boundary.gate.await(), "A response boundary not observed");
    const auto generation_a = pending.boundary.generation;
    require(generation_a != 0 && pending.boundary.accepted != failed_upload,
            "wrong response boundary");
    require(f.settled(), "A fake worker did not retire before B admission");
    const auto a_state = status(f);
    require(a_state.find("\"name\":\"A\"") != std::string::npos, "A identity missing");
    if (failed_upload) {
        require(a_state.find("\"running\":true") != std::string::npos,
                "failed upload released reservation before response/staging unwind");
        require(f.runner->calls == 0, "failed upload started a worker");
        busy(f, "B"); busy(f, "C");
        pending.complete();
        f.unchanged_and_clean();
    } else {
        require(a_state.find("\"running\":false") != std::string::npos &&
                    a_state.find("\"error\":true") != std::string::npos,
                "A worker did not finalize before its delayed acceptance response");
    }
    Request b; train_request(b, "B");
    const auto accepted_b = request(f, b);
    require(accepted_b.code == 200 && f.runner->b.await(), "B fake worker did not enter");
    const auto b_state = status(f);
    require(public_id(accepted_b.body) == public_id(b_state) &&
                public_id(b_state) != public_id(a_state) &&
                b_state.find("\"running\":true") != std::string::npos,
            "B did not acquire an independent active identity");
    busy(f, "C");
    if (!failed_upload) pending.complete();
    require(pending.response_fault == response_fault, "response exception was not observed");
    if (!response_fault) {
        require(pending.response && pending.response->code == (failed_upload ? 500 : 200),
                "wrong delayed A response");
        if (!failed_upload)
            require(public_id(pending.response->body) == public_id(a_state),
                    "delayed A response reported B's identity");
    }
    require(status(f) == b_state, "A response unwind changed B's ownership/state");
    // Explicitly replay obsolete completion callbacks, not an injected thread
    // or a claim that every old OS-fault scenario was dynamically reproduced.
    for (int repeat = 0; repeat < 2; ++repeat) {
        WebServiceJobOwnershipTestAccess::replay_training_finish(*f.service, generation_a);
        require(status(f) == b_state, "stale training finish changed B");
        WebServiceJobOwnershipTestAccess::replay_step_finish(*f.service, generation_a);
        require(status(f) == b_state, "stale step finish changed B");
        busy(f, "C");
    }
    f.runner->b.release();
    require(f.settled() && !f.runner->b.timed_out(), "B did not retire naturally");
    f.unchanged_and_clean();
    Request c; train_request(c, "C");
    require(request(f, c).code == 200 && f.settled(), "C retry did not start after B retired");
    require(f.runner->calls == (failed_upload ? 2 : 3), "unexpected worker invocation");
    f.unchanged_and_clean();
    std::cout << (failed_upload ? "upload-rejection" : "worker-before-response")
              << (response_fault ? "/exception" : "/normal")
              << ": A/B/C ownership, stale finish x2, retry and cleanup passed\n";
}
std::size_t fd_count() {
    std::size_t count = 0;
    for (const auto& item : fs::directory_iterator("/proc/self/fd")) { (void)item; ++count; }
    return count;
}
}

namespace resonantia::web {
void test_training_response_boundary(std::uint64_t generation, bool accepted) {
    if (!response_boundary) return;
    response_boundary->generation = generation;
    response_boundary->accepted = accepted;
    require(response_boundary->gate.enter(), "response gate wait expired");
    if (response_boundary->throw_after) throw ResponseFault{};
}
}

int main() {
    try {
        setenv("CUDA_VISIBLE_DEVICES", "", 1);
        const auto descriptors = fd_count();
        for (bool upload : {true, false})
            for (bool error : {false, true}) scenario(upload, error);
        require(fd_count() == descriptors, "file descriptor leak");
        std::cout << "4/4 deterministic host cases; no GPU, subprocess or interposition\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Job ownership test: " << error.what() << '\n';
        return 1;
    }
}
