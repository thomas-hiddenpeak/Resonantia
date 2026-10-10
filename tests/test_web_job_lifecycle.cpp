// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>

#include "tools/web/web_service.h"

// Faults are restricted to the request thread, in this isolated test process.
// Start only after this request reserves a job. Read-only private test access
// identifies that boundary; all behavioral assertions use HTTP/status responses.
// Sweep observed allocations, not implementation-specific indices or sizes.
// Workers never inherit the fault. Pre-validation filesystem OOM is out of scope.
namespace {
thread_local bool count_allocations = false;
thread_local std::size_t allocation_count = 0;
thread_local std::size_t fail_at = 0;
thread_local bool injected = false;
thread_local const std::atomic<bool>* reserved = nullptr;
}
void* operator new(std::size_t size) {
    if (count_allocations && reserved->load() && ++allocation_count == fail_at) {
        injected = true;
        throw std::bad_alloc();
    }
    if (void* result = std::malloc(size ? size : 1)) return result;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace resonantia::web {
struct WebServiceLifecycleTestAccess {
    static const std::atomic<bool>& running(const WebService& service) {
        return service.job_.running;
    }
};
}

namespace {
namespace fs = std::filesystem;
using namespace resonantia::web;
using namespace std::chrono_literals;

class FailingRunner final : public ProcessRunner {
public:
    mutable std::atomic<bool> entered{false};
    mutable std::atomic<bool> released{false};
    mutable std::atomic<bool> timed_out{false};
    mutable std::atomic<int> calls{0};
    ProcessResult run(const ProcessSpec&) const override {
        ++calls;
        entered = true;
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while (!released && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(1ms);
        if (!released) timed_out = true;
        return {ProcessTermination::kExited, 1, 0, 0, 0};
    }
};

struct Fixture {
    TempDir root{fs::temp_directory_path(), "resonantia-web-job-"};
    const std::string name = std::string(64, 'v');  // Valid, non-SSO metadata.
    fs::path voice = fs::path(root.path()) / "runs" / name;
    std::shared_ptr<FailingRunner> runner = std::make_shared<FailingRunner>();
    std::shared_ptr<WebService> service;
    Fixture() {
        for (const char* sub : {"raw", "work"}) {
            fs::create_directories(voice / sub);
            std::ofstream(voice / sub / "old.wav", std::ios::binary) << "RIFFxxxxWAVEold";
        }
        Paths paths;
        paths.runs = (fs::path(root.path()) / "runs").string();
        paths.build = (fs::path(root.path()) / "build").string();
        paths.models = (fs::path(root.path()) / "models").string();
        service = std::make_shared<WebService>(paths, runner);
    }
    bool wait_for_worker_exit() {
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while (service.use_count() != 1 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(1ms);
        return service.use_count() == 1;
    }
    ~Fixture() {
        count_allocations = false;
        runner->released = true;
        // Retire worker ownership before deleting any fixture data.
        if (!wait_for_worker_exit()) std::abort();
    }
};

void job_request(Request& req, const Fixture& fixture, const std::string& route) {
    req.method = "POST";
    req.path = "/api/" + route;
    req.content_type = "multipart/form-data; boundary=job-boundary";
    for (const auto& item : {std::pair<std::string, std::string>{"name", fixture.name},
                             {"step", "denoise"}, {"steps", "1"}}) {
        req.body += "--job-boundary\r\nContent-Disposition: form-data; name=\"" +
                    item.first + "\"\r\n\r\n" + item.second + "\r\n";
    }
    req.body += "--job-boundary--\r\n";
}

struct Response {
    std::string wire;
    std::size_t allocations = 0;
    bool fault = false;
    bool exception = false;
};
Response request(Fixture& fixture, const Request& req, bool count = false,
                 std::size_t nth = 0) {
    int sockets[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets))
        throw std::runtime_error("socketpair failed");
    Response result;
    allocation_count = 0;
    injected = false;
    fail_at = nth;
    reserved = &WebServiceLifecycleTestAccess::running(*fixture.service);
    count_allocations = count;
    try {
        fixture.service->handle_request(sockets[0], req);
    } catch (const std::exception&) {
        // HttpTransport normally converts this to an error response. The direct
        // boundary lets the test examine service state after the same unwind.
        result.exception = true;
    }
    count_allocations = false;
    result.allocations = allocation_count;
    result.fault = injected;
    ::close(sockets[0]);
    char buffer[4096];
    ssize_t length;
    while ((length = ::recv(sockets[1], buffer, sizeof buffer, 0)) > 0)
        result.wire.append(buffer, static_cast<std::size_t>(length));
    ::close(sockets[1]);
    return result;
}
std::string status(Fixture& fixture) {
    Request req;
    req.method = "GET";
    req.path = "/api/train/status";
    return request(fixture, req).wire;
}
bool contains(const std::string& text, const char* value) {
    return text.find(value) != std::string::npos;
}
bool settle(Fixture& fixture) {
    fixture.runner->released = true;
    return fixture.wait_for_worker_exit() && !fixture.runner->timed_out;
}
bool old_material_and_no_staging(const Fixture& fixture) {
    for (const char* sub : {"raw", "work"}) {
        std::ifstream input(fixture.voice / sub / "old.wav", std::ios::binary);
        if (std::string(std::istreambuf_iterator<char>(input), {}) != "RIFFxxxxWAVEold")
            return false;
    }
    for (const auto& entry : fs::directory_iterator(fixture.voice)) {
        if (entry.path().filename().string().front() == '.') return false;
    }
    return true;
}
}

int main() {
    try {
        int failures = 0;
        for (const std::string route : {"step", "train"}) {
            std::size_t budget;
            {
                Fixture baseline;
                Request req;
                job_request(req, baseline, route);
                const Response response = request(baseline, req, true);
                budget = response.allocations;
                if (response.exception || response.wire.rfind("HTTP/1.1 200 ", 0) ||
                    !settle(baseline) || budget == 0 || budget > 512)
                    throw std::runtime_error("invalid allocation sweep baseline");
            }
            int observed = 0, active_response_faults = 0, stuck = 0;
            for (std::size_t nth = 1; nth <= budget; ++nth) {
                Fixture fixture;
                Request req;
                job_request(req, fixture, route);
                const Response response = request(fixture, req, true, nth);
                if (response.fault) ++observed;
                const auto deadline = std::chrono::steady_clock::now() + 2s;
                while (fixture.service.use_count() != 1 && !fixture.runner->entered &&
                       std::chrono::steady_clock::now() < deadline)
                    std::this_thread::sleep_for(1ms);
                bool ok = response.fault;
                if (fixture.runner->entered) {
                    // A failed response allocation must not release an already
                    // handed-off job while its fake CLI still owns the slot.
                    if (response.exception) ++active_response_faults;
                    ok &= contains(status(fixture), "\"running\":true");
                } else {
                    ok &= !response.exception &&
                          response.wire.rfind("HTTP/1.1 503 ", 0) == 0;
                }
                ok &= settle(fixture);
                const bool running = contains(status(fixture), "\"running\":true");
                if (running) ++stuck;
                ok &= !running && old_material_and_no_staging(fixture);
                const int prior_calls = fixture.runner->calls;
                const Response retry = request(fixture, req);
                ok &= retry.wire.rfind("HTTP/1.1 200 ", 0) == 0 && !retry.exception;
                ok &= settle(fixture);
                const std::string final_status = status(fixture);
                ok &= fixture.runner->calls == prior_calls + 1 &&
                      contains(final_status, "\"running\":false") &&
                      contains(final_status, "\"done\":true") &&
                      contains(final_status, "\"error\":true") &&
                      old_material_and_no_staging(fixture);
                if (!ok) {
                    ++failures;
                    std::cerr << route << " allocation " << nth
                              << ": injected=" << response.fault << " stuck=" << running
                              << " retry=" << retry.wire.substr(0, retry.wire.find('\r')) << '\n';
                }
            }
            if (active_response_faults == 0) ++failures;
            std::cout << route << ": " << observed << '/' << budget
                      << " one-shot faults, " << stuck << " stuck reservations, "
                      << active_response_faults << " response faults with active worker\n";
        }
        std::cout << "job lifecycle failures: " << failures << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        count_allocations = false;
        std::cerr << "job lifecycle test error: " << e.what() << '\n';
        return 1;
    }
}
