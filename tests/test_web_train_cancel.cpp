// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
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
    void gate_on(const std::string& program) {
        std::lock_guard<std::mutex> lock(gate_mu_);
        gate_program_ = program; gate_entered_ = false; gate_released_ = false;
    }
    void cooperative(bool enabled) { std::lock_guard<std::mutex> lock(mu_); cooperative_=enabled; }
    fs::path token_path() const { std::lock_guard<std::mutex> lock(gate_mu_); return token_path_; }
    bool await_gate() const {
        std::unique_lock<std::mutex> lock(gate_mu_);
        return gate_cv_.wait_for(lock, 500ms, [&] { return gate_entered_; });
    }
    void release_gate() {
        { std::lock_guard<std::mutex> lock(gate_mu_); gate_released_ = true; }
        gate_cv_.notify_all();
    }
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
        fs::path output, token;
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
            if (cooperative_) {
                token = argument(spec, "--cancel-file");
                require(read(token)=="0", "new training token is not clear");
                std::lock_guard<std::mutex> gate_lock(gate_mu_); token_path_=token;
            }
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
        {
            std::unique_lock<std::mutex> lock(gate_mu_);
            if (program == gate_program_) {
                gate_entered_ = true; gate_cv_.notify_all();
                if (!gate_cv_.wait_for(lock, 1s, [&] { return gate_released_; }))
                    return {ProcessTermination::kExited, 1, 0, 0, 0};
            }
        }
        if (program != failing_program_ && cooperative_ && program == "vc_train" && read(token)=="1")
            return {ProcessTermination::kExited, 2, 0, 0, 0};
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
    mutable std::mutex gate_mu_;
    mutable std::condition_variable gate_cv_;
    std::string gate_program_;
    mutable bool gate_entered_ = false, gate_released_ = false;
    fs::path build_;
    mutable std::mutex mu_;
    bool cooperative_ = false;
    mutable fs::path token_path_;
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
        runner->release_gate();
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
std::string job_id(const Response& response) {
    const std::string key = "\"job_id\":\"";
    const auto start = response.body.find(key);
    require(start != std::string::npos, "accepted job/status must expose job_id");
    const auto from = start + key.size(), end = response.body.find('"', from);
    require(end != std::string::npos && end > from, "job_id must be nonempty");
    return response.body.substr(from, end - from);
}
Response cancel(Fixture& fixture, const std::string& id) {
    Request req;
    req.method = "POST"; req.path = "/api/train/cancel";
    req.content_type = "multipart/form-data; boundary=result-boundary";
    req.body = "--result-boundary\r\nContent-Disposition: form-data; name=\"job_id\"\r\n\r\n" +
               id + "\r\n--result-boundary--\r\n";
    return request(fixture, req);
}
void cancelled(Fixture& fixture) {
    const auto limit = std::chrono::steady_clock::now() + 500ms;
    while (std::chrono::steady_clock::now() < limit) {
        const auto s = status(fixture);
        if (s.body.find("\"cancelled\":true") != std::string::npos &&
            s.body.find("\"running\":false") != std::string::npos) {
            require_terminal(fixture, false, "cancelled");
            return;
        }
        std::this_thread::sleep_for(1ms);
    }
    throw std::runtime_error("cancelled terminal state not reached within 500ms");
}
void queued_case(const fs::path& root) {
    Fixture f(root / "queued");
    f.runner->gate_on("vc_convert");
    Request convert, train;
    make_request(convert, "convert-main"); make_request(train, "train-model");
    Response converted{}; std::exception_ptr error;
    std::jthread holder([&] { try { converted=request(f,convert); } catch (...) { error=std::current_exception(); } });
    require(f.runner->await_gate(), "GPU holder never entered");
    const auto accepted=request(f,train); require(accepted.status==200,"queued train rejected");
    const auto id=job_id(accepted);
    require(cancel(f,id).status==202,"first queued cancel must be accepted");
    cancelled(f); // Must finish while the unrelated synchronous GPU holder remains blocked.
    require(cancel(f,id).status==200,"terminal duplicate cancel must be idempotent");
    const auto again=request(f,train);require(again.status==200,"queue slot not reusable");
    const auto second=job_id(again);require(second!=id,"job ids reused");
    require(cancel(f,id).status==409,"stale cancel affected new queued job");
    require(status(f).body.find("\"cancel_requested\":false")!=std::string::npos,"stale cancel changed new job");
    require(cancel(f,second).status==202,"second queued cancellation rejected");cancelled(f);
    f.runner->release_gate();holder.join();if(error)std::rethrow_exception(error);
    require(converted.status==200,"queued cancellation interrupted GPU holder");
    f.require_old();f.require_clean();
    f.runner->reset("");
    require(request(f,train).status==200,"retry after queued cancellation failed");require_terminal(f,false,"done");
    f.require_clean();std::cout<<"queued: prompt cancellation while GPU holder runs; stale id/retry/cleanup passed\n";
}
void stage_case(const fs::path& root,const std::string& program,bool upload=false) {
    Fixture f(root/(upload?"uploaded":program));f.runner->gate_on(program);
    Request train;make_request(train,"train-model");
    if(upload) {
        const std::string end="--result-boundary--\r\n";
        train.body.resize(train.body.size()-end.size());
        train.body+="--result-boundary\r\nContent-Disposition: form-data; name=\"files\"; filename=\"upload.wav\"\r\n\r\n"+kWav+"\r\n"+end;
    }
    const auto accepted=request(f,train);require(accepted.status==200,"training rejected");
    const auto id=job_id(accepted);require(f.runner->await_gate(),"stage never entered");
    require(cancel(f,id).status==202,"running cancellation not accepted");
    const auto pending=status(f);
    require(pending.body.find("\"cancel_requested\":true")!=std::string::npos &&
            pending.body.find("\"running\":true")!=std::string::npos &&
            pending.body.find("\"cancelled\":false")!=std::string::npos,"request was falsely reported stopped");
    require(cancel(f,id).status==200,"running duplicate cancel not idempotent");
    require(request(f,train).status==409,"slot released before current stage returned");
    f.runner->release_gate();cancelled(f);
    const auto calls=f.runner->calls();
    const std::vector<std::string> expected=program=="vc_preprocess"?std::vector<std::string>{"vc_preprocess"}:
        program=="vc_train"?std::vector<std::string>{"vc_preprocess","vc_train"}:
        std::vector<std::string>{"vc_preprocess","vc_train","build_index"};
    require(calls==expected,"cancelled job ran downstream CLI");
    if(upload) {
        require(read(f.voice/"raw/upload.wav")==kWav && read(f.voice/"work/upload.wav")==kWav &&
                !fs::exists(f.voice/"steps.txt"),"completed upload commit was undone");
        for(const char* path:{"clips/old.wav","model.safetensors","model.index"})
            require(read(f.voice/path)==f.old.at(path),"cancelled upload replaced old voice");
    } else f.require_old();
    f.require_clean();f.runner->reset("");
    const auto retry=request(f,train);require(retry.status==200,"retry rejected");require_terminal(f,false,"done");
    require(job_id(retry)!=id && cancel(f,id).status==409,"old cancellation matched retry");
    require(cancel(f,job_id(retry)).status==409,"completed publication claimed cancellation");
    f.require_clean();std::cout<<program<<(upload?" with committed upload":"")<<": pending/terminal/preservation/retry passed\n";
}
void race_case(const fs::path& root,int n) {
    Fixture f(root/("race-"+std::to_string(n)));f.runner->gate_on("build_index");
    Request train;make_request(train,"train-model");const auto accepted=request(f,train);
    const auto id=job_id(accepted);require(f.runner->await_gate(),"index gate not reached");
    Response response{};std::exception_ptr error;
    std::jthread cancelling([&] {try{response=cancel(f,id);}catch(...){error=std::current_exception();}});
    f.runner->release_gate();cancelling.join();if(error)std::rethrow_exception(error);
    require(f.settled(),"race worker did not settle");const auto terminal=status(f);
    if(terminal.body.find("\"cancelled\":true")!=std::string::npos) {
        require(response.status==202 || response.status==200,"cancelled race ack incorrect");f.require_old();
    } else {
        require(response.status==409,"late cancel not explicitly rejected");require_terminal(f,false,"done");
        require(read(f.voice/"model.safetensors")=="new-model" && read(f.voice/"model.index")=="new-index" &&
                read(f.voice/"clips/result.wav")==kWav,"publication race left partial artifacts");
    }
    f.require_clean();
}

#ifdef RESONANTIA_TEST_MATERIAL_COMMIT_GATE
class CommitGate {
public:
    explicit CommitGate(fs::path voice) : voice_(std::move(voice)) {}
    void pause(const fs::path& voice) {
        std::unique_lock<std::mutex> lock(mu_);
        if (voice != voice_) return;
        entered_ = true; changed_.notify_all();
        if (!changed_.wait_for(lock, 2s, [&] { return released_; }))
            throw std::runtime_error("commit gate exceeded two seconds");
    }
    bool entered() {
        std::unique_lock<std::mutex> lock(mu_);
        return changed_.wait_for(lock, 500ms, [&] { return entered_; });
    }
    void release() {
        { std::lock_guard<std::mutex> lock(mu_); released_ = true; }
        changed_.notify_all();
    }
private:
    fs::path voice_;
    std::mutex mu_;
    std::condition_variable changed_;
    bool entered_ = false, released_ = false;
};
// Set only in this test's submitting thread; no shared mutable hook pointer.
thread_local CommitGate* current_commit_gate = nullptr;

void material_commit_case(const fs::path& root, bool request_cancel) {
    Fixture f(root / (request_cancel ? "commit-cancel" : "commit-control"));
    CommitGate gate(f.voice);
    Request train; make_request(train, "train-model");
    const std::string ending = "--result-boundary--\r\n";
    train.body.resize(train.body.size() - ending.size());
    train.body += "--result-boundary\r\nContent-Disposition: form-data; name=\"files\"; filename=\"upload.wav\"\r\n\r\n" + kWav + "\r\n" + ending;
    Response accepted{};
    std::exception_ptr submit_error;
    std::jthread submit([&] {
        current_commit_gate = &gate;
        try { accepted = request(f, train); }
        catch (...) { submit_error = std::current_exception(); }
        current_commit_gate = nullptr;
    });
    // Declared after submit: an assertion failure releases the gate before join.
    struct ReleaseGate { CommitGate& gate; ~ReleaseGate() { gate.release(); } } release{gate};
    require(gate.entered(), "commit gate was not reached inside the transaction");
    require(read(f.voice / "raw/upload.wav") == kWav && !fs::exists(f.voice / "work") &&
                !fs::exists(f.voice / "steps.txt"), "gate was not between raw/work publication");
    const auto during = status(f);
    const auto id = job_id(during);
    require(during.body.find("\"stage\":\"material-commit\"") != std::string::npos &&
                during.body.find("\"running\":true") != std::string::npos,
            "commit did not retain training reservation");
    for (const char* path : {"clips/old.wav", "model.safetensors", "model.index"})
        require(read(f.voice / path) == f.old.at(path), "commit changed old trained artifacts");
    require(f.runner->calls().empty(), "CLI started before material commit finished");
    if (request_cancel) {
        require(cancel(f, id).status == 202 && cancel(f, id).status == 200,
                "commit cancellation/duplicate not accepted");
        const auto pending = status(f);
        for (const char* field : {"\"running\":true", "\"done\":false", "\"cancel_requested\":true", "\"cancelled\":false"})
            require(pending.body.find(field) != std::string::npos, "commit cancellation falsely reported stopped");
        Request retry, step, material;
        material.method = "POST"; material.content_type = train.content_type; material.body = train.body;
        make_request(retry, "train-model"); make_request(step, "step"); material.path = "/api/material";
        require(request(f, retry).status == 409 && request(f, step).status == 409 &&
                    request(f, material).status == 409,
                "cancel released active commit reservation or GPU ownership");
    }
    gate.release(); submit.join();
    if (submit_error) std::rethrow_exception(submit_error);
    require(accepted.status == 200 && job_id(accepted) == id, "commit acceptance lost identity");
    if (request_cancel) cancelled(f); else require_terminal(f, false, "done");
    require(read(f.voice / "raw/upload.wav") == kWav && read(f.voice / "work/upload.wav") == kWav &&
                !fs::exists(f.voice / "steps.txt"), "started material commit did not finish completely");
    f.require_clean();
    if (!request_cancel) {
        require(read(f.voice / "model.safetensors") == "new-model" &&
                    read(f.voice / "model.index") == "new-index", "gate control failed normal training");
        std::cout << "commit control: pause observed; complete materials and normal publication passed\n";
        return;
    }
    require(f.runner->calls().empty(), "cancelled commit started a downstream CLI");
    for (const char* path : {"clips/old.wav", "model.safetensors", "model.index"})
        require(read(f.voice / path) == f.old.at(path), "cancelled commit replaced old trained artifacts");
    require(cancel(f, id).status == 200, "terminal duplicate cancel not idempotent");
    Request next; make_request(next, "train-model");
    f.runner->gate_on("vc_preprocess");
    const auto second = request(f, next);
    require(second.status == 200 && job_id(second) != id && f.runner->await_gate(), "next worker did not start");
    const auto second_id = job_id(second);
    require(cancel(f, id).status == 409 && status(f).body.find("\"cancel_requested\":false") != std::string::npos,
            "old commit job id cancelled new worker");
    require(cancel(f, second_id).status == 202 && cancel(f, second_id).status == 200,
            "new worker cancellation not accepted");
    require(request(f, next).status == 409 && status(f).body.find("\"running\":true") != std::string::npos,
            "new worker released reservation before its stage ended");
    f.runner->release_gate(); cancelled(f); f.require_clean();
    f.runner->reset("");
    require(request(f, next).status == 200, "normal retry after commit cancellation failed");
    require_terminal(f, false, "done"); f.require_clean();
    std::cout << "commit cancel: mid-transaction 202/duplicate 200; busy until complete; materials/old model retained; no CLI; stale id/new worker/retry passed\n";
}
#endif

void error_during_cancel(const fs::path& root) {
    Fixture f(root / "error-during-cancel");
    f.runner->reset("vc_train", {ProcessTermination::kExited, 7, 0, 0, 0});
    f.runner->gate_on("vc_train");
    Request train; make_request(train, "train-model");
    const auto accepted = request(f, train);
    require(accepted.status == 200 && f.runner->await_gate(), "failure stage not started");
    require(cancel(f, job_id(accepted)).status == 202, "failure-stage cancellation not accepted");
    f.runner->release_gate(); require_terminal(f, true, "error");
    const auto terminal = status(f);
    require(terminal.body.find("\"cancelled\":false") != std::string::npos &&
                terminal.body.find("\"cancel_requested\":true") != std::string::npos,
            "real process failure was hidden by cancellation");
    require(f.runner->calls() == std::vector<std::string>{"vc_preprocess", "vc_train"},
            "failed cancelled job ran downstream CLI");
    f.require_old(); f.require_clean(); f.runner->reset("");
    require(request(f, train).status == 200, "retry after failed cancellation rejected");
    require_terminal(f, false, "done"); f.require_clean();
}
void iteration_protocol_case(const fs::path& root) {
    Fixture f(root/"iteration-protocol");f.runner->cooperative(true);
    Request train;make_request(train,"train-model");std::string previous;fs::path previous_token;
    for(int i=0;i<2;++i) {
        f.runner->gate_on("vc_train");const auto accepted=request(f,train);
        require(accepted.status==200 && f.runner->await_gate(),"cooperative CLI not entered");
        const auto id=job_id(accepted);const auto token=f.runner->token_path();
        require(token!=previous_token && read(token)=="0","job reused cancellation token");
        if(!previous.empty())require(cancel(f,previous).status==409 && read(token)=="0","stale id touched current token");
        require(cancel(f,id).status==202 && read(token)=="1","cancel was not delivered to CLI");
        require(cancel(f,id).status==200 && read(token)=="1","duplicate token delivery failed");
        require(request(f,train).status==409 && status(f).body.find("\"running\":true")!=std::string::npos,
                "reservation released before child returned");
        f.runner->release_gate();cancelled(f);
        require(!fs::exists(token),"cancelled child token not cleaned");
        require(f.runner->calls()==std::vector<std::string>{"vc_preprocess","vc_train"},"cancelled child ran indexing");
        f.require_old();f.require_clean();previous=id;previous_token=token;f.runner->reset("");
    }
    require(request(f,train).status==200,"cooperative retry failed");require_terminal(f,false,"done");f.require_clean();
    std::cout<<"iteration protocol: delivery/duplicate/natural-return barrier/stale token/retry passed\n";
}
void dedicated_exit_error_cases(const fs::path& root) {
    const std::vector<ProcessResult> results={
        {ProcessTermination::kExited,2,0,0,0},
        {ProcessTermination::kExited,2,0,0,EIO},
        {ProcessTermination::kExited,2,0,EIO,0},
        {ProcessTermination::kExited,1,0,0,0}};
    for(std::size_t i=0;i<results.size();++i){
        Fixture f(root/("exit-error-"+std::to_string(i)));f.runner->cooperative(true);
        f.runner->reset("vc_train",results[i]);f.runner->gate_on("vc_train");Request train;make_request(train,"train-model");
        const auto accepted=request(f,train);require(accepted.status==200 && f.runner->await_gate(),"exit error gate missing");
        if(i!=0)require(cancel(f,job_id(accepted)).status==202,"concurrent cancel rejected");
        f.runner->release_gate();require_terminal(f,true,"error");
        require(status(f).body.find("\"cancelled\":false")!=std::string::npos,"error classified as cancelled");
        f.require_old();f.require_clean();
    }
    std::cout<<"dedicated exit: unsolicited exit2/system error/log error/ordinary error dominate cancellation passed\n";
}
void unsupported_step(const fs::path& root) {
    Fixture f(root/"step");f.runner->gate_on("vc_preprocess");Request req;make_request(req,"step");
    require(request(f,req).status==200 && f.runner->await_gate(),"step not started");
    require(cancel(f,job_id(status(f))).status==409,"step cancellation was enabled");
    f.runner->release_gate();require_terminal(f,false,"step-done");f.require_clean();
}
}
#ifdef RESONANTIA_TEST_MATERIAL_COMMIT_GATE
namespace resonantia::web {
void test_material_raw_installed(const std::filesystem::path& directory) {
    if (current_commit_gate) current_commit_gate->pause(directory);
}
}
#endif
int main() {
    try {
        const auto started=std::chrono::steady_clock::now();const auto fds=fd_count();fs::path path;
        {
            TempDir root(fs::temp_directory_path(),"resonantia-train-cancel-");path=root.path();
#ifdef RESONANTIA_TEST_MATERIAL_COMMIT_GATE
            material_commit_case(path, false);
            material_commit_case(path, true);
#else
            queued_case(path);
            for(const std::string program:{"vc_preprocess","vc_train","build_index"})stage_case(path,program);
            stage_case(path,"vc_preprocess",true);
            for(int n=0;n<8;++n)race_case(path,n);
            error_during_cancel(path);
            unsupported_step(path);
            iteration_protocol_case(path);
            dedicated_exit_error_cases(path);
#endif
        }
        require(!fs::exists(path) && fd_count()==fds,"cancel tests left fixtures or FDs");
        require(std::chrono::steady_clock::now()-started<30s,"cancel suite exceeded 30 seconds");
#ifdef RESONANTIA_TEST_MATERIAL_COMMIT_GATE
        std::cout << "Mid-commit cancellation: control + cancellation passed; no real CLI/syscall interception; no leftover fixtures/FD delta\n";
#else
        std::cout<<"Training cancellation: queue + 3 stages + committed upload + 8 completion races + failure during cancel + step exclusion passed; no real CLI/signals, no leftover fixtures/FD delta\n";
#endif
        return 0;
    } catch(const std::exception& e) {std::cerr<<"Cancellation contract failure: "<<e.what()<<'\n';return 1;}
}
