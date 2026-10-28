// Host-only HTTP upload contracts. Model processes are always mocked.
#include "tools/web/web_service.h"
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <future>
#include <thread>

namespace {
namespace fs = std::filesystem;
using namespace resonantia::web;
void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
void write(const fs::path& path, const std::string& bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary); out << bytes;
    out.close(); check(bool(out), "fixture write failed");
}
std::string pcm() {
    std::string wav = "RIFF";
    auto le = [&](unsigned n, int bytes) { while (bytes--) { wav += char(n & 255); n >>= 8; } };
    le(44, 4); wav += "WAVEfmt "; le(16, 4); le(1, 2); le(1, 2);
    le(44100, 4); le(88200, 4); le(2, 2); le(16, 2); wav += "data";
    le(8, 4); le(10, 2); le(20, 2); le(30, 2); le(40, 2);
    return wav;
}
struct Runner : ProcessRunner {
    mutable std::atomic<int> decodes{0}, models{0};
    std::function<ProcessResult(const ProcessSpec&)> decode;
    mutable std::vector<std::vector<std::string>> model_commands;
    bool cleanup_success = false;
    bool corrupt_cleanup = false;
    ProcessResult run(const ProcessSpec& spec) const override {
        if (fs::path(spec.argv.front()).filename() == "timeout") {
            ++decodes;
            if (decode) return decode(spec);
            write(spec.argv.back(), pcm());
            return {ProcessTermination::kExited, 0};
        }
        ++models;
        model_commands.push_back(spec.argv);
        for (std::size_t i = 0; i + 1 < spec.argv.size(); ++i) {
            if (spec.argv[i] == "--output-dir") write(fs::path(spec.argv[i+1])/"processed.wav",
                corrupt_cleanup ? "RIFFxxxxWAVEinvalid-result" : pcm());
        }
        for (std::size_t i = 0; i + 1 < spec.argv.size(); ++i)
            if (spec.argv[i] == "--output") write(spec.argv[i + 1], pcm());
        // Training stops at its first mocked CLI. Never starts a real model.
        return {ProcessTermination::kExited, spec.argv.front().ends_with("vc_convert") ||
            (cleanup_success && spec.argv.front().ends_with("vc_preprocess")) ? 0 : 1};
    }
};
struct Reply { int status; std::string body; };
Reply request(const std::shared_ptr<WebService>& service, const std::string& route,
              const std::vector<std::pair<std::string, std::string>>& files,
              const std::string& extra = "", const std::string& method = "POST") {
    Request req; req.method = method; req.path = route;
    req.content_type = "multipart/form-data; boundary=audio-test";
    req.body = "--audio-test\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\nvoice\r\n";
    req.body += extra;
    for (const auto& [name, bytes] : files)
        req.body += "--audio-test\r\nContent-Disposition: form-data; name=\"files\"; filename=\"" +
                    name + "\"\r\n\r\n" + bytes + "\r\n";
    req.body += "--audio-test--\r\n";
    int sockets[2]; check(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) == 0, "socketpair");
    std::exception_ptr failure;
    std::thread worker([&] {
        try { service->handle_request(sockets[0], req); } catch (...) { failure = std::current_exception(); }
        close(sockets[0]);
    });
    std::string wire; char buffer[8192]; ssize_t n;
    while ((n = read(sockets[1], buffer, sizeof buffer)) > 0) wire.append(buffer, n);
    close(sockets[1]); worker.join();
    if (failure) std::rethrow_exception(failure);
    check(wire.starts_with("HTTP/1.1 "), "missing HTTP response");
    return {std::stoi(wire.substr(9, 3)), wire.substr(wire.find("\r\n\r\n") + 4)};
}
void run() {
    TempDir temp("/tmp", "resonantia-audio-contract-");
    Paths paths; paths.runs = temp.path() + "/runs"; paths.build = temp.path() + "/build";
    paths.ffmpeg = "/bin/true"; // Host contracts do not depend on installed codecs.
    auto runner = std::make_shared<Runner>();
    auto service = std::make_shared<WebService>(paths, runner);
    const auto field = [](const std::string& key, const std::string& value) {
        return "--audio-test\r\nContent-Disposition: form-data; name=\""+key+"\"\r\n\r\n"+value+"\r\n";
    };
    const std::string candidate = "candidate_epoch_0000000200.safetensors";
    write(fs::path(paths.runs)/"voice/full_training/checkpoints"/candidate, "trained-model");
    write(fs::path(paths.runs)/"voice/full_training/result/model.index", "trained-index");
    auto release_fields = field("candidate",candidate)+field("publish_name","approved");
    check(request(service,"/api/train/publish",{},release_fields).status==400,"publication requires review");
    release_fields+=field("reviewed","1");
    check(request(service,"/api/train/publish",{},release_fields).status==200,"reviewed candidate publication");
    check(read_file(paths.runs+"/approved/model.safetensors")=="trained-model","wrong candidate published");
    check(read_file(paths.runs+"/approved/model.index")=="trained-index","index not published");
    check(request(service,"/api/train/publish",{},release_fields).status==409,"existing voice must not be overwritten");
    check(request(service,"/api/train/publish",{},field("candidate","../model.safetensors")+
                  field("publish_name","escape")+field("reviewed","1")).status==400,"candidate traversal");
    check(request(service,"/api/train/runs",{},"","GET").body.find(candidate)!=std::string::npos,"candidate discovery");
    check(request(service,"/api/convert",{{"source.wav",pcm()}},field("filter_radius","2")).status==400,
          "even F0 window must fail before model execution");
    check(request(service,"/api/convert",{{"source.wav",pcm()}},field("workflow","bogus")).status==400,"unknown workflow");
    check(request(service,"/api/convert",{{"source.wav",pcm()}},field("output_kind","mix")).status==400,"mix without song");
    check(request(service,"/api/convert",{{"source.wav",pcm()}},field("pitch_edits","smooth:0:4")).status==400,"oversized pitch edit");
    check(runner->models==0,"invalid fields must not start model processes");
    runner->cleanup_success=true;
    auto cover = request(service,"/api/convert",{{"source.wav",pcm()}},field("workflow","song")+field("output_kind","mix"));
    check(cover.status==200 && cover.body.size()>24 && cover.body[22]==2,"cover must return stereo WAV");
    check(runner->model_commands.size()==3,"song must run two cleanup stages then conversion");
    for (size_t i=0;i<2;++i) {
        const auto& argv=runner->model_commands[i];
        check(std::find(argv.begin(),argv.end(),i==0?"--separate":"--deharmony")!=argv.end(),"cleanup order");
        check(std::find(argv.begin(),argv.end(),"--preserve-stereo")!=argv.end(),"cleanup must preserve stereo");
    }
    runner->corrupt_cleanup=true;runner->model_commands.clear();
    check(request(service,"/api/convert",{{"source.wav",pcm()}},field("workflow","harmonies")).status==500,
          "corrupt cleanup output is a server error, not a client validation error");
    check(runner->model_commands.size()==1,"corrupt cleanup must not reach conversion");
    runner->corrupt_cleanup=false;
    runner->models=0;runner->model_commands.clear();runner->cleanup_success=false;
    const std::string stereo_field = "--audio-test\r\nContent-Disposition: form-data; name=\"preserve_stereo\"\r\n\r\n1\r\n";
    runner->decode = [&](const ProcessSpec& spec) {
        auto ac = std::find(spec.argv.begin(), spec.argv.end(), "-ac");
        check(ac != spec.argv.end() && *(ac+1) == "2", "stereo request lost before decoder");
        auto bytes = pcm(); bytes[22] = 2; bytes[28] = char(176400 & 255);
        bytes[29] = char((176400 >> 8) & 255); bytes[30] = char(176400 >> 16); bytes[32] = 4;
        write(spec.argv.back(), bytes); return ProcessResult{ProcessTermination::kExited, 0};
    };
    check(request(service, "/api/material", {{"stereo.mp3", "encoded"}}, stereo_field).status == 200,
          "stereo material decode failed");
    check(read_file(paths.runs + "/voice/raw/stereo.wav")[22] == 2, "stereo material was downmixed");
    runner->decode = {}; runner->decodes = 0;
    auto reply = request(service, "/api/material", {{"song.MP3", "compressed"}});
    check(reply.status == 200, "MP3 material must normalize successfully; got " + std::to_string(reply.status));
    check(read_file(paths.runs + "/voice/raw/song.wav") == pcm(), "normalized WAV missing");
    check(runner->decodes == 1 && runner->models == 0, "upload must decode once without models");
    for (const auto& extension : {"wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "wma", "aiff", "aif"}) {
        const auto file = std::string("audio.") + extension;
        reply = request(service, "/api/material", {{file, std::string(extension) == "wav" ? pcm() : "encoded"}});
        check(reply.status == 200, "format rejected: " + file);
        check(read_file(paths.runs + "/voice/raw/audio.wav") == pcm(), "wrong normalized output");
    }
    const std::string old = "RIFFxxxxWAVEunchanged-wav-bytes";
    check(request(service, "/api/material", {{"old.WAV ", old}}).status == 200,
          "filename sanitization must preserve the existing WAV path");
    check(request(service, "/api/material", {{"old.WAV", old}}).status == 200, "WAV compatibility");
    write(paths.runs + "/voice/steps.txt", "dereverb\n");
    const auto preserved = [&] {
        check(read_file(paths.runs + "/voice/raw/old.wav") == old &&
              read_file(paths.runs + "/voice/work/old.wav") == old &&
              read_file(paths.runs + "/voice/steps.txt") == "dereverb\n", "failed batch changed old material");
        for (const auto& entry : fs::recursive_directory_iterator(paths.runs))
            check(!entry.path().filename().string().starts_with('.'), "staging not cleaned");
    };
    for (const auto& files : std::vector<std::vector<std::pair<std::string, std::string>>>{
             {{"a.mp3", "x"}, {"a.FLAC", "x"}}, {{"a.mp4", "x"}}, {{"a.mp3", ""}}, {{"a.wav", "bad"}}}) {
        const auto before = runner->decodes.load();
        check(request(service, "/api/material", files).status == 400, "bad upload accepted");
        check(runner->decodes == before, "invalid batch launched decoder"); preserved();
    }
    for (const auto& route : {"/api/material", "/api/train", "/api/preprocess", "/api/convert"}) {
        for (const int code : {1, 124, 137}) {
            runner->decode = [code](const ProcessSpec& spec) {
                write(spec.argv.back(), pcm()); // Valid-looking partial output MUST be rejected.
                return ProcessResult{ProcessTermination::kExited, code};
            };
            const auto before = runner->models.load();
            reply = request(service, route, {{"a.mp3", "bad"}});
            check(reply.status == 400, std::string(route) + " decoder failure not propagated");
            if (code != 1) check(reply.body.find("timed out") != std::string::npos, "timeout diagnostic missing");
            check(runner->models == before, "model called after decode failure"); preserved();
        }
    }
    runner->decode = [](const ProcessSpec& spec) {
        write(spec.argv.back(), "RIFFxxxxWAVEbroken");
        return ProcessResult{ProcessTermination::kExited, 0};
    };
    check(request(service, "/api/material", {{"a.mp3", "x"}}).status == 400, "malformed decoded WAV accepted");
    preserved();
    runner->decode = [](const ProcessSpec& spec) {
        write(spec.argv.back(), pcm()); fs::resize_file(spec.argv.back(), (64u << 20) + 1);
        return ProcessResult{ProcessTermination::kExited, 0};
    };
    check(request(service, "/api/material", {{"a.mp3", "x"}}).status == 413, "truncated oversized output accepted");
    preserved();
    runner->decode = [](const ProcessSpec& spec) {
        std::string wav = pcm(); const unsigned size = 48u << 20;
        for (int i = 0; i < 4; ++i) { wav[4+i] = char((size-8) >> (8*i)); wav[40+i] = char((size-44) >> (8*i)); }
        write(spec.argv.back(), wav); fs::resize_file(spec.argv.back(), size);
        return ProcessResult{ProcessTermination::kExited, 0};
    };
    check(request(service, "/api/material", {{"a.mp3", "x"}, {"b.mp3", "x"}, {"c.mp3", "x"}}).status == 413,
          "aggregate decoded limit not enforced"); preserved();
    // Whole batch stays staged even if an earlier file decoded successfully.
    int count = 0;
    runner->decode = [&](const ProcessSpec& spec) {
        write(spec.argv.back(), pcm()); return ProcessResult{ProcessTermination::kExited, ++count == 2 ? 1 : 0};
    };
    check(request(service, "/api/material", {{"a.mp3", "x"}, {"b.mp3", "x"}}).status == 400, "partial batch committed");
    preserved();

    std::promise<void> entered, release;
    auto gate = release.get_future().share();
    runner->decode = [&](const ProcessSpec& spec) {
        entered.set_value(); gate.wait(); write(spec.argv.back(), pcm());
        return ProcessResult{ProcessTermination::kExited, 1};
    };
    auto pending = std::async(std::launch::async, [&] { return request(service, "/api/material", {{"a.mp3", "x"}}); });
    entered.get_future().wait();
    reply = request(service, "/api/material", {{"b.mp3", "x"}});
    release.set_value(); check(pending.get().status == 400, "first decoder result");
    check(reply.status == 409, "concurrent decoder was not rejected"); preserved();

    paths.ffmpeg = temp.path() + "/missing-ffmpeg";
    auto missing = std::make_shared<WebService>(paths, runner);
    check(request(missing, "/api/material", {{"a.mp3", "x"}}).status == 503, "missing dependency not reported"); preserved();
    check(request(missing, "/api/material", {{"old.WAV", old}}).status == 200, "WAV requires FFmpeg");
    runner->decode = {};
    check(request(service, "/api/preprocess", {{"a.aac", "x"}}).body == pcm(), "preprocess did not normalize audio");
    check(request(service, "/api/convert", {{"a.flac", "x"}}).status == 200, "convert did not normalize audio");
    check(request(service, "/api/train", {{"a.wma", "x"}}).status == 200, "training did not accept normalized material");
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (service.use_count() != 1 && std::chrono::steady_clock::now() < end) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    check(service.use_count() == 1, "mock training worker did not retire");
    check(read_file(paths.runs + "/voice/raw/a.wav") == pcm(), "training raw not normalized");

    // Exercise the same deadline wrapper and production process runner with a
    // short sleeping fixture. No test waits for the production 120 s deadline.
    ProcessSpec deadline;
    deadline.argv = {"/usr/bin/timeout", "--kill-after=0.1s", "0.1s",
                     fs::read_symlink("/proc/self/exe").string(), "--sleep-child"};
    deadline.log_path = temp.path() + "/deadline.log";
    const auto started = std::chrono::steady_clock::now();
    const auto result = PosixProcessRunner().run(deadline);
    check(result.exit_code == 124, "deadline wrapper did not time out");
    check(std::chrono::steady_clock::now() - started < std::chrono::seconds(3), "deadline failed to bound wait");
    const auto pid = std::stoi(read_file(deadline.log_path));
    check(kill(pid, 0) == -1 && errno == ESRCH, "timed out child was not reaped");
}

// Real decoder integration: one second of provenance-tracked human speech,
// transcoded only for codec coverage. Does not execute a voice/model process.
void codecs(const std::string& ffmpeg, const std::string& speech) {
    TempDir temp("/tmp", "resonantia-audio-codecs-");
    Paths paths; paths.runs = temp.path() + "/runs"; paths.ffmpeg = ffmpeg;
    auto runner = std::make_shared<PosixProcessRunner>();
    auto service = std::make_shared<WebService>(paths, runner);
    const std::vector<std::pair<std::string, std::string>> encoders{
        {"wav", "pcm_s16le"}, {"MP3", "libmp3lame"}, {"m4a", "aac"}, {"m4a", "alac"},
        {"aac", "aac"}, {"flac", "flac"}, {"ogg", "libvorbis"}, {"opus", "libopus"},
        {"ogg", "libopus"}, {"wma", "wmav2"}, {"aiff", "pcm_s16be"}, {"aif", "pcm_s16be"}};
    for (const auto& [ext, encoder] : encoders) {
        const auto file = fs::path(temp.path()) / ("speech." + ext);
        ProcessSpec spec;
        spec.argv = {ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", speech,
                     "-t", "1", "-ac", "1", "-ar", "44100", "-c:a", encoder, file.string()};
        if (encoder == "libopus") spec.argv[spec.argv.size()-4] = "48000";
        spec.log_path = fs::path(temp.path()) / "encode.log";
        check(runner->run(spec).success(), "fixture encoding failed: " + ext + "/" + encoder + " " + log_tail(spec.log_path));
        const auto reply = request(service, "/api/material", {{file.filename().string(), read_file(file)}});
        check(reply.status == 200, "real codec failed: " + ext + "/" + encoder + " " + reply.body);
        const auto decoded = read_file(paths.runs + "/voice/work/speech.wav");
        // One-second AAC/WMA frames may contain padding. Assert nonempty data,
        // normalization and duration within 150 ms, not perceptual quality.
        const auto data = decoded.find("data", 36);
        check(decoded.starts_with("RIFF") && data != std::string::npos, "decoded PCM missing");
        const auto n = decoded.size() - data - 8;
        check(n > 74970 && n < 101430, "decoded duration outside 1 +/- .15 seconds");
        check(decoded.find_first_not_of('\0', data + 8) != std::string::npos, "real speech became silence");
        std::cout << ext << '/' << encoder << ": " << n / 88200.0 << " s\n";
    }
    for (const auto& ext : {"mp3", "m4a", "aac", "flac", "ogg", "opus", "wma", "aiff"})
        check(request(service, "/api/material", {{std::string("bad.") + ext, "#EXTM3U\nhttp://127.0.0.1/secret\n"}}).status == 400,
              "playlist/invalid audio accepted");
    check(request(service, "/api/material", {{"wrong.mp3", read_file(temp.path() + "/speech.flac")}}).status == 400,
          "wrong container extension accepted");
    ProcessSpec no_audio;
    const auto video = temp.path() + "/video.m4a";
    no_audio.argv = {ffmpeg, "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=size=16x16:duration=0.1", "-c:v", "mpeg4", "-an", "-f", "mp4", video};
    no_audio.log_path = temp.path() + "/video.log";
    check(runner->run(no_audio).success(), "no-audio fixture failed");
    check(request(service, "/api/material", {{"video.m4a", read_file(video)}}).status == 400, "video without audio accepted");
}
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--sleep-child") {
            std::cout << getpid() << std::endl;
            for (;;) pause();
        }
        if (argc == 3) codecs(argv[1], argv[2]); else run();
        std::cout << "Web audio upload checks passed\n"; return 0;
    }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
