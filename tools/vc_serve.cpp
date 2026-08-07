/**
 * @file vc_serve.cpp
 * @brief Minimal pure-C++ HTTP server exposing the voice-conversion pipeline.
 *
 * Serves the static WebUI and a single JSON-free endpoint the frontend calls:
 *   POST /api/convert  (multipart/form-data: audio + params) -> WAV bytes.
 * Hand-written HTTP/1.1 over POSIX sockets — no third-party dependency, no
 * Python at runtime. Single-threaded (one conversion at a time; local tool).
 *
 * Usage:
 *   vc_serve --hubert <path> --model <G.safetensors> --rmvpe <path>
 *            [--index <index>] [--port 8080] [--webroot <dir>]
 *            [--speakers 109] [--version v2] [--sr 40000]
 */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "voxmutatio/core/device.h"
#include "voxmutatio/core/types.h"
#include "voxmutatio/pipeline/pipeline.h"

namespace {

using voxmutatio::pipeline::VoiceConversionPipeline;

struct Request {
    std::string method, path;
    std::string content_type;
    std::string body;
};

struct Part {
    std::string name, filename, data;
};

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// Read a full HTTP request (headers + Content-Length body, binary-safe).
bool read_request(int fd, Request& req) {
    std::string buf;
    char tmp[8192];
    // Read until end of headers.
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf.append(tmp, n);
        header_end = buf.find("\r\n\r\n");
        if (buf.size() > (64u << 20)) return false;  // 64MB guard
    }
    std::string head = buf.substr(0, header_end);
    std::string body = buf.substr(header_end + 4);

    std::istringstream hs(head);
    std::string line;
    std::getline(hs, line);
    {
        std::istringstream ls(line);
        ls >> req.method >> req.path;
    }
    size_t content_length = 0;
    while (std::getline(hs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = lower(line.substr(0, colon));
        std::string val = line.substr(colon + 1);
        while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) val.erase(val.begin());
        if (key == "content-length") content_length = std::stoul(val);
        else if (key == "content-type") req.content_type = val;
    }
    // Read the rest of the body.
    while (body.size() < content_length) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) break;
        body.append(tmp, n);
    }
    req.body = std::move(body);
    return true;
}

std::vector<Part> parse_multipart(const std::string& body, const std::string& boundary) {
    std::vector<Part> parts;
    std::string delim = "--" + boundary;
    size_t pos = body.find(delim);
    while (pos != std::string::npos) {
        pos += delim.size();
        if (body.compare(pos, 2, "--") == 0) break;  // closing boundary
        if (body.compare(pos, 2, "\r\n") == 0) pos += 2;
        size_t hend = body.find("\r\n\r\n", pos);
        if (hend == std::string::npos) break;
        std::string headers = body.substr(pos, hend - pos);
        size_t cstart = hend + 4;
        size_t next = body.find(delim, cstart);
        if (next == std::string::npos) break;
        size_t cend = next;
        if (cend >= 2 && body.compare(cend - 2, 2, "\r\n") == 0) cend -= 2;
        Part part;
        part.data = body.substr(cstart, cend - cstart);
        // Parse Content-Disposition name / filename.
        std::string h = lower(headers);
        auto grab = [&](const std::string& key) -> std::string {
            size_t k = h.find(key + "=\"");
            if (k == std::string::npos) return "";
            k += key.size() + 2;
            size_t e = headers.find('"', k);
            return (e == std::string::npos) ? "" : headers.substr(k, e - k);
        };
        part.name = grab("name");
        part.filename = grab("filename");
        parts.push_back(std::move(part));
        pos = next;
    }
    return parts;
}

void send_response(int fd, int status, const std::string& status_text,
                   const std::string& content_type, const std::string& body) {
    std::ostringstream os;
    os << "HTTP/1.1 " << status << " " << status_text << "\r\n"
       << "Content-Type: " << content_type << "\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Connection: close\r\n\r\n";
    std::string head = os.str();
    send(fd, head.data(), head.size(), MSG_NOSIGNAL);
    if (!body.empty()) send(fd, body.data(), body.size(), MSG_NOSIGNAL);
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string mime_for(const std::string& path) {
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".html") return "text/html; charset=utf-8";
    if (path.size() >= 3 && path.substr(path.size() - 3) == ".js") return "application/javascript";
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".css") return "text/css";
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".wav") return "audio/wav";
    return "application/octet-stream";
}

void serve_static(int fd, const std::string& webroot, const std::string& path) {
    std::string rel = (path == "/") ? "/index.html" : path;
    if (rel.find("..") != std::string::npos) { send_response(fd, 403, "Forbidden", "text/plain", "no"); return; }
    std::string full = webroot + rel;
    std::string content = read_file(full);
    if (content.empty()) { send_response(fd, 404, "Not Found", "text/plain", "not found"); return; }
    send_response(fd, 200, "OK", mime_for(full), content);
}

std::string field(const std::vector<Part>& parts, const std::string& name, const std::string& def) {
    for (const auto& p : parts) if (p.name == name && p.filename.empty()) return p.data;
    return def;
}

void handle_convert(int fd, VoiceConversionPipeline& pipe, const Request& req) {
    size_t b = req.content_type.find("boundary=");
    if (b == std::string::npos) { send_response(fd, 400, "Bad Request", "text/plain", "no boundary"); return; }
    std::string boundary = req.content_type.substr(b + 9);
    if (!boundary.empty() && boundary.front() == '"') boundary = boundary.substr(1, boundary.size() - 2);
    auto parts = parse_multipart(req.body, boundary);

    const Part* audio = nullptr;
    for (const auto& p : parts) if (!p.filename.empty()) { audio = &p; break; }
    if (!audio) { send_response(fd, 400, "Bad Request", "text/plain", "no audio"); return; }

    int speaker = std::atoi(field(parts, "speaker_id", "0").c_str());
    int f0_up = std::atoi(field(parts, "f0_up_key", "0").c_str());
    double index_rate = std::atof(field(parts, "index_rate", "0").c_str());
    double rms_mix = std::atof(field(parts, "rms_mix_rate", "0.5").c_str());
    double protect = std::atof(field(parts, "protect", "0.5").c_str());

    std::string in_path = "/tmp/vcserve_in.wav";
    std::string out_path = "/tmp/vcserve_out.wav";
    { std::ofstream o(in_path, std::ios::binary); o.write(audio->data.data(), audio->data.size()); }

    pipe.set_runtime_params(f0_up, 0.0, index_rate, rms_mix, protect);
    auto res = pipe.convert_file(in_path, out_path, speaker);
    if (!res.success) {
        send_response(fd, 500, "Internal Server Error", "text/plain", res.error_message);
        return;
    }
    std::string wav = read_file(out_path);
    printf("[vc_serve] converted %zu bytes -> %zu bytes (spk=%d pitch=%d idx=%.2f) %.0fms\n",
           audio->data.size(), wav.size(), speaker, f0_up, index_rate, res.total_ms);
    send_response(fd, 200, "OK", "audio/wav", wav);
}

}  // namespace

int main(int argc, char** argv) {
    voxmutatio::VCConfig cfg;
    int port = 8080;
    std::string webroot = "webui";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--hubert") cfg.hubert_model_path = next();
        else if (a == "--model") cfg.synthesizer_model_path = next();
        else if (a == "--rmvpe") cfg.rmvpe_model_path = next();
        else if (a == "--index") cfg.index_path = next();
        else if (a == "--port") port = std::atoi(next());
        else if (a == "--webroot") webroot = next();
        else if (a == "--speakers") cfg.num_speakers = std::atoi(next());
        else if (a == "--sr") cfg.model_sample_rate = std::atoi(next());
        else if (a == "--version") { std::string v = next(); cfg.version = (v == "v2") ? voxmutatio::ModelVersion::kV2 : voxmutatio::ModelVersion::kV1; }
        else { fprintf(stderr, "Unknown argument: %s\n", a.c_str()); return 1; }
    }
    if (cfg.hubert_model_path.empty() || cfg.synthesizer_model_path.empty()) {
        fprintf(stderr, "error: --hubert and --model are required\n");
        return 1;
    }

    std::signal(SIGPIPE, SIG_IGN);
    voxmutatio::Device device;
    if (auto err = device.init("cuda", 0)) { fprintf(stderr, "CUDA init: %s\n", err->c_str()); return 1; }

    VoiceConversionPipeline pipe;
    fprintf(stderr, "Loading models...\n");
    if (!pipe.init(cfg)) { fprintf(stderr, "pipeline init failed\n"); return 1; }

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        fprintf(stderr, "bind failed on port %d\n", port); return 1;
    }
    if (listen(srv, 8) < 0) { fprintf(stderr, "listen failed\n"); return 1; }
    printf("Resonantia WebUI on http://localhost:%d  (webroot: %s)\n", port, webroot.c_str());

    for (;;) {
        int fd = accept(srv, nullptr, nullptr);
        if (fd < 0) continue;
        Request req;
        if (read_request(fd, req)) {
            if (req.method == "POST" && req.path == "/api/convert") handle_convert(fd, pipe, req);
            else if (req.method == "GET") serve_static(fd, webroot, req.path);
            else send_response(fd, 405, "Method Not Allowed", "text/plain", "no");
        }
        close(fd);
    }
    return 0;
}
