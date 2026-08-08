/**
 * @file vc_serve.cpp
 * @brief Pure-C++ HTTP server that orchestrates the full Resonantia workflow.
 *
 * Serves the WebUI and drives training + inference by spawning the verified
 * CLI tools (vc_preprocess, vc_train, build_index, vc_convert) as subprocesses.
 * No inline model loading -> no GPU contention with training; each stage reuses
 * an already-tested binary. Hand-written HTTP/1.1 over POSIX sockets, no
 * third-party dependency, no Python at runtime.
 *
 * Endpoints:
 *   GET  /                     -> webui/index.html (+ static assets)
 *   GET  /api/voices           -> {"voices":[...],"training":{...}}
 *   POST /api/train            -> multipart {files[], name, mode, steps, seg}
 *   GET  /api/train/status     -> {"stage","running","done","error","log"}
 *   POST /api/convert          -> multipart {audio, voice, f0_up_key, ...} -> WAV
 *
 * Usage:
 *   vc_serve [--port 8080] [--webroot webui] [--repo <path>]
 */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ---- resolved paths (set in main) ----
struct Paths {
    std::string repo, models, build, runs, webroot;
    std::string hubert, rmvpe, gmodel, dmodel;
} g;

// ---- single training job state ----
struct Job {
    std::mutex mu;
    std::string name, stage = "idle";
    std::atomic<bool> running{false};
    bool done = false, error = false;
} g_job;

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// Only allow a safe subset for names used in shell commands / paths.
bool safe_name(const std::string& s) {
    if (s.empty() || s.size() > 64) return false;
    for (char c : s)
        if (!(std::isalnum((unsigned char)c) || c == '_' || c == '-')) return false;
    return true;
}
std::string safe_filename(const std::string& s) {
    std::string base = fs::path(s).filename().string(), out;
    for (char c : base)
        if (std::isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.') out += c;
    return out.empty() ? "clip.wav" : out;
}
std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': break;
            case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; std::snprintf(b, 8, "\\u%04x", c); o += b; }
                else o += c;
        }
    }
    return o;
}

// ---- HTTP ----
struct Request { std::string method, path, query, content_type, body; };
struct Part { std::string name, filename, data; };

bool read_request(int fd, Request& req) {
    std::string buf; char tmp[8192];
    size_t he = std::string::npos;
    while (he == std::string::npos) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf.append(tmp, n);
        he = buf.find("\r\n\r\n");
        if (buf.size() > (256u << 20)) return false;
    }
    std::string head = buf.substr(0, he), body = buf.substr(he + 4);
    std::istringstream hs(head); std::string line; std::getline(hs, line);
    { std::istringstream ls(line); std::string target; ls >> req.method >> target;
      auto q = target.find('?');
      if (q == std::string::npos) req.path = target;
      else { req.path = target.substr(0, q); req.query = target.substr(q + 1); } }
    size_t clen = 0;
    while (std::getline(hs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto c = line.find(':'); if (c == std::string::npos) continue;
        std::string k = lower(line.substr(0, c)), v = line.substr(c + 1);
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
        if (k == "content-length") clen = std::stoul(v);
        else if (k == "content-type") req.content_type = v;
    }
    while (body.size() < clen) { ssize_t n = recv(fd, tmp, sizeof(tmp), 0); if (n <= 0) break; body.append(tmp, n); }
    req.body = std::move(body);
    return true;
}

std::vector<Part> parse_multipart(const std::string& body, const std::string& boundary) {
    std::vector<Part> parts; std::string delim = "--" + boundary;
    size_t pos = body.find(delim);
    while (pos != std::string::npos) {
        pos += delim.size();
        if (body.compare(pos, 2, "--") == 0) break;
        if (body.compare(pos, 2, "\r\n") == 0) pos += 2;
        size_t hend = body.find("\r\n\r\n", pos); if (hend == std::string::npos) break;
        std::string headers = body.substr(pos, hend - pos);
        size_t cstart = hend + 4, next = body.find(delim, cstart);
        if (next == std::string::npos) break;
        size_t cend = next; if (cend >= 2 && body.compare(cend - 2, 2, "\r\n") == 0) cend -= 2;
        Part part; part.data = body.substr(cstart, cend - cstart);
        std::string h = lower(headers);
        auto grab = [&](const std::string& key) -> std::string {
            size_t k = h.find(key + "=\""); if (k == std::string::npos) return "";
            k += key.size() + 2; size_t e = headers.find('"', k);
            return (e == std::string::npos) ? "" : headers.substr(k, e - k);
        };
        part.name = grab("name"); part.filename = grab("filename");
        parts.push_back(std::move(part)); pos = next;
    }
    return parts;
}

void send_response(int fd, int status, const char* text, const std::string& ctype, const std::string& body) {
    std::ostringstream os;
    os << "HTTP/1.1 " << status << " " << text << "\r\nContent-Type: " << ctype
       << "\r\nContent-Length: " << body.size() << "\r\nConnection: close\r\n\r\n";
    std::string h = os.str();
    send(fd, h.data(), h.size(), MSG_NOSIGNAL);
    if (!body.empty()) send(fd, body.data(), body.size(), MSG_NOSIGNAL);
}
void send_json(int fd, const std::string& j) { send_response(fd, 200, "OK", "application/json", j); }

std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary); if (!f) return "";
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}
std::string mime_for(const std::string& p) {
    if (p.size() >= 5 && p.substr(p.size() - 5) == ".html") return "text/html; charset=utf-8";
    if (p.size() >= 3 && p.substr(p.size() - 3) == ".js") return "application/javascript";
    if (p.size() >= 4 && p.substr(p.size() - 4) == ".css") return "text/css";
    if (p.size() >= 4 && p.substr(p.size() - 4) == ".wav") return "audio/wav";
    return "application/octet-stream";
}
void serve_static(int fd, const std::string& path) {
    std::string rel = (path == "/") ? "/index.html" : path;
    if (rel.find("..") != std::string::npos) { send_response(fd, 403, "Forbidden", "text/plain", "no"); return; }
    std::string full = g.webroot + rel, content = read_file(full);
    if (content.empty()) { send_response(fd, 404, "Not Found", "text/plain", "not found"); return; }
    send_response(fd, 200, "OK", mime_for(full), content);
}
std::string field(const std::vector<Part>& ps, const std::string& n, const std::string& def) {
    for (auto& p : ps) if (p.name == n && p.filename.empty()) return p.data;
    return def;
}
std::string query_get(const std::string& q, const std::string& key) {
    std::istringstream ss(q); std::string kv;
    while (std::getline(ss, kv, '&')) { auto e = kv.find('='); if (e != std::string::npos && kv.substr(0, e) == key) return kv.substr(e + 1); }
    return "";
}

// ---- workflow ----
void set_stage(const std::string& s) { std::lock_guard<std::mutex> lk(g_job.mu); g_job.stage = s; }
void finish(const std::string& stage, bool error) {
    std::lock_guard<std::mutex> lk(g_job.mu);
    g_job.stage = stage; g_job.error = error; g_job.done = true; g_job.running = false;
}

void run_training(std::string name, std::string mode, int steps, int seg, int epochs, bool separate, bool dereverb, bool denoise, bool deharmony, bool vad) {
    std::string dir = g.runs + "/" + name, log = dir + "/train.log";
    std::string clips = dir + "/clips", model = dir + "/model.safetensors", index = dir + "/model.index";
    auto sh = [&](const std::string& cmd) {
        std::string full = "{ " + cmd + " ; } >> '" + log + "' 2>&1";
        return std::system(full.c_str());
    };
    { std::ofstream(log, std::ios::trunc) << "[serve] training '" << name << "' mode=" << mode
        << (epochs > 0 ? " epochs=" + std::to_string(epochs) : " steps=" + std::to_string(steps))
        << " seg=" << seg << (separate ? " +separate" : "") << "\n"; }

    // If the material was preprocessed step-by-step (runs/<name>/work), slice that
    // (separation already applied); otherwise slice raw and apply separation flags.
    bool preprocessed = fs::exists(dir + "/work") && !fs::is_empty(dir + "/work");
    std::string src = preprocessed ? dir + "/work" : dir + "/raw";
    set_stage((!preprocessed && (separate || dereverb || denoise)) ? "separating" : "preprocessing");
    std::string sep_flags;
    if (!preprocessed) {
        if (separate) sep_flags += " --separate";
        if (deharmony) sep_flags += " --deharmony";
        if (dereverb) sep_flags += " --dereverb";
        if (denoise) sep_flags += " --denoise";
        if (!sep_flags.empty()) sep_flags += " --sep-dir '" + g.models + "/separation'";
    }
    // Smart slicing (Silero VAD) supersedes RMS --trim when enabled.
    std::string slice_flags = vad ? " --vad --vad-dir '" + g.models + "/vad'" : " --trim";
    int rc = sh("'" + g.build + "/vc_preprocess' --input '" + src + "' --output-dir '" + clips +
                "' --sr 40000 --seg-sec 3.0" + slice_flags + sep_flags +
                " 2>&1 | grep --line-buffered -E 'clip|Wrote|error|skip|separation|reverb|noise|speech|enabled'");
    if (rc != 0) { finish("error", true); return; }

    set_stage("training");
    std::string t = "'" + g.build + "/vc_train'";
    if (mode == "gan") t += " --gan --dmodel '" + g.dmodel + "'";
    t += " --hubert '" + g.hubert + "' --rmvpe '" + g.rmvpe + "' --pretrained '" + g.gmodel +
         "' --target '" + clips + "' --out '" + model + "'" +
         (epochs > 0 ? " --epochs " + std::to_string(epochs) : " --steps " + std::to_string(steps)) +
         " --seg " + std::to_string(seg) +
         " 2>&1 | grep --line-buffered -E 'step|mel|Device|frames|Epochs|Exported|clips|done|error|GAN'";
    rc = sh(t);
    if (rc != 0) { finish("error", true); return; }

    set_stage("indexing");
    sh("'" + g.build + "/build_index' --hubert '" + g.hubert + "' --input-dir '" + clips +
       "' --output '" + index + "' 2>&1 | grep --line-buffered -E 'Wrote|index|error|frames'");

    std::ofstream(log, std::ios::app) << "[serve] done.\n";
    finish("done", false);
}

std::string voices_json() {
    std::string j = "{\"base\":\"base (pretrained)\",\"voices\":[";
    bool first = true;
    if (fs::is_directory(g.runs)) {
        std::vector<std::string> names;
        for (auto& e : fs::directory_iterator(g.runs))
            if (e.is_directory() && fs::exists(e.path() / "model.safetensors"))
                names.push_back(e.path().filename().string());
        std::sort(names.begin(), names.end());
        for (auto& n : names) { if (!first) j += ","; j += "\"" + json_escape(n) + "\""; first = false; }
    }
    { std::lock_guard<std::mutex> lk(g_job.mu);
      j += "],\"training\":{\"running\":" + std::string(g_job.running ? "true" : "false") +
           ",\"name\":\"" + json_escape(g_job.name) + "\",\"stage\":\"" + json_escape(g_job.stage) + "\"}}"; }
    return j;
}

// ---- per-step material preprocessing (apply each step individually, preview, then train) ----
void run_step(std::string name, std::string step) {
    std::string dir = g.runs + "/" + name, work = dir + "/work", tmp = dir + "/work_tmp", log = dir + "/train.log";
    std::ofstream(log, std::ios::app) << "[serve] preprocess step '" << step << "' on '" << name << "'\n";
    set_stage("step:" + step);
    fs::remove_all(tmp); fs::create_directories(tmp);
    std::string flag = step == "separate" ? "--separate" : step == "dereverb" ? "--dereverb"
                     : step == "deharmony" ? "--deharmony" : "--denoise";
    int rc = std::system(("{ '" + g.build + "/vc_preprocess' --input '" + work + "' --output-dir '" + tmp +
        "' --sr 44100 --no-slice " + flag + " --sep-dir '" + g.models + "/separation' 2>&1 | " +
        "grep --line-buffered -E 'processed|error|enabled' ; } >> '" + log + "' 2>&1").c_str());
    if (rc != 0 || fs::is_empty(tmp)) { fs::remove_all(tmp); finish("error", true); return; }
    fs::remove_all(work); fs::rename(tmp, work);
    std::ofstream(dir + "/steps.txt", std::ios::app) << step << "\n";
    finish("step-done", false);
}

// POST /api/material : multipart {name, files[]} -> store raw + init work set.
void handle_material(int fd, const Request& req) {
    size_t b = req.content_type.find("boundary="); if (b == std::string::npos) { send_response(fd, 400, "Bad Request", "text/plain", "no boundary"); return; }
    std::string boundary = req.content_type.substr(b + 9);
    if (!boundary.empty() && boundary.front() == '"') boundary = boundary.substr(1, boundary.size() - 2);
    auto parts = parse_multipart(req.body, boundary);
    std::string name = field(parts, "name", "");
    if (!safe_name(name)) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid name\"}"); return; }
    std::string dir = g.runs + "/" + name, raw = dir + "/raw", work = dir + "/work";
    fs::remove_all(work); fs::remove_all(raw); fs::create_directories(raw); fs::create_directories(work);
    fs::remove(dir + "/steps.txt");
    int nfiles = 0;
    for (auto& p : parts)
        if (!p.filename.empty() && !p.data.empty()) {
            std::string fn = safe_filename(p.filename);
            { std::ofstream o(raw + "/" + fn, std::ios::binary); o.write(p.data.data(), p.data.size()); }
            fs::copy_file(raw + "/" + fn, work + "/" + fn, fs::copy_options::overwrite_existing);
            ++nfiles;
        }
    if (nfiles == 0) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"no audio files\"}"); return; }
    send_json(fd, "{\"name\":\"" + json_escape(name) + "\",\"files\":" + std::to_string(nfiles) + "}");
}

// POST /api/step : multipart {name, step} -> apply one separation step to the work set.
void handle_step(int fd, const Request& req) {
    if (g_job.running) { send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"busy\"}"); return; }
    size_t b = req.content_type.find("boundary="); if (b == std::string::npos) { send_response(fd, 400, "Bad Request", "text/plain", "no boundary"); return; }
    std::string boundary = req.content_type.substr(b + 9);
    if (!boundary.empty() && boundary.front() == '"') boundary = boundary.substr(1, boundary.size() - 2);
    auto parts = parse_multipart(req.body, boundary);
    std::string name = field(parts, "name", ""), step = field(parts, "step", "");
    if (!safe_name(name) || !fs::exists(g.runs + "/" + name + "/work")) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"upload material first\"}"); return; }
    if (step != "separate" && step != "dereverb" && step != "denoise" && step != "deharmony") { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"unknown step\"}"); return; }
    { std::lock_guard<std::mutex> lk(g_job.mu); g_job.name = name; g_job.stage = "step:" + step; g_job.done = false; g_job.error = false; }
    g_job.running = true;
    std::thread(run_step, name, step).detach();
    send_json(fd, "{\"step\":\"" + json_escape(step) + "\"}");
}

// GET /api/preview?name=X : return the first work-set file as WAV for listening.
void handle_preview(int fd, const Request& req) {
    std::string name = query_get(req.query, "name");
    if (!safe_name(name)) { send_response(fd, 400, "Bad Request", "text/plain", "bad name"); return; }
    std::string work = g.runs + "/" + name + "/work";
    if (!fs::exists(work)) { send_response(fd, 404, "Not Found", "text/plain", "no material"); return; }
    std::string first;
    for (auto& e : fs::directory_iterator(work)) { std::string ext = e.path().extension().string();
        if (ext == ".wav" || ext == ".flac") { first = e.path().string(); break; } }
    if (first.empty()) { send_response(fd, 404, "Not Found", "text/plain", "empty"); return; }
    send_response(fd, 200, "OK", "audio/wav", read_file(first));
}

void handle_train(int fd, const Request& req) {
    if (g_job.running) { send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"a training job is already running\"}"); return; }
    size_t b = req.content_type.find("boundary="); if (b == std::string::npos) { send_response(fd, 400, "Bad Request", "text/plain", "no boundary"); return; }
    std::string boundary = req.content_type.substr(b + 9);
    if (!boundary.empty() && boundary.front() == '"') boundary = boundary.substr(1, boundary.size() - 2);
    auto parts = parse_multipart(req.body, boundary);

    std::string name = field(parts, "name", ""), mode = field(parts, "mode", "decoder");
    int steps = std::atoi(field(parts, "steps", "300").c_str());
    int seg = std::atoi(field(parts, "seg", "40").c_str());
    int epochs = std::atoi(field(parts, "epochs", "0").c_str());
    bool separate = field(parts, "separate", "0") == "1";
    bool dereverb = field(parts, "dereverb", "0") == "1";
    bool denoise = field(parts, "denoise", "0") == "1";
    bool deharmony = field(parts, "deharmony", "0") == "1";
    bool vad = field(parts, "vad", "0") == "1";
    if (!safe_name(name)) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid voice name (letters, digits, _ or - only)\"}"); return; }
    if (mode != "decoder" && mode != "gan") mode = "decoder";
    steps = std::max(1, std::min(steps, 100000));
    seg = std::max(8, std::min(seg, 200));
    epochs = std::max(0, std::min(epochs, 2000));

    std::string raw = g.runs + "/" + name + "/raw";
    fs::create_directories(raw);
    int nfiles = 0;
    for (auto& p : parts)
        if (!p.filename.empty() && !p.data.empty()) {
            std::ofstream o(raw + "/" + safe_filename(p.filename), std::ios::binary);
            o.write(p.data.data(), p.data.size()); ++nfiles;
        }
    if (nfiles == 0) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"no audio files uploaded\"}"); return; }

    { std::lock_guard<std::mutex> lk(g_job.mu); g_job.name = name; g_job.stage = "queued"; g_job.done = false; g_job.error = false; }
    g_job.running = true;
    std::thread(run_training, name, mode, steps, seg, epochs, separate, dereverb, denoise, deharmony, vad).detach();
    send_json(fd, "{\"job\":\"" + json_escape(name) + "\",\"files\":" + std::to_string(nfiles) + "}");
}

void handle_status(int fd, const Request& req) {
    std::string qn = query_get(req.query, "name");
    std::string stage, jname; bool running, done, error;
    { std::lock_guard<std::mutex> lk(g_job.mu); stage = g_job.stage; jname = g_job.name; running = g_job.running; done = g_job.done; error = g_job.error; }
    std::string which = qn.empty() ? jname : qn, logtail;
    if (safe_name(which)) {
        std::string all = read_file(g.runs + "/" + which + "/train.log");
        if (all.size() > 4000) all = all.substr(all.size() - 4000);
        logtail = all;
    }
    send_json(fd, std::string("{\"name\":\"") + json_escape(jname) + "\",\"stage\":\"" + json_escape(stage) +
        "\",\"running\":" + (running ? "true" : "false") + ",\"done\":" + (done ? "true" : "false") +
        ",\"error\":" + (error ? "true" : "false") + ",\"log\":\"" + json_escape(logtail) + "\"}");
}

void handle_convert(int fd, const Request& req) {
    size_t b = req.content_type.find("boundary="); if (b == std::string::npos) { send_response(fd, 400, "Bad Request", "text/plain", "no boundary"); return; }
    std::string boundary = req.content_type.substr(b + 9);
    if (!boundary.empty() && boundary.front() == '"') boundary = boundary.substr(1, boundary.size() - 2);
    auto parts = parse_multipart(req.body, boundary);
    const Part* audio = nullptr;
    for (auto& p : parts) if (!p.filename.empty()) { audio = &p; break; }
    if (!audio) { send_response(fd, 400, "Bad Request", "text/plain", "no audio"); return; }

    std::string voice = field(parts, "voice", "base");
    int pitch = std::atoi(field(parts, "f0_up_key", "0").c_str());
    double index_rate = std::atof(field(parts, "index_rate", "0").c_str());
    double rms = std::atof(field(parts, "rms_mix_rate", "0.5").c_str());
    double protect = std::atof(field(parts, "protect", "0.5").c_str());
    int filter_radius = std::atoi(field(parts, "filter_radius", "3").c_str());

    std::string model = g.gmodel, index;
    if (voice != "base" && voice != "base (pretrained)") {
        if (!safe_name(voice)) { send_response(fd, 400, "Bad Request", "text/plain", "bad voice"); return; }
        std::string vm = g.runs + "/" + voice + "/model.safetensors";
        if (fs::exists(vm)) model = vm;
        std::string vi = g.runs + "/" + voice + "/model.index";
        if (fs::exists(vi)) index = vi;
    }

    std::string in = "/tmp/vcserve_in.wav", out = "/tmp/vcserve_out.wav";
    { std::ofstream o(in, std::ios::binary); o.write(audio->data.data(), audio->data.size()); }
    std::string cmd = "'" + g.build + "/vc_convert' --hubert '" + g.hubert + "' --rmvpe '" + g.rmvpe +
        "' --model '" + model + "' --input '" + in + "' --output '" + out +
        "' --version v2 --speakers 109 --sr 40000 --pitch " + std::to_string(pitch) +
        " --rms-mix " + std::to_string(rms) + " --protect " + std::to_string(protect) +
        " --filter-radius " + std::to_string(filter_radius);
    if (!index.empty()) cmd += " --index '" + index + "' --index-rate " + std::to_string(index_rate);
    cmd += " > /tmp/vcserve_convert.log 2>&1";
    int rc = std::system(cmd.c_str());
    std::string wav = read_file(out);
    if (rc != 0 || wav.empty()) { send_response(fd, 500, "Internal Server Error", "text/plain", "conversion failed"); return; }
    printf("[vc_serve] convert voice=%s -> %zu bytes\n", voice.c_str(), wav.size());
    send_response(fd, 200, "OK", "audio/wav", wav);
}

}  // namespace

int main(int argc, char** argv) {
    int port = 8080;
    std::string webroot = "webui", repo;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--port") port = std::atoi(nx());
        else if (a == "--webroot") webroot = nx();
        else if (a == "--repo") repo = nx();
        else { fprintf(stderr, "Unknown argument: %s\n", a.c_str()); return 1; }
    }
    g.webroot = fs::absolute(webroot).string();
    g.repo = repo.empty() ? fs::path(g.webroot).parent_path().string() : fs::absolute(repo).string();
    g.models = g.repo + "/models";
    g.build = g.repo + "/build";
    g.runs = g.repo + "/runs";
    g.hubert = g.models + "/hubert_base/model.safetensors";
    g.rmvpe = g.models + "/rmvpe.safetensors";
    g.gmodel = g.models + "/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    g.dmodel = g.models + "/pretrained_v2/pretrained_v2/f0D40k.safetensors";
    fs::create_directories(g.runs);

    std::signal(SIGPIPE, SIG_IGN);
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port);
    if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) { fprintf(stderr, "bind failed on %d\n", port); return 1; }
    if (listen(srv, 16) < 0) { fprintf(stderr, "listen failed\n"); return 1; }
    printf("Resonantia WebUI on http://localhost:%d\n  repo=%s\n", port, g.repo.c_str());

    for (;;) {
        int fd = accept(srv, nullptr, nullptr);
        if (fd < 0) continue;
        Request req;
        if (read_request(fd, req)) {
            if (req.method == "GET" && req.path == "/api/voices") send_json(fd, voices_json());
            else if (req.method == "GET" && req.path == "/api/train/status") handle_status(fd, req);
            else if (req.method == "POST" && req.path == "/api/train") handle_train(fd, req);
            else if (req.method == "POST" && req.path == "/api/material") handle_material(fd, req);
            else if (req.method == "POST" && req.path == "/api/step") handle_step(fd, req);
            else if (req.method == "GET" && req.path == "/api/preview") handle_preview(fd, req);
            else if (req.method == "POST" && req.path == "/api/convert") handle_convert(fd, req);
            else if (req.method == "GET") serve_static(fd, req.path);
            else send_response(fd, 405, "Method Not Allowed", "text/plain", "no");
        }
        close(fd);
    }
    return 0;
}
