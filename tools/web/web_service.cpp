#include "tools/web/web_service.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "voxmutatio/separation/sep_registry.h"

namespace resonantia::web {

namespace fs = std::filesystem;

namespace {

constexpr std::uintmax_t kMaximumStepHistoryBytes = 64 * 1024;

bool nonempty_regular_file(const fs::path& path) {
    std::error_code error;
    if (!fs::is_regular_file(path, error) || error) return false;
    const std::uintmax_t size = fs::file_size(path, error);
    return !error && size != 0;
}

bool valid_wav_file(const fs::path& path) {
    if (!nonempty_regular_file(path)) return false;
    std::error_code error;
    if (fs::file_size(path, error) < 12 || error) return false;
    std::ifstream input(path, std::ios::binary);
    std::array<char, 12> header{};
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    return input.gcount() == static_cast<std::streamsize>(header.size()) &&
           std::string_view(header.data(), 4) == "RIFF" &&
           std::string_view(header.data() + 8, 4) == "WAVE";
}

bool exceeds_buffered_response_limit(const fs::path& path) {
    std::error_code error;
    const std::uintmax_t size = fs::file_size(path, error);
    return !error && size > kMaxFileBytes;
}

std::optional<fs::path> first_valid_wav(const fs::path& directory) {
    std::error_code error;
    for (fs::directory_iterator entry(directory, error), end; !error && entry != end;
         entry.increment(error)) {
        if (!entry->is_regular_file(error) || error ||
            lower(entry->path().extension().string()) != ".wav") {
            if (error) return std::nullopt;
            continue;
        }
        if (valid_wav_file(entry->path())) return entry->path();
    }
    return std::nullopt;
}

}  // namespace

WebService::WebService(Paths paths, std::shared_ptr<const ProcessRunner> process_runner)
    : paths_(std::move(paths)), process_runner_(std::move(process_runner)) {
    if (!process_runner_) throw std::invalid_argument("process runner is required");
}

// ---- workflow ----
void WebService::set_stage(const std::string& s) { std::lock_guard<std::mutex> lk(job_.mu); job_.stage = s; }
void WebService::finish(const std::string& stage, bool error) {
    std::lock_guard<std::mutex> lk(job_.mu);
    job_.stage = stage; job_.error = error; job_.done = true; job_.running = false;
}

void WebService::run_training(std::string name, std::string mode, int steps,
                              int segment, int epochs, bool separate,
                              bool dereverb, bool deecho, bool denoise,
                              bool deharmony, bool vad) {
    std::lock_guard<std::mutex> gpu_lock(gpu_mutex_);
    const fs::path directory = fs::path(paths_.runs) / name;
    const fs::path log = directory / "train.log";
    TempDir staged(directory, ".train_stage_");
    const fs::path staged_root = staged.path();
    const fs::path clips = staged_root / "clips";
    const fs::path model = staged_root / "model.safetensors";
    const fs::path index = staged_root / "model.index";
    {
        std::ofstream output(log, std::ios::trunc);
        output << "[serve] training '" << name << "' mode=" << mode
               << (epochs > 0 ? " epochs=" + std::to_string(epochs)
                              : " steps=" + std::to_string(steps))
               << " seg=" << segment << (separate ? " +separate" : "") << '\n';
        if (!output) {
            finish("error", true);
            return;
        }
    }

    // If the material was preprocessed step-by-step, slice that work set without
    // repeating separation. Otherwise slice raw material with the selected stages.
    const fs::path work = directory / "work";
    const bool preprocessed =
        nonempty_regular_file(directory / "steps.txt") && first_valid_wav(work);
    const fs::path source = preprocessed ? work : directory / "raw";
    const bool has_separation =
        separate || deharmony || dereverb || deecho || denoise;
    set_stage(!preprocessed && has_separation ? "separating" : "preprocessing");

    ProcessSpec preprocess;
    preprocess.argv = {paths_.build + "/vc_preprocess", "--input", source.string(),
                       "--output-dir", clips.string(), "--sr", "40000",
                       "--seg-sec", "3.0"};
    if (vad) {
        preprocess.argv.insert(preprocess.argv.end(),
                               {"--vad", "--vad-dir", paths_.models + "/vad"});
    } else {
        preprocess.argv.push_back("--trim");
    }
    if (!preprocessed) {
        if (separate) preprocess.argv.push_back("--separate");
        if (deharmony) preprocess.argv.push_back("--deharmony");
        if (dereverb) preprocess.argv.push_back("--dereverb");
        if (deecho) preprocess.argv.push_back("--deecho");
        if (denoise) preprocess.argv.push_back("--denoise");
        if (has_separation) {
            preprocess.argv.insert(preprocess.argv.end(),
                                   {"--sep-dir", paths_.models + "/separation"});
        }
    }
    preprocess.log_path = log;
    preprocess.log_filter_terms = {"clip", "Wrote", "error", "skip",
                                   "separation", "reverb", "noise",
                                   "speech", "enabled"};
    if (!process_runner_->run(preprocess).success() || !first_valid_wav(clips)) {
        finish("error", true);
        return;
    }

    set_stage("training");
    ProcessSpec training;
    training.argv = {paths_.build + "/vc_train"};
    if (mode == "gan") {
        training.argv.insert(training.argv.end(), {"--gan", "--dmodel", paths_.dmodel});
    }
    training.argv.insert(training.argv.end(),
                         {"--hubert", paths_.hubert, "--rmvpe", paths_.rmvpe,
                          "--pretrained", paths_.gmodel, "--target", clips.string(),
                          "--out", model.string()});
    if (epochs > 0) {
        training.argv.insert(training.argv.end(),
                             {"--epochs", std::to_string(epochs)});
    } else {
        training.argv.insert(training.argv.end(),
                             {"--steps", std::to_string(steps)});
    }
    training.argv.insert(training.argv.end(), {"--seg", std::to_string(segment)});
    training.log_path = log;
    training.log_filter_terms = {"step", "mel", "Device", "frames", "Epochs",
                                 "Exported", "clips", "done", "error", "GAN"};
    if (!process_runner_->run(training).success() ||
        !nonempty_regular_file(model)) {
        finish("error", true);
        return;
    }

    set_stage("indexing");
    ProcessSpec indexing;
    indexing.argv = {paths_.build + "/build_index", "--hubert", paths_.hubert,
                     "--input-dir", clips.string(), "--output", index.string()};
    indexing.log_path = log;
    indexing.log_filter_terms = {"Wrote", "index", "error", "frames"};
    if (!process_runner_->run(indexing).success() ||
        !nonempty_regular_file(index)) {
        finish("error", true);
        return;
    }

    std::string publish_error;
    if (!replace_training_artifacts(directory, clips, model, index, publish_error)) {
        std::fprintf(stderr, "[vc_serve] training publish failed for '%s': %s\n",
                     name.c_str(), publish_error.c_str());
        finish("error", true);
        return;
    }
    std::ofstream(log, std::ios::app) << "[serve] done.\n";
    finish("done", false);
}

std::string WebService::voices_json() {
    std::string j = "{\"base\":\"base (pretrained)\",\"voices\":[";
    bool first = true;
    if (fs::is_directory(paths_.runs)) {
        std::vector<std::string> names;
        for (auto& e : fs::directory_iterator(paths_.runs))
            if (e.is_directory() && fs::exists(e.path() / "model.safetensors"))
                names.push_back(e.path().filename().string());
        std::sort(names.begin(), names.end());
        for (auto& n : names) { if (!first) j += ","; j += "\"" + json_escape(n) + "\""; first = false; }
    }
    { std::lock_guard<std::mutex> lk(job_.mu);
      j += "],\"training\":{\"running\":" + std::string(job_.running ? "true" : "false") +
           ",\"name\":\"" + json_escape(job_.name) + "\",\"stage\":\"" + json_escape(job_.stage) + "\"}}"; }
    return j;
}

// Installed separation models grouped by stage, for the WebUI per-stage picker.
std::string WebService::separation_models_json() const {
    auto models = voxmutatio::separation::list_sep_models(paths_.models + "/separation");
    std::string j = "{";
    bool first_task = true;
    for (const auto& task : voxmutatio::separation::sep_tasks()) {
        if (!first_task) j += ",";
        first_task = false;
        j += "\"" + json_escape(task) + "\":[";
        bool first = true;
        for (const auto& m : models) {
            if (m.task != task) continue;
            if (!first) j += ",";
            first = false;
            j += "{\"name\":\"" + json_escape(m.name) + "\",\"label\":\"" + json_escape(m.label) + "\"}";
        }
        j += "]";
    }
    j += "}";
    return j;
}

void WebService::run_step(std::string name, std::string step, std::string model,
              std::unique_lock<std::mutex> gpu_lk) {
    if (!gpu_lk.owns_lock()) throw std::logic_error("step worker does not own GPU reservation");
    const fs::path dir = fs::path(paths_.runs) / name;
    const fs::path work = dir / "work";
    TempDir staged(dir, ".step_stage_");
    const fs::path staged_root = staged.path();
    const fs::path staged_work = staged_root / "work";
    const fs::path staged_steps = staged_root / "steps.txt";
    const fs::path log = dir / "train.log";
    std::ofstream(log, std::ios::app) << "[serve] preprocess step '" << step << "'"
        << (model.empty() ? "" : " model '" + model + "'") << " on '" << name << "'\n";
    set_stage("step:" + step);
    const std::string flag =
        step == "separate" ? "--separate"
        : step == "dereverb" ? "--dereverb"
        : step == "deecho" ? "--deecho"
        : step == "deharmony" ? "--deharmony"
                               : "--denoise";
    ProcessSpec process;
    process.argv = {paths_.build + "/vc_preprocess", "--input", work.string(),
                    "--output-dir", staged_work.string(), "--sr", "44100",
                    "--no-slice", flag};
    if (!model.empty()) {
        process.argv.insert(process.argv.end(),
                            {"--" + step + "-model", model});
    }
    process.argv.insert(process.argv.end(),
                        {"--sep-dir", paths_.models + "/separation"});
    process.log_path = log;
    process.log_filter_terms = {"processed", "error", "enabled"};
    const ProcessResult process_result = process_runner_->run(process);
    if (!process_result.success() || !first_valid_wav(staged_work)) {
        finish("error", true);
        return;
    }
    const fs::path steps = dir / "steps.txt";
    std::error_code marker_error;
    const bool has_steps = fs::exists(steps, marker_error);
    if (marker_error) {
        finish("error", true);
        return;
    }
    if (has_steps) {
        const bool regular = fs::is_regular_file(steps, marker_error);
        const std::uintmax_t size =
            regular && !marker_error ? fs::file_size(steps, marker_error) : 0;
        if (!regular || marker_error || step.size() + 1 > kMaximumStepHistoryBytes ||
            size > kMaximumStepHistoryBytes - step.size() - 1) {
            finish("error", true);
            return;
        }
        fs::copy_file(steps, staged_steps, fs::copy_options::none, marker_error);
        if (marker_error) {
            finish("error", true);
            return;
        }
    }
    {
        std::ofstream marker(staged_steps, std::ios::app);
        marker << step << '\n';
        marker.close();
        if (!marker) {
            finish("error", true);
            return;
        }
    }
    std::string error;
    if (!replace_step_artifacts(dir, staged_work, staged_steps, error)) {
        std::fprintf(stderr, "[vc_serve] step publish failed for '%s': %s\n",
                     name.c_str(), error.c_str());
        finish("error", true);
        return;
    }
    finish("step-done", false);
}

// POST /api/material : multipart {name, files[]} -> store raw + init work set.
void WebService::handle_material(int fd, const Request& req) {
    std::vector<Part> parts;
    if (!request_parts(fd, req, parts)) return;
    std::string name = field(parts, "name", "");
    if (!safe_name(name)) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid name\"}"); return; }
    std::vector<Upload> uploads;
    if (!collect_uploads(fd, parts, uploads)) return;
    if (uploads.empty()) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"no audio files\"}"); return; }

    fs::path dir = fs::path(paths_.runs) / name;
    fs::create_directories(dir);
    TempDir staged(dir, ".material_upload_");
    fs::path staged_raw = fs::path(staged.path()) / "raw";
    fs::path staged_work = fs::path(staged.path()) / "work";
    std::string error;
    if (!write_uploads(uploads, staged_raw, staged_work, error)) {
        send_response(fd, 500, "Internal Server Error", "application/json",
                      "{\"error\":\"upload staging failed\"}");
        return;
    }

    std::unique_lock<std::mutex> gpu_lk(gpu_mutex_, std::try_to_lock);
    if (!gpu_lk.owns_lock() || job_.running) {
        send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"workflow busy; upload not committed\"}");
        return;
    }
    if (!replace_material_dirs(dir, staged_raw, staged_work, error)) {
        send_response(fd, 500, "Internal Server Error", "application/json",
                      "{\"error\":\"upload commit failed\"}");
        return;
    }
    std::error_code ignored;
    fs::remove(dir / "steps.txt", ignored);
    send_json(fd, "{\"name\":\"" + json_escape(name) + "\",\"files\":" + std::to_string(uploads.size()) + "}");
}

// POST /api/step : multipart {name, step} -> apply one separation step to the work set.
void WebService::handle_step(int fd, const Request& req) {
    if (job_.running) { send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"busy\"}"); return; }
    std::vector<Part> parts;
    if (!request_parts(fd, req, parts)) return;
    std::string name = field(parts, "name", ""), step = field(parts, "step", "");
    if (!safe_name(name) || !fs::exists(paths_.runs + "/" + name + "/work")) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"upload material first\"}"); return; }
    if (step != "separate" && step != "dereverb" && step != "deecho" && step != "denoise" && step != "deharmony") { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"unknown step\"}"); return; }
    std::string model = field(parts, "model", "");
    if (!model.empty() &&
        (!safe_name(model) ||
         !voxmutatio::separation::sep_model_available(paths_.models + "/separation", step, model))) {
        send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"unknown model\"}"); return;
    }
    bool expected = false;
    if (!job_.running.compare_exchange_strong(expected, true)) { send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"busy\"}"); return; }
    std::promise<bool> gpu_ready;
    std::future<bool> gpu_ready_result = gpu_ready.get_future();
    const std::shared_ptr<WebService> self = shared_from_this();
    try {
        std::thread worker([self, name, step, model, ready = std::move(gpu_ready)]() mutable {
            bool readiness_reported = false;
            try {
                std::unique_lock<std::mutex> gpu_lk(self->gpu_mutex_, std::try_to_lock);
                if (!gpu_lk.owns_lock()) {
                    self->job_.running.store(false);
                    ready.set_value(false);
                    return;
                }
                {
                    std::lock_guard<std::mutex> lk(self->job_.mu);
                    self->job_.name = name;
                    self->job_.stage = "step:" + step;
                    self->job_.done = false;
                    self->job_.error = false;
                }
                ready.set_value(true);
                readiness_reported = true;
                self->run_step(name, step, model, std::move(gpu_lk));
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[vc_serve] preprocessing exception: %s\n", e.what());
                if (!readiness_reported) {
                    self->job_.running.store(false);
                    try { ready.set_exception(std::current_exception()); } catch (...) {}
                } else {
                    self->finish("error", true);
                }
            } catch (...) {
                std::fprintf(stderr, "[vc_serve] preprocessing exception\n");
                if (!readiness_reported) {
                    self->job_.running.store(false);
                    try { ready.set_exception(std::current_exception()); } catch (...) {}
                } else {
                    self->finish("error", true);
                }
            }
        });
        worker.detach();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[vc_serve] cannot start preprocessing worker: %s\n", e.what());
        job_.running.store(false);
        send_response(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"cannot start worker\"}");
        return;
    }
    try {
        if (!gpu_ready_result.get()) {
            send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"GPU busy\"}");
            return;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[vc_serve] cannot reserve GPU for preprocessing: %s\n", e.what());
        send_response(fd, 503, "Service Unavailable", "application/json",
                      "{\"error\":\"cannot start worker\"}");
        return;
    }
    send_json(fd, "{\"step\":\"" + json_escape(step) + "\"}");
}

// GET /api/preview?name=X : return the first work-set file as WAV for listening.
void WebService::handle_preview(int fd, const Request& req) {
    std::string name = query_get(req.query, "name");
    if (!safe_name(name)) { send_response(fd, 400, "Bad Request", "text/plain", "bad name"); return; }
    std::string work = paths_.runs + "/" + name + "/work";
    if (!fs::exists(work)) { send_response(fd, 404, "Not Found", "text/plain", "no material"); return; }
    const std::optional<fs::path> first = first_valid_wav(work);
    if (!first) { send_response(fd, 404, "Not Found", "text/plain", "empty"); return; }
    if (exceeds_buffered_response_limit(*first)) {
        send_response(fd, 413, "Payload Too Large", "application/json",
                      "{\"error\":\"audio response exceeds 64 MiB\"}");
        return;
    }
    const std::string wav = read_file(first->string());
    if (wav.empty()) {
        send_response(fd, 500, "Internal Server Error", "text/plain",
                      "preview read failed");
        return;
    }
    send_response(fd, 200, "OK", "audio/wav", wav);
}

void WebService::handle_train(int fd, const Request& req) {
    if (job_.running) { send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"a training job is already running\"}"); return; }
    std::vector<Part> parts;
    if (!request_parts(fd, req, parts)) return;

    std::string name = field(parts, "name", ""), mode = field(parts, "mode", "decoder");
    int steps = 0, seg = 0, epochs = 0;
    bool separate = false, dereverb = false, deecho = false, denoise = false;
    bool deharmony = false, vad = false;
    if (!safe_name(name)) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid voice name (letters, digits, _ or - only)\"}"); return; }
    if (mode != "decoder" && mode != "gan") {
        send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid training mode\"}"); return;
    }
    if (!parse_bool_field(field(parts, "separate", "0"), separate) ||
        !parse_bool_field(field(parts, "dereverb", "0"), dereverb) ||
        !parse_bool_field(field(parts, "deecho", "0"), deecho) ||
        !parse_bool_field(field(parts, "denoise", "0"), denoise) ||
        !parse_bool_field(field(parts, "deharmony", "0"), deharmony) ||
        !parse_bool_field(field(parts, "vad", "0"), vad)) {
        send_response(fd, 400, "Bad Request", "application/json",
                      "{\"error\":\"boolean parameters must be 0 or 1\"}");
        return;
    }
    if (!parse_int_field(field(parts, "steps", "300"), 1, 100000, steps) ||
        !parse_int_field(field(parts, "seg", "40"), 8, 200, seg) ||
        !parse_int_field(field(parts, "epochs", "0"), 0, 2000, epochs)) {
        send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"training parameter out of range\"}"); return;
    }

    std::vector<Upload> uploads;
    if (!collect_uploads(fd, parts, uploads)) return;
    int nfiles = static_cast<int>(uploads.size());
    fs::path voice_dir = fs::path(paths_.runs) / name;
    fs::create_directories(voice_dir);
    bool has_work = fs::exists(paths_.runs + "/" + name + "/work") && !fs::is_empty(paths_.runs + "/" + name + "/work");
    if (nfiles == 0 && !has_work) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"no audio files uploaded\"}"); return; }

    std::optional<TempDir> staged;
    fs::path staged_raw, staged_work;
    if (!uploads.empty()) {
        staged.emplace(voice_dir, ".train_upload_");
        staged_raw = fs::path(staged->path()) / "raw";
        staged_work = fs::path(staged->path()) / "work";
        std::string error;
        if (!write_uploads(uploads, staged_raw, staged_work, error)) {
            send_response(fd, 500, "Internal Server Error", "application/json", "{\"error\":\"upload staging failed\"}");
            return;
        }
    }

    bool expected = false;
    if (!job_.running.compare_exchange_strong(expected, true)) {
        send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"a training job is already running\"}"); return; }
    { std::lock_guard<std::mutex> lk(job_.mu); job_.name = name; job_.stage = "queued"; job_.done = false; job_.error = false; }
    if (staged) {
        std::unique_lock<std::mutex> gpu_lk(gpu_mutex_, std::try_to_lock);
        if (!gpu_lk.owns_lock()) {
            finish("error", true);
            send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"GPU busy; upload not committed\"}");
            return;
        }
        std::string error;
        try {
            if (!replace_material_dirs(voice_dir, staged_raw, staged_work, error)) {
                finish("error", true);
                send_response(fd, 500, "Internal Server Error", "application/json", "{\"error\":\"upload commit failed\"}");
                return;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[vc_serve] upload commit exception: %s\n", e.what());
            finish("error", true);
            send_response(fd, 500, "Internal Server Error", "application/json", "{\"error\":\"upload commit failed\"}");
            return;
        }
        std::error_code ignored;
        fs::remove(voice_dir / "steps.txt", ignored);
    }
    const std::shared_ptr<WebService> self = shared_from_this();
    try {
        std::thread worker([self, name, mode, steps, seg, epochs, separate, dereverb,
                            deecho, denoise, deharmony, vad] {
            try {
                self->run_training(name, mode, steps, seg, epochs, separate, dereverb,
                                   deecho, denoise, deharmony, vad);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[vc_serve] training exception: %s\n", e.what());
                self->finish("error", true);
            } catch (...) {
                std::fprintf(stderr, "[vc_serve] training exception\n");
                self->finish("error", true);
            }
        });
        worker.detach();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[vc_serve] cannot start training worker: %s\n", e.what());
        finish("error", true);
        send_response(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"cannot start worker\"}");
        return;
    }
    send_json(fd, "{\"job\":\"" + json_escape(name) + "\",\"files\":" + std::to_string(nfiles) + "}");
}

void WebService::handle_status(int fd, const Request& req) {
    std::string qn = query_get(req.query, "name");
    std::string stage, jname; bool running, done, error;
    { std::lock_guard<std::mutex> lk(job_.mu); stage = job_.stage; jname = job_.name; running = job_.running; done = job_.done; error = job_.error; }
    std::string which = qn.empty() ? jname : qn, logtail;
    if (safe_name(which)) logtail = log_tail(paths_.runs + "/" + which + "/train.log", 4000);
    send_json(fd, std::string("{\"name\":\"") + json_escape(jname) + "\",\"stage\":\"" + json_escape(stage) +
        "\",\"running\":" + (running ? "true" : "false") + ",\"done\":" + (done ? "true" : "false") +
        ",\"error\":" + (error ? "true" : "false") + ",\"log\":\"" + json_escape(logtail) + "\"}");
}

// Apply selected SOTA input preprocessing (vocal/deharmony/dereverb/denoise) to
// `in`, writing a cleaned full-length WAV path to `cleaned`. cleaned=in if none.
bool WebService::preprocess_audio_file(const std::string& in, const std::string& temp_dir, bool sep,
                           bool deharm, bool derev, bool deecho, bool denoise,
                           std::string& cleaned, std::string& log) const {
    if (!(sep || deharm || derev || deecho || denoise)) { cleaned = in; return true; }
    std::string outdir = temp_dir + "/preprocessed";
    log = temp_dir + "/preprocess.log";
    fs::create_directories(outdir);
    ProcessSpec process;
    process.argv = {paths_.build + "/vc_preprocess", "--input", in,
                    "--output-dir", outdir, "--sr", "40000", "--no-slice"};
    if (sep) process.argv.push_back("--separate");
    if (deharm) process.argv.push_back("--deharmony");
    if (derev) process.argv.push_back("--dereverb");
    if (deecho) process.argv.push_back("--deecho");
    if (denoise) process.argv.push_back("--denoise");
    process.argv.insert(process.argv.end(),
                        {"--sep-dir", paths_.models + "/separation"});
    process.log_path = log;
    if (!process_runner_->run(process).success()) return false;
    const std::optional<fs::path> output = first_valid_wav(outdir);
    if (!output) return false;
    cleaned = output->string();
    return true;
}

// POST /api/preprocess : multipart {audio, pp_*} -> cleaned WAV (preview, no conversion).
void WebService::handle_preprocess(int fd, const Request& req) {
    std::vector<Part> parts;
    if (!request_parts(fd, req, parts)) return;
    std::vector<Upload> uploads;
    if (!collect_uploads(fd, parts, uploads)) return;
    if (uploads.size() != 1) { send_response(fd, 400, "Bad Request", "text/plain", "exactly one audio file required"); return; }
    const Part* audio = uploads.front().part;
    std::unique_lock<std::mutex> gpu_lk(gpu_mutex_, std::try_to_lock);
    if (!gpu_lk.owns_lock()) { send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"GPU 忙（正在训练或转换），请稍候重试\"}"); return; }
    TempDir temp("/tmp", "resonantia-preprocess-");
    std::string in = temp.path() + "/input.wav", cleaned, log;
    { std::ofstream o(in, std::ios::binary | std::ios::trunc); o.write(audio->data.data(), audio->data.size());
      if (!o) { send_response(fd, 500, "Internal Server Error", "text/plain", "input write failed"); return; } }
    bool sep = false, deharm = false, derev = false, denoise = false, deecho = false;
    if (!parse_bool_field(field(parts, "pp_separate", "0"), sep) ||
        !parse_bool_field(field(parts, "pp_deharmony", "0"), deharm) ||
        !parse_bool_field(field(parts, "pp_dereverb", "0"), derev) ||
        !parse_bool_field(field(parts, "pp_denoise", "0"), denoise) ||
        !parse_bool_field(field(parts, "pp_deecho", "0"), deecho)) {
        send_response(fd, 400, "Bad Request", "application/json",
                      "{\"error\":\"boolean parameters must be 0 or 1\"}");
        return;
    }
    if (!preprocess_audio_file(in, temp.path(), sep, deharm, derev, deecho, denoise, cleaned, log)) {
        send_response(fd, 500, "Internal Server Error", "text/plain", "预处理失败：" + log_tail(log)); return; }
    if (exceeds_buffered_response_limit(cleaned)) {
        send_response(fd, 413, "Payload Too Large", "application/json",
                      "{\"error\":\"audio response exceeds 64 MiB\"}");
        return;
    }
    const std::string wav = read_file(cleaned);
    if (wav.empty()) {
        send_response(fd, 500, "Internal Server Error", "text/plain",
                      "preprocess output read failed");
        return;
    }
    send_response(fd, 200, "OK", "audio/wav", wav);
}

void WebService::handle_convert(int fd, const Request& req) {
    std::vector<Part> parts;
    if (!request_parts(fd, req, parts)) return;
    std::vector<Upload> uploads;
    if (!collect_uploads(fd, parts, uploads)) return;
    if (uploads.size() > 1) { send_response(fd, 400, "Bad Request", "text/plain", "at most one audio file allowed"); return; }
    const Part* audio = uploads.empty() ? nullptr : uploads.front().part;

    // Preprocessed input may come from a cascade work-set (session) instead of a fresh upload.
    std::string session = field(parts, "session", "");
    std::string conv_in;
    if (!session.empty() && !safe_name(session)) {
        send_response(fd, 400, "Bad Request", "text/plain", "bad session"); return;
    }
    if (!session.empty()) {
        std::string work = paths_.runs + "/" + session + "/work";
        if (fs::exists(work))
            for (auto& e : fs::directory_iterator(work)) {
                if (lower(e.path().extension().string()) == ".wav") { conv_in = e.path().string(); break; }
            }
    }
    if (conv_in.empty() && !audio) { send_response(fd, 400, "Bad Request", "text/plain", "no audio"); return; }
    std::unique_lock<std::mutex> gpu_lk(gpu_mutex_, std::try_to_lock);
    if (!gpu_lk.owns_lock()) { send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"GPU 忙（正在训练或转换），请稍候重试\"}"); return; }
    TempDir temp("/tmp", "resonantia-convert-");

    std::string voice = field(parts, "voice", "base");
    int pitch = 0, filter_radius = 0;
    double index_rate = 0.0, rms = 0.0, protect = 0.0;
    if (!parse_int_field(field(parts, "f0_up_key", "0"), -48, 48, pitch) ||
        !parse_double_field(field(parts, "index_rate", "0"), 0.0, 1.0, index_rate) ||
        !parse_double_field(field(parts, "rms_mix_rate", "0.5"), 0.0, 1.0, rms) ||
        !parse_double_field(field(parts, "protect", "0.5"), 0.0, 1.0, protect) ||
        !parse_int_field(field(parts, "filter_radius", "3"), 0, 31, filter_radius)) {
        send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"conversion parameter out of range\"}");
        return;
    }
    bool pp_sep = false, pp_deharm = false, pp_derev = false;
    bool pp_denoise = false, pp_deecho = false;
    if (!parse_bool_field(field(parts, "pp_separate", "0"), pp_sep) ||
        !parse_bool_field(field(parts, "pp_deharmony", "0"), pp_deharm) ||
        !parse_bool_field(field(parts, "pp_dereverb", "0"), pp_derev) ||
        !parse_bool_field(field(parts, "pp_denoise", "0"), pp_denoise) ||
        !parse_bool_field(field(parts, "pp_deecho", "0"), pp_deecho)) {
        send_response(fd, 400, "Bad Request", "application/json",
                      "{\"error\":\"boolean parameters must be 0 or 1\"}");
        return;
    }

    std::string model = paths_.gmodel, index;
    if (voice != "base" && voice != "base (pretrained)") {
        if (!safe_name(voice)) { send_response(fd, 400, "Bad Request", "text/plain", "bad voice"); return; }
        std::string vm = paths_.runs + "/" + voice + "/model.safetensors";
        if (!nonempty_regular_file(vm)) {
            send_response(fd, 404, "Not Found", "application/json",
                          "{\"error\":\"voice model not found\"}");
            return;
        }
        model = vm;
        std::string vi = paths_.runs + "/" + voice + "/model.index";
        if (nonempty_regular_file(vi)) index = vi;
    }
    if (index_rate > 0.0 && index.empty()) {
        send_response(fd, 400, "Bad Request", "application/json",
                      "{\"error\":\"index_rate > 0 requires a compatible voice index\"}");
        return;
    }

    std::string in = temp.path() + "/input.wav", out = temp.path() + "/output.wav";
    std::string preprocess_log;
    if (conv_in.empty()) {
        { std::ofstream o(in, std::ios::binary | std::ios::trunc); o.write(audio->data.data(), audio->data.size());
          if (!o) { send_response(fd, 500, "Internal Server Error", "text/plain", "input write failed"); return; } }
        if (!preprocess_audio_file(in, temp.path(), pp_sep, pp_deharm, pp_derev, pp_deecho, pp_denoise,
                                   conv_in, preprocess_log)) {
            send_response(fd, 500, "Internal Server Error", "text/plain", "输入预处理失败：" + log_tail(preprocess_log)); return; }
    }
    std::string convert_log = temp.path() + "/convert.log";
    ProcessSpec process;
    process.argv = {paths_.build + "/vc_convert", "--hubert", paths_.hubert,
                    "--rmvpe", paths_.rmvpe, "--model", model, "--input", conv_in,
                    "--output", out, "--version", "v2", "--speakers", "109",
                    "--sr", "40000", "--pitch", std::to_string(pitch),
                    "--rms-mix", std::to_string(rms), "--protect",
                    std::to_string(protect), "--filter-radius",
                    std::to_string(filter_radius)};
    if (index_rate > 0.0) {
        process.argv.insert(process.argv.end(),
                            {"--index", index, "--index-rate",
                             std::to_string(index_rate)});
    }
    process.log_path = convert_log;
    const ProcessResult process_result = process_runner_->run(process);
    if (!process_result.success() || !valid_wav_file(out)) {
        send_response(fd, 500, "Internal Server Error", "text/plain",
                      "转换失败：" + log_tail(convert_log));
        return;
    }
    if (exceeds_buffered_response_limit(out)) {
        send_response(fd, 413, "Payload Too Large", "application/json",
                      "{\"error\":\"audio response exceeds 64 MiB\"}");
        return;
    }
    std::string wav = read_file(out);
    if (wav.empty()) {
        send_response(fd, 500, "Internal Server Error", "text/plain",
                      "转换失败：output read failed");
        return;
    }
    printf("[vc_serve] convert voice=%s -> %zu bytes\n", voice.c_str(), wav.size());
    send_response(fd, 200, "OK", "audio/wav", wav);
}

const char* WebService::allowed_api_method(const std::string& path) {
    if (path == "/api/voices" || path == "/api/sep-models" ||
        path == "/api/train/status" || path == "/api/preview") {
        return "GET";
    }
    if (path == "/api/train" || path == "/api/material" || path == "/api/step" ||
        path == "/api/preprocess" || path == "/api/convert") {
        return "POST";
    }
    return nullptr;
}

void WebService::handle_request(int fd, const Request& request) {
    if (request.method == "GET" && request.path == "/api/voices") {
        send_json(fd, voices_json());
    } else if (request.method == "GET" && request.path == "/api/sep-models") {
        send_json(fd, separation_models_json());
    } else if (request.method == "GET" && request.path == "/api/train/status") {
        handle_status(fd, request);
    } else if (request.method == "POST" && request.path == "/api/train") {
        handle_train(fd, request);
    } else if (request.method == "POST" && request.path == "/api/material") {
        handle_material(fd, request);
    } else if (request.method == "POST" && request.path == "/api/step") {
        handle_step(fd, request);
    } else if (request.method == "GET" && request.path == "/api/preview") {
        handle_preview(fd, request);
    } else if (request.method == "POST" && request.path == "/api/preprocess") {
        handle_preprocess(fd, request);
    } else if (request.method == "POST" && request.path == "/api/convert") {
        handle_convert(fd, request);
    } else if (request.path.rfind("/api/", 0) == 0) {
        if (const char* allowed = allowed_api_method(request.path)) {
            send_response(fd, 405, "Method Not Allowed", "application/json",
                          "{\"error\":\"method not allowed\"}",
                          "Allow: " + std::string(allowed) + "\r\n");
        } else {
            send_response(fd, 404, "Not Found", "application/json",
                          "{\"error\":\"unknown API endpoint\"}");
        }
    } else if (request.method == "GET") {
        serve_static(fd, paths_.webroot, request.path);
    } else {
        send_response(fd, 405, "Method Not Allowed", "text/plain", "no", "Allow: GET\r\n");
    }
}

}  // namespace resonantia::web
