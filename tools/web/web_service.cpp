#include "tools/web/web_service.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <limits>
#include <random>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "voxmutatio/separation/sep_registry.h"
#include "voxmutatio/io/audio_io.h"

namespace resonantia::web {

namespace fs = std::filesystem;

#ifdef RESONANTIA_TEST_JOB_OWNERSHIP
// Defined only by the private host test; the regular service has no callback.
void test_training_response_boundary(std::uint64_t generation, bool accepted);
#endif

namespace {

constexpr std::uintmax_t kMaximumStepHistoryBytes = 64 * 1024;

std::string service_instance_id() {
    std::random_device entropy;
    std::string id;
    for (int i = 0; i < 4; ++i) {
        char hex[9];
        std::snprintf(hex, sizeof hex, "%08x", static_cast<unsigned>(entropy()));
        id += hex;
    }
    return id;
}

bool nonempty_regular_file(const fs::path& path) {
    std::error_code error;
    if (!fs::is_regular_file(path, error) || error) return false;
    const std::uintmax_t size = fs::file_size(path, error);
    return !error && size != 0;
}

bool candidate_name(const std::string& name) {
    return name.size() == 38 && name.starts_with("candidate_epoch_") &&
           name.ends_with(".safetensors") &&
           name.substr(16, 10).find_first_not_of("0123456789") == std::string::npos;
}

bool regular_candidate(const fs::path& path) {
    std::error_code ec;
    return fs::is_regular_file(fs::symlink_status(path, ec)) && !ec && nonempty_regular_file(path);
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

// Constant gain only: preserve the original stereo residual and the converted
// vocal envelope. No compressor, limiter, or automatic loudness normalization.
bool mix_cover(const std::string& original, const std::string& all_vocals,
               const std::string& converted, const std::string& output) {
    namespace io = voxmutatio::io;
    auto source = io::read_stereo_audio(original);
    auto vocals = io::read_stereo_audio(all_vocals);
    auto voice = io::read_audio(converted, 44100);
    if (!source || !vocals || !voice || source->data.empty() || voice->data.empty() ||
        source->frames() != vocals->frames() ||
        std::abs(static_cast<long long>(source->frames()) -
                 static_cast<long long>(voice->data.size())) > 2) return false;
    double peak = 0;
    for (std::size_t i = 0; i < source->data.size(); ++i) {
        source->data[i] = source->data[i] - vocals->data[i] +
            voice->data[std::min(i / 2, voice->data.size() - 1)];
        if (!std::isfinite(source->data[i])) return false;
        peak = std::max(peak, static_cast<double>(std::abs(source->data[i])));
    }
    const double gain = peak > 0.95 ? 0.95 / peak : 1.0;
    for (float& value : source->data) value *= gain;
    std::fprintf(stderr, "[vc_serve] cover fixed gain=%.6f dB frames=%zu\n",
                 20 * std::log10(gain), source->frames());
    return io::write_stereo_audio(output, source->data.data(),
                                  static_cast<int>(source->frames()), 44100);
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

std::optional<fs::path> latest_full_checkpoint(const fs::path& directory) {
    if(!fs::is_directory(directory))return std::nullopt;
    std::optional<fs::path> latest;
    for(const auto& entry:fs::directory_iterator(directory)) {
        const auto name=entry.path().filename().string();
        if(name.size()==33 && name.starts_with("state_step_") && name.ends_with(".safetensors") &&
           name.substr(11,10).find_first_not_of("0123456789")==std::string::npos &&
           fs::is_regular_file(entry.symlink_status()) && entry.file_size()>0 && (!latest || entry.path()>*latest))
            latest=entry.path();
    }
    return latest;
}

void copy_training_clips(const fs::path& source,const fs::path& destination) {
    fs::create_directories(destination);std::size_t count=0;
    for(const auto& entry:fs::directory_iterator(source)) {
        if(lower(entry.path().extension().string())!=".wav")continue;
        if(!fs::is_regular_file(entry.symlink_status()) || !valid_wav_file(entry.path()) || ++count>4096)
            throw std::runtime_error("invalid or excessive prepared training clips");
        fs::copy_file(entry.path(),destination/entry.path().filename());
    }
    if(!count)throw std::runtime_error("no prepared training clips");
}

}  // namespace

WebService::WebService(Paths paths, std::shared_ptr<const ProcessRunner> process_runner)
    : paths_(std::move(paths)), process_runner_(std::move(process_runner)),
      instance_id_(service_instance_id()) {
    if (!process_runner_) throw std::invalid_argument("process runner is required");
}

// ---- workflow ----
bool WebService::reserve_job(const std::string& name, JobKind kind, JobTicket& ticket) {
    std::lock_guard<std::mutex> lock(job_.mu);
    if (job_.running) return false;
    if (job_.generation == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("job generation exhausted");
    ticket.generation = job_.generation + 1;
    ticket.public_id = instance_id_ + "-" + std::to_string(ticket.generation);
    // Prepare all allocating metadata before reserving. Moves after CAS cannot
    // strand a reservation or expose half-initialized identity to cancellation.
    std::string stored_id = ticket.public_id, stored_name = name, stage = "queued";
    bool expected = false;
    if (!job_.running.compare_exchange_strong(expected, true)) return false;
    job_.generation = ticket.generation;
    job_.public_id = std::move(stored_id); job_.name = std::move(stored_name);
    job_.stage = std::move(stage); job_.kind = kind;
    job_.done = false; job_.error = false;
    job_.cancel_requested = false; job_.cancelled = false;
    job_.cancel_error = false; job_.cancel_source.reset();
    return true;
}
void WebService::set_stage(std::uint64_t generation, const std::string& stage) {
    std::lock_guard<std::mutex> lock(job_.mu);
    if (job_.generation == generation) job_.stage = stage;
}
void WebService::finish(std::uint64_t generation, const std::string& stage, bool error) {
    std::lock_guard<std::mutex> lock(job_.mu);
    if (job_.generation != generation) return;
    job_.stage = stage; job_.error = error; job_.done = true; job_.running = false;
}
WebService::TrainingCancelLease::~TrainingCancelLease() {
    // Wait out any already-started request write before staging can disappear.
    // Never nest the channel mutex and job mutex in either lock order.
    const bool clean = source->close();
    std::lock_guard<std::mutex> lock(service.job_.mu);
    if (service.job_.generation == generation) {
        service.job_.cancel_error = service.job_.cancel_error || !clean;
        service.job_.cancel_source.reset();
    }
}
void WebService::finish_training(std::uint64_t generation, TrainingOutcome outcome) {
    std::lock_guard<std::mutex> lock(job_.mu);
    if (job_.generation != generation || job_.kind != JobKind::kTraining) return;
    if (job_.cancel_error) outcome = TrainingOutcome::kError;
    job_.cancelled = outcome == TrainingOutcome::kCancelled;
    job_.error = outcome == TrainingOutcome::kError;
    job_.stage = job_.cancelled ? "cancelled" : job_.error ? "error" : "done";
    job_.done = true;
    // Called only after the request owner's staging scope, or the worker's
    // run_training scope (child reap, staging and GPU lock), has ended.
    job_.running = false;
}
bool WebService::cancellation_requested(std::uint64_t generation) {
    std::lock_guard<std::mutex> lock(job_.mu);
    return job_.generation != generation || job_.cancel_requested;
}
bool WebService::training_checkpoint(std::uint64_t generation, const std::string& stage) {
    std::lock_guard<std::mutex> lock(job_.mu);
    if (job_.generation != generation || job_.cancel_requested || job_.cancel_error) return false;
    job_.stage = stage;
    return true;
}

WebService::TrainingOutcome WebService::run_training(std::uint64_t generation, std::string name, std::string mode, int steps,
                              int segment, int epochs, bool separate,
                              bool dereverb, bool deecho, bool denoise,
                              bool deharmony, bool vad, bool resume, bool prepared) {
    std::unique_lock<std::mutex> gpu_lock(gpu_mutex_, std::defer_lock);
    while (!gpu_lock.try_lock()) {
        std::unique_lock<std::mutex> lock(job_.mu);
        if (job_.generation != generation || job_.cancel_requested)
            return TrainingOutcome::kCancelled;
        job_.changed.wait_for(lock, std::chrono::milliseconds(20), [&] {
            return job_.generation != generation || job_.cancel_requested;
        });
    }
    if (!training_checkpoint(generation, "preparing")) return TrainingOutcome::kCancelled;
    const fs::path directory = fs::path(paths_.runs) / name;
    const bool full=mode=="joint-full";
    const fs::path full_dir=directory/"full_training";
    if(full&&!resume) {
        if(!fs::create_directory(full_dir))throw std::runtime_error("full training directory already exists");
    }
    const fs::path log = directory / "train.log";
    TempDir staged(directory, ".train_stage_");
    const fs::path staged_root = staged.path();
    const fs::path clips = staged_root / "clips";
    const fs::path model = staged_root / "model.safetensors";
    const fs::path index = staged_root / "model.index";
    {
        std::ofstream output(log, resume ? std::ios::app : std::ios::trunc);
        output << "[serve] training '" << name << "' mode=" << mode
               << (epochs > 0 ? " epochs=" + std::to_string(epochs)
                              : " steps=" + std::to_string(steps))
               << " seg=" << segment << (separate ? " +separate" : "") << '\n';
        if (!output) {
            return TrainingOutcome::kError;
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
    if (!training_checkpoint(generation, !preprocessed && has_separation ? "separating" : "preprocessing"))
        return TrainingOutcome::kCancelled;

    if(full && resume) {
        copy_training_clips(full_dir/"clips",clips);
    } else if(full && prepared) {
        copy_training_clips(source,clips);
    } else {
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
        return TrainingOutcome::kError;
    }
    }
    if(full&&!resume) {
        // Commit a complete fixed dataset before the trainer starts. Staging
        // cleanup can never delete this copy or its checkpoints on cancellation.
        const auto fixed_staged=staged_root/"fixed_clips";
        copy_training_clips(clips,fixed_staged);fs::rename(fixed_staged,full_dir/"clips");
    }

    if (!training_checkpoint(generation, "training")) return TrainingOutcome::kCancelled;
    ProcessSpec training;
    training.argv = {paths_.build + "/vc_train"};
    if (mode == "gan" || mode == "joint" || full) {
        training.argv.insert(training.argv.end(), {"--gan", "--dmodel", paths_.dmodel});
    }
    if(mode=="joint")training.argv.insert(training.argv.end(),{"--gan-profile","joint"});
    if(full) {
        training.argv.insert(training.argv.end(),{"--gan-profile","joint-full",
            "--cache-dir",(full_dir/"cache").string(),"--checkpoint-dir",(full_dir/"checkpoints").string()});
        if(resume) {
            if(const auto saved=latest_full_checkpoint(full_dir/"checkpoints"))
                training.argv.insert(training.argv.end(),{"--resume",saved->string()});
            else std::ofstream(log,std::ios::app)<<"[serve] no completed checkpoint; restarting fixed clips from step 0\n";
        }
    }
    training.argv.insert(training.argv.end(),
                         {"--hubert", paths_.hubert, "--rmvpe", paths_.rmvpe,
                          "--pretrained", paths_.gmodel, "--target", (full?full_dir/"clips":clips).string(),
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
                                 "Exported", "clips", "done", "error", "GAN", "cancelled",
                                 "Cache", "Checkpoint", "Candidate", "seconds", "identity", "Resumed"};
    ProcessResult training_result;
    {
        const auto token = staged_root / "cancel";
        TrainingCancelLease channel(*this, generation,
            std::make_shared<training_control::TrainingCancelSource>(token.string()));
        training.argv.insert(training.argv.end(), {"--cancel-file", token.string()});
        {
            std::lock_guard<std::mutex> lock(job_.mu);
            // Covers cancellation between the stage checkpoint and registration.
            if (job_.generation != generation || job_.cancel_requested)
                return TrainingOutcome::kCancelled;
            job_.cancel_source = channel.source;
        }
        // Cancellation only changes the job channel. The synchronous runner
        // drains logs and naturally reaps the child before this lease closes.
        training_result = process_runner_->run(training);
    }
    bool requested;
    {
        std::lock_guard<std::mutex> lock(job_.mu);
        if (job_.generation != generation || job_.cancel_error) return TrainingOutcome::kError;
        requested = job_.cancel_requested;
    }
    if (training_result.termination != ProcessTermination::kExited ||
        training_result.io_error || training_result.system_error || training_result.signal_number)
        return TrainingOutcome::kError;
    if (training_result.exit_code == static_cast<int>(training_control::TrainingResult::kCancelled))
        return requested ? TrainingOutcome::kCancelled : TrainingOutcome::kError;
    if (!training_result.success() || !nonempty_regular_file(model)) return TrainingOutcome::kError;

    if (!training_checkpoint(generation, "indexing")) return TrainingOutcome::kCancelled;
    ProcessSpec indexing;
    indexing.argv = {paths_.build + "/build_index", "--hubert", paths_.hubert,
                     "--input-dir", clips.string(), "--output", index.string()};
    indexing.log_path = log;
    indexing.log_filter_terms = {"Wrote", "index", "error", "frames"};
    if (!process_runner_->run(indexing).success() ||
        !nonempty_regular_file(index)) {
        return TrainingOutcome::kError;
    }

    if (!training_checkpoint(generation, "publishing")) return TrainingOutcome::kCancelled;
    std::string publish_error;
    const auto destination=full?full_dir/"result":directory;
    fs::create_directories(destination);
    if (!replace_training_artifacts(destination, clips, model, index, publish_error)) {
        std::fprintf(stderr, "[vc_serve] training publish failed for '%s': %s\n",
                     name.c_str(), publish_error.c_str());
        return TrainingOutcome::kError;
    }
    std::ofstream(log, std::ios::app) << (full?"[serve] done; review required; candidate retained in full_training/result, existing voice unchanged.\n":"[serve] done.\n");
    return TrainingOutcome::kDone;
}

std::string WebService::voices_json() {
    std::lock_guard<std::mutex> material_lock(material_mutex_);
    std::string j = "{\"base\":\"base (pretrained)\",\"voices\":[";
    bool first = true;
    std::vector<std::string> reviewed;
    if (fs::is_directory(paths_.runs)) {
        std::vector<std::string> names;
        for (auto& e : fs::directory_iterator(paths_.runs))
            if (e.is_directory() && fs::exists(e.path() / "model.safetensors"))
                names.push_back(e.path().filename().string());
        std::sort(names.begin(), names.end());
        for (auto& n : names) {
            if (!first) j += ","; j += "\"" + json_escape(n) + "\""; first = false;
            if (regular_candidate(fs::path(paths_.runs)/n/"clips/release.json")) reviewed.push_back(n);
        }
    }
    j += "],\"reviewed_voices\":["; first = true;
    for (const auto& name : reviewed) {
        if (!first) j += ','; first = false; j += "\"" + json_escape(name) + "\"";
    }
    { std::lock_guard<std::mutex> lk(job_.mu);
      j += "],\"training_profiles\":[\"decoder\",\"gan\",\"joint\",\"joint-full\"],\"training\":{\"running\":" + std::string(job_.running ? "true" : "false") +
           ",\"name\":\"" + json_escape(job_.name) + "\",\"stage\":\"" + json_escape(job_.stage) + "\"}}"; }
    return j;
}

void WebService::handle_training_runs(int fd) {
    std::string json = "{\"runs\":[";
    bool first = true;
    std::error_code ec;
    std::vector<fs::path> runs;
    for (fs::directory_iterator it(paths_.runs, ec), end; !ec && it != end; it.increment(ec))
        if (safe_name(it->path().filename().string()) && fs::is_directory(it->path()/"full_training"))
            runs.push_back(it->path());
    std::sort(runs.begin(), runs.end());
    for (const auto& run : runs) {
        const auto full = run/"full_training";
        std::vector<std::string> candidates;
        std::error_code scan_error;
        for (fs::directory_iterator it(full/"checkpoints", scan_error), end;
             !scan_error && it != end; it.increment(scan_error)) {
            const auto name = it->path().filename().string();
            if (candidate_name(name) && regular_candidate(it->path())) candidates.push_back(name);
        }
        std::sort(candidates.begin(), candidates.end());
        if (!first) json += ','; first = false;
        json += "{\"name\":\"" + json_escape(run.filename().string()) + "\",\"resumable\":" +
            (latest_full_checkpoint(full/"checkpoints") && first_valid_wav(full/"clips") ? "true" : "false") +
            ",\"result_ready\":" + (regular_candidate(full/"result/model.safetensors") &&
               regular_candidate(full/"result/model.index") ? "true" : "false") + ",\"candidates\":[";
        bool first_candidate = true;
        for (const auto& name : candidates) {
            if (!first_candidate) json += ','; first_candidate = false;
            json += "\"" + json_escape(name) + "\"";
        }
        json += "]}";
    }
    send_json(fd, json + "]}");
}

void WebService::handle_publish(int fd, const Request& request) {
    std::vector<Part> parts;
    if (!request_parts(fd, request, parts)) return;
    const auto run = field(parts, "name", ""), candidate = field(parts, "candidate", "");
    const auto name = field(parts, "publish_name", "");
    if (!safe_name(run) || !safe_name(name) || !candidate_name(candidate) ||
        field(parts, "reviewed", "0") != "1") {
        send_response(fd, 400, "Bad Request", "text/plain", "请选择候选、填写新声线名称并确认听审通过"); return;
    }
    std::unique_lock gpu_lock(gpu_mutex_, std::try_to_lock);
    if (!gpu_lock.owns_lock() || job_.running) {
        send_response(fd, 409, "Conflict", "text/plain", "GPU 工作中，请结束后发布"); return;
    }
    const auto source = fs::path(paths_.runs)/run/"full_training";
    const auto model = source/"checkpoints"/candidate, index = source/"result/model.index";
    if (!regular_candidate(model) || !regular_candidate(index)) {
        send_response(fd, 404, "Not Found", "text/plain", "候选或训练索引尚未就绪"); return;
    }
    std::lock_guard material_lock(material_mutex_);
    const auto destination = fs::path(paths_.runs)/name;
    for (const auto* artifact : {"model.safetensors", "model.index", "clips"}) {
        if (fs::exists(fs::symlink_status(destination/artifact))) {
            send_response(fd, 409, "Conflict", "text/plain", "声线已存在，请使用新名称；已有文件不会被覆盖"); return;
        }
    }
    fs::create_directories(destination);
    TempDir staging(destination, ".release_");
    const fs::path staged(staging.path());
    fs::create_directory(staged/"clips");
    fs::copy_file(model, staged/"model.safetensors");
    fs::copy_file(index, staged/"model.index");
    std::ofstream record(staged/"clips/release.json");
    record << "{\"reviewed\":true,\"run\":\"" << json_escape(run)
           << "\",\"candidate\":\"" << json_escape(candidate) << "\"}\n";
    record.close();
    std::string error;
    if (!record || !replace_training_artifacts(destination, staged/"clips", staged/"model.safetensors",
                                               staged/"model.index", error)) {
        send_response(fd, 500, "Internal Server Error", "text/plain", "发布失败，旧声线保留"); return;
    }
    send_json(fd, "{\"published\":\"" + json_escape(name) + "\"}");
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

void WebService::run_step(std::uint64_t generation, std::string name, std::string step, std::string model,
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
    set_stage(generation, "step:" + step);
    const std::string flag =
        step == "separate" ? "--separate"
        : step == "dereverb" ? "--dereverb"
        : step == "deecho" ? "--deecho"
        : step == "deharmony" ? "--deharmony"
                               : "--denoise";
    ProcessSpec process;
    process.argv = {paths_.build + "/vc_preprocess", "--input", work.string(),
                    "--output-dir", staged_work.string(), "--sr", "44100",
                    "--no-slice", "--preserve-stereo", flag};
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
        finish(generation, "error", true);
        return;
    }
    const fs::path steps = dir / "steps.txt";
    std::error_code marker_error;
    const bool has_steps = fs::exists(steps, marker_error);
    if (marker_error) {
        finish(generation, "error", true);
        return;
    }
    if (has_steps) {
        const bool regular = fs::is_regular_file(steps, marker_error);
        const std::uintmax_t size =
            regular && !marker_error ? fs::file_size(steps, marker_error) : 0;
        if (!regular || marker_error || step.size() + 1 > kMaximumStepHistoryBytes ||
            size > kMaximumStepHistoryBytes - step.size() - 1) {
            finish(generation, "error", true);
            return;
        }
        fs::copy_file(steps, staged_steps, fs::copy_options::none, marker_error);
        if (marker_error) {
            finish(generation, "error", true);
            return;
        }
    }
    {
        std::ofstream marker(staged_steps, std::ios::app);
        marker << step << '\n';
        marker.close();
        if (!marker) {
            finish(generation, "error", true);
            return;
        }
    }
    std::string error;
    bool committed = false;
    {
        std::lock_guard<std::mutex> material_lock(material_mutex_);
        committed = replace_step_artifacts(dir, staged_work, staged_steps, error);
    }
    if (!committed) {
        std::fprintf(stderr, "[vc_serve] step publish failed for '%s': %s\n",
                     name.c_str(), error.c_str());
        finish(generation, "error", true);
        return;
    }
    finish(generation, "step-done", false);
}

// POST /api/material : multipart {name, files[]} -> store raw + init work set.
bool WebService::stage_uploads(int fd, const std::vector<Upload>& uploads,
                              const fs::path& raw, const std::optional<fs::path>& work, bool stereo) {
    const auto result = normalize_uploads(uploads, raw, work, paths_, *process_runner_, decode_mutex_, stereo);
    if (result.status == 200) return true;
    const char* reason = result.status == 400 ? "Bad Request" : result.status == 409 ? "Conflict" :
                         result.status == 413 ? "Payload Too Large" :
                         result.status == 503 ? "Service Unavailable" : "Internal Server Error";
    send_response(fd, result.status, reason, "application/json", "{\"error\":\"" + json_escape(result.error) + "\"}");
    return false;
}

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
    bool stereo = false;
    if (!parse_bool_field(field(parts, "preserve_stereo", "0"), stereo)) {
        send_response(fd, 400, "Bad Request", "text/plain", "invalid preserve_stereo"); return;
    }
    if (!stage_uploads(fd, uploads, staged_raw, staged_work, stereo)) return;

    std::unique_lock<std::mutex> gpu_lk(gpu_mutex_, std::try_to_lock);
    if (!gpu_lk.owns_lock() || job_.running) {
        send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"workflow busy; upload not committed\"}");
        return;
    }
    bool committed = false;
    {
        std::lock_guard<std::mutex> material_lock(material_mutex_);
        committed = replace_material_dirs(dir, staged_raw, staged_work, error);
    }
    if (!committed) {
        send_response(fd, 500, "Internal Server Error", "application/json",
                      "{\"error\":\"upload commit failed\"}");
        return;
    }
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
    JobTicket ticket;
    try {
        if (!reserve_job(name, JobKind::kStep, ticket)) {
            send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"busy\"}"); return;
        }
    } catch (const std::exception&) {
        send_response(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"cannot prepare job\"}"); return;
    }
    std::future<bool> gpu_ready_result;
    try {
        // Every fallible preparation after reservation must release the slot on
        // failure, including promise allocation before a worker exists.
        std::promise<bool> gpu_ready;
        gpu_ready_result = gpu_ready.get_future();
        const std::shared_ptr<WebService> self = shared_from_this();
        std::thread worker([self, generation = ticket.generation, name, step, model, ready = std::move(gpu_ready)]() mutable {
            bool readiness_reported = false;
            try {
                std::unique_lock<std::mutex> gpu_lk(self->gpu_mutex_, std::try_to_lock);
                if (!gpu_lk.owns_lock()) {
                    self->finish(generation, "error", true);
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
                self->run_step(generation, name, step, model, std::move(gpu_lk));
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[vc_serve] preprocessing exception: %s\n", e.what());
                if (!readiness_reported) {
                    self->finish(generation, "error", true);
                    try { ready.set_exception(std::current_exception()); } catch (...) {}
                } else {
                    self->finish(generation, "error", true);
                }
            } catch (...) {
                std::fprintf(stderr, "[vc_serve] preprocessing exception\n");
                if (!readiness_reported) {
                    self->finish(generation, "error", true);
                    try { ready.set_exception(std::current_exception()); } catch (...) {}
                } else {
                    self->finish(generation, "error", true);
                }
            }
        });
        worker.detach();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[vc_serve] cannot start preprocessing worker: %s\n", e.what());
        finish(ticket.generation, "error", true);
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
    send_json(fd, "{\"step\":\"" + json_escape(step) + "\",\"job_id\":\"" + ticket.public_id + "\"}");
}

// GET /api/preview?name=X : return the first work-set file as WAV for listening.
void WebService::handle_preview(int fd, const Request& req) {
    std::string name = query_get(req.query, "name");
    if (!safe_name(name)) { send_response(fd, 400, "Bad Request", "text/plain", "bad name"); return; }
    std::unique_lock<std::mutex> material_lock(material_mutex_);
    std::string work = paths_.runs + "/" + name + "/work";
    if (!fs::exists(work)) { material_lock.unlock(); send_response(fd, 404, "Not Found", "text/plain", "no material"); return; }
    const std::optional<fs::path> first = first_valid_wav(work);
    if (!first) { material_lock.unlock(); send_response(fd, 404, "Not Found", "text/plain", "empty"); return; }
    TempDir preview_temp("/tmp", "resonantia-preview-");
    std::string preview_path = first->string();
    if (exceeds_buffered_response_limit(*first)) {
        auto stereo = voxmutatio::io::read_stereo_audio(preview_path);
        if (stereo && 44 + stereo->frames() * 4 <= kMaxFileBytes) {
            const auto compact = preview_temp.path() + "/preview.wav";
            if (voxmutatio::io::write_stereo_audio(compact, stereo->data.data(),
                    static_cast<int>(stereo->frames()), 44100)) preview_path = compact;
        }
    }
    if (exceeds_buffered_response_limit(preview_path)) {
        material_lock.unlock();
        send_response(fd, 413, "Payload Too Large", "application/json",
                      "{\"error\":\"audio response exceeds 64 MiB\"}");
        return;
    }
    const std::string wav = read_file(preview_path);
    material_lock.unlock();
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
    bool deharmony = false, vad = false, resume = false, prepared = false;
    if (!safe_name(name)) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid voice name (letters, digits, _ or - only)\"}"); return; }
    const bool full=mode=="joint-full";
    if (mode != "decoder" && mode != "gan" && mode != "joint" && !full) {
        send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid training mode\"}"); return;
    }
    if (!parse_bool_field(field(parts, "separate", "0"), separate) ||
        !parse_bool_field(field(parts, "dereverb", "0"), dereverb) ||
        !parse_bool_field(field(parts, "deecho", "0"), deecho) ||
        !parse_bool_field(field(parts, "denoise", "0"), denoise) ||
        !parse_bool_field(field(parts, "deharmony", "0"), deharmony) ||
        !parse_bool_field(field(parts, "vad", "0"), vad) ||
        !parse_bool_field(field(parts, "resume", "0"), resume) ||
        !parse_bool_field(field(parts, "prepared", "0"), prepared)) {
        send_response(fd, 400, "Bad Request", "application/json",
                      "{\"error\":\"boolean parameters must be 0 or 1\"}");
        return;
    }
    if (!parse_int_field(field(parts, "steps", mode=="joint"?"50":"300"), 1, 100000, steps) ||
        !parse_int_field(field(parts, "seg", mode=="joint"||full?"32":"40"), 8, 200, seg) ||
        !parse_int_field(field(parts, "epochs", full?"200":"0"), 0, 2000, epochs)) {
        send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"training parameter out of range\"}"); return;
    }
    if(mode=="joint" && (steps>50 || seg!=32 || epochs!=0)) {
        send_response(fd,400,"Bad Request","application/json",
                      "{\"error\":\"joint training requires 1..50 steps, seg 32 and epochs 0\"}");return;
    }
    if((!full&&(resume||prepared)) || (full&&(epochs<1||seg!=32||!field(parts,"steps","").empty())) ||
       ((resume||prepared)&&(separate||dereverb||deecho||denoise||deharmony||vad)) || (resume&&prepared)) {
        send_response(fd,400,"Bad Request","application/json",
            "{\"error\":\"joint-full requires epochs >0, seg 32, no steps; resume/prepared are full-only and cannot combine with cleanup or each other\"}");return;
    }

    std::vector<Upload> uploads;
    if (!collect_uploads(fd, parts, uploads)) return;
    int nfiles = static_cast<int>(uploads.size());
    fs::path voice_dir = fs::path(paths_.runs) / name;
    const auto full_dir=voice_dir/"full_training";
    if(full && !resume && fs::exists(full_dir)) {
        send_response(fd,409,"Conflict","application/json","{\"error\":\"full training state already exists; resume it or use a new name\"}");return;
    }
    if(resume && (nfiles || fs::is_symlink(full_dir) || fs::is_symlink(full_dir/"clips") || !first_valid_wav(full_dir/"clips"))) {
        send_response(fd,400,"Bad Request","application/json","{\"error\":\"resume requires retained full training clips and no uploads\"}");return;
    }
    fs::create_directories(voice_dir);
    bool has_work = fs::exists(paths_.runs + "/" + name + "/work") && !fs::is_empty(paths_.runs + "/" + name + "/work");
    if (nfiles == 0 && !has_work && !resume) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"no audio files uploaded\"}"); return; }

    TrainingOwner owner(*this); // Declared first: staging retires before the reservation.
    std::optional<TempDir> staged;
    fs::path staged_raw, staged_work;
    if (!uploads.empty()) {
        staged.emplace(voice_dir, ".train_upload_");
        staged_raw = fs::path(staged->path()) / "raw";
        staged_work = fs::path(staged->path()) / "work";
        if (!stage_uploads(fd, uploads, staged_raw, staged_work)) return;
    }

    JobTicket ticket;
    try {
        if (!reserve_job(name, JobKind::kTraining, ticket)) {
            send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"a training job is already running\"}"); return;
        }
        owner.generation = ticket.generation;
    } catch (const std::exception&) {
        send_response(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"cannot prepare job\"}"); return;
    }
    const auto accepted = [&] {
#ifdef RESONANTIA_TEST_JOB_OWNERSHIP
        test_training_response_boundary(ticket.generation, true);
#endif
        send_json(fd, "{\"job\":\"" + json_escape(name) + "\",\"job_id\":\"" + ticket.public_id +
                      "\",\"files\":" + std::to_string(nfiles) + "}");
    };
    if (staged) {
        std::unique_lock<std::mutex> gpu_lk(gpu_mutex_, std::try_to_lock);
        if (!gpu_lk.owns_lock()) {
            send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"GPU busy; upload not committed\"}");
            return;
        }
        std::string error;
        bool committed = false, cancelled_before_commit = false;
        try {
            std::lock_guard<std::mutex> material_lock(material_mutex_);
            cancelled_before_commit = !training_checkpoint(ticket.generation, "material-commit");
            if (!cancelled_before_commit)
                committed = replace_material_dirs(voice_dir, staged_raw, staged_work, error);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[vc_serve] upload commit exception: %s\n", e.what());
        }
        if (cancelled_before_commit) {
            gpu_lk.unlock();
            owner.outcome = TrainingOutcome::kCancelled; accepted(); return;
        }
        if (!committed) {
            // The request owner retires the reservation after upload staging.
#ifdef RESONANTIA_TEST_JOB_OWNERSHIP
            test_training_response_boundary(ticket.generation, false);
#endif
            send_response(fd, 500, "Internal Server Error", "application/json", "{\"error\":\"upload commit failed\"}");
            return;
        }
    }
    // Retire request-owned upload staging before handing the reservation to a
    // worker that can complete before the acceptance response is written.
    staged.reset();
    if (cancellation_requested(ticket.generation)) {
        owner.outcome = TrainingOutcome::kCancelled; accepted(); return;
    }
    set_stage(ticket.generation, "queued");
    try {
        const std::shared_ptr<WebService> self = shared_from_this();
        std::thread worker([self, generation = ticket.generation, name, mode, steps, seg, epochs, separate, dereverb,
                            deecho, denoise, deharmony, vad, resume, prepared] {
            TrainingOwner completion(*self); completion.generation = generation;
            try {
                completion.outcome = self->run_training(generation, name, mode, steps, seg, epochs, separate, dereverb,
                                   deecho, denoise, deharmony, vad, resume, prepared);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[vc_serve] training exception: %s\n", e.what());
            } catch (...) {
                std::fprintf(stderr, "[vc_serve] training exception\n");
            }
        });
        owner.release(); // The successfully constructed worker owns finalization now.
        worker.detach();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[vc_serve] cannot start training worker: %s\n", e.what());
        send_response(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"cannot start worker\"}");
        return;
    }
    accepted();
}

void WebService::handle_status(int fd, const Request& req) {
    std::string qn = query_get(req.query, "name");
    std::string stage, jname, id, kind, state; bool running, done, error, requested, cancelled, cancellable;
    {
        std::lock_guard<std::mutex> lock(job_.mu);
        stage = job_.stage; jname = job_.name; id = job_.public_id;
        running = job_.running; done = job_.done; error = job_.error;
        requested = job_.cancel_requested; cancelled = job_.cancelled;
        kind = job_.kind == JobKind::kTraining ? "train" : job_.kind == JobKind::kStep ? "step" : "none";
        cancellable = running && job_.kind == JobKind::kTraining && stage != "publishing";
        state = !running ? (cancelled ? "cancelled" : error ? "error" : done ? "done" : "idle") :
                requested ? "cancel_requested" : stage == "queued" ? "queued" :
                stage == "publishing" ? "publishing" : "running";
    }
    std::string which = qn.empty() ? jname : qn, logtail;
    if (safe_name(which)) logtail = log_tail(paths_.runs + "/" + which + "/train.log", 4000);
    const auto full_dir=fs::path(paths_.runs)/which/"full_training";
    const bool resumable=safe_name(which)&&first_valid_wav(full_dir/"clips").has_value()&&
        latest_full_checkpoint(full_dir/"checkpoints").has_value();
    const bool review_required=safe_name(which)&&nonempty_regular_file(full_dir/"result/model.safetensors");
    send_json(fd, std::string("{\"name\":\"") + json_escape(jname) + "\",\"stage\":\"" + json_escape(stage) +
        "\",\"job_id\":\"" + id + "\",\"kind\":\"" + kind + "\",\"state\":\"" + state +
        "\",\"running\":" + (running ? "true" : "false") + ",\"done\":" + (done ? "true" : "false") +
        ",\"error\":" + (error ? "true" : "false") + ",\"cancel_requested\":" + (requested ? "true" : "false") +
        ",\"cancelled\":" + (cancelled ? "true" : "false") + ",\"cancellable\":" + (cancellable ? "true" : "false") +
        ",\"resumable\":"+(resumable?"true":"false")+",\"review_required\":"+(review_required?"true":"false")+
        ",\"cancel_boundary\":\"stage_or_iteration\",\"log\":\"" + json_escape(logtail) + "\"}");
}

void WebService::handle_cancel_train(int fd, const Request& req) {
    std::vector<Part> parts;
    if (!request_parts(fd, req, parts)) return;
    if (parts.size() != 1 || parts[0].name != "job_id" || !parts[0].filename.empty() ||
        !safe_name(parts[0].data)) {
        send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"exactly one valid job_id is required\"}"); return;
    }
    const std::string id = parts[0].data;
    int code = 202; const char* failure = nullptr; bool stopped = false;
    std::shared_ptr<training_control::TrainingCancelSource> channel;
    {
        std::lock_guard<std::mutex> lock(job_.mu);
        if (job_.public_id.empty()) { code = 404; failure = "no job"; }
        else if (id != job_.public_id) { code = 409; failure = "job id does not match current job"; }
        else if (job_.kind != JobKind::kTraining) { code = 409; failure = "only training jobs can be cancelled"; }
        else if (job_.cancelled && !job_.running) { code = 200; stopped = true; }
        else if (!job_.running || job_.stage == "publishing") { code = 409; failure = "too late: publication started or job already ended"; }
        else {
            code = job_.cancel_requested ? 200 : 202; job_.cancel_requested = true;
            channel = job_.cancel_source;
        }
    }
    if (failure) {
        send_response(fd, code, code == 404 ? "Not Found" : "Conflict", "application/json",
                      std::string("{\"error\":\"") + failure + "\"}"); return;
    }
    job_.changed.notify_all();
    if (channel && channel->request() == training_control::CancelDelivery::kError) {
        send_response(fd, 500, "Internal Server Error", "application/json",
                      "{\"error\":\"cancellation channel failed; waiting for current stage to finish\"}");
        return;
    }
    send_response(fd, code, code == 202 ? "Accepted" : "OK", "application/json",
                  "{\"job_id\":\"" + id + "\",\"cancel_requested\":true,\"cancelled\":" +
                  (stopped ? "true" : "false") + ",\"state\":\"" +
                  (stopped ? "cancelled" : "cancel_requested") + "\"}");
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
                    "--output-dir", outdir, "--sr", "44100", "--no-slice", "--preserve-stereo"};
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
    std::unique_lock<std::mutex> gpu_lk(gpu_mutex_, std::try_to_lock);
    if (!gpu_lk.owns_lock()) { send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"GPU 忙（正在训练或转换），请稍候重试\"}"); return; }
    TempDir temp("/tmp", "resonantia-preprocess-");
    const fs::path upload_dir = fs::path(temp.path()) / "upload";
    std::string in = (upload_dir / uploads.front().filename).string(), cleaned, log;
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
    if (!stage_uploads(fd, uploads, upload_dir)) return;
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
    const std::string workflow = field(parts, "workflow", "manual");
    const bool song = workflow == "song";
    if (workflow != "manual" && workflow != "dry" && workflow != "harmonies" && !song) {
        send_response(fd, 400, "Bad Request", "text/plain", "invalid workflow"); return;
    }
    const std::string output_kind = field(parts, "output_kind", "vocal");
    if ((output_kind != "vocal" && output_kind != "mix") || (output_kind == "mix" && !song)) {
        send_response(fd, 400, "Bad Request", "text/plain", "mix output requires song workflow"); return;
    }
    std::unique_lock<std::mutex> gpu_lk(gpu_mutex_, std::try_to_lock);
    if (!gpu_lk.owns_lock()) { send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"GPU 忙（正在训练或转换），请稍候重试\"}"); return; }
    TempDir temp("/tmp", "resonantia-convert-");
    if (!session.empty()) {
        std::lock_guard<std::mutex> material_lock(material_mutex_);
        const auto input = first_valid_wav(fs::path(paths_.runs) / session /
                                          (workflow == "manual" ? "work" : "raw"));
        if (input) {
            conv_in = temp.path() + "/source.wav";
            fs::copy_file(*input, conv_in);
        }
    }
    if (conv_in.empty() && !audio) { send_response(fd, 400, "Bad Request", "text/plain", "no audio"); return; }

    std::string voice = field(parts, "voice", "base");
    const auto pitch_edits = field(parts, "pitch_edits", "");
    std::vector<voxmutatio::PitchEdit> validated_edits;
    if (!voxmutatio::parse_pitch_edits(pitch_edits, validated_edits)) {
        send_response(fd, 400, "Bad Request", "text/plain", "局部修音需按时间排序且不重叠；每段 0.02–2 秒，最多 16 段"); return;
    }
    int pitch = 0, filter_radius = 0;
    double index_rate = 0.0, rms = 0.0, protect = 0.0;
    if (!parse_int_field(field(parts, "f0_up_key", "0"), -48, 48, pitch) ||
        !parse_double_field(field(parts, "index_rate", "0"), 0.0, 1.0, index_rate) ||
        !parse_double_field(field(parts, "rms_mix_rate", "0.5"), 0.0, 1.0, rms) ||
        !parse_double_field(field(parts, "protect", "0.5"), 0.0, 1.0, protect) ||
        !parse_int_field(field(parts, "filter_radius", "3"), 0, 31, filter_radius) ||
        (filter_radius != 0 && (filter_radius < 3 || filter_radius % 2 == 0))) {
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
    const auto candidate_run = field(parts, "training_run", "");
    const auto candidate = field(parts, "candidate", "");
    if (!candidate_run.empty() || !candidate.empty()) {
        if (!safe_name(candidate_run) || !candidate_name(candidate)) {
            send_response(fd, 400, "Bad Request", "text/plain", "invalid training candidate"); return;
        }
        const auto full = fs::path(paths_.runs)/candidate_run/"full_training";
        model = (full/"checkpoints"/candidate).string();
        if (!regular_candidate(model)) {
            send_response(fd, 404, "Not Found", "text/plain", "candidate not found"); return;
        }
        if (regular_candidate(full/"result/model.index")) index = (full/"result/model.index").string();
    } else if (voice != "base" && voice != "base (pretrained)") {
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

    std::string out = temp.path() + "/output.wav";
    std::string preprocess_log;
    if (conv_in.empty()) {
        const fs::path upload_dir = fs::path(temp.path()) / "upload";
        if (!stage_uploads(fd, uploads, upload_dir, std::nullopt, workflow != "manual")) return;
        conv_in = (upload_dir / uploads.front().filename).string();
    }
    const std::string original = conv_in;
    std::string all_vocals;
    if (song) {
        if (!preprocess_audio_file(original, temp.path() + "/vocals", true, false, false, false, false,
                                   all_vocals, preprocess_log)) {
            send_response(fd, 500, "Internal Server Error", "text/plain", "分离人声失败：" + log_tail(preprocess_log)); return;
        }
        conv_in = all_vocals;
    }
    const std::string before_cleanup = conv_in;
    if (!preprocess_audio_file(before_cleanup, temp.path() + "/cleanup",
            workflow == "manual" && pp_sep, song || workflow == "harmonies" || pp_deharm,
            pp_derev, pp_deecho, pp_denoise, conv_in, preprocess_log)) {
        send_response(fd, 500, "Internal Server Error", "text/plain", "主唱清理失败：" + log_tail(preprocess_log)); return;
    }
    if (conv_in != original) {
        // Match the reviewed delivery path: retain stereo through cleanup,
        // then mono 40k PCM16 before the model's 16k analysis frontend.
        auto lead = voxmutatio::io::read_audio(conv_in, 40000);
        if (!lead || lead->data.empty()) {
            send_response(fd, 500, "Internal Server Error", "text/plain", "清理后的人声文件无效"); return;
        }
        if (lead->data.size() > 40000u * 600) {
            send_response(fd, 413, "Payload Too Large", "text/plain", "清理后的人声超过 10 分钟"); return;
        }
        float peak = 0;
        for (float value : lead->data) peak = std::max(peak, std::abs(value));
        if (peak > 0.99f) for (float& value : lead->data) value *= 0.99f / peak;
        conv_in = temp.path() + "/lead40.wav";
        if (!voxmutatio::io::write_audio(conv_in, lead->data.data(), static_cast<int>(lead->data.size()), 40000)) {
            send_response(fd, 500, "Internal Server Error", "text/plain", "主唱保存失败"); return;
        }
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
    if (!pitch_edits.empty()) process.argv.insert(process.argv.end(), {"--pitch-edits", pitch_edits});
    const auto conversion_start = std::chrono::steady_clock::now();
    const ProcessResult process_result = process_runner_->run(process);
    if (!process_result.success() || !valid_wav_file(out)) {
        send_response(fd, 500, "Internal Server Error", "text/plain",
                      "转换失败：" + log_tail(convert_log));
        return;
    }
    if (output_kind == "mix") {
        const std::string mixed = temp.path() + "/cover.wav";
        if (!mix_cover(original, all_vocals, out, mixed)) {
            send_response(fd, 500, "Internal Server Error", "text/plain", "伴奏合成失败：音频长度或格式不匹配"); return;
        }
        out = mixed;
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
    const double conversion_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - conversion_start).count();
    const std::string timings = log_tail(convert_log, 2048);
    // Preserve CLI stage timings in the service journal before RAII removes the
    // request directory. Child wall time also includes initialization and I/O.
    std::fprintf(stderr, "[vc_serve] convert voice=%s -> %zu bytes, child/readback %.3f ms\n%s\n",
                 voice.c_str(), wav.size(), conversion_ms, timings.c_str());
    send_response(fd, 200, "OK", "audio/wav", wav);
}

const char* WebService::allowed_api_method(const std::string& path) {
    if (path == "/api/train/runs") return "GET";
    if (path == "/api/train/publish") return "POST";
    if (path == "/api/voices" || path == "/api/sep-models" ||
        path == "/api/train/status" || path == "/api/preview") {
        return "GET";
    }
    if (path == "/api/train" || path == "/api/train/cancel" || path == "/api/material" || path == "/api/step" ||
        path == "/api/preprocess" || path == "/api/convert") {
        return "POST";
    }
    return nullptr;
}

void WebService::handle_request(int fd, const Request& request) {
    if (request.method == "GET" && request.path == "/api/train/runs") {
        handle_training_runs(fd);
    } else if (request.method == "POST" && request.path == "/api/train/publish") {
        handle_publish(fd, request);
    } else if (request.method == "GET" && request.path == "/api/voices") {
        send_json(fd, voices_json());
    } else if (request.method == "GET" && request.path == "/api/sep-models") {
        send_json(fd, separation_models_json());
    } else if (request.method == "GET" && request.path == "/api/train/status") {
        handle_status(fd, request);
    } else if (request.method == "POST" && request.path == "/api/train/cancel") {
        handle_cancel_train(fd, request);
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
