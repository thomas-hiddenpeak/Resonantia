#pragma once

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#include "tools/web/http_transport.h"
#include "tools/web/task_store.h"
#include "tools/web/task_artifacts.h"
#include "tools/app_support/training_cancel.h"
#include "tools/web/workflow_support.h"

namespace voxmutatio { struct VCConfig; }
namespace resonantia::web {

class WebWorkflowIdentity;

class WebService final : public std::enable_shared_from_this<WebService> {
public:
    WebService(Paths paths, std::shared_ptr<const ProcessRunner> process_runner);
    WebService(const WebService&) = delete;
    WebService& operator=(const WebService&) = delete;

    void handle_request(int fd, const Request& request);

private:
    // Read-only access in the isolated lifecycle test targets post-reservation
    // allocation faults without adding a production fault callback.
    friend struct WebServiceLifecycleTestAccess;
    // The isolated ownership test replays late finalization with an old ticket.
    friend struct WebServiceJobOwnershipTestAccess;

    enum class JobKind { kNone, kStep, kTraining, kConversion, kPreprocess };
    enum class TrainingOutcome { kDone, kError, kCancelled, kPaused };
    struct JobTicket { std::uint64_t generation = 0; std::string public_id; std::uint64_t attempt=1; };
    struct TrainingOwner {
        WebService& service;
        std::uint64_t generation = 0;
        TrainingOutcome outcome = TrainingOutcome::kError;
        explicit TrainingOwner(WebService& owner) : service(owner) {}
        TrainingOwner(const TrainingOwner&) = delete;
        ~TrainingOwner() { if (generation) service.finish_training(generation, outcome); }
        void release() noexcept { generation = 0; }
    };
    struct TrainingCancelLease {
        WebService& service;
        std::uint64_t generation;
        std::shared_ptr<training_control::TrainingCancelSource> source;
        TrainingCancelLease(WebService& owner, std::uint64_t id,
                            std::shared_ptr<training_control::TrainingCancelSource> channel)
            : service(owner), generation(id), source(std::move(channel)) {}
        TrainingCancelLease(const TrainingCancelLease&) = delete;
        ~TrainingCancelLease();
    };
    struct JobState {
        std::mutex mu;
        std::condition_variable changed;
        std::uint64_t generation = 0;
        std::uint64_t attempt = 1;
        std::string public_id;
        JobKind kind = JobKind::kNone;
        bool cancel_requested = false;
        bool pause_requested = false;
        bool recoverable_training = false;
        std::string storage_notice;
        std::string control_path;
        bool cancel_error = false;
        std::shared_ptr<training_control::TrainingCancelSource> cancel_source;
        bool cancelled = false;
        std::string name;
        std::string stage = "idle";
        std::chrono::steady_clock::time_point progress_saved{};
        std::atomic<bool> running{false};
        bool done = false;
        bool error = false;
    };

    // These helpers require job_.mu; terminal persistence is contained so a
    // cleanup owner never throws while releasing its generation.
    void persist_job_locked(const std::string& state, bool terminal = false);
    void persist_terminal_locked(const std::string& state) noexcept;
    void handle_tasks(int fd, const Request& request);
    void handle_pitch_report(int fd, const Request& request);
    void handle_pitch_preview(int fd, const Request& request);
    void handle_task_rename(int fd, const Request& request);
    void handle_storage(int fd);
    TaskStorageUsage storage_usage(bool force = false);
    void require_storage_room(const std::string& id = {});
    void guard_task_storage(std::uint64_t generation, bool force);
    void handle_task_control(int fd, const Request& request);
    void enable_progress(ProcessSpec& process, std::uint64_t generation);
    void observe_progress(std::uint64_t generation, std::string_view line);
    void audio_boundary(std::uint64_t generation, std::string_view stage = {});
    void enable_audio_control(ProcessSpec& process, std::uint64_t generation,
                              const std::filesystem::path& checkpoint);
    bool run_audio_process(ProcessSpec& process, std::uint64_t generation);
    ProcessResult run_process(ProcessSpec& process);
    std::filesystem::path job_log_path(std::uint64_t generation);

    bool reserve_job(const std::string& name, JobKind kind, JobTicket& ticket,
                     const std::string& requested_id = {});
    void set_stage(std::uint64_t generation, const std::string& stage);
    void finish(std::uint64_t generation, const std::string& stage, bool error);
    void finish_training(std::uint64_t generation, TrainingOutcome outcome);
    bool training_checkpoint(std::uint64_t generation, const std::string& stage);
    bool cancellation_requested(std::uint64_t generation);
    TrainingOutcome run_training(std::uint64_t generation, std::string name, std::string mode, int steps, int segment,
                      int epochs, bool separate, bool dereverb, bool deecho,
                      bool denoise, bool deharmony, bool vad, bool resume = false, bool prepared = false);
    void run_step(std::uint64_t generation, std::string name, std::string step, std::string model,
                  std::unique_lock<std::mutex> gpu_lock);

    [[nodiscard]] std::string voices_json();
    [[nodiscard]] std::string separation_models_json() const;
    [[nodiscard]] bool preprocess_audio_file(const std::string& input,
                                             const std::string& temporary_directory,
                                             bool separate, bool deharmony,
                                             bool dereverb, bool deecho, bool denoise,
                                             std::string& cleaned, std::string& log,
                                             std::uint64_t generation = 0);

    void handle_material(int fd, const Request& request);
    bool stage_uploads(int fd, const std::vector<Upload>& uploads,
                       const std::filesystem::path& raw,
                       const std::optional<std::filesystem::path>& work = std::nullopt,
                       bool stereo = false);
    void handle_step(int fd, const Request& request);
    void handle_preview(int fd, const Request& request);
    void handle_train(int fd, const Request& request);
    void handle_status(int fd, const Request& request);
    void handle_cancel_train(int fd, const Request& request);
    void handle_preprocess(int fd, const Request& request);
    void handle_convert(int fd, const Request& request);
    struct ConversionPlan {
        std::string directory, input, name, workflow, output_kind, voice, model, index, pitch_edits;
        int pitch=0, filter_radius=3;
        double index_rate=0, rms=0.5, protect=0.5;
        bool separate=false, deharmony=false, dereverb=false, deecho=false, denoise=false;
        bool preprocess_only=false;
        std::string source_task;
        bool pitch_preview=false;
        double preview_start=0, preview_end=0;
        std::string analysis_id;
    };
    // Owns only values and task-owned paths, never request or socket references.
    std::string convert_audio(const ConversionPlan& plan, std::uint64_t generation=0,
                              TaskArtifacts* artifacts=nullptr);
    WebWorkflowIdentity audio_identity(const ConversionPlan& plan) const;
    ConversionPlan restore_audio_plan(const TaskRecord& record) const;
    voxmutatio::VCConfig conversion_config(const ConversionPlan& plan) const;
    void resolve_saved_audio_model(ConversionPlan& plan) const;
    void inherit_audio_stages(const JobTicket& ticket, const ConversionPlan& plan, TaskArtifacts& artifacts);
    void start_audio_task(int fd, ConversionPlan plan, const std::string& task_id);
    void run_audio_task(JobTicket ticket, ConversionPlan plan) noexcept;
    void prepare_training_task(const JobTicket& ticket);
    void run_training_task(JobTicket ticket) noexcept;
    void execute_training_task(const JobTicket& ticket, const TaskRecord& record);
    void finish_audio_task(const JobTicket& ticket, std::string_view result,
                           std::string_view error, bool paused=false) noexcept;
    void handle_training_runs(int fd);
    void handle_publish(int fd, const Request& request);

    [[nodiscard]] static const char* allowed_api_method(const std::string& path);

    Paths paths_;
    std::shared_ptr<const ProcessRunner> process_runner_;
    const std::string instance_id_;
    std::unique_ptr<TaskStore> tasks_;
    std::atomic<bool> task_storage_failed_{false};
    std::mutex storage_mutex_;
    std::optional<TaskStorageUsage> storage_cached_;
    std::chrono::steady_clock::time_point storage_checked_{};
    JobState job_;
    std::mutex gpu_mutex_;
    std::mutex decode_mutex_;
    // Writers lock GPU before material; preview only locks material. Never hold
    // this lock during CLI execution or socket writes.
    std::mutex material_mutex_;
};

}  // namespace resonantia::web
