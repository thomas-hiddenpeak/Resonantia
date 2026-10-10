#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "tools/web/http_transport.h"
#include "tools/app_support/training_cancel.h"
#include "tools/web/workflow_support.h"

namespace resonantia::web {

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

    enum class JobKind { kNone, kStep, kTraining };
    enum class TrainingOutcome { kDone, kError, kCancelled };
    struct JobTicket { std::uint64_t generation = 0; std::string public_id; };
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
        std::string public_id;
        JobKind kind = JobKind::kNone;
        bool cancel_requested = false;
        bool cancel_error = false;
        std::shared_ptr<training_control::TrainingCancelSource> cancel_source;
        bool cancelled = false;
        std::string name;
        std::string stage = "idle";
        std::atomic<bool> running{false};
        bool done = false;
        bool error = false;
    };

    bool reserve_job(const std::string& name, JobKind kind, JobTicket& ticket);
    void set_stage(std::uint64_t generation, const std::string& stage);
    void finish(std::uint64_t generation, const std::string& stage, bool error);
    void finish_training(std::uint64_t generation, TrainingOutcome outcome);
    bool training_checkpoint(std::uint64_t generation, const std::string& stage);
    bool cancellation_requested(std::uint64_t generation);
    TrainingOutcome run_training(std::uint64_t generation, std::string name, std::string mode, int steps, int segment,
                      int epochs, bool separate, bool dereverb, bool deecho,
                      bool denoise, bool deharmony, bool vad);
    void run_step(std::uint64_t generation, std::string name, std::string step, std::string model,
                  std::unique_lock<std::mutex> gpu_lock);

    [[nodiscard]] std::string voices_json();
    [[nodiscard]] std::string separation_models_json() const;
    [[nodiscard]] bool preprocess_audio_file(const std::string& input,
                                             const std::string& temporary_directory,
                                             bool separate, bool deharmony,
                                             bool dereverb, bool deecho, bool denoise,
                                             std::string& cleaned, std::string& log) const;

    void handle_material(int fd, const Request& request);
    void handle_step(int fd, const Request& request);
    void handle_preview(int fd, const Request& request);
    void handle_train(int fd, const Request& request);
    void handle_status(int fd, const Request& request);
    void handle_cancel_train(int fd, const Request& request);
    void handle_preprocess(int fd, const Request& request);
    void handle_convert(int fd, const Request& request);

    [[nodiscard]] static const char* allowed_api_method(const std::string& path);

    Paths paths_;
    std::shared_ptr<const ProcessRunner> process_runner_;
    const std::string instance_id_;
    JobState job_;
    std::mutex gpu_mutex_;
    // Writers lock GPU before material; preview only locks material. Never hold
    // this lock during CLI execution or socket writes.
    std::mutex material_mutex_;
};

}  // namespace resonantia::web
