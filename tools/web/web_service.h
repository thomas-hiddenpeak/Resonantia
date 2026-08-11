#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "tools/web/http_transport.h"
#include "tools/web/workflow_support.h"

namespace resonantia::web {

class WebService final : public std::enable_shared_from_this<WebService> {
public:
    WebService(Paths paths, std::shared_ptr<const ProcessRunner> process_runner);
    WebService(const WebService&) = delete;
    WebService& operator=(const WebService&) = delete;

    void handle_request(int fd, const Request& request);

private:
    struct JobState {
        std::mutex mu;
        std::string name;
        std::string stage = "idle";
        std::atomic<bool> running{false};
        bool done = false;
        bool error = false;
    };

    void set_stage(const std::string& stage);
    void finish(const std::string& stage, bool error);
    void run_training(std::string name, std::string mode, int steps, int segment,
                      int epochs, bool separate, bool dereverb, bool deecho,
                      bool denoise, bool deharmony, bool vad);
    void run_step(std::string name, std::string step, std::string model,
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
    void handle_preprocess(int fd, const Request& request);
    void handle_convert(int fd, const Request& request);

    [[nodiscard]] static const char* allowed_api_method(const std::string& path);

    Paths paths_;
    std::shared_ptr<const ProcessRunner> process_runner_;
    JobState job_;
    std::mutex gpu_mutex_;
};

}  // namespace resonantia::web
