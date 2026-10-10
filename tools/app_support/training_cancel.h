// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
#pragma once

#include <mutex>
#include <string>

// Private CLI/Web orchestration, independent of CUDA and trainer APIs.
namespace resonantia::training_control {

enum class TrainingResult { kCompleted = 0, kError = 1, kCancelled = 2 };
enum class CancelDelivery { kDelivered, kClosed, kError };

// Optional reader opens once, so an old CLI cannot follow a replaced path.
class TrainingCancelReader {
public:
    explicit TrainingCancelReader(const std::string& path);
    ~TrainingCancelReader();
    TrainingCancelReader(const TrainingCancelReader&) = delete;
    TrainingCancelReader& operator=(const TrainingCancelReader&) = delete;
    TrainingResult poll();
    int error_code() const { return error_; }
private:
    int fd_ = -1;
    int error_ = 0;
};

// One source belongs to one staged job. request()/close() serialize: after
// close returns, even a retained shared_ptr cannot write or recreate a path.
class TrainingCancelSource {
public:
    explicit TrainingCancelSource(const std::string& path);
    ~TrainingCancelSource();
    TrainingCancelSource(const TrainingCancelSource&) = delete;
    TrainingCancelSource& operator=(const TrainingCancelSource&) = delete;
    CancelDelivery request();
    bool close();
private:
    std::mutex mutex_;
    int fd_ = -1;
    int error_ = 0;
    bool requested_ = false;
};

// The body is indivisible to cancellation. A reported body failure wins over
// cancellation during that body. The final poll also protects the export gate.
template<class CompleteStep>
TrainingResult run_training_iterations(int steps, TrainingCancelReader& reader,
                                       CompleteStep complete_step) {
    for (int iteration = 0; iteration < steps; ++iteration) {
        const auto decision = reader.poll();
        if (decision != TrainingResult::kCompleted) return decision;
        if (!complete_step(iteration)) return TrainingResult::kError;
    }
    return reader.poll();
}

}  // namespace resonantia::training_control
