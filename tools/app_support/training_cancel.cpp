// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
#include "tools/app_support/training_cancel.h"

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

namespace resonantia::training_control {

TrainingCancelReader::TrainingCancelReader(const std::string& path) {
    if (path.empty()) return; // Backward-compatible CLI: no channel I/O.
    // NONBLOCK prevents a malformed FIFO path from hanging before fstat.
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd_ < 0) { error_ = errno; return; }
    struct stat state{};
    if (::fstat(fd_, &state) != 0) error_ = errno;
    else if (!S_ISREG(state.st_mode) || state.st_size != 1) error_ = EINVAL;
}
TrainingCancelReader::~TrainingCancelReader() { if (fd_ >= 0) ::close(fd_); }
TrainingResult TrainingCancelReader::poll() {
    if (error_) return TrainingResult::kError;
    if (fd_ < 0) return TrainingResult::kCompleted;
    char bytes[2]{};
    ssize_t count;
    do { count = ::pread(fd_, bytes, sizeof bytes, 0); } while (count < 0 && errno == EINTR);
    if (count != 1 || (bytes[0] != '0' && bytes[0] != '1')) {
        error_ = count < 0 ? errno : EINVAL;
        return TrainingResult::kError;
    }
    return bytes[0] == '1' ? TrainingResult::kCancelled : TrainingResult::kCompleted;
}

TrainingCancelSource::TrainingCancelSource(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "create training cancel channel");
    ssize_t count;
    do { count = ::pwrite(fd_, "0", 1, 0); } while (count < 0 && errno == EINTR);
    if (count != 1) {
        const int error = count < 0 ? errno : EIO;
        ::close(fd_); fd_ = -1;
        throw std::system_error(error, std::generic_category(), "initialize training cancel channel");
    }
}
TrainingCancelSource::~TrainingCancelSource() { close(); }
CancelDelivery TrainingCancelSource::request() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_) return CancelDelivery::kError;
    if (fd_ < 0) return CancelDelivery::kClosed;
    if (requested_) return CancelDelivery::kDelivered;
    ssize_t count;
    do { count = ::pwrite(fd_, "1", 1, 0); } while (count < 0 && errno == EINTR);
    if (count != 1) {
        error_ = count < 0 ? errno : EIO;
        return CancelDelivery::kError;
    }
    requested_ = true;
    return CancelDelivery::kDelivered;
}
bool TrainingCancelSource::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fd_ >= 0) {
        const int fd = fd_; fd_ = -1;
        // Linux releases the descriptor even on close errors; never retry and
        // risk closing a descriptor reused by another request thread.
        if (::close(fd) != 0 && !error_) error_ = errno;
    }
    return error_ == 0;
}

}  // namespace resonantia::training_control
