#pragma once

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace resonantia_test {

struct SubprocessResult {
    int launch_error = 0;
    int wait_error = 0;
    int status = 0;
    bool timed_out = false;
    // Linux reports ru_maxrss in KiB.  Capturing it at reap time makes model
    // resource comparisons independent of the parent Catch2 process high-water
    // mark.
    long max_rss_kib = 0;

    [[nodiscard]] bool exited_with(int code) const noexcept {
        return !timed_out && launch_error == 0 && wait_error == 0 &&
               WIFEXITED(status) &&
               WEXITSTATUS(status) == code;
    }

    [[nodiscard]] bool killed_by(int signal) const noexcept {
        return !timed_out && launch_error == 0 && wait_error == 0 &&
               WIFSIGNALED(status) &&
               WTERMSIG(status) == signal;
    }
};

// Re-executes this Catch2 binary with a single named test. Re-exec, rather than
// running the body directly after fork(), gives CUDA-facing helpers a clean
// process state and contains fatal mmap signals to the child.
inline SubprocessResult run_self_test(const char* test_name,
                                      int timeout_ms = 30'000,
                                      const char* result_path = nullptr) {
    SubprocessResult result;
    if (timeout_ms <= 0) {
        result.launch_error = EINVAL;
        return result;
    }
    std::array<char, 4096> executable{};
    const ssize_t length =
        ::readlink("/proc/self/exe", executable.data(), executable.size() - 1);
    if (length < 0 || static_cast<std::size_t>(length) >= executable.size() - 1) {
        result.launch_error = length < 0 ? errno : ENAMETOOLONG;
        return result;
    }
    executable[static_cast<std::size_t>(length)] = '\0';

    // Build the optional child environment before fork.  The fork-to-exec
    // window may inherit CUDA/runtime locks, so it must use only
    // async-signal-safe calls (setenv is deliberately avoided there).
    std::string result_environment;
    std::vector<char*> child_environment;
    if (result_path != nullptr) {
        constexpr std::string_view prefix =
            "RESONANTIA_SUBPROCESS_RESULT_PATH=";
        for (char** entry = ::environ; entry != nullptr && *entry != nullptr;
             ++entry) {
            if (!std::string_view(*entry).starts_with(prefix)) {
                child_environment.push_back(*entry);
            }
        }
        result_environment = std::string(prefix) + result_path;
        child_environment.push_back(result_environment.data());
        child_environment.push_back(nullptr);
    }

    const pid_t child = ::fork();
    if (child < 0) {
        result.launch_error = errno;
        return result;
    }
    if (child == 0) {
        const struct rlimit no_core {0, 0};
        if (::setrlimit(RLIMIT_CORE, &no_core) != 0) _exit(126);
        if (result_path != nullptr) {
            char* const arguments[] = {
                executable.data(), const_cast<char*>(test_name),
                const_cast<char*>("--reporter"), const_cast<char*>("compact"),
                nullptr};
            ::execve(executable.data(), arguments, child_environment.data());
        } else {
            ::execl(executable.data(), executable.data(), test_name,
                    "--reporter", "compact", static_cast<char*>(nullptr));
        }
        _exit(127);
    }

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    for (;;) {
        struct rusage usage {};
        const pid_t waited = ::wait4(child, &result.status, WNOHANG, &usage);
        if (waited == child) {
            result.max_rss_kib = usage.ru_maxrss;
            return result;
        }
        if (waited < 0 && errno != EINTR) {
            result.wait_error = errno;
            return result;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            result.timed_out = true;
            if (::kill(child, SIGKILL) != 0 && errno != ESRCH) {
                result.wait_error = errno;
                return result;
            }
            pid_t reaped = -1;
            do {
                reaped = ::wait4(child, &result.status, 0, &usage);
            } while (reaped < 0 && errno == EINTR);
            if (reaped == child) result.max_rss_kib = usage.ru_maxrss;
            if (reaped < 0 && errno != ECHILD) result.wait_error = errno;
            return result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

}  // namespace resonantia_test
