#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace resonantia_test {

inline std::filesystem::path artifact_directory() {
    if (const char* configured = std::getenv("RESONANTIA_TEST_OUTPUT_DIR");
        configured != nullptr && *configured != '\0') {
        return configured;
    }
    return std::filesystem::temp_directory_path();
}

inline std::filesystem::path artifact_path(std::string_view filename) {
    return artifact_directory() / filename;
}

inline std::filesystem::path unique_artifact_path(std::string_view stem,
                                                  std::string_view extension) {
    static std::atomic<unsigned long long> sequence{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return artifact_directory() /
           (std::string(stem) + "-" + std::to_string(stamp) + "-" +
            std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) +
            std::string(extension));
}

class ScopedArtifact {
public:
    explicit ScopedArtifact(std::filesystem::path path) : path_(std::move(path)) {}
    ScopedArtifact(const ScopedArtifact&) = delete;
    ScopedArtifact& operator=(const ScopedArtifact&) = delete;

    ~ScopedArtifact() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace resonantia_test
