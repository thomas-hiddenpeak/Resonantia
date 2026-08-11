// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

class TempDir {
public:
    TempDir() {
        std::array<char, 64> pattern {};
        constexpr char value[] = "/tmp/resonantia-vc-bounded.XXXXXX";
        std::memcpy(pattern.data(), value, sizeof(value));
        if (char* created = ::mkdtemp(pattern.data())) path_ = created;
    }
    ~TempDir() {
        std::error_code error;
        if (!path_.empty()) fs::remove_all(path_, error);
    }
    [[nodiscard]] const fs::path& path() const noexcept { return path_; }

private:
    fs::path path_;
};

struct ProcessResult {
    int exit_code = -1;
    std::string output;
};

ProcessResult run_process(const std::vector<std::string>& arguments) {
    int pipe_fds[2] = {-1, -1};
    if (::pipe(pipe_fds) != 0) return {};
    const pid_t child = ::fork();
    if (child < 0) {
        ::close(pipe_fds[0]);
        ::close(pipe_fds[1]);
        return {};
    }
    if (child == 0) {
        ::close(pipe_fds[0]);
        if (::dup2(pipe_fds[1], STDOUT_FILENO) < 0 ||
            ::dup2(pipe_fds[1], STDERR_FILENO) < 0) {
            _exit(126);
        }
        ::close(pipe_fds[1]);
        std::vector<char*> argv;
        argv.reserve(arguments.size() + 1);
        for (const auto& argument : arguments) {
            argv.push_back(const_cast<char*>(argument.c_str()));
        }
        argv.push_back(nullptr);
        ::execv(arguments.front().c_str(), argv.data());
        _exit(127);
    }

    ::close(pipe_fds[1]);
    ProcessResult result;
    std::array<char, 4096> buffer {};
    for (;;) {
        const ssize_t count = ::read(pipe_fds[0], buffer.data(), buffer.size());
        if (count > 0) {
            result.output.append(buffer.data(), static_cast<std::size_t>(count));
            continue;
        }
        if (count < 0 && errno == EINTR) continue;
        break;
    }
    ::close(pipe_fds[0]);
    int status = 0;
    while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    if (WIFEXITED(status)) result.exit_code = WEXITSTATUS(status);
    return result;
}

std::uint32_t read_u32_le(const std::uint8_t* data) {
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8) |
           (static_cast<std::uint32_t>(data[2]) << 16) |
           (static_cast<std::uint32_t>(data[3]) << 24);
}

std::uint16_t read_u16_le(const std::uint8_t* data) {
    return static_cast<std::uint16_t>(data[0]) |
           static_cast<std::uint16_t>(data[1] << 8);
}

bool valid_40k_mono_pcm16_wav(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::array<std::uint8_t, 44> header {};
    input.read(reinterpret_cast<char*>(header.data()), header.size());
    return input && std::memcmp(header.data(), "RIFF", 4) == 0 &&
           std::memcmp(header.data() + 8, "WAVE", 4) == 0 &&
           std::memcmp(header.data() + 12, "fmt ", 4) == 0 &&
           read_u16_le(header.data() + 20) == 1 &&
           read_u16_le(header.data() + 22) == 1 &&
           read_u32_le(header.data() + 24) == 40'000 &&
           read_u16_le(header.data() + 34) == 16 &&
           std::memcmp(header.data() + 36, "data", 4) == 0 &&
           read_u32_le(header.data() + 40) > 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: test_vc_convert_bounded_contract <vc_convert> "
                     "<vc_batch> <hubert> <rmvpe> <vits-v2> <speech>\n";
        return 2;
    }
    TempDir temp;
    if (temp.path().empty()) return 1;
    const fs::path output = temp.path() / "converted.wav";
    const ProcessResult result = run_process({
        argv[1], "--hubert", argv[3], "--rmvpe", argv[4], "--model", argv[5],
        "--version", "v2", "--sr", "40000", "--input", argv[6], "--output",
        output.string(),
    });
    if (result.exit_code != 0 ||
        result.output.find("File path: bounded HuBERT") == std::string::npos ||
        result.output.find("Done (") == std::string::npos ||
        !valid_40k_mono_pcm16_wav(output)) {
        std::cerr << "vc_convert bounded contract failed: exit="
                  << result.exit_code << "\n" << result.output;
        return 1;
    }

    const fs::path input_directory = temp.path() / "batch-input";
    const fs::path output_directory = temp.path() / "batch-output";
    fs::create_directory(input_directory);
    fs::copy_file(argv[6], input_directory / "sample.wav");
    const ProcessResult batch = run_process({
        argv[2], "--hubert", argv[3], "--rmvpe", argv[4], "--model", argv[5],
        "--version", "v2", "--sr", "40000", "--input-dir",
        input_directory.string(), "--output-dir", output_directory.string(),
    });
    if (batch.exit_code != 0 ||
        batch.output.find("File path: bounded HuBERT") == std::string::npos ||
        batch.output.find("Batch complete: 1 succeeded, 0 failed") ==
            std::string::npos ||
        !valid_40k_mono_pcm16_wav(output_directory / "sample.wav")) {
        std::cerr << "vc_batch bounded contract failed: exit="
                  << batch.exit_code << "\n" << batch.output;
        return 1;
    }

    std::cout << "bounded vc_convert and vc_batch caller contract passed\n";
    return 0;
}
