// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

class TempDir {
public:
    TempDir() {
        std::array<char, 64> pattern {};
        constexpr char value[] = "/tmp/resonantia-build-index.XXXXXX";
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

int run_build(const std::string& executable, const std::string& hubert,
              const fs::path& input, const fs::path& output) {
    const pid_t child = ::fork();
    if (child < 0) return -1;
    if (child == 0) {
        const std::string input_string = input.string();
        const std::string output_string = output.string();
        char* const arguments[] = {
            const_cast<char*>(executable.c_str()),
            const_cast<char*>("--hubert"), const_cast<char*>(hubert.c_str()),
            const_cast<char*>("--input-dir"),
            const_cast<char*>(input_string.c_str()),
            const_cast<char*>("--output"),
            const_cast<char*>(output_string.c_str()), nullptr};
        ::execv(executable.c_str(), arguments);
        _exit(127);
    }
    int status = 0;
    while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::vector<std::uint8_t> read_binary(const fs::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const std::streamoff size = input.tellg();
    if (size <= 0) return {};
    input.seekg(0);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!input) return {};
    return bytes;
}

std::uint64_t read_u64_le(const std::uint8_t* data) {
    std::uint64_t value = 0;
    for (int byte = 0; byte < 8; ++byte) {
        value |= static_cast<std::uint64_t>(data[byte]) << (byte * 8);
    }
    return value;
}

bool valid_index(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() < 16) return false;
    const std::uint64_t rows = read_u64_le(bytes.data());
    const std::uint64_t dim = read_u64_le(bytes.data() + 8);
    if (rows == 0 || dim != 768 ||
        rows > (std::numeric_limits<std::size_t>::max() - 16) /
                   (dim * sizeof(float)) ||
        bytes.size() != 16 + static_cast<std::size_t>(rows * dim) * sizeof(float)) {
        return false;
    }
    for (std::size_t offset = 16; offset < bytes.size(); offset += sizeof(float)) {
        float value = 0.0f;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        if (!std::isfinite(value)) return false;
    }
    return true;
}

bool has_stage_file(const fs::path& directory) {
    std::error_code error;
    for (fs::directory_iterator it(directory, error), end;
         !error && it != end; it.increment(error)) {
        if (it->path().filename().string().find(".stage.") != std::string::npos) {
            return true;
        }
    }
    return error || false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: test_build_index_contract <build_index> <hubert> "
                     "<speech-a> <speech-b>\n";
        return 2;
    }

    TempDir temp;
    if (temp.path().empty()) return 1;
    const fs::path first_input = temp.path() / "first";
    const fs::path second_input = temp.path() / "second";
    fs::create_directories(first_input);
    fs::create_directories(second_input);

    // Create the same name->audio mapping in opposite creation order. A stable
    // build must sort paths rather than inherit directory iteration order.
    fs::copy_file(argv[4], first_input / "b.wav");
    fs::copy_file(argv[3], first_input / "a.wav");
    fs::copy_file(argv[3], second_input / "a.wav");
    fs::copy_file(argv[4], second_input / "b.wav");

    const fs::path first_output = temp.path() / "first.index";
    const fs::path old_witness = temp.path() / "old-index-witness";
    {
        std::ofstream marker(first_output, std::ios::binary);
        marker << "old-index";
    }
    fs::create_hard_link(first_output, old_witness);

    if (run_build(argv[1], argv[2], first_input, first_output) != 0) return 1;
    const auto first = read_binary(first_output);
    if (!valid_index(first) || read_binary(old_witness) !=
                                   std::vector<std::uint8_t>{'o', 'l', 'd', '-',
                                                              'i', 'n', 'd', 'e', 'x'}) {
        return 1;
    }

    const fs::path second_output = temp.path() / "second.index";
    if (run_build(argv[1], argv[2], second_input, second_output) != 0) return 1;
    const auto second = read_binary(second_output);
    if (!valid_index(second) || first != second || has_stage_file(temp.path())) {
        return 1;
    }

    const fs::path blocked_target = temp.path() / "blocked.index";
    fs::create_directory(blocked_target);
    {
        std::ofstream marker(blocked_target / "marker", std::ios::binary);
        marker << "preserve";
    }
    if (run_build(argv[1], argv[2], first_input, blocked_target) == 0 ||
        read_binary(blocked_target / "marker") !=
            std::vector<std::uint8_t>{'p', 'r', 'e', 's', 'e', 'r', 'v', 'e'} ||
        has_stage_file(temp.path())) {
        return 1;
    }

    std::cout << "build_index atomic and deterministic contract passed\n";
    return 0;
}
