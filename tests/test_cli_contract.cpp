// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

// Asset-free process-level CLI contract tests. The four executable paths are
// supplied by CTest so this test does not assume any particular build layout:
//   test_cli_contract <vc_convert> <vc_batch> <vc_train> <vc_preprocess>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct ProcessResult {
    int exit_code = -1;
    std::string output;
};

ProcessResult run_process(const std::string& executable,
                          const std::vector<std::string>& arguments) {
    int pipe_fds[2] = {-1, -1};
    if (::pipe(pipe_fds) != 0) {
        return {-1, "pipe failed: " + std::to_string(errno)};
    }

    const pid_t child = ::fork();
    if (child == -1) {
        const int saved_errno = errno;
        ::close(pipe_fds[0]);
        ::close(pipe_fds[1]);
        return {-1, "fork failed: " + std::to_string(saved_errno)};
    }
    if (child == 0) {
        ::close(pipe_fds[0]);
        ::dup2(pipe_fds[1], STDOUT_FILENO);
        ::dup2(pipe_fds[1], STDERR_FILENO);
        ::close(pipe_fds[1]);

        std::vector<char*> argv;
        argv.reserve(arguments.size() + 2);
        argv.push_back(const_cast<char*>(executable.c_str()));
        for (const auto& argument : arguments) {
            argv.push_back(const_cast<char*>(argument.c_str()));
        }
        argv.push_back(nullptr);
        ::execv(executable.c_str(), argv.data());
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
        if (count == -1 && errno == EINTR) continue;
        break;
    }
    ::close(pipe_fds[0]);

    int status = 0;
    while (::waitpid(child, &status, 0) == -1 && errno == EINTR) {}
    if (WIFEXITED(status)) result.exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) result.exit_code = 128 + WTERMSIG(status);
    return result;
}

class TempDir {
public:
    TempDir() {
        std::array<char, 64> pattern {};
        const std::string value = "/tmp/resonantia-cli-contract.XXXXXX";
        std::copy(value.begin(), value.end(), pattern.begin());
        if (char* created = ::mkdtemp(pattern.data())) path_ = created;
    }

    ~TempDir() {
        std::error_code error;
        if (!path_.empty()) fs::remove_all(path_, error);
    }

    [[nodiscard]] const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

int failures = 0;

void expect_rejected(const std::string& label, const std::string& executable,
                     const std::vector<std::string>& arguments,
                     std::string_view expected_message) {
    const ProcessResult result = run_process(executable, arguments);
    if (result.exit_code != 0 && result.output.find(expected_message) != std::string::npos) {
        return;
    }
    ++failures;
    std::cerr << "FAIL " << label << ": exit=" << result.exit_code
              << ", expected output containing: " << expected_message << "\n"
              << "actual output:\n" << result.output << "\n";
}

void expect_success_contains(const std::string& label, const std::string& executable,
                             const std::vector<std::string>& arguments,
                             std::initializer_list<std::string_view> expected_messages) {
    const ProcessResult result = run_process(executable, arguments);
    bool matched = result.exit_code == 0;
    for (const auto expected : expected_messages) {
        matched = matched && result.output.find(expected) != std::string::npos;
    }
    if (matched) return;
    ++failures;
    std::cerr << "FAIL " << label << ": exit=" << result.exit_code
              << ", output did not describe every accepted option\n"
              << "actual output:\n" << result.output << "\n";
}

std::vector<std::string> convert_required() {
    return {"--hubert", "missing-hubert", "--model", "missing-model",
            "--rmvpe", "missing-rmvpe", "--input", "missing-input.wav",
            "--output", "missing-output.wav"};
}

std::vector<std::string> batch_required() {
    return {"--hubert", "missing-hubert", "--model", "missing-model",
            "--rmvpe", "missing-rmvpe", "--input-dir", "missing-input",
            "--output-dir", "missing-output"};
}

std::vector<std::string> train_required() {
    return {"--hubert", "missing-hubert", "--rmvpe", "missing-rmvpe",
            "--pretrained", "missing-model", "--target", "missing-target.wav",
            "--out", "missing-output.safetensors"};
}

std::vector<std::string> preprocess_required(const fs::path& root) {
    return {"--input", (root / "missing-input.wav").string(),
            "--output-dir", (root / "output").string()};
}

void append(std::vector<std::string>& destination,
            std::initializer_list<std::string> values) {
    destination.insert(destination.end(), values.begin(), values.end());
}

struct DependencyCase {
    std::string label;
    std::vector<std::string> arguments;
    std::string_view expected_message;
};

std::vector<DependencyCase> dependency_cases(std::vector<std::string> base) {
    auto with = [&base](std::initializer_list<std::string> values) {
        auto arguments = base;
        append(arguments, values);
        return arguments;
    };

    return {
        {"selected HuBERT requires --hubert",
         with({"--rmvpe", "missing-rmvpe"}),
         "error: --hubert is required when --content-encoder=hubert"},
        {"selected WavLM requires --wavlm",
         with({"--content-encoder", "wavlm", "--rmvpe", "missing-rmvpe"}),
         "error: --wavlm is required when --content-encoder=wavlm"},
        {"selected RMVPE requires --rmvpe",
         with({"--hubert", "missing-hubert"}),
         "error: --rmvpe is required when --f0-method=rmvpe"},
        {"selected FCPE requires --fcpe",
         with({"--hubert", "missing-hubert", "--f0-method", "fcpe"}),
         "error: --fcpe is required when --f0-method=fcpe"},
        {"positive index rate requires --index",
         with({"--hubert", "missing-hubert", "--rmvpe", "missing-rmvpe",
               "--index-rate", "0.5"}),
         "error: --index is required when --index-rate > 0"},
    };
}

void expect_dependency_contracts(const std::string& command,
                                 const std::string& executable,
                                 std::vector<std::string> base) {
    for (const auto& test_case : dependency_cases(std::move(base))) {
        expect_rejected(command + " " + test_case.label, executable,
                        test_case.arguments, test_case.expected_message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: test_cli_contract <vc_convert> <vc_batch> <vc_train> "
                     "<vc_preprocess>\n";
        return 2;
    }
    const std::string convert = argv[1];
    const std::string batch = argv[2];
    const std::string train = argv[3];
    const std::string preprocess = argv[4];

    expect_success_contains(
        "vc_convert help lists the supported model contract", convert, {"--help"},
        {"--filter-radius", "--version", "--speakers", "--sr", "40 kHz",
         "--rmvpe", "--fcpe", "required when f0-method=rmvpe",
         "required when f0-method=fcpe", "required when index-rate > 0",
         "required when content-encoder=hubert",
         "required when content-encoder=wavlm", "safetensors path"});
    expect_success_contains(
        "vc_batch help lists the supported model contract", batch, {"--help"},
        {"--filter-radius", "--version", "--speakers", "--sr", "40 kHz",
         "--rmvpe", "--fcpe", "required when f0-method=rmvpe",
         "required when f0-method=fcpe", "required when index-rate > 0",
         "required when content-encoder=hubert",
         "required when content-encoder=wavlm", "safetensors path"});
    expect_success_contains(
        "vc_convert help short-circuits an earlier unknown argument", convert,
        {"--bogus", "--help"}, {"Usage:", "--content-encoder"});
    expect_success_contains(
        "vc_batch help short-circuits an earlier unknown argument", batch,
        {"--bogus", "--help"}, {"Usage:", "--content-encoder"});

    expect_rejected("vc_convert unknown argument without help", convert, {"--bogus"},
                    "Unknown argument: --bogus");
    expect_rejected("vc_batch unknown argument without help", batch, {"--bogus"},
                    "Unknown argument: --bogus");
    expect_rejected("vc_train unknown argument without help", train, {"--bogus"},
                    "Unknown argument: --bogus");
    expect_rejected("vc_preprocess unknown argument without help", preprocess, {"--bogus"},
                    "Unknown argument: --bogus");

    expect_dependency_contracts(
        "vc_convert", convert,
        {"--model", "missing-model", "--input", "missing-input.wav",
         "--output", "missing-output.wav"});
    expect_dependency_contracts(
        "vc_batch", batch,
        {"--model", "missing-model", "--input-dir", "missing-input",
         "--output-dir", "missing-output"});

    {
        auto args = convert_required();
        append(args, {"--speaker", "junk"});
        expect_rejected("vc_convert malformed integer", convert, args,
                        "error: invalid value for --speaker");
    }
    {
        auto args = convert_required();
        append(args, {"--index-rate", "NaN"});
        expect_rejected("vc_convert non-finite float", convert, args,
                        "error: invalid value for --index-rate");
    }
    {
        auto args = convert_required();
        append(args, {"--sr", "999999999999999999999999"});
        expect_rejected("vc_convert integer overflow", convert, args,
                        "error: invalid value for --sr");
    }
    {
        auto args = convert_required();
        append(args, {"--sr", "48000"});
        expect_rejected("vc_convert unsupported model rate", convert, args,
                        "error: --sr must be 40000");
    }
    {
        auto args = convert_required();
        append(args, {"--pitch"});
        expect_rejected("vc_convert missing numeric value", convert, args,
                        "error: missing value for --pitch");
    }
    {
        auto args = convert_required();
        append(args, {"--pitch", " 12"});
        expect_rejected("vc_convert leading whitespace", convert, args,
                        "error: invalid value for --pitch");
    }
    {
        auto args = convert_required();
        append(args, {"--pitch", "+-12"});
        expect_rejected("vc_convert double sign", convert, args,
                        "error: invalid value for --pitch");
    }
    {
        auto args = convert_required();
        append(args, {"--pitch", "49"});
        expect_rejected("vc_convert value range", convert, args,
                        "error: invalid value for --pitch");
    }

    {
        auto args = batch_required();
        append(args, {"--pitch", "12semitones"});
        expect_rejected("vc_batch trailing characters", batch, args,
                        "error: invalid value for --pitch");
    }
    {
        auto args = batch_required();
        append(args, {"--protect", "inf"});
        expect_rejected("vc_batch non-finite float", batch, args,
                        "error: invalid value for --protect");
    }
    {
        auto args = batch_required();
        append(args, {"--speakers", "999999999999999999999999"});
        expect_rejected("vc_batch integer overflow", batch, args,
                        "error: invalid value for --speakers");
    }
    {
        auto args = batch_required();
        append(args, {"--sr"});
        expect_rejected("vc_batch missing numeric value", batch, args,
                        "error: missing value for --sr");
    }
    {
        auto args = batch_required();
        append(args, {"--index-rate", "0.5 "});
        expect_rejected("vc_batch trailing whitespace", batch, args,
                        "error: invalid value for --index-rate");
    }
    {
        auto args = batch_required();
        append(args, {"--index-rate", "+-0"});
        expect_rejected("vc_batch double sign", batch, args,
                        "error: invalid value for --index-rate");
    }
    {
        auto args = batch_required();
        append(args, {"--sr", "7999"});
        expect_rejected("vc_batch value range", batch, args,
                        "error: invalid value for --sr");
    }
    {
        auto args = batch_required();
        append(args, {"--sr", "48000"});
        expect_rejected("vc_batch unsupported model rate", batch, args,
                        "error: --sr must be 40000");
    }

    {
        auto args = train_required();
        append(args, {"--steps", "3steps"});
        expect_rejected("vc_train trailing characters", train, args,
                        "error: invalid value for --steps");
    }
    {
        auto args = train_required();
        append(args, {"--lr", "nan"});
        expect_rejected("vc_train non-finite float", train, args,
                        "error: invalid value for --lr");
    }
    {
        auto args = train_required();
        append(args, {"--seed", "999999999999999999999999"});
        expect_rejected("vc_train unsigned overflow", train, args,
                        "error: invalid value for --seed");
    }
    {
        auto args = train_required();
        append(args, {"--seg"});
        expect_rejected("vc_train missing numeric value", train, args,
                        "error: missing value for --seg");
    }
    {
        auto args = train_required();
        append(args, {"--lr", "0x1p-2"});
        expect_rejected("vc_train hexadecimal float", train, args,
                        "error: invalid value for --lr");
    }
    {
        auto args = train_required();
        append(args, {"--steps", "100001"});
        expect_rejected("vc_train value range", train, args,
                        "error: invalid value for --steps");
    }

    TempDir temp;
    if (temp.path().empty()) {
        std::cerr << "FAIL could not create temporary directory\n";
        return 1;
    }
    {
        auto args = preprocess_required(temp.path());
        append(args, {"--sr", "forty-k"});
        expect_rejected("vc_preprocess malformed integer", preprocess, args,
                        "error: invalid value for --sr");
    }
    {
        auto args = preprocess_required(temp.path());
        append(args, {"--min-rms", "NaN"});
        expect_rejected("vc_preprocess non-finite float", preprocess, args,
                        "error: invalid value for --min-rms");
    }
    {
        auto args = preprocess_required(temp.path());
        append(args, {"--seg-sec", "1e9999"});
        expect_rejected("vc_preprocess float overflow", preprocess, args,
                        "error: invalid value for --seg-sec");
    }
    {
        auto args = preprocess_required(temp.path());
        append(args, {"--seg-sec", "1000000000"});
        expect_rejected("vc_preprocess derived sample-count range", preprocess, args,
                        "error: invalid value for --seg-sec");
    }
    {
        auto args = preprocess_required(temp.path());
        append(args, {"--vad-thresh"});
        expect_rejected("vc_preprocess missing numeric value", preprocess, args,
                        "error: missing value for --vad-thresh");
    }
    {
        auto args = preprocess_required(temp.path());
        append(args, {"--vad-thresh", "1.1"});
        expect_rejected("vc_preprocess value range", preprocess, args,
                        "error: invalid value for --vad-thresh");
    }
    {
        auto args = preprocess_required(temp.path());
        expect_rejected("vc_preprocess unreadable input", preprocess, args,
                        "error: no input audio could be read");
    }

    const fs::path registry = temp.path() / "registry";
    fs::create_directories(registry);
    std::ofstream(registry / "wrong_model.safetensors", std::ios::binary).put('\0');
    std::ofstream(registry / "wrong_model.json")
        << "{\"task\":\"dereverb\",\"label\":\"Wrong task\"}\n";
    {
        auto args = preprocess_required(temp.path());
        append(args, {"--separate", "--sep-dir", registry.string(),
                      "--separate-model", "wrong_model"});
        expect_rejected("vc_preprocess task mismatch", preprocess, args,
                        "error: model 'wrong_model' is not registered for task 'separate'");
    }
    {
        auto args = preprocess_required(temp.path());
        append(args, {"--separate", "--sep-dir", registry.string()});
        expect_rejected("vc_preprocess missing canonical model", preprocess, args,
                        "error: model 'vocal_roformer' is not registered for task 'separate'");
    }

    if (failures != 0) {
        std::cerr << failures << " CLI contract checks failed\n";
        return 1;
    }
    std::cout << "CLI contract checks passed\n";
    return 0;
}
