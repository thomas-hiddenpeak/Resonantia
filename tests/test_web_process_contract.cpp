// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <sys/resource.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

#include "tools/web/workflow_support.h"

namespace fs = std::filesystem;
using namespace resonantia::web;
using namespace std::chrono_literals;

namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
std::string read(const fs::path& path) {
    std::ifstream input(path);
    return std::string(std::istreambuf_iterator<char>(input), {});
}
std::size_t descriptor_count() {
    std::size_t count = 0;
    for (const auto& entry : fs::directory_iterator("/proc/self/fd")) {
        (void)entry;
        ++count;
    }
    return count;
}
void require_no_children() {
    // Some Jetson kernels omit /proc/<pid>/task/<tid>/children. Read PPIDs
    // from ordinary proc stat metadata; never compete with the runner's waitpid
    // or signal a process. Both live children and zombies retain this PPID.
    for (const auto& entry : fs::directory_iterator("/proc")) {
        const std::string name = entry.path().filename().string();
        if (name.empty() || name.find_first_not_of("0123456789") != std::string::npos)
            continue;
        const fs::path stat_path = entry.path() / "stat";
        std::ifstream input(stat_path);
        std::string stat;
        std::getline(input, stat);
        if (stat.empty()) {
            std::error_code error;
            const bool exists = fs::exists(stat_path, error);
            require(!error && !exists, "cannot read existing process stat");
            continue;  // An unrelated process exited during the read.
        }
        const auto comm_end = stat.rfind(')');
        require(comm_end != std::string::npos, "invalid process stat");
        std::istringstream fields(stat.substr(comm_end + 1));
        char state = 0;
        pid_t parent = -1;
        require(bool(fields >> state >> parent), "invalid process parent metadata");
        require(parent != getpid(), "runner left a live or unreaped child");
    }
}

int child_main(const std::string& mode, const char* marker) {
    {
        std::ofstream output(marker, std::ios::app);
        output << mode << '\n';
        output.close();
        if (!output) return 91;
    }
    std::cout << mode << '\n' << std::flush;
    if (mode == "--child-signal") {
        std::signal(SIGTERM, SIG_DFL);
        std::raise(SIGTERM);  // Only this short-lived test helper signals itself.
        return 92;
    }
    return 0;
}
}

int main(int argc, char** argv) {
    if (argc == 3 && (std::string(argv[1]) == "--child-ok" ||
                      std::string(argv[1]) == "--child-signal"))
        return child_main(argv[1], argv[2]);
    try {
        require(argc == 1, "unexpected arguments");
        const auto started = std::chrono::steady_clock::now();
        const fs::path self = fs::canonical(argv[0]);
        const std::size_t baseline_fds = descriptor_count();
        require_no_children();
        fs::path fixture_path;
        {
            TempDir tree(fs::temp_directory_path(), "resonantia-process-contract-");
            fixture_path = tree.path();
            const fs::path marker = fixture_path / "child-invocations";
            PosixProcessRunner runner;
            int failures_seen = 0;
            for (const std::string scenario : {"empty-argv", "missing-program", "log-directory", "signal"}) {
                ProcessSpec spec;
                spec.argv = {self.string(), "--child-ok", marker.string()};
                spec.log_path = fixture_path / (scenario + ".log");
                ProcessTermination expected = ProcessTermination::kLaunchFailed;
                int expected_errno = 0;
                if (scenario == "empty-argv") {
                    spec.argv.clear();
                    expected_errno = EINVAL;
                } else if (scenario == "missing-program") {
                    spec.argv.front() = (fixture_path / "no-such-program").string();
                    expected = ProcessTermination::kExecFailed;
                    expected_errno = ENOENT;
                } else if (scenario == "log-directory") {
                    spec.log_path = fixture_path;  // Fails naturally; no permission changes.
                    expected_errno = EISDIR;
                } else {
                    spec.argv[1] = "--child-signal";
                    expected = ProcessTermination::kSignaled;
                }
                const std::string calls_before = read(marker);
                const auto case_start = std::chrono::steady_clock::now();
                const ProcessResult result = runner.run(spec);
                require(std::chrono::steady_clock::now() - case_start < 2s,
                        scenario + ": natural failure exceeded 2s budget");
                require(!result.success() && result.termination == expected &&
                            result.system_error == expected_errno && result.io_error == 0 &&
                            result.exit_code == -1 &&
                            result.signal_number == (scenario == "signal" ? SIGTERM : 0),
                        scenario + ": incorrect failure classification");
                require_no_children();
                require(descriptor_count() == baseline_fds, scenario + ": descriptor leak");
                require(read(marker) == calls_before + (scenario == "signal" ? "--child-signal\n" : ""),
                        scenario + ": unexpected child execution");
                if (scenario == "signal")
                    require(read(spec.log_path).find("--child-signal\n") != std::string::npos,
                            "signal child output was not drained");
                ++failures_seen;

                // Every failure is followed by a successful invocation of the
                // same real runner, never concurrently with another helper.
                ProcessSpec recovery;
                recovery.argv = {self.string(), "--child-ok", marker.string()};
                recovery.log_path = fixture_path / "recovery.log";
                const ProcessResult recovered = runner.run(recovery);
                require(recovered.success(), scenario + ": next invocation failed");
                require_no_children();
                require(descriptor_count() == baseline_fds, scenario + ": recovery leaked descriptors");
                require(read(marker) == calls_before +
                            (scenario == "signal" ? "--child-signal\n" : "") + "--child-ok\n",
                        scenario + ": recovery child did not run exactly once");
                std::cout << scenario << ": expected failure, no children/FD delta, recovery success\n";
            }
            require(failures_seen == 4, "not all natural failure cases ran");
            std::uintmax_t bytes = 0;
            for (const auto& entry : fs::directory_iterator(fixture_path)) {
                require(entry.is_regular_file(), "unexpected fixture artifact");
                bytes += entry.file_size();
            }
            require(bytes <= 8192, "fixture/log output exceeded 8KiB");
            std::cout << "Temporary output=" << bytes << " bytes; max helper children=1\n";
        }
        require(!fs::exists(fixture_path), "temporary fixture was not removed");
        require_no_children();
        require(descriptor_count() == baseline_fds, "final descriptor count changed");
        const auto elapsed = std::chrono::steady_clock::now() - started;
        require(elapsed < 10s, "process group exceeded 10s test budget");
        rusage own{}, child{};
        require(getrusage(RUSAGE_SELF, &own) == 0 && getrusage(RUSAGE_CHILDREN, &child) == 0,
                "cannot inspect test memory usage");
        require(own.ru_maxrss + child.ru_maxrss <= 256 * 1024,
                "test parent/child RSS budget exceeded 256MiB");
        std::cout << "Natural process failures: 4/4; recoveries: 4/4; elapsed="
                  << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
                  << " ms; parent+child peak RSS upper bound=" << own.ru_maxrss + child.ru_maxrss
                  << " KiB; fixture removed, no remaining children\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Process contract failure: " << error.what() << '\n';
        return 1;
    }
}
