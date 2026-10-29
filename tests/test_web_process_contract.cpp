// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include <sys/resource.h>
#include <sys/prctl.h>
#include <sys/wait.h>
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
#include <thread>

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
    if(mode=="--child-hold") {
        {std::ofstream out(marker);out<<getpid()<<'\n';if(!out)return 91;}
        std::this_thread::sleep_for(5s);return 0;
    }
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
void inherited_resource(const fs::path& self,const fs::path& root) {
    int previous=0;require(prctl(PR_GET_CHILD_SUBREAPER,&previous)==0&&prctl(PR_SET_CHILD_SUBREAPER,1)==0,"subreaper setup");
    struct Restore{int value;~Restore(){prctl(PR_SET_CHILD_SUBREAPER,value);}} restore{previous};
    struct Children {
        pid_t owner=-1,worker=-1;
        ~Children(){for(auto pid:{owner,worker})if(pid>0){kill(pid,SIGKILL);int status;while(waitpid(pid,&status,0)<0&&errno==EINTR){}}}
    } children;
    const auto marker=root/"orphan-worker",lease=root/"gpu-lease";
    children.owner=fork();require(children.owner>=0,"fork runner owner");
    if(children.owner==0) {
        ProcessSpec spec;spec.argv={self.string(),"--child-hold",marker.string()};spec.log_path=root/"orphan.log";spec.resource_lock=lease;
        const auto result=PosixProcessRunner{}.run(spec);_exit(result.success()?0:91);
    }
    const auto until=std::chrono::steady_clock::now()+2s;
    while(std::chrono::steady_clock::now()<until){std::ifstream in(marker);in>>children.worker;if(children.worker>0)break;std::this_thread::sleep_for(1ms);}
    require(children.worker>0,"worker did not start");
    require(kill(children.owner,SIGKILL)==0,"kill isolated runner owner");
    int status;require(waitpid(children.owner,&status,0)==children.owner,"reap isolated owner");children.owner=-1;
    ProcessSpec competing;competing.argv={self.string(),"--child-ok",(root/"competing").string()};competing.log_path=root/"competing.log";competing.resource_lock=lease;
    const auto denied=PosixProcessRunner{}.run(competing);
    require(denied.termination==ProcessTermination::kLaunchFailed&&denied.system_error==EBUSY&&!fs::exists(root/"competing"),
            "runner death released GPU ownership while its worker was alive");
    require(kill(children.worker,SIGTERM)==0,"stop owned fixture worker");
    require(waitpid(children.worker,&status,0)==children.worker,"reap adopted worker");children.worker=-1;
    require(PosixProcessRunner{}.run(competing).success(),"resource was not released after worker exit");
}
}

int main(int argc, char** argv) {
    if (argc == 3 && (std::string(argv[1]) == "--child-ok" ||
                      std::string(argv[1]) == "--child-signal" || std::string(argv[1]) == "--child-hold"))
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
            ProcessSpec observed;
            observed.argv={self.string(),"--child-ok",marker.string()};observed.log_path=fixture_path/"observed.log";
            observed.log_filter_terms={"not-in-this-line"};int count=0;
            observed.observe_line=[&](std::string_view line){require(line=="--child-ok","observer line framing");++count;};
            require(runner.run(observed).success()&&count==1,"observer must see filtered child lines");
            observed.log_filter_terms.clear();
            observed.observe_line=[](std::string_view){throw std::runtime_error("observer storage failure");};
            auto failed=runner.run(observed);
            require(!failed.success()&&failed.termination==ProcessTermination::kExited&&failed.exit_code==0&&failed.io_error!=0,
                    "observer failure must be separate from successful child exit");
            require(read(observed.log_path).find("--child-ok")!=std::string::npos,"observer exception lost child log");
            require_no_children();require(descriptor_count()==baseline_fds,"observer exception leaked resources");
            inherited_resource(self,fixture_path);
            require_no_children();require(descriptor_count()==baseline_fds,"resource inheritance leaked children or descriptors");
            std::uintmax_t bytes = 0;
            for (const auto& entry : fs::directory_iterator(fixture_path)) {
                require(entry.is_regular_file(), "unexpected fixture artifact");
                bytes += entry.file_size();
            }
            require(bytes <= 8192, "fixture/log output exceeded 8KiB");
            std::cout << "Temporary output=" << bytes << " bytes; max worker children=1 plus isolated runner owner\n";
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
