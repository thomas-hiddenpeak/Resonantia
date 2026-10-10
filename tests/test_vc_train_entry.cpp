// Host contract for the actual vc_train main linked with throwing app stubs.
#include "tools/web/workflow_support.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace resonantia::web;
namespace {
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
std::string read(const fs::path& p) {
    std::ifstream in(p); return {std::istreambuf_iterator<char>(in), {}};
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected entry fixture executable");
        setenv("CUDA_VISIBLE_DEVICES", "", 1);
        TempDir root(fs::temp_directory_path(), "resonantia-train-entry-");
        const fs::path dir = root.path();
        const std::vector<std::pair<std::string, int>> cases{
            {"success", 0}, {"cancelled", 2}, {"standard", 1}, {"unknown", 1},
            {"parse-exception", 1}, {"cancel-and-throw", 1},
            {"diagnostic-exception", 1}, {"unknown-diagnostic", 1}};
        for (const auto& [name, expected] : cases) {
            const auto marker = dir / (name + ".marker");
            ProcessSpec spec; spec.argv = {argv[1], name, marker.string()};
            spec.log_path = dir / (name + ".log");
            const auto result = PosixProcessRunner{}.run(spec);
            std::cout << name << ": exit=" << result.exit_code << " expected=" << expected << '\n';
            require(result.termination == ProcessTermination::kExited && result.exit_code == expected &&
                        !result.io_error && !result.system_error, name + ": wrong process outcome");
            require(read(marker) == "unwound\n", name + ": ordinary stack did not unwind exactly once");
            if (name == "standard" || name == "parse-exception" || name == "cancel-and-throw")
                require(read(spec.log_path).find("error:") != std::string::npos &&
                            read(spec.log_path).find("exception") != std::string::npos,
                        name + ": missing original exception diagnostic");
            if (name == "unknown") require(read(spec.log_path).find("unknown exception") != std::string::npos,
                                           "unknown exception missing diagnostic");
        }
        std::cout << "CLI entry: 8/8 host cases; standard/unknown/parse/body/diagnostic errors return 1, normal 0 and cancel 2 preserved; no signals/GPU\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Entry contract: " << error.what() << '\n'; return 1;
    }
}
