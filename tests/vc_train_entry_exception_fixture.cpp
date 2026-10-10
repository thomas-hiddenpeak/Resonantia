// Host-only replacement app functions linked with the unchanged production CLI
// main. No CUDA, models, syscall interposition, or production test hook.
#include "tools/app_support/vc_train_app.h"
#include "tools/app_support/training_cancel.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <streambuf>

namespace {
std::filesystem::path marker;
struct UnwindMarker {
    ~UnwindMarker() { std::ofstream(marker, std::ios::app) << "unwound\n"; }
};
class ThrowingBuffer : public std::streambuf {
    int_type overflow(int_type) override { throw std::runtime_error("diagnostic sink failed"); }
};
ThrowingBuffer throwing_buffer;
struct RestoreDiagnostics {
    std::streambuf* previous = nullptr;
    ~RestoreDiagnostics() {
        if (previous) {
            std::cerr.exceptions(std::ios::goodbit);
            std::cerr.clear();
            std::cerr.rdbuf(previous);
        }
    }
} diagnostics;
}
namespace voxmutatio::app_support {
ParseStatus parse_vc_train_options(int argc, char* const argv[], VcTrainOptions& options,
                                  std::ostream&, std::ostream&) {
    // Before the fix, uncaught exceptions use this test-only exit sentinel.
    // This observes a missing catch without generating SIGABRT or a core dump.
    std::set_terminate([] { std::_Exit(86); });
    if (argc != 3) return ParseStatus::kExitFailure;
    options.target_path = argv[1];
    marker = argv[2];
    if (options.target_path == "parse-exception") {
        UnwindMarker cleanup;
        throw std::runtime_error("parse fixture exception");
    }
    if (options.target_path == "diagnostic-exception" || options.target_path == "unknown-diagnostic") {
        diagnostics.previous = std::cerr.rdbuf(&throwing_buffer);
        std::cerr.exceptions(std::ios::badbit | std::ios::failbit);
    }
    return ParseStatus::kRun;
}
int run_vc_train(VcTrainOptions options) {
    UnwindMarker cleanup;
    if (options.target_path == "success") return 0;
    if (options.target_path == "cancelled") return 2;
    if (options.target_path == "unknown" || options.target_path == "unknown-diagnostic") throw 7;
    if (options.target_path == "cancel-and-throw") {
        using namespace resonantia::training_control;
        const auto path = marker.string() + ".token";
        TrainingCancelSource source(path);
        TrainingCancelReader reader(path);
        return static_cast<int>(run_training_iterations(2, reader, [&](int) -> bool {
            source.request();
            throw std::runtime_error("step failed after cancellation request");
        }));
    }
    throw std::runtime_error("training fixture exception");
}
}
