/**
 * @file vc_train.cpp
 * @brief Thin process entry point for decoder and GAN fine-tuning.
 */

#include <exception>
#include <iostream>
#include <utility>

#include "tools/app_support/vc_train_app.h"

namespace {
void report_exception(const char* message) noexcept {
    // Reporting must not turn an already determined failure into another
    // uncaught exception (for example, if the stream/allocation fails).
    try {
        std::cerr << "error: vc_train exception: " << message << '\n';
    } catch (...) {
        // The caller still returns failure even when no diagnostic is writable.
    }
}
}

int main(int argc, char** argv) {
    try {
        voxmutatio::app_support::VcTrainOptions options;
        const auto status = voxmutatio::app_support::parse_vc_train_options(
            argc, argv, options, std::cout, std::cerr);
        if (status != voxmutatio::app_support::ParseStatus::kRun) {
            return voxmutatio::app_support::parse_exit_code(status);
        }
        return voxmutatio::app_support::run_vc_train(std::move(options));
    } catch (const std::exception& error) {
        report_exception(error.what());
        return 1;
    } catch (...) {
        report_exception("unknown exception");
        return 1;
    }
}
