/**
 * @file vc_preprocess.cpp
 * @brief Thin process entry point for training-audio preprocessing.
 */

#include <iostream>

#include "tools/app_support/vc_preprocess_app.h"

int main(int argc, char** argv) {
    voxmutatio::app_support::VcPreprocessOptions options;
    const auto status = voxmutatio::app_support::parse_vc_preprocess_options(
        argc, argv, options, std::cout, std::cerr);
    if (status != voxmutatio::app_support::ParseStatus::kRun) {
        return voxmutatio::app_support::parse_exit_code(status);
    }
    return voxmutatio::app_support::run_vc_preprocess(options);
}
