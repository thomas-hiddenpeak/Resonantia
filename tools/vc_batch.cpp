/**
 * @file vc_batch.cpp
 * @brief Thin process entry point for batch voice conversion.
 */

#include <iostream>

#include "tools/app_support/vc_batch_app.h"

int main(int argc, char** argv) {
    voxmutatio::app_support::VcBatchOptions options;
    const auto status = voxmutatio::app_support::parse_vc_batch_options(
        argc, argv, options, std::cout, std::cerr);
    if (status != voxmutatio::app_support::ParseStatus::kRun) {
        return voxmutatio::app_support::parse_exit_code(status);
    }
    return voxmutatio::app_support::run_vc_batch(options);
}
