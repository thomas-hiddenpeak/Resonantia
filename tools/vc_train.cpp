/**
 * @file vc_train.cpp
 * @brief Thin process entry point for decoder and GAN fine-tuning.
 */

#include <iostream>
#include <utility>

#include "tools/app_support/vc_train_app.h"

int main(int argc, char** argv) {
    voxmutatio::app_support::VcTrainOptions options;
    const auto status = voxmutatio::app_support::parse_vc_train_options(
        argc, argv, options, std::cout, std::cerr);
    if (status != voxmutatio::app_support::ParseStatus::kRun) {
        return voxmutatio::app_support::parse_exit_code(status);
    }
    return voxmutatio::app_support::run_vc_train(std::move(options));
}
