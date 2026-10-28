// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
#include "tools/app_support/vc_train_app.h"
#include "tools/app_support/training_cancel.h"
namespace voxmutatio::app_support {
int run_full_joint_training(const VcTrainOptions&,
    resonantia::training_control::TrainingCancelReader&);
}
