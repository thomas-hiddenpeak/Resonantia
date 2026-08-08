// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// Separation model registry (spec 004 S7). Enumerates the models available per
// preprocessing stage so the CLI/server/WebUI can offer real, installed choices
// instead of a single hard-coded model. Built-in defaults are always offered
// when their weights are present; additional models are discovered from any
// <name>.json in the separation dir that carries a "task" field (drop-in).

#pragma once

#include <string>
#include <vector>

namespace voxmutatio::separation {

struct SepModel {
    std::string name;   // file stem: <name>.safetensors (+ _config.i64 ...)
    std::string label;  // human-readable label
    std::string task;   // one of sep_tasks()
};

/// Preprocessing stages, in cascade order (keys match vc_preprocess flags).
const std::vector<std::string>& sep_tasks();

/// Models installed under `dir` for every task (built-in defaults whose weights
/// exist + json-tagged drop-in models). Ordered by task, default first.
std::vector<SepModel> list_sep_models(const std::string& dir);

/// Default (first available) model name for `task` in `dir`, or "" if none.
std::string default_sep_model(const std::string& dir, const std::string& task);

/// Whether `name` is an installed model for `task` in `dir`.
bool sep_model_available(const std::string& dir, const std::string& task,
                         const std::string& name);

}  // namespace voxmutatio::separation
