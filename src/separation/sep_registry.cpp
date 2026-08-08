// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/separation/sep_registry.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace fs = std::filesystem;

namespace voxmutatio::separation {

namespace {

struct Builtin { const char* task; const char* name; const char* label; };

// Canonical SOTA MelBand-RoFormer per stage (always offered when installed).
const Builtin kBuiltins[] = {
    {"separate",  "vocal_roformer",    "人声分离 · MelBand-RoFormer"},
    {"deharmony", "karaoke_roformer",  "去和声 · Karaoke RoFormer"},
    {"dereverb",  "dereverb_roformer", "去混响 · MelBand-RoFormer"},
    {"deecho",    "deecho_roformer",   "去回声 · MelBand-RoFormer"},
    {"denoise",   "denoise_roformer",  "去噪 · MelBand-RoFormer"},
};

// Minimal JSON string-field extractor: returns value of "key":"value" or "".
std::string json_string_field(const std::string& text, const std::string& key) {
    std::string pat = "\"" + key + "\"";
    size_t k = text.find(pat);
    if (k == std::string::npos) return "";
    size_t c = text.find(':', k + pat.size());
    if (c == std::string::npos) return "";
    size_t q1 = text.find('"', c + 1);
    if (q1 == std::string::npos) return "";
    size_t q2 = text.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return text.substr(q1 + 1, q2 - q1 - 1);
}

bool weights_exist(const std::string& dir, const std::string& name) {
    return fs::exists(fs::path(dir) / (name + ".safetensors"));
}

}  // namespace

const std::vector<std::string>& sep_tasks() {
    static const std::vector<std::string> tasks = {
        "separate", "deharmony", "dereverb", "deecho", "denoise"};
    return tasks;
}

std::vector<SepModel> list_sep_models(const std::string& dir) {
    std::vector<SepModel> out;
    std::unordered_set<std::string> seen;  // task|name

    // 1. Built-in defaults whose weights are installed.
    for (const auto& b : kBuiltins) {
        if (weights_exist(dir, b.name)) {
            out.push_back({b.name, b.label, b.task});
            seen.insert(std::string(b.task) + "|" + b.name);
        }
    }

    // 2. Drop-in models: any <name>.json tagged with a known "task".
    std::error_code ec;
    if (fs::is_directory(dir, ec)) {
        std::vector<fs::path> jsons;
        for (const auto& e : fs::directory_iterator(dir, ec))
            if (e.path().extension() == ".json") jsons.push_back(e.path());
        std::sort(jsons.begin(), jsons.end());
        for (const auto& jp : jsons) {
            std::ifstream f(jp);
            if (!f) continue;
            std::stringstream ss;
            ss << f.rdbuf();
            std::string text = ss.str();
            std::string task = json_string_field(text, "task");
            if (task.empty()) continue;
            const auto& tasks = sep_tasks();
            if (std::find(tasks.begin(), tasks.end(), task) == tasks.end()) continue;
            std::string name = jp.stem().string();
            if (!weights_exist(dir, name)) continue;
            std::string key = task + "|" + name;
            if (seen.count(key)) continue;
            std::string label = json_string_field(text, "label");
            out.push_back({name, label.empty() ? name : label, task});
            seen.insert(key);
        }
    }
    return out;
}

std::string default_sep_model(const std::string& dir, const std::string& task) {
    for (const auto& b : kBuiltins)
        if (task == b.task && weights_exist(dir, b.name)) return b.name;
    for (const auto& m : list_sep_models(dir))
        if (m.task == task) return m.name;
    return "";
}

bool sep_model_available(const std::string& dir, const std::string& task,
                         const std::string& name) {
    for (const auto& m : list_sep_models(dir))
        if (m.task == task && m.name == name) return true;
    return false;
}

}  // namespace voxmutatio::separation
