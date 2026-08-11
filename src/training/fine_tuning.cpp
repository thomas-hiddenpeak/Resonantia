// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/training/fine_tuning.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>

namespace voxmutatio::training {

namespace {

std::string trim_copy(std::string value) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    const auto first = std::find_if(value.begin(), value.end(), not_space);
    const auto last = std::find_if(value.rbegin(), value.rend(), not_space).base();
    if (first >= last) return {};
    return std::string(first, last);
}

}  // namespace

bool FineTuneTrainer::init(const FineTuneConfig& config) {
    config_ = config;
    current_metrics_ = {};
    initialized_ = false;
    std::cerr << "FineTuneTrainer is not implemented; use vc_train's decoder or GAN trainer\n";
    return false;
}

bool FineTuneTrainer::load_dataset(const std::string& list_path) {
    dataset_.clear();
    std::ifstream file(list_path);
    if (!file.is_open()) {
        return false;
    }
    
    std::vector<TrainingEntry> parsed;
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        line = trim_copy(std::move(line));
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Parse line: audio_path | speaker_id | label
        std::istringstream iss(line);
        std::string audio_path, speaker_id_str, label;
        
        if (std::getline(iss, audio_path, '|') &&
            std::getline(iss, speaker_id_str, '|')) {
            std::getline(iss, label);
            
            TrainingEntry entry;
            entry.audio_path = trim_copy(std::move(audio_path));
            speaker_id_str = trim_copy(std::move(speaker_id_str));
            label = trim_copy(std::move(label));
            int speaker_id = 0;
            const auto [end, error] = std::from_chars(
                speaker_id_str.data(), speaker_id_str.data() + speaker_id_str.size(), speaker_id);
            if (entry.audio_path.empty() || error != std::errc{} ||
                end != speaker_id_str.data() + speaker_id_str.size() || speaker_id < 0) {
                return false;
            }
            entry.speaker_id = speaker_id;
            entry.label = label;
            
            parsed.push_back(std::move(entry));
        } else {
            return false;
        }
    }
    if (file.bad()) return false;
    if (parsed.empty()) return false;
    dataset_ = std::move(parsed);
    return true;
}

bool FineTuneTrainer::train() {
    std::cerr << "FineTuneTrainer::train is unsupported; no optimization was performed\n";
    return false;
}

bool FineTuneTrainer::save_checkpoint(const std::string& path,
                                       const std::string& name) {
    (void)path;
    (void)name;
    std::cerr << "FineTuneTrainer::save_checkpoint is unsupported\n";
    return false;
}

bool FineTuneTrainer::load_checkpoint(const std::string& path) {
    (void)path;
    std::cerr << "FineTuneTrainer::load_checkpoint is unsupported\n";
    return false;
}

}  // namespace voxmutatio::training
