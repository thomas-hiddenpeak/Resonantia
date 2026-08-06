#pragma once

#include <string>
#include <vector>

#include "voxmutatio/core/types.h"

namespace voxmutatio::training {

/// Fine-tuning configuration for voice conversion models
struct FineTuneConfig {
    // Model paths
    std::string hubert_model_path;       // Pre-trained HuBERT weights
    std::string synthesizer_model_path;  // Pre-trained VITS weights
    
    // Training data
    std::string train_list_path;         // Path to training list file
    std::string log_dir;                 // Output directory for logs and checkpoints
    
    // Training parameters
    int epochs = 100;
    int batch_size = 16;
    double learning_rate = 0.0001;
    int save_every_epoch = 10;
    int save_every_steps = 1000;
    
    // Device
    std::string device = "cuda";
    int gpu_device = 0;
    bool use_half_precision = false;
    
    // Model variant
    ModelVersion version = ModelVersion::kV1;
    bool has_f0 = true;
};

/// Training dataset entry
struct TrainingEntry {
    std::string audio_path;
    int speaker_id;
    std::string label;  // Optional metadata
};

/// Training metrics snapshot
struct TrainingMetrics {
    int epoch;
    int step;
    double loss;
    double grad_norm;
    double learning_rate;
    double elapsed_seconds;
};

/// Fine-tuning trainer for voice conversion models
class FineTuneTrainer {
public:
    /// Initialize trainer and load pre-trained models
    bool init(const FineTuneConfig& config);
    
    /// Load training dataset from list file
    bool load_dataset(const std::string& list_path);
    
    /// Run training loop
    bool train();
    
    /// Save checkpoint
    bool save_checkpoint(const std::string& path, 
                        const std::string& name = "checkpoint");
    
    /// Load checkpoint for resume
    bool load_checkpoint(const std::string& path);
    
    /// Get current metrics
    [[nodiscard]] const TrainingMetrics& metrics() const noexcept {
        return current_metrics_;
    }
    
    /// Get training dataset
    [[nodiscard]] const std::vector<TrainingEntry>& dataset() const noexcept {
        return dataset_;
    }
    
    /// Get config
    [[nodiscard]] const FineTuneConfig& config() const noexcept {
        return config_;
    }

private:
    FineTuneConfig config_;
    std::vector<TrainingEntry> dataset_;
    TrainingMetrics current_metrics_;
    
    bool initialized_ = false;
};

}  // namespace voxmutatio::training
