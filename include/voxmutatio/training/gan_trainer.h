// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.
//
// GAN fine-tuning: the trainable NSF-HiFiGAN decoder (generator) trained
// against the MPD-V2 discriminator with LSGAN adversarial + feature-matching
// + mel losses. Alternating D/G updates on the autograd engine (spec 002).

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "voxmutatio/autograd/tensor.h"
#include "voxmutatio/training/discriminator.h"
#include "voxmutatio/training/generator_trainer.h"
#include "voxmutatio/training/mel_loss.h"

namespace voxmutatio::training {

class GANTrainer {
 public:
  bool init(const std::string& g_model_path, const std::string& d_model_path,
            int speaker_id, const MelSpecConfig& mel_cfg,
            float g_lr = 1e-4f, float d_lr = 1e-4f);

  struct Losses { float d, g, mel, fm, adv; };

  /// One alternating D-then-G update on a segment.
  /// z[192,T], har[1,L], target[1,L] (real 40k), L = T*upp.
  Losses train_step(const std::vector<float>& z, const std::vector<float>& har,
                    const std::vector<float>& target, int T, int L);

  bool export_model(const std::string& src, const std::string& out) {
    return gen_.export_model(src, out);
  }
  GeneratorTrainer& generator() { return gen_; }

 private:
  GeneratorTrainer gen_;
  Discriminator disc_;
  std::unique_ptr<MelLoss> mel_;
  std::unique_ptr<autograd::AdamW> g_opt_, d_opt_;
  int n_mels_ = 80;
};

}  // namespace voxmutatio::training
