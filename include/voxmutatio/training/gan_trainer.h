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
#include "voxmutatio/training/posterior_encoder.h"

namespace voxmutatio::training {

class GANTrainer {
 public:
  bool init(const std::string& g_model_path, const std::string& d_model_path,
            int speaker_id, const MelSpecConfig& mel_cfg,
            float g_lr = 1e-4f, float d_lr = 1e-4f);

  struct Losses { float d, g, mel, fm, adv, kl; };

  /// Decoder-only GAN (frozen posterior z): z[192,T], har[1,L], target[1,L].
  Losses train_step(const std::vector<float>& z, const std::vector<float>& har,
                    const std::vector<float>& target, int T, int L);

  /// Full GAN: enc_q(spec) -> z_q -> {flow -> z_p (KL), dec -> y_hat}.
  /// spec[n_spec*T], har[1,L], target[1,L], m_p/logs_p[192*T] (prior consts).
  Losses train_step_full(const std::vector<float>& spec,
                         const std::vector<float>& har,
                         const std::vector<float>& target,
                         const std::vector<float>& m_p,
                         const std::vector<float>& logs_p, int T, int L);

  bool export_model(const std::string& src, const std::string& out) {
    return gen_.export_model(src, out);
  }
  GeneratorTrainer& generator() { return gen_; }

  /// RVC ExponentialLR: multiply both optimizers' LR by gamma (call per epoch).
  void decay_lr(float gamma) {
    if (g_opt_) g_opt_->set_lr(g_opt_->lr() * gamma);
    if (d_opt_) d_opt_->set_lr(d_opt_->lr() * gamma);
  }
  [[nodiscard]] float g_lr() const { return g_opt_ ? g_opt_->lr() : 0.0f; }

 private:
  GeneratorTrainer gen_;
  PosteriorEncoder enc_q_;
  Flow flow_;
  Discriminator disc_;
  std::unique_ptr<MelLoss> mel_;
  std::unique_ptr<autograd::AdamW> g_opt_, d_opt_;
  int n_mels_ = 80;
  int n_spec_ = 1025;
};

}  // namespace voxmutatio::training
