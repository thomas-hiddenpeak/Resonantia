// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/training/gan_trainer.h"

#include <cmath>

namespace voxmutatio::training {

namespace ag = voxmutatio::autograd;

namespace {

// mean(x) and mean(x^2) as scalar autograd tensors.
ag::Tensor mean_(const ag::Tensor& x) {
  return ag::scale(ag::sum(x), 1.0f / static_cast<float>(x.numel()));
}
ag::Tensor mean_sq(const ag::Tensor& x) {
  return ag::scale(ag::sum(ag::mul(x, x)), 1.0f / static_cast<float>(x.numel()));
}
ag::Tensor add_opt(const ag::Tensor& acc, const ag::Tensor& t) {
  return acc.n ? ag::add(acc, t) : t;
}

// LSGAN discriminator loss (grad part; drops the +1 constant per sub-D):
//   sum_i [ mean((real_i-1)^2) + mean(fake_i^2) ]
//   mean((r-1)^2) = mean(r^2) - 2 mean(r) + 1
ag::Tensor disc_loss(const std::vector<SubDiscResult>& real,
                     const std::vector<SubDiscResult>& fake) {
  ag::Tensor loss;
  for (size_t i = 0; i < real.size(); ++i) {
    auto tr = ag::add(mean_sq(real[i].score), ag::scale(mean_(real[i].score), -2.0f));
    auto tf = mean_sq(fake[i].score);
    loss = add_opt(loss, ag::add(tr, tf));
  }
  return loss;
}

// LSGAN generator adversarial loss (grad part): sum_i mean((fake_i-1)^2).
ag::Tensor gen_adv_loss(const std::vector<SubDiscResult>& fake) {
  ag::Tensor loss;
  for (const auto& f : fake) {
    auto t = ag::add(mean_sq(f.score), ag::scale(mean_(f.score), -2.0f));
    loss = add_opt(loss, t);
  }
  return loss;
}

// Feature-matching loss: sum over sub-Ds and layers of mean|real - fake|.
ag::Tensor fm_loss(const std::vector<SubDiscResult>& real,
                   const std::vector<SubDiscResult>& fake) {
  ag::Tensor loss;
  for (size_t i = 0; i < real.size(); ++i) {
    for (size_t j = 0; j < real[i].fmaps.size(); ++j) {
      auto d = ag::add(real[i].fmaps[j], ag::scale(fake[i].fmaps[j], -1.0f));
      auto m = ag::scale(ag::sum(ag::abs_op(d)), 1.0f / static_cast<float>(d.numel()));
      loss = add_opt(loss, m);
    }
  }
  return loss;
}

}  // namespace

bool GANTrainer::init(const std::string& g_path, const std::string& d_path,
                      int speaker_id, const MelSpecConfig& mel_cfg,
                      float g_lr, float d_lr) {
  if (!gen_.init(g_path, speaker_id)) return false;
  if (!disc_.init(d_path)) return false;
  mel_ = std::make_unique<MelLoss>(mel_cfg);
  n_mels_ = mel_cfg.n_mels;
  g_opt_ = std::make_unique<ag::AdamW>(gen_.params(), g_lr, 0.8f, 0.99f);
  d_opt_ = std::make_unique<ag::AdamW>(disc_.params(), d_lr, 0.8f, 0.99f);
  return true;
}

GANTrainer::Losses GANTrainer::train_step(const std::vector<float>& z,
                                          const std::vector<float>& har,
                                          const std::vector<float>& target,
                                          int T, int L) {
  auto zc = ag::Tensor::from_host(z, {192, T}, false);
  auto hc = ag::Tensor::from_host(har, {1, L}, false);
  auto y_real = ag::Tensor::from_host(target, {1, L}, false);

  int Tm = 0;
  auto tgt_host = mel_->target_log_mel(target.data(), L, Tm);
  auto mel_tgt = ag::Tensor::from_host(tgt_host, {Tm, n_mels_}, false);

  Losses out{};
  const int n_sub = 9;

  // ---- Discriminator step (fake detached, no gradient to generator) ----
  {
    auto y_hat = gen_.decode(zc, hc, T);
    auto y_det = ag::Tensor::from_host(y_hat.to_host(), {1, L}, false);
    auto dr = disc_.forward(y_real, L);
    auto df = disc_.forward(y_det, L);
    auto ld = disc_loss(dr, df);
    ag::backward(ld);
    d_opt_->step();
    out.d = ld.to_host()[0] + static_cast<float>(n_sub);  // + dropped constant
  }

  // ---- Generator step (mel + feature-matching + adversarial) ----
  {
    auto y_hat = gen_.decode(zc, hc, T);
    auto gm = mel_->log_mel(y_hat, L);
    auto lmel = mel_->l1(gm, mel_tgt, Tm);
    auto df = disc_.forward(y_hat, L);
    auto dr = disc_.forward(y_real, L);
    auto ladv = gen_adv_loss(df);
    auto lfm = fm_loss(dr, df);
    auto lg = ag::add(ag::add(ag::scale(lmel, 45.0f), ag::scale(lfm, 2.0f)), ladv);
    ag::backward(lg);
    g_opt_->step();
    out.mel = lmel.to_host()[0];
    out.fm = lfm.to_host()[0];
    out.adv = ladv.to_host()[0] + static_cast<float>(n_sub);
    out.g = lg.to_host()[0] + static_cast<float>(n_sub);
  }

  return out;
}

}  // namespace voxmutatio::training
