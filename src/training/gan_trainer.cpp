// Copyright 2024 Resonantia Authors.
// Licensed under the MIT License.

#include "voxmutatio/training/gan_trainer.h"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace voxmutatio::training {

namespace ag = voxmutatio::autograd;

namespace {

// Lightweight phase profiler gated by the VOX_PROFILE env var.
struct Prof {
  bool on;
  std::chrono::high_resolution_clock::time_point t0;
  Prof() : on(std::getenv("VOX_PROFILE") != nullptr) { reset(); }
  void reset() {
    if (on) { cudaDeviceSynchronize(); t0 = std::chrono::high_resolution_clock::now(); }
  }
  void mark(const char* tag) {
    if (!on) return;
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "  [prof] %-16s %8.1f ms\n", tag, ms);
    t0 = t1;
  }
};
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

// VITS KL divergence (mean over elements; drops the constant -0.5 offset):
//   kl = logs_p - logs_q - 0.5 + 0.5*(z_p - m_p)^2 * exp(-2 logs_p)
ag::Tensor kl_loss(const ag::Tensor& z_p, const ag::Tensor& logs_q,
                   const ag::Tensor& m_p, const ag::Tensor& logs_p) {
  auto diff = ag::add(z_p, ag::scale(m_p, -1.0f));
  auto sq = ag::mul(diff, diff);
  auto inv = ag::exp_op(ag::scale(logs_p, -2.0f));
  auto term = ag::mul(sq, inv);
  auto klt = ag::add(ag::add(logs_p, ag::scale(logs_q, -1.0f)), ag::scale(term, 0.5f));
  return ag::scale(ag::sum(klt), 1.0f / static_cast<float>(z_p.numel()));
}

}  // namespace

bool GANTrainer::init(const std::string& g_path, const std::string& d_path,
                      int speaker_id, const MelSpecConfig& mel_cfg,
                      float g_lr, float d_lr) {
  if (!gen_.init(g_path, speaker_id)) return false;
  if (!enc_q_.init(g_path, speaker_id)) return false;
  if (!flow_.init(g_path, speaker_id)) return false;
  if (!disc_.init(d_path)) return false;
  mel_ = std::make_unique<MelLoss>(mel_cfg);
  n_mels_ = mel_cfg.n_mels;
  n_spec_ = mel_cfg.n_fft;  // placeholder; overwritten per full-step spec channels
  n_spec_ = 1025;

  // Generator optimizer spans decoder + posterior encoder + flow.
  std::vector<ag::Tensor> gp = gen_.params();
  for (auto& t : enc_q_.params()) gp.push_back(t);
  for (auto& t : flow_.params()) gp.push_back(t);
  g_opt_ = std::make_unique<ag::AdamW>(gp, g_lr, 0.8f, 0.99f);
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

GANTrainer::Losses GANTrainer::train_step_full(
    const std::vector<float>& spec, const std::vector<float>& har,
    const std::vector<float>& target, const std::vector<float>& m_p,
    const std::vector<float>& logs_p, int T, int L) {
  auto spec_t = ag::Tensor::from_host(spec, {n_spec_, T}, false);
  auto hc = ag::Tensor::from_host(har, {1, L}, false);
  auto y_real = ag::Tensor::from_host(target, {1, L}, false);
  auto mp = ag::Tensor::from_host(m_p, {192, T}, false);
  auto lsp = ag::Tensor::from_host(logs_p, {192, T}, false);

  int Tm = 0;
  auto tgt_host = mel_->target_log_mel(target.data(), L, Tm);
  auto mel_tgt = ag::Tensor::from_host(tgt_host, {Tm, n_mels_}, false);

  Losses out{};
  const int n_sub = 9;

  Prof prof;
  // Single posterior sample + decode, shared by D and G steps (matches
  // reference VITS: one enc_q sample, one generator forward per step).
  ag::Tensor mq, lq;
  auto zq = enc_q_.forward(spec_t, T, true, mq, lq);
  auto zp = flow_.forward(zq, T);
  auto y_hat = gen_.decode(zq, hc, T);
  auto y_det = ag::Tensor::from_host(y_hat.to_host(), {1, L}, false);
  prof.mark("fwd(encq+flow+dec)");

  // ---- Discriminator step (fake detached) ----
  {
    auto dr = disc_.forward(y_real, L);
    auto df = disc_.forward(y_det, L);
    auto ld = disc_loss(dr, df);
    prof.mark("D.fwd(disc x2)");
    ag::backward(ld);
    d_opt_->step();
    prof.mark("D.backward+step");
    out.d = ld.to_host()[0] + static_cast<float>(n_sub);
  }

  // ---- Generator step (mel + kl + feature-matching + adversarial) ----
  {
    auto gm = mel_->log_mel(y_hat, L);
    auto lmel = mel_->l1(gm, mel_tgt, Tm);
    auto lkl = kl_loss(zp, lq, mp, lsp);
    auto df = disc_.forward(y_hat, L);
    auto dr = disc_.forward(y_real, L);
    auto ladv = gen_adv_loss(df);
    auto lfm = fm_loss(dr, df);
    auto lg = ag::add(ag::add(ag::add(ag::scale(lmel, 45.0f), ag::scale(lfm, 2.0f)), ladv), lkl);
    prof.mark("G.fwd(mel+disc x2)");
    ag::backward(lg);
    g_opt_->step();
    prof.mark("G.backward+step");
    out.mel = lmel.to_host()[0];
    out.fm = lfm.to_host()[0];
    out.adv = ladv.to_host()[0] + static_cast<float>(n_sub);
    out.kl = lkl.to_host()[0];
    out.g = lg.to_host()[0] + static_cast<float>(n_sub);
  }

  return out;
}

}  // namespace voxmutatio::training
