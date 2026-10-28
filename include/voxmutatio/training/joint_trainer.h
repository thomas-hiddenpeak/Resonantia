// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace voxmutatio::training {
// One synchronous, borrowed host clip. T is bounded to 600, batch=1, v2/40k.
struct JointTrainingClip {
  int frames = 0;
  std::vector<float> phone;       // [T,768], nearest 2x content features
  std::vector<int> pitch;         // [T], 1..255
  std::vector<float> f0;          // [T], Hz; 0 denotes unvoiced
  std::vector<float> spectrum;    // [1025,T], sqrt(STFT power + 2e-7)
  std::vector<float> audio;       // [400*T], mono 40k
};
struct JointStepOptions {
  int crop_start = 0;
  int crop_frames = 32;
  bool update = true;
  // Optional fixed standard-Gaussian inputs for independent numerical tests.
  std::vector<float> posterior_noise;  // [192,T]; empty => trainer RNG
  std::vector<float> nsf_noise;        // [crop_frames*400]; empty => trainer RNG
};
struct JointStepResult {
  float discriminator = 0, generator = 0, mel = 0, kl = 0, feature = 0, adversarial = 0;
};
struct JointStepTrace {
  std::vector<float> prior_mean, prior_logs, posterior_mean, posterior_logs;
  std::vector<float> posterior_z, flow_z, audio;
  std::vector<float> audio_gradient;
  std::vector<std::vector<float>> discriminator_real, discriminator_fake;
};
struct JointParameterSnapshot {
  std::vector<int> shape;
  std::vector<float> value, gradient, first_moment, second_moment;
};
// Experimental aligned joint training. Private implementation owns all tensors,
// optimizer/RNG state; instances are not thread-safe. No input retained after step.
class JointTrainer {
 public:
  JointTrainer();
  ~JointTrainer();
  JointTrainer(JointTrainer&&) noexcept;
  JointTrainer& operator=(JointTrainer&&) noexcept;
  bool init(const std::string& generator, const std::string& discriminator,
            int speaker = 0, float learning_rate = 1e-4f, std::uint32_t seed = 0);
  // Throws invalid_argument for bad clips/options, logic_error if uninitialized;
  // runtime failures throw and invalidate the trainer before any further step/export.
  JointStepResult step(const JointTrainingClip& clip, const JointStepOptions& options,
                       JointStepTrace* trace = nullptr);
  bool export_model(const std::string& output) const;
  // Versioned complete training state, distinct from an inference model. Saves
  // all G/D parameters, AdamW state, RNG and inference-preserved originals.
  // run_state (<=65536 bytes) is caller-owned epoch/data/shuffle metadata.
  // Same-directory staging/rename preserves an old file on write failure.
  bool save_checkpoint(const std::string& output, const std::string& run_state) const;
  // Self-contained transactional restore. Failure preserves the old trainer and
  // run_state; success replaces both. Unknown versions/invalid state fail closed.
  bool load_checkpoint(const std::string& input, std::string& run_state);
  void set_learning_rate(float learning_rate);
  [[nodiscard]] float learning_rate() const;
  [[nodiscard]] int step_count() const;
  // Diagnostic snapshots; persistent resume uses the checkpoint API above.
  // Gradient is from the last backward. Request a step trace to retain D's
  // optimizer gradient separately from the subsequent G backward through D.
  std::vector<std::string> parameter_names(bool discriminator = false) const;
  JointParameterSnapshot inspect_parameter(const std::string& name, bool discriminator = false) const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace voxmutatio::training
