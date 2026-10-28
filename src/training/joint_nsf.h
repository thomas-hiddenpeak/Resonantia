// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace voxmutatio::training::detail {
// Fixed 40 kHz / 400-sample hop source used by the bounded joint decoder.
inline std::vector<float> joint_sine_source(std::span<const float> f0,
                                           std::span<const float> noise) {
  if(f0.empty() || f0.size()>600 || noise.size()!=f0.size()*400)
    throw std::invalid_argument("joint NSF dimensions exceed bounded contract");
  for(float v:f0)if(!std::isfinite(v) || v<0 || v>20000)
    throw std::invalid_argument("joint NSF F0 out of range");
  for(float v:noise)if(!std::isfinite(v))throw std::invalid_argument("non-finite joint NSF noise");
  std::vector<float> sine(noise.size());double accumulated=0;float previous=0;
  for(std::size_t t=0;t<f0.size();++t) {
    const float rad=f0[t]/40000.f;
    for(int j=0;j<400;++j) {
      // These are separate upstream tensor operations. Materialize FP32
      // products so host compiler FMA contraction cannot move the phase or
      // combine periodic excitation with its independent noise contribution.
      volatile float within=rad*(j+1);
      const float phase=within+previous;
      const float angle=(2*std::numbers::pi_v<float>)*phase;
      volatile float tone=f0[t]>0?.1f*std::sin(angle):0.f;
      volatile float random=(f0[t]>0?.003f:.1f/3)*noise[t*400+j];
      sine[t*400+j]=tone+random;
    }
    volatile float end=rad*400;
    accumulated+=std::fmod(end+.5f,1.f)-.5f;
    previous=std::fmod(static_cast<float>(accumulated),1.f);
  }
  return sine;
}
} // namespace voxmutatio::training::detail
