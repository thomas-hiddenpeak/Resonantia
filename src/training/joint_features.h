// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
#include "voxmutatio/separation/stft.h"
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace voxmutatio::training::detail {
// One aligned mono 40 kHz clip; caller owns rate validation. Returns [1025,T].
inline std::vector<float> joint_spectrum(std::span<const float> audio) {
  if(audio.size()<32*400 || audio.size()>600*400 || audio.size()%400)
    throw std::invalid_argument("joint spectrum requires 32..600 complete frames");
  for(float x:audio)if(!std::isfinite(x))throw std::invalid_argument("non-finite joint spectrum input");
  separation::Stft stft(2048,400,2048,true,824);
  std::vector<float> re,im;int frames=0;
  stft.forward(audio.data(),static_cast<int>(audio.size()),re,im,frames);
  if(frames!=static_cast<int>(audio.size()/400) || re.size()!=static_cast<std::size_t>(1025*frames) || im.size()!=re.size())
    throw std::runtime_error("joint spectrum shape mismatch");
  for(std::size_t j=0;j<re.size();++j) {
    volatile float real_power=re[j]*re[j],imag_power=im[j]*im[j];
    re[j]=std::sqrt((real_power+imag_power)+2e-7f);
    if(!std::isfinite(re[j]))throw std::runtime_error("non-finite joint spectrum");
  }
  return re;
}
} // namespace voxmutatio::training::detail
