// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
#include "tools/app_support/joint_run_state.h"
#include "src/training/joint_checkpoint.h"
#include "voxmutatio/training/joint_trainer.h"

namespace voxmutatio::app_support::joint_full {
constexpr std::size_t kFeatureCacheMaxBytes=8*1024*1024;
inline void validate_cached_clip(const training::JointTrainingClip& c) {
  if(c.frames<32||c.frames>600||c.phone.size()!=std::size_t(c.frames*768)||c.f0.size()!=std::size_t(c.frames)||
      c.pitch.size()!=std::size_t(c.frames)||c.audio.size()!=std::size_t(c.frames*400)||c.spectrum.size()!=std::size_t(c.frames*1025))
    throw std::invalid_argument("invalid frozen feature shape");
  for(const auto* v:{&c.phone,&c.f0,&c.audio,&c.spectrum})for(float x:*v)
    if(!std::isfinite(x))throw std::invalid_argument("nonfinite frozen feature");
  for(float x:c.f0)if(x<0||x>20000)throw std::invalid_argument("invalid cached F0");
  for(float x:c.spectrum)if(x<0)throw std::invalid_argument("negative cached spectrum");
  for(int p:c.pitch)if(p<1||p>255)throw std::invalid_argument("invalid cached coarse pitch");
}
inline std::string clip_digest(const training::JointTrainingClip& c) {
  static_assert(sizeof(int)==4);Sha256 h;h.update(&c.frames,sizeof(c.frames));
  for(const auto* v:{&c.phone,&c.f0,&c.audio,&c.spectrum})h.update(v->data(),v->size()*sizeof(float));
  h.update(c.pitch.data(),c.pitch.size()*sizeof(int));return h.finish();
}
inline void save_feature_cache(const std::filesystem::path& path,const std::string& key,const training::JointTrainingClip& clip) {
  using namespace training::detail;
  if(!valid_digest(key))throw std::invalid_argument("invalid frozen feature key");validate_cached_clip(clip);
  std::vector<CheckpointTensor> tensors{
    {"phone",{clip.frames,768},[&]{return clip.phone;}},
    {"f0",{clip.frames},[&]{return clip.f0;}},
    {"audio",{clip.frames*400},[&]{return clip.audio;}},
    {"spectrum",{1025,clip.frames},[&]{return clip.spectrum;}},
    {"pitch",{clip.frames},[&]{return std::vector<float>(clip.pitch.begin(),clip.pitch.end());}}};
  write_joint_checkpoint(path,tensors,"RESONANTIA_JOINT_FEATURES 1\n"+key+"\n"+clip_digest(clip)+"\n");
}
inline training::JointTrainingClip load_feature_cache(const std::filesystem::path& path,const std::string& expected_key) {
  using namespace training::detail;
  if(!valid_digest(expected_key))throw std::invalid_argument("invalid frozen feature key");
  io::SafetensorsLoader loader;
  if(!loader.load_snapshot(path,kFeatureCacheMaxBytes))throw std::runtime_error("cannot read frozen feature cache: "+path.string());
  auto names=loader.tensor_names();std::sort(names.begin(),names.end());
  if(names!=std::vector<std::string>{"audio","f0","phone","pitch","spectrum","state"})
    throw std::invalid_argument("unknown frozen cache tensor manifest");
  const auto* state=loader.get_tensor("state");
  if(!state||state->shape.size()!=1||state->shape[0]<1||state->shape[0]>256)
    throw std::invalid_argument("invalid frozen cache metadata");
  const auto bytes=checkpoint_tensor(loader,"state",{int(state->shape[0])});std::string text;
  for(float b:bytes){if(b<0||b>255||b!=std::trunc(b))throw std::invalid_argument("invalid cache text byte");text.push_back(char(b));}
  std::istringstream in(text);std::string magic,key,digest;int version;
  if(!(in>>magic>>version>>key>>digest)||magic!="RESONANTIA_JOINT_FEATURES"||version!=1||
      key!=expected_key||!valid_digest(digest))throw std::invalid_argument("frozen feature recipe mismatch");
  in>>std::ws;if(!in.eof())throw std::invalid_argument("trailing cache metadata");
  const auto* phone=loader.get_tensor("phone");
  if(!phone||phone->shape.size()!=2||phone->shape[0]<32||phone->shape[0]>600)
    throw std::invalid_argument("invalid cache frame count");
  training::JointTrainingClip c;c.frames=phone->shape[0];
  c.phone=checkpoint_tensor(loader,"phone",{c.frames,768});c.f0=checkpoint_tensor(loader,"f0",{c.frames});
  c.audio=checkpoint_tensor(loader,"audio",{c.frames*400});c.spectrum=checkpoint_tensor(loader,"spectrum",{1025,c.frames});
  const auto pitch=checkpoint_tensor(loader,"pitch",{c.frames});c.pitch.reserve(c.frames);
  for(float p:pitch){if(p<1||p>255||p!=std::trunc(p))throw std::invalid_argument("invalid cached coarse pitch");c.pitch.push_back(int(p));}
  validate_cached_clip(c);if(clip_digest(c)!=digest)throw std::invalid_argument("frozen feature payload checksum mismatch");return c;
}
} // namespace voxmutatio::app_support::joint_full
