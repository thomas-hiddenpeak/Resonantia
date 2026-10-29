// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
#include "voxmutatio/core/sha256.h"
#include "voxmutatio/io/file_digest.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace voxmutatio::app_support::joint_full {
using core::Sha256;
using core::sha256;
using core::valid_digest;
// Preserve the existing training identity helper's accepted symlink paths.
inline std::string sha256_file(const std::filesystem::path& path,std::uint64_t max_bytes=2ULL*1024*1024*1024) {
  return io::sha256_file(path,max_bytes,true);
}

// One epoch visits every clip exactly once at batch=1; random decoder crops are
// separate from the trainer's posterior/NSF RNG. Serialization is bounded/versioned.
struct RunState {
  std::string identity;int epochs=0,step=0;float initial_lr=0,decay=0;
  std::vector<int> order;std::mt19937 random;
  static RunState begin(std::string id,int clips,int epoch_count,unsigned seed,float lr,float lr_decay) {
    if(clips<1||clips>4096)throw std::invalid_argument("joint-full requires 1..4096 clips");
    RunState s;s.identity=std::move(id);s.epochs=epoch_count;s.initial_lr=lr;s.decay=lr_decay;
    s.order.resize(clips);std::iota(s.order.begin(),s.order.end(),0);s.random.seed(seed);
    std::shuffle(s.order.begin(),s.order.end(),s.random);s.validate();return s;
  }
  int epoch() const {return step/static_cast<int>(order.size());}
  int slot() const {return step%static_cast<int>(order.size());}
  bool done() const {return step==epochs*static_cast<int>(order.size());}
  int current_clip() const {if(done())throw std::logic_error("training already complete");return order.at(slot());}
  int crop(int frames){if(done()||frames<32||frames>600)throw std::invalid_argument("invalid crop request");
    return std::uniform_int_distribution<int>(0,frames-32)(random);}
  void advance(){if(done())throw std::logic_error("training already complete");++step;
    if(slot()==0&&!done())std::shuffle(order.begin(),order.end(),random);}
  float learning_rate() const {float value=initial_lr;for(int i=0;i<epoch();++i)value*=decay;
    if(!std::isfinite(value)||value<=0)throw std::invalid_argument("learning rate underflow");return value;}
  void validate() const {
    if(!valid_digest(identity)||order.empty()||order.size()>4096||epochs<1||epochs>2000||step<0||
        step>epochs*static_cast<int>(order.size())||!std::isfinite(initial_lr)||initial_lr<=0||initial_lr>1||
        !std::isfinite(decay)||decay<=0||decay>1)throw std::invalid_argument("invalid joint-full state");
    auto sorted=order;std::sort(sorted.begin(),sorted.end());
    for(std::size_t i=0;i<sorted.size();++i)if(sorted[i]!=int(i))throw std::invalid_argument("invalid clip permutation");
    (void)learning_rate();
  }
  std::string serialize() const {
    validate();std::ostringstream o;o.imbue(std::locale::classic());o<<std::setprecision(std::numeric_limits<float>::max_digits10);
    o<<"RESONANTIA_JOINT_RUN 1\n"<<identity<<' '<<epochs<<' '<<step<<' '<<initial_lr<<' '<<decay<<' '<<order.size()<<'\n';
    for(int i:order)o<<i<<' ';o<<'\n'<<random<<'\n';auto text=o.str();
    if(text.size()>65536)throw std::invalid_argument("run state exceeds bound");return text;
  }
  static RunState parse(const std::string& text) {
    if(text.empty()||text.size()>65536)throw std::invalid_argument("run state exceeds bound");
    std::istringstream in(text);in.imbue(std::locale::classic());std::string magic;int version,n;
    RunState s;if(!(in>>magic>>version)||magic!="RESONANTIA_JOINT_RUN"||version!=1||
      !(in>>s.identity>>s.epochs>>s.step>>s.initial_lr>>s.decay>>n)||n<1||n>4096)
      throw std::invalid_argument("invalid run state header");
    s.order.resize(n);for(int& i:s.order)if(!(in>>i))throw std::invalid_argument("truncated permutation");
    if(!(in>>s.random))throw std::invalid_argument("invalid crop RNG");in>>std::ws;
    if(!in.eof())throw std::invalid_argument("trailing run state");s.validate();return s;
  }
};
} // namespace voxmutatio::app_support::joint_full
