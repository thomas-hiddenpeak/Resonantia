// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
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
// Streaming SHA-256 for dataset/cache identity; no runtime command or dependency.
class Sha256 {
 public:
  void update(const void* data,std::size_t n) {
    if(n>std::numeric_limits<std::uint64_t>::max()/8-bytes_)
      throw std::invalid_argument("SHA256 input too large");
    bytes_+=n;auto p=static_cast<const unsigned char*>(data);
    while(n){const auto take=std::min(n,64-used_);std::memcpy(block_.data()+used_,p,take);
      used_+=take;p+=take;n-=take;if(used_==64){compress();used_=0;}}
  }
  std::string finish() const {
    auto copy=*this;const auto bits=bytes_*8;copy.block_[copy.used_++]=0x80;
    if(copy.used_>56){std::fill(copy.block_.begin()+copy.used_,copy.block_.end(),0);copy.compress();copy.used_=0;}
    std::fill(copy.block_.begin()+copy.used_,copy.block_.begin()+56,0);
    for(int i=0;i<8;++i)copy.block_[63-i]=static_cast<unsigned char>(bits>>(8*i));
    copy.compress();std::ostringstream out;out<<std::hex<<std::setfill('0');
    for(auto w:copy.h_)out<<std::setw(8)<<w;return out.str();
  }
 private:
  void compress() {
    static constexpr std::uint32_t k[]={
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::uint32_t w[64];for(int i=0;i<16;++i)w[i]=(std::uint32_t(block_[4*i])<<24)|
      (std::uint32_t(block_[4*i+1])<<16)|(std::uint32_t(block_[4*i+2])<<8)|block_[4*i+3];
    for(int i=16;i<64;++i){const auto x=w[i-15],y=w[i-2];
      w[i]=w[i-16]+(std::rotr(x,7)^std::rotr(x,18)^(x>>3))+w[i-7]+(std::rotr(y,17)^std::rotr(y,19)^(y>>10));}
    auto a=h_[0],b=h_[1],c=h_[2],d=h_[3],e=h_[4],f=h_[5],g=h_[6],h=h_[7];
    for(int i=0;i<64;++i){const auto t1=h+(std::rotr(e,6)^std::rotr(e,11)^std::rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];
      const auto t2=(std::rotr(a,2)^std::rotr(a,13)^std::rotr(a,22))+((a&b)^(a&c)^(b&c));
      h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
    h_[0]+=a;h_[1]+=b;h_[2]+=c;h_[3]+=d;h_[4]+=e;h_[5]+=f;h_[6]+=g;h_[7]+=h;
  }
  std::array<std::uint32_t,8> h_{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  std::array<unsigned char,64> block_{};std::size_t used_=0;std::uint64_t bytes_=0;
};
inline std::string sha256(std::string_view text){Sha256 h;h.update(text.data(),text.size());return h.finish();}
inline bool valid_digest(std::string_view s){return s.size()==64 && s.find_first_not_of("0123456789abcdef")==s.npos;}
inline std::string sha256_file(const std::filesystem::path& path,std::uint64_t max_bytes=2ULL*1024*1024*1024) {
  struct Fd{int v;~Fd(){if(v>=0)close(v);}} fd{open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NONBLOCK)};
  struct stat a{},b{};
  if(fd.v<0 || fstat(fd.v,&a) || !S_ISREG(a.st_mode) || a.st_size<0 || std::uint64_t(a.st_size)>max_bytes)
    throw std::runtime_error("cannot hash bounded regular file: "+path.string());
  Sha256 hash;std::array<char,65536> buffer{};std::uint64_t count=0;
  for(;;){const auto n=read(fd.v,buffer.data(),buffer.size());if(n<0&&errno==EINTR)continue;
    if(n<0)throw std::runtime_error("file hash read failed");if(n==0)break;
    count+=n;if(count>max_bytes || count>std::uint64_t(a.st_size))throw std::runtime_error("file changed while hashing");
    hash.update(buffer.data(),n);}
  if(fstat(fd.v,&b) || count!=std::uint64_t(a.st_size) || a.st_dev!=b.st_dev || a.st_ino!=b.st_ino ||
      a.st_size!=b.st_size || a.st_mtim.tv_sec!=b.st_mtim.tv_sec || a.st_mtim.tv_nsec!=b.st_mtim.tv_nsec ||
      a.st_ctim.tv_sec!=b.st_ctim.tv_sec || a.st_ctim.tv_nsec!=b.st_ctim.tv_nsec)
    throw std::runtime_error("file changed while hashing");
  return hash.finish();
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
