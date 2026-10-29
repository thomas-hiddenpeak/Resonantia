// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
namespace voxmutatio::core {
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
} // namespace voxmutatio::core
