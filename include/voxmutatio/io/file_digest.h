// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
#include "voxmutatio/core/sha256.h"
#include <cerrno>
#include <filesystem>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
namespace voxmutatio::io {
using core::Sha256;
// Hash a bounded, stable regular file; reject final symlinks by default. The
// explicit exception supports the running /proc/self/exe and legacy consumers.
inline std::string sha256_file(const std::filesystem::path& path,std::uint64_t max_bytes=2ULL*1024*1024*1024,
                               bool follow_symlinks=false) {
  struct Fd{int v;~Fd(){if(v>=0)close(v);}} fd{open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NONBLOCK|(follow_symlinks?0:O_NOFOLLOW))};
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

} // namespace voxmutatio::io
