// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
#include "voxmutatio/io/safetensors.h"
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace voxmutatio::training::detail {
constexpr std::size_t kJointCheckpointMaxBytes=2ULL*1024*1024*1024;
constexpr std::size_t kJointCheckpointMaxText=160*1024;
constexpr std::size_t kJointCheckpointMaxRunState=65536;
struct CheckpointTensor {
  std::string name;
  std::vector<int> shape;
  std::function<std::vector<float>()> read;
};
inline std::size_t checkpoint_elements(const std::vector<int>& shape) {
  if(shape.empty() || shape.size()>4)throw std::invalid_argument("invalid checkpoint rank");
  std::size_t n=1;
  for(int d:shape) {
    if(d<=0 || n>kJointCheckpointMaxBytes/sizeof(float)/static_cast<unsigned>(d))
      throw std::invalid_argument("checkpoint tensor exceeds bound");
    n*=d;
  }
  return n;
}
// Unique temporary file owns its FILE/fd and pathname through every exception.
class CheckpointFile {
 public:
  explicit CheckpointFile(const std::string& path) {
    auto pattern=path+".tmp.XXXXXX";name_.assign(pattern.begin(),pattern.end());name_.push_back(0);
    const int fd=mkostemp(name_.data(),O_CLOEXEC);
    if(fd<0)throw std::runtime_error("cannot create checkpoint staging file");
    file_=fdopen(fd,"wb");
    if(!file_){close(fd);unlink(name_.data());throw std::runtime_error("checkpoint fdopen failed");}
  }
  CheckpointFile(const CheckpointFile&)=delete;
  CheckpointFile& operator=(const CheckpointFile&)=delete;
  ~CheckpointFile(){if(file_)std::fclose(file_);if(!name_.empty())unlink(name_.data());}
  void write(const void* data,std::size_t bytes) {
    if(bytes && std::fwrite(data,1,bytes,file_)!=bytes)throw std::runtime_error("checkpoint write failed");
  }
  void commit(const std::string& path) {
    if(std::fflush(file_) || std::ferror(file_) || fsync(fileno(file_)))
      throw std::runtime_error("checkpoint flush failed");
    auto* f=file_;file_=nullptr;
    if(std::fclose(f))throw std::runtime_error("checkpoint close failed");
    if(rename(name_.data(),path.c_str()))throw std::runtime_error("checkpoint publish failed");
    name_.clear();
  }
 private:
  std::vector<char> name_;
  std::FILE* file_=nullptr;
};
inline void write_joint_checkpoint(const std::string& path,
    const std::vector<CheckpointTensor>& entries,const std::string& text) {
  static_assert(std::endian::native==std::endian::little);
  if(path.empty() || path.find('\0')!=std::string::npos)throw std::invalid_argument("invalid checkpoint path");
  if(text.empty() || text.size()>kJointCheckpointMaxText)throw std::invalid_argument("checkpoint metadata exceeds bound");
  std::ostringstream header;header.imbue(std::locale::classic());header<<'{';
  std::size_t offset=0;std::set<std::string> names;
  for(const auto& e:entries) {
    if(e.name.empty() || e.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")!=std::string::npos ||
        !names.insert(e.name).second || !e.read || e.name=="state")throw std::invalid_argument("invalid checkpoint tensor name");
    const auto bytes=checkpoint_elements(e.shape)*sizeof(float);
    if(bytes>kJointCheckpointMaxBytes-offset)throw std::invalid_argument("checkpoint payload exceeds bound");
    if(offset)header<<',';
    header<<'"'<<e.name<<"\":{\"dtype\":\"F32\",\"shape\":[";
    for(std::size_t i=0;i<e.shape.size();++i){if(i)header<<',';header<<e.shape[i];}
    header<<"],\"data_offsets\":["<<offset<<','<<offset+bytes<<"]}";offset+=bytes;
  }
  if(offset)header<<',';
  // The existing runtime deliberately accepts only F32 safetensors. Encode
  // metadata bytes as exact integers [0,255], without broadening that contract.
  header<<"\"state\":{\"dtype\":\"F32\",\"shape\":["<<text.size()<<"],\"data_offsets\":["<<offset<<','<<offset+text.size()*sizeof(float)<<"]}}";
  auto json=header.str();json.append((8-json.size()%8)%8,' ');
  if(json.size()+8+text.size()*sizeof(float)>kJointCheckpointMaxBytes-offset)throw std::invalid_argument("checkpoint file exceeds bound");
  const std::uint64_t length=json.size();CheckpointFile file(path);
  file.write(&length,sizeof(length));file.write(json.data(),json.size());
  for(const auto& e:entries) {
    const auto values=e.read();
    if(values.size()!=checkpoint_elements(e.shape))throw std::runtime_error("checkpoint tensor changed shape");
    for(float value:values)if(!std::isfinite(value))throw std::runtime_error("nonfinite checkpoint tensor");
    file.write(values.data(),values.size()*sizeof(float));
  }
  std::vector<float> encoded;encoded.reserve(text.size());
  for(unsigned char byte:text)encoded.push_back(byte);
  file.write(encoded.data(),encoded.size()*sizeof(float));file.commit(path);
}
inline std::vector<float> checkpoint_tensor(const io::SafetensorsLoader& loader,
    const std::string& name,const std::vector<int>& shape) {
  const auto n=checkpoint_elements(shape);const auto* t=loader.get_tensor(name);
  if(!t || t->dtype!="F32" || t->shape!=std::vector<int64_t>(shape.begin(),shape.end()) ||
      t->data_nbytes!=n*sizeof(float) || !loader.data(name))throw std::invalid_argument("checkpoint layout mismatch: "+name);
  std::vector<float> values(n);std::memcpy(values.data(),loader.data(name),t->data_nbytes);
  for(float value:values)if(!std::isfinite(value))throw std::invalid_argument("nonfinite checkpoint tensor: "+name);
  if(name.find(".v.")!=std::string::npos)
    for(float value:values)if(value<0)throw std::invalid_argument("negative optimizer variance");
  if(name.ends_with(".weight_v") && name.find(".p.")!=std::string::npos) {
    const auto inner=n/shape[0];
    for(int row=0;row<shape[0];++row) {
      double sum=0;for(std::size_t j=0;j<inner;++j){const double v=values[row*inner+j];sum+=v*v;}
      if(!(static_cast<float>(sum)>0) || !std::isfinite(static_cast<float>(sum)))
        throw std::invalid_argument("invalid restored weight norm");
    }
  }
  return values;
}
} // namespace voxmutatio::training::detail
