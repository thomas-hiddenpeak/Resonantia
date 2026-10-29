#pragma once
#include "voxmutatio/io/file_digest.h"
#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace resonantia::web {
// Web orchestration has its own explicit protocol. Native CLI checkpoints still
// bind /proc/self/exe through OperationIdentity. Bump this protocol when Web-side
// preparation/mixing/training orchestration changes numerical semantics.
class WebWorkflowIdentity {
public:
    // Audited production releases with identical audio/training orchestration:
    // 2026-10-29 09:27 and 14:04. Never accept a caller-supplied producer digest.
    static constexpr std::array<std::string_view,2> kLegacyProducers={
        "91f7c8a4952e90cc4d069e7c83d4f77b6b599d63e1a20beddcd48f56b0b4a52a",
        "d95014094dcdebefc688e50effefb68ac679d68a3e3d562d7743ac2027c565e4"};
    explicit WebWorkflowIdentity(std::string_view domain) {
        append(hashes_[0],"format","web-workflow-v2");append(hashes_[0],"domain",domain);
        append(hashes_[0],"orchestration","2026-10-29.1");
        for(std::size_t i=0;i<kLegacyProducers.size();++i) {
            auto& h=hashes_[i+1];append(h,"format","operation-blocks-v1");append(h,"domain",domain);
            append(h,"executable",kLegacyProducers[i]);
        }
    }
    void field(std::string_view name,std::string_view value) {for(auto& h:hashes_)append(h,name,value);}
    void field(std::string_view name,const WebWorkflowIdentity& nested) {
        for(std::size_t i=0;i<hashes_.size();++i)append(hashes_[i],name,nested.hashes_[i].finish());
    }
    void file(std::string_view name,const std::filesystem::path& path){field(name,voxmutatio::io::sha256_file(path));}
    std::string finish()const{return hashes_[0].finish();}
    std::vector<std::string> legacy()const{return {hashes_[1].finish(),hashes_[2].finish()};}
    std::vector<std::string> identities()const{return {finish(),hashes_[1].finish(),hashes_[2].finish()};}
private:
    static void append(voxmutatio::core::Sha256& hash,std::string_view name,std::string_view value) {
        for(auto text:{name,value}){const auto length=std::to_string(text.size())+":";hash.update(length.data(),length.size());hash.update(text.data(),text.size());}
    }
    std::array<voxmutatio::core::Sha256,3> hashes_;
};
} // namespace resonantia::web
