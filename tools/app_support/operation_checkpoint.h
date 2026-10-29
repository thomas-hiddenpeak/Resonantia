// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
#include "voxmutatio/io/block_checkpoint.h"
#include "voxmutatio/io/file_digest.h"
#include "voxmutatio/core/types.h"
#include "tools/app_support/training_cancel.h"
#include <iomanip>
#include <sstream>

namespace voxmutatio::app_support {
// Length-prefixed fields prevent ambiguous concatenation. The executable digest
// deliberately follows /proc/self/exe: it is the running inode, even after an
// on-disk binary replacement. User input/model paths never follow final symlinks.
class OperationIdentity {
public:
    explicit OperationIdentity(std::string_view domain) {
        field("format","operation-blocks-v1");field("domain",domain);
        field("executable",io::sha256_file("/proc/self/exe",256ULL*1024*1024,true));
    }
    void field(std::string_view name,std::string_view value) {
        for(auto part:{name,value}){const auto n=std::to_string(part.size())+":";hash_.update(n.data(),n.size());hash_.update(part.data(),part.size());}
    }
    void file(std::string_view name,const std::filesystem::path& path){field(name,io::sha256_file(path));}
    std::string finish()const{return hash_.finish();}
private:
    core::Sha256 hash_;
};
inline std::string conversion_recipe(const VCConfig& c,int speaker) {
    std::ostringstream o;o.imbue(std::locale::classic());o<<std::setprecision(std::numeric_limits<double>::max_digits10);
    o<<speaker<<' '<<c.f0_up_key<<' '<<c.formant_shift<<' '<<c.index_rate<<' '<<int(c.f0_method)<<' '
     <<int(c.content_encoder)<<' '<<c.rms_mix_rate<<' '<<c.protect<<' '<<c.filter_radius<<' '
     <<c.edge_pad_sec<<' '<<c.apply_highpass<<' '<<int(c.synthesis_mode)<<' '<<c.synthesis_seed<<' '
     <<c.interp_unvoiced<<' '<<c.max_chunk_sec<<' '<<c.chunk_pad_sec<<' '<<c.chunk_query_sec<<' '
     <<c.target_sample_rate<<' '<<c.output_format<<' '<<c.device<<' '<<c.use_half_precision<<' '<<c.gpu_device<<' '
     <<int(c.version)<<' '<<c.has_f0<<' '<<c.num_speakers<<' '<<c.model_sample_rate;
    for(const auto& e:c.pitch_edits)o<<' '<<int(e.mode)<<' '<<e.start<<' '<<e.end<<' '<<e.semitones;
    return o.str();
}

// The owner store serializes the whole CLI run; children bind file/stage identity.
// Resume can create a not-yet-started child, but never replace an existing one.
// Browser/task identities do not cross this interface.
class OperationCheckpointSession {
public:
    OperationCheckpointSession(std::filesystem::path root,std::string identity,bool resume,const std::string& pause_file)
        :root_(std::move(root)),identity_(std::move(identity)),reader_(pause_file) {
        if(reader_.error_code())throw std::runtime_error("cannot open pause channel");
        if(resume&&!std::filesystem::is_directory(root_))throw std::runtime_error("checkpoint root missing");
        if(std::filesystem::is_symlink(std::filesystem::symlink_status(root_)))throw std::runtime_error("checkpoint root is a symlink");
        auto mode=resume?io::BlockCheckpoint::Mode::Resume:io::BlockCheckpoint::Mode::Create;
        if(resume&&!std::filesystem::exists(std::filesystem::symlink_status(root_/"owner/identity"))) {
            // No stage can legitimately precede the owner identity. An empty
            // root or its unfinished owner can be initialized, while lost
            // identity beside existing stage data must never be reconstructed.
            for(const auto& entry:std::filesystem::directory_iterator(root_))
                if(entry.path().filename()!="owner")throw std::runtime_error("checkpoint owner identity missing beside stage data");
            if(!std::filesystem::exists(std::filesystem::symlink_status(root_/"owner")))mode=io::BlockCheckpoint::Mode::Create;
        }
        owner_=std::make_unique<io::BlockCheckpoint>(root_/"owner",identity_,
            mode,stop_callback());
        owner_->boundary();
    }
    std::unique_ptr<io::BlockCheckpoint> stage(const std::string& key) {
        if(key.empty()||key.size()>64||key.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-")!=key.npos)
            throw std::invalid_argument("invalid checkpoint stage key");
        const auto path=root_/key;
        const auto status=std::filesystem::symlink_status(path);
        if(std::filesystem::exists(status)&&(!std::filesystem::is_directory(status)||std::filesystem::is_symlink(status)))
            throw std::runtime_error("invalid checkpoint stage directory");
        return std::make_unique<io::BlockCheckpoint>(path,core::sha256(identity_+":"+key),
            std::filesystem::exists(status)?io::BlockCheckpoint::Mode::Resume:io::BlockCheckpoint::Mode::Create,stop_callback());
    }
    void boundary()const{owner_->boundary();}
private:
    std::function<bool()> stop_callback(){return [this]{
        const auto result=reader_.poll();
        if(result==resonantia::training_control::TrainingResult::kError)throw std::runtime_error("pause channel read failed");
        return result==resonantia::training_control::TrainingResult::kCancelled;
    };}
    std::filesystem::path root_;
    std::string identity_;
    resonantia::training_control::TrainingCancelReader reader_;
    std::unique_ptr<io::BlockCheckpoint> owner_;
};
} // namespace voxmutatio::app_support
