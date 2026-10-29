// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#include "voxmutatio/io/block_checkpoint.h"
#include "voxmutatio/core/sha256.h"
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace voxmutatio::io {
namespace {
constexpr std::size_t header_size=160;
using Header=std::array<unsigned char,header_size>;
struct Fd {
    int value=-1;
    explicit Fd(int v=-1):value(v){}
    ~Fd(){if(value>=0)::close(value);}
    Fd(const Fd&)=delete;Fd& operator=(const Fd&)=delete;
};
[[noreturn]] void bad(const char* message){throw std::runtime_error(message);}
bool valid_key(const std::string& key) {
    return !key.empty()&&key.size()<=64&&key.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-")==key.npos;
}
void check_key(const std::string& key){if(!valid_key(key))throw std::invalid_argument("invalid checkpoint key");}
void transfer(int fd,void* data,std::size_t bytes,bool writing) {
    auto* p=static_cast<unsigned char*>(data);
    while(bytes) {
        const auto n=writing ? ::write(fd,p,bytes) : ::read(fd,p,bytes);
        if(n<0&&errno==EINTR)continue;
        if(n<=0)bad(writing?"checkpoint write failed":"checkpoint truncated/read failed");
        p+=n;bytes-=static_cast<std::size_t>(n);
    }
}
void put64(Header& h,std::size_t p,std::uint64_t value){for(int i=0;i<8;++i)h[p+i]=static_cast<unsigned char>(value>>(8*i));}
std::uint64_t get64(const Header& h,std::size_t p){std::uint64_t v=0;for(int i=0;i<8;++i)v|=std::uint64_t(h[p+i])<<(8*i);return v;}
std::string digest(const std::string& identity,const std::string& key,std::uint64_t offset,std::span<const float> data) {
    core::Sha256 h;
    const auto prefix=identity+"\n"+key+"\n"+std::to_string(offset)+"\n"+std::to_string(data.size())+"\n";
    h.update(prefix.data(),prefix.size());h.update(data.data(),data.size_bytes());return h.finish();
}
void finite(std::span<const float> data){for(float f:data)if(!std::isfinite(f))bad("checkpoint contains non-finite data");}
struct Dir {
    DIR* value;
    ~Dir(){if(value)::closedir(value);}
};
} // namespace

struct BlockCheckpoint::Impl {
    Fd directory,lock;
    std::string identity;
    std::function<bool()> pause;
    Limits limits;
    std::uint64_t bytes=0,hits=0;
    std::uint32_t blocks=0;
    bool healthy=true;

    Impl(const std::filesystem::path& path,std::string id,Mode mode,std::function<bool()> stop,Limits bounds)
        :identity(std::move(id)),pause(std::move(stop)),limits(bounds) {
        static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559);
        static_assert(std::endian::native==std::endian::little,"checkpoint format requires little endian IEEE float");
        if(!core::valid_digest(identity)||limits.block_bytes<4||limits.block_bytes>32ULL*1024*1024||
           limits.total_bytes<header_size+4||limits.total_bytes>2ULL*1024*1024*1024||!limits.blocks||limits.blocks>4096)
            throw std::invalid_argument("invalid checkpoint identity or limits");
        if(mode!=Mode::Create&&mode!=Mode::Resume)throw std::invalid_argument("invalid checkpoint mode");
        if(mode==Mode::Create)std::filesystem::create_directories(path);
        directory.value=::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
        if(directory.value<0)bad("cannot open checkpoint directory");
        lock.value=::openat(directory.value,".lock",O_RDWR|O_CREAT|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC,0600);
        struct stat st{};
        if(lock.value<0||::fstat(lock.value,&st)||!S_ISREG(st.st_mode)||st.st_nlink!=1||::flock(lock.value,LOCK_EX|LOCK_NB))
            bad("checkpoint directory is already owned or lock is invalid");
        const auto manifest="RESONANTIA_BLOCKS 1\n"+identity+"\n";
        bool initialize=mode==Mode::Create;
        if(mode==Mode::Resume) {
            Fd f(::openat(directory.value,"identity",O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC));
            if(f.value<0&&errno==ENOENT)initialize=true;
            else {
                if(f.value<0||::fstat(f.value,&st)||!S_ISREG(st.st_mode)||st.st_nlink!=1||st.st_size!=static_cast<off_t>(manifest.size()))
                    bad("checkpoint identity missing or malformed");
                std::string text(manifest.size(),'\0');transfer(f.value,text.data(),text.size(),false);
                if(text!=manifest)bad("checkpoint identity mismatch");
            }
        }
        // Bound traversal as well as aggregate disk use. Never follow files or
        // remove unknown paths; only this format's incomplete publications.
        Dir dir{::fdopendir(::dup(directory.value))};if(!dir.value)bad("cannot scan checkpoint directory");
        std::uint32_t scanned=0;
        for(;;) {
            errno=0;auto* entry=::readdir(dir.value);
            if(!entry){if(errno)bad("checkpoint directory scan failed");break;}
            std::string name=entry->d_name;
            if(name=="."||name==".."||name==".lock")continue;
            if(++scanned>8194)bad("checkpoint directory entry limit exceeded");
            if(::fstatat(directory.value,name.c_str(),&st,AT_SYMLINK_NOFOLLOW)||!S_ISREG(st.st_mode)||st.st_nlink!=1)
                bad("checkpoint contains non-regular or linked entry");
            if(mode==Mode::Create)bad("new checkpoint directory is not empty");
            if(name=="identity")continue;
            if(initialize) {
                if(name=="identity.pending"&&st.st_size<=256)continue;
                bad("checkpoint identity missing with existing block data");
            }
            if(name.ends_with(".block.pending")&&valid_key(name.substr(0,name.size()-14))) {
                if(::unlinkat(directory.value,name.c_str(),0))bad("cannot remove incomplete checkpoint block");
                continue;
            }
            if(!name.ends_with(".block")||!valid_key(name.substr(0,name.size()-6))||
               st.st_size<static_cast<off_t>(header_size+4)||std::uint64_t(st.st_size)>header_size+limits.block_bytes)
                bad("invalid checkpoint block entry");
            if(++blocks>limits.blocks||std::uint64_t(st.st_size)>limits.total_bytes-bytes)
                bad("checkpoint disk limit exceeded");
            bytes+=st.st_size;
        }
        if(initialize) {
            if(mode==Mode::Resume&&::unlinkat(directory.value,"identity.pending",0)&&errno!=ENOENT)
                bad("cannot remove incomplete checkpoint identity");
            publish("identity",std::span<const unsigned char>(
                reinterpret_cast<const unsigned char*>(manifest.data()),manifest.size()),{});
        }
        if(::fsync(directory.value))bad("checkpoint directory sync failed");
    }

    void publish(const std::string& name,std::span<const unsigned char> head,std::span<const float> data) try {
        const auto pending=name+".pending";
        Fd f(::openat(directory.value,pending.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600));
        if(f.value<0)bad("cannot create checkpoint staging file");
        bool renamed=false;
        try {
            transfer(f.value,const_cast<unsigned char*>(head.data()),head.size_bytes(),true);
            transfer(f.value,const_cast<float*>(data.data()),data.size_bytes(),true);
            if(::fsync(f.value))bad("checkpoint file sync failed");
            if(::renameat(directory.value,pending.c_str(),directory.value,name.c_str()))bad("checkpoint publication failed");
            renamed=true;
            if(::fsync(directory.value))bad("checkpoint publication directory sync failed");
        }catch(...) {
            if(!renamed)::unlinkat(directory.value,pending.c_str(),0);
            throw;
        }
    } catch(...) {healthy=false;throw;}

    std::optional<std::vector<float>> read(const std::string& key,std::uint64_t offset,std::uint64_t count) {
        if(!healthy)bad("checkpoint storage failed; reopen before further work");
        check_key(key);
        if(!count||count>limits.block_bytes/sizeof(float))throw std::invalid_argument("checkpoint count out of bounds");
        const auto name=key+".block";
        Fd f(::openat(directory.value,name.c_str(),O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC));
        if(f.value<0){if(errno==ENOENT)return {};bad("cannot read checkpoint block");}
        struct stat a{},b{};
        const auto size=header_size+count*sizeof(float);
        if(::fstat(f.value,&a)||!S_ISREG(a.st_mode)||a.st_nlink!=1||a.st_size!=static_cast<off_t>(size))
            bad("checkpoint block size or type mismatch");
        Header h{};transfer(f.value,h.data(),h.size(),false);
        if(std::memcmp(h.data(),"RSBLK001",8)||std::memcmp(h.data()+8,identity.data(),64)||
           get64(h,72)!=offset||get64(h,80)!=count||get64(h,152)!=0)
            bad("checkpoint block metadata mismatch");
        std::vector<float> data(static_cast<std::size_t>(count));transfer(f.value,data.data(),data.size()*sizeof(float),false);
        if(::fstat(f.value,&b)||a.st_size!=b.st_size||a.st_mtim.tv_sec!=b.st_mtim.tv_sec||a.st_mtim.tv_nsec!=b.st_mtim.tv_nsec||
           a.st_ctim.tv_sec!=b.st_ctim.tv_sec||a.st_ctim.tv_nsec!=b.st_ctim.tv_nsec)bad("checkpoint changed while reading");
        finite(data);const auto checksum=digest(identity,key,offset,data);
        if(std::memcmp(h.data()+88,checksum.data(),64))bad("checkpoint block checksum mismatch");
        return data;
    }
};

BlockCheckpoint::BlockCheckpoint(const std::filesystem::path& directory,std::string identity,Mode mode,std::function<bool()> pause)
    :BlockCheckpoint(directory,std::move(identity),mode,std::move(pause),Limits{}){}
BlockCheckpoint::BlockCheckpoint(const std::filesystem::path& directory,std::string identity,Mode mode,std::function<bool()> pause,Limits limits)
    :impl_(std::make_unique<Impl>(directory,std::move(identity),mode,std::move(pause),limits)){}
BlockCheckpoint::~BlockCheckpoint()=default;
std::optional<std::vector<float>> BlockCheckpoint::load(const std::string& key,std::uint64_t offset,std::uint64_t count) {
    auto value=impl_->read(key,offset,count);if(value)++impl_->hits;return value;
}
void BlockCheckpoint::save(const std::string& key,std::uint64_t offset,std::span<const float> data) {
    check_key(key);
    if(data.empty()||data.size()>impl_->limits.block_bytes/sizeof(float))
        throw std::invalid_argument("checkpoint count out of bounds");
    finite(data);
    if(auto old=impl_->read(key,offset,data.size())) {
        if(std::memcmp(old->data(),data.data(),data.size_bytes()))bad("cannot replace a committed checkpoint block");
        return;
    }
    const auto size=header_size+data.size_bytes();
    if(impl_->blocks>=impl_->limits.blocks||size>impl_->limits.total_bytes-impl_->bytes)bad("checkpoint disk limit exceeded");
    Header h{};std::memcpy(h.data(),"RSBLK001",8);std::memcpy(h.data()+8,impl_->identity.data(),64);
    put64(h,72,offset);put64(h,80,data.size());const auto checksum=digest(impl_->identity,key,offset,data);
    std::memcpy(h.data()+88,checksum.data(),64);
    impl_->publish(key+".block",h,data);++impl_->blocks;impl_->bytes+=size;
}
void BlockCheckpoint::boundary()const {
    if(!impl_->healthy)bad("checkpoint storage failed; reopen before further work");
    if(impl_->pause&&impl_->pause())throw OperationPaused();
}
std::uint64_t BlockCheckpoint::restored_blocks()const noexcept{return impl_->hits;}
} // namespace voxmutatio::io
