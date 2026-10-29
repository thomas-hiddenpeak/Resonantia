#include "tools/app_support/joint_initialization.h"
#include "voxmutatio/core/sha256.h"
#include <cerrno>
#include <dirent.h>
#include <fcntl.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace voxmutatio::app_support::joint_full {
namespace {
struct Fd {int value;~Fd(){if(value>=0)::close(value);}};
struct Dir {DIR* value;~Dir(){if(value)::closedir(value);}};
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
bool temporary(std::string_view name,std::string_view target) {
    if(!name.starts_with(target)||name.size()!=target.size()+11)return false;
    const auto suffix=name.substr(target.size());
    return suffix.starts_with(".tmp.")&&suffix.substr(5).find_first_not_of(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")==suffix.npos;
}
struct InitialDirectory {
    Fd directory;
    std::optional<std::string> identity;
    std::vector<std::string> pending;
    explicit InitialDirectory(const std::filesystem::path& path)
        :directory{::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)} {
        require(directory.value>=0,"cannot open initial training checkpoint directory");
        Dir scan{::fdopendir(::openat(directory.value,".",O_RDONLY|O_DIRECTORY|O_CLOEXEC))};
        require(scan.value!=nullptr,"cannot inspect initial training checkpoint directory");
        bool partial_state=false;std::uint64_t bytes=0;std::size_t count=0;
        for(;;) {
            errno=0;const auto* entry=::readdir(scan.value);
            if(!entry){require(errno==0,"initial training directory scan failed");break;}
            const std::string name=entry->d_name;if(name=="."||name=="..")continue;
            require(++count<=18,"too many initial training checkpoint entries");
            struct stat st{};
            require(::fstatat(directory.value,name.c_str(),&st,AT_SYMLINK_NOFOLLOW)==0&&
                    S_ISREG(st.st_mode)&&st.st_nlink==1&&st.st_size>=0,
                    "initial training checkpoint entry must be a regular unlinked file");
            if(name==".lock")continue;
            if(name=="identity.txt") {
                require(st.st_size==65,"invalid initial training identity");
                Fd file{::openat(directory.value,name.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK)};
                struct stat opened{};
                require(file.value>=0&&::fstat(file.value,&opened)==0&&opened.st_dev==st.st_dev&&opened.st_ino==st.st_ino,
                        "initial training identity changed while opening");
                char value[66];ssize_t n;do{n=::read(file.value,value,sizeof value);}while(n<0&&errno==EINTR);
                require(n==65&&value[64]=='\n'&&core::valid_digest(std::string_view(value,64)),"invalid initial training identity");
                identity=std::string(value,64);continue;
            }
            const bool state=temporary(name,"state_step_0000000000.safetensors");
            require(state||temporary(name,"identity.txt"),"initial resume requires no published checkpoint, candidate or unknown file");
            require(std::uint64_t(st.st_size)<=(state?4ULL*1024*1024*1024:128ULL),"initial training partial file is too large");
            bytes+=std::uint64_t(st.st_size);
            require(bytes<=8ULL*1024*1024*1024&&pending.size()<16,"initial training staging limit exceeded");
            partial_state|=state;pending.push_back(name);
        }
        require(!partial_state||identity.has_value(),"initial state partial exists without a committed recipe identity");
    }
};
}
void validate_initial_training_directory(const std::filesystem::path& directory) {
    InitialDirectory checked(directory);
}
void recover_initial_training_directory(const std::filesystem::path& directory,std::string_view identity) {
    require(core::valid_digest(identity),"invalid expected training identity");
    InitialDirectory checked(directory);
    require(!checked.identity||*checked.identity==identity,"initial training directory belongs to another dataset or recipe");
    // The caller still owns RunLock. Published checkpoints were rejected during
    // inspection, so these partial initialization files cannot contain progress.
    for(const auto& name:checked.pending)
        require(::unlinkat(checked.directory.value,name.c_str(),0)==0,"cannot remove incomplete initial training write");
    require(::fsync(checked.directory.value)==0,"cannot sync initial training checkpoint directory");
}
}
