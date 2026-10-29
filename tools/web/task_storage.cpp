#include "tools/web/task_storage.h"
#include "voxmutatio/io/file_digest.h"
#include <array>
#include <cerrno>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace resonantia::web {
namespace fs=std::filesystem;
namespace {
struct Fd {
    int value;
    explicit Fd(int v):value(v){}
    ~Fd(){if(value>=0)::close(value);}
    Fd(const Fd&)=delete;Fd& operator=(const Fd&)=delete;
};
struct Directory {
    DIR* value;
    explicit Directory(int fd) {value=::fdopendir(fd);if(!value){::close(fd);throw std::runtime_error("cannot inspect task directory");}}
    ~Directory(){::closedir(value);}
    Directory(const Directory&)=delete;Directory& operator=(const Directory&)=delete;
};
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
std::uint64_t add(std::uint64_t a,std::uint64_t b){require(b<=std::numeric_limits<std::uint64_t>::max()-a,"task storage size overflow");return a+b;}
int directory(const fs::path& path){const int fd=::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);require(fd>=0,"cannot open managed directory");return fd;}
int child_directory(int parent,const char* name){const int fd=::openat(parent,name,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);require(fd>=0,"cannot open managed child directory");return fd;}
struct Scan {
    std::size_t entries=0;
    std::uint64_t entry(int parent,const char* name,unsigned depth) {
        require(++entries<=262144,"task storage scan entry limit exceeded");
        struct stat st{};
        if(::fstatat(parent,name,&st,AT_SYMLINK_NOFOLLOW)!=0){
            // Atomic checkpoint publication can retire a temporary entry after
            // readdir. This is a live measurement, not a filesystem snapshot.
            if(errno==ENOENT)return 0;
            throw std::runtime_error("cannot inspect managed file");
        }
        if(S_ISREG(st.st_mode)){require(st.st_size>=0,"invalid managed file size");return static_cast<std::uint64_t>(st.st_size);}
        require(S_ISDIR(st.st_mode),"managed storage contains a symlink or special file");
        Fd fd(::openat(parent,name,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));
        if(fd.value<0&&errno==ENOENT)return 0;
        require(fd.value>=0,"cannot inspect managed child directory");return tree(fd.value,depth+1);
    }
    std::uint64_t tree(int fd,unsigned depth) {
        require(depth<=16,"task storage scan depth exceeded");
        Directory stream(child_directory(fd,"."));std::uint64_t bytes=0;
        for(;;){errno=0;auto* e=::readdir(stream.value);if(!e){require(errno==0,"task directory read failed");break;}
            const std::string_view name=e->d_name;if(name=="."||name=="..")continue;
            bytes=add(bytes,entry(fd,e->d_name,depth));
        }return bytes;
    }
};
bool full_training(const TaskRecord& r){const auto p=r.parameters.find("recovery");return r.kind=="training"&&p!=r.parameters.end()&&p->second=="training-full-v1";}
int owned_training(const fs::path& runs,const TaskRecord& r) {
    require(!r.name.empty()&&r.name.size()<=64&&r.name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")==r.name.npos,"invalid managed training name");
    Fd parent(directory(runs)),run(child_directory(parent.value,r.name.c_str())),full(child_directory(run.value,"full_training"));
    Fd marker(::openat(full.value,"web-task",O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK));struct stat st{};std::array<char,97> bytes{};
    require(marker.value>=0&&::fstat(marker.value,&st)==0&&S_ISREG(st.st_mode)&&st.st_nlink==1&&st.st_size==static_cast<off_t>(r.id.size()),"invalid training ownership marker");
    const auto n=::read(marker.value,bytes.data(),bytes.size());
    require(n==st.st_size&&std::string_view(bytes.data(),static_cast<std::size_t>(n))==r.id,"training directory belongs to another task");
    const int fd=full.value;full.value=-1;return fd;
}
void protected_file(int parent,const char* name) {
    Fd file(::openat(parent,name,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK));struct stat st{};
    require(file.value>=0&&::fstat(file.value,&st)==0&&S_ISREG(st.st_mode)&&st.st_nlink==1&&st.st_size>0,"protected result or source is unavailable; cleanup refused");
}
std::uint64_t remove_copy(int parent,const char* name) {
    struct stat st{};
    if(::fstatat(parent,name,&st,AT_SYMLINK_NOFOLLOW)!=0){require(errno==ENOENT,"cannot inspect redundant copy");return 0;}
    require(S_ISREG(st.st_mode)&&st.st_nlink==1&&st.st_size>=0,"redundant copy is not a private regular file");
    require(::unlinkat(parent,name,0)==0,"cannot remove redundant copy");return static_cast<std::uint64_t>(st.st_size);
}
}
TaskStorageUsage measure_task_storage(const fs::path& root,const fs::path& runs,const std::vector<TaskRecord>& records) {
    require(records.size()<=4096,"task history bound exceeded");
    Fd base(directory(root));Scan scan;TaskStorageUsage result;
    struct statvfs space{};require(::fstatvfs(base.value,&space)==0,"cannot inspect available disk space");
    require(space.f_frsize>0&&space.f_bavail<=std::numeric_limits<std::uint64_t>::max()/space.f_frsize,"available disk space overflow");
    result.available_bytes=static_cast<std::uint64_t>(space.f_bavail)*space.f_frsize;
    for(const auto& r:records){require(TaskStore::valid_id(r.id),"invalid storage task ID");require(result.tasks.emplace(r.id,0).second,"duplicate storage task ID");}
    Directory stream(child_directory(base.value,"."));
    for(;;){errno=0;auto* e=::readdir(stream.value);if(!e){require(errno==0,"task root read failed");break;}
        const std::string name=e->d_name;if(name=="."||name=="..")continue;
        const auto bytes=scan.entry(base.value,name.c_str(),0);result.managed_bytes=add(result.managed_bytes,bytes);
        if(auto found=result.tasks.find(name);found!=result.tasks.end())found->second=bytes;
    }
    for(const auto& r:records)if(full_training(r)){
        Fd full(owned_training(runs,r));const auto bytes=scan.tree(full.value,0);
        result.managed_bytes=add(result.managed_bytes,bytes);result.tasks.at(r.id)=add(result.tasks.at(r.id),bytes);
    }
    return result;
}
std::string storage_limit_reason(const TaskStorageUsage& usage,const TaskStorageLimits& limits,const std::string& id) {
    if(usage.available_bytes<limits.reserve_bytes)return "磁盘可用空间低于保护线，请先释放空间";
    if(usage.managed_bytes>=limits.managed_bytes)return "任务文件总量达到保护线，请先清理已完成任务的副本";
    const auto task=usage.tasks.find(id);
    if(task!=usage.tasks.end()&&task->second>=limits.task_bytes)return "本任务文件达到保护线，已保存内容保留，请检查磁盘与任务目录";
    return {};
}
std::uint64_t prune_task_copies(const fs::path& root,const fs::path& runs,const TaskRecord& r) {
    require(TaskStore::valid_id(r.id)&&r.state=="succeeded","only a successful task may be cleaned");
    Fd parent(directory(root)),task(child_directory(parent.value,r.id.c_str()));
    if(r.kind!="convert"&&r.kind!="preprocess"&&!full_training(r))return 0;
    Fd work(child_directory(task.value,"work"));
    std::uint64_t freed=0;
    if(full_training(r)) {
        Fd full(owned_training(runs,r)),result(child_directory(full.value,"result"));
        protected_file(result.value,"model.safetensors");protected_file(result.value,"model.index");
        for(const auto* name:{"generator.safetensors","discriminator.safetensors"})freed=add(freed,remove_copy(work.value,name));
    } else {
        require(r.result=="work/result.wav","completed audio has no retained result");
        protected_file(work.value,"source.wav");protected_file(work.value,"result.wav");
        const auto digest=r.parameters.find("result_sha256");
        require(digest!=r.parameters.end()&&voxmutatio::core::valid_digest(digest->second)&&
            voxmutatio::io::sha256_file(root/r.id/"work/result.wav")==digest->second,
            "completed result checksum unavailable or changed; cleanup refused");
        for(const auto* name:{"model.safetensors","model.index","cover.wav"})freed=add(freed,remove_copy(work.value,name));
        // A mixed final result does not replace the independently useful vocal.
        const auto kind=r.parameters.find("output_kind");
        if(kind==r.parameters.end()||kind->second!="mix")freed=add(freed,remove_copy(work.value,"output.wav"));
    }
    require(::fsync(work.value)==0,"cannot sync task cleanup");return freed;
}
} // namespace resonantia::web
