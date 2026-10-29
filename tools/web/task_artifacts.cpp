#include "tools/web/task_artifacts.h"
#include "voxmutatio/io/file_digest.h"
#include <cerrno>
#include <fcntl.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
namespace resonantia::web {
namespace fs=std::filesystem;
namespace {
struct Fd{int v;~Fd(){if(v>=0)::close(v);}};
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void key(const std::string& s){require(!s.empty()&&s.size()<=64&&s.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-")==s.npos,"invalid stage key");}
void sync(const fs::path& p,bool directory=false) {
    Fd fd{::open(p.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|(directory?O_DIRECTORY:0))};
    require(fd.v>=0&&::fsync(fd.v)==0,"artifact sync failed");
}
std::optional<std::string> read(const fs::path& path) {
    Fd fd{::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK)};
    if(fd.v<0&&errno==ENOENT)return {};
    struct stat st{};require(fd.v>=0&&::fstat(fd.v,&st)==0&&S_ISREG(st.st_mode)&&st.st_nlink==1&&st.st_size>0&&st.st_size<=8192,"invalid artifact marker");
    std::string text(st.st_size,'\0');std::size_t done=0;
    while(done<text.size()){auto n=::read(fd.v,text.data()+done,text.size()-done);if(n<0&&errno==EINTR)continue;require(n>0,"marker read failed");done+=n;}
    char extra;require(::read(fd.v,&extra,1)==0,"marker changed while reading");return text;
}
void remove_pending(const fs::path& path) {
    const auto s=fs::symlink_status(path);if(!fs::exists(s))return;
    require(fs::is_regular_file(s),"artifact staging path is not regular");
    fs::remove(path);
}
void publish(const fs::path& path,const std::string& text) {
    require(text.size()<=8192,"artifact marker too large");
    const auto pending=path.string()+".pending";remove_pending(pending);
    Fd fd{::open(pending.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600)};
    require(fd.v>=0,"marker staging failed");
    try {
        std::size_t done=0;while(done<text.size()) {
            const auto n=::write(fd.v,text.data()+done,text.size()-done);if(n<0&&errno==EINTR)continue;require(n>0,"marker write failed");done+=n;
        }
        require(::fsync(fd.v)==0,"marker file sync failed");
        require(::rename(pending.c_str(),path.c_str())==0,"marker publish failed");sync(path.parent_path(),true);
    }catch(...){::unlink(pending.c_str());throw;}
}
fs::path artifact(const fs::path& root,const fs::path& relative) {
    require(!relative.empty()&&!relative.is_absolute()&&relative.native().size()<=4096,"invalid stage output path");
    fs::path result=root;
    for(const auto& part:relative) {
        require(part!=".."&&part!="."&&!part.empty(),"invalid stage output component");result/=part;
        require(!fs::is_symlink(fs::symlink_status(result)),"stage output uses symlink");
    }
    require(fs::is_regular_file(fs::symlink_status(result)),"stage output missing or not regular");return result;
}
std::optional<fs::path> completed_stage(const fs::path& root,const std::string& identity,
                                      const std::string& stage,bool verify_checksum) {
    key(stage);const auto text=read(root/("stage."+stage+".done"));if(!text)return {};
    std::istringstream in(*text);std::string header,saved_identity,saved_stage,path,hash;
    std::getline(in,header);in>>saved_identity>>saved_stage>>std::quoted(path)>>hash;require(bool(in),"malformed stage marker");
    in>>std::ws;require(in.eof()&&header=="RESONANTIA_STAGE 1"&&saved_identity==identity&&saved_stage==stage&&voxmutatio::core::valid_digest(hash),"stage identity mismatch");
    auto file=artifact(root,path);
    if(verify_checksum)require(voxmutatio::io::sha256_file(file)==hash,"completed stage artifact checksum mismatch");
    return file;
}
}
TaskArtifacts::TaskArtifacts(fs::path root,std::string identity,const std::vector<std::string>& compatible):root_(fs::absolute(root).lexically_normal()),identity_(std::move(identity)) {
    require(voxmutatio::core::valid_digest(identity_),"invalid workflow identity");
    require(fs::is_directory(fs::symlink_status(root_)),"invalid task work directory");
    require(compatible.size()<=2,"too many compatible workflow identities");
    for(const auto& digest:compatible)require(voxmutatio::core::valid_digest(digest),"invalid compatible workflow identity");
    const auto marker=[](const std::string& digest){return "RESONANTIA_WORKFLOW 1\n"+digest+"\n";};
    if(const auto saved=read(root_/"workflow.identity")) {
        if(*saved!=marker(identity_)) {
            bool matched=false;
            for(const auto& digest:compatible)if(*saved==marker(digest)){identity_=digest;matched=true;break;}
            require(matched,"保存任务的素材、参数或计算程序校验不一致，无法安全复用；旧结果仍保留");
        }
    } else publish(root_/"workflow.identity",marker(identity_));
}
std::optional<fs::path> TaskArtifacts::completed(const std::string& stage)const {
    return completed_stage(root_,identity_,stage,true);
}
std::optional<fs::path> TaskArtifacts::retained_output(const fs::path& root,const std::string& stage,bool verify_checksum) {
    require(fs::is_directory(fs::symlink_status(root)),"invalid task work directory");
    const auto saved=read(root/"workflow.identity");if(!saved)return {};
    std::istringstream in(*saved);std::string header,identity;std::getline(in,header);in>>identity>>std::ws;
    require(in.eof()&&header=="RESONANTIA_WORKFLOW 1"&&voxmutatio::core::valid_digest(identity),"invalid saved workflow identity");
    return completed_stage(root,identity,stage,verify_checksum);
}
void TaskArtifacts::commit(const std::string& stage,const fs::path& output) {
    key(stage);const auto relative=fs::absolute(output).lexically_normal().lexically_relative(root_);
    const auto file=artifact(root_,relative);
    if(const auto old=completed(stage)){require(*old==file,"cannot replace completed stage");return;}
    const auto digest=voxmutatio::io::sha256_file(file);sync(file);
    for(auto parent=file.parent_path();;parent=parent.parent_path()){sync(parent,true);if(parent==root_)break;}
    std::ostringstream text;text<<"RESONANTIA_STAGE 1\n"<<identity_<<'\n'<<stage<<'\n'<<std::quoted(relative.string())<<'\n'<<digest<<'\n';
    publish(root_/("stage."+stage+".done"),text.str());
}
void TaskArtifacts::publish_result(const fs::path& input) {
    const auto file=artifact(root_,fs::absolute(input).lexically_normal().lexically_relative(root_));
    const auto pending=root_/"result.wav.pending";remove_pending(pending);
    fs::copy_file(file,pending);sync(pending);
    require(::rename(pending.c_str(),(root_/"result.wav").c_str())==0,"result publish failed");sync(root_,true);
}
} // namespace resonantia::web
