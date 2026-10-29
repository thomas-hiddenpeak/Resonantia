#include "tools/web/task_store.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace resonantia::web {
namespace fs = std::filesystem;
namespace {
constexpr std::size_t kMaxRecordBytes = 65536;
struct Fd { int value; ~Fd() { if(value >= 0) ::close(value); } };
void require(bool ok, const char* message) { if(!ok) throw std::runtime_error(message); }
bool one_of(const std::string& value, std::initializer_list<const char*> choices) {
    return std::any_of(choices.begin(), choices.end(), [&](auto item){return value == item;});
}
void validate(const TaskRecord& r) {
    require(TaskStore::valid_id(r.id), "invalid task ID");
    require(r.attempt>0&&r.attempt<=2147483647,"invalid execution attempt");
    require(one_of(r.kind,{"convert","preprocess","training","step"}), "invalid task kind");
    require(one_of(r.state,{"preparing","queued","running","pause_requested","paused",
            "interrupted","succeeded","failed","cancelled"}), "invalid task state");
    for(const auto* value : {&r.name,&r.stage,&r.error,&r.unit,&r.result,&r.phase})
        require(value->size() <= 4096 && value->find('\0') == value->npos, "invalid task text");
    require(r.result.empty() || r.result=="work/result.wav", "invalid task result path");
    require(r.total == 0 || r.completed <= r.total, "invalid task progress");
    require(r.parameters.size() <= 64, "too many task parameters");
    if(const auto retention=r.parameters.find("retention");retention!=r.parameters.end())
        require(retention->second=="pruned-v1"&&r.state=="succeeded","invalid task retention state");
    if(const auto digest=r.parameters.find("result_sha256");digest!=r.parameters.end())
        require((r.kind=="convert"||r.kind=="preprocess")&&r.state=="succeeded"&&digest->second.size()==64&&
            digest->second.find_first_not_of("0123456789abcdef")==digest->second.npos,"invalid completed audio digest");
    if(const auto parent=r.parameters.find("source_task");parent!=r.parameters.end())
        require(TaskStore::valid_id(parent->second)&&parent->second!=r.id&&(r.kind=="convert"||r.kind=="preprocess"),"invalid source task");
    for(const auto& [key,value] : r.parameters)
        require(!key.empty() && key.size() <= 64 && value.size() <= 4096 &&
                key.find('\0') == key.npos && value.find('\0') == value.npos, "invalid task parameter");
}
std::string encode(const TaskRecord& r) {
    validate(r);
    std::ostringstream out;
    out << "RESONANTIA_TASK 2\n";
    for(const auto* value : {&r.id,&r.kind,&r.name,&r.state,&r.stage,&r.error,&r.unit,&r.result,&r.phase})
        out << std::quoted(*value) << '\n';
    out << r.revision << ' ' << r.created_ms << ' ' << r.updated_ms << ' ' << r.started_ms << ' '
        << r.finished_ms << ' ' << r.progress_at_ms << ' ' << r.completed << ' ' << r.total << ' ' << r.attempt << '\n';
    out << r.parameters.size() << '\n';
    for(const auto& [key,value] : r.parameters) out << std::quoted(key) << ' ' << std::quoted(value) << '\n';
    auto bytes=out.str(); require(bytes.size() <= kMaxRecordBytes,"task record exceeds limit");
    return bytes;
}
TaskRecord decode(const std::string& bytes) {
    std::istringstream in(bytes); std::string header;
    std::getline(in,header); require(header=="RESONANTIA_TASK 1"||header=="RESONANTIA_TASK 2","unsupported task record version");
    TaskRecord r;
    for(auto* value : {&r.id,&r.kind,&r.name,&r.state,&r.stage,&r.error,&r.unit,&r.result,&r.phase}) in >> std::quoted(*value);
    const auto number = [&]() {
        std::string token;in >> token;
        require(!token.empty() && token.size()<=20 && token.find_first_not_of("0123456789")==token.npos,
                "invalid task counter");
        std::uint64_t value=0;
        const auto parsed=std::from_chars(token.data(),token.data()+token.size(),value);
        require(parsed.ec==std::errc{} && parsed.ptr==token.data()+token.size(),"task counter overflow");
        return value;
    };
    r.revision=number();r.created_ms=number();r.updated_ms=number();r.started_ms=number();
    r.finished_ms=number();r.progress_at_ms=number();r.completed=number();r.total=number();
    if(header=="RESONANTIA_TASK 2")r.attempt=number();
    const auto count=number();
    require(bool(in) && r.revision != 0 && count <= 64,"malformed task record");
    for(std::size_t i=0;i<count;++i) {
        std::string key,value; in >> std::quoted(key) >> std::quoted(value);
        require(bool(in) && r.parameters.emplace(key,value).second,"malformed task parameters");
    }
    in >> std::ws; require(in.eof(),"trailing task record bytes"); validate(r); return r;
}
void checked_directory(const fs::path& path) {
    require(fs::is_directory(fs::symlink_status(path)),"task path is not a regular directory");
}
void sync_directory(const fs::path& path) {
    Fd fd{::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)};
    require(fd.value>=0 && ::fsync(fd.value)==0,"task directory sync failed");
}
} // namespace

bool TaskStore::valid_id(const std::string& id) noexcept {
    if(id.size()<8 || id.size()>96) return false;
    return std::all_of(id.begin(),id.end(),[](unsigned char c){
        return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-';
    });
}
TaskStore::TaskStore(fs::path root,std::size_t capacity):root_(std::move(root)),capacity_(capacity) {
    require(!root_.empty() && capacity_>0 && capacity_<=4096,"invalid task store configuration");
    root_=fs::absolute(root_).lexically_normal(); fs::create_directories(root_); checked_directory(root_);
    Fd fd{::open((root_/".owner.lock").c_str(),O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW,0600)};
    struct stat st{};
    require(fd.value>=0 && ::fstat(fd.value,&st)==0 && S_ISREG(st.st_mode),"cannot open task store lock");
    require(::flock(fd.value,LOCK_EX|LOCK_NB)==0,"task store already owned by another service");
    lock_fd_=fd.value;fd.value=-1;
}
TaskStore::~TaskStore() { if(lock_fd_>=0) ::close(lock_fd_); }
fs::path TaskStore::task_directory(const std::string& id) const {
    require(valid_id(id),"invalid task ID"); return root_/id;
}
std::optional<TaskRecord> TaskStore::read_locked(const std::string& id) const {
    const auto dir=task_directory(id);
    if(!fs::exists(fs::symlink_status(dir))) return std::nullopt;
    checked_directory(dir);
    Fd fd{::open((dir/"record").c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK)};
    if(fd.value<0 && errno==ENOENT) return std::nullopt;
    struct stat st{};
    require(fd.value>=0 && ::fstat(fd.value,&st)==0 && S_ISREG(st.st_mode) &&
            st.st_size>0 && st.st_size<=static_cast<off_t>(kMaxRecordBytes),"invalid task record file");
    std::string bytes(static_cast<std::size_t>(st.st_size),'\0'); std::size_t offset=0;
    while(offset<bytes.size()) {
        const auto n=::read(fd.value,bytes.data()+offset,bytes.size()-offset);
        if(n<0 && errno==EINTR) continue;
        require(n>0,"incomplete task record"); offset+=static_cast<std::size_t>(n);
    }
    char extra; require(::read(fd.value,&extra,1)==0,"task record changed while reading");
    auto r=decode(bytes);require(r.id==id,"task record identity mismatch");return r;
}
std::optional<TaskRecord> TaskStore::get(const std::string& id) const {
    std::lock_guard lock(mutex_);return read_locked(id);
}
std::vector<TaskRecord> TaskStore::list_locked() const {
    std::vector<TaskRecord> records;std::size_t scanned=0;
    for(const auto& e:fs::directory_iterator(root_)) {
        require(++scanned<=16384,"task directory scan limit exceeded");
        const auto id=e.path().filename().string();if(!valid_id(id))continue;
        if(auto r=read_locked(id)) records.push_back(std::move(*r));
        require(records.size()<=4096,"task history capacity exceeded");
    }
    std::sort(records.begin(),records.end(),[](const auto& a,const auto& b){
        return a.created_ms!=b.created_ms?a.created_ms>b.created_ms:a.id>b.id;
    });return records;
}
std::vector<TaskRecord> TaskStore::list(std::size_t offset,std::size_t limit) const {
    require(limit>0 && limit<=100,"invalid task page size");std::lock_guard lock(mutex_);
    auto records=list_locked(); if(offset>=records.size())return {};
    const auto last=offset+std::min(limit,records.size()-offset);
    return {records.begin()+offset,records.begin()+last};
}
std::vector<TaskRecord> TaskStore::snapshot() const {
    std::lock_guard lock(mutex_);return list_locked();
}
void TaskStore::write_locked(const TaskRecord& record) {
    const auto bytes=encode(record),dir=task_directory(record.id).string();
    fs::create_directory(dir);checked_directory(dir);sync_directory(root_);
    auto pattern=(fs::path(dir)/"record.tmp.XXXXXX").string();
    std::vector<char> name(pattern.begin(),pattern.end());name.push_back('\0');
    Fd fd{::mkstemp(name.data())};require(fd.value>=0,"cannot stage task record");
    struct Unlink { const char* path;~Unlink(){::unlink(path);} } cleanup{name.data()};
    require(::fcntl(fd.value,F_SETFD,FD_CLOEXEC)==0,"cannot secure task record descriptor");
    std::size_t offset=0;
    while(offset<bytes.size()) {
        const auto n=::write(fd.value,bytes.data()+offset,bytes.size()-offset);
        if(n<0&&errno==EINTR)continue;require(n>0,"task record write failed");offset+=n;
    }
    require(::fsync(fd.value)==0,"task record sync failed");
    const int file=fd.value;fd.value=-1;require(::close(file)==0,"task record close failed");
    require(::rename(name.data(),(fs::path(dir)/"record").c_str())==0,"task record publish failed");
    // A failure here means the rename committed but crash durability is unknown.
    // Callers must reload the record and report the storage failure, not rerun work.
    sync_directory(dir);
}
bool TaskStore::save(TaskRecord& record,std::uint64_t expected_revision) {
    validate(record);require(record.revision==expected_revision,"task revision mismatch");
    std::lock_guard lock(mutex_);const auto old=read_locked(record.id);
    if((old&&old->revision!=expected_revision)||(!old&&expected_revision!=0))return false;
    if(!old) {
        const auto records=list_locked();
        const auto retained=std::count_if(records.begin(),records.end(),[](const auto& r){return !r.parameters.contains("retention");});
        if(records.size()>=4096||static_cast<std::size_t>(retained)>=capacity_)throw TaskCapacityError();
    }
    require(expected_revision<std::numeric_limits<std::uint64_t>::max(),"task revision overflow");
    auto next=record;next.revision=expected_revision+1;write_locked(next);record.revision=next.revision;return true;
}
std::size_t TaskStore::recover_interrupted(std::uint64_t now_ms) {
    std::lock_guard lock(mutex_);auto records=list_locked();std::size_t count=0;
    for(auto& r:records) {
        if(!one_of(r.state,{"preparing","queued","running","pause_requested"}))continue;
        require(r.revision<std::numeric_limits<std::uint64_t>::max(),"task revision overflow");
        r.state="interrupted";r.updated_ms=now_ms;r.error="Service restarted; explicit recovery is required.";
        ++r.revision;write_locked(r);++count;
    }return count;
}
} // namespace resonantia::web
