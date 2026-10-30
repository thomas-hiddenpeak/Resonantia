#include "tools/web/web_service.h"
#include "tools/web/workflow_identity.h"
#include "tools/web/pitch_review.h"
#include <iomanip>
#include <sstream>
#include "tools/app_support/progress_protocol.h"
#include "tools/app_support/operation_checkpoint.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <limits>
#include <random>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "voxmutatio/separation/sep_registry.h"
#include "voxmutatio/io/audio_io.h"

namespace resonantia::web {

namespace fs = std::filesystem;

#ifdef RESONANTIA_TEST_JOB_OWNERSHIP
// Defined only by the private host test; the regular service has no callback.
void test_training_response_boundary(std::uint64_t generation, bool accepted);
#endif

namespace {

constexpr std::uintmax_t kMaximumStepHistoryBytes = 64 * 1024;

std::uint64_t wall_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

struct WorkflowError : std::runtime_error {
    int status;
    WorkflowError(int code,const std::string& message):std::runtime_error(message),status(code){}
};

std::string display_name(const TaskRecord& r) {
    const auto name=r.parameters.find("display_name");return name==r.parameters.end()?r.name:name->second;
}

bool valid_display_name(const std::string& name) {
    if(name.empty()||name.size()>255)return false;
    // Validate UTF-8 without applying filename filtering: this is display text.
    for(std::size_t i=0;i<name.size();) {
        const auto c=static_cast<unsigned char>(name[i]);
        if(c<0x80){if(c<0x20||c==0x7f)return false;++i;continue;}
        const std::size_t width=c>=0xc2&&c<=0xdf?2:c>=0xe0&&c<=0xef?3:c>=0xf0&&c<=0xf4?4:0;
        if(!width||width>name.size()-i)return false;
        unsigned cp=c&(0x7f>>width);
        for(std::size_t j=1;j<width;++j){const auto b=static_cast<unsigned char>(name[i+j]);if((b&0xc0)!=0x80)return false;cp=(cp<<6)|(b&0x3f);}
        if((width==3&&cp<0x800)||(width==4&&cp<0x10000)||(cp>=0xd800&&cp<=0xdfff)||cp>0x10ffff||(cp>=0x80&&cp<=0x9f))return false;
        i+=width;
    }
    return true;
}

bool pitch_preview_task(const TaskRecord& r) {
    return r.parameters.contains("recovery")&&r.parameters.at("recovery")=="audio-pitch-preview-v1";
}

bool mixed_result(const TaskRecord& r) {
    const auto kind=r.parameters.find("output_kind");return r.kind=="convert"&&kind!=r.parameters.end()&&kind->second=="mix";
}

std::string task_json(const TaskRecord& r,const fs::path& directory={}) {
    std::string out="{";
    for(const auto& [key,value] : std::map<std::string,std::string>{
        {"id",r.id},{"kind",r.kind},{"name",r.name},{"display_name",display_name(r)},{"state",r.state},
        {"stage",r.stage},{"phase",r.phase},{"error",r.error},{"unit",r.unit}})
        out += "\""+key+"\":\""+json_escape(value)+"\",";
    for(const auto& [key,value] : std::map<std::string,std::uint64_t>{
        {"revision",r.revision},{"attempt",r.attempt},{"created_ms",r.created_ms},{"updated_ms",r.updated_ms},
        {"started_ms",r.started_ms},{"finished_ms",r.finished_ms},
        {"progress_at_ms",r.progress_at_ms},{"completed",r.completed},{"total",r.total}})
        out += "\""+key+"\":"+std::to_string(value)+",";
    if(!directory.empty()) {
        out+="\"outputs\":[";std::string notice;
        if(r.state=="succeeded"&&!r.result.empty()) {
            const std::string kind=mixed_result(r)?"mix":r.kind=="convert"?"vocal":"result";
            const auto entry=[&](const std::string& k,const std::string& label){return "{\"kind\":\""+k+"\",\"label\":\""+label+"\",\"url\":\"/api/task/result?id="+r.id+"&variant="+k+"\"}";};
            out+=entry(kind,pitch_preview_task(r)?"修正后（局部试听）":kind=="mix"?"完整翻唱":kind=="vocal"?"干净人声":"处理后音频");
            if(pitch_preview_task(r))out+=","+entry("baseline","修正前（同一片段）");
            if(kind=="mix") {
                bool available=false;
                try {available=TaskArtifacts::retained_output(directory/"work","conversion",false).has_value();}catch(const std::exception&){}
                if(available)out+=","+entry("vocal","干净人声");
                else notice="此历史任务的干净人声未保留或不可读取，完整翻唱仍可试听；不会自动重新转换。";
            }
        }
        out+="],\"outputs_notice\":\""+json_escape(notice)+"\",";
    }
    out += "\"result_url\":\"" + (r.state=="succeeded"&&!r.result.empty()?
        "/api/task/result?id="+r.id:std::string{}) + "\",\"parameters\":{";bool comma=false;
    for(const auto& [key,value]:r.parameters) {
        if(comma)out+=",";comma=true;
        out+="\""+json_escape(key)+"\":\""+json_escape(value)+"\"";
    }
    const bool full=r.kind=="training"&&r.parameters.contains("recovery")&&r.parameters.at("recovery")=="training-full-v1";
    const bool recoverable=full||((r.kind=="convert"||r.kind=="preprocess")&&r.parameters.contains("recovery")&&(r.parameters.at("recovery")=="audio-v1"||pitch_preview_task(r)));
    const bool pause=recoverable&&(r.state=="queued"||r.state=="running"||r.state=="pause_requested")&&r.stage!="saving"&&r.stage!="publishing";
    const bool resume=recoverable&&(r.state=="paused"||r.state=="interrupted"||r.state=="failed"||(full&&r.state=="cancelled"));
    const bool cleanable=r.state=="succeeded"&&!r.parameters.contains("retention")&&
        ((r.kind!="convert"&&r.kind!="preprocess")||r.parameters.contains("result_sha256"));
    return out+"},\"actions\":{\"pause\":"+(pause?"true":"false")+",\"resume\":"+(resume?"true":"false")+
        ",\"cleanup\":"+(cleanable?"true":"false")+",\"revise\":"+
        ((r.kind=="convert"||r.kind=="preprocess")&&r.state=="succeeded"&&recoverable&&!pitch_preview_task(r)?"true":"false")+"}}";
}

void sync_artifact(const fs::path& path,bool directory=false) {
    const int fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|(directory?O_DIRECTORY:0));
    if(fd<0)throw std::runtime_error("cannot open task artifact for sync");
    const auto result=::fsync(fd);const auto closed=::close(fd);
    if(result||closed)throw std::runtime_error("cannot sync task artifact");
}

void snapshot_log(const fs::path& directory,const std::string& contents) {
    // The record is committed after this file. A crash can leave a newer log,
    // but cannot leave a terminal record pointing at an unflushed snapshot.
    const auto temporary=directory/"task.log.pending";
    const int fd=::open(temporary.c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd<0)throw std::runtime_error("cannot save task log");
    struct Owner{int fd;~Owner(){if(fd>=0)::close(fd);}} owner{fd};
    std::size_t done=0;
    while(done<contents.size()) {
        const auto n=::write(fd,contents.data()+done,contents.size()-done);
        if(n<0&&errno==EINTR)continue;
        if(n<=0)throw std::runtime_error("cannot write task log");done+=n;
    }
    if(::fsync(fd))throw std::runtime_error("cannot sync task log");
    owner.fd=-1;if(::close(fd))throw std::runtime_error("cannot close task log");
    fs::rename(temporary,directory/"task.log");
}

std::string service_instance_id() {
    std::random_device entropy;
    std::string id;
    for (int i = 0; i < 4; ++i) {
        char hex[9];
        std::snprintf(hex, sizeof hex, "%08x", static_cast<unsigned>(entropy()));
        id += hex;
    }
    return id;
}

bool nonempty_regular_file(const fs::path& path) {
    std::error_code error;
    if (!fs::is_regular_file(path, error) || error) return false;
    const std::uintmax_t size = fs::file_size(path, error);
    return !error && size != 0;
}

bool candidate_name(const std::string& name) {
    return name.size() == 38 && name.starts_with("candidate_epoch_") &&
           name.ends_with(".safetensors") &&
           name.substr(16, 10).find_first_not_of("0123456789") == std::string::npos;
}

bool regular_candidate(const fs::path& path) {
    std::error_code ec;
    return fs::is_regular_file(fs::symlink_status(path, ec)) && !ec && nonempty_regular_file(path);
}

bool valid_wav_file(const fs::path& path) {
    if (!nonempty_regular_file(path)) return false;
    std::error_code error;
    if (fs::file_size(path, error) < 12 || error) return false;
    std::ifstream input(path, std::ios::binary);
    std::array<char, 12> header{};
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    return input.gcount() == static_cast<std::streamsize>(header.size()) &&
           std::string_view(header.data(), 4) == "RIFF" &&
           std::string_view(header.data() + 8, 4) == "WAVE";
}

bool exceeds_buffered_response_limit(const fs::path& path) {
    std::error_code error;
    const std::uintmax_t size = fs::file_size(path, error);
    return !error && size > kMaxFileBytes;
}

// Constant gain only: preserve the original stereo residual and the converted
// vocal envelope. No compressor, limiter, or automatic loudness normalization.
bool mix_cover(const std::string& original, const std::string& all_vocals,
               const std::string& converted, const std::string& output) {
    namespace io = voxmutatio::io;
    auto source = io::read_stereo_audio(original);
    auto vocals = io::read_stereo_audio(all_vocals);
    auto voice = io::read_audio(converted, 44100);
    if (!source || !vocals || !voice || source->data.empty() || voice->data.empty() ||
        source->frames() != vocals->frames()) return false;
    // Song cleanup is 44.1k, its lead snapshot is 40k, and bounded conversion
    // goes through 16k before synthesizing 40k. Each resampler floors its frame
    // count. Accept that exact roundtrip in addition to the old two-frame
    // allowance; do not hide actual truncation behind a guessed time tolerance.
    const std::uint64_t lead_frames = std::uint64_t(source->frames()) * 40000 / 44100;
    const std::uint64_t model_frames = (lead_frames * 16000 / 40000) * 40000 / 16000;
    const std::uint64_t roundtrip_frames = model_frames * 44100 / 40000;
    if (std::abs(static_cast<long long>(source->frames()) -
                 static_cast<long long>(voice->data.size())) > 2 &&
        voice->data.size() != roundtrip_frames) return false;
    double peak = 0;
    for (std::size_t i = 0; i < source->data.size(); ++i) {
        source->data[i] = source->data[i] - vocals->data[i] +
            voice->data[std::min(i / 2, voice->data.size() - 1)];
        if (!std::isfinite(source->data[i])) return false;
        peak = std::max(peak, static_cast<double>(std::abs(source->data[i])));
    }
    const double gain = peak > 0.95 ? 0.95 / peak : 1.0;
    for (float& value : source->data) value *= gain;
    std::fprintf(stderr, "[vc_serve] cover fixed gain=%.6f dB frames=%zu\n",
                 20 * std::log10(gain), source->frames());
    return io::write_stereo_audio(output, source->data.data(),
                                  static_cast<int>(source->frames()), 44100);
}

std::optional<fs::path> first_valid_wav(const fs::path& directory) {
    std::error_code error;
    for (fs::directory_iterator entry(directory, error), end; !error && entry != end;
         entry.increment(error)) {
        if (!entry->is_regular_file(error) || error ||
            lower(entry->path().extension().string()) != ".wav") {
            if (error) return std::nullopt;
            continue;
        }
        if (valid_wav_file(entry->path())) return entry->path();
    }
    return std::nullopt;
}

std::optional<fs::path> latest_full_checkpoint(const fs::path& directory) {
    if(!fs::is_directory(directory))return std::nullopt;
    std::optional<fs::path> latest;
    for(const auto& entry:fs::directory_iterator(directory)) {
        const auto name=entry.path().filename().string();
        if(name.size()==33 && name.starts_with("state_step_") && name.ends_with(".safetensors") &&
           name.substr(11,10).find_first_not_of("0123456789")==std::string::npos &&
           fs::is_regular_file(entry.symlink_status()) && entry.file_size()>0 && (!latest || entry.path()>*latest))
            latest=entry.path();
    }
    return latest;
}

void copy_training_clips(const fs::path& source,const fs::path& destination) {
    fs::create_directories(destination);std::size_t count=0;
    for(const auto& entry:fs::directory_iterator(source)) {
        if(lower(entry.path().extension().string())!=".wav")continue;
        if(!fs::is_regular_file(entry.symlink_status()) || !valid_wav_file(entry.path()) || ++count>4096)
            throw std::runtime_error("invalid or excessive prepared training clips");
        fs::copy_file(entry.path(),destination/entry.path().filename());
    }
    if(!count)throw std::runtime_error("no prepared training clips");
}

WebWorkflowIdentity training_clips_identity(const fs::path& directory) {
    if(!fs::is_directory(fs::symlink_status(directory)))throw std::runtime_error("training clips directory missing or unsafe");
    std::vector<fs::path> files;std::size_t scanned=0;
    for(const auto& entry:fs::directory_iterator(directory)) {
        if(++scanned>8192)throw std::runtime_error("too many training directory entries");
        if(lower(entry.path().extension().string())!=".wav")continue;
        if(!fs::is_regular_file(entry.symlink_status())||files.size()>=4096||!valid_wav_file(entry.path()))
            throw std::runtime_error("invalid fixed training clip");
        files.push_back(entry.path());
    }
    if(files.empty())throw std::runtime_error("empty fixed training clips");
    std::sort(files.begin(),files.end());
    WebWorkflowIdentity identity("web-training-clips-v1");
    for(const auto& path:files){identity.field("name",path.filename().string());identity.file("wav",path);}
    return identity;
}
void sync_training_clips(const fs::path& directory) {
    for(const auto& entry:fs::directory_iterator(directory))sync_artifact(entry.path());
    sync_artifact(directory,true);
}
void write_training_marker(const fs::path& path,const std::string& value) {
    const auto temporary=path.string()+".pending";
    const auto existing=fs::symlink_status(temporary);
    if(fs::exists(existing)) {
        if(!fs::is_regular_file(existing))throw std::runtime_error("unsafe training marker staging");
        fs::remove(temporary);
    }
    {std::ofstream out(temporary,std::ios::binary);out<<value;if(!out)throw std::runtime_error("cannot write training marker");}
    sync_artifact(temporary);fs::rename(temporary,path);sync_artifact(path.parent_path(),true);
}

// A former server can leave an owned CLI alive. Signal only its retained,
// validated attempt channel; never recreate it or identify processes by PID.
// A missing channel means execution had not registered one (or already retired).
bool stop_interrupted_execution(const fs::path& path) {
    const int fd=::open(path.c_str(),O_RDWR|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);
    if(fd<0)return errno==ENOENT;
    struct Owner{int fd;~Owner(){if(fd>=0)::close(fd);}} owner{fd};
    struct stat st{};char value=0;
    if(::fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size!=1||st.st_nlink!=1||st.st_uid!=::geteuid())return false;
    ssize_t n;do{n=::pread(fd,&value,1,0);}while(n<0&&errno==EINTR);
    if(n!=1||(value!='0'&&value!='1'))return false;
    if(value=='0'){do{n=::pwrite(fd,"1",1,0);}while(n<0&&errno==EINTR);if(n!=1)return false;}
    if(::fsync(fd))return false;
    owner.fd=-1;return ::close(fd)==0;
}

}  // namespace

WebService::WebService(Paths paths, std::shared_ptr<const ProcessRunner> process_runner)
    : paths_(std::move(paths)), process_runner_(std::move(process_runner)),
      instance_id_(service_instance_id()) {
    if (!process_runner_) throw std::invalid_argument("process runner is required");
    if (!paths_.tasks.empty()) {
        tasks_ = std::make_unique<TaskStore>(paths_.tasks);
        tasks_->recover_interrupted(wall_time_ms());
        for(std::size_t offset=0;;offset+=100) {
            const auto records=tasks_->list(offset,100);
            for(auto r:records) {
                if(r.state!="interrupted"||!r.parameters.contains("recovery")||
                   (r.parameters.at("recovery")!="audio-v1"&&r.parameters.at("recovery")!="training-full-v1"))continue;
                if(!stop_interrupted_execution(tasks_->task_directory(r.id)/("control."+std::to_string(r.attempt)))) {
                    r.error="Previous worker control channel could not be signalled; GPU ownership may still be held.";
                    if(!tasks_->save(r,r.revision))throw std::runtime_error("cannot retain interrupted control failure");
                }
            }
            if(records.size()<100)break;
        }
    }
}

void WebService::persist_job_locked(const std::string& state, bool terminal) {
    if(!tasks_)return;
    if(task_storage_failed_)throw std::runtime_error("task storage unavailable");
    try {
        auto r=tasks_->get(job_.public_id);
        if(!r)throw std::runtime_error("task record missing");
        const auto now=wall_time_ms();
        if(r->stage!=job_.stage) {
            r->progress_at_ms=now;r->phase.clear();r->completed=r->total=0;r->unit.clear();
        }
        r->stage=job_.stage;r->state=state;r->updated_ms=now;
        if(!r->started_ms&&state=="running")r->started_ms=now;
        if(terminal) {
            r->finished_ms=state=="paused"?0:now;
            const auto log=log_tail((tasks_->task_directory(r->id)/"live.log").string(),65536);
            snapshot_log(tasks_->task_directory(r->id),log);
            if(state=="failed")r->error="Stage failed: "+r->stage+". See retained task log.";
        }
        if(!tasks_->save(*r,r->revision))throw std::runtime_error("task revision changed");
    } catch(...) {task_storage_failed_=true;throw;}
}
void WebService::enable_progress(ProcessSpec& process,std::uint64_t generation) {
    if(!tasks_||!generation)return;
    process.argv.push_back("--progress");
    process.observe_line=[this,generation](std::string_view line){observe_progress(generation,line);};
}
void WebService::observe_progress(std::uint64_t generation,std::string_view line) {
    const auto progress=voxmutatio::app_support::parse_progress_message(line);
    if(!progress)return;
    guard_task_storage(generation,progress->total>0&&progress->completed==progress->total);
    std::lock_guard lock(job_.mu);
    if(job_.generation!=generation||!job_.running)return;
    if(task_storage_failed_)throw std::runtime_error("task progress storage unavailable");
    try {
        auto r=tasks_->get(job_.public_id);
        if(!r)throw std::runtime_error("task progress record missing");
        const bool changed=r->phase!=progress->phase||r->unit!=progress->unit||r->total!=progress->total;
        if(!changed&&progress->completed<r->completed)throw std::runtime_error("task progress moved backwards");
        if(!changed&&progress->completed==r->completed)return;
        const auto now=std::chrono::steady_clock::now();
        if(!changed&&progress->completed!=progress->total&&
           now-job_.progress_saved<std::chrono::seconds(1))return;
        r->phase=progress->phase;r->unit=progress->unit;r->completed=progress->completed;r->total=progress->total;
        r->progress_at_ms=r->updated_ms=wall_time_ms();
        if(!tasks_->save(*r,r->revision))throw std::runtime_error("task progress revision changed");
        job_.progress_saved=now;
    }catch(...){task_storage_failed_=true;throw;}
}

void WebService::persist_terminal_locked(const std::string& state) noexcept {
    try {persist_job_locked(state,true);}
    catch(const std::exception& e){std::fprintf(stderr,"[vc_serve] cannot retain task result: %s\n",e.what());}
    catch(...){task_storage_failed_=true;std::fprintf(stderr,"[vc_serve] cannot retain task result\n");}
}
TaskStorageUsage WebService::storage_usage(bool force) {
    if(!tasks_)throw std::runtime_error("task storage is disabled");
    std::lock_guard lock(storage_mutex_);const auto now=std::chrono::steady_clock::now();
    if(force||!storage_cached_||now-storage_checked_>=std::chrono::seconds(5)) {
        auto usage=measure_task_storage(paths_.tasks,paths_.runs,tasks_->snapshot());
        storage_cached_=std::move(usage);storage_checked_=now;
    }
    return *storage_cached_;
}
void WebService::require_storage_room(const std::string& id) {
    if(!tasks_||!paths_.enforce_task_storage)return;
    try {
        const auto reason=storage_limit_reason(storage_usage(true),paths_.storage_limits,id);
        if(!reason.empty())throw WorkflowError(507,reason);
    }catch(const WorkflowError&){throw;}
    catch(const std::exception& e){throw WorkflowError(503,std::string("磁盘保护状态暂不可确认：")+e.what());}
}
void WebService::guard_task_storage(std::uint64_t generation,bool force) {
    if(!tasks_||!paths_.enforce_task_storage||!generation)return;
    std::string id;
    {std::lock_guard lock(job_.mu);
        if(job_.generation!=generation||!job_.running||job_.pause_requested||
           (job_.kind!=JobKind::kConversion&&job_.kind!=JobKind::kPreprocess&&!job_.recoverable_training))return;
        id=job_.public_id;
    }
    std::string reason;
    try{reason=storage_limit_reason(storage_usage(force),paths_.storage_limits,id);}
    catch(const std::exception& e){reason=std::string("磁盘保护核查失败，等待安全暂停：")+e.what();}
    if(reason.empty())return;
    std::shared_ptr<training_control::TrainingCancelSource> channel;
    {std::lock_guard lock(job_.mu);
        if(job_.generation!=generation||!job_.running||job_.stage=="publishing"||job_.stage=="saving")return;
        job_.pause_requested=true;job_.storage_notice=reason;channel=job_.cancel_source;
        try {
            auto r=tasks_->get(id);if(!r)throw std::runtime_error("storage guard task missing");
            r->error=reason;if(!tasks_->save(*r,r->revision))throw std::runtime_error("storage guard revision changed");
            persist_job_locked("pause_requested");
        } catch(...){task_storage_failed_=true;throw;}
    }
    if(channel&&channel->request()==training_control::CancelDelivery::kError) {
        std::lock_guard lock(job_.mu);if(job_.generation==generation)job_.cancel_error=true;
        throw std::runtime_error("storage protection pause delivery failed");
    }
}
void WebService::handle_storage(int fd) {
    try {
        const auto usage=storage_usage();const auto records=tasks_->snapshot();
        const auto retained=std::count_if(records.begin(),records.end(),[](const auto& r){return !r.parameters.contains("retention");});
        send_json(fd,"{\"managed_bytes\":"+std::to_string(usage.managed_bytes)+",\"available_bytes\":"+std::to_string(usage.available_bytes)+
            ",\"managed_limit_bytes\":"+std::to_string(paths_.storage_limits.managed_bytes)+",\"task_limit_bytes\":"+std::to_string(paths_.storage_limits.task_bytes)+
            ",\"reserve_bytes\":"+std::to_string(paths_.storage_limits.reserve_bytes)+",\"retained_tasks\":"+std::to_string(retained)+
            ",\"history_tasks\":"+std::to_string(records.size())+",\"retained_limit\":256,\"history_limit\":4096,\"reason\":\""+
            json_escape(storage_limit_reason(usage,paths_.storage_limits))+"\"}");
    }catch(const std::exception& e){send_response(fd,503,"Service Unavailable","application/json","{\"error\":\""+json_escape(e.what())+"\"}");}
}

void WebService::handle_tasks(int fd,const Request& req) {
    const auto unavailable=[&]{send_response(fd,503,"Service Unavailable","application/json",
        "{\"error\":\"task storage unavailable; no new background work is accepted\"}");};
    if(!tasks_||task_storage_failed_){unavailable();return;}
    try {
        if(req.path=="/api/tasks") {
            int offset=0,limit=25;
            auto o=query_get(req.query,"offset"),l=query_get(req.query,"limit");
            if((!o.empty()&&!parse_int_field(o,0,4096,offset))||
               (!l.empty()&&!parse_int_field(l,1,100,limit))) {
                send_response(fd,400,"Bad Request","application/json","{\"error\":\"invalid task page\"}");return;
            }
            std::string out="{\"tasks\":[";bool comma=false;
            for(const auto& r:tasks_->list(offset,limit)){if(comma)out+=",";comma=true;out+=task_json(r);}
            std::string active_id;
            {std::lock_guard lock(job_.mu);if(job_.running)active_id=job_.public_id;}
            send_json(fd,out+"],\"offset\":"+std::to_string(offset)+",\"limit\":"+std::to_string(limit)+
                ",\"active_id\":\""+json_escape(active_id)+"\"}");return;
        }
        const auto id=query_get(req.query,"id");
        if(!TaskStore::valid_id(id)) {
            send_response(fd,400,"Bad Request","application/json","{\"error\":\"invalid task ID\"}");return;
        }
        const auto r=tasks_->get(id);
        if(!r){send_response(fd,404,"Not Found","application/json","{\"error\":\"task not found\"}");return;}
        if(req.path=="/api/task/log") {
            std::string log;
            const auto directory=tasks_->task_directory(id);
            if(r->state!="queued"&&r->state!="running"&&r->state!="pause_requested"&&fs::is_regular_file(directory/"task.log"))log=log_tail((directory/"task.log").string(),65536);
            else if(r->kind=="convert"||r->kind=="preprocess") {
                for(const auto* name:{"stages.log","vocals/preprocess.log","cleanup/preprocess.log","convert.log"})
                    log+=log_tail((directory/"work"/name).string(),20000);
            } else log=log_tail((directory/"live.log").string(),65536);
            send_response(fd,200,"OK","text/plain; charset=utf-8",log);
        } else if(req.path=="/api/task/source") {
            if(r->state!="succeeded"||(r->kind!="convert"&&r->kind!="preprocess")) {
                send_response(fd,409,"Conflict","application/json","{\"error\":\"task has no reusable audio source\"}");return;
            }
            send_audio_file(fd,tasks_->task_directory(id)/"work/source.wav",req.range);
        } else if(req.path=="/api/task/result") {
            if(r->state!="succeeded"||r->result.empty()) {
                send_response(fd,409,"Conflict","application/json","{\"error\":\"task has no completed audio result\"}");return;
            }
            auto path=tasks_->task_directory(id)/r->result;
            const auto variant=query_get(req.query,"variant");
            if(!variant.empty()&&variant!="mix"&&variant!="vocal"&&variant!="result"&&variant!="baseline") {
                send_response(fd,400,"Bad Request","application/json","{\"error\":\"未知音频类型\"}");return;
            }
            if((variant=="mix"&&!mixed_result(*r))||(variant=="vocal"&&r->kind!="convert")) {
                send_response(fd,404,"Not Found","application/json","{\"error\":\"此任务没有这份音频\"}");return;
            }
            if(variant=="baseline") {
                try {
                    if(!pitch_preview_task(*r))throw std::runtime_error("此任务不是局部试听");
                    const auto baseline=TaskArtifacts::retained_output(tasks_->task_directory(id)/"work","baseline");
                    if(!baseline)throw std::runtime_error("修正前片段未保留");
                    path=*baseline;
                } catch(const std::exception& e) {send_response(fd,409,"Conflict","application/json","{\"error\":\""+json_escape(e.what())+"\"}");return;}
            }
            if(variant=="vocal"&&mixed_result(*r)) {
                try {
                    const auto vocal=TaskArtifacts::retained_output(tasks_->task_directory(id)/"work","conversion");
                    if(!vocal){send_response(fd,404,"Not Found","application/json","{\"error\":\"此任务未保留干净人声\"}");return;}
                    path=*vocal;
                } catch(const std::exception&) {
                    send_response(fd,409,"Conflict","application/json","{\"error\":\"干净人声缺失或校验失败，完整翻唱仍保留\"}");return;
                }
            }
            send_audio_file(fd,path,req.range);
        } else send_json(fd,task_json(*r,tasks_->task_directory(id)));
    } catch(const std::exception& e) {
        task_storage_failed_=true;
        std::fprintf(stderr,"[vc_serve] task query failed: %s\n",e.what());unavailable();
    }
}

void WebService::handle_task_rename(int fd,const Request& req) {
    std::vector<Part> parts;if(!request_parts(fd,req,parts))return;
    const auto id=field(parts,"id",""),expected=field(parts,"expected_name","");
    auto name=field(parts,"name","");const auto first=name.find_first_not_of(' '),last=name.find_last_not_of(' ');
    name=first==std::string::npos?"":name.substr(first,last-first+1);
    if(!TaskStore::valid_id(id)||!valid_display_name(name)||expected.size()>4096||
       parts.size()!=3||std::any_of(parts.begin(),parts.end(),[](const auto& p){return !p.filename.empty()||(p.name!="id"&&p.name!="name"&&p.name!="expected_name");})||
       std::any_of(parts.begin(),parts.end(),[&](const auto& p){return std::count_if(parts.begin(),parts.end(),[&](const auto& other){return other.name==p.name;})!=1;})) {
        send_response(fd,400,"Bad Request","application/json","{\"error\":\"任务名称须为1至255字节的有效文字，不能含控制字符\"}");return;
    }
    try {
        std::lock_guard lock(job_.mu);
        if(!tasks_||task_storage_failed_)throw WorkflowError(503,"任务存储暂不可用");
        auto r=tasks_->get(id);if(!r)throw WorkflowError(404,"任务不存在");
        const auto current=display_name(*r);
        if(current!=name) {
            if(current!=expected)throw WorkflowError(409,"任务已被重命名，请刷新后再修改");
            r->parameters["display_name"]=name;r->updated_ms=wall_time_ms();
            if(!tasks_->save(*r,r->revision))throw WorkflowError(409,"任务记录已变化，请重试");
        }
        send_json(fd,task_json(*r));
    }catch(const WorkflowError& e){send_response(fd,e.status,e.status==404?"Not Found":e.status==409?"Conflict":"Service Unavailable","application/json","{\"error\":\""+json_escape(e.what())+"\"}");}
     catch(const std::exception&){send_response(fd,503,"Service Unavailable","application/json","{\"error\":\"名称未能保存，请刷新后核实\"}");}
}

ProcessResult WebService::run_process(ProcessSpec& process) {
    if(!process.argv.empty()) {
        const auto program=fs::path(process.argv.front()).filename().string();
        if(program=="vc_convert"||program=="vc_preprocess"||program=="vc_train"||program=="build_index")
            process.resource_lock=fs::path(paths_.runs)/".gpu-worker.lock";
    }
    return process_runner_->run(process);
}

void WebService::audio_boundary(std::uint64_t generation,std::string_view stage) {
    if(!generation)return;
    guard_task_storage(generation,true);
    std::lock_guard lock(job_.mu);
    if(job_.generation!=generation||job_.cancel_error)throw std::runtime_error("task control ownership or I/O failed");
    if(job_.pause_requested||(job_.recoverable_training&&job_.cancel_requested))throw voxmutatio::io::OperationPaused();
    // A control request either wins before this boundary, or observes the new
    // stage (including the non-interruptible publication stage).
    if(!stage.empty()){job_.stage=stage;persist_job_locked("running");}
}
void WebService::enable_audio_control(ProcessSpec& process,std::uint64_t generation,const fs::path& checkpoint) {
    if(!tasks_||!generation)return;
    std::lock_guard lock(job_.mu);
    if(job_.generation!=generation)throw std::runtime_error("stale audio task");
    if(job_.kind!=JobKind::kConversion&&job_.kind!=JobKind::kPreprocess&&!job_.recoverable_training)return;
    if(job_.control_path.empty())throw std::runtime_error("audio task control missing");
    process.argv.insert(process.argv.end(),{"--checkpoint-dir",checkpoint.string(),"--pause-file",job_.control_path});
    if(fs::exists(fs::symlink_status(checkpoint)))process.argv.push_back("--resume");
}
bool WebService::run_audio_process(ProcessSpec& process,std::uint64_t generation) {
    const auto result=run_process(process);
    if(result.termination==ProcessTermination::kExited&&result.exit_code==2&&
       !result.io_error&&!result.system_error&&!result.signal_number&&generation) {
        std::lock_guard lock(job_.mu);
        if(job_.generation==generation&&(job_.pause_requested||(job_.recoverable_training&&job_.cancel_requested))&&!job_.cancel_error)
            throw voxmutatio::io::OperationPaused();
    }
    return result.success();
}

WebWorkflowIdentity WebService::audio_identity(const ConversionPlan& p)const {
    WebWorkflowIdentity identity(p.pitch_preview?"web-pitch-preview-v1":"web-audio-v1");
    if(p.pitch_preview)for(const auto& value:{p.analysis_id,std::to_string(p.preview_start),std::to_string(p.preview_end)})identity.field("preview",value);
    for(const auto& item:std::vector<std::string>{p.workflow,p.output_kind,p.voice,p.pitch_edits,
        std::to_string(p.pitch),std::to_string(p.filter_radius),std::to_string(p.index_rate),std::to_string(p.rms),
        std::to_string(p.protect),std::to_string(p.separate),std::to_string(p.deharmony),std::to_string(p.dereverb),
        std::to_string(p.deecho),std::to_string(p.denoise),std::to_string(p.preprocess_only)})identity.field("recipe",item);
    identity.file("input",p.input);
    if(!p.preprocess_only) {
        identity.file("conversion-program",fs::path(paths_.build)/"vc_convert");
        identity.file("hubert",paths_.hubert);identity.file("rmvpe",paths_.rmvpe);identity.file("generator",p.model);
        if(p.index_rate>0)identity.file("index",p.index);
    }
    bool cleanup=false;
    for(const auto& [enabled,model]:std::vector<std::pair<bool,std::string>>{
        {p.workflow=="song"||(p.workflow=="manual"&&p.separate),"vocal_roformer"},
        {p.workflow=="song"||p.workflow=="harmonies"||p.deharmony,"karaoke_roformer"},
        {p.dereverb,"dereverb_roformer"},{p.deecho,"deecho_roformer"},{p.denoise,"denoise_roformer"}}) {
        if(!enabled)continue;cleanup=true;
        for(const auto* suffix:{".safetensors",".json","_config.i64","_dim_inputs.i64","_freq_indices.i64","_num_bands_per_freq.i64"})
            identity.file(model+suffix,fs::path(paths_.models)/"separation"/(model+suffix));
    }
    if(cleanup)identity.file("cleanup-program",fs::path(paths_.build)/"vc_preprocess");
    return identity;
}
WebService::ConversionPlan WebService::restore_audio_plan(const TaskRecord& r)const {
    if((r.kind!="convert"&&r.kind!="preprocess")||!r.parameters.contains("recovery")||(r.parameters.at("recovery")!="audio-v1"&&!pitch_preview_task(r)))
        throw WorkflowError(409,"此任务没有可验证的恢复记录");
    ConversionPlan p;p.directory=(tasks_->task_directory(r.id)/"work").string();p.name=r.name;
    p.input=p.directory+"/source.wav";p.model=p.directory+"/model.safetensors";p.index=p.directory+"/model.index";
    p.preprocess_only=r.kind=="preprocess";
    p.pitch_preview=pitch_preview_task(r);
    if(p.pitch_preview) {
        p.analysis_id=r.parameters.at("analysis_id");
        if(!parse_double_field(r.parameters.at("preview_start"),0,600,p.preview_start)||
           !parse_double_field(r.parameters.at("preview_end"),0,600,p.preview_end)||p.preview_end<=p.preview_start||p.preview_end-p.preview_start>15||
           !voxmutatio::core::valid_digest(p.analysis_id))throw WorkflowError(409,"保存的试听范围无效");
    }
    if(r.parameters.contains("source_task"))p.source_task=r.parameters.at("source_task");
    const auto& v=r.parameters;
    p.voice=v.at("voice");p.workflow=v.at("workflow");p.output_kind=v.at("output_kind");p.pitch_edits=v.at("pitch_edits");
    if(!parse_int_field(v.at("f0_up_key"),-48,48,p.pitch)||!parse_int_field(v.at("filter_radius"),0,31,p.filter_radius)||
       !parse_double_field(v.at("index_rate"),0,1,p.index_rate)||!parse_double_field(v.at("rms_mix_rate"),0,1,p.rms)||
       !parse_double_field(v.at("protect"),0,1,p.protect)||!parse_bool_field(v.at("pp_separate"),p.separate)||
       !parse_bool_field(v.at("pp_deharmony"),p.deharmony)||!parse_bool_field(v.at("pp_dereverb"),p.dereverb)||
       !parse_bool_field(v.at("pp_deecho"),p.deecho)||!parse_bool_field(v.at("pp_denoise"),p.denoise))
        throw WorkflowError(409,"保存的任务参数无效");
    if((p.workflow!="song"&&p.workflow!="harmonies"&&p.workflow!="dry"&&p.workflow!="manual")||
       (p.output_kind!="vocal"&&p.output_kind!="mix")||(p.output_kind=="mix"&&p.workflow!="song"))
        throw WorkflowError(409,"保存的转换配方无效");
    return p;
}

void WebService::resolve_saved_audio_model(ConversionPlan& p)const {
    if(p.preprocess_only)return;
    const bool base=p.voice=="base"||p.voice=="base (pretrained)";
    if(!base&&!safe_name(p.voice))throw WorkflowError(409,"保存的声线身份无效");
    // Cleanup may remove only redundant snapshots. A published replacement is
    // usable only after the complete parent's workflow identity is verified.
    if(!nonempty_regular_file(p.model))p.model=base?paths_.gmodel:paths_.runs+"/"+p.voice+"/model.safetensors";
    if(p.index_rate>0&&!nonempty_regular_file(p.index))p.index=paths_.runs+"/"+p.voice+"/model.index";
}

voxmutatio::VCConfig WebService::conversion_config(const ConversionPlan& p)const {
    voxmutatio::VCConfig c;
    c.hubert_model_path=paths_.hubert;c.rmvpe_model_path=paths_.rmvpe;
    c.synthesizer_model_path=p.model;c.index_path=p.index;c.version=voxmutatio::ModelVersion::kV2;c.num_speakers=109;
    c.f0_up_key=p.pitch;c.filter_radius=p.filter_radius;c.index_rate=p.index_rate;c.rms_mix_rate=p.rms;c.protect=p.protect;
    if(!voxmutatio::parse_pitch_edits(p.pitch_edits,c.pitch_edits))throw WorkflowError(409,"保存的局部修音无效");
    return c;
}

void WebService::handle_pitch_report(int fd,const Request& req) {
    try {
        if(!tasks_||task_storage_failed_)throw WorkflowError(503,"任务存储不可用");
        const auto id=query_get(req.query,"id");
        if(!TaskStore::valid_id(id))throw WorkflowError(400,"invalid task ID");
        auto record=tasks_->get(id);if(!record)throw WorkflowError(404,"任务不存在");
        const bool fallback=record->state=="failed"&&record->parameters.contains("source_task");
        if(fallback)record=tasks_->get(record->parameters.at("source_task"));
        if(!record||record->kind!="convert"||record->state!="succeeded"||pitch_preview_task(*record))
            throw WorkflowError(409,"请选择已完成的整段转换；局部试听或未完成任务不能作为基线");
        const auto plan=restore_audio_plan(*record);const auto config=conversion_config(plan);
        const auto evidence=pitch_review::read_evidence(plan.directory);
        const auto curve=voxmutatio::f0::prepare_pitch_curve(evidence.raw,plan.filter_radius,plan.pitch,config.pitch_edits,evidence.pad,evidence.duration);
        const auto issues=voxmutatio::f0::find_pitch_issues(evidence.raw,curve,plan.filter_radius,evidence.pad,evidence.duration);
        const auto analysis=voxmutatio::core::sha256(evidence.identity+voxmutatio::app_support::conversion_recipe(config,0));
        std::ostringstream out;out.imbue(std::locale::classic());out<<std::setprecision(8);
        out<<"{\"task_id\":\""<<id<<"\",\"baseline_id\":\""<<record->id<<"\",\"analysis_id\":\""<<analysis
           <<"\",\"duration\":"<<evidence.duration<<",\"start_time\":"<<-evidence.pad<<",\"step\":0.01,\"notice\":\""
           <<(fallback?"失败的修音没有应用；以下分析与试听使用上一个成功版本。":"疑点来自音高轨迹，只是复核线索；正常换音也可能被标记，未标记处也可能有问题。")<<"\",\"raw\":[";
        for(std::size_t i=0;i<evidence.raw.size();++i){if(i)out<<',';out<<evidence.raw[i];}
        out<<"],\"processed\":[";for(std::size_t i=0;i<curve.size();++i){if(i)out<<',';out<<curve[i];}
        out<<"],\"issues\":[";
        for(std::size_t i=0;i<issues.size();++i) {
            if(i)out<<',';const auto& issue=issues[i];out<<"{\"start\":"<<issue.start<<",\"end\":"<<issue.end<<",\"reason\":\""<<json_escape(issue.reason)<<"\",\"suggestion\":";
            if(issue.suggestion) {const auto& e=*issue.suggestion;out<<"{\"start\":"<<e.start<<",\"end\":"<<e.end<<",\"semitones\":"<<e.semitones<<"}";}else out<<"null";
            out<<'}';
        }
        out<<"],\"baseline_edits\":\""<<json_escape(plan.pitch_edits)<<"\"}";send_json(fd,out.str());
    } catch(const WorkflowError& e) {send_response(fd,e.status,"Conflict","application/json","{\"error\":\""+json_escape(e.what())+"\"}");}
      catch(const std::exception& e) {send_response(fd,409,"Conflict","application/json","{\"error\":\"音高分析不可用："+json_escape(e.what())+"\"}");}
}

void WebService::handle_pitch_preview(int fd,const Request& req) {
    std::vector<Part> parts;if(!request_parts(fd,req,parts))return;
    try {
        if(!tasks_||task_storage_failed_)throw WorkflowError(503,"任务存储不可用");
        const auto id=field(parts,"task_id",""),parent_id=field(parts,"source_task","");
        if(!TaskStore::valid_id(id)||!TaskStore::valid_id(parent_id))throw WorkflowError(400,"invalid task ID");
        for(const auto& part:parts)if(!part.filename.empty()||(part.name!="task_id"&&part.name!="source_task"&&part.name!="pitch_edits"&&part.name!="preview_start"&&part.name!="preview_end"&&part.name!="analysis_id"))
            throw WorkflowError(400,"局部试听只接受修音范围；其他设置沿用成功版本");
        for(const auto& part:parts)if(std::count_if(parts.begin(),parts.end(),[&](const Part& other){return other.name==part.name;})!=1)
            throw WorkflowError(400,"试听字段不可重复");
        if(tasks_->get(id))throw WorkflowError(409,"task already exists; query its state");
        auto parent=tasks_->get(parent_id);
        if(!parent||parent->kind!="convert"||parent->state!="succeeded"||pitch_preview_task(*parent))throw WorkflowError(409,"试听基线不是已完成的整段转换");
        auto plan=restore_audio_plan(*parent);resolve_saved_audio_model(plan);
        const auto evidence=pitch_review::read_evidence(plan.directory);
        const auto config=conversion_config(plan);
        const auto analysis=voxmutatio::core::sha256(evidence.identity+voxmutatio::app_support::conversion_recipe(config,0));
        if(field(parts,"analysis_id","")!=analysis)throw WorkflowError(409,"分析已变化，请重新读取音高");
        if(!parse_double_field(field(parts,"preview_start",""),0,evidence.duration,plan.preview_start)||
           !parse_double_field(field(parts,"preview_end",""),0,evidence.duration,plan.preview_end)||plan.preview_end<=plan.preview_start||plan.preview_end-plan.preview_start>15)
            throw WorkflowError(400,"试听范围须在音频内，且不超过15秒");
        plan.pitch_edits=field(parts,"pitch_edits","");
        const auto corrected=conversion_config(plan);
        // Validate before creating a durable task. No model work at this boundary.
        const auto next_curve=voxmutatio::f0::prepare_pitch_curve(evidence.raw,plan.filter_radius,plan.pitch,corrected.pitch_edits,evidence.pad,evidence.duration);
        const auto old_curve=voxmutatio::f0::prepare_pitch_curve(evidence.raw,plan.filter_radius,plan.pitch,config.pitch_edits,evidence.pad,evidence.duration);
        if(old_curve==next_curve)throw WorkflowError(400,"没有需要试听的音高变化");
        for(std::size_t i=0;i<old_curve.size();++i)if(old_curve[i]!=next_curve[i]&&(i/100.-evidence.pad<plan.preview_start-.011||i/100.-evidence.pad>plan.preview_end+.011))
            throw WorkflowError(400,"请只预览当前片段内的修音变化");
        plan.pitch_preview=true;plan.source_task=parent_id;plan.analysis_id=analysis;plan.output_kind="vocal";
        plan.name="修音试听 · "+parent->name;
        start_audio_task(fd,std::move(plan),id);
    } catch(const WorkflowError& e) {send_response(fd,e.status,"Conflict","application/json","{\"error\":\""+json_escape(e.what())+"\"}");}
      catch(const std::exception& e) {send_response(fd,409,"Conflict","application/json","{\"error\":\"不能创建试听："+json_escape(e.what())+"\"}");}
}
void WebService::inherit_audio_stages(const JobTicket& ticket,const ConversionPlan& p,TaskArtifacts& artifacts) {
    if(p.source_task.empty())return;
    auto current=tasks_->get(ticket.public_id);
    if(!current)throw std::runtime_error("revision task missing");
    if(current->parameters.contains("revision_ready"))return;
    const auto parent=tasks_->get(p.source_task);
    if(!parent||parent->state!="succeeded")throw std::runtime_error("原任务不可用于创建新版");
    auto previous=restore_audio_plan(*parent);resolve_saved_audio_model(previous);
    if(!fs::is_regular_file(fs::symlink_status(fs::path(previous.directory)/"workflow.identity")))
        throw std::runtime_error("原任务缺少已保存的依赖身份，无法安全复用");
    const auto previous_identity=audio_identity(previous);
    TaskArtifacts saved(previous.directory,previous_identity.finish(),previous_identity.legacy());
    if(voxmutatio::io::sha256_file(previous.input)!=voxmutatio::io::sha256_file(p.input))
        throw std::runtime_error("新版素材与原任务不一致");
    std::string reused;
    const auto inherit=[&](const std::string& stage,std::optional<fs::path> source) {
        if(!source)return;
        audio_boundary(ticket.generation);
        if(!artifacts.completed(stage)) {
            if(*source==fs::path(previous.input))artifacts.commit(stage,p.input);
            else {
                if(fs::file_size(*source)>1024ull*1024*1024)throw std::runtime_error("复用音频超过1GiB");
                TempDir staging(p.directory,"inherit-");const fs::path copy=fs::path(staging.path())/"audio.wav";
                fs::copy_file(*source,copy);sync_artifact(copy);
                const auto output=fs::path(p.directory)/("inherited-"+stage+".wav");
                fs::rename(copy,output);artifacts.commit(stage,output);
            }
        }
        if(!reused.empty())reused+=",";reused+=stage;
    };
    if(p.workflow=="song"&&previous.workflow=="song")inherit("separation",saved.completed("separation"));
    const bool same_cleanup=p.workflow==previous.workflow&&p.separate==previous.separate&&
        p.deharmony==previous.deharmony&&p.dereverb==previous.dereverb&&p.deecho==previous.deecho&&p.denoise==previous.denoise;
    if(same_cleanup)inherit("cleanup",saved.completed("cleanup"));
    const bool same_conversion=!p.preprocess_only&&!previous.preprocess_only&&same_cleanup&&
        p.pitch==previous.pitch&&p.pitch_edits==previous.pitch_edits&&p.filter_radius==previous.filter_radius&&
        p.index_rate==previous.index_rate&&p.rms==previous.rms&&p.protect==previous.protect&&
        voxmutatio::io::sha256_file(p.model)==voxmutatio::io::sha256_file(previous.model)&&
        (p.index_rate==0||voxmutatio::io::sha256_file(p.index)==voxmutatio::io::sha256_file(previous.index));
    if(same_conversion) {
        std::optional<fs::path> conversion;
        // Cleanup removes output.wav, but a vocal-only final result is the
        // identical retained artifact. A mixed result cannot substitute for it.
        if(parent->parameters.contains("retention")&&!fs::exists(fs::symlink_status(fs::path(previous.directory)/"output.wav"))&&
           !fs::exists(fs::symlink_status(fs::path(previous.directory)/"inherited-conversion.wav"))) {
            if(previous.output_kind=="vocal"&&parent->parameters.contains("result_sha256")) {
                const auto file=fs::path(previous.directory)/"result.wav";
                if(voxmutatio::io::sha256_file(file)!=parent->parameters.at("result_sha256"))throw std::runtime_error("原任务结果校验失败");
                conversion=file;
            }
        } else conversion=saved.completed("conversion");
        inherit("conversion",conversion);
    }
    {
        std::ofstream log(fs::path(p.directory)/"stages.log",std::ios::app);
        log<<"[serve] revision from "<<p.source_task<<"; imported completed stages: "<<(reused.empty()?"none":reused)<<'\n';
        if(!log)throw std::runtime_error("cannot retain revision log");
    }
    std::lock_guard lock(job_.mu);current=tasks_->get(ticket.public_id);
    current->parameters["revision_ready"]="1";current->parameters["reused_stages"]=reused;
    if(!tasks_->save(*current,current->revision))throw std::runtime_error("cannot retain revision preparation");
}

void WebService::handle_task_control(int fd,const Request& req) {
    std::vector<Part> parts;if(!request_parts(fd,req,parts))return;
    const auto id=field(parts,"id","");int attempt=0;
    if(!TaskStore::valid_id(id)||!parse_int_field(field(parts,"attempt",""),1,2147483647,attempt)) {
        send_response(fd,400,"Bad Request","application/json","{\"error\":\"invalid task ID or execution attempt\"}");return;
    }
    JobTicket ticket;bool reserved=false;
    try {
        if(!tasks_)throw WorkflowError(503,"任务存储不可用");
        if(req.path=="/api/task/cleanup") {
            std::lock_guard lock(job_.mu);
            if(job_.running)throw WorkflowError(409,"任务运行期间暂不清理，请先等待或安全暂停");
            auto r=tasks_->get(id);if(!r)throw WorkflowError(404,"任务不存在");
            if(r->attempt!=std::uint64_t(attempt)||r->state!="succeeded")throw WorkflowError(409,"只有当前已成功的任务可以清理副本");
            std::uint64_t freed=0;
            if(!r->parameters.contains("retention")) {
                freed=prune_task_copies(paths_.tasks,paths_.runs,*r);
                r->parameters["retention"]="pruned-v1";r->updated_ms=wall_time_ms();
                try{if(!tasks_->save(*r,r->revision))throw std::runtime_error("cleanup revision changed");}
                catch(...){task_storage_failed_=true;throw;}
                {std::lock_guard storage_lock(storage_mutex_);storage_cached_.reset();}
            }
            send_json(fd,"{\"task_id\":\""+id+"\",\"freed_bytes\":"+std::to_string(freed)+"}");return;
        }
        if(req.path=="/api/task/pause") {
            std::shared_ptr<training_control::TrainingCancelSource> source;bool durable=true;int code=202;
            {
                std::lock_guard lock(job_.mu);
                if(job_.running&&job_.public_id==id) {
                    if(job_.attempt!=std::uint64_t(attempt))throw WorkflowError(409,"执行已变化，请刷新任务状态");
                    if((job_.kind!=JobKind::kConversion&&job_.kind!=JobKind::kPreprocess&&!job_.recoverable_training)||job_.stage=="saving"||job_.stage=="publishing")
                        throw WorkflowError(409,"当前阶段不支持暂停，任务仍在后台运行");
                    job_.pause_requested=true;source=job_.cancel_source;
                    try{persist_job_locked("pause_requested");}catch(...){durable=false;}
                } else {
                    if(task_storage_failed_)throw WorkflowError(503,"任务状态暂不可确认");
                    const auto r=tasks_->get(id);
                    if(!r)throw WorkflowError(404,"任务不存在");
                    if(r->attempt!=std::uint64_t(attempt)||r->state!="paused")throw WorkflowError(409,"此执行已结束或状态已变化");
                    code=200;
                }
                job_.changed.notify_all();
            }
            if(source&&source->request()==training_control::CancelDelivery::kError) {
                std::lock_guard lock(job_.mu);
                if(job_.public_id==id&&job_.attempt==std::uint64_t(attempt))job_.cancel_error=true;
                throw WorkflowError(503,"暂停通道写入失败，任务状态暂不可确认");
            }
            send_response(fd,code,code==202?"Accepted":"OK","application/json",
                "{\"task_id\":\""+id+"\",\"attempt\":"+std::to_string(attempt)+",\"durable\":"+(durable?"true":"false")+"}");return;
        }
        ConversionPlan plan;bool duplicate=false,training=false;
        {
            std::lock_guard lock(job_.mu);
            if(task_storage_failed_)throw WorkflowError(503,"任务存储不可用，未启动新的执行");
            auto r=tasks_->get(id);if(!r)throw WorkflowError(404,"任务不存在");
            if(r->attempt==std::uint64_t(attempt)+1&&r->parameters.contains("resumed_from")&&
               r->parameters.at("resumed_from")==std::to_string(attempt)) {
                duplicate=true;ticket.public_id=id;ticket.attempt=r->attempt;
            } else {
                if(r->attempt!=std::uint64_t(attempt))throw WorkflowError(409,"执行已变化，请刷新任务状态");
                if(job_.running)throw WorkflowError(409,"已有任务运行，请先等待或暂停");
                require_storage_room(id);
                training=r->kind=="training"&&r->parameters.contains("recovery")&&r->parameters.at("recovery")=="training-full-v1";
                if(r->state!="paused"&&r->state!="interrupted"&&r->state!="failed"&&!(training&&r->state=="cancelled"))throw WorkflowError(409,"此任务无需恢复");
                if(!training)plan=restore_audio_plan(*r);
                if(r->attempt>=2147483647||job_.generation==std::numeric_limits<std::uint64_t>::max())throw WorkflowError(409,"执行次数已达到上限");
                ticket.public_id=id;ticket.attempt=++r->attempt;ticket.generation=job_.generation+1;
                std::string stored_id=id,stored_name=r->name,stage="queued";
                r->parameters["resumed_from"]=std::to_string(attempt);r->state="queued";r->stage="queued";
                r->phase.clear();r->unit.clear();r->completed=r->total=0;r->error.clear();r->result.clear();r->finished_ms=0;
                r->updated_ms=r->progress_at_ms=wall_time_ms();
                try{if(!tasks_->save(*r,r->revision))throw std::runtime_error("resume revision changed");}
                catch(...){task_storage_failed_=true;throw;}
                job_.generation=ticket.generation;job_.attempt=ticket.attempt;job_.public_id=std::move(stored_id);
                job_.name=std::move(stored_name);job_.stage=std::move(stage);
                job_.kind=training?JobKind::kTraining:plan.preprocess_only?JobKind::kPreprocess:JobKind::kConversion;
                job_.recoverable_training=training;job_.storage_notice.clear();
                job_.done=job_.error=job_.cancel_requested=job_.pause_requested=job_.cancelled=job_.cancel_error=false;
                job_.cancel_source.reset();job_.control_path.clear();job_.running=true;reserved=true;
            }
        }
        if(!duplicate) {
            auto self=shared_from_this();std::thread worker([self,ticket,training,plan=std::move(plan)]()mutable{
                if(training)self->run_training_task(ticket);else self->run_audio_task(ticket,std::move(plan));});
            worker.detach();reserved=false;
        }
        send_response(fd,202,"Accepted","application/json","{\"task_id\":\""+id+"\",\"attempt\":"+std::to_string(ticket.attempt)+"}");
    }catch(const WorkflowError& e) {
        if(reserved) {if(job_.recoverable_training)finish_training(ticket.generation,TrainingOutcome::kError);else finish_audio_task(ticket,"",e.what());}
        send_response(fd,e.status,e.status==409?"Conflict":e.status==404?"Not Found":"Service Unavailable","application/json","{\"error\":\""+json_escape(e.what())+"\"}");
    }catch(const std::exception& e) {
        if(reserved) {if(job_.recoverable_training)finish_training(ticket.generation,TrainingOutcome::kError);else finish_audio_task(ticket,"",e.what());}
        send_response(fd,503,"Service Unavailable","application/json","{\"error\":\"cannot restore task; retained work is unchanged\"}");
    }
}

// ---- workflow ----
bool WebService::reserve_job(const std::string& name, JobKind kind, JobTicket& ticket, const std::string& requested_id) {
    std::lock_guard<std::mutex> lock(job_.mu);
    if (job_.running) return false;
    require_storage_room();
    if (job_.generation == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("job generation exhausted");
    ticket.generation = job_.generation + 1;
    ticket.public_id = requested_id.empty()?instance_id_ + "-" + std::to_string(ticket.generation):requested_id;
    // Prepare all allocating metadata before reserving. Moves after CAS cannot
    // strand a reservation or expose half-initialized identity to cancellation.
    std::string stored_id = ticket.public_id, stored_name = name, stage = "queued";
    if (task_storage_failed_) throw std::runtime_error("task storage unavailable");
    if (tasks_) {
        TaskRecord r; r.id=ticket.public_id;r.name=name;
        r.kind=kind==JobKind::kTraining?"training":kind==JobKind::kConversion?"convert":kind==JobKind::kPreprocess?"preprocess":"step";
        r.created_ms=r.updated_ms=wall_time_ms();
        try { if(!tasks_->save(r,0))return false; }
        catch(const TaskCapacityError&) {throw WorkflowError(409,"任务记录已达到容量上限；已有任务和结果仍可查看");}
        catch(...) {task_storage_failed_=true;throw;}
    }
    bool expected = false;
    if (!job_.running.compare_exchange_strong(expected, true)) return false;
    job_.generation = ticket.generation;job_.attempt=ticket.attempt;
    job_.public_id = std::move(stored_id); job_.name = std::move(stored_name);
    job_.stage = std::move(stage); job_.kind = kind;
    job_.done = false; job_.error = false;
    job_.cancel_requested = false; job_.cancelled = false;job_.pause_requested=false;job_.control_path.clear();
    job_.recoverable_training=false;job_.storage_notice.clear();
    job_.cancel_error = false; job_.cancel_source.reset();
    return true;
}
void WebService::set_stage(std::uint64_t generation, const std::string& stage) {
    std::lock_guard<std::mutex> lock(job_.mu);
    if (job_.generation == generation) {
        job_.stage = stage;
        persist_job_locked(job_.pause_requested?"pause_requested":stage=="queued"?"queued":"running");
    }
}
void WebService::finish(std::uint64_t generation, const std::string& stage, bool error) {
    std::lock_guard<std::mutex> lock(job_.mu);
    if (job_.generation != generation) return;
    persist_terminal_locked(error?"failed":"succeeded");
    job_.stage = stage; job_.error = error || task_storage_failed_; job_.done = true; job_.running = false;
}
WebService::TrainingCancelLease::~TrainingCancelLease() {
    // Wait out any already-started request write before staging can disappear.
    // Never nest the channel mutex and job mutex in either lock order.
    const bool clean = source->close();
    std::lock_guard<std::mutex> lock(service.job_.mu);
    if (service.job_.generation == generation) {
        service.job_.cancel_error = service.job_.cancel_error || !clean;
        service.job_.cancel_source.reset();
    }
}
void WebService::finish_training(std::uint64_t generation, TrainingOutcome outcome) {
    std::lock_guard<std::mutex> lock(job_.mu);
    if (job_.generation != generation || job_.kind != JobKind::kTraining) return;
    if (job_.cancel_error) outcome = TrainingOutcome::kError;
    persist_terminal_locked(outcome==TrainingOutcome::kPaused?"paused":outcome==TrainingOutcome::kCancelled?"cancelled":
                            outcome==TrainingOutcome::kError?"failed":"succeeded");
    if(task_storage_failed_)outcome=TrainingOutcome::kError;
    job_.cancelled = outcome == TrainingOutcome::kCancelled;
    job_.error = outcome == TrainingOutcome::kError;
    job_.stage = outcome==TrainingOutcome::kPaused?"paused":job_.cancelled ? "cancelled" : job_.error ? "error" : "done";
    job_.done = outcome!=TrainingOutcome::kPaused;
    // Called only after the request owner's staging scope, or the worker's
    // run_training scope (child reap, staging and GPU lock), has ended.
    job_.running = false;
}
bool WebService::cancellation_requested(std::uint64_t generation) {
    std::lock_guard<std::mutex> lock(job_.mu);
    return job_.generation != generation || job_.cancel_requested;
}
bool WebService::training_checkpoint(std::uint64_t generation, const std::string& stage) {
    std::lock_guard<std::mutex> lock(job_.mu);
    if (job_.generation != generation || job_.cancel_requested || job_.cancel_error) return false;
    job_.stage = stage;
    persist_job_locked("running");
    return true;
}

fs::path WebService::job_log_path(std::uint64_t generation) {
    std::lock_guard lock(job_.mu);
    if(job_.generation!=generation)throw std::runtime_error("stale task log owner");
    return tasks_?tasks_->task_directory(job_.public_id)/"live.log":
        fs::path(paths_.runs)/job_.name/"train.log";
}

WebService::TrainingOutcome WebService::run_training(std::uint64_t generation, std::string name, std::string mode, int steps,
                              int segment, int epochs, bool separate,
                              bool dereverb, bool deecho, bool denoise,
                              bool deharmony, bool vad, bool resume, bool prepared) {
    std::unique_lock<std::mutex> gpu_lock(gpu_mutex_, std::defer_lock);
    while (!gpu_lock.try_lock()) {
        std::unique_lock<std::mutex> lock(job_.mu);
        if (job_.generation != generation || job_.cancel_requested)
            return TrainingOutcome::kCancelled;
        job_.changed.wait_for(lock, std::chrono::milliseconds(20), [&] {
            return job_.generation != generation || job_.cancel_requested;
        });
    }
    if (!training_checkpoint(generation, "preparing")) return TrainingOutcome::kCancelled;
    const fs::path directory = fs::path(paths_.runs) / name;
    const bool full=mode=="joint-full";
    const fs::path full_dir=directory/"full_training";
    if(full&&!resume) {
        if(!fs::create_directory(full_dir))throw std::runtime_error("full training directory already exists");
    }
    const fs::path log = job_log_path(generation);
    TempDir staged(directory, ".train_stage_");
    const fs::path staged_root = staged.path();
    const fs::path clips = staged_root / "clips";
    const fs::path model = staged_root / "model.safetensors";
    const fs::path index = staged_root / "model.index";
    {
        std::ofstream output(log, resume ? std::ios::app : std::ios::trunc);
        output << "[serve] training '" << name << "' mode=" << mode
               << (epochs > 0 ? " epochs=" + std::to_string(epochs)
                              : " steps=" + std::to_string(steps))
               << " seg=" << segment << (separate ? " +separate" : "") << '\n';
        if (!output) {
            return TrainingOutcome::kError;
        }
    }

    // If the material was preprocessed step-by-step, slice that work set without
    // repeating separation. Otherwise slice raw material with the selected stages.
    const fs::path work = directory / "work";
    const bool preprocessed =
        nonempty_regular_file(directory / "steps.txt") && first_valid_wav(work);
    const fs::path source = preprocessed ? work : directory / "raw";
    const bool has_separation =
        separate || deharmony || dereverb || deecho || denoise;
    if (!training_checkpoint(generation, !preprocessed && has_separation ? "separating" : "preprocessing"))
        return TrainingOutcome::kCancelled;

    if(full && resume) {
        copy_training_clips(full_dir/"clips",clips);
    } else if(full && prepared) {
        copy_training_clips(source,clips);
    } else {
    ProcessSpec preprocess;
    preprocess.argv = {paths_.build + "/vc_preprocess", "--input", source.string(),
                       "--output-dir", clips.string(), "--sr", "40000",
                       "--seg-sec", "3.0"};
    if (vad) {
        preprocess.argv.insert(preprocess.argv.end(),
                               {"--vad", "--vad-dir", paths_.models + "/vad"});
    } else {
        preprocess.argv.push_back("--trim");
    }
    if (!preprocessed) {
        if (separate) preprocess.argv.push_back("--separate");
        if (deharmony) preprocess.argv.push_back("--deharmony");
        if (dereverb) preprocess.argv.push_back("--dereverb");
        if (deecho) preprocess.argv.push_back("--deecho");
        if (denoise) preprocess.argv.push_back("--denoise");
        if (has_separation) {
            preprocess.argv.insert(preprocess.argv.end(),
                                   {"--sep-dir", paths_.models + "/separation"});
        }
    }
    preprocess.log_path = log;
    preprocess.log_filter_terms = {"clip", "Wrote", "error", "skip",
                                   "separation", "reverb", "noise",
                                   "speech", "enabled"};
    enable_progress(preprocess,generation);
    if (!run_process(preprocess).success() || !first_valid_wav(clips)) {
        return TrainingOutcome::kError;
    }
    }
    if(full&&!resume) {
        // Commit a complete fixed dataset before the trainer starts. Staging
        // cleanup can never delete this copy or its checkpoints on cancellation.
        const auto fixed_staged=staged_root/"fixed_clips";
        copy_training_clips(clips,fixed_staged);fs::rename(fixed_staged,full_dir/"clips");
    }

    if (!training_checkpoint(generation, "training")) return TrainingOutcome::kCancelled;
    ProcessSpec training;
    training.argv = {paths_.build + "/vc_train"};
    if (mode == "gan" || mode == "joint" || full) {
        training.argv.insert(training.argv.end(), {"--gan", "--dmodel", paths_.dmodel});
    }
    if(mode=="joint")training.argv.insert(training.argv.end(),{"--gan-profile","joint"});
    if(full) {
        training.argv.insert(training.argv.end(),{"--gan-profile","joint-full",
            "--cache-dir",(full_dir/"cache").string(),"--checkpoint-dir",(full_dir/"checkpoints").string()});
        if(resume) {
            if(const auto saved=latest_full_checkpoint(full_dir/"checkpoints"))
                training.argv.insert(training.argv.end(),{"--resume",saved->string()});
            else std::ofstream(log,std::ios::app)<<"[serve] no completed checkpoint; restarting fixed clips from step 0\n";
        }
    }
    training.argv.insert(training.argv.end(),
                         {"--hubert", paths_.hubert, "--rmvpe", paths_.rmvpe,
                          "--pretrained", paths_.gmodel, "--target", (full?full_dir/"clips":clips).string(),
                          "--out", model.string()});
    if (epochs > 0) {
        training.argv.insert(training.argv.end(),
                             {"--epochs", std::to_string(epochs)});
    } else {
        training.argv.insert(training.argv.end(),
                             {"--steps", std::to_string(steps)});
    }
    training.argv.insert(training.argv.end(), {"--seg", std::to_string(segment)});
    training.log_path = log;
    training.log_filter_terms = {"step", "mel", "Device", "frames", "Epochs",
                                 "Exported", "clips", "done", "error", "GAN", "cancelled",
                                 "Cache", "Checkpoint", "Candidate", "seconds", "identity", "Resumed"};
    ProcessResult training_result;
    {
        const auto token = staged_root / "cancel";
        TrainingCancelLease channel(*this, generation,
            std::make_shared<training_control::TrainingCancelSource>(token.string()));
        training.argv.insert(training.argv.end(), {"--cancel-file", token.string()});
        {
            std::lock_guard<std::mutex> lock(job_.mu);
            // Covers cancellation between the stage checkpoint and registration.
            if (job_.generation != generation || job_.cancel_requested)
                return TrainingOutcome::kCancelled;
            job_.cancel_source = channel.source;
        }
        // Cancellation only changes the job channel. The synchronous runner
        // drains logs and naturally reaps the child before this lease closes.
        enable_progress(training,generation);
        training_result = run_process(training);
    }
    bool requested;
    {
        std::lock_guard<std::mutex> lock(job_.mu);
        if (job_.generation != generation || job_.cancel_error) return TrainingOutcome::kError;
        requested = job_.cancel_requested;
    }
    if (training_result.termination != ProcessTermination::kExited ||
        training_result.io_error || training_result.system_error || training_result.signal_number)
        return TrainingOutcome::kError;
    if (training_result.exit_code == static_cast<int>(training_control::TrainingResult::kCancelled))
        return requested ? TrainingOutcome::kCancelled : TrainingOutcome::kError;
    if (!training_result.success() || !nonempty_regular_file(model)) return TrainingOutcome::kError;

    if (!training_checkpoint(generation, "indexing")) return TrainingOutcome::kCancelled;
    ProcessSpec indexing;
    indexing.argv = {paths_.build + "/build_index", "--hubert", paths_.hubert,
                     "--input-dir", clips.string(), "--output", index.string()};
    indexing.log_path = log;
    indexing.log_filter_terms = {"Wrote", "index", "error", "frames"};
    enable_progress(indexing,generation);
    if (!run_process(indexing).success() ||
        !nonempty_regular_file(index)) {
        return TrainingOutcome::kError;
    }

    if (!training_checkpoint(generation, "publishing")) return TrainingOutcome::kCancelled;
    std::string publish_error;
    const auto destination=full?full_dir/"result":directory;
    fs::create_directories(destination);
    if (!replace_training_artifacts(destination, clips, model, index, publish_error)) {
        std::fprintf(stderr, "[vc_serve] training publish failed for '%s': %s\n",
                     name.c_str(), publish_error.c_str());
        return TrainingOutcome::kError;
    }
    std::ofstream(log, std::ios::app) << (full?"[serve] done; review required; candidate retained in full_training/result, existing voice unchanged.\n":"[serve] done.\n");
    return TrainingOutcome::kDone;
}

std::string WebService::voices_json() {
    std::lock_guard<std::mutex> material_lock(material_mutex_);
    std::string j = "{\"base\":\"base (pretrained)\",\"voices\":[";
    bool first = true;
    std::vector<std::string> reviewed;
    if (fs::is_directory(paths_.runs)) {
        std::vector<std::string> names;
        for (auto& e : fs::directory_iterator(paths_.runs))
            if (e.is_directory() && fs::exists(e.path() / "model.safetensors"))
                names.push_back(e.path().filename().string());
        std::sort(names.begin(), names.end());
        for (auto& n : names) {
            if (!first) j += ","; j += "\"" + json_escape(n) + "\""; first = false;
            if (regular_candidate(fs::path(paths_.runs)/n/"clips/release.json")) reviewed.push_back(n);
        }
    }
    j += "],\"reviewed_voices\":["; first = true;
    for (const auto& name : reviewed) {
        if (!first) j += ','; first = false; j += "\"" + json_escape(name) + "\"";
    }
    { std::lock_guard<std::mutex> lk(job_.mu);
      j += "],\"training_profiles\":[\"decoder\",\"gan\",\"joint\",\"joint-full\"],\"training\":{\"running\":" + std::string(job_.running ? "true" : "false") +
           ",\"name\":\"" + json_escape(job_.name) + "\",\"stage\":\"" + json_escape(job_.stage) + "\"}}"; }
    return j;
}

void WebService::prepare_training_task(const JobTicket& ticket) {
    auto r=tasks_->get(ticket.public_id);
    if(!r||r->parameters.at("mode")!="joint-full")throw std::runtime_error("missing full training recipe");
    const fs::path directory=fs::path(paths_.runs)/r->name,full=directory/"full_training";
    const bool imported=r->parameters.at("resume")=="1";
    const bool preprocessed=nonempty_regular_file(directory/"steps.txt")&&first_valid_wav(directory/"work");
    const auto source=imported?full/"clips":preprocessed?directory/"work":directory/"raw";
    const auto work=tasks_->task_directory(ticket.public_id)/"work";
    fs::create_directory(work);copy_training_clips(source,work/"source");
    fs::copy_file(paths_.gmodel,work/"generator.safetensors");
    fs::copy_file(paths_.dmodel,work/"discriminator.safetensors");
    sync_training_clips(work/"source");sync_artifact(work/"generator.safetensors");
    sync_artifact(work/"discriminator.safetensors");sync_artifact(work,true);
    if(!imported&&!fs::create_directory(full))throw std::runtime_error("full training directory already exists");
    if(fs::exists(fs::symlink_status(full/"web-task")))throw std::runtime_error("full training already belongs to a task");
    write_training_marker(full/"web-task",ticket.public_id);
    std::lock_guard lock(job_.mu);
    r=tasks_->get(ticket.public_id);
    r->parameters["recovery"]="training-full-v1";
    r->parameters["inputs_preprocessed"]=preprocessed?"1":"0";
    if(!tasks_->save(*r,r->revision))throw std::runtime_error("cannot retain fixed training recipe");
    job_.recoverable_training=true;
}

void WebService::run_training_task(JobTicket ticket) noexcept {
    TrainingOwner completion(*this);completion.generation=ticket.generation;
    try {
        const auto token=(tasks_->task_directory(ticket.public_id)/("control."+std::to_string(ticket.attempt))).string();
        TrainingCancelLease channel(*this,ticket.generation,std::make_shared<training_control::TrainingCancelSource>(token));
        {
            std::lock_guard lock(job_.mu);job_.control_path=token;job_.cancel_source=channel.source;
        }
        std::unique_lock gpu(gpu_mutex_,std::defer_lock);
        while(!gpu.try_lock()) {
            audio_boundary(ticket.generation);
            std::unique_lock lock(job_.mu);job_.changed.wait_for(lock,std::chrono::milliseconds(20));
        }
        audio_boundary(ticket.generation);
        const auto r=tasks_->get(ticket.public_id);if(!r)throw std::runtime_error("training task disappeared");
        execute_training_task(ticket,*r);
        completion.outcome=TrainingOutcome::kDone;
    }catch(const voxmutatio::io::OperationPaused&) {
        std::lock_guard lock(job_.mu);
        completion.outcome=job_.cancel_requested?TrainingOutcome::kCancelled:TrainingOutcome::kPaused;
    }catch(const std::exception& e) {
        try {std::ofstream(job_log_path(ticket.generation),std::ios::app)<<"[serve] "<<e.what()<<'\n';}catch(...){}
    }catch(...){}
}

void WebService::execute_training_task(const JobTicket& ticket,const TaskRecord& r) {
    const auto& v=r.parameters;int epochs=0,segment=0;
    if(v.at("mode")!="joint-full"||!safe_name(r.name)||
       !parse_int_field(v.at("epochs"),1,2000,epochs)||!parse_int_field(v.at("seg"),32,32,segment))
        throw std::runtime_error("invalid fixed training recipe");
    const auto flag=[&](const char* key){bool b=false;if(!parse_bool_field(v.at(key),b))throw std::runtime_error("invalid fixed training option");return b;};
    const bool imported=flag("resume"),prepared=flag("prepared"),preprocessed=flag("inputs_preprocessed");
    const bool vad=flag("vad");
    const fs::path work=tasks_->task_directory(r.id)/"work",full=fs::path(paths_.runs)/r.name/"full_training";
    if(!fs::is_directory(fs::symlink_status(full))||!fs::is_regular_file(fs::symlink_status(full/"web-task"))||
       fs::file_size(full/"web-task")>128||read_file(full/"web-task")!=r.id)
        throw std::runtime_error("training directory ownership changed");
    const auto source=work/"source",clips=full/"clips",model=full/"model.safetensors",index=full/"model.index";
    const auto log=job_log_path(ticket.generation);
    const auto boundary=[&](const char* stage){audio_boundary(ticket.generation,stage);};
    boundary("checking");
    WebWorkflowIdentity identity("web-training-full-v1");identity.field("source",training_clips_identity(source));
    // Display-only edits must never change the frozen training recipe.
    for(const auto& [key,value]:v)if(key!="resumed_from"&&key!="display_name")identity.field(key,value);
    for(const auto* program:{"vc_train","vc_preprocess","build_index"})identity.file(program,fs::path(paths_.build)/program);
    identity.file("generator",work/"generator.safetensors");identity.file("discriminator",work/"discriminator.safetensors");
    identity.file("hubert",paths_.hubert);identity.file("rmvpe",paths_.rmvpe);
    if(!imported&&!prepared) {
        if(vad) {
            identity.file("vad",fs::path(paths_.models)/"vad/silero_vad.safetensors");
        }
        if(!preprocessed)for(const auto& [option,name]:std::vector<std::pair<const char*,const char*>>{
            {"separate","vocal_roformer"},{"deharmony","karaoke_roformer"},{"dereverb","dereverb_roformer"},
            {"deecho","deecho_roformer"},{"denoise","denoise_roformer"}}) {
            if(!flag(option))continue;
            for(const auto* ext:{".safetensors",".json","_config.i64","_dim_inputs.i64","_freq_indices.i64","_num_bands_per_freq.i64"})
                identity.file(std::string(name)+ext,fs::path(paths_.models)/"separation"/(std::string(name)+ext));
        }
    }
    TaskArtifacts artifacts(full,identity.finish(),identity.legacy());
    const auto alternatives=identity.identities();
    const auto version=std::find(alternatives.begin(),alternatives.end(),artifacts.identity())-alternatives.begin();
    const auto clip_digest=[&](const fs::path& path){return training_clips_identity(path).identities().at(version);};
    boundary("preprocessing");
    if(const auto manifest=artifacts.completed("clips")) {
        if(read_file(*manifest)!=clip_digest(clips))throw std::runtime_error("fixed training clips changed");
        std::ofstream(log,std::ios::app)<<"[serve] Reusing verified fixed clips\n";
    } else {
        TempDir staged(full,".clips_stage_");const auto output=fs::path(staged.path())/"clips";
        if(imported||prepared)copy_training_clips(source,output);
        else {
            ProcessSpec process;process.argv={paths_.build+"/vc_preprocess","--input",source.string(),"--output-dir",output.string(),"--sr","40000","--seg-sec","3.0"};
            if(vad)process.argv.insert(process.argv.end(),{"--vad","--vad-dir",paths_.models+"/vad"});else process.argv.push_back("--trim");
            if(!preprocessed) {
                for(const auto* option:{"separate","deharmony","dereverb","deecho","denoise"})if(flag(option))process.argv.push_back(std::string("--")+option);
                process.argv.insert(process.argv.end(),{"--sep-dir",paths_.models+"/separation"});
            }
            process.log_path=log;enable_progress(process,ticket.generation);
            enable_audio_control(process,ticket.generation,full/"preprocess-checkpoints");
            if(!run_audio_process(process,ticket.generation))throw std::runtime_error("training material preparation failed");
        }
        const auto digest=clip_digest(output);sync_training_clips(output);
        if(fs::exists(fs::symlink_status(clips))) {
            if(clip_digest(clips)!=digest)throw std::runtime_error("existing training clips do not match completed preparation");
        } else fs::rename(output,clips);
        write_training_marker(full/"clips.sha256",digest);artifacts.commit("clips",full/"clips.sha256");
    }
    boundary("training");
    if(!artifacts.completed("training")) {
        ProcessSpec process;process.argv={paths_.build+"/vc_train","--gan","--gan-profile","joint-full",
            "--hubert",paths_.hubert,"--rmvpe",paths_.rmvpe,"--pretrained",(work/"generator.safetensors").string(),
            "--dmodel",(work/"discriminator.safetensors").string(),"--target",clips.string(),"--out",model.string(),
            "--epochs",std::to_string(epochs),"--seg",std::to_string(segment),
            "--cache-dir",(full/"cache").string(),"--checkpoint-dir",(full/"checkpoints").string()};
        if(const auto saved=latest_full_checkpoint(full/"checkpoints"))process.argv.insert(process.argv.end(),{"--resume",saved->string()});
        else {
            if(fs::is_directory(full/"checkpoints"))process.argv.push_back("--resume-initial");
            std::ofstream(log,std::ios::app)<<"[serve] No complete training checkpoint; validating unfinished initialization before starting fixed clips at step 0\n";
        }
        {std::lock_guard lock(job_.mu);process.argv.insert(process.argv.end(),{"--cancel-file",job_.control_path});}
        process.log_path=log;enable_progress(process,ticket.generation);
        if(!run_audio_process(process,ticket.generation)||!nonempty_regular_file(model))throw std::runtime_error("joint-full training failed; retained complete checkpoints are unchanged");
        artifacts.commit("training",model);
    } else std::ofstream(log,std::ios::app)<<"[serve] Reusing verified trained generator\n";
    // The CLI's periodic exports need not include the last epoch (e.g. a
    // one-epoch validation). Every completed Web run still needs a selectable
    // final candidate. Copy the verified output, never an optimizer checkpoint.
    const auto epoch=std::to_string(epochs);
    const auto final_candidate=full/"checkpoints"/("candidate_epoch_"+std::string(10-epoch.size(),'0')+epoch+".safetensors");
    if(!artifacts.completed("final-candidate")) {
        fs::create_directories(full/"checkpoints");
        if(!fs::is_directory(fs::symlink_status(full/"checkpoints")))throw std::runtime_error("invalid candidate directory");
        if(fs::exists(fs::symlink_status(final_candidate))) {
            if(voxmutatio::io::sha256_file(final_candidate)!=voxmutatio::io::sha256_file(model))
                throw std::runtime_error("final candidate differs from completed generator");
        } else {
            TempDir staged(full/"checkpoints",".candidate_stage_");const fs::path copy=fs::path(staged.path())/"model.safetensors";
            fs::copy_file(model,copy);sync_artifact(copy);fs::rename(copy,final_candidate);sync_artifact(full/"checkpoints",true);
        }
        artifacts.commit("final-candidate",final_candidate);
    }
    boundary("indexing");
    if(!artifacts.completed("indexing")) {
        ProcessSpec process;process.argv={paths_.build+"/build_index","--hubert",paths_.hubert,"--input-dir",clips.string(),"--output",index.string()};
        process.log_path=log;enable_progress(process,ticket.generation);
        if(!run_process(process).success()||!nonempty_regular_file(index))throw std::runtime_error("index building failed; trained generator retained");
        artifacts.commit("indexing",index);
    }
    boundary("publishing");
    // Publication takes copies: the verified completed stages remain reusable if
    // installation or the terminal task-record write fails.
    TempDir staged(full,".result_stage_");const fs::path output=staged.path();
    copy_training_clips(clips,output/"clips");fs::copy_file(model,output/"model.safetensors");fs::copy_file(index,output/"model.index");
    sync_training_clips(output/"clips");sync_artifact(output/"model.safetensors");sync_artifact(output/"model.index");
    fs::create_directories(full/"result");std::string error;
    if(!replace_training_artifacts(full/"result",output/"clips",output/"model.safetensors",output/"model.index",error))throw std::runtime_error("training result installation failed: "+error);
    sync_artifact(full/"result",true);sync_artifact(full,true);
    std::ofstream(log,std::ios::app)<<"[serve] done; review required; candidate retained in full_training/result\n";
}

void WebService::handle_training_runs(int fd) {
    std::string json = "{\"runs\":[";
    bool first = true;
    std::error_code ec;
    std::vector<fs::path> runs;
    for (fs::directory_iterator it(paths_.runs, ec), end; !ec && it != end; it.increment(ec))
        if (safe_name(it->path().filename().string()) && fs::is_directory(it->path()/"full_training"))
            runs.push_back(it->path());
    std::sort(runs.begin(), runs.end());
    for (const auto& run : runs) {
        const auto full = run/"full_training";
        std::vector<std::string> candidates;
        std::error_code scan_error;
        for (fs::directory_iterator it(full/"checkpoints", scan_error), end;
             !scan_error && it != end; it.increment(scan_error)) {
            const auto name = it->path().filename().string();
            if (candidate_name(name) && regular_candidate(it->path())) candidates.push_back(name);
        }
        std::sort(candidates.begin(), candidates.end());
        std::string task_id;
        // A managed run must lead back to its original task, even if no full
        // optimizer checkpoint has been committed yet. Bound the marker read.
        const int marker=::open((full/"web-task").c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);
        if(marker>=0) {
            struct stat st{};std::array<char,97> bytes{};ssize_t count=-1;
            if(::fstat(marker,&st)==0&&S_ISREG(st.st_mode)&&st.st_nlink==1&&st.st_size>=8&&st.st_size<=96) {
                count=::read(marker,bytes.data(),bytes.size());
            }
            ::close(marker);
            if(count>=8&&count==st.st_size) {std::string id(bytes.data(),static_cast<std::size_t>(count));if(TaskStore::valid_id(id))task_id=std::move(id);}
        }
        if (!first) json += ','; first = false;
        json += "{\"name\":\"" + json_escape(run.filename().string()) + "\",\"task_id\":\""+json_escape(task_id)+"\",\"resumable\":" +
            (latest_full_checkpoint(full/"checkpoints") && first_valid_wav(full/"clips") ? "true" : "false") +
            ",\"result_ready\":" + (regular_candidate(full/"result/model.safetensors") &&
               regular_candidate(full/"result/model.index") ? "true" : "false") + ",\"candidates\":[";
        bool first_candidate = true;
        for (const auto& name : candidates) {
            if (!first_candidate) json += ','; first_candidate = false;
            json += "\"" + json_escape(name) + "\"";
        }
        json += "]}";
    }
    send_json(fd, json + "]}");
}

void WebService::handle_publish(int fd, const Request& request) {
    std::vector<Part> parts;
    if (!request_parts(fd, request, parts)) return;
    const auto run = field(parts, "name", ""), candidate = field(parts, "candidate", "");
    const auto name = field(parts, "publish_name", "");
    if (!safe_name(run) || !safe_name(name) || !candidate_name(candidate) ||
        field(parts, "reviewed", "0") != "1") {
        send_response(fd, 400, "Bad Request", "text/plain", "请选择候选、填写新声线名称并确认听审通过"); return;
    }
    std::unique_lock gpu_lock(gpu_mutex_, std::try_to_lock);
    if (!gpu_lock.owns_lock() || job_.running) {
        send_response(fd, 409, "Conflict", "text/plain", "GPU 工作中，请结束后发布"); return;
    }
    const auto source = fs::path(paths_.runs)/run/"full_training";
    const auto model = source/"checkpoints"/candidate, index = source/"result/model.index";
    if (!regular_candidate(model) || !regular_candidate(index)) {
        send_response(fd, 404, "Not Found", "text/plain", "候选或训练索引尚未就绪"); return;
    }
    std::lock_guard material_lock(material_mutex_);
    const auto destination = fs::path(paths_.runs)/name;
    for (const auto* artifact : {"model.safetensors", "model.index", "clips"}) {
        if (fs::exists(fs::symlink_status(destination/artifact))) {
            send_response(fd, 409, "Conflict", "text/plain", "声线已存在，请使用新名称；已有文件不会被覆盖"); return;
        }
    }
    fs::create_directories(destination);
    TempDir staging(destination, ".release_");
    const fs::path staged(staging.path());
    fs::create_directory(staged/"clips");
    fs::copy_file(model, staged/"model.safetensors");
    fs::copy_file(index, staged/"model.index");
    std::ofstream record(staged/"clips/release.json");
    record << "{\"reviewed\":true,\"run\":\"" << json_escape(run)
           << "\",\"candidate\":\"" << json_escape(candidate) << "\"}\n";
    record.close();
    std::string error;
    if (!record || !replace_training_artifacts(destination, staged/"clips", staged/"model.safetensors",
                                               staged/"model.index", error)) {
        send_response(fd, 500, "Internal Server Error", "text/plain", "发布失败，旧声线保留"); return;
    }
    send_json(fd, "{\"published\":\"" + json_escape(name) + "\"}");
}

// Installed separation models grouped by stage, for the WebUI per-stage picker.
std::string WebService::separation_models_json() const {
    auto models = voxmutatio::separation::list_sep_models(paths_.models + "/separation");
    std::string j = "{";
    bool first_task = true;
    for (const auto& task : voxmutatio::separation::sep_tasks()) {
        if (!first_task) j += ",";
        first_task = false;
        j += "\"" + json_escape(task) + "\":[";
        bool first = true;
        for (const auto& m : models) {
            if (m.task != task) continue;
            if (!first) j += ",";
            first = false;
            j += "{\"name\":\"" + json_escape(m.name) + "\",\"label\":\"" + json_escape(m.label) + "\"}";
        }
        j += "]";
    }
    j += "}";
    return j;
}

void WebService::run_step(std::uint64_t generation, std::string name, std::string step, std::string model,
              std::unique_lock<std::mutex> gpu_lk) {
    if (!gpu_lk.owns_lock()) throw std::logic_error("step worker does not own GPU reservation");
    const fs::path dir = fs::path(paths_.runs) / name;
    const fs::path work = dir / "work";
    TempDir staged(dir, ".step_stage_");
    const fs::path staged_root = staged.path();
    const fs::path staged_work = staged_root / "work";
    const fs::path staged_steps = staged_root / "steps.txt";
    const fs::path log = job_log_path(generation);
    std::ofstream(log, std::ios::app) << "[serve] preprocess step '" << step << "'"
        << (model.empty() ? "" : " model '" + model + "'") << " on '" << name << "'\n";
    set_stage(generation, "step:" + step);
    const std::string flag =
        step == "separate" ? "--separate"
        : step == "dereverb" ? "--dereverb"
        : step == "deecho" ? "--deecho"
        : step == "deharmony" ? "--deharmony"
                               : "--denoise";
    ProcessSpec process;
    process.argv = {paths_.build + "/vc_preprocess", "--input", work.string(),
                    "--output-dir", staged_work.string(), "--sr", "44100",
                    "--no-slice", "--preserve-stereo", flag};
    if (!model.empty()) {
        process.argv.insert(process.argv.end(),
                            {"--" + step + "-model", model});
    }
    process.argv.insert(process.argv.end(),
                        {"--sep-dir", paths_.models + "/separation"});
    process.log_path = log;
    process.log_filter_terms = {"processed", "error", "enabled"};
    enable_progress(process,generation);
    const ProcessResult process_result = run_process(process);
    if (!process_result.success() || !first_valid_wav(staged_work)) {
        finish(generation, "error", true);
        return;
    }
    const fs::path steps = dir / "steps.txt";
    std::error_code marker_error;
    const bool has_steps = fs::exists(steps, marker_error);
    if (marker_error) {
        finish(generation, "error", true);
        return;
    }
    if (has_steps) {
        const bool regular = fs::is_regular_file(steps, marker_error);
        const std::uintmax_t size =
            regular && !marker_error ? fs::file_size(steps, marker_error) : 0;
        if (!regular || marker_error || step.size() + 1 > kMaximumStepHistoryBytes ||
            size > kMaximumStepHistoryBytes - step.size() - 1) {
            finish(generation, "error", true);
            return;
        }
        fs::copy_file(steps, staged_steps, fs::copy_options::none, marker_error);
        if (marker_error) {
            finish(generation, "error", true);
            return;
        }
    }
    {
        std::ofstream marker(staged_steps, std::ios::app);
        marker << step << '\n';
        marker.close();
        if (!marker) {
            finish(generation, "error", true);
            return;
        }
    }
    std::string error;
    bool committed = false;
    {
        std::lock_guard<std::mutex> material_lock(material_mutex_);
        committed = replace_step_artifacts(dir, staged_work, staged_steps, error);
    }
    if (!committed) {
        std::fprintf(stderr, "[vc_serve] step publish failed for '%s': %s\n",
                     name.c_str(), error.c_str());
        finish(generation, "error", true);
        return;
    }
    finish(generation, "step-done", false);
}

// POST /api/material : multipart {name, files[]} -> store raw + init work set.
bool WebService::stage_uploads(int fd, const std::vector<Upload>& uploads,
                              const fs::path& raw, const std::optional<fs::path>& work, bool stereo) {
    const auto result = normalize_uploads(uploads, raw, work, paths_, *process_runner_, decode_mutex_, stereo);
    if (result.status == 200) return true;
    const char* reason = result.status == 400 ? "Bad Request" : result.status == 409 ? "Conflict" :
                         result.status == 413 ? "Payload Too Large" :
                         result.status == 503 ? "Service Unavailable" : "Internal Server Error";
    send_response(fd, result.status, reason, "application/json", "{\"error\":\"" + json_escape(result.error) + "\"}");
    return false;
}

void WebService::handle_material(int fd, const Request& req) {
    std::vector<Part> parts;
    if (!request_parts(fd, req, parts)) return;
    std::string name = field(parts, "name", "");
    if (!safe_name(name)) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid name\"}"); return; }
    std::vector<Upload> uploads;
    if (!collect_uploads(fd, parts, uploads)) return;
    if (uploads.empty()) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"no audio files\"}"); return; }

    fs::path dir = fs::path(paths_.runs) / name;
    fs::create_directories(dir);
    TempDir staged(dir, ".material_upload_");
    fs::path staged_raw = fs::path(staged.path()) / "raw";
    fs::path staged_work = fs::path(staged.path()) / "work";
    std::string error;
    bool stereo = false;
    if (!parse_bool_field(field(parts, "preserve_stereo", "0"), stereo)) {
        send_response(fd, 400, "Bad Request", "text/plain", "invalid preserve_stereo"); return;
    }
    if (!stage_uploads(fd, uploads, staged_raw, staged_work, stereo)) return;

    std::unique_lock<std::mutex> gpu_lk(gpu_mutex_, std::try_to_lock);
    if (!gpu_lk.owns_lock() || job_.running) {
        send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"workflow busy; upload not committed\"}");
        return;
    }
    bool committed = false;
    {
        std::lock_guard<std::mutex> material_lock(material_mutex_);
        committed = replace_material_dirs(dir, staged_raw, staged_work, error);
    }
    if (!committed) {
        send_response(fd, 500, "Internal Server Error", "application/json",
                      "{\"error\":\"upload commit failed\"}");
        return;
    }
    send_json(fd, "{\"name\":\"" + json_escape(name) + "\",\"files\":" + std::to_string(uploads.size()) + "}");
}

// POST /api/step : multipart {name, step} -> apply one separation step to the work set.
void WebService::handle_step(int fd, const Request& req) {
    if (job_.running) { send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"busy\"}"); return; }
    std::vector<Part> parts;
    if (!request_parts(fd, req, parts)) return;
    std::string name = field(parts, "name", ""), step = field(parts, "step", "");
    if (!safe_name(name) || !fs::exists(paths_.runs + "/" + name + "/work")) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"upload material first\"}"); return; }
    if (step != "separate" && step != "dereverb" && step != "deecho" && step != "denoise" && step != "deharmony") { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"unknown step\"}"); return; }
    std::string model = field(parts, "model", "");
    if (!model.empty() &&
        (!safe_name(model) ||
         !voxmutatio::separation::sep_model_available(paths_.models + "/separation", step, model))) {
        send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"unknown model\"}"); return;
    }
    JobTicket ticket;
    try {
        if (!reserve_job(name, JobKind::kStep, ticket)) {
            send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"busy\"}"); return;
        }
    } catch (const WorkflowError& e) {
        send_response(fd,e.status,"Conflict","application/json","{\"error\":\""+json_escape(e.what())+"\"}");return;
    } catch (const std::exception&) {
        send_response(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"cannot prepare job\"}"); return;
    }
    std::future<bool> gpu_ready_result;
    try {
        // Every fallible preparation after reservation must release the slot on
        // failure, including promise allocation before a worker exists.
        std::promise<bool> gpu_ready;
        gpu_ready_result = gpu_ready.get_future();
        const std::shared_ptr<WebService> self = shared_from_this();
        std::thread worker([self, generation = ticket.generation, name, step, model, ready = std::move(gpu_ready)]() mutable {
            bool readiness_reported = false;
            try {
                std::unique_lock<std::mutex> gpu_lk(self->gpu_mutex_, std::try_to_lock);
                if (!gpu_lk.owns_lock()) {
                    self->finish(generation, "error", true);
                    ready.set_value(false);
                    return;
                }
                {
                    std::lock_guard<std::mutex> lk(self->job_.mu);
                    self->job_.name = name;
                    self->job_.stage = "step:" + step;
                    self->job_.done = false;
                    self->job_.error = false;
                }
                ready.set_value(true);
                readiness_reported = true;
                self->run_step(generation, name, step, model, std::move(gpu_lk));
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[vc_serve] preprocessing exception: %s\n", e.what());
                if (!readiness_reported) {
                    self->finish(generation, "error", true);
                    try { ready.set_exception(std::current_exception()); } catch (...) {}
                } else {
                    self->finish(generation, "error", true);
                }
            } catch (...) {
                std::fprintf(stderr, "[vc_serve] preprocessing exception\n");
                if (!readiness_reported) {
                    self->finish(generation, "error", true);
                    try { ready.set_exception(std::current_exception()); } catch (...) {}
                } else {
                    self->finish(generation, "error", true);
                }
            }
        });
        worker.detach();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[vc_serve] cannot start preprocessing worker: %s\n", e.what());
        finish(ticket.generation, "error", true);
        send_response(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"cannot start worker\"}");
        return;
    }
    try {
        if (!gpu_ready_result.get()) {
            send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"GPU busy\"}");
            return;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[vc_serve] cannot reserve GPU for preprocessing: %s\n", e.what());
        send_response(fd, 503, "Service Unavailable", "application/json",
                      "{\"error\":\"cannot start worker\"}");
        return;
    }
    send_json(fd, "{\"step\":\"" + json_escape(step) + "\",\"job_id\":\"" + ticket.public_id + "\"}");
}

// GET /api/preview?name=X : return the first work-set file as WAV for listening.
void WebService::handle_preview(int fd, const Request& req) {
    std::string name = query_get(req.query, "name");
    if (!safe_name(name)) { send_response(fd, 400, "Bad Request", "text/plain", "bad name"); return; }
    std::unique_lock<std::mutex> material_lock(material_mutex_);
    std::string work = paths_.runs + "/" + name + "/work";
    if (!fs::exists(work)) { material_lock.unlock(); send_response(fd, 404, "Not Found", "text/plain", "no material"); return; }
    const std::optional<fs::path> first = first_valid_wav(work);
    if (!first) { material_lock.unlock(); send_response(fd, 404, "Not Found", "text/plain", "empty"); return; }
    TempDir preview_temp("/tmp", "resonantia-preview-");
    std::string preview_path = first->string();
    if (exceeds_buffered_response_limit(*first)) {
        std::error_code error;
        const auto size = fs::file_size(*first, error);
        const auto frames = !error && size <= kMaxStereoUploadBytes
            ? decoded_wav_frames(*first, size, true) : std::nullopt;
        if (frames && *frames <= kMaxStereoUploadFrames) {
            // Snapshot under the material lock; send in bounded blocks after
            // unlocking so replacement and slow clients cannot mix generations.
            const auto snapshot = fs::path(preview_temp.path()) / "preview.wav";
            fs::copy_file(*first, snapshot, fs::copy_options::none, error);
            material_lock.unlock();
            if (error) {
                send_response(fd, 500, "Internal Server Error", "text/plain", "preview snapshot failed");
                return;
            }
            send_audio_file(fd, snapshot, req.range);
            return;
        }
        auto stereo = voxmutatio::io::read_stereo_audio(preview_path);
        if (stereo && 44 + stereo->frames() * 4 <= kMaxStereoUploadBytes) {
            const auto compact = preview_temp.path() + "/preview.wav";
            if (voxmutatio::io::write_stereo_audio(compact, stereo->data.data(),
                    static_cast<int>(stereo->frames()), 44100)) {
                if (exceeds_buffered_response_limit(compact)) {
                    stereo.reset();
                    material_lock.unlock();
                    send_audio_file(fd, compact, req.range);
                    return;
                }
                preview_path = compact;
            }
        }
    }
    if (exceeds_buffered_response_limit(preview_path)) {
        material_lock.unlock();
        send_response(fd, 413, "Payload Too Large", "application/json",
                      "{\"error\":\"audio response exceeds 64 MiB\"}");
        return;
    }
    const std::string wav = read_file(preview_path);
    material_lock.unlock();
    if (wav.empty()) {
        send_response(fd, 500, "Internal Server Error", "text/plain",
                      "preview read failed");
        return;
    }
    send_response(fd, 200, "OK", "audio/wav", wav);
}

void WebService::handle_train(int fd, const Request& req) {
    if (job_.running) { send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"a training job is already running\"}"); return; }
    std::vector<Part> parts;
    if (!request_parts(fd, req, parts)) return;
    const auto task_id=field(parts,"task_id","");
    if(!task_id.empty()) {
        if(!tasks_||!TaskStore::valid_id(task_id)||!safe_name(task_id)) {send_response(fd,400,"Bad Request","application/json","{\"error\":\"invalid persistent training task ID (8..64 safe characters)\"}");return;}
        if(tasks_->get(task_id)) {send_response(fd,409,"Conflict","application/json","{\"error\":\"task already exists\",\"task_id\":\""+task_id+"\"}");return;}
    }

    std::string name = field(parts, "name", ""), mode = field(parts, "mode", "decoder");
    int steps = 0, seg = 0, epochs = 0;
    bool separate = false, dereverb = false, deecho = false, denoise = false;
    bool deharmony = false, vad = false, resume = false, prepared = false;
    if (!safe_name(name)) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid voice name (letters, digits, _ or - only)\"}"); return; }
    const bool full=mode=="joint-full";
    if (mode != "decoder" && mode != "gan" && mode != "joint" && !full) {
        send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"invalid training mode\"}"); return;
    }
    if (!parse_bool_field(field(parts, "separate", "0"), separate) ||
        !parse_bool_field(field(parts, "dereverb", "0"), dereverb) ||
        !parse_bool_field(field(parts, "deecho", "0"), deecho) ||
        !parse_bool_field(field(parts, "denoise", "0"), denoise) ||
        !parse_bool_field(field(parts, "deharmony", "0"), deharmony) ||
        !parse_bool_field(field(parts, "vad", "0"), vad) ||
        !parse_bool_field(field(parts, "resume", "0"), resume) ||
        !parse_bool_field(field(parts, "prepared", "0"), prepared)) {
        send_response(fd, 400, "Bad Request", "application/json",
                      "{\"error\":\"boolean parameters must be 0 or 1\"}");
        return;
    }
    if (!parse_int_field(field(parts, "steps", mode=="joint"?"50":"300"), 1, 100000, steps) ||
        !parse_int_field(field(parts, "seg", mode=="joint"||full?"32":"40"), 8, 200, seg) ||
        !parse_int_field(field(parts, "epochs", full?"200":"0"), 0, 2000, epochs)) {
        send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"training parameter out of range\"}"); return;
    }
    if(mode=="joint" && (steps>50 || seg!=32 || epochs!=0)) {
        send_response(fd,400,"Bad Request","application/json",
                      "{\"error\":\"joint training requires 1..50 steps, seg 32 and epochs 0\"}");return;
    }
    if((!full&&(resume||prepared)) || (full&&(epochs<1||seg!=32||!field(parts,"steps","").empty())) ||
       ((resume||prepared)&&(separate||dereverb||deecho||denoise||deharmony||vad)) || (resume&&prepared)) {
        send_response(fd,400,"Bad Request","application/json",
            "{\"error\":\"joint-full requires epochs >0, seg 32, no steps; resume/prepared are full-only and cannot combine with cleanup or each other\"}");return;
    }

    std::vector<Upload> uploads;
    if (!collect_uploads(fd, parts, uploads)) return;
    int nfiles = static_cast<int>(uploads.size());
    fs::path voice_dir = fs::path(paths_.runs) / name;
    const auto full_dir=voice_dir/"full_training";
    if(full&&resume&&fs::exists(fs::symlink_status(full_dir/"web-task"))) {
        send_response(fd,409,"Conflict","application/json","{\"error\":\"此训练由任务中心管理，请从原任务继续\"}");return;
    }
    if(full && !resume && fs::exists(full_dir)) {
        send_response(fd,409,"Conflict","application/json","{\"error\":\"full training state already exists; resume it or use a new name\"}");return;
    }
    if(resume && (nfiles || fs::is_symlink(full_dir) || fs::is_symlink(full_dir/"clips") || !first_valid_wav(full_dir/"clips"))) {
        send_response(fd,400,"Bad Request","application/json","{\"error\":\"resume requires retained full training clips and no uploads\"}");return;
    }
    fs::create_directories(voice_dir);
    bool has_work = fs::exists(paths_.runs + "/" + name + "/work") && !fs::is_empty(paths_.runs + "/" + name + "/work");
    if (nfiles == 0 && !has_work && !resume) { send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"no audio files uploaded\"}"); return; }

    TrainingOwner owner(*this); // Declared first: staging retires before the reservation.
    std::optional<TempDir> staged;
    fs::path staged_raw, staged_work;
    if (!uploads.empty()) {
        staged.emplace(voice_dir, ".train_upload_");
        staged_raw = fs::path(staged->path()) / "raw";
        staged_work = fs::path(staged->path()) / "work";
        if (!stage_uploads(fd, uploads, staged_raw, staged_work)) return;
    }

    JobTicket ticket;
    try {
        if (!reserve_job(name, JobKind::kTraining, ticket,task_id)) {
            send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"a training job is already running\"}"); return;
        }
        owner.generation = ticket.generation;
        if(tasks_) {
            std::lock_guard lock(job_.mu);
            auto r=tasks_->get(ticket.public_id);
            // A name may have been edited after admission, before preparation.
            r->parameters.insert({{"mode",mode},{"steps",std::to_string(steps)},{"seg",std::to_string(seg)},
                {"epochs",std::to_string(epochs)},{"separate",separate?"1":"0"},
                {"dereverb",dereverb?"1":"0"},{"deecho",deecho?"1":"0"},{"denoise",denoise?"1":"0"},
                {"deharmony",deharmony?"1":"0"},{"vad",vad?"1":"0"},
                {"resume",resume?"1":"0"},{"prepared",prepared?"1":"0"}});
            if(!tasks_->save(*r,r->revision))throw std::runtime_error("cannot retain training parameters");
        }
    } catch (const WorkflowError& e) {
        send_response(fd,e.status,"Conflict","application/json","{\"error\":\""+json_escape(e.what())+"\"}");return;
    } catch (const std::exception&) {
        send_response(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"cannot prepare job\"}"); return;
    }
    const auto accepted = [&] {
#ifdef RESONANTIA_TEST_JOB_OWNERSHIP
        test_training_response_boundary(ticket.generation, true);
#endif
        send_json(fd, "{\"job\":\"" + json_escape(name) + "\",\"job_id\":\"" + ticket.public_id +
                      "\",\"files\":" + std::to_string(nfiles) + "}");
    };
    if (staged) {
        std::unique_lock<std::mutex> gpu_lk(gpu_mutex_, std::try_to_lock);
        if (!gpu_lk.owns_lock()) {
            send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"GPU busy; upload not committed\"}");
            return;
        }
        std::string error;
        bool committed = false, cancelled_before_commit = false;
        try {
            std::lock_guard<std::mutex> material_lock(material_mutex_);
            cancelled_before_commit = !training_checkpoint(ticket.generation, "material-commit");
            if (!cancelled_before_commit)
                committed = replace_material_dirs(voice_dir, staged_raw, staged_work, error);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[vc_serve] upload commit exception: %s\n", e.what());
        }
        if (cancelled_before_commit) {
            gpu_lk.unlock();
            owner.outcome = TrainingOutcome::kCancelled; accepted(); return;
        }
        if (!committed) {
            // The request owner retires the reservation after upload staging.
#ifdef RESONANTIA_TEST_JOB_OWNERSHIP
            test_training_response_boundary(ticket.generation, false);
#endif
            send_response(fd, 500, "Internal Server Error", "application/json", "{\"error\":\"upload commit failed\"}");
            return;
        }
    }
    // Retire request-owned upload staging before handing the reservation to a
    // worker that can complete before the acceptance response is written.
    staged.reset();
    if (cancellation_requested(ticket.generation)) {
        owner.outcome = TrainingOutcome::kCancelled; accepted(); return;
    }
    if(tasks_&&full) {
        try {
            prepare_training_task(ticket);set_stage(ticket.generation,"queued");
            auto self=shared_from_this();std::thread worker([self,ticket]{self->run_training_task(ticket);});
            owner.release();worker.detach();accepted();return;
        }catch(const std::exception& e) {
            std::fprintf(stderr,"[vc_serve] cannot prepare persistent training: %s\n",e.what());
            send_response(fd,503,"Service Unavailable","application/json","{\"error\":\"cannot prepare persistent training; inspect task history\"}");return;
        }
    }
    set_stage(ticket.generation, "queued");
    try {
        const std::shared_ptr<WebService> self = shared_from_this();
        std::thread worker([self, generation = ticket.generation, name, mode, steps, seg, epochs, separate, dereverb,
                            deecho, denoise, deharmony, vad, resume, prepared] {
            TrainingOwner completion(*self); completion.generation = generation;
            try {
                completion.outcome = self->run_training(generation, name, mode, steps, seg, epochs, separate, dereverb,
                                   deecho, denoise, deharmony, vad, resume, prepared);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[vc_serve] training exception: %s\n", e.what());
            } catch (...) {
                std::fprintf(stderr, "[vc_serve] training exception\n");
            }
        });
        owner.release(); // The successfully constructed worker owns finalization now.
        worker.detach();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[vc_serve] cannot start training worker: %s\n", e.what());
        send_response(fd, 503, "Service Unavailable", "application/json", "{\"error\":\"cannot start worker\"}");
        return;
    }
    accepted();
}

void WebService::handle_status(int fd, const Request& req) {
    std::string qn = query_get(req.query, "name");
    std::string stage, jname, id, kind, state; bool running, done, error, requested, cancelled, cancellable;
    {
        std::lock_guard<std::mutex> lock(job_.mu);
        stage = job_.stage; jname = job_.name; id = job_.public_id;
        running = job_.running; done = job_.done; error = job_.error;
        requested = job_.cancel_requested; cancelled = job_.cancelled;
        kind = job_.kind == JobKind::kTraining ? "train" : job_.kind == JobKind::kStep ? "step" :
            job_.kind==JobKind::kConversion?"convert":job_.kind==JobKind::kPreprocess?"preprocess":"none";
        cancellable = running && job_.kind == JobKind::kTraining && stage != "publishing";
        state = !running ? (stage=="paused"?"paused":cancelled ? "cancelled" : error ? "error" : done ? "done" : "idle") :
                requested ? "cancel_requested" : stage == "queued" ? "queued" :
                stage == "publishing" ? "publishing" : "running";
    }
    std::string which = qn.empty() ? jname : qn, logtail;
    if(tasks_&&which==jname&&TaskStore::valid_id(id))
        logtail=log_tail((tasks_->task_directory(id)/(running?"live.log":"task.log")).string(),4000);
    else if (safe_name(which)) logtail = log_tail(paths_.runs + "/" + which + "/train.log", 4000);
    const auto full_dir=fs::path(paths_.runs)/which/"full_training";
    const bool resumable=safe_name(which)&&first_valid_wav(full_dir/"clips").has_value()&&
        latest_full_checkpoint(full_dir/"checkpoints").has_value();
    const bool review_required=safe_name(which)&&nonempty_regular_file(full_dir/"result/model.safetensors");
    send_json(fd, std::string("{\"name\":\"") + json_escape(jname) + "\",\"stage\":\"" + json_escape(stage) +
        "\",\"job_id\":\"" + id + "\",\"kind\":\"" + kind + "\",\"state\":\"" + state +
        "\",\"running\":" + (running ? "true" : "false") + ",\"done\":" + (done ? "true" : "false") +
        ",\"error\":" + (error ? "true" : "false") + ",\"cancel_requested\":" + (requested ? "true" : "false") +
        ",\"cancelled\":" + (cancelled ? "true" : "false") + ",\"cancellable\":" + (cancellable ? "true" : "false") +
        ",\"resumable\":"+(resumable?"true":"false")+",\"review_required\":"+(review_required?"true":"false")+
        ",\"cancel_boundary\":\"stage_or_iteration\",\"log\":\"" + json_escape(logtail) + "\"}");
}

void WebService::handle_cancel_train(int fd, const Request& req) {
    std::vector<Part> parts;
    if (!request_parts(fd, req, parts)) return;
    if (parts.size() != 1 || parts[0].name != "job_id" || !parts[0].filename.empty() ||
        !safe_name(parts[0].data)) {
        send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"exactly one valid job_id is required\"}"); return;
    }
    const std::string id = parts[0].data;
    int code = 202; const char* failure = nullptr; bool stopped = false;
    std::shared_ptr<training_control::TrainingCancelSource> channel;
    {
        std::lock_guard<std::mutex> lock(job_.mu);
        if (job_.public_id.empty()) { code = 404; failure = "no job"; }
        else if (id != job_.public_id) { code = 409; failure = "job id does not match current job"; }
        else if (job_.kind != JobKind::kTraining) { code = 409; failure = "only training jobs can be cancelled"; }
        else if (job_.recoverable_training && job_.attempt > 1) { code = 409; failure = "resumed training requires task controls with an execution attempt"; }
        else if (job_.cancelled && !job_.running) { code = 200; stopped = true; }
        else if (!job_.running || job_.stage == "publishing") { code = 409; failure = "too late: publication started or job already ended"; }
        else {
            code = job_.cancel_requested ? 200 : 202; job_.cancel_requested = true;
            channel = job_.cancel_source;
        }
    }
    if (failure) {
        send_response(fd, code, code == 404 ? "Not Found" : "Conflict", "application/json",
                      std::string("{\"error\":\"") + failure + "\"}"); return;
    }
    job_.changed.notify_all();
    if (channel && channel->request() == training_control::CancelDelivery::kError) {
        send_response(fd, 500, "Internal Server Error", "application/json",
                      "{\"error\":\"cancellation channel failed; waiting for current stage to finish\"}");
        return;
    }
    send_response(fd, code, code == 202 ? "Accepted" : "OK", "application/json",
                  "{\"job_id\":\"" + id + "\",\"cancel_requested\":true,\"cancelled\":" +
                  (stopped ? "true" : "false") + ",\"state\":\"" +
                  (stopped ? "cancelled" : "cancel_requested") + "\"}");
}

// Apply selected SOTA input preprocessing (vocal/deharmony/dereverb/denoise) to
// `in`, writing a cleaned full-length WAV path to `cleaned`. cleaned=in if none.
bool WebService::preprocess_audio_file(const std::string& in, const std::string& temp_dir, bool sep,
                           bool deharm, bool derev, bool deecho, bool denoise,
                           std::string& cleaned, std::string& log,std::uint64_t generation) {
    if (!(sep || deharm || derev || deecho || denoise)) { cleaned = in; return true; }
    std::string outdir = temp_dir + "/preprocessed";
    log = temp_dir + "/preprocess.log";
    fs::create_directories(outdir);
    ProcessSpec process;
    process.argv = {paths_.build + "/vc_preprocess", "--input", in,
                    "--output-dir", outdir, "--sr", "44100", "--no-slice", "--preserve-stereo"};
    if (sep) process.argv.push_back("--separate");
    if (deharm) process.argv.push_back("--deharmony");
    if (derev) process.argv.push_back("--dereverb");
    if (deecho) process.argv.push_back("--deecho");
    if (denoise) process.argv.push_back("--denoise");
    process.argv.insert(process.argv.end(),
                        {"--sep-dir", paths_.models + "/separation"});
    process.log_path = log;
    enable_progress(process,generation);
    enable_audio_control(process,generation,fs::path(temp_dir)/"blocks");
    if (!run_audio_process(process,generation)) return false;
    const std::optional<fs::path> output = first_valid_wav(outdir);
    if (!output) return false;
    cleaned = output->string();
    return true;
}

// POST /api/preprocess : multipart {audio, pp_*} -> cleaned WAV (preview, no conversion).
void WebService::handle_preprocess(int fd, const Request& req) {
    if(job_.running) {
        send_response(fd,409,"Conflict","application/json","{\"error\":\"another background task is active\"}");return;
    }
    std::vector<Part> parts;
    if (!request_parts(fd, req, parts)) return;
    std::vector<Upload> uploads;
    if (!collect_uploads(fd, parts, uploads)) return;
    if (uploads.size() != 1) { send_response(fd, 400, "Bad Request", "text/plain", "exactly one audio file required"); return; }
    std::unique_lock<std::mutex> gpu_lk(gpu_mutex_, std::try_to_lock);
    if (!gpu_lk.owns_lock()) { send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"GPU 忙（正在训练或转换），请稍候重试\"}"); return; }
    TempDir temp("/tmp", "resonantia-preprocess-");
    const fs::path upload_dir = fs::path(temp.path()) / "upload";
    std::string in = (upload_dir / uploads.front().filename).string(), cleaned, log;
    bool sep = false, deharm = false, derev = false, denoise = false, deecho = false;
    if (!parse_bool_field(field(parts, "pp_separate", "0"), sep) ||
        !parse_bool_field(field(parts, "pp_deharmony", "0"), deharm) ||
        !parse_bool_field(field(parts, "pp_dereverb", "0"), derev) ||
        !parse_bool_field(field(parts, "pp_denoise", "0"), denoise) ||
        !parse_bool_field(field(parts, "pp_deecho", "0"), deecho)) {
        send_response(fd, 400, "Bad Request", "application/json",
                      "{\"error\":\"boolean parameters must be 0 or 1\"}");
        return;
    }
    if (!stage_uploads(fd, uploads, upload_dir)) return;
    if (!preprocess_audio_file(in, temp.path(), sep, deharm, derev, deecho, denoise, cleaned, log)) {
        send_response(fd, 500, "Internal Server Error", "text/plain", "预处理失败：" + log_tail(log)); return; }
    if (exceeds_buffered_response_limit(cleaned)) {
        send_response(fd, 413, "Payload Too Large", "application/json",
                      "{\"error\":\"audio response exceeds 64 MiB\"}");
        return;
    }
    const std::string wav = read_file(cleaned);
    if (wav.empty()) {
        send_response(fd, 500, "Internal Server Error", "text/plain",
                      "preprocess output read failed");
        return;
    }
    send_response(fd, 200, "OK", "audio/wav", wav);
}

void WebService::handle_convert(int fd, const Request& req) {
    std::vector<Part> parts;
    if (!request_parts(fd, req, parts)) return;
    const bool background=req.path=="/api/tasks/convert"||req.path=="/api/tasks/preprocess";
    const bool preprocess_only=req.path=="/api/tasks/preprocess";
    const auto task_id=field(parts,"task_id","");
    if(background) {
        if(!tasks_||task_storage_failed_) {
            send_response(fd,503,"Service Unavailable","application/json","{\"error\":\"task storage unavailable\"}");return;
        }
        if(!TaskStore::valid_id(task_id)) {
            send_response(fd,400,"Bad Request","application/json","{\"error\":\"a stable task_id is required\"}");return;
        }
        if(tasks_->get(task_id)) {
            send_response(fd,409,"Conflict","application/json","{\"error\":\"task already exists; query its state\",\"task_id\":\""+task_id+"\"}");return;
        }
    }
    if(job_.running) {
        send_response(fd,409,"Conflict","application/json","{\"error\":\"another background task is active\"}");return;
    }
    const auto source_task=field(parts,"source_task","");
    std::optional<TaskRecord> parent;
    if(!source_task.empty()) {
        if(!background||!TaskStore::valid_id(source_task)||!field(parts,"session","").empty()||
           std::any_of(parts.begin(),parts.end(),[](const Part& part){return !part.filename.empty();})) {
            send_response(fd,400,"Bad Request","application/json","{\"error\":\"source_task requires a background task and cannot be combined with upload/session\"}");return;
        }
        parent=tasks_->get(source_task);
        if(!parent){send_response(fd,404,"Not Found","application/json","{\"error\":\"source task not found\"}");return;}
        if(parent->state!="succeeded"||(parent->kind!="convert"&&parent->kind!="preprocess")||
           !parent->parameters.contains("recovery")||parent->parameters.at("recovery")!="audio-v1") {
            send_response(fd,409,"Conflict","application/json","{\"error\":\"only completed audio tasks can create a revision\"}");return;
        }
        for(const auto* key:{"voice","workflow","output_kind","pitch_edits","f0_up_key","index_rate",
            "rms_mix_rate","protect","filter_radius","pp_separate","pp_deharmony","pp_dereverb","pp_deecho","pp_denoise"}) {
            if(parent->parameters.contains(key)&&std::none_of(parts.begin(),parts.end(),[&](const Part& part){return part.name==key;}))
                parts.push_back({key,"",parent->parameters.at(key)});
        }
    }
    std::vector<Upload> uploads;
    if (!collect_uploads(fd, parts, uploads)) return;
    if (uploads.size() > 1) { send_response(fd, 400, "Bad Request", "text/plain", "at most one audio file allowed"); return; }
    const Part* audio = uploads.empty() ? nullptr : uploads.front().part;
    std::string input_name=audio?uploads.front().filename:"audio";

    // Preprocessed input may come from a cascade work-set (session) instead of a fresh upload.
    std::string session = field(parts, "session", "");
    std::string conv_in;
    if (!session.empty() && !safe_name(session)) {
        send_response(fd, 400, "Bad Request", "text/plain", "bad session"); return;
    }
    const std::string workflow = field(parts, "workflow", "manual");
    const bool song = workflow == "song";
    if (workflow != "manual" && workflow != "dry" && workflow != "harmonies" && !song) {
        send_response(fd, 400, "Bad Request", "text/plain", "invalid workflow"); return;
    }
    const std::string output_kind = field(parts, "output_kind", "vocal");
    if ((output_kind != "vocal" && output_kind != "mix") || (output_kind == "mix" && !song)) {
        send_response(fd, 400, "Bad Request", "text/plain", "mix output requires song workflow"); return;
    }
    std::unique_lock<std::mutex> gpu_lk(gpu_mutex_, std::try_to_lock);
    if (!gpu_lk.owns_lock()) { send_response(fd, 409, "Conflict", "application/json", "{\"error\":\"GPU 忙（正在训练或转换），请稍候重试\"}"); return; }
    TempDir temp("/tmp", "resonantia-convert-");
    if (!session.empty()) {
        std::lock_guard<std::mutex> material_lock(material_mutex_);
        const auto input = first_valid_wav(fs::path(paths_.runs) / session /
                                          (workflow == "manual" ? "work" : "raw"));
        if (input) {
            input_name=input->filename().string();
            conv_in = temp.path() + "/source.wav";
            fs::copy_file(*input, conv_in);
        }
    }
    if(parent) {
        conv_in=(tasks_->task_directory(source_task)/"work/source.wav").string();input_name=parent->name;
    }
    if (conv_in.empty() && !audio) { send_response(fd, 400, "Bad Request", "text/plain", "no audio"); return; }

    std::string voice = field(parts, "voice", "base");
    const auto pitch_edits = field(parts, "pitch_edits", "");
    std::vector<voxmutatio::PitchEdit> validated_edits;
    if (!voxmutatio::parse_pitch_edits(pitch_edits, validated_edits)) {
        send_response(fd, 400, "Bad Request", "text/plain", "局部修音需按时间排序且不重叠；每段 0.02–2 秒，最多 16 段"); return;
    }
    int pitch = 0, filter_radius = 0;
    double index_rate = 0.0, rms = 0.0, protect = 0.0;
    if (!parse_int_field(field(parts, "f0_up_key", "0"), -48, 48, pitch) ||
        !parse_double_field(field(parts, "index_rate", "0"), 0.0, 1.0, index_rate) ||
        !parse_double_field(field(parts, "rms_mix_rate", "0.5"), 0.0, 1.0, rms) ||
        !parse_double_field(field(parts, "protect", "0.5"), 0.0, 1.0, protect) ||
        !parse_int_field(field(parts, "filter_radius", "3"), 0, 31, filter_radius) ||
        (filter_radius != 0 && (filter_radius < 3 || filter_radius % 2 == 0))) {
        send_response(fd, 400, "Bad Request", "application/json", "{\"error\":\"conversion parameter out of range\"}");
        return;
    }
    bool pp_sep = false, pp_deharm = false, pp_derev = false;
    bool pp_denoise = false, pp_deecho = false;
    if (!parse_bool_field(field(parts, "pp_separate", "0"), pp_sep) ||
        !parse_bool_field(field(parts, "pp_deharmony", "0"), pp_deharm) ||
        !parse_bool_field(field(parts, "pp_dereverb", "0"), pp_derev) ||
        !parse_bool_field(field(parts, "pp_denoise", "0"), pp_denoise) ||
        !parse_bool_field(field(parts, "pp_deecho", "0"), pp_deecho)) {
        send_response(fd, 400, "Bad Request", "application/json",
                      "{\"error\":\"boolean parameters must be 0 or 1\"}");
        return;
    }

    std::string model = paths_.gmodel, index;
    const auto candidate_run = field(parts, "training_run", "");
    const auto candidate = field(parts, "candidate", "");
    if (!candidate_run.empty() || !candidate.empty()) {
        if (!safe_name(candidate_run) || !candidate_name(candidate)) {
            send_response(fd, 400, "Bad Request", "text/plain", "invalid training candidate"); return;
        }
        const auto full = fs::path(paths_.runs)/candidate_run/"full_training";
        model = (full/"checkpoints"/candidate).string();
        if (!regular_candidate(model)) {
            send_response(fd, 404, "Not Found", "text/plain", "candidate not found"); return;
        }
        if (regular_candidate(full/"result/model.index")) index = (full/"result/model.index").string();
    } else if(parent&&parent->kind=="convert"&&voice==parent->parameters.at("voice")) {
        auto saved=restore_audio_plan(*parent);resolve_saved_audio_model(saved);model=saved.model;
        if(nonempty_regular_file(saved.index))index=saved.index;
        else if(safe_name(voice)&&nonempty_regular_file(fs::path(paths_.runs)/voice/"model.index"))index=(fs::path(paths_.runs)/voice/"model.index").string();
    } else if (voice != "base" && voice != "base (pretrained)") {
        if (!safe_name(voice)) { send_response(fd, 400, "Bad Request", "text/plain", "bad voice"); return; }
        std::string vm = paths_.runs + "/" + voice + "/model.safetensors";
        if (!nonempty_regular_file(vm)) {
            send_response(fd, 404, "Not Found", "application/json",
                          "{\"error\":\"voice model not found\"}");
            return;
        }
        model = vm;
        std::string vi = paths_.runs + "/" + voice + "/model.index";
        if (nonempty_regular_file(vi)) index = vi;
    }
    if(parent&&!preprocess_only&&voice==parent->parameters.at("voice")&&candidate_run.empty()&&candidate.empty()&&parent->kind=="convert") {
        auto saved=restore_audio_plan(*parent);resolve_saved_audio_model(saved);model=saved.model;
        if(nonempty_regular_file(saved.index))index=saved.index;
        else if(index_rate>0&&parent->parameters.at("index_rate")=="0.000000"&&
                voxmutatio::io::sha256_file(model)!=voxmutatio::io::sha256_file(voice=="base"||voice=="base (pretrained)"?paths_.gmodel:paths_.runs+"/"+voice+"/model.safetensors")) {
            send_response(fd,409,"Conflict","application/json","{\"error\":\"原版本未保存可验证的索引，请选择声线或保持检索率为0\"}");return;
        }
    }
    if (index_rate > 0.0 && index.empty()) {
        send_response(fd, 400, "Bad Request", "application/json",
                      "{\"error\":\"index_rate > 0 requires a compatible voice index\"}");
        return;
    }

    std::string out = temp.path() + "/output.wav";
    std::string preprocess_log;
    if (conv_in.empty()) {
        const fs::path upload_dir = fs::path(temp.path()) / "upload";
        if (!stage_uploads(fd, uploads, upload_dir, std::nullopt, workflow != "manual")) return;
        conv_in = (upload_dir / uploads.front().filename).string();
    }
    ConversionPlan plan;
    plan.directory=temp.path();plan.input=conv_in;plan.name=input_name;plan.workflow=workflow;plan.output_kind=output_kind;
    plan.voice=voice;plan.model=model;plan.index=index;plan.pitch_edits=pitch_edits;
    plan.pitch=pitch;plan.filter_radius=filter_radius;plan.index_rate=index_rate;plan.rms=rms;plan.protect=protect;
    plan.separate=pp_sep;plan.deharmony=pp_deharm;plan.dereverb=pp_derev;plan.deecho=pp_deecho;plan.denoise=pp_denoise;
    plan.preprocess_only=preprocess_only;plan.source_task=source_task;
    if(background) {start_audio_task(fd,std::move(plan),task_id);return;}
    try {
        const auto result=convert_audio(plan);
        if(exceeds_buffered_response_limit(result))throw WorkflowError(413,"audio response exceeds 64 MiB");
        auto bytes=read_file(result);
        if(bytes.empty())throw WorkflowError(500,"转换失败：output read failed");
        send_response(fd,200,"OK","audio/wav",bytes);
    } catch(const WorkflowError& e) {
        send_response(fd,e.status,e.status==413?"Payload Too Large":"Internal Server Error","text/plain",e.what());
    }
}

std::string WebService::convert_audio(const ConversionPlan& p,std::uint64_t generation,TaskArtifacts* artifacts) {
    const auto stage=[&](const std::string& value){if(generation)audio_boundary(generation,value);};
    const auto cached=[&](const std::string& key)->std::optional<fs::path> {
        if(!artifacts)return {};
        auto file=artifacts->completed(key);
        if(file){std::ofstream log(fs::path(p.directory)/"stages.log",std::ios::app);log<<"[serve] reused completed "<<key<<'\n';if(!log)throw std::runtime_error("cannot retain recovery log");}
        return file;
    };
    const auto commit=[&](const std::string& key,const std::string& file){if(artifacts)artifacts->commit(key,file);};
    const bool song=p.workflow=="song";
    const auto original=p.input;auto input=original;
    std::string vocals,log;
    if(song) {
        stage("separating");
        if(auto saved=cached("separation"))vocals=saved->string();
        else {
            if(!preprocess_audio_file(original,p.directory+"/vocals",true,false,false,false,false,vocals,log,generation))
                throw WorkflowError(500,"分离人声失败："+log_tail(log));
            commit("separation",vocals);
        }
        input=vocals;
    }
    stage("cleaning");
    const auto before=input;
    if(auto saved=cached("cleanup"))input=saved->string();
    else {
        if(!preprocess_audio_file(before,p.directory+"/cleanup",p.workflow=="manual"&&p.separate,
            song||p.workflow=="harmonies"||p.deharmony,p.dereverb,p.deecho,p.denoise,input,log,generation))
            throw WorkflowError(500,"主唱清理失败："+log_tail(log));
        commit("cleanup",input);
    }
    if(p.preprocess_only)return input;
    if(input!=original) {
        stage("preparing-vocal");
        auto lead=voxmutatio::io::read_audio(input,40000);
        if(!lead||lead->data.empty())throw WorkflowError(500,"清理后的人声文件无效");
        if(lead->data.size()>40000u*600)throw WorkflowError(413,"清理后的人声超过 10 分钟");
        float peak=0;for(float value:lead->data)peak=std::max(peak,std::abs(value));
        if(peak>0.99f)for(float& value:lead->data)value*=0.99f/peak;
        input=p.directory+"/lead40.wav";
        if(!voxmutatio::io::write_audio(input,lead->data.data(),static_cast<int>(lead->data.size()),40000))
            throw WorkflowError(500,"主唱保存失败");
    }
    stage("converting");
    std::string output=p.directory+"/output.wav";
    if(p.pitch_preview&&artifacts&&!artifacts->completed("conversion")) {
        stage("preview-checking");
        const auto parent=tasks_->get(p.source_task);
        if(!parent||parent->state!="succeeded"||pitch_preview_task(*parent))throw WorkflowError(409,"试听基线不可用");
        auto previous=restore_audio_plan(*parent);resolve_saved_audio_model(previous);
        std::optional<fs::path> baseline;
        if(previous.output_kind=="vocal"&&parent->parameters.contains("retention")&&parent->parameters.contains("result_sha256")) {
            const auto file=tasks_->task_directory(parent->id)/parent->result;
            if(voxmutatio::io::sha256_file(file)==parent->parameters.at("result_sha256"))baseline=file;
        }else baseline=TaskArtifacts::retained_output(previous.directory,"conversion");
        if(!baseline)throw WorkflowError(409,"基线干净人声未保留，不能制作对照试听");
        if(!artifacts->completed("baseline")) {
            const auto excerpt=p.directory+"/baseline.wav";
            pitch_review::write_excerpt(*baseline,excerpt,p.preview_start,p.preview_end);artifacts->commit("baseline",excerpt);
        }
        std::string pause_file;{std::lock_guard lock(job_.mu);pause_file=job_.control_path;}
        const auto cores=pitch_review::seed_preview(previous.directory,fs::path(p.directory)/"conversion_blocks",input,
            fs::path(paths_.build)/"vc_convert",conversion_config(previous),conversion_config(p),p.preview_start,p.preview_end,pause_file,
            [&]{audio_boundary(generation);});
        std::ofstream preview_log(fs::path(p.directory)/"stages.log",std::ios::app);
        preview_log<<"[serve] pitch preview "<<p.preview_start<<".."<<p.preview_end<<" seconds; recompute "<<cores<<" synthesis cores; reuse all F0 and unaffected cores\n";
        if(!preview_log)throw std::runtime_error("cannot retain preview log");
        stage("converting");
    }
    if(auto saved=cached("conversion"))output=saved->string();
    else {
    ProcessSpec process;
    process.argv={paths_.build+"/vc_convert","--hubert",paths_.hubert,"--rmvpe",paths_.rmvpe,
        "--model",p.model,"--input",input,"--output",output,"--version","v2","--speakers","109",
        "--sr","40000","--pitch",std::to_string(p.pitch),"--rms-mix",std::to_string(p.rms),
        "--protect",std::to_string(p.protect),"--filter-radius",std::to_string(p.filter_radius)};
    if(p.index_rate>0)process.argv.insert(process.argv.end(),{"--index",p.index,"--index-rate",std::to_string(p.index_rate)});
    if(!p.pitch_edits.empty())process.argv.insert(process.argv.end(),{"--pitch-edits",p.pitch_edits});
    process.log_path=p.directory+"/convert.log";
    const auto start=std::chrono::steady_clock::now();
    enable_progress(process,generation);
    enable_audio_control(process,generation,fs::path(p.directory)/"conversion_blocks");
    if(!run_audio_process(process,generation)||!valid_wav_file(output))
        throw WorkflowError(500,"转换失败："+log_tail(process.log_path));
    const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::fprintf(stderr,"[vc_serve] convert voice=%s child %.3f ms\n%s\n",p.voice.c_str(),elapsed,
                 log_tail(process.log_path,2048).c_str());
    commit("conversion",output);
    }
    if(p.output_kind=="mix") {
        stage("mixing");
        const auto mixed=p.directory+"/cover.wav";
        if(auto saved=cached("mixing"))output=saved->string();
        else {
            if(!mix_cover(original,vocals,output,mixed))throw WorkflowError(500,"伴奏合成失败：音频长度或格式不匹配");
            output=mixed;commit("mixing",output);
        }
    }
    return output;
}

void WebService::start_audio_task(int fd,ConversionPlan plan,const std::string& task_id) {
    JobTicket ticket;bool reserved=false;
    try {
        if(!reserve_job(plan.name,plan.preprocess_only?JobKind::kPreprocess:JobKind::kConversion,ticket,task_id)) {
            send_response(fd,409,"Conflict","application/json","{\"error\":\"another task is active\"}");return;
        }
        reserved=true;
        const auto directory=tasks_->task_directory(task_id)/"work";fs::create_directory(directory);
        fs::copy_file(plan.input,directory/"source.wav");
        plan.input=(directory/"source.wav").string();plan.directory=directory.string();
        // The published voice can later be replaced atomically. This copy belongs
        // to the task so that its execution never follows a newer voice model.
        if(!plan.preprocess_only) {
            fs::copy_file(plan.model,directory/"model.safetensors");plan.model=(directory/"model.safetensors").string();
            if(plan.index_rate>0) {fs::copy_file(plan.index,directory/"model.index");plan.index=(directory/"model.index").string();}
        }
        for(const auto& entry:fs::directory_iterator(directory))sync_artifact(entry.path());
        sync_artifact(directory,true);
        {
            std::lock_guard lock(job_.mu);auto r=tasks_->get(task_id);
            // Keep display edits accepted while copying the task's inputs.
            r->parameters.insert({{"recovery",plan.pitch_preview?"audio-pitch-preview-v1":"audio-v1"},{"voice",plan.voice},{"workflow",plan.workflow},{"output_kind",plan.output_kind},
                {"f0_up_key",std::to_string(plan.pitch)},{"filter_radius",std::to_string(plan.filter_radius)},
                {"index_rate",std::to_string(plan.index_rate)},{"rms_mix_rate",std::to_string(plan.rms)},
                {"protect",std::to_string(plan.protect)},{"pitch_edits",plan.pitch_edits},
                {"pp_separate",plan.separate?"1":"0"},{"pp_deharmony",plan.deharmony?"1":"0"},
                {"pp_dereverb",plan.dereverb?"1":"0"},{"pp_deecho",plan.deecho?"1":"0"},
                {"pp_denoise",plan.denoise?"1":"0"}});
            if(!plan.source_task.empty())r->parameters["source_task"]=plan.source_task;
            if(plan.pitch_preview) {r->parameters["analysis_id"]=plan.analysis_id;r->parameters["preview_start"]=std::to_string(plan.preview_start);r->parameters["preview_end"]=std::to_string(plan.preview_end);}
            if(!tasks_->save(*r,r->revision))throw std::runtime_error("cannot retain conversion parameters");
        }
        auto self=shared_from_this();
        std::thread worker([self,ticket,plan=std::move(plan)]() mutable {self->run_audio_task(ticket,std::move(plan));});
        worker.detach();reserved=false;
        send_response(fd,202,"Accepted","application/json","{\"task_id\":\""+task_id+"\"}");
    } catch(const WorkflowError& e) {
        if(reserved)finish_audio_task(ticket,"",e.what());
        send_response(fd,e.status,"Conflict","application/json","{\"error\":\""+json_escape(e.what())+"\"}");
    } catch(const std::exception& e) {
        if(reserved)finish_audio_task(ticket,"",e.what());
        send_response(fd,503,"Service Unavailable","application/json","{\"error\":\"cannot prepare background task; inspect task history\"}");
    }
}
void WebService::run_audio_task(JobTicket ticket,ConversionPlan plan) noexcept {
    try {
        {
            const auto token=(tasks_->task_directory(ticket.public_id)/("control."+std::to_string(ticket.attempt))).string();
            TrainingCancelLease channel(*this,ticket.generation,std::make_shared<training_control::TrainingCancelSource>(token));
            {
                std::lock_guard lock(job_.mu);job_.control_path=token;job_.cancel_source=channel.source;
            }
            std::unique_lock gpu(gpu_mutex_,std::defer_lock);
            while(!gpu.try_lock()) {
                audio_boundary(ticket.generation);
                std::unique_lock lock(job_.mu);job_.changed.wait_for(lock,std::chrono::milliseconds(20));
            }
            audio_boundary(ticket.generation,"checking");
            const auto identity=audio_identity(plan);
            TaskArtifacts artifacts(plan.directory,identity.finish(),identity.legacy());
            inherit_audio_stages(ticket,plan,artifacts);
            audio_boundary(ticket.generation);
            auto output=convert_audio(plan,ticket.generation,&artifacts);
            if(plan.pitch_preview) {
                audio_boundary(ticket.generation,"preview");
                if(const auto cached=artifacts.completed("preview"))output=cached->string();
                else {const auto excerpt=plan.directory+"/preview.wav";pitch_review::write_excerpt(output,excerpt,plan.preview_start,plan.preview_end);artifacts.commit("preview",excerpt);output=excerpt;}
            }
            audio_boundary(ticket.generation,"saving");
            artifacts.publish_result(output);
        }
        finish_audio_task(ticket,"work/result.wav","");
    } catch(const voxmutatio::io::OperationPaused&){finish_audio_task(ticket,"","",true);}
      catch(const std::exception& e){finish_audio_task(ticket,"",e.what());}
      catch(...){finish_audio_task(ticket,"","unknown conversion failure");}
}
void WebService::finish_audio_task(const JobTicket& ticket,std::string_view result,std::string_view error,bool paused) noexcept {
    std::lock_guard lock(job_.mu);
    if(job_.generation!=ticket.generation)return;
    try {
        auto r=tasks_->get(ticket.public_id);
        if(!r)throw std::runtime_error("missing audio task record");
        std::string log;
        for(const auto* name:{"stages.log","vocals/preprocess.log","cleanup/preprocess.log","convert.log"})
            log+=log_tail((tasks_->task_directory(r->id)/"work"/name).string(),20000);
        if(!error.empty()){log.push_back('\n');log.append(error.data(),std::min<std::size_t>(error.size(),4096));}
        snapshot_log(tasks_->task_directory(r->id),log.substr(log.size()>65536?log.size()-65536:0));
        const bool failed=job_.cancel_error||!error.empty();
        if(job_.cancel_error){paused=false;r->error="pause control I/O failed";}
        else r->error=paused&&!failed?job_.storage_notice:std::string(error.substr(0,4096));
        r->state=failed?"failed":paused?"paused":"succeeded";r->result=result;
        if(r->state=="succeeded")r->parameters["result_sha256"]=voxmutatio::io::sha256_file(tasks_->task_directory(r->id)/r->result);
        r->updated_ms=wall_time_ms();r->finished_ms=paused?0:r->updated_ms;
        if(!tasks_->save(*r,r->revision))throw std::runtime_error("cannot retain audio task result");
    } catch(const std::exception& e) {
        task_storage_failed_=true;std::fprintf(stderr,"[vc_serve] cannot retain audio task: %s\n",e.what());
    } catch(...){task_storage_failed_=true;}
    job_.done=true;job_.error=!error.empty()||job_.cancel_error||task_storage_failed_;job_.running=false;
}

const char* WebService::allowed_api_method(const std::string& path) {
    if (path == "/api/task/pitch") return "GET";
    if (path == "/api/tasks/pitch-preview") return "POST";
    if (path == "/api/task/rename") return "POST";
    if (path == "/api/tasks" || path == "/api/task" || path == "/api/task/log" || path == "/api/task/result" || path == "/api/task/source") return "GET";
    if (path == "/api/train/runs" || path == "/api/storage") return "GET";
    if (path == "/api/task/pause" || path == "/api/task/resume" || path == "/api/task/cleanup") return "POST";
    if (path == "/api/train/publish" || path == "/api/tasks/convert" || path == "/api/tasks/preprocess") return "POST";
    if (path == "/api/voices" || path == "/api/sep-models" ||
        path == "/api/train/status" || path == "/api/preview") {
        return "GET";
    }
    if (path == "/api/train" || path == "/api/train/cancel" || path == "/api/material" || path == "/api/step" ||
        path == "/api/preprocess" || path == "/api/convert") {
        return "POST";
    }
    return nullptr;
}

void WebService::handle_request(int fd, const Request& request) {
    if (request.method == "GET" && request.path == "/api/task/pitch") {
        handle_pitch_report(fd,request);
    } else if (request.method == "POST" && request.path == "/api/tasks/pitch-preview") {
        handle_pitch_preview(fd,request);
    } else if (request.method == "POST" && request.path == "/api/task/rename") {
        handle_task_rename(fd,request);
    } else if (request.method == "GET" && (request.path == "/api/tasks" || request.path == "/api/task" || request.path == "/api/task/log" || request.path == "/api/task/result" || request.path == "/api/task/source")) {
        handle_tasks(fd,request);
    } else if (request.method == "GET" && request.path == "/api/storage") {
        handle_storage(fd);
    } else if (request.method == "POST" && (request.path == "/api/task/pause" || request.path == "/api/task/resume" || request.path == "/api/task/cleanup")) {
        handle_task_control(fd,request);
    } else if (request.method == "GET" && request.path == "/api/train/runs") {
        handle_training_runs(fd);
    } else if (request.method == "POST" && request.path == "/api/train/publish") {
        handle_publish(fd, request);
    } else if (request.method == "GET" && request.path == "/api/voices") {
        send_json(fd, voices_json());
    } else if (request.method == "GET" && request.path == "/api/sep-models") {
        send_json(fd, separation_models_json());
    } else if (request.method == "GET" && request.path == "/api/train/status") {
        handle_status(fd, request);
    } else if (request.method == "POST" && request.path == "/api/train/cancel") {
        handle_cancel_train(fd, request);
    } else if (request.method == "POST" && request.path == "/api/train") {
        handle_train(fd, request);
    } else if (request.method == "POST" && request.path == "/api/material") {
        handle_material(fd, request);
    } else if (request.method == "POST" && request.path == "/api/step") {
        handle_step(fd, request);
    } else if (request.method == "GET" && request.path == "/api/preview") {
        handle_preview(fd, request);
    } else if (request.method == "POST" && request.path == "/api/preprocess") {
        handle_preprocess(fd, request);
    } else if (request.method == "POST" && (request.path == "/api/convert" || request.path == "/api/tasks/convert" || request.path == "/api/tasks/preprocess")) {
        handle_convert(fd, request);
    } else if (request.path.rfind("/api/", 0) == 0) {
        if (const char* allowed = allowed_api_method(request.path)) {
            send_response(fd, 405, "Method Not Allowed", "application/json",
                          "{\"error\":\"method not allowed\"}",
                          "Allow: " + std::string(allowed) + "\r\n");
        } else {
            send_response(fd, 404, "Not Found", "application/json",
                          "{\"error\":\"unknown API endpoint\"}");
        }
    } else if (request.method == "GET") {
        serve_static(fd, paths_.webroot, request.path);
    } else {
        send_response(fd, 405, "Method Not Allowed", "text/plain", "no", "Allow: GET\r\n");
    }
}

}  // namespace resonantia::web
