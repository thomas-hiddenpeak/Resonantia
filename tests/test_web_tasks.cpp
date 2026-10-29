// Service lifecycle contracts; synthetic runner never executes models or CUDA.
#include "tools/web/web_service.h"
#include "tools/web/task_store.h"
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <array>
#include <future>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>

using namespace resonantia::web;
namespace fs=std::filesystem;
using namespace std::chrono_literals;
namespace {
void check(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
void write(const fs::path& p,const std::string& s){fs::create_directories(p.parent_path());std::ofstream f(p);f<<s;check(bool(f),"fixture write");}
std::string arg(const ProcessSpec& s,const std::string& key){for(size_t i=1;i+1<s.argv.size();++i)if(s.argv[i]==key)return s.argv[i+1];throw std::runtime_error("missing argument "+key);}
struct BlockingRunner:ProcessRunner {
    mutable std::mutex mu;mutable std::condition_variable cv;
    mutable bool entered=false;bool released=false,fail=false;
    ProcessResult run(const ProcessSpec& s) const override {
        const auto program=fs::path(s.argv[0]).filename().string();
        if(program=="vc_preprocess")write(fs::path(arg(s,"--output-dir"))/"clip.wav","RIFFxxxxWAVEfixture");
        else if(program=="vc_train") {
            std::unique_lock lock(mu);entered=true;cv.notify_all();
            check(cv.wait_for(lock,3s,[&]{return released;}),"test did not release worker");
            write(s.log_path,"Step 1/2 loss=1\nretained test log\n");
            if(fail)return {ProcessTermination::kExited,1,0,0,0};
            write(arg(s,"--out"),"fixture model");
        } else if(program=="build_index")write(arg(s,"--output"),"fixture index");
        else throw std::runtime_error("unexpected fixture program");
        return {ProcessTermination::kExited,0,0,0,0};
    }
    void wait(){std::unique_lock lock(mu);check(cv.wait_for(lock,3s,[&]{return entered;}),"worker did not start");}
    void release(){std::lock_guard lock(mu);released=true;cv.notify_all();}
};
struct Reply{int status;std::string body;};
Reply call(WebService& service,const std::string& method,const std::string& path,const std::string& query="",const std::map<std::string,std::string>& fields={},const std::string& range="",bool disconnected=false){
    Request r;r.method=method;r.path=path;r.query=query;r.range=range;
    if(method=="POST"){
        r.content_type="multipart/form-data; boundary=tasks";
        for(const auto& [k,v]:fields)r.body+="--tasks\r\nContent-Disposition: form-data; name=\""+k+"\"\r\n\r\n"+v+"\r\n";
        r.body+="--tasks--\r\n";
    }
    int fd[2];check(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,fd)==0,"socketpair");
    const timeval timeout{3,0};for(int f:fd)setsockopt(f,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof timeout);
    if(disconnected){close(fd[1]);service.handle_request(fd[0],r);close(fd[0]);return {0,{}};}
    service.handle_request(fd[0],r);close(fd[0]);std::string wire;char b[4096];
    for(;;){auto n=recv(fd[1],b,sizeof b,0);if(n==0)break;check(n>0,"response timeout");wire.append(b,n);check(wire.size()<65536,"response bound");}close(fd[1]);
    auto p=wire.find("\r\n\r\n");check(p!=wire.npos,"HTTP framing");return {std::stoi(wire.substr(9,3)),wire.substr(p+4)};
}
std::string json_string(const std::string& s,const std::string& key){auto p=s.find("\""+key+"\":\"");check(p!=s.npos,"missing "+key+" in "+s);p+=key.size()+4;return s.substr(p,s.find('"',p)-p);}
void settled(std::shared_ptr<WebService>& s){auto until=std::chrono::steady_clock::now()+3s;while(s.use_count()!=1&&std::chrono::steady_clock::now()<until)std::this_thread::sleep_for(1ms);check(s.use_count()==1,"worker lifetime");}
const std::string wav=[] {
    std::string s="RIFF";auto le=[&](unsigned v,int n){while(n--){s+=char(v&255);v>>=8;}};
    le(44,4);s+="WAVEfmt ";le(16,4);le(1,2);le(1,2);le(40000,4);le(80000,4);le(2,2);le(16,2);
    s+="data";le(8,4);for(int i=0;i<4;++i)le(i*10,2);return s;
}();
struct AudioRunner:ProcessRunner {
    mutable std::mutex mu;mutable std::condition_variable cv;mutable bool entered=false;
    bool released=false,fail=false;mutable int calls=0;mutable std::string captured_input;
    ProcessResult run(const ProcessSpec& s) const override {
        std::unique_lock lock(mu);++calls;
        if(fs::path(s.argv[0]).filename()=="vc_preprocess") {
            write(fs::path(arg(s,"--output-dir"))/"result.wav",wav);return {ProcessTermination::kExited,0,0,0,0};
        }
        entered=true;captured_input=read_file(arg(s,"--input"));
        if(s.observe_line)s.observe_line("RESONANTIA_PROGRESS 1 synthesis 16000 32000 samples16k");
        cv.notify_all();
        check(cv.wait_for(lock,3s,[&]{return released;}),"audio worker not released");
        write(s.log_path,"retained conversion failure details\n");
        write(arg(s,"--output"),wav);
        return {ProcessTermination::kExited,fail?1:0,0,0,0};
    }
    void wait(){std::unique_lock lock(mu);check(cv.wait_for(lock,3s,[&]{return entered;}),"audio worker did not start");}
    void release(){std::lock_guard lock(mu);released=true;cv.notify_all();}
};
void audio_assets(Paths& p,const fs::path& root) {
    p.models=(root/"models").string();p.hubert=(root/"models/hubert").string();p.rmvpe=(root/"models/rmvpe").string();
    for(const auto& path:{p.hubert,p.rmvpe,p.build+"/vc_convert",p.build+"/vc_preprocess"})write(path,"fixed fixture dependency");
    for(const auto* model:{"vocal_roformer","karaoke_roformer","dereverb_roformer"})
        for(const auto* ext:{".safetensors",".json","_config.i64","_dim_inputs.i64","_freq_indices.i64","_num_bands_per_freq.i64"})
            write(fs::path(p.models)/"separation"/(std::string(model)+ext),"fixed separation fixture");
}
void audio_revisions(const fs::path& root) {
    Paths p;p.runs=(root/"runs").string();p.tasks=(root/"tasks").string();p.build=(root/"build").string();
    p.gmodel=(root/"models/base").string();write(p.gmodel,"fixed model");audio_assets(p,root);
    write(root/"runs/session/raw/source.wav",wav);
    auto runner=std::make_shared<AudioRunner>();runner->release();auto service=std::make_shared<WebService>(p,runner);
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","revision-parent"},{"session","session"},{"workflow","song"}}).status==202,"parent accepted");settled(service);
    const auto parent=call(*service,"GET","/api/task","id=revision-parent");
    check(parent.body.find("succeeded")!=parent.body.npos,"parent completed");check(runner->calls==3,"parent pipeline");
    auto revised=call(*service,"POST","/api/tasks/convert","",{{"task_id","revision-pitch"},{"source_task","revision-parent"},{"f0_up_key","-2"}});
    check(revised.status==202,"completed task revision rejected: "+revised.body);settled(service);
    auto state=call(*service,"GET","/api/task","id=revision-pitch");
    check(state.body.find("succeeded")!=state.body.npos,"revision failed: "+state.body);
    check(runner->calls==4,"pitch change repeated separation/cleanup");
    check(state.body.find("\"source_task\":\"revision-parent\"")!=state.body.npos,"parent relation not saved");
    check(call(*service,"GET","/api/task","id=revision-parent").body==parent.body,"revision mutated parent record");
    check(call(*service,"GET","/api/task/source","id=revision-parent").body==wav,"original unavailable without upload");
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","revision-pitch"},{"source_task","revision-parent"}}).status==409,"duplicate revision executed twice");
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","revision-mixed"},{"source_task","revision-pitch"},{"output_kind","mix"}}).status==202,"mix revision accepted");settled(service);
    state=call(*service,"GET","/api/task","id=revision-mixed");check(state.body.find("succeeded")!=state.body.npos,"remix failed: "+state.body);
    check(runner->calls==4,"output change repeated model work");
    check(state.body.find("\"outputs\":[")!=state.body.npos&&state.body.find("\"kind\":\"vocal\"")!=state.body.npos&&state.body.find("\"kind\":\"mix\"")!=state.body.npos,"both audio results must be discoverable");
    check(call(*service,"GET","/api/task/result","id=revision-mixed&variant=vocal").body==wav,"vocal endpoint returned mix instead of converted vocal");
    check(call(*service,"GET","/api/task/result","id=revision-mixed&variant=vocal",{},"bytes=0-7").body==wav.substr(0,8),"vocal range");
    check(call(*service,"GET","/api/task/result","id=revision-mixed&variant=mix").body==call(*service,"GET","/api/task/result","id=revision-mixed").body,"default result compatibility");
    check(call(*service,"GET","/api/task/result","id=revision-parent&variant=mix").status==404,"vocal-only result mislabeled mix");
    check(call(*service,"GET","/api/task/result","id=revision-mixed&variant=../../source").status==400,"invalid variant accepted");
    check(call(*service,"POST","/api/task/cleanup","",{{"id","revision-mixed"},{"attempt","1"}}).status==200,"mix cleanup");
    check(call(*service,"GET","/api/task/result","id=revision-mixed&variant=vocal").body==wav,"cleanup deleted unique vocal");
    write(root/"tasks/revision-mixed/work/inherited-conversion.wav","corrupt vocal");
    check(call(*service,"GET","/api/task/result","id=revision-mixed&variant=vocal").status==409,"corrupt vocal served");
    check(call(*service,"GET","/api/tasks").status==200,"bad optional vocal poisoned history");
    fs::remove(root/"tasks/revision-mixed/work/inherited-conversion.wav");
    check(call(*service,"GET","/api/task","id=revision-mixed").body.find("人声")!=std::string::npos,"missing old vocal lacks explanation");
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","bad-source-1"},{"source_task","../escape"}}).status==400,"invalid parent accepted");
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","bad-source-2"},{"source_task","absent-parent"}}).status==404,"missing parent accepted");
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","bad-source-3"},{"source_task","revision-parent"},{"session","session"}}).status==400,"ambiguous source accepted");
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","bad-source-4"},{"source_task","revision-parent"},{"f0_up_key","99"}}).status==400,"invalid edited parameter accepted");
    check(call(*service,"POST","/api/task/cleanup","",{{"id","revision-parent"},{"attempt","1"}}).status==200,"parent cleanup");
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","revision-pruned"},{"source_task","revision-parent"},{"output_kind","mix"}}).status==202,"pruned parent rejected");settled(service);
    state=call(*service,"GET","/api/task","id=revision-pruned");check(state.body.find("succeeded")!=state.body.npos,"pruned revision failed: "+state.body);
    check(runner->calls==4,"retained vocal result not reused after cleanup");
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","revision-cleaning"},{"source_task","revision-parent"},{"pp_dereverb","1"}}).status==202,"cleanup change admission");settled(service);
    state=call(*service,"GET","/api/task","id=revision-cleaning");check(state.body.find("succeeded")!=state.body.npos,"cleanup revision failed: "+state.body);
    check(runner->calls==6,"cleanup change must reuse separation but rerun cleanup/conversion");
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","revision-drybase"},{"session","session"},{"workflow","dry"}}).status==202,"dry parent");settled(service);
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","revision-drycopy"},{"source_task","revision-drybase"},{"protect","0.4"}}).status==202,"dry revision");settled(service);
    state=call(*service,"GET","/api/task","id=revision-drycopy");check(state.body.find("succeeded")!=state.body.npos,"dry revision failed: "+state.body);
    check(runner->captured_input==wav,"dry input acquired extra resampling/normalization");check(runner->calls==8,"dry revisions computation count");
    check(call(*service,"GET","/api/task/source","id=revision-parent",{},"bytes=0-7").body==wav.substr(0,8),"source range mismatch");
    write(root/"tasks/revision-pitch/work/inherited-cleanup.wav","bad cached artifact");
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","revision-badstage"},{"source_task","revision-pitch"},{"f0_up_key","2"}}).status==202,"bad stage admission");settled(service);
    state=call(*service,"GET","/api/task","id=revision-badstage");check(state.body.find("failed")!=state.body.npos&&runner->calls==8,"corrupt cache computed or succeeded");
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","revision-badparent"},{"source_task","revision-badstage"}}).status==409,"failed parent allowed");
    write(root/"tasks/revision-pitch/work/source.wav","corrupt source");
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id","revision-corrupt"},{"source_task","revision-pitch"}}).status==202,"corrupt verification admission");settled(service);
    state=call(*service,"GET","/api/task","id=revision-corrupt");check(state.body.find("failed")!=state.body.npos&&runner->calls==8,"corrupt source computed or succeeded");
}
void storage_protection(const fs::path& root) {
    Paths p;p.runs=(root/"runs").string();p.tasks=(root/"tasks").string();p.build=(root/"build").string();
    p.gmodel=(root/"models/base").string();write(p.gmodel,"model fixture");audio_assets(p,root);write(root/"runs/session/raw/source.wav",wav);
    p.enforce_task_storage=true;p.storage_limits.managed_bytes=65536;p.storage_limits.task_bytes=65536;p.storage_limits.reserve_bytes=0;
    auto runner=std::make_shared<AudioRunner>();auto s=std::make_shared<WebService>(p,runner);
    std::map<std::string,std::string> fields{{"task_id","disk-protected"},{"session","session"},{"workflow","dry"}};
    check(call(*s,"POST","/api/tasks/convert","",fields).status==202,"disk fixture not accepted");runner->wait();
    write(root/"tasks/disk-protected/work/retained-extra.bin",std::string(70000,'x'));
    runner->release();settled(s);
    const auto paused=call(*s,"GET","/api/task","id=disk-protected");
    check(paused.body.find("\"state\":\"paused\"")!=std::string::npos,"disk boundary did not safely pause: "+paused.body);
    check(paused.body.find("保护线")!=std::string::npos,"storage pause reason was lost");
    check(call(*s,"POST","/api/task/resume","",{{"id","disk-protected"},{"attempt","1"}}).status==507,"over-budget resume accepted");
    fields["task_id"]="disk-new-task";check(call(*s,"POST","/api/tasks/convert","",fields).status==507,"over-budget new work accepted");
    check(call(*s,"GET","/api/tasks").status==200&&call(*s,"GET","/api/storage").status==200,"quota pressure broke history");
    fs::remove(root/"tasks/disk-protected/work/retained-extra.bin");
    check(call(*s,"POST","/api/task/resume","",{{"id","disk-protected"},{"attempt","1"}}).status==202,"freeing space did not allow explicit resume");settled(s);
    check(call(*s,"GET","/api/task/result","id=disk-protected").body==wav&&runner->calls==1,"storage resume repeated completed conversion or lost result");
}
void audio_tasks(const fs::path& root) {
    Paths p;p.runs=(root/"runs").string();p.tasks=(root/"tasks").string();p.build=(root/"build").string();
    p.gmodel=(root/"models/base.safetensors").string();write(p.gmodel,"model fixture");
    audio_assets(p,root);
    write(root/"runs/session/raw/source.wav",wav);
    auto runner=std::make_shared<AudioRunner>();auto service=std::make_shared<WebService>(p,runner);
    std::map<std::string,std::string> fields{{"task_id","client-convert-1"},{"session","session"},{"workflow","dry"}};
    auto accepted=call(*service,"POST","/api/tasks/convert","",fields);
    check(accepted.status==202,"async conversion acceptance: "+accepted.body);runner->wait();
    check(call(*service,"GET","/api/tasks","offset=100").body.find("\"active_id\":\"client-convert-1\"")!=std::string::npos,"active task absent outside history page");
    const auto live=call(*service,"GET","/api/task","id=client-convert-1");
    check(live.body.find("\"phase\":\"synthesis\"")!=std::string::npos&&
          live.body.find("\"completed\":16000")!=std::string::npos&&
          live.body.find("\"total\":32000")!=std::string::npos,"real child counters not persisted");
    check(call(*service,"POST","/api/tasks/convert","",fields).status==409,"duplicate cannot rerun");
    check(call(*service,"GET","/api/task/result","id=client-convert-1").status==409,"no incomplete result");
    check(call(*service,"POST","/api/task/cleanup","",{{"id","client-convert-1"},{"attempt","1"}}).status==409,"active task cleanup accepted");
    const auto rename=[&](const std::string& name,const std::string& expected){return call(*service,"POST","/api/task/rename","",{{"id","client-convert-1"},{"name",name},{"expected_name",expected}});};
    check(rename("我的翻唱 · 降八度","source.wav").status==200,"running task rename rejected");
    check(rename("我的翻唱 · 降八度","source.wav").status==200,"rename lost-response retry not idempotent");
    check(rename("覆盖他人的改名","source.wav").status==409,"stale rename overwrote current title");
    for(const auto& bad:{std::string("   "),std::string(256,'x'),std::string("bad\nname"),std::string("\xc0\xaf",2)})
        check(rename(bad,"我的翻唱 · 降八度").status==400,"invalid display name accepted");
    check(call(*service,"GET","/api/task/rename").status==405,"rename method guard");
    check(call(*service,"POST","/api/task/rename","",{{"id","missing-task"},{"name","测试"},{"expected_name","old"}}).status==404,"rename absent task");
    write(root/"runs/session/raw/source.wav","replaced session");
    check(runner->captured_input==wav,"input snapshot identity");runner->release();settled(service);
    auto done=call(*service,"GET","/api/task","id=client-convert-1");
    check(done.body.find("\"state\":\"succeeded\"")!=done.body.npos,"audio terminal state: "+done.body);
    check(done.body.find("\"name\":\"source.wav\"")!=done.body.npos,"task history lost input name");
    check(done.body.find("\"display_name\":\"我的翻唱 · 降八度\"")!=done.body.npos,"worker completion lost display name");
    auto result=call(*service,"GET","/api/task/result","id=client-convert-1");
    check(result.status==200&&result.body==wav,"durable audio result");
    check(call(*service,"GET","/api/storage").status==200,"storage usage missing");
    check(call(*service,"POST","/api/task/cleanup","",{{"id","client-convert-1"},{"attempt","2"}}).status==409,"stale cleanup accepted");
    const auto cleaned=call(*service,"POST","/api/task/cleanup","",{{"id","client-convert-1"},{"attempt","1"}});
    check(cleaned.status==200,"completed copy cleanup rejected: "+cleaned.body);
    check(!fs::exists(root/"tasks/client-convert-1/work/model.safetensors"),"redundant model snapshot retained");
    check(call(*service,"GET","/api/task/result","id=client-convert-1").body==wav,"cleanup lost completed audio");
    check(call(*service,"POST","/api/tasks/convert","",fields).status==409,"cleaned task ID duplicated computation");
    auto partial=call(*service,"GET","/api/task/result","id=client-convert-1",{},"bytes=1-5");
    check(partial.status==206&&partial.body==wav.substr(1,5),"bounded byte range");
    check(call(*service,"GET","/api/task/result","id=client-convert-1",{},"bytes=-4").body==wav.substr(wav.size()-4),"suffix range");
    check(call(*service,"GET","/api/task/result","id=client-convert-1",{},"bytes=49-").body==wav.substr(49),"open range");
    for(const auto& range:{"bytes=999-","bytes=8-4","bytes=-0","bytes=1-2,4-5","bytes=0-18446744073709551616","items=1-2"})
        check(call(*service,"GET","/api/task/result","id=client-convert-1",{},range).status==416,"invalid range accepted");
    service.reset();service=std::make_shared<WebService>(p,runner);
    check(call(*service,"GET","/api/task/result","id=client-convert-1").body==wav,"result survives service restart");
    check(runner->calls==1,"GET/restart/duplicate reran model");service.reset();
    write(root/"runs/session/raw/source.wav",wav);runner=std::make_shared<AudioRunner>();runner->fail=true;
    service=std::make_shared<WebService>(p,runner);fields["task_id"]="client-convert-2";fields["workflow"]="song";
    check(call(*service,"POST","/api/tasks/convert","",fields).status==202,"failed conversion accepted");
    runner->wait();runner->release();settled(service);
    auto failed=call(*service,"GET","/api/task","id=client-convert-2");
    check(failed.body.find("\"state\":\"failed\"")!=failed.body.npos,"audio failed state retained");
    check(fs::exists(root/"tasks/client-convert-2/work/vocals/preprocessed/result.wav"),"successful separation lost on conversion failure");
    check(call(*service,"GET","/api/task/log","id=client-convert-2").body.find("failure details")!=std::string::npos,"failure diagnostics retained");
    check(call(*service,"GET","/api/task/result","id=client-convert-2").status==409,"failed output incorrectly downloadable");
    service.reset();runner=std::make_shared<AudioRunner>();service=std::make_shared<WebService>(p,runner);
    fields["task_id"]="client-convert-lost";fields["workflow"]="dry";
    (void)call(*service,"POST","/api/tasks/convert","",fields,"",true);
    runner->wait();runner->release();settled(service);
    check(call(*service,"GET","/api/task/result","id=client-convert-lost").body==wav,"lost acceptance response killed output");
    check(call(*service,"POST","/api/tasks/convert","",fields).status==409&&runner->calls==1,"retry after lost response duplicated work");
    check(call(*service,"POST","/api/tasks/preprocess","",{{"task_id","client-clean-1"},{"session","session"},{"workflow","dry"}}).status==202,"background cleanup accepted");
    settled(service);
    check(call(*service,"GET","/api/task/result","id=client-clean-1").body==wav,"background cleanup retained result");
    fs::remove(root/"tasks/client-clean-1/work/result.wav");
    check(call(*service,"GET","/api/task/result","id=client-clean-1").status==410,"missing result must be explicit");
    check(call(*service,"GET","/api/tasks").status==200,"one missing audio broke all task history");
    service.reset();runner=std::make_shared<AudioRunner>();service=std::make_shared<WebService>(p,runner);
    fields["task_id"]="client-log-failure";
    check(call(*service,"POST","/api/tasks/convert","",fields).status==202,"log failure fixture accepted");
    runner->wait();fs::create_directory(root/"tasks/client-log-failure/task.log.pending");
    runner->release();settled(service);
    check(call(*service,"GET","/api/tasks").status==503,"failed terminal persistence presented normal state");
    check(fs::exists(root/"tasks/client-log-failure/work/result.wav"),"persistence failure deleted output");
    service.reset();service=std::make_shared<WebService>(p,runner);
    check(call(*service,"GET","/api/task","id=client-log-failure").body.find("\"state\":\"interrupted\"")!=std::string::npos,"startup fabricated success after persistence failure");
}
struct ControlledAudioRunner:ProcessRunner {
    mutable std::mutex mu;mutable std::condition_variable cv;
    mutable int conversions=0,preprocesses=0;bool release=false,fail_on_pause=false;
    ProcessResult run(const ProcessSpec& s)const override {
        const auto program=fs::path(s.argv[0]).filename();
        if(program=="vc_preprocess") {
            {std::lock_guard lock(mu);++preprocesses;}
            write(fs::path(arg(s,"--output-dir"))/"source.wav",wav);
            return {ProcessTermination::kExited,0,0,0,0};
        }
        check(program=="vc_convert","unexpected control fixture child");
        int attempt;{std::lock_guard lock(mu);attempt=++conversions;cv.notify_all();}
        check(!arg(s,"--checkpoint-dir").empty()&&!arg(s,"--pause-file").empty(),"child has no durable control");
        // Fake block existence tests orchestration only; real model/CLI tests
        // separately validate the actual checkpoint format and numerical output.
        fs::create_directories(arg(s,"--checkpoint-dir"));
        if(attempt==1) {
            resonantia::training_control::TrainingCancelReader pause(arg(s,"--pause-file"));
            const auto until=std::chrono::steady_clock::now()+3s;
            while(std::chrono::steady_clock::now()<until) {
                auto state=pause.poll();
                if(state==resonantia::training_control::TrainingResult::kCancelled)return {ProcessTermination::kExited,fail_on_pause?1:2,0,0,0};
                check(state==resonantia::training_control::TrainingResult::kCompleted,"bad control channel");
                std::this_thread::sleep_for(1ms);
            }
            throw std::runtime_error("pause never delivered");
        }
        check(std::find(s.argv.begin(),s.argv.end(),"--resume")!=s.argv.end(),"child did not resume saved blocks");
        std::unique_lock lock(mu);check(cv.wait_for(lock,3s,[&]{return release;}),"resume not released");
        write(arg(s,"--output"),wav);return {ProcessTermination::kExited,0,0,0,0};
    }
    void wait(int n){std::unique_lock lock(mu);check(cv.wait_for(lock,3s,[&]{return conversions>=n;}),"controlled child not started");}
    void finish(){std::lock_guard lock(mu);release=true;cv.notify_all();}
};
void audio_controls(const fs::path& root) {
    Paths p;p.runs=(root/"runs").string();p.tasks=(root/"tasks").string();p.build=(root/"build").string();
    p.gmodel=(root/"models/base.safetensors").string();write(p.gmodel,"model fixture");audio_assets(p,root);
    write(root/"runs/session/raw/source.wav",wav);
    auto runner=std::make_shared<ControlledAudioRunner>();auto service=std::make_shared<WebService>(p,runner);
    const std::string id="controlled-audio";
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id",id},{"session","session"},{"workflow","song"},{"output_kind","vocal"}}).status==202,"control task acceptance");
    runner->wait(1);
    auto status=call(*service,"GET","/api/task","id="+id);
    check(status.body.find("\"attempt\":1")!=std::string::npos,"missing stable execution attempt");
    check(call(*service,"POST","/api/task/pause","",{{"id",id},{"attempt","2"}}).status==409,"wrong attempt paused active task");
    check(call(*service,"POST","/api/task/pause","",{{"id",id},{"attempt","1"}}).status==202,"pause rejected");
    settled(service);status=call(*service,"GET","/api/task","id="+id);
    check(status.body.find("\"state\":\"paused\"")!=status.body.npos,"safe pause not retained: "+status.body);
    check(call(*service,"GET","/api/task/result","id="+id).status==409,"paused output was published");
    check(call(*service,"POST","/api/task/pause","",{{"id",id},{"attempt","1"}}).status==200,"pause acknowledgement not idempotent");
    service.reset();service=std::make_shared<WebService>(p,runner);
    check(call(*service,"POST","/api/task/resume","",{{"id",id},{"attempt","1"}}).status==202,"restart resume rejected");
    runner->wait(2);
    check(call(*service,"POST","/api/task/resume","",{{"id",id},{"attempt","1"}}).status==202,"lost resume response not idempotentent");
    check(call(*service,"POST","/api/task/pause","",{{"id",id},{"attempt","1"}}).status==409,"late pause reached new execution");
    check(runner->preprocesses==2&&runner->conversions==2,"resume repeated completed separation/cleanup");
    runner->finish();settled(service);
    status=call(*service,"GET","/api/task","id="+id);
    check(status.body.find("\"state\":\"succeeded\"")!=status.body.npos&&status.body.find("\"attempt\":2")!=status.body.npos,"resumed terminal state missing");
    check(call(*service,"GET","/api/task/result","id="+id).status==200,"resumed result unavailable");
}
void failed_mix_resume(const fs::path& root) {
    Paths p;p.runs=(root/"runs").string();p.tasks=(root/"tasks").string();p.build=(root/"build").string();
    p.gmodel=(root/"models/base.safetensors").string();write(p.gmodel,"model fixture");audio_assets(p,root);
    write(root/"runs/session/raw/source.wav",wav);
    auto runner=std::make_shared<AudioRunner>();auto service=std::make_shared<WebService>(p,runner);
    const std::string id="failed-mix-retry";
    check(call(*service,"POST","/api/tasks/convert","",{{"task_id",id},{"session","session"},{"workflow","song"},{"output_kind","mix"}}).status==202,"mix fixture acceptance");
    runner->wait();const auto work=fs::path(p.tasks)/id/"work";fs::create_directory(work/"cover.wav");
    runner->release();settled(service);
    auto state=call(*service,"GET","/api/task","id="+id);
    check(state.body.find("\"state\":\"failed\"")!=state.body.npos&&runner->calls==3,"mix failure fixture did not complete model stages");
    check(fs::exists(work/"stage.conversion.done"),"completed conversion not retained before mix failure");
    const auto frontend=read_file(p.hubert);write(p.hubert,"changed dependency");
    check(call(*service,"POST","/api/task/resume","",{{"id",id},{"attempt","1"}}).status==202,"identity recheck not accepted");
    settled(service);state=call(*service,"GET","/api/task","id="+id);
    check(state.body.find("\"state\":\"failed\"")!=state.body.npos&&runner->calls==3,"dependency change reran or reused mismatching models");
    write(p.hubert,frontend);fs::remove(work/"cover.wav");
    check(call(*service,"POST","/api/task/resume","",{{"id",id},{"attempt","2"}}).status==202,"mix retry rejected");
    settled(service);state=call(*service,"GET","/api/task","id="+id);
    check(state.body.find("\"state\":\"succeeded\"")!=state.body.npos&&runner->calls==3,"mix retry reran completed model stages: "+state.body);
    check(call(*service,"GET","/api/task/result","id="+id).status==200,"repaired mix not retained");
}
void control_failures(const fs::path& root) {
    for(bool storage_failure:{false,true}) {
        const auto base=root/(storage_failure?"storage":"child");
        Paths p;p.runs=(base/"runs").string();p.tasks=(base/"tasks").string();p.build=(base/"build").string();
        p.gmodel=(base/"models/base.safetensors").string();write(p.gmodel,"model fixture");audio_assets(p,base);
        write(base/"runs/session/raw/source.wav",wav);
        auto runner=std::make_shared<ControlledAudioRunner>();runner->fail_on_pause=!storage_failure;
        auto service=std::make_shared<WebService>(p,runner);const std::string id="control-error-task";
        check(call(*service,"POST","/api/tasks/convert","",{{"task_id",id},{"session","session"},{"workflow","dry"}}).status==202,"error fixture acceptance");
        runner->wait(1);const auto directory=fs::path(p.tasks)/id;
        if(storage_failure){fs::rename(directory/"record",directory/"record.saved");fs::create_directory(directory/"record");}
        const auto response=call(*service,"POST","/api/task/pause","",{{"id",id},{"attempt","1"}});
        check(response.status==202,"storage failure blocked safe stop request");
        settled(service);
        if(storage_failure) {
            check(response.body.find("\"durable\":false")!=response.body.npos,"failed pause persistence hidden");
            check(call(*service,"GET","/api/task","id="+id).status==503,"storage failure fabricated durable pause");
            service.reset();fs::remove(directory/"record");fs::rename(directory/"record.saved",directory/"record");
            service=std::make_shared<WebService>(p,runner);
            check(call(*service,"GET","/api/task","id="+id).body.find("\"state\":\"interrupted\"")!=std::string::npos,"storage recovery fabricated completed state");
        }else check(call(*service,"GET","/api/task","id="+id).body.find("\"state\":\"failed\"")!=std::string::npos,"child failure was hidden as pause");
    }
}
void streamed_artifact(const fs::path& root) {
    const auto path=root/"large.wav";write(path,"");constexpr std::uint64_t bytes=70ULL<<20;
    fs::resize_file(path,bytes);
    int fd[2];check(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,fd)==0,"stream socketpair");
    const timeval timeout{3,0};for(int f:fd) {setsockopt(f,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof timeout);setsockopt(f,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof timeout);}
    auto writer=std::async(std::launch::async,[&]{send_audio_file(fd[0],path,"");close(fd[0]);});
    std::array<char,8192> buffer{};std::string headers;bool body=false;std::uint64_t received=0;
    for(;;) {
        const auto n=recv(fd[1],buffer.data(),buffer.size(),0);
        if(n==0)break;check(n>0,"stream timed out");
        if(body)received+=n;
        else {
            headers.append(buffer.data(),n);const auto boundary=headers.find("\r\n\r\n");
            if(boundary!=headers.npos) {received+=headers.size()-boundary-4;headers.resize(boundary);body=true;}
            else check(headers.size()<4096,"unbounded response headers");
        }
    }
    close(fd[1]);writer.get();
    check(headers.starts_with("HTTP/1.1 200")&&received==bytes,"large artifact transfer truncated at old buffer ceiling");
    check(headers.find("Content-Length: "+std::to_string(bytes))!=headers.npos,"stream length mismatch");
    // A vanished peer must only end this download, not throw or mutate the task.
    check(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,fd)==0,"closed stream socketpair");
    close(fd[1]);send_audio_file(fd[0],path,"bytes=0-9");close(fd[0]);
}
struct FullTrainingRunner:ProcessRunner {
    mutable std::mutex mu;mutable std::condition_variable cv;
    mutable int trains=0,indexes=0,cleans=0;
    bool fail_index=true,pause_preprocess=false,pause_training=true,block_resumed=false,released_resume=false;
    ProcessResult run(const ProcessSpec& s)const override {
        const auto program=fs::path(s.argv[0]).filename();
        if(program=="vc_preprocess") {
            std::unique_lock lock(mu);++cleans;cv.notify_all();
            if(pause_preprocess&&cleans==1) {
                const auto token=arg(s,"--pause-file");
                write(fs::path(arg(s,"--checkpoint-dir"))/"fixture-block","committed cleanup window");
                const auto until=std::chrono::steady_clock::now()+3s;
                while(read_file(token)!="1"&&std::chrono::steady_clock::now()<until) {
                    lock.unlock();std::this_thread::sleep_for(1ms);lock.lock();
                }
                check(read_file(token)=="1","preparation pause was not delivered");
                return {ProcessTermination::kExited,2,0,0,0};
            }
            if(pause_preprocess)check(std::find(s.argv.begin(),s.argv.end(),"--resume")!=s.argv.end(),"cleanup blocks were not resumed");
            write(fs::path(arg(s,"--output-dir"))/"clip.wav",wav);
        } else if(program=="vc_train") {
            const auto token=arg(s,"--cancel-file");
            std::unique_lock lock(mu);++trains;cv.notify_all();
            check(read_file(fs::path(arg(s,"--target"))/"clip.wav")==wav,"training did not use fixed clips");
            const auto checkpoint=fs::path(arg(s,"--checkpoint-dir"))/"state_step_0000000001.safetensors";
            if(pause_training&&trains==1) {
                const auto until=std::chrono::steady_clock::now()+3s;
                while(read_file(token)!="1"&&std::chrono::steady_clock::now()<until) {
                    lock.unlock();std::this_thread::sleep_for(1ms);lock.lock();
                }
                check(read_file(token)=="1","training pause channel was not signalled");
                write(checkpoint,"complete G D optimizer RNG fixture");
                write(s.log_path,"Checkpoint retained at complete step 1\n");
                return {ProcessTermination::kExited,2,0,0,0};
            }
            if(pause_training)check(arg(s,"--resume")==checkpoint.string(),"complete training state was not resumed");
            if(block_resumed)check(cv.wait_for(lock,3s,[&]{return released_resume;}),"resumed trainer was not released");
            write(arg(s,"--out"),"trained model fixture");
        } else if(program=="build_index") {
            ++indexes;
            if(fail_index)return {ProcessTermination::kExited,1,0,0,0};
            write(arg(s,"--output"),"index fixture");
        } else throw std::runtime_error("unexpected full training program");
        return {ProcessTermination::kExited,0,0,0,0};
    }
    void wait(){std::unique_lock lock(mu);check(cv.wait_for(lock,3s,[&]{return trains>0;}),"full trainer did not start");}
    void wait_cleanup(){std::unique_lock lock(mu);check(cv.wait_for(lock,3s,[&]{return cleans>0;}),"full preparation did not start");}
    void wait_resumed(){std::unique_lock lock(mu);check(cv.wait_for(lock,3s,[&]{return trains>1;}),"resumed trainer did not start");}
    void release_resume(){std::lock_guard lock(mu);released_resume=true;cv.notify_all();}
};
void full_training_tasks(const fs::path& root) {
    Paths p;p.runs=(root/"runs").string();p.tasks=(root/"tasks").string();p.build=(root/"build").string();
    audio_assets(p,root);p.gmodel=(root/"models/base").string();p.dmodel=(root/"models/discriminator").string();
    for(const auto& file:{p.gmodel,p.dmodel,p.build+"/vc_train",p.build+"/build_index"})write(file,"fixed training fixture");
    write(root/"runs/voice/raw/clip.wav",wav);write(root/"runs/voice/work/clip.wav",wav);
    auto runner=std::make_shared<FullTrainingRunner>();auto s=std::make_shared<WebService>(p,runner);
    const std::map<std::string,std::string> fields{{"name","voice"},{"task_id","full-task"},{"mode","joint-full"},{"epochs","2"},{"prepared","1"}};
    auto accepted=call(*s,"POST","/api/train","",fields);
    check(accepted.status==200&&json_string(accepted.body,"job_id")=="full-task","training needs stable submitted ID");
    runner->wait();
    check(call(*s,"GET","/api/train/runs").body.find("\"task_id\":\"full-task\"")!=std::string::npos,"managed training history lost original task ID");
    check(call(*s,"GET","/api/task","id=full-task").body.find("\"pause\":true")!=std::string::npos,"full training pause unavailable");
    check(call(*s,"POST","/api/task/pause","",{{"id","full-task"},{"attempt","1"}}).status==202,"full training pause rejected");
    settled(s);
    check(call(*s,"GET","/api/task","id=full-task").body.find("\"state\":\"paused\"")!=std::string::npos,"full training not paused");
    check(runner->indexes==0,"paused training built an index");
    check(call(*s,"POST","/api/task/rename","",{{"id","full-task"},{"name","完整训练 · 候选版"},{"expected_name","voice"}}).status==200,"paused training rename rejected");
    write(root/"runs/voice/raw/clip.wav","new material");write(root/"runs/voice/work/clip.wav","new work");
    s.reset();s=std::make_shared<WebService>(p,runner);
    check(call(*s,"POST","/api/train","",fields).status==409,"duplicate training accepted");
    write(p.hubert,"changed frontend");
    check(call(*s,"POST","/api/task/resume","",{{"id","full-task"},{"attempt","1"}}).status==202,"full training resume rejected");
    settled(s);
    check(call(*s,"GET","/api/task","id=full-task").body.find("\"state\":\"failed\"")!=std::string::npos,"changed training frontend accepted");
    check(runner->trains==1&&runner->indexes==0,"changed identity reached a GPU process");
    write(p.hubert,"fixed fixture dependency");
    runner->block_resumed=true;
    check(call(*s,"POST","/api/task/resume","",{{"id","full-task"},{"attempt","2"}}).status==202,"restored training dependencies rejected");
    runner->wait_resumed();
    const auto stale_cancel=call(*s,"POST","/api/train/cancel","",{{"job_id","full-task"}});
    runner->release_resume();
    check(stale_cancel.status==409,"legacy cancellation from a prior attempt stopped resumed training");
    settled(s);
    check(call(*s,"GET","/api/task","id=full-task").body.find("\"state\":\"failed\"")!=std::string::npos,"index failure lost");
    check(runner->trains==2&&runner->indexes==1,"unexpected training execution count");
    runner->fail_index=false;
    check(call(*s,"POST","/api/task/resume","",{{"id","full-task"},{"attempt","3"}}).status==202,"index retry rejected");
    settled(s);
    check(call(*s,"GET","/api/task","id=full-task").body.find("\"state\":\"succeeded\"")!=std::string::npos,"training result not retained");
    check(runner->trains==2&&runner->indexes==2,"index retry repeated training");
    check(read_file(root/"runs/voice/full_training/result/model.safetensors")=="trained model fixture","review candidate missing");
    check(call(*s,"GET","/api/train/runs").body.find("candidate_epoch_0000000002.safetensors")!=std::string::npos,"final epoch missing from reviewable candidates below periodic export interval");
    check(!fs::exists(root/"runs/voice/model.safetensors"),"training auto-published voice");
    check(read_file(root/"runs/voice/raw/clip.wav")=="new material","resume changed source recordings");
    s.reset();runner=std::make_shared<FullTrainingRunner>();runner->pause_preprocess=true;runner->pause_training=false;runner->fail_index=false;
    s=std::make_shared<WebService>(p,runner);
    write(root/"runs/cleanup/raw/clip.wav",wav);write(root/"runs/cleanup/work/clip.wav",wav);
    auto cleaning=call(*s,"POST","/api/train","",{{"task_id","clean-training"},{"name","cleanup"},{"mode","joint-full"},{"epochs","2"}});
    check(cleaning.status==200,"training preparation not accepted");runner->wait_cleanup();
    check(call(*s,"POST","/api/task/pause","",{{"id","clean-training"},{"attempt","1"}}).status==202,"training preparation pause rejected");
    settled(s);
    check(call(*s,"GET","/api/task","id=clean-training").body.find("\"state\":\"paused\"")!=std::string::npos,"preparation pause classified as failure");
    check(runner->trains==0,"paused preparation launched trainer");
    s.reset();s=std::make_shared<WebService>(p,runner);
    check(call(*s,"POST","/api/task/resume","",{{"id","clean-training"},{"attempt","1"}}).status==202,"training preparation resume rejected");
    settled(s);
    check(call(*s,"GET","/api/task","id=clean-training").body.find("\"state\":\"succeeded\"")!=std::string::npos,"training preparation resume failed");
    check(runner->cleans==2&&runner->trains==1&&runner->indexes==1,"training preparation repeated completed work");
}
void task_capacity(const fs::path& root) {
    Paths p;p.runs=(root/"runs").string();p.tasks=(root/"tasks").string();p.build=(root/"build").string();
    p.gmodel=(root/"model").string();write(p.gmodel,"model");audio_assets(p,root);
    write(root/"runs/session/raw/source.wav",wav);
    {
        TaskStore store(p.tasks);
        for(int i=0;i<256;++i){TaskRecord r;r.id="retained-task-"+std::to_string(i);r.kind="convert";r.state="succeeded";check(store.save(r,0),"capacity fixture");}
    }
    auto s=std::make_shared<WebService>(p,std::make_shared<AudioRunner>());
    const auto response=call(*s,"POST","/api/tasks/convert","",{{"task_id","over-capacity"},{"session","session"},{"workflow","dry"}});
    check(response.status==409,"capacity must reject admission without reporting storage failure");
    check(call(*s,"GET","/api/tasks","limit=1").status==200,"capacity failure poisoned retained task queries");
    check(call(*s,"GET","/api/task","id=over-capacity").status==404,"capacity rejection accepted work");
    write(root/"runs/training/work/clip.wav",wav);
    check(call(*s,"POST","/api/train","",{{"name","training"},{"steps","2"}}).status==409,"training capacity admission reported a storage failure");
    check(call(*s,"GET","/api/tasks","limit=1").status==200,"training capacity failure poisoned history");
}
void interrupted_controls(const fs::path& root) {
    Paths p;p.runs=(root/"runs").string();p.tasks=(root/"tasks").string();
    {
        TaskStore store(p.tasks);
        for(const auto& id:{"old-active","old-paused","old-missing","old-broken"}) {
            TaskRecord r;r.id=id;r.kind="training";r.name="voice";r.state=std::string(id)=="old-paused"?"paused":"running";
            r.attempt=4;r.parameters={{"recovery","training-full-v1"}};check(store.save(r,0),"seed old execution");
            if(r.id!="old-missing")write(store.task_directory(r.id)/"control.4",r.id=="old-broken"?"bad":"0");
        }
    }
    resonantia::training_control::TrainingCancelReader orphan((root/"tasks/old-active/control.4").string());
    auto s=std::make_shared<WebService>(p,std::make_shared<BlockingRunner>());
    check(orphan.poll()==resonantia::training_control::TrainingResult::kCancelled,"restart did not safely stop surviving owned trainer");
    check(read_file(root/"tasks/old-paused/control.4")=="0","restart signalled an inactive execution");
    check(!fs::exists(root/"tasks/old-missing/control.4"),"restart fabricated an old channel");
    check(read_file(root/"tasks/old-broken/control.4")=="bad","restart rewrote invalid old channel");
    const auto bad=call(*s,"GET","/api/task","id=old-broken");
    check(bad.status==200&&bad.body.find("control channel")!=std::string::npos,"recovery channel error not visible in history");
}
void run(const fs::path& root){
    Paths p;p.runs=(root/"runs").string();p.tasks=(root/"tasks").string();p.build=(root/"build").string();
    write(root/"runs/voice/work/a.wav","RIFFxxxxWAVEfixture");
    auto runner=std::make_shared<BlockingRunner>();auto service=std::make_shared<WebService>(p,runner);
    check(call(*service,"POST","/api/train","",{{"name","voice"},{"steps","2"},{"task_id",std::string(65,'x')}}).status==400,"training accepted ID that legacy controls cannot address");
    check(call(*service,"GET","/api/tasks").body=="{\"tasks\":[],\"offset\":0,\"limit\":25,\"active_id\":\"\"}","empty task list");
    check(call(*service,"GET","/api/tasks","limit=101").status==400,"page limit");
    check(call(*service,"GET","/api/task","id=../bad").status==400,"ID path rejection");
    check(call(*service,"GET","/api/task","id=absent-task").status==404,"absent task");
    check(call(*service,"POST","/api/tasks").status==405,"method guard");
    auto accepted=call(*service,"POST","/api/train","",{{"name","voice"},{"steps","2"}});
    check(accepted.status==200,"training accepted");auto id=json_string(accepted.body,"job_id");runner->wait();
    auto active=call(*service,"GET","/api/task","id="+id);
    check(active.status==200&&active.body.find("\"state\":\"running\"")!=active.body.npos,"live durable training");
    check(active.body.find("\"steps\":\"2\"")!=active.body.npos,"frozen parameters");
    check(call(*service,"POST","/api/task/rename","",{{"id",id},{"name","中文训练名称"},{"expected_name","voice"}}).status==200,"training rename rejected");
    runner->release();settled(service);
    auto done=call(*service,"GET","/api/task","id="+id);
    check(done.body.find("\"state\":\"succeeded\"")!=done.body.npos,"terminal state");
    check(done.body.find("\"display_name\":\"中文训练名称\"")!=done.body.npos&&fs::exists(root/"runs/voice/model.safetensors"),"display rename changed training destination");
    // Subsequent jobs can overwrite train.log; the old task's bounded log must survive.
    write(root/"runs/voice/train.log","new job log");
    check(call(*service,"GET","/api/task/log","id="+id).body.find("retained test log")!=std::string::npos,"historical log snapshot");
    service.reset();service=std::make_shared<WebService>(p,runner);
    check(call(*service,"GET","/api/task","id="+id).body==done.body,"terminal state survives restart");service.reset();
    {
        TaskStore store(p.tasks);TaskRecord r;r.id="interrupted-job";r.kind="training";r.name="voice";r.state="running";r.stage="training";r.completed=2;r.total=10;r.unit="steps";check(store.save(r,0),"seed interrupted record");
    }
    write(root/"tasks/interrupted-job/live.log","retained before restart\n");
    service=std::make_shared<WebService>(p,runner);
    check(call(*service,"GET","/api/task/log","id=interrupted-job").body=="retained before restart\n","restart lost interrupted task logs");
    auto interrupted=call(*service,"GET","/api/task","id=interrupted-job");
    check(interrupted.body.find("\"state\":\"interrupted\"")!=interrupted.body.npos&&interrupted.body.find("\"completed\":2")!=interrupted.body.npos,"restart retains real progress");
    service.reset();
    auto failure=std::make_shared<BlockingRunner>();failure->fail=true;service=std::make_shared<WebService>(p,failure);
    accepted=call(*service,"POST","/api/train","",{{"name","voice"},{"steps","2"}});id=json_string(accepted.body,"job_id");failure->wait();failure->release();settled(service);
    check(call(*service,"GET","/api/task","id="+id).body.find("\"state\":\"failed\"")!=std::string::npos,"failed task retained");
    check(call(*service,"GET","/api/task/log","id="+id).body.find("retained test log")!=std::string::npos,"failure log retained");
}
}
int main(){try{TempDir temp("/tmp","web-tasks-test-");audio_revisions(fs::path(temp.path())/"revisions");storage_protection(fs::path(temp.path())/"storage");task_capacity(fs::path(temp.path())/"capacity");interrupted_controls(fs::path(temp.path())/"restart-control");full_training_tasks(fs::path(temp.path())/"full-training");run(fs::path(temp.path())/"training");audio_tasks(fs::path(temp.path())/"audio");audio_controls(fs::path(temp.path())/"controls");failed_mix_resume(fs::path(temp.path())/"mix-retry");control_failures(fs::path(temp.path())/"control-failures");streamed_artifact(fs::path(temp.path())/"stream");std::cout<<"durable service task contracts passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
