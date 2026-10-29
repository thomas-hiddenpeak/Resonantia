#include "tools/web/task_storage.h"
#include "tools/web/workflow_support.h"
#include "voxmutatio/io/file_digest.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <atomic>
using namespace resonantia::web;
namespace fs=std::filesystem;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F>void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}check(bad,"unsafe storage operation accepted");}
void put(const fs::path& p,const std::string& s){fs::create_directories(p.parent_path());std::ofstream o(p);o<<s;check(bool(o),"fixture write");}
int main()try{
    TempDir temp("/tmp","resonantia-task-disk-");const fs::path base=temp.path(),root=base/"tasks",runs=base/"runs";
    fs::create_directories(root);fs::create_directories(runs);
    TaskRecord audio;audio.id="audio-task";audio.kind="convert";audio.name="recording.wav";audio.state="succeeded";audio.result="work/result.wav";
    const auto work=root/audio.id/"work";put(work/"source.wav","source");put(work/"result.wav","result");put(work/"model.safetensors","redundant");put(work/"output.wav","duplicate");put(work/"notes.txt","user notes");put(root/audio.id/"task.log","retained log");
    audio.parameters["result_sha256"]=voxmutatio::io::sha256_file(work/"result.wav");
    put(work/"result.wav","broken");rejects([&]{prune_task_copies(root,runs,audio);});
    check(fs::exists(work/"output.wav")&&fs::exists(work/"model.safetensors"),"corrupt result lost recovery copies");put(work/"result.wav","result");
    put(runs/"FormalVoice/model.safetensors",std::string(10000,'f'));
    auto usage=measure_task_storage(root,runs,{audio});check(usage.managed_bytes==usage.tasks.at(audio.id),"unmanaged voice entered task budget");check(usage.available_bytes>0,"filesystem space missing");
    TaskStorageLimits limits;limits.managed_bytes=usage.managed_bytes;limits.task_bytes=usage.managed_bytes+1;limits.reserve_bytes=0;
    check(!storage_limit_reason(usage,limits).empty(),"managed budget boundary ignored");limits.managed_bytes++;
    check(storage_limit_reason(usage,limits,audio.id).empty(),"inside budget rejected");limits.task_bytes=usage.tasks.at(audio.id);check(!storage_limit_reason(usage,limits,audio.id).empty(),"task budget ignored");
    limits.task_bytes++;limits.reserve_bytes=usage.available_bytes+1;check(!storage_limit_reason(usage,limits).empty(),"free-space floor ignored");
    for(const auto* state:{"running","paused","interrupted","failed","cancelled"}){auto r=audio;r.state=state;rejects([&]{prune_task_copies(root,runs,r);});}
    check(prune_task_copies(root,runs,audio)==18,"redundant copy count");check(prune_task_copies(root,runs,audio)==0,"cleanup is not idempotent");
    check(read_file(work/"source.wav")=="source"&&read_file(work/"result.wav")=="result"&&read_file(work/"notes.txt")=="user notes","cleanup deleted protected content");
    audio.parameters["output_kind"]="mix";put(work/"output.wav","unique converted vocal");put(work/"cover.wav","redundant mixed copy");
    check(prune_task_copies(root,runs,audio)==20,"mix cleanup removed unique vocal");
    check(read_file(work/"output.wav")=="unique converted vocal","mix lost vocal output");
    audio.parameters["output_kind"]="vocal";fs::remove(work/"output.wav");
    fs::create_symlink(runs/"FormalVoice/model.safetensors",work/"model.safetensors");rejects([&]{prune_task_copies(root,runs,audio);});rejects([&]{measure_task_storage(root,runs,{audio});});fs::remove(work/"model.safetensors");
    fs::create_hard_link(work/"source.wav",work/"output.wav");rejects([&]{prune_task_copies(root,runs,audio);});fs::remove(work/"output.wav");
    TaskRecord training;training.id="train-task";training.name="TrainFixture";training.kind="training";training.state="succeeded";training.parameters["recovery"]="training-full-v1";
    const auto full=runs/training.name/"full_training",tw=root/training.id/"work";
    put(full/"web-task",training.id);put(full/"result/model.safetensors","model");put(full/"result/model.index","index");put(full/"checkpoints/state.safetensors","unique state");put(full/"checkpoints/candidate.safetensors","candidate");put(full/"clips/clip.wav","fixed clip");put(tw/"source/clip.wav","source clip");put(tw/"generator.safetensors","generator");put(tw/"discriminator.safetensors","discriminator");
    auto combined=measure_task_storage(root,runs,{audio,training});check(combined.tasks.at(training.id)>22,"owned training assets were not counted");
    check(prune_task_copies(root,runs,training)==22,"training snapshots not cleaned");
    check(read_file(full/"checkpoints/state.safetensors")=="unique state"&&read_file(full/"checkpoints/candidate.safetensors")=="candidate"&&read_file(tw/"source/clip.wav")=="source clip","training recovery/source deleted");
    put(full/"web-task","wrong-task");rejects([&]{measure_task_storage(root,runs,{audio,training});});rejects([&]{prune_task_copies(root,runs,training);});
    put(full/"web-task",training.id);fs::remove(work/"result.wav");rejects([&]{prune_task_copies(root,runs,audio);});
    std::atomic<bool> writer_failed=false;
    std::jthread writer([&](std::stop_token stop){try{while(!stop.stop_requested()){
        for(int n=0;n<16;++n)put(work/("moving"+std::to_string(n)+".tmp"),"published atomically");
        for(int n=0;n<16;++n)fs::rename(work/("moving"+std::to_string(n)+".tmp"),work/("moving"+std::to_string(n)+".bin"));
    }}catch(...){writer_failed=true;}});
    std::string scan_failure;
    try{for(int n=0;n<500;++n)(void)measure_task_storage(root,runs,{audio,training});}
    catch(const std::exception& e){scan_failure=e.what();}
    writer.request_stop();writer.join();check(!writer_failed,"atomic writer failed");
    check(scan_failure.empty(),("normal atomic publication broke disk monitoring: "+scan_failure).c_str());
    std::cout<<"storage boundaries, ownership, no-follow scans and protected cleanup passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
