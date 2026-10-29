#include "tools/web/task_store.h"
#include "tools/web/workflow_support.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace resonantia::web;
namespace fs = std::filesystem;

void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    check(threw, "invalid operation accepted");
}
TaskRecord record(const std::string& id) {
    TaskRecord r; r.id=id; r.kind="convert"; r.name="中文歌曲.mp3";
    r.created_ms=10; r.updated_ms=10;
    r.parameters={{"voice","Tom_Cybersinger"},{"workflow","song"}};
    return r;
}
int main() try {
    TempDir temp("/tmp","resonantia-task-store-");
    const auto root=fs::path(temp.path())/"tasks";
    {
        TaskStore store(root,2);
        rejects([&]{TaskStore second(root);});
        check(!store.get("task_0001"),"missing record");
        auto r=record("task_0001");
        check(store.save(r,0)&&r.revision==1,"create record");
        check(store.get(r.id)->parameters==r.parameters,"parameters roundtrip");
        check(store.get(r.id)->attempt==1,"initial execution attempt");
        auto old=r;
        r.state="running";r.stage="deharmony";r.completed=3;r.total=20;r.unit="chunks";
        r.updated_ms=20;r.progress_at_ms=20;r.attempt=2;
        check(store.save(r,1)&&r.revision==2,"update record");
        check(store.get(r.id)->attempt==2,"execution attempt was not persisted");
        old.state="cancelled";
        check(!store.save(old,1)&&store.get(r.id)->state=="running","stale writer changed state");
        auto duplicate=record(r.id);
        check(!store.save(duplicate,0),"duplicate ID replaced original");
        auto completed=record("task_0002");completed.state="succeeded";
        check(store.save(completed,0),"second record");
        auto extra=record("task_0003"); rejects([&]{store.save(extra,0);});
        check(store.list(0,1).size()==1&&store.list(1,1).size()==1&&store.list(2,1).empty(),"pagination");
        for (const auto& id : {"../outside","bad/name","bad\\name",".hidden","x"}) {
            rejects([&]{(void)store.get(id);});
            rejects([&]{(void)store.task_directory(id);});
        }
        auto malformed=r;malformed.state="invented";
        rejects([&]{store.save(malformed,2);});
        malformed=r;malformed.error.assign(65537,'x');rejects([&]{store.save(malformed,2);});
        malformed=r;malformed.completed=21;rejects([&]{store.save(malformed,2);});
        malformed=r;malformed.attempt=0;rejects([&]{store.save(malformed,2);});
        check(store.get(r.id)->revision==2,"invalid update changed old generation");
    }
    {
        TaskStore restarted(root,2);
        check(restarted.get("task_0001")->completed==3,"restart lost progress");
        check(restarted.recover_interrupted(30)==1,"startup must retire only live executions");
        const auto recovered=restarted.get("task_0001");
        check(recovered->state=="interrupted"&&recovered->completed==3&&recovered->stage=="deharmony",
              "startup lost recovery context");
        check(restarted.get("task_0002")->state=="succeeded","completed task changed after restart");
        check(restarted.recover_interrupted(40)==0,"startup recovery is not idempotent");
        auto record_path=restarted.task_directory("task_0001")/"record";
        const auto valid=read_file(record_path);
        auto malformed=valid;
        const auto counters=malformed.find("3 10 ");
        check(counters!=std::string::npos,"record counter fixture");
        malformed.replace(counters,1,"-1");
        {std::ofstream out(record_path);out<<malformed;}
        rejects([&]{(void)restarted.get("task_0001");});
        {std::ofstream out(record_path);out<<valid;}
        fs::remove(record_path);fs::create_symlink(root/"task_0002"/"record",record_path);
        rejects([&]{(void)restarted.get("task_0001");});
        fs::remove(record_path);
        std::ofstream(record_path)<<"unsupported version\n";
        rejects([&]{(void)restarted.get("task_0001");});
    }
    {
        TaskStore retained(fs::path(temp.path())/"retained",1);
        auto first=record("retained-first");first.state="succeeded";check(retained.save(first,0),"retention fixture");
        auto next=record("retained-next");rejects([&]{retained.save(next,0);});
        first.parameters["retention"]="pruned-v1";check(retained.save(first,first.revision),"cleanup metadata save");
        check(retained.save(next,0),"cleaned history did not free admission slot");
        auto duplicate=record(first.id);check(!retained.save(duplicate,0),"cleanup resurrected an old task ID");
        check(retained.list().size()==2,"cleanup lost history");
        auto bad=next;bad.parameters["retention"]="pruned-v1";rejects([&]{retained.save(bad,bad.revision);});
        bad=first;bad.parameters["retention"]="unknown";rejects([&]{retained.save(bad,bad.revision);});
    }
    std::cout<<"Persistent Web task store checks passed\n";
    return 0;
} catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
