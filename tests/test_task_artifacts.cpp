#include "tools/web/task_artifacts.h"
#include "tools/web/workflow_identity.h"
#include "tools/app_support/operation_checkpoint.h"
#include "tools/web/workflow_support.h"
#include <fstream>
#include <iostream>
using namespace resonantia::web;
namespace fs=std::filesystem;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,"accepted invalid artifact");}
void put(const fs::path& p,const std::string& s){std::ofstream f(p);f<<s;check(bool(f),"write fixture");}
int main(int argc,char** argv)try {
    if(argc==3&&std::string(argv[1])=="--emit-identities") {
        WebWorkflowIdentity web("web-audio-v1");web.field("recipe","fixed");web.file("input",argv[2]);
        voxmutatio::app_support::OperationIdentity native("conversion");native.field("recipe","fixed");native.file("input",argv[2]);
        std::cout<<web.finish()<<'\n'<<native.finish()<<'\n';return 0;
    }
    {
        TempDir fixture("/tmp","web-identity-release-");const fs::path dir=fixture.path();put(dir/"input.wav","fixed bytes");
        fs::copy_file("/proc/self/exe",dir/"updated-web");{std::ofstream changed(dir/"updated-web",std::ios::app|std::ios::binary);changed.put('X');}
        PosixProcessRunner runner;
        for(const auto& [exe,log]:std::vector<std::pair<std::string,std::string>>{{"/proc/self/exe","original.log"},{(dir/"updated-web").string(),"updated.log"}}) {
            // /proc/self/exe in a child would name the runner before exec, so
            // resolve the current test executable before passing argv.
            ProcessSpec p;p.argv={exe=="/proc/self/exe"?fs::read_symlink(exe).string():exe,"--emit-identities",(dir/"input.wav").string()};p.log_path=(dir/log).string();check(runner.run(p).success(),"identity producer failed");
        }
        std::istringstream a(read_file(dir/"original.log")),b(read_file(dir/"updated.log"));std::string wa,na,wb,nb;a>>wa>>na;b>>wb>>nb;
        check(wa==wb,"Web-only binary update invalidated unchanged task dependencies");
        check(na!=nb,"native CLI lost executable binding");
    }
    {
        TempDir fixture("/tmp","web-legacy-identity-");const fs::path dir=fixture.path();put(dir/"input.wav","fixed bytes");put(dir/"vc_train","native program v1");
        // Frozen independent SHA fixtures reproduce the deployed legacy format.
        const std::vector<std::string> legacy_audio{"178c62a2f198c7c24f8754e1e9185a8864f25997db81cea504113239e7e2c00e","14af22caf352381a201a4193ac4a667e5ece968efaa383a083311bee4f3990d7"};
        const std::vector<std::string> legacy_training{"0e40f6d32b42813aa98bff7080278aeb121ce53f090f4e6e001eef95e9c656b7","8ce7e78c2122ed21a626f65bef527c4cdacafdef9c12418db6cc861273ffe62e"};
        WebWorkflowIdentity audio("web-audio-v1");audio.field("recipe","fixed");audio.file("input",dir/"input.wav");check(audio.legacy()==legacy_audio,"legacy audio reference mismatch");
        WebWorkflowIdentity clips("web-training-clips-v1");clips.field("name","clip.wav");clips.file("wav",dir/"input.wav");
        WebWorkflowIdentity training("web-training-full-v1");training.field("source",clips);training.field("recipe","fixed");training.file("vc_train",dir/"vc_train");check(training.legacy()==legacy_training,"nested training producer identity mismatch");
        for(std::size_t version=0;version<2;++version) {
            const auto work=dir/std::to_string(version);fs::create_directory(work);put(work/"completed.wav","verified stage");
            TaskArtifacts old(work,legacy_audio[version]);old.commit("conversion",work/"completed.wav");
            const auto original=read_file(work/"workflow.identity"),stage=read_file(work/"stage.conversion.done");
            TaskArtifacts current(work,audio.finish(),audio.legacy());check(current.identity()==legacy_audio[version],"lost exact matched legacy identity");
            check(current.completed("conversion").has_value(),"legacy stage not reusable");current.commit("conversion",work/"completed.wav");
            check(original==read_file(work/"workflow.identity")&&stage==read_file(work/"stage.conversion.done"),"compatibility rewrote legacy evidence");
            WebWorkflowIdentity changed("web-audio-v1");changed.field("recipe","changed");changed.file("input",dir/"input.wav");rejects([&]{TaskArtifacts invalid(work,changed.finish(),changed.legacy());});
        }
        put(dir/"input.wav","changed source");WebWorkflowIdentity changed("web-audio-v1");changed.field("recipe","fixed");changed.file("input",dir/"input.wav");rejects([&]{TaskArtifacts invalid(dir/"0",changed.finish(),changed.legacy());});
        put(dir/"vc_train","changed native program");WebWorkflowIdentity changed_training("web-training-full-v1");changed_training.field("source",clips);changed_training.field("recipe","fixed");changed_training.file("vc_train",dir/"vc_train");check(changed_training.finish()!=training.finish()&&changed_training.legacy()!=legacy_training,"changed native program accepted");
        fs::create_directory(dir/"unknown");TaskArtifacts unknown(dir/"unknown",std::string(64,'f'));rejects([&]{TaskArtifacts invalid(dir/"unknown",audio.finish(),audio.legacy());});
    }
    TempDir temp("/tmp","resonantia-task-artifacts-");fs::path root=temp.path();const std::string id(64,'a');
    check(!TaskArtifacts::retained_output(root,"conversion")&&!fs::exists(root/"workflow.identity"),"playback created missing workflow marker");
    {
        TaskArtifacts a(root,id);put(root/"output.wav","completed bytes");
        check(!a.completed("conversion"),"file existence treated as completion");
        a.commit("conversion",root/"output.wav");check(a.completed("conversion")==root/"output.wav","commit failed");
        const auto marker=read_file(root/"workflow.identity");
        check(TaskArtifacts::retained_output(root,"conversion")==root/"output.wav","read-only playback lost saved stage");
        check(marker==read_file(root/"workflow.identity"),"playback rewrote identity");
        a.commit("conversion",root/"output.wav");
        fs::copy_file(root/"stage.conversion.done",root/"stage.cleanup.done");
        rejects([&]{a.completed("cleanup");});
        fs::remove(root/"stage.cleanup.done");
        rejects([&]{a.completed("../outside");});
        rejects([&]{a.commit("other",root/"../outside");});
        a.publish_result(root/"output.wav");a.publish_result(root/"output.wav");
        check(read_file(root/"result.wav")=="completed bytes","result publication retry");
    }
    rejects([&]{TaskArtifacts wrong(root,std::string(64,'b'));});
    TaskArtifacts restarted(root,id);check(restarted.completed("conversion").has_value(),"lost stage after restart");
    put(root/"output.wav","corrupted bytes");rejects([&]{restarted.completed("conversion");});
    rejects([&]{TaskArtifacts::retained_output(root,"conversion");});
    check(read_file(root/"result.wav")=="completed bytes","corrupt stage changed published result");
    fs::remove(root/"output.wav");fs::create_symlink(root/"result.wav",root/"output.wav");
    rejects([&]{restarted.completed("conversion");});
    rejects([&]{TaskArtifacts::retained_output(root,"conversion",false);});
    fs::remove(root/"stage.conversion.done");fs::create_directory(root/"stage.conversion.done.pending");
    rejects([&]{restarted.commit("conversion",root/"result.wav");});
    check(!restarted.completed("conversion"),"failed marker write became complete");
    std::cout<<"task stage identity, immutable completion and atomic result checks passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
