#include "tools/web/workflow_support.h"
#include "tools/app_support/progress_protocol.h"
#include "tools/app_support/training_cancel.h"
#include "voxmutatio/io/audio_io.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
namespace fs=std::filesystem;
using namespace resonantia::web;
using namespace resonantia::training_control;
using voxmutatio::app_support::parse_progress_message;
namespace {
void check(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void aligned(const fs::path& a,const fs::path& b) {
    auto x=voxmutatio::io::read_stereo_audio(a.string()),y=voxmutatio::io::read_stereo_audio(b.string());
    check(x&&y&&x->data.size()==y->data.size(),"restored cleanup shape");
    double err=0,energy=0,peak=0;
    for(std::size_t i=0;i<x->data.size();++i){check(std::isfinite(y->data[i]),"nonfinite cleanup");
        double d=double(x->data[i])-y->data[i];err+=d*d;energy+=double(x->data[i])*x->data[i];peak=std::max(peak,std::abs(d));}
    double relative=energy?std::sqrt(err/energy):std::sqrt(err);
    std::cout<<"CLI cleanup resume relative L2="<<relative<<" max_abs="<<peak<<std::endl;
    check(relative<1e-6&&peak<2e-6,"CLI cleanup exceeds float reduction roundoff");
}
}
int main(int argc,char** argv)try {
    check(argc==8,"usage: test convert preprocess hubert rmvpe generator speech separation_dir");
    TempDir temp("/tmp","resonantia-checkpoint-cli-");const fs::path root=temp.path();PosixProcessRunner runner;
    auto call=[&](std::vector<std::string> args,const std::string& log,const std::function<void(std::string_view)>& callback={}) {
        ProcessSpec spec;spec.argv=std::move(args);spec.log_path=root/log;spec.observe_line=callback;return runner.run(spec);
    };
    for(int i=1;i<=2;++i){auto r=call({argv[i],"--help"},"help.log");check(r.success(),"help failed");
        auto help=read_file(root/"help.log");for(auto option:{"--checkpoint-dir","--resume","--pause-file"})check(help.find(option)!=help.npos,"missing checkpoint CLI contract");}
    auto audio=voxmutatio::io::read_audio(argv[6],40000);check(audio&&audio->data.size()>=240000,"need real six-second source");
    const auto input=root/"source.wav";check(voxmutatio::io::write_audio(input.string(),audio->data.data(),240000,40000),"prepare input");
    for(bool cleanup:{false,true}) {
        auto base=cleanup?std::vector<std::string>{argv[2],"--input",input.string(),"--no-slice","--preserve-stereo","--sr","44100","--deharmony","--sep-dir",argv[7]}:
            std::vector<std::string>{argv[1],"--input",input.string(),"--hubert",argv[3],"--rmvpe",argv[4],"--model",argv[5],"--version","v2","--sr","40000","--rms-mix","0","--filter-radius","3"};
        const auto prefix=cleanup?"cleanup":"conversion";const auto checkpoints=root/(std::string(prefix)+"_blocks");
        const auto target=root/(std::string(prefix)+"_output");const auto baseline=root/(std::string(prefix)+"_baseline");
        auto reference=base;reference.insert(reference.end(),{cleanup?"--output-dir":"--output",baseline.string()});
        check(call(reference,"reference.log").success(),"baseline failed: "+log_tail(root/"reference.log",2000));
        auto args=base;args.insert(args.end(),{cleanup?"--output-dir":"--output",target.string(),"--checkpoint-dir",checkpoints.string(),"--progress"});
        {
            TrainingCancelSource pause((root/(std::string(prefix)+".pause")).string());
            auto first=args;first.insert(first.end(),{"--pause-file",(root/(std::string(prefix)+".pause")).string()});
            bool requested=false;
            auto r=call(first,"pause.log",[&](std::string_view line){
                const auto p=parse_progress_message(line);
                if(p&&!requested&&((cleanup&&p->phase=="deharmony"&&p->completed==1&&p->total>=2)||
                    (!cleanup&&p->phase=="f0"&&p->completed>0))) {
                    check(pause.request()==CancelDelivery::kDelivered,"pause request failed");requested=true;
                }
            });
            check(requested&&r.exit_code==2&&r.io_error==0,"safe pause did not exit 2: "+log_tail(root/"pause.log",2000));
            check(!fs::exists(cleanup?target/"source.wav":target),"paused operation published output");
        }
        auto wrong=args;wrong.push_back("--resume");wrong.insert(wrong.end(),cleanup?std::initializer_list<std::string>{"--min-rms","0.05"}:std::initializer_list<std::string>{"--pitch","1"});
        auto refused=call(wrong,"mismatch.log");
        check(refused.exit_code==1&&read_file(root/"mismatch.log").find("identity mismatch")!=std::string::npos,"changed recipe accepted");
        auto resume=args;resume.push_back("--resume");check(call(resume,"resume.log").success(),"resume failed: "+log_tail(root/"resume.log",2000));
        if(cleanup)aligned(baseline/"source.wav",target/"source.wav");
        else check(read_file(baseline)==read_file(target),"CLI resumed conversion changed PCM WAV");
        const auto before=read_file(cleanup?target/"source.wav":target);
        check(call(args,"duplicate.log").exit_code==1,"duplicate new run overwrote checkpoint");
        check(read_file(cleanup?target/"source.wav":target)==before,"refused duplicate changed output");
        auto bad_channel=resume;bad_channel.insert(bad_channel.end(),{"--pause-file",(root/"missing.pause").string()});
        check(call(bad_channel,"bad-channel.log").exit_code==1,"invalid pause channel reported paused");
    }
    std::cout<<"CLI safe pause, new-process resume, identity rejection and error distinction passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
