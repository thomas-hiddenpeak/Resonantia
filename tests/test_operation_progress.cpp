// Real short human audio and model assets. Progress must not change either
// conversion samples or Roformer overlap-add samples; no full songs or training.
#include "tools/web/workflow_support.h"
#include "tools/app_support/progress_protocol.h"
#include "voxmutatio/io/audio_io.h"
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <map>
#include <stdexcept>
using namespace resonantia::web;
using namespace voxmutatio::app_support;
namespace fs=std::filesystem;
namespace {
void check(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void verify_events(const std::vector<ProgressMessage>& events,const std::string& phase,const std::string& unit,std::uint64_t minimum_total) {
    bool found=false;std::uint64_t last=0,total=0;
    for(const auto& e:events)if(e.phase==phase) {
        check(e.unit==unit&&e.completed>=last&&e.completed<=e.total,"invalid observed domain counters");
        if(found)check(e.total==total,"domain total changed");
        last=e.completed;total=e.total;found=true;
    }
    check(found&&total>=minimum_total&&last==total,"missing completed "+phase+" boundary");
}
}
int main(int argc,char** argv)try {
    check(argc==8,"usage: test convert preprocess hubert rmvpe generator source separation_dir");
    TempDir temp("/tmp","resonantia-progress-real-");const fs::path root=temp.path();
    auto audio=voxmutatio::io::read_audio(argv[6],40000);
    check(audio&&audio->data.size()>=240000,"real source must contain six seconds");
    const auto input=root/"input.wav";
    check(voxmutatio::io::write_audio(input.string(),audio->data.data(),240000,40000),"cannot prepare real short input");
    const std::vector<std::string> conversion={argv[1],"--hubert",argv[3],"--rmvpe",argv[4],"--model",argv[5],
        "--version","v2","--sr","40000","--input",input.string(),"--rms-mix","0","--filter-radius","3"};
    PosixProcessRunner runner;
    std::vector<ProgressMessage> events;
    for(int monitored=0;monitored<2;++monitored) {
        ProcessSpec spec;spec.argv=conversion;spec.log_path=root/("convert"+std::to_string(monitored)+".log");
        spec.argv.insert(spec.argv.end(),{"--output",(root/("converted"+std::to_string(monitored)+".wav")).string()});
        if(monitored)spec.argv.push_back("--progress");
        spec.observe_line=[&](std::string_view line){if(auto p=parse_progress_message(line))events.push_back(*p);};
        auto result=runner.run(spec);check(result.success(),"conversion failed: "+log_tail(spec.log_path,4000));
        if(!monitored)check(events.empty(),"progress emitted without opt-in");
    }
    verify_events(events,"synthesis","samples16k",96000);verify_events(events,"f0","frames100hz",600);
    check(read_file(root/"converted0.wav")==read_file(root/"converted1.wav"),"progress changed conversion WAV bytes");
    events.clear();
    for(int monitored=0;monitored<2;++monitored) {
        ProcessSpec spec;spec.log_path=root/("cleanup"+std::to_string(monitored)+".log");
        spec.argv={argv[2],"--input",input.string(),"--output-dir",(root/("clean"+std::to_string(monitored))).string(),
            "--sr","44100","--preserve-stereo","--no-slice","--deharmony","--sep-dir",argv[7]};
        if(monitored)spec.argv.push_back("--progress");
        spec.observe_line=[&](std::string_view line){if(auto p=parse_progress_message(line))events.push_back(*p);};
        auto result=runner.run(spec);check(result.success(),"cleanup failed: "+log_tail(spec.log_path,4000));
        if(!monitored)check(events.empty(),"cleanup progress emitted without opt-in");
    }
    verify_events(events,"deharmony","chunks",2);verify_events(events,"files","files",1);
    const auto a=voxmutatio::io::read_stereo_audio((root/"clean0/input.wav").string());
    const auto b=voxmutatio::io::read_stereo_audio((root/"clean1/input.wav").string());
    check(a&&b&&a->data.size()==b->data.size(),"cleanup frame count changed");
    // GPU iSTFT uses float atomicAdd (src/separation/stft.cu); repeat executions
    // can differ in the last float bits even without an observer. This is far
    // stricter than reference-model alignment and is not a quality tolerance.
    double squared_error=0,energy=0,maximum=0;
    for(std::size_t i=0;i<a->data.size();++i) {
        check(std::isfinite(b->data[i]),"cleanup non-finite output");
        const double d=double(a->data[i])-b->data[i];squared_error+=d*d;energy+=double(a->data[i])*a->data[i];
        maximum=std::max(maximum,std::abs(d));
    }
    const double relative=energy?std::sqrt(squared_error/energy):std::sqrt(squared_error);
    std::cout<<"cleanup relative L2="<<relative<<" max_abs="<<maximum<<'\n';
    check(relative<1e-6&&maximum<2e-6,"observer changed cleanup beyond float reduction roundoff");
    std::cout<<"Six-second real conversion and deharmony: counters complete; conversion bytes identical; cleanup within float roundoff\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
