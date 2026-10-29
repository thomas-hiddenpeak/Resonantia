// Real six-second human speech, real models, explicit safe-boundary pause.
#include "voxmutatio/pipeline/pipeline.h"
#include "voxmutatio/separation/roformer.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/io/file_digest.h"
#include "tools/web/workflow_support.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace voxmutatio;
namespace fs=std::filesystem;
namespace {
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void aligned(const std::vector<float>& a,const std::vector<float>& b) {
    require(a.size()==b.size(),"restored audio length mismatch");
    double delta=0,energy=0,maximum=0;
    for(std::size_t i=0;i<a.size();++i){require(std::isfinite(b[i]),"nonfinite restored audio");
        const double d=double(a[i])-b[i];delta+=d*d;energy+=double(a[i])*a[i];maximum=std::max(maximum,std::abs(d));}
    const double relative=energy?std::sqrt(delta/energy):std::sqrt(delta);
    std::cout<<"restored relative L2="<<relative<<" max_abs="<<maximum<<std::endl;
    require(relative<1e-6&&maximum<2e-6,"resume exceeds float reduction roundoff");
}
}
int main(int argc,char** argv)try {
    require(argc==6,"usage: test hubert rmvpe generator speech separation_dir");
    resonantia::web::TempDir temp("/tmp","resonantia-checkpoint-real-");const fs::path root=temp.path();
    auto audio=io::read_audio(argv[4],40000);require(audio&&audio->data.size()>=240000,"need six seconds of real speech");
    const auto input=root/"input.wav";require(io::write_audio(input.string(),audio->data.data(),240000,40000),"prepare input");
    VCConfig cfg;cfg.hubert_model_path=argv[1];cfg.rmvpe_model_path=argv[2];cfg.synthesizer_model_path=argv[3];
    cfg.version=ModelVersion::kV2;cfg.rms_mix_rate=0;cfg.filter_radius=3;
    cfg.max_chunk_sec=1;cfg.chunk_query_sec=.1;cfg.chunk_pad_sec=.25;
    const auto identity=core::sha256("checkpoint-real-test-v1:1:.1:.25:rms0:filter3:v2:"+io::sha256_file(input)+
        io::sha256_file(argv[1])+io::sha256_file(argv[2])+io::sha256_file(argv[3]));
    using Mode=io::BlockCheckpoint::Mode;
    {
        pipeline::VoiceConversionPipeline p;require(p.init(cfg),"baseline init");
        auto r=p.convert_file_bounded(input.string(),(root/"baseline.wav").string());require(r.success,r.error_message);
    }
    const auto result=root/"restored.wav";{std::ofstream f(result);f<<"existing target";}
    bool stop=false;
    {
        io::BlockCheckpoint c(root/"conversion",identity,Mode::Create,[&]{return stop;});
        pipeline::VoiceConversionPipeline p;require(p.init(cfg),"pause init");
        const auto r=p.convert_file_bounded(input.string(),result.string(),0,[&](const OperationProgress& e){
            if(e.phase=="synthesis"&&e.completed>0&&e.completed<e.total)stop=true;
        },&c);
        require(stop&&r.paused&&!r.success,"conversion did not pause between chunks");
        require(resonantia::web::read_file(result)=="existing target","pause replaced target");
    }
    {
        io::BlockCheckpoint c(root/"conversion",identity,Mode::Resume);
        pipeline::VoiceConversionPipeline p;require(p.init(cfg),"resume init");
        const auto r=p.convert_file_bounded(input.string(),result.string(),0,{},&c);
        require(r.success&&!r.paused,r.error_message);require(c.restored_blocks()>=2,"F0 and completed synthesis were not reused");
        std::cout<<"conversion restored blocks="<<c.restored_blocks()<<std::endl;
    }
    require(resonantia::web::read_file(root/"baseline.wav")==resonantia::web::read_file(result),"restored PCM WAV differs from continuous run");
    {
        io::BlockCheckpoint c(root/"conversion",identity,Mode::Resume);
        pipeline::VoiceConversionPipeline p;require(p.init(cfg),"full replay init");
        const auto r=p.convert_file_bounded(input.string(),(root/"replayed.wav").string(),0,{},&c);
        require(r.success&&r.hubert_ms==0&&r.index_ms==0&&r.synth_ms==0,"completed replay recomputed expensive synthesis");
        require(resonantia::web::read_file(result)==resonantia::web::read_file(root/"replayed.wav"),"full replay changed WAV");
    }
    auto source=io::read_audio(input.string(),44100);require(bool(source),"separation input");
    std::vector<float> baseline,restored;const int frames=static_cast<int>(source->data.size());
    const auto sepidentity=core::sha256("real-roformer-karaoke-v1:"+io::sha256_file(input)+
        io::sha256_file(fs::path(argv[5])/"karaoke_roformer.safetensors"));
    {separation::Roformer p(argv[5],"karaoke_roformer");require(p.valid(),"baseline roformer init");
        baseline=p.separate_mono(source->data.data(),frames,44100);}
    stop=false;
    {
        io::BlockCheckpoint c(root/"separation",sepidentity,Mode::Create,[&]{return stop;});
        separation::Roformer p(argv[5],"karaoke_roformer");require(p.valid(),"pause roformer init");
        bool paused=false;
        try{(void)p.separate_mono(source->data.data(),frames,44100,[&](const OperationProgress& e){
            if(e.completed==1&&e.total>=2)stop=true;
        },&c);}catch(const io::OperationPaused&){paused=true;}
        require(paused&&stop,"roformer did not pause after first of multiple windows");
    }
    {
        io::BlockCheckpoint c(root/"separation",sepidentity,Mode::Resume);
        separation::Roformer p(argv[5],"karaoke_roformer");require(p.valid(),"resume roformer init");
        restored=p.separate_mono(source->data.data(),frames,44100,{},&c);
        require(c.restored_blocks()==1,"roformer did not reuse exactly the completed first window");aligned(baseline,restored);
    }
    {
        io::BlockCheckpoint c(root/"separation",sepidentity,Mode::Resume);
        separation::Roformer p(argv[5],"karaoke_roformer");require(p.valid(),"replay roformer init");
        const auto replay=p.separate_mono(source->data.data(),frames,44100,{},&c);
        require(c.restored_blocks()>=2&&replay==restored,"full overlap-add replay not bit-identical");
    }
    std::cout<<"Real conversion and Roformer safe pause/reopen/reuse passed; no full song or training\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
