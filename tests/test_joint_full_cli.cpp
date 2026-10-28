// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
// Real assets; exercises CLI cache, epoch LR, cooperative stop and whole-job resume.
#include "tools/web/workflow_support.h"
#include "tools/app_support/joint_feature_cache.h"
#include "voxmutatio/io/audio_io.h"
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <map>
#include <thread>
using namespace voxmutatio;
using namespace voxmutatio::app_support::joint_full;
using namespace resonantia::web;
namespace fs=std::filesystem;
namespace {
std::string read(const fs::path& p){std::ifstream f(p);return {std::istreambuf_iterator<char>(f),{}};}
fs::path latest(const fs::path& dir){std::vector<fs::path> files;for(const auto& e:fs::directory_iterator(dir))
    if(e.path().filename().string().starts_with("state_step_")&&e.path().extension()==".safetensors")files.push_back(e.path());
    REQUIRE_FALSE(files.empty());std::sort(files.begin(),files.end());return files.back();}
RunState state_in(const fs::path& path){io::SafetensorsLoader loader;REQUIRE(loader.load(path));
    const auto* t=loader.get_tensor("state");REQUIRE(t);std::string text;
    for(std::size_t i=0;i<t->data_nbytes/4;++i){float x;std::memcpy(&x,loader.data("state")+i*4,4);text.push_back(char(x));}
    std::istringstream in(text);std::string magic,rng,run;int v,speaker,steps;float lr;
    REQUIRE(bool(in>>magic>>v>>speaker>>steps>>lr>>std::quoted(rng)>>std::quoted(run)));
    auto state=RunState::parse(run);REQUIRE(state.step==steps);REQUIRE(state.learning_rate()==lr);return state;}
}
TEST_CASE("Full CLI resume preserves real-audio epoch order, model and optimizer", "[full-cli]") {
    const auto executable=std::getenv("RESONANTIA_VC_TRAIN");REQUIRE(executable);
    TempDir root(fs::temp_directory_path(),"resonantia-joint-full-cli-");const fs::path dir=root.path();
    const auto clips=dir/"clips",cache=dir/"cache",a=dir/"continuous",b=dir/"resumed",c=dir/"cancelled";
    fs::create_directory(clips);const auto source=io::read_audio("fixtures/speech_librispeech.wav",40000);REQUIRE(source);REQUIRE(source->data.size()>=102400);
    REQUIRE(io::write_audio((clips/"a.wav").string(),source->data.data(),51200,40000));
    const auto token=dir/"cancel";{std::ofstream(token)<<'0';}
    auto spec_for=[&](const fs::path& state,const fs::path& log){ProcessSpec spec;spec.log_path=log;
      spec.argv={executable,"--gan","--gan-profile","joint-full","--epochs","6","--hubert","../models/hubert_base",
        "--rmvpe","../models/rmvpe.safetensors","--pretrained","../models/pretrained_v2/pretrained_v2/f0G40k.safetensors",
        "--dmodel","../models/pretrained_v2/pretrained_v2/f0D40k.safetensors","--target",clips.string(),"--cache-dir",cache.string(),
        "--checkpoint-dir",state.string(),"--out",(state/"output.safetensors").string(),"--cancel-file",token.string()};return spec;};
    const auto continuous=spec_for(a,dir/"continuous.log");const auto result=PosixProcessRunner{}.run(continuous);
    INFO(read(continuous.log_path));REQUIRE(result.success());REQUIRE(state_in(latest(a)).step==6);
    const auto fork=a/"state_step_0000000005.safetensors";REQUIRE(fs::exists(fork));
    REQUIRE_FALSE(fs::exists(a/"state_step_0000000000.safetensors")); // Rolling retention.
    // Both sides start step 6 from the EXACT SAME prefix, not two independent
    // fresh runs. Atomic CUDA reductions can differ before any checkpoint; a
    // multi-step divergence between independent histories does not isolate resume.
    auto resumed=spec_for(b,dir/"resumed.log");resumed.argv.insert(resumed.argv.end(),{"--resume",fork.string()});
    const auto finished=PosixProcessRunner{}.run(resumed);INFO(read(resumed.log_path));REQUIRE(finished.success());
    REQUIRE(state_in(latest(a)).serialize()==state_in(latest(b)).serialize());
    REQUIRE(read(resumed.log_path).find("cache_hits=1/1")!=std::string::npos);
    {
    io::SafetensorsLoader x,y;REQUIRE(x.load(latest(a)));REQUIRE(y.load(latest(b)));
    struct Metric {double error=0,reference=0;std::size_t count=0;};
    std::map<std::string,Metric> modules;
    for(const auto& name:x.tensor_names()) {
      const auto* tx=x.get_tensor(name);const auto* ty=y.get_tensor(name);REQUIRE(ty);REQUIRE(tx->shape==ty->shape);
      double error=0,reference=0;bool finite=true;const auto n=tx->data_nbytes/4;
      for(std::size_t i=0;i<n;++i){float av,bv;std::memcpy(&av,x.data(name)+i*4,4);std::memcpy(&bv,y.data(name)+i*4,4);
        finite=finite&&std::isfinite(bv);const double d=double(av)-bv;error+=d*d;reference+=double(av)*av;}
      INFO(name);REQUIRE(finite);
      if(name=="state")REQUIRE(error==0);
      else {
        // Exactly the approved independent test's module groups: enc_p/enc_q/
        // flow/dec/emb_g for G; each discriminator separately for D.
        const auto start=name.find("model.");REQUIRE(start!=std::string::npos);
        const auto model_name=name.substr(start);
        const auto end=model_name.find('.',name.starts_with("D.")?model_name.find('.',6)+1:6);
        const auto key=name.substr(0,start)+model_name.substr(0,end);
        auto& group=modules[key];group.error+=error;group.reference+=reference;group.count+=n;
        if(name.find(".p.")!=std::string::npos && std::sqrt(error/n)>=1e-6)
          std::printf("[resume diagnostic tensor] %s RMS=%.9g\n",name.c_str(),std::sqrt(error/n));
      }
    }
    for(const auto& [name,metric]:modules) {
      const auto rms=std::sqrt(metric.error/metric.count);
      const auto relative=metric.reference?std::sqrt(metric.error/metric.reference):0.;
      std::printf("[resume module] %s RMS=%.9g relative=%.9g\n",name.c_str(),rms,relative);INFO(name);
      if(name.find(".p.")!=std::string::npos)CHECK(rms<1e-6);
      else if(!metric.reference)CHECK(metric.error==0);
      else CHECK(relative<1e-3);
    }
    }
    // A separate real cancellation verifies persistence, no final export and
    // continuation to the exact epoch/crop RNG state, without conflating an
    // independently initialized training history with checkpoint preservation.
    auto interrupted=spec_for(c,dir/"interrupted.log");
    auto future=std::async(std::launch::async,[&]{return PosixProcessRunner{}.run(interrupted);});
    bool requested=false;
    while(future.wait_for(std::chrono::milliseconds(20))!=std::future_status::ready) {
      if(!requested&&read(interrupted.log_path).find("joint-full step=1/6")!=std::string::npos){
        std::fstream channel(token,std::ios::in|std::ios::out);channel.put('1');channel.flush();REQUIRE(channel.good());requested=true;}
    }
    const auto stopped=future.get();INFO(read(interrupted.log_path));REQUIRE(requested);
    REQUIRE(stopped.termination==ProcessTermination::kExited);REQUIRE(stopped.exit_code==2);
    REQUIRE_FALSE(fs::exists(c/"output.safetensors"));
    const auto saved=latest(c);const auto saved_state=state_in(saved);REQUIRE(saved_state.step>=1);REQUIRE(saved_state.step<6);
    {std::ofstream(token)<<'0';}
    auto continued=spec_for(c,dir/"continued.log");continued.argv.insert(continued.argv.end(),{"--resume",saved.string()});
    REQUIRE(PosixProcessRunner{}.run(continued).success());
    REQUIRE(state_in(latest(a)).serialize()==state_in(latest(c)).serialize());
    // The caller must reject a valid checkpoint used with a different dataset,
    // leaving its old inference output untouched.
    const auto before=sha256_file(b/"output.safetensors");
    REQUIRE(io::write_audio((clips/"a.wav").string(),source->data.data()+1600,51200,40000));
    auto mismatch=spec_for(b,dir/"mismatch.log");mismatch.argv.insert(mismatch.argv.end(),{"--resume",latest(b).string()});
    REQUIRE_FALSE(PosixProcessRunner{}.run(mismatch).success());REQUIRE(sha256_file(b/"output.safetensors")==before);
}
