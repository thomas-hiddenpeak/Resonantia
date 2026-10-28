// Existing option parsing and actual pre-cancel CLI contract (CUDA build required).
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <sys/stat.h>
#include <thread>
#include <atomic>
#include <chrono>
#include "tools/app_support/training_cancel.h"
#include "tools/web/workflow_support.h"
#include "tools/app_support/vc_train_app.h"
namespace fs=std::filesystem;
using namespace resonantia::web;
using namespace voxmutatio::app_support;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::string read(const fs::path& path){std::ifstream f(path,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
int main(int argc,char** argv){
 try{
  require(argc==2,"expected vc_train executable");
  // Even a regression in early cancellation cannot load a device/model here.
  setenv("CUDA_VISIBLE_DEVICES","",1);
  TempDir root(fs::temp_directory_path(),"resonantia-training-cancel-");
  const fs::path dir=root.path(),token=dir/"cancel",output=dir/"old-model";
  {std::ofstream(token)<<'1';std::ofstream(output)<<"old-model";}
  using namespace resonantia::training_control;
  std::ofstream(dir/"invalid")<<'x';
  // Parsing remains compatible without the option; optional path cannot be empty/repeated.
  auto parse=[&](std::vector<std::string> args,VcTrainOptions& options){
   std::vector<char*> av;for(auto& arg:args)av.push_back(arg.data());std::ostringstream out,err;
   return parse_vc_train_options(static_cast<int>(av.size()),av.data(),options,out,err);
  };
  std::vector<std::string> args={"vc_train","--hubert","h","--rmvpe","r","--pretrained","p","--target","t","--out","o"};
  VcTrainOptions defaults;require(parse(args,defaults)==ParseStatus::kRun && defaults.cancel_file.empty() && defaults.steps==300,"default parsing changed");
  auto joint=args;joint.insert(joint.end(),{"--gan","--gan-profile","joint"});
  VcTrainOptions jo;
  require(parse(joint,jo)==ParseStatus::kRun,"joint profile parsing failed");
  require(jo.steps==50 && jo.segment_frames==32 && jo.epochs==0 && jo.learning_rate==1e-4f &&
          jo.learning_rate_decay==1.f && jo.synthesis_mode==voxmutatio::SynthesisMode::kSampled,
          "joint bounded defaults do not match qualified recipe");
  for(const auto& extra:std::vector<std::vector<std::string>>{{"--steps","51"},{"--epochs","1"},
      {"--seg","40"},{"--synthesis-mode","mean"},{"--lr-decay","0.9"},
      {"--gan-profile","joint"}}) {
    auto invalid=joint;invalid.insert(invalid.end(),extra.begin(),extra.end());VcTrainOptions rejected;
    require(parse(invalid,rejected)==ParseStatus::kExitFailure,"unsupported joint combination accepted");
  }
  auto without_gan=args;without_gan.insert(without_gan.end(),{"--gan-profile","joint"});VcTrainOptions no_gan;
  require(parse(without_gan,no_gan)==ParseStatus::kExitFailure,"joint profile without GAN accepted");
  auto with=args;with.insert(with.end(),{"--cancel-file",token.string()});VcTrainOptions valid;
  require(parse(with,valid)==ParseStatus::kRun && valid.cancel_file==token.string(),"optional token parse failed");
  with.insert(with.end(),{"--cancel-file",token.string()});VcTrainOptions twice;
  require(parse(with,twice)==ParseStatus::kExitFailure,"duplicate token accepted");
  auto empty=args;empty.insert(empty.end(),{"--cancel-file",""});VcTrainOptions blank;
  require(parse(empty,blank)==ParseStatus::kExitFailure,"empty token accepted");
  ProcessSpec spec;
  spec.argv={argv[1],"--hubert",(dir/"absent-hubert").string(),"--rmvpe",(dir/"absent-rmvpe").string(),"--pretrained",(dir/"absent-G").string(),"--target",(dir/"absent-audio").string(),"--out",output.string(),"--cancel-file",token.string()};
  spec.log_path=dir/"cli.log";
  const auto result=PosixProcessRunner{}.run(spec);
  require(result.termination==ProcessTermination::kExited && result.exit_code==2 && result.io_error==0,"pre-cancel CLI must normally exit 2 before CUDA/model initialization");
  require(read(output)=="old-model","pre-cancel replaced old output");
  require(read(spec.log_path).find("Device:")==std::string::npos && read(spec.log_path).find("CUDA init")==std::string::npos,"pre-cancel touched CUDA initialization");
  auto joint_spec=spec;
  joint_spec.argv.insert(joint_spec.argv.end(),{"--gan","--gan-profile","joint"});
  const auto joint_result=PosixProcessRunner{}.run(joint_spec);
  require(joint_result.termination==ProcessTermination::kExited && joint_result.exit_code==2,
          "pre-cancel joint CLI must exit 2 before loading data/models");
  require(read(output)=="old-model","pre-cancel joint CLI replaced output");
  auto full=args;full.insert(full.end(),{"--gan","--gan-profile","joint-full","--cache-dir","cache","--checkpoint-dir","state"});
  VcTrainOptions fo;
  require(parse(full,fo)==ParseStatus::kRun,"joint-full parsing failed");
  require(fo.epochs==200 && fo.segment_frames==32 && fo.learning_rate==1e-4f && fo.learning_rate_decay==.999875f,
          "joint-full defaults differ from approved recipe");
  for(const auto& extra:std::vector<std::vector<std::string>>{{"--steps","50"},{"--epochs","0"},
      {"--seg","40"},{"--synthesis-mode","mean"},{"--resume",""},{"--cache-dir","cache2"}}) {
    auto invalid=full;invalid.insert(invalid.end(),extra.begin(),extra.end());VcTrainOptions rejected;
    require(parse(invalid,rejected)==ParseStatus::kExitFailure,"invalid full training options accepted");
  }
  for(const auto& extra:std::vector<std::vector<std::string>>{{"--resume","checkpoint"},
      {"--cache-dir","cache"},{"--checkpoint-dir","state"}}) {
    auto invalid=joint;invalid.insert(invalid.end(),extra.begin(),extra.end());VcTrainOptions rejected;
    require(parse(invalid,rejected)==ParseStatus::kExitFailure,"full-only option accepted in pilot");
  }
  auto full_spec=spec;
  full_spec.argv.insert(full_spec.argv.end(),{"--gan","--gan-profile","joint-full","--cache-dir","cache","--checkpoint-dir","state"});
  const auto full_result=PosixProcessRunner{}.run(full_spec);
  require(full_result.termination==ProcessTermination::kExited && full_result.exit_code==2,
          "pre-cancel full CLI must exit 2 before loading data/models");
  spec.argv.back()=(dir/"invalid").string();
  const auto bad=PosixProcessRunner{}.run(spec);
  require(bad.termination==ProcessTermination::kExited && bad.exit_code==1 && bad.io_error==0,"bad token exit is not error");
  require(read(output)=="old-model","bad channel changed old model");
  std::cout<<"cancel option parsing and actual pre-cancel CLI: exit 2 before CUDA/output mutation; passed\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"Training cancel contract: "<<e.what()<<'\n';return 1;}
}
