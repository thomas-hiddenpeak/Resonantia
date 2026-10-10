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
  spec.argv.back()=(dir/"invalid").string();
  const auto bad=PosixProcessRunner{}.run(spec);
  require(bad.termination==ProcessTermination::kExited && bad.exit_code==1 && bad.io_error==0,"bad token exit is not error");
  require(read(output)=="old-model","bad channel changed old model");
  std::cout<<"cancel option parsing and actual pre-cancel CLI: exit 2 before CUDA/output mutation; passed\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"Training cancel contract: "<<e.what()<<'\n';return 1;}
}
