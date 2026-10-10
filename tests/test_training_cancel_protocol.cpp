// Existing cancellation protocol and natural helper cases, independent of the CUDA CLI.
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
namespace fs=std::filesystem;
using namespace resonantia::web;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::string read(const fs::path& path){std::ifstream f(path,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
int main(int argc,char** argv){
 try{
  using namespace resonantia::training_control;
  using namespace std::chrono_literals;
  if(argc==4 && std::string(argv[1])=="--host-step-helper") {
   const fs::path dir=argv[3];TrainingCancelReader reader(argv[2]);
   const auto outcome=run_training_iterations(2,reader,[&](int iteration){
    if(iteration!=0)return false;
    std::ofstream(dir/"entered")<<"fake-D";
    const auto deadline=std::chrono::steady_clock::now()+2s;
    while(!fs::exists(dir/"release") && std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(1ms);
    if(!fs::exists(dir/"release"))return false;
    std::ofstream(dir/"completed")<<"fake-G";
    return true;
   });
   if(outcome==TrainingResult::kCompleted)std::ofstream(dir/"exported")<<"unexpected-export";
   std::cout<<"host helper finished naturally\n";
   return static_cast<int>(outcome);
  }
  require(argc==1,"unexpected protocol test argument");
  setenv("CUDA_VISIBLE_DEVICES","",1);
  TempDir root(fs::temp_directory_path(),"resonantia-training-protocol-");
  const fs::path dir=root.path(),token=dir/"cancel";
  std::ofstream(token)<<'1';
  TrainingCancelReader disabled("");
  require(disabled.poll()==TrainingResult::kCompleted,"default cancellation must be disabled");
  TrainingCancelReader missing((dir/"missing").string());
  require(missing.poll()==TrainingResult::kError,"missing requested channel is an error");
  for(const std::string bytes:{"", "00", "x"}) {
   std::ofstream(dir/"invalid",std::ios::trunc)<<bytes;
   TrainingCancelReader invalid((dir/"invalid").string());
   require(invalid.poll()==TrainingResult::kError,"malformed token accepted");
  }
  fs::create_symlink(token,dir/"link");
  TrainingCancelReader link((dir/"link").string());
  require(link.poll()==TrainingResult::kError,"symlink token accepted");
  require(mkfifo((dir/"fifo").c_str(),0600)==0,"fifo fixture failed");
  TrainingCancelReader fifo((dir/"fifo").string());
  require(fifo.poll()==TrainingResult::kError,"nonregular token accepted");
  const auto controlled=dir/"controlled";
  TrainingCancelSource source(controlled.string());
  struct stat st{};require(stat(controlled.c_str(),&st)==0 && (st.st_mode&0777)==0600,"token permissions changed");
  bool duplicate=false;try{TrainingCancelSource other(controlled.string());}catch(...){duplicate=true;}
  require(duplicate && read(controlled)=="0","token must be exclusive and initially zero");
  TrainingCancelReader reader(controlled.string());
  require(reader.poll()==TrainingResult::kCompleted,"initial token unexpectedly cancelled");
  std::vector<int> order;
  const auto stopped=run_training_iterations(3,reader,[&](int i){
   order.push_back(10+i); // Fake complete GAN D/G pair: no poll inside the body.
   require(source.request()==CancelDelivery::kDelivered,"request failed");
   order.push_back(20+i);
   return true;
  });
  require(stopped==TrainingResult::kCancelled && order==std::vector<int>{10,20},"complete step or next-step boundary broken");
  require(source.request()==CancelDelivery::kDelivered && read(controlled)=="1","duplicate request is not idempotent");
  require(source.close(),"source close failed");
  fs::remove(controlled);
  TrainingCancelSource replacement(controlled.string());
  require(source.request()==CancelDelivery::kClosed && read(controlled)=="0","retired token wrote to replacement");
  require(reader.poll()==TrainingResult::kCancelled,"reader reopened a replacement path");
  TrainingCancelReader next(controlled.string());
  const auto failed=run_training_iterations(1,next,[&](int){replacement.request();return false;});
  require(failed==TrainingResult::kError,"body error must outrank concurrent cancellation");
  const auto lastpath=dir/"last-step";TrainingCancelSource lastsource(lastpath.string());
  TrainingCancelReader lastreader(lastpath.string());int exported=0;
  const auto last=run_training_iterations(1,lastreader,[&](int){lastsource.request();return true;});
  if(last==TrainingResult::kCompleted)++exported;
  require(last==TrainingResult::kCancelled && exported==0,"last iteration cancel allowed export");
  require(run_training_iterations(2,disabled,[&](int){return true;})==TrainingResult::kCompleted,"default loop changed");
  // Concurrent writer/close is bounded; close prevents any later path reuse.
  for(int i=0;i<32;++i){
   const auto path=dir/("race-"+std::to_string(i));TrainingCancelSource raced(path.string());
   std::jthread writer([&]{raced.request();});require(raced.close(),"concurrent close failed");writer.join();
   fs::remove(path);TrainingCancelSource fresh(path.string());
   require(raced.request()==CancelDelivery::kClosed && read(path)=="0","late request reached new channel");
  }
  {
   const auto helperdir=dir/"helper";fs::create_directory(helperdir);
   TrainingCancelSource channel((helperdir/"cancel").string());
   ProcessSpec helper;helper.argv={argv[0],"--host-step-helper",(helperdir/"cancel").string(),helperdir.string()};
   helper.log_path=helperdir/"log";ProcessResult outcome;std::atomic<bool> returned{false};std::exception_ptr error;
   std::jthread runner([&]{try{outcome=PosixProcessRunner{}.run(helper);}catch(...){error=std::current_exception();}returned=true;});
   // Always release the helper before joining after an assertion failure.
   struct Release { fs::path path;~Release(){std::ofstream(path)<<"release";} } release{helperdir/"release"};
   const auto deadline=std::chrono::steady_clock::now()+1s;
   while(!fs::exists(helperdir/"entered") && std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(1ms);
   require(fs::exists(helperdir/"entered"),"host helper never entered its complete step");
   require(channel.request()==CancelDelivery::kDelivered && !returned,"runner returned before the whole step completed");
   require(!fs::exists(helperdir/"completed"),"host helper crossed unreleased boundary");
   std::ofstream(helperdir/"release")<<"release";runner.join();if(error)std::rethrow_exception(error);
   require(outcome.termination==ProcessTermination::kExited && outcome.exit_code==2 && !outcome.io_error && !outcome.system_error,
           "real runner did not naturally reap cancelled host helper");
   require(read(helperdir/"completed")=="fake-G" && !fs::exists(helperdir/"exported"),"helper split the step or exported after cancel");
   require(read(helper.log_path).find("host helper finished naturally")!=std::string::npos,"runner lost final helper log");
   require(channel.close(),"helper source close failed");
  }
  std::cout<<"natural host helper: complete body, exit2, log drain/reap passed (no GPU training)\n";
  std::cout<<"token/complete-step/error-priority/close/reuse host contracts passed\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"Training protocol contract: "<<e.what()<<'\n';return 1;}
}
