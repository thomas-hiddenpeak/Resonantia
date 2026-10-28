// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#include "tools/app_support/joint_full_training.h"
#include "tools/app_support/joint_feature_cache.h"
#include "src/io/bounded_audio_io.h"
#include "src/training/joint_features.h"
#include "voxmutatio/content/hubert_encoder.h"
#include "voxmutatio/core/device.h"
#include "voxmutatio/f0/rmvpe.h"
#include "voxmutatio/io/audio_io.h"
#include <chrono>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <sys/file.h>

namespace voxmutatio::app_support {
namespace {
namespace fs=std::filesystem;
using namespace joint_full;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
struct RunLock {
  int fd=-1;
  explicit RunLock(const fs::path& dir) {
    fs::create_directories(dir);fd=open((dir/".lock").c_str(),O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd<0)throw std::runtime_error("cannot open joint-full directory lock");
    if(flock(fd,LOCK_EX|LOCK_NB)){close(fd);fd=-1;throw std::runtime_error("checkpoint directory is already in use");}
  }
  ~RunLock(){if(fd>=0)close(fd);}
  RunLock(const RunLock&)=delete;RunLock& operator=(const RunLock&)=delete;
};
struct CachedInput {fs::path audio,cache;std::string audio_hash,key,payload_hash;int frames=0;};
void validate_audio(const io::detail::WavMetadata& m) {
  if(m.sample_rate!=40000 || m.frame_count<32*400 || m.frame_count>600*400)
    throw std::invalid_argument("joint input requires 40 kHz WAV clips between 0.32 and 6 seconds; run vc_preprocess first");
}
int coarse(float f0) {
  if(f0<=0)return 1;
  const double min=1127.*std::log(1.+50./700.),max=1127.*std::log(1.+1100./700.);
  const double scaled=(1127.*std::log(1.+f0/700.)-min)*254./(max-min)+1.;
  return std::clamp(static_cast<int>(std::lround(std::clamp(scaled,1.,255.))),1,255);
}
training::JointTrainingClip prepare_clip(const CachedInput& input,content::HubertEncoder& hubert,f0::RmvpeExtractor& rmvpe) {
  const auto start=Clock::now();training::JointTrainingClip clip;
  io::detail::WavChunkReader reader;
  if(!reader.open(input.audio))throw std::runtime_error("cannot open joint-full audio");validate_audio(reader.metadata());
  if(!reader.read_mono_frames(0,reader.metadata().frame_count,clip.audio)||!reader.verify_unchanged()||
     sha256_file(input.audio)!=input.audio_hash)throw std::runtime_error("joint-full audio changed during preparation");
  const auto audio16=io::resample_linear(clip.audio.data(),static_cast<int>(clip.audio.size()),40000,16000);
  if(audio16.empty())throw std::runtime_error("joint-full resampling failed");
  const auto content_start=Clock::now();const auto features=hubert.extract(audio16.data(),static_cast<int>(audio16.size()));
  const auto content_seconds=elapsed(content_start);
  if(features.empty()||features.size()%768||features.size()/768>300)throw std::runtime_error("joint-full HuBERT layout failed");
  clip.phone.resize(features.size()*2);
  for(std::size_t frame=0;frame<features.size()/768;++frame)
    for(int repeat=0;repeat<2;++repeat)std::copy_n(features.data()+frame*768,768,clip.phone.data()+(frame*2+repeat)*768);
  const auto f0_start=Clock::now();clip.f0=rmvpe.infer(audio16.data(),static_cast<int>(audio16.size()));const auto f0_seconds=elapsed(f0_start);
  clip.frames=static_cast<int>(std::min({clip.phone.size()/768,clip.f0.size(),clip.audio.size()/400}));
  if(clip.frames<32||clip.frames>600)throw std::runtime_error("joint-full aligned frame count out of bounds");
  clip.phone.resize(clip.frames*768);clip.f0.resize(clip.frames);clip.audio.resize(clip.frames*400);
  clip.pitch.reserve(clip.frames);for(float f:clip.f0)clip.pitch.push_back(coarse(f));
  const auto spec_start=Clock::now();clip.spectrum=training::detail::joint_spectrum(clip.audio);const auto spec_seconds=elapsed(spec_start);
  std::cout<<"  frozen timings seconds: hubert="<<content_seconds<<" f0="<<f0_seconds<<" spectrum="<<spec_seconds
           <<" total="<<elapsed(start)<<'\n';return clip;
}
std::string numbered(const char* prefix,int value,const char* suffix) {
  std::ostringstream o;o<<prefix<<std::setfill('0')<<std::setw(10)<<value<<suffix;return o.str();
}
bool checkpoint_name(const std::string& n) {
  return n.size()==33&&n.starts_with("state_step_")&&n.ends_with(".safetensors")&&
    n.substr(11,10).find_first_not_of("0123456789")==std::string::npos;
}
void save_state(training::JointTrainer& trainer,const RunState& state,const fs::path& dir) {
  const auto start=Clock::now();const auto file=dir/numbered("state_step_",state.step,".safetensors");
  if(!trainer.save_checkpoint(file,state.serialize()))throw std::runtime_error("joint-full full checkpoint save failed");
  std::cout<<"Checkpoint "<<file<<" step="<<state.step<<" seconds="<<elapsed(start)<<std::endl;
  std::vector<fs::path> checkpoints;
  for(const auto& entry:fs::directory_iterator(dir))
    if(checkpoint_name(entry.path().filename())&&fs::is_regular_file(entry.symlink_status()))checkpoints.push_back(entry.path());
  std::sort(checkpoints.begin(),checkpoints.end());
  // Prune only our exact checkpoint filenames, after a new checkpoint committed.
  for(std::size_t i=0;i+2<checkpoints.size();++i)fs::remove(checkpoints[i]);
}
void save_candidate(training::JointTrainer& trainer,const RunState& state,const fs::path& dir) {
  if(state.slot()!=0 || state.epoch()==0 || state.epoch()>200 || state.epoch()%50)return;
  const auto file=dir/numbered("candidate_epoch_",state.epoch(),".safetensors");
  if(!trainer.export_model(file))throw std::runtime_error("joint-full candidate export failed");
  std::cout<<"Candidate "<<file<<std::endl;
}
} // namespace

int run_full_joint_training(const VcTrainOptions& o,resonantia::training_control::TrainingCancelReader& cancel) {
  using namespace resonantia::training_control;
  if(!o.gan_mode||o.gan_profile!="joint-full"||o.epochs<1||o.epochs>2000||o.segment_frames!=32||
      o.synthesis_mode!=SynthesisMode::kSampled||o.cache_dir.empty()||o.checkpoint_dir.empty()||
      o.speaker<0||o.speaker>108)
    throw std::invalid_argument("invalid joint-full options");
  // Validate all audio metadata and recipe bounds before CUDA/model work.
  std::vector<fs::path> files;
  if(fs::is_directory(o.target_path)) {
    for(const auto& entry:fs::directory_iterator(o.target_path)) {
      auto ext=entry.path().extension().string();for(char& c:ext)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      if(ext==".wav")files.push_back(entry.path());
    }
    std::sort(files.begin(),files.end());
  } else files.emplace_back(o.target_path);
  if(files.empty()||files.size()>4096)throw std::invalid_argument("joint-full requires 1..4096 WAV clips");
  (void)RunState::begin(sha256("preflight"),files.size(),o.epochs,o.seed,o.learning_rate,o.learning_rate_decay);
  // Ensure the entire requested schedule, including the last decay, stays finite/nonzero.
  {auto s=RunState::begin(sha256("preflight"),files.size(),o.epochs,o.seed,o.learning_rate,o.learning_rate_decay);
    s.step=s.epochs*static_cast<int>(s.order.size());s.validate();}
  for(const auto& file:files){io::detail::WavChunkReader reader;if(!reader.open(file))throw std::runtime_error("cannot read joint-full WAV: "+file.string());validate_audio(reader.metadata());}
  const fs::path state_dir=o.checkpoint_dir,cache_dir=o.cache_dir;
  RunLock lock(state_dir);
  if(o.resume_path.empty())for(const auto& entry:fs::directory_iterator(state_dir))
    if(entry.path().filename()!=".lock")throw std::invalid_argument("checkpoint directory is not empty; use a new directory or --resume");
  fs::create_directories(cache_dir);
  const fs::path hubert_file=fs::is_directory(o.hubert_path)?fs::path(o.hubert_path)/"model.safetensors":fs::path(o.hubert_path);
  const fs::path dmodel=o.discriminator_path.empty()?fs::path(o.pretrained_path).parent_path()/"f0D40k.safetensors":fs::path(o.discriminator_path);
  std::cout<<"Preparing model/data identity on CPU before CUDA feature extraction"<<std::endl;
  const auto generator_hash=sha256_file(o.pretrained_path),discriminator_hash=sha256_file(dmodel);
  const auto hubert_hash=sha256_file(hubert_file),rmvpe_hash=sha256_file(o.rmvpe_path);
  const std::string frontend="joint-frozen-v1;40k;mono;linear16k;hubert12-768-nearest2;rmvpe-htk-zero;spec2048-hop400-pad824-eps2e-7\n"+hubert_hash+'\n'+rmvpe_hash+'\n';
  std::vector<CachedInput> inputs;inputs.reserve(files.size());
  for(const auto& file:files){const auto hash=sha256_file(file);const auto key=sha256(frontend+hash);inputs.push_back({file,cache_dir/(key+".safetensors"),hash,key,{}});}
  if(const auto result=cancel.poll();result!=TrainingResult::kCompleted)return static_cast<int>(result);
  Device device;if(auto error=device.init("cuda",0))throw std::runtime_error("CUDA init: "+*error);
  std::cout<<"Device: "<<device.name()<<"\nJoint-full: clips="<<files.size()<<" epochs="<<o.epochs
           <<" batch=1 steps="<<files.size()*o.epochs<<" crop=32 cache="<<cache_dir<<std::endl;
  const auto preparation_start=Clock::now();std::size_t hits=0;
  {
    std::unique_ptr<content::HubertEncoder> hubert;std::unique_ptr<f0::RmvpeExtractor> rmvpe;
    for(std::size_t i=0;i<inputs.size();++i) {
      if(const auto result=cancel.poll();result!=TrainingResult::kCompleted)return static_cast<int>(result);
      auto& input=inputs[i];training::JointTrainingClip clip;
      if(fs::exists(input.cache)) {clip=load_feature_cache(input.cache,input.key);++hits;}
      else {
        if(!hubert){hubert=std::make_unique<content::HubertEncoder>();content::HubertConfig hc;hc.model_path=hubert_file.string();
          hc.output_dim=768;hc.num_layers=12;hc.use_final_proj=false;
          if(!hubert->init(hc))throw std::runtime_error("joint-full HuBERT initialization failed");
          rmvpe=std::make_unique<f0::RmvpeExtractor>();f0::RmvpeConfig fc;fc.model_path=o.rmvpe_path;
          if(!rmvpe->init(fc))throw std::runtime_error("joint-full RMVPE initialization failed");}
        clip=prepare_clip(input,*hubert,*rmvpe);save_feature_cache(input.cache,input.key,clip);
        // Verify the serialized representation before ever training on it.
        const auto reloaded=load_feature_cache(input.cache,input.key);
        if(clip_digest(clip)!=clip_digest(reloaded))throw std::runtime_error("frozen cache roundtrip mismatch");
      }
      input.frames=clip.frames;input.payload_hash=clip_digest(clip);
      std::cout<<"Cache "<<i+1<<'/'<<inputs.size()<<" frames="<<clip.frames<<" hits="<<hits<<std::endl;
    }
  } // Frozen models are released before the joint graph/optimizer is allocated.
  for(const auto& input:inputs)if(sha256_file(input.audio)!=input.audio_hash)
    throw std::runtime_error("training audio changed during feature preparation");
  if(sha256_file(hubert_file)!=hubert_hash||sha256_file(o.rmvpe_path)!=rmvpe_hash||
     sha256_file(o.pretrained_path)!=generator_hash||sha256_file(dmodel)!=discriminator_hash)
    throw std::runtime_error("training model changed during feature preparation");
  std::ostringstream recipe;recipe.imbue(std::locale::classic());recipe<<std::setprecision(std::numeric_limits<float>::max_digits10)
    <<"joint-full-v1;batch1;fp32;crop32;adamw.8,.99,1e-9,.01;fresh-noise\n"<<generator_hash<<'\n'<<discriminator_hash<<'\n'
    <<frontend<<o.epochs<<' '<<o.speaker<<' '<<o.seed<<' '<<o.learning_rate<<' '<<o.learning_rate_decay<<'\n';
  for(const auto& input:inputs)recipe<<std::quoted(input.audio.filename().string())<<' '<<input.key<<' '<<input.payload_hash<<'\n';
  const auto identity=sha256(recipe.str());RunState state=RunState::begin(identity,inputs.size(),o.epochs,o.seed,o.learning_rate,o.learning_rate_decay);
  const auto marker=state_dir/"identity.txt";
  if(fs::exists(marker)) {
    if(fs::file_size(marker)>128)throw std::runtime_error("invalid training identity marker");
    std::ifstream in(marker);std::string value;std::getline(in,value);
    if(value!=identity)throw std::invalid_argument("checkpoint directory belongs to another dataset/recipe");
  }
  if(const auto result=cancel.poll();result!=TrainingResult::kCompleted)return static_cast<int>(result);
  training::JointTrainer trainer;
  if(o.resume_path.empty()) {
    if(!trainer.init(o.pretrained_path,dmodel,o.speaker,o.learning_rate,o.seed))throw std::runtime_error("joint-full trainer init failed");
  } else {
    std::string metadata;
    std::cout<<"Checkpoint restore: "<<o.resume_path<<std::endl;
    if(!trainer.load_checkpoint(o.resume_path,metadata))throw std::runtime_error("joint-full checkpoint load failed");
    auto restored=RunState::parse(metadata);
    if(restored.identity!=identity||restored.epochs!=state.epochs||restored.order.size()!=state.order.size()||
       restored.initial_lr!=state.initial_lr||restored.decay!=state.decay||restored.step!=trainer.step_count()||
       restored.learning_rate()!=trainer.learning_rate())throw std::invalid_argument("resume dataset, recipe, step or learning rate mismatch");
    state=std::move(restored);std::cout<<"Resumed step="<<state.step<<" epoch="<<state.epoch()<<" next_clip="<<state.slot()<<std::endl;
  }
  if(sha256_file(o.pretrained_path)!=generator_hash||sha256_file(dmodel)!=discriminator_hash)
    throw std::runtime_error("pretrained model changed during trainer initialization");
  {training::detail::CheckpointFile file(marker);const auto text=identity+'\n';file.write(text.data(),text.size());file.commit(marker);}
  if(o.resume_path.empty())save_state(trainer,state,state_dir);
  std::cout<<"Preparation seconds="<<elapsed(preparation_start)<<" cache_hits="<<hits<<'/'<<inputs.size()<<" identity="<<identity<<std::endl;
  const auto training_start=Clock::now();const int starting_step=state.step;double load_seconds=0,step_seconds=0;
  auto save_on_stop=[&](TrainingResult result){save_state(trainer,state,state_dir);
    std::cout<<"Joint-full stopped at complete step="<<state.step<<"; resume checkpoint retained; no model exported"<<std::endl;
    return static_cast<int>(result);};
  while(!state.done()) {
    if(const auto result=cancel.poll();result!=TrainingResult::kCompleted)return save_on_stop(result);
    const auto& input=inputs.at(state.current_clip());const auto load_start=Clock::now();
    auto clip=load_feature_cache(input.cache,input.key);
    if(clip_digest(clip)!=input.payload_hash)throw std::runtime_error("cached training data changed during run");
    const double load_time=elapsed(load_start);load_seconds+=load_time;
    training::JointStepOptions step;step.crop_frames=32;step.crop_start=state.crop(clip.frames);
    const auto step_start=Clock::now();const auto loss=trainer.step(clip,step);const double step_time=elapsed(step_start);step_seconds+=step_time;
    state.advance();trainer.set_learning_rate(state.learning_rate());
    std::cout<<std::setprecision(9)<<"joint-full step="<<state.step<<'/'<<inputs.size()*o.epochs<<" epoch="<<state.epoch()
      <<" clip="<<state.slot()<<'/'<<inputs.size()<<" frames="<<clip.frames<<" crop="<<step.crop_start<<" lr="<<trainer.learning_rate()
      <<" D="<<loss.discriminator<<" G="<<loss.generator<<" mel="<<loss.mel<<" kl="<<loss.kl<<" fm="<<loss.feature<<" adv="<<loss.adversarial
      <<" cache_seconds="<<load_time<<" train_seconds="<<step_time<<std::endl;
    if(state.slot()==0) {
      if(state.epoch()%5==0||state.done())save_state(trainer,state,state_dir);
      save_candidate(trainer,state,state_dir);
    }
  }
  if(const auto result=cancel.poll();result!=TrainingResult::kCompleted)return save_on_stop(result);
  // Also supports a completed resume if output export failed after the last checkpoint.
  if(state.step==starting_step)save_candidate(trainer,state,state_dir);
  if(!trainer.export_model(o.output_path))throw std::runtime_error("joint-full final export failed");
  std::cout<<"Joint-full completed: steps_this_process="<<state.step-starting_step<<" training_seconds="<<elapsed(training_start)
    <<" cache_read_seconds="<<load_seconds<<" joint_step_seconds="<<step_seconds<<" model="<<o.output_path<<std::endl;
  return 0;
}
} // namespace voxmutatio::app_support
