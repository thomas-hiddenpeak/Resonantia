// Real-checkpoint contracts and the frozen independent ADR-0008 one-step gate.
#include <catch2/catch_test_macros.hpp>
#include "voxmutatio/training/joint_trainer.h"
#include "voxmutatio/autograd/tensor.h"
#include "voxmutatio/io/safetensors.h"
#include "training/export_weights.h"
#include "training/joint_features.h"
#include "test_artifacts.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>

using namespace voxmutatio;
namespace fs=std::filesystem;
namespace {
const std::string G="../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
const std::string D="../models/pretrained_v2/pretrained_v2/f0D40k.safetensors";
const fs::path reference="fixtures/joint_training";
std::vector<float> data(const io::SafetensorsLoader& l,const std::string& name) {
  auto t=l.get_tensor(name);REQUIRE(t);REQUIRE(t->dtype=="F32");
  std::vector<float> out(t->data_nbytes/4);std::memcpy(out.data(),l.data(name),t->data_nbytes);return out;
}
std::vector<float> fixture(const std::string& name,int rows,int cols) {
  std::ifstream f(reference/(name+".bin"),std::ios::binary);int32_t h[3]{};
  f.read(reinterpret_cast<char*>(h),12);REQUIRE(f.good());REQUIRE(h[0]==2);REQUIRE(h[1]==rows);REQUIRE(h[2]==cols);
  std::vector<float> v(static_cast<std::size_t>(rows)*cols);f.read(reinterpret_cast<char*>(v.data()),v.size()*4);
  REQUIRE(f.good());REQUIRE(f.peek()==std::char_traits<char>::eof());return v;
}
struct Error {
  double error=0,reference=0;std::size_t count=0;
  void add(const std::vector<float>& a,const std::vector<float>& b) {
    REQUIRE(a.size()==b.size());count+=a.size();
    bool finite=true;
    for(std::size_t i=0;i<a.size();++i) {
      finite=finite&&std::isfinite(a[i])&&std::isfinite(b[i]);
      const double d=double(a[i])-b[i];error+=d*d;reference+=double(b[i])*b[i];
    }
    REQUIRE(finite);
  }
  double rms()const{return std::sqrt(error/count);}
  double relative()const{return reference?std::sqrt(error/reference):0;}
};
}

TEST_CASE("Joint clip frontend matches the frozen real RVC spectrum", "[joint][numerical]") {
  const auto actual=training::detail::joint_spectrum(fixture("target",51200,1));
  Error error;error.add(actual,fixture("spec",1025,128));
  INFO("joint spectrum relative="<<error.relative());CHECK(error.relative()<1e-4);
}

TEST_CASE("Joint full checkpoint restores RNG optimizers and inference preservation", "[joint][resume]") {
  training::JointTrainingClip c;c.frames=128;c.phone=fixture("phone",128,768);
  auto pitch=fixture("pitch",128,1);c.pitch.assign(pitch.begin(),pitch.end());
  c.f0=fixture("f0",128,1);c.spectrum=fixture("spec",1025,128);c.audio=fixture("target",51200,1);
  training::JointStepOptions o;o.crop_start=48;
  resonantia_test::ScopedArtifact saved(resonantia_test::unique_artifact_path("joint-resume",".safetensors"));
  resonantia_test::ScopedArtifact expected(resonantia_test::unique_artifact_path("joint-continuous",".safetensors"));
  resonantia_test::ScopedArtifact actual(resonantia_test::unique_artifact_path("joint-resumed",".safetensors"));
  resonantia_test::ScopedArtifact inferred(resonantia_test::unique_artifact_path("joint-resumed-g",".safetensors"));
  const std::string run_state="epoch 1\nnext_clip 7\nidentity test-fixture";
  training::JointTrainer trainer;REQUIRE(trainer.init(G,D,3,1e-4f,17));
  trainer.step(c,o);trainer.set_learning_rate(9e-5f);
  REQUIRE(trainer.step_count()==1);REQUIRE(trainer.save_checkpoint(saved.path().string(),run_state));
  const auto first=trainer.step(c,o);REQUIRE(trainer.save_checkpoint(expected.path().string(),run_state));
  trainer=training::JointTrainer{};std::string state="unchanged";
  REQUIRE(trainer.load_checkpoint(saved.path().string(),state));REQUIRE(state==run_state);
  REQUIRE(trainer.step_count()==1);REQUIRE(trainer.learning_rate()==9e-5f);
  REQUIRE(trainer.save_checkpoint(actual.path().string(),state));
  {
    io::SafetensorsLoader a,b;REQUIRE(a.load(saved.path().string()));REQUIRE(b.load(actual.path().string()));
    REQUIRE(a.tensor_names().size()==b.tensor_names().size());
    for(const auto& name:a.tensor_names()) {
      auto t=a.get_tensor(name);REQUIRE(b.get_tensor(name));
      REQUIRE(t->data_nbytes==b.get_tensor(name)->data_nbytes);
      CHECK(std::memcmp(a.data(name),b.data(name),t->data_nbytes)==0);
    }
  }
  CHECK_FALSE(trainer.load_checkpoint(G,state));CHECK(state==run_state);CHECK(trainer.step_count()==1);
  CHECK_FALSE(trainer.load_checkpoint("/missing-checkpoint",state));CHECK(state==run_state);
  {
    resonantia_test::ScopedArtifact bad(resonantia_test::unique_artifact_path("joint-corrupt",".safetensors"));
    fs::copy_file(saved.path(),bad.path());
    std::size_t version_offset=0,variance_offset=0;
    {io::SafetensorsLoader source;REQUIRE(source.load(bad.path().string()));
      version_offset=source.get_tensor("state")->data_offset+std::string("RESONANTIA_JOINT_CHECKPOINT ").size()*4;
      for(const auto& name:source.tensor_names())if(name.starts_with("D.v.")){variance_offset=source.get_tensor(name)->data_offset;break;}}
    const auto corrupt=[&](std::size_t offset,float value){
      std::fstream file(bad.path(),std::ios::binary|std::ios::in|std::ios::out);REQUIRE(file.good());
      file.seekp(offset);file.write(reinterpret_cast<const char*>(&value),4);REQUIRE(file.good());
    };
    corrupt(version_offset,float('2'));
    CHECK_FALSE(trainer.load_checkpoint(bad.path().string(),state));CHECK(trainer.step_count()==1);CHECK(state==run_state);
    corrupt(version_offset,float('1'));REQUIRE(variance_offset>0);corrupt(variance_offset,-1);
    CHECK_FALSE(trainer.load_checkpoint(bad.path().string(),state));CHECK(trainer.step_count()==1);CHECK(state==run_state);
    fs::resize_file(bad.path(),16);
    CHECK_FALSE(trainer.load_checkpoint(bad.path().string(),state));CHECK(trainer.step_count()==1);CHECK(state==run_state);
  }
  CHECK_FALSE(trainer.save_checkpoint(saved.path().string(),std::string(65537,'x')));
  CHECK_FALSE(trainer.save_checkpoint((saved.path()/"child").string(),run_state));
  CHECK_THROWS_AS(trainer.set_learning_rate(0),std::invalid_argument);
  const auto second=trainer.step(c,o);CHECK(std::abs(second.generator-first.generator)<1e-4f);
  REQUIRE(trainer.save_checkpoint(actual.path().string(),state));
  {
    io::SafetensorsLoader a,b;REQUIRE(a.load(expected.path().string()));REQUIRE(b.load(actual.path().string()));
    REQUIRE(a.tensor_names().size()==b.tensor_names().size());
    std::map<std::string,Error> errors;
    for(const auto& name:a.tensor_names()) {
      auto t=a.get_tensor(name);REQUIRE(b.get_tensor(name));
      if(name=="state")CHECK(std::memcmp(a.data(name),b.data(name),t->data_nbytes)==0);
      else {
        const auto start=name.find("model.");REQUIRE(start!=std::string::npos);
        const auto model_name=name.substr(start);
        const auto end=model_name.find('.',name.starts_with("D.")?model_name.find('.',6)+1:6);
        errors[name.substr(0,start)+model_name.substr(0,end)].add(data(b,name),data(a,name));
      }
    }
    for(const auto& [name,error]:errors) {
      INFO(name<<" resume RMS="<<error.rms()<<" rel="<<error.relative());
      if(name.find(".p.")!=std::string::npos)CHECK(error.rms()<1e-6);
      else if(!error.reference)CHECK(error.error==0);
      else CHECK(error.relative()<1e-3);
    }
  }
  REQUIRE(trainer.export_model(inferred.path().string()));
  io::SafetensorsLoader base,out;REQUIRE(base.load(G));REQUIRE(out.load(inferred.path().string()));
  for(const auto& name:base.tensor_names())if(name.starts_with("model.enc_q."))CHECK(data(base,name)==data(out,name));
  auto a=data(base,"model.emb_g.weight"),b=data(out,"model.emb_g.weight");
  for(std::size_t i=0;i<a.size();++i)if(i/256!=3)CHECK(a[i]==b[i]);
}

TEST_CASE("Joint real decoder layers match independent fixed-input backward", "[joint][kernel-gradient]") {
  namespace ag=voxmutatio::autograd;
  struct Case {const char* name;bool transpose;int stride,pad,dilation;};
  for(const auto& c:std::vector<Case>{{"ups.0",true,10,3,1},{"ups.1",true,10,3,1},
      {"ups.2",true,2,1,1},{"ups.3",true,2,1,1},{"resblocks.8.convs1.2",false,1,25,5},
      {"resblocks.11.convs2.2",false,1,5,1}}) {
    io::SafetensorsLoader fixture;
    REQUIRE(fixture.load((fs::path("fixtures/joint_training_kernels")/(std::string(c.name)+".safetensors")).string()));
    const auto tensor=[&](const std::string& name,bool grad) {
      auto t=fixture.get_tensor(name);REQUIRE(t);
      return ag::Tensor::from_host(data(fixture,name),std::vector<int>(t->shape.begin(),t->shape.end()),grad);
    };
    auto x=tensor("input",true),v=tensor("weight_v",true),g=tensor("weight_g",true),b=tensor("bias",true);
    auto w=ag::weight_norm(v,g);const int cin=x.shape()[0],length=x.shape()[1],cout=w.shape()[c.transpose?1:0],k=w.shape()[2];
    auto y=c.transpose?ag::conv_transpose1d(x,w,b,cin,length,cout,k,c.stride,c.pad):
        ag::conv1d(x,w,b,cin,length,cout,k,c.stride,c.pad,c.dilation,1);
    Error forward;forward.add(y.to_host(),data(fixture,"output"));
    INFO(c.name<<" fixed forward relative="<<forward.relative());CHECK(forward.relative()<1e-4);
    {
      // Exact fixed-input outputs protect near-zero Leaky-ReLU branch choices
      // for both the small-input and channels-last reference paths.
      const auto actual=y.to_host(),expected=data(fixture,"output");
      std::size_t mismatches=0;
      for(std::size_t j=0;j<actual.size();++j)mismatches+=actual[j]!=expected[j];
      INFO(c.name<<" ordered forward mismatches="<<mismatches);
      CHECK(mismatches==0);
    }
    ag::backward(ag::sum(ag::mul(y,tensor("output_gradient",false))));
    for(const auto& [name,value]:std::vector<std::pair<std::string,ag::Tensor>>{
         {"input",x},{"weight_v",v},{"weight_g",g},{"bias",b}}) {
      Error e;e.add(value.grad_to_host(),data(fixture,name+"_gradient"));
      std::printf("[joint kernel] %s %s relative=%.9g RMS=%.9g\n",c.name,name.c_str(),e.relative(),e.rms());
      INFO(c.name<<" "<<name<<" fixed backward relative="<<e.relative());
      if(std::sqrt(e.reference)<1e-8)CHECK(e.rms()<1e-6);else CHECK(e.relative()<1e-3);
    }
  }
}

TEST_CASE("Joint checkpoint shapes are validated before training and failed reinit rolls back", "[joint][model-contract]") {
  io::SafetensorsLoader original;REQUIRE(original.load(G));
  std::vector<training::detail::ExportTensor> entries;
  for(const auto& name:original.tensor_names()) {
    const auto* t=original.get_tensor(name);std::vector<int> shape(t->shape.begin(),t->shape.end());
    if(name=="model.enc_q.pre.weight")shape={192,1,1025}; // same bytes, incompatible layout
    entries.push_back({name,shape,data(original,name)});
  }
  resonantia_test::ScopedArtifact bad(resonantia_test::unique_artifact_path("joint-invalid-shape",".safetensors"));
  REQUIRE(training::detail::write_safetensors_atomic(bad.path().string(),entries));
  training::JointTrainer trainer;REQUIRE(trainer.init(G,D));
  CHECK_FALSE(trainer.init(bad.path().string(),D));
  CHECK_THROWS_AS(trainer.parameter_names(),std::logic_error);
  CHECK_FALSE(trainer.export_model(bad.path().string()));
}

TEST_CASE("Joint real clip one-step forward gradients optimizer and export match independent reference", "[joint][numerical]") {
  training::JointTrainingClip c;c.frames=128;c.phone=fixture("phone",128,768);
  auto pitch=fixture("pitch",128,1);c.pitch.assign(pitch.begin(),pitch.end());
  c.f0=fixture("f0",128,1);c.spectrum=fixture("spec",1025,128);c.audio=fixture("target",51200,1);
  training::JointStepOptions o;o.crop_start=48;o.crop_frames=32;
  o.posterior_noise=fixture("eps",192,128);o.nsf_noise=fixture("nsf",12800,1);
  training::JointTrainer trainer;REQUIRE(trainer.init(G,D));
  auto bad=c;bad.frames=601;CHECK_THROWS_AS(trainer.step(bad,o),std::invalid_argument);
  bad=c;bad.phone[0]=std::numeric_limits<float>::quiet_NaN();CHECK_THROWS_AS(trainer.step(bad,o),std::invalid_argument);
  auto invalid=o;invalid.crop_start=97;CHECK_THROWS_AS(trainer.step(c,invalid),std::invalid_argument);
  resonantia_test::ScopedArtifact zero(resonantia_test::unique_artifact_path("joint-zero",".safetensors"));
  REQUIRE(trainer.export_model(zero.path().string()));
  io::SafetensorsLoader base,z;REQUIRE(base.load(G));REQUIRE(z.load(zero.path().string()));
  REQUIRE(base.tensor_names().size()==560);REQUIRE(z.tensor_names().size()==560);
  for(const auto& name:base.tensor_names())CHECK(data(base,name)==data(z,name));
  training::JointStepTrace trace;auto loss=trainer.step(c,o,&trace);CHECK(std::isfinite(loss.generator));
  for(const auto& [name,value]:std::vector<std::pair<std::string,std::vector<float>*>>{
      {"prior_mean",&trace.prior_mean},{"prior_logs",&trace.prior_logs},
      {"posterior_mean",&trace.posterior_mean},{"posterior_logs",&trace.posterior_logs},
      {"posterior_z",&trace.posterior_z},{"flow_z",&trace.flow_z},{"audio",&trace.audio}}) {
    auto ref=fixture(name,name=="audio"?1:192,name=="audio"?12800:128);Error e;e.add(*value,ref);
    INFO(name<<" forward relative L2="<<e.relative());CHECK(e.relative()<(name=="audio"?1e-3:1e-4));
    if(name=="audio") {
      double sa=0,sb=0,aa=0,bb=0,ab=0;const double n=ref.size();
      for(std::size_t i=0;i<ref.size();++i){double a=(*value)[i],b=ref[i];sa+=a;sb+=b;aa+=a*a;bb+=b*b;ab+=a*b;}
      CHECK((ab-sa*sb/n)/std::sqrt((aa-sa*sa/n)*(bb-sb*sb/n))>.999);
    }
  }
  for(bool disc:{false,true}) {
    std::array<io::SafetensorsLoader,4> refs;const char* kinds[]={"value","gradient","first","second"};
    std::array<std::map<std::string,Error>,4> errors;
    for(int k=0;k<4;++k)REQUIRE(refs[k].load((reference/((disc?"D_":"G_")+std::string(kinds[k])+".safetensors")).string()));
    for(const auto& name:trainer.parameter_names(disc)) {
      auto p=trainer.inspect_parameter(name,disc);std::array<const std::vector<float>*,4> values{&p.value,&p.gradient,&p.first_moment,&p.second_moment};
      const auto end=name.find('.',disc?name.find('.',6)+1:6);const auto group=name.substr(0,end);
      for(int k=0;k<4;++k)errors[k][group].add(*values[k],data(refs[k],name));
    }
    for(int k=0;k<4;++k)for(const auto& [name,e]:errors[k]) {
      INFO(name<<" "<<kinds[k]<<" relative L2="<<e.relative()<<" RMS="<<e.rms());
      if(k==0 || (k==1 && std::sqrt(e.reference)<1e-8))CHECK(e.rms()<1e-6);
      else if(!e.reference)CHECK(e.error==0);
      else CHECK(e.relative()<1e-3);
    }
  }
  resonantia_test::ScopedArtifact one(resonantia_test::unique_artifact_path("joint-one",".safetensors"));
  REQUIRE(trainer.export_model(one.path().string()));io::SafetensorsLoader exported;REQUIRE(exported.load(one.path().string()));
  for(const auto& name:base.tensor_names()) {
    auto expected=name.starts_with("model.enc_q.")?data(base,name):trainer.inspect_parameter(name).value;
    if(name=="model.emb_g.weight") {auto all=data(base,name);std::copy(expected.begin(),expected.end(),all.begin());expected=std::move(all);}
    CHECK(data(exported,name)==expected);
  }
  // A failed export must preserve an existing artifact byte for byte.
  CHECK_FALSE(trainer.export_model((one.path()/"invalid-child").string()));
  io::SafetensorsLoader retained;REQUIRE(retained.load(one.path().string()));
  CHECK(data(retained,"model.emb_g.weight")==data(exported,"model.emb_g.weight"));
}
