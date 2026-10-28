// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#include <catch2/catch_test_macros.hpp>
#include "voxmutatio/training/joint_trainer.h"
#include "../src/training/joint_nsf.h"
#include "../src/training/joint_features.h"
#include "../src/training/joint_checkpoint.h"
#include "test_artifacts.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <limits>
#include <stdexcept>
using namespace voxmutatio::training;
TEST_CASE("Checkpoint staging preserves old files when serialization fails", "[joint][contract][checkpoint]") {
  namespace fs=std::filesystem;
  resonantia_test::ScopedArtifact file(resonantia_test::unique_artifact_path("checkpoint-atomic",".safetensors"));
  const std::string old="previous complete checkpoint";
  {std::ofstream out(file.path());out<<old;}
  std::vector<detail::CheckpointTensor> entries{{"G.p.test",{2},[]{return std::vector<float>{1,2};}},
    {"D.p.test",{1},[]{throw std::runtime_error("injected read failure");return std::vector<float>{};}}};
  CHECK_THROWS(detail::write_joint_checkpoint(file.path().string(),entries,"state"));
  std::ifstream retained(file.path());
  CHECK(std::string(std::istreambuf_iterator<char>(retained),{})==old);
  for(const auto& p:fs::directory_iterator(file.path().parent_path()))
    CHECK_FALSE(p.path().filename().string().starts_with(file.path().filename().string()+".tmp."));
  entries.pop_back();detail::write_joint_checkpoint(file.path().string(),entries,"state");
  voxmutatio::io::SafetensorsLoader loader;REQUIRE(loader.load(file.path().string()));
  CHECK(loader.tensor_names().size()==2);CHECK(detail::checkpoint_tensor(loader,"G.p.test",{2})==std::vector<float>{1,2});
  CHECK_THROWS(detail::checkpoint_tensor(loader,"G.p.test",{3}));
  CHECK_THROWS(detail::write_joint_checkpoint(file.path().string(),entries,std::string(detail::kJointCheckpointMaxText+1,'x')));
}
TEST_CASE("Joint spectrum retains the RVC power floor and bounded extent", "[joint][contract]") {
  std::vector<float> audio(32*400,0.f);
  const auto spectrum=detail::joint_spectrum(audio);
  REQUIRE(spectrum.size()==1025*32);
  CHECK(std::all_of(spectrum.begin(),spectrum.end(),[](float x){return x==std::sqrt(2e-7f);}));
  CHECK_THROWS_AS(detail::joint_spectrum({}),std::invalid_argument);
  CHECK_THROWS_AS(detail::joint_spectrum(std::vector<float>(601*400)),std::invalid_argument);
  audio.push_back(0);CHECK_THROWS_AS(detail::joint_spectrum(audio),std::invalid_argument);
  audio.pop_back();audio[0]=std::numeric_limits<float>::infinity();
  CHECK_THROWS_AS(detail::joint_spectrum(audio),std::invalid_argument);
}
TEST_CASE("Joint NSF preserves separate FP32 phase and excitation rounding", "[joint][contract]") {
  const std::vector<float> f0{220.13f,183.29f,0,301.17f,195.31f,276.19f};
  std::vector<float> noise(2400);
  for(int j=0;j<2400;++j)noise[j]=float((j*13)%29-14)/37.f;
  // Independent upstream SineGen, CPU FP32. Tolerance covers the last-bit
  // sinf/SLEEF difference, while rejecting fused phase/amplitude operations.
  const std::vector<int> indices{0,1,39,127,398,399,400,401,799,800,801,1199,
      1200,1201,1299,1477,1599,1600,1699,1799,1999,2000,2101,2199,2299,2399};
  const std::vector<float> expected{0x1.305864p-9f,0x1.bf8b8ep-8f,0x1.92684ep-4f,
      -0x1.8498d4p-4f,0x1.816918p-4f,0x1.8a3a18p-4f,0x1.884da4p-4f,
      0x1.8faf2ap-4f,0x1.516900p-6f,0x1.d854e2p-9f,-0x1.623fa8p-7f,0.f,
      0x1.b9f166p-6f,0x1.ef0b7ep-6f,-0x1.902ad6p-4f,0x1.22309ep-4f,
      0x1.ddf2d4p-6f,0x1.f8a7bap-6f,-0x1.57cd96p-6f,0x1.ca09f6p-7f,
      -0x1.8e71a0p-10f,0x1.fb7960p-9f,-0x1.84c2d6p-4f,0x1.1b23d6p-4f,
      0x1.60c58ap-5f,-0x1.994dccp-4f};
  const auto result=detail::joint_sine_source(f0,noise);
  REQUIRE(result.size()==noise.size());
  for(std::size_t j=0;j<indices.size();++j) {
    INFO("sample="<<indices[j]);CHECK(std::abs(result[indices[j]]-expected[j])<2e-8f);
  }
  CHECK_THROWS_AS(detail::joint_sine_source({},{}),std::invalid_argument);
  CHECK_THROWS_AS(detail::joint_sine_source(f0,{}),std::invalid_argument);
  CHECK_THROWS_AS(detail::joint_sine_source(std::vector<float>(601),{}),std::invalid_argument);
  auto bad=f0;bad[0]=std::numeric_limits<float>::infinity();
  CHECK_THROWS_AS(detail::joint_sine_source(bad,noise),std::invalid_argument);
  bad=f0;bad[0]=-1;CHECK_THROWS_AS(detail::joint_sine_source(bad,noise),std::invalid_argument);
  noise[0]=std::numeric_limits<float>::quiet_NaN();
  CHECK_THROWS_AS(detail::joint_sine_source(f0,noise),std::invalid_argument);
}
TEST_CASE("Joint trainer initialization and uninitialized calls fail closed", "[joint][contract]") {
  JointTrainer trainer;JointTrainingClip clip;JointStepOptions options;
  CHECK_FALSE(trainer.init("/missing-joint-generator", "/missing-joint-discriminator"));
  CHECK_THROWS_AS(trainer.step(clip,options),std::logic_error);
  CHECK_FALSE(trainer.export_model("/missing-output/model.safetensors"));
  CHECK_THROWS_AS(trainer.parameter_names(),std::logic_error);
  CHECK_THROWS_AS(trainer.inspect_parameter("model.emb_g.weight"),std::logic_error);
  CHECK_FALSE(trainer.init("missing","missing",109));
}
