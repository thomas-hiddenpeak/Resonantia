/**
 * @file test_vits_gen_debug.cpp
 * @brief Debug generator stages: har source, conv_pre.
 */
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include "voxmutatio/synthesizer/synthesizer.h"

namespace {
struct Ref { std::vector<int> shape; std::vector<float> data; bool ok=false; };
Ref load_ref(const std::string& p){ Ref r; std::ifstream f(p,std::ios::binary); if(!f.is_open())return r;
  int32_t nd; f.read((char*)&nd,4); if(nd<=0||nd>8)return r; int64_t tot=1;
  for(int i=0;i<nd;++i){int32_t s;f.read((char*)&s,4);r.shape.push_back(s);tot*=s;}
  r.data.resize(tot); f.read((char*)r.data.data(),tot*4); r.ok=f.good()||f.eof(); return r; }
double rms(const std::vector<float>&a,const std::vector<float>&b){ if(a.size()!=b.size()||a.empty())return -1;
  double s=0; for(size_t i=0;i<a.size();++i){double d=(double)a[i]-b[i];s+=d*d;} return std::sqrt(s/a.size()); }
double corr(const std::vector<float>&a,const std::vector<float>&b){ size_t n=std::min(a.size(),b.size());
  double ma=0,mb=0; for(size_t i=0;i<n;++i){ma+=a[i];mb+=b[i];} ma/=n;mb/=n; double c=0,va=0,vb=0;
  for(size_t i=0;i<n;++i){double da=a[i]-ma,db=b[i]-mb;c+=da*db;va+=da*da;vb+=db*db;} return c/(std::sqrt(va*vb)+1e-12);}
}

TEST_CASE("VITS har source", "[vitsdbg][har]") {
    using namespace voxmutatio::synthesizer;
    auto nsff0 = load_ref("../tests/fixtures/vits_ref_nsff0.bin");
    auto har_ref = load_ref("../tests/fixtures/vits_ref_har.bin");
    REQUIRE(nsff0.ok); REQUIRE(har_ref.ok);
    int T = nsff0.shape[0];
    SynthesizerConfig cfg; cfg.model_path="../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    Synthesizer s; s.init(cfg);
    auto har = s.debug_har(nsff0.data.data(), T);
    std::cout<<"har C++ "<<har.size()<<" ref "<<har_ref.data.size()<<std::endl;
    std::cout<<"har RMS "<<rms(har,har_ref.data)<<" corr "<<corr(har,har_ref.data)<<std::endl;
    std::cout<<"har[0..5]: ";for(int i=0;i<6;++i)std::cout<<har[i]<<" ";std::cout<<std::endl;
    std::cout<<"ref[0..5]: ";for(int i=0;i<6;++i)std::cout<<har_ref.data[i]<<" ";std::cout<<std::endl;
    // sample from voiced region
    int mid=100000;
    std::cout<<"har[mid..+5]: ";for(int i=0;i<6;++i)std::cout<<har[mid+i]<<" ";std::cout<<std::endl;
    std::cout<<"ref[mid..+5]: ";for(int i=0;i<6;++i)std::cout<<har_ref.data[mid+i]<<" ";std::cout<<std::endl;
    CHECK(corr(har,har_ref.data)>0.99);
}

TEST_CASE("VITS conv_pre", "[vitsdbg][convpre]") {
    using namespace voxmutatio::synthesizer;
    auto z = load_ref("../tests/fixtures/vits_ref_z.bin");
    auto cp_ref = load_ref("../tests/fixtures/vits_ref_convpre.bin");
    REQUIRE(z.ok); REQUIRE(cp_ref.ok);
    int T = z.shape[1];
    SynthesizerConfig cfg; cfg.model_path="../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    Synthesizer s; s.init(cfg);
    auto cp = s.debug_convpre(z.data.data(), T, 0);
    std::cout<<"convpre RMS "<<rms(cp,cp_ref.data)<<" corr "<<corr(cp,cp_ref.data)<<std::endl;
    CHECK(rms(cp,cp_ref.data)<1e-3);
}

TEST_CASE("VITS gen stage0", "[vitsdbg][stage0]") {
    using namespace voxmutatio::synthesizer;
    auto z = load_ref("../tests/fixtures/vits_ref_z.bin");
    auto nsff0 = load_ref("../tests/fixtures/vits_ref_nsff0.bin");
    auto ups0 = load_ref("../tests/fixtures/vits_ref_ups0.bin");
    auto nc0 = load_ref("../tests/fixtures/vits_ref_noiseconv0.bin");
    auto st0 = load_ref("../tests/fixtures/vits_ref_stage0.bin");
    REQUIRE(z.ok); REQUIRE(nsff0.ok); REQUIRE(ups0.ok);
    int T = z.shape[1];
    SynthesizerConfig cfg; cfg.model_path="../models/pretrained_v2/pretrained_v2/f0G40k.safetensors";
    Synthesizer s; s.init(cfg);
    auto u0 = s.debug_gen_stage0(z.data.data(), nsff0.data.data(), T, 0, 0);
    std::cout<<"ups0 C++ "<<u0.size()<<" ref "<<ups0.data.size()<<" RMS "<<rms(u0,ups0.data)<<" corr "<<corr(u0,ups0.data)<<std::endl;
    auto n0 = s.debug_gen_stage0(z.data.data(), nsff0.data.data(), T, 0, 1);
    std::cout<<"noiseconv0 C++ "<<n0.size()<<" ref "<<nc0.data.size()<<" RMS "<<rms(n0,nc0.data)<<" corr "<<corr(n0,nc0.data)<<std::endl;
    auto s0 = s.debug_gen_stage0(z.data.data(), nsff0.data.data(), T, 0, 2);
    std::cout<<"stage0 C++ "<<s0.size()<<" ref "<<st0.data.size()<<" RMS "<<rms(s0,st0.data)<<" corr "<<corr(s0,st0.data)<<std::endl;
    CHECK(corr(u0,ups0.data)>0.99);
    CHECK(corr(s0,st0.data)>0.99);
}
