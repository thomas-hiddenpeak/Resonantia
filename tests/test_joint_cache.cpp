// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#include "tools/app_support/joint_feature_cache.h"
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <unistd.h>
using namespace voxmutatio;
using namespace voxmutatio::app_support::joint_full;
TEST_CASE("Frozen feature cache roundtrip, wrong recipe and corruption fail closed", "[cache]") {
    auto path=std::filesystem::temp_directory_path()/("joint-cache-"+std::to_string(getpid())+".safetensors");
    struct Cleanup{std::filesystem::path p;~Cleanup(){std::error_code e;std::filesystem::remove(p,e);}} cleanup{path};
    training::JointTrainingClip clip;clip.frames=32;clip.phone.assign(32*768,.125f);clip.f0.assign(32,220);
    clip.pitch.assign(32,80);clip.spectrum.assign(1025*32,.1f);clip.audio.assign(32*400,.05f);
    clip.f0[0]=1500; // Raw F0 is not capped at the 1100 Hz coarse-quantizer limit.
    const auto key=sha256("fixture recipe");save_feature_cache(path,key,clip);
    auto loaded=load_feature_cache(path,key);
    REQUIRE(loaded.frames==clip.frames);REQUIRE(loaded.audio==clip.audio);REQUIRE(loaded.phone==clip.phone);
    REQUIRE(loaded.f0==clip.f0);REQUIRE(loaded.pitch==clip.pitch);REQUIRE(loaded.spectrum==clip.spectrum);
    REQUIRE_THROWS(load_feature_cache(path,sha256("other model")));
    auto hash=sha256_file(path);clip.pitch[0]=256;
    REQUIRE_THROWS(save_feature_cache(path,key,clip));REQUIRE(sha256_file(path)==hash);
    clip.pitch[0]=80;clip.f0[0]=-1;REQUIRE_THROWS(save_feature_cache(path,key,clip));
    // Change a finite value, not just a header or NaN: payload digest must catch it.
    io::SafetensorsLoader loader;REQUIRE(loader.load(path));const auto offset=loader.get_tensor("phone")->data_offset;
    // Loader offsets are absolute within the mapped file.
    {std::fstream f(path,std::ios::in|std::ios::out|std::ios::binary);f.seekp(offset);float changed=.25f;f.write(reinterpret_cast<char*>(&changed),4);}
    REQUIRE_THROWS(load_feature_cache(path,key));
}
