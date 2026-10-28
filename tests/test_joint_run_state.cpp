// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#include "tools/app_support/joint_run_state.h"
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <filesystem>
#include <unistd.h>
using namespace voxmutatio::app_support::joint_full;
TEST_CASE("SHA256 identity matches published known answers and streaming", "[host]") {
    REQUIRE(sha256("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    REQUIRE(sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    REQUIRE(sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    const std::string million(1'000'000,'a');
    REQUIRE(sha256(million)=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    const auto path=std::filesystem::temp_directory_path()/("joint-sha-"+std::to_string(getpid()));
    struct Cleanup {std::filesystem::path p;~Cleanup(){std::error_code e;std::filesystem::remove(p,e);}} cleanup{path};
    {std::ofstream f(path,std::ios::binary);f<<million;}
    REQUIRE(sha256_file(path)==sha256(million));
    REQUIRE_THROWS(sha256_file(path,100));
    REQUIRE_THROWS(sha256_file(path/"missing"));
}
TEST_CASE("Epoch state survives mid-epoch restore with identical crops and next epoch", "[host]") {
    auto state=RunState::begin(sha256("recipe"),7,3,42,1e-4f,.999875f);
    REQUIRE(state.step==0);REQUIRE(state.epoch()==0);REQUIRE(state.slot()==0);
    for(int i=0;i<9;++i) {state.crop(120);state.advance();}
    auto restored=RunState::parse(state.serialize());
    REQUIRE(restored.serialize()==state.serialize());
    for(int i=9;i<21;++i) {
        REQUIRE(restored.current_clip()==state.current_clip());
        REQUIRE(restored.crop(340)==state.crop(340));
        REQUIRE(restored.learning_rate()==state.learning_rate());
        state.advance();restored.advance();
    }
    REQUIRE(state.done());REQUIRE(restored.done());REQUIRE(state.epoch()==3);
    REQUIRE(state.serialize()==restored.serialize());
    REQUIRE_THROWS(state.current_clip());REQUIRE_THROWS(state.advance());
    REQUIRE_THROWS(restored.crop(31));
}
TEST_CASE("Epoch state rejects corruption and inconsistent counters", "[host]") {
    auto state=RunState::begin(sha256("recipe"),3,200,1,1e-4f,.999875f);
    REQUIRE(state.learning_rate()==1e-4f);
    for(int i=0;i<3;++i)state.advance();
    REQUIRE(state.learning_rate()==1e-4f*.999875f);
    REQUIRE_THROWS(RunState::parse(state.serialize()+"garbage"));
    REQUIRE_THROWS(RunState::parse(std::string(65537,'x')));
    auto invalid=state;invalid.order[1]=invalid.order[0];REQUIRE_THROWS(invalid.serialize());
    invalid=state;invalid.step=601;REQUIRE_THROWS(invalid.serialize());
    REQUIRE_THROWS(RunState::begin("bad",3,200,0,1e-4f,.999875f));
    REQUIRE_THROWS(RunState::begin(sha256("x"),4097,200,0,1e-4f,.999875f));
    REQUIRE_THROWS(RunState::begin(sha256("x"),3,0,0,1e-4f,.999875f));
    REQUIRE_THROWS(RunState::begin(sha256("x"),3,200,0,1e-4f,0));
}
