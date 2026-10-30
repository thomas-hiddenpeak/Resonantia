#include "voxmutatio/io/block_checkpoint.h"
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <limits>
#include <unistd.h>
namespace fs=std::filesystem;
using voxmutatio::io::BlockCheckpoint;
using Mode=BlockCheckpoint::Mode;
struct Directory {
    fs::path path;
    Directory(){char p[]="/tmp/resonantia-block-test-XXXXXX";auto r=mkdtemp(p);if(!r)throw std::runtime_error("mkdtemp");path=r;}
    ~Directory(){std::error_code e;fs::remove_all(path,e);}
};
const std::string identity(64,'a');
TEST_CASE("inspection is read only, shares readers and excludes writers") {
    Directory d;const auto root=d.path/"store";const std::vector<float> data{0,120,121};
    {BlockCheckpoint writer(root,identity,Mode::Create);writer.save("f0_0",0,data);
     REQUIRE_THROWS(BlockCheckpoint(root,identity,Mode::ReadOnly));}
    {std::ofstream f(root/"crash.block.pending");f<<"partial";}
    REQUIRE(BlockCheckpoint::stored_identity(root)==identity);
    {
        BlockCheckpoint a(root,identity,Mode::ReadOnly),b(root,identity,Mode::ReadOnly);
        REQUIRE(a.load("f0_0",0,3)==data);REQUIRE(a.entries().size()==1);
        CHECK(a.entries()[0].key=="f0_0");CHECK(a.entries()[0].offset==0);CHECK(a.entries()[0].count==3);
        REQUIRE_THROWS(a.save("f0_0",0,data));REQUIRE_THROWS(a.save("f0_3",3,data));
        REQUIRE_THROWS(BlockCheckpoint(root,identity,Mode::Resume));
        CHECK(fs::exists(root/"crash.block.pending"));
    }
    fs::remove(root/".lock");
    REQUIRE_THROWS(BlockCheckpoint(root,identity,Mode::ReadOnly));CHECK_FALSE(fs::exists(root/".lock"));
    fs::remove(root/"identity");REQUIRE_THROWS(BlockCheckpoint::stored_identity(root));
    REQUIRE_THROWS(BlockCheckpoint(root,identity,Mode::ReadOnly));CHECK_FALSE(fs::exists(root/"identity"));
}
TEST_CASE("durable blocks survive owner release and pause, bind shape and identity") {
    Directory d;const auto root=d.path/"store";bool stop=false;
    const std::vector<float> data{0,1,-2,3.5f};
    {
        BlockCheckpoint s(root,identity,Mode::Create,[&]{return stop;});
        REQUIRE_FALSE(s.load("f0_0",0,4));
        REQUIRE_THROWS(BlockCheckpoint(root,identity,Mode::Resume));
        s.save("f0_0",0,data);s.save("f0_0",0,data);
        REQUIRE(s.load("f0_0",0,4)==data);
        REQUIRE(s.restored_blocks()>=1);
        REQUIRE_THROWS(s.load("f0_0",1,4));REQUIRE_THROWS(s.load("f0_0",0,3));
        REQUIRE_THROWS(s.save("f0_0",0,std::vector<float>{1,2,3,4}));
        s.boundary();stop=true;REQUIRE_THROWS_AS(s.boundary(),voxmutatio::io::OperationPaused);
    }
    REQUIRE_THROWS(BlockCheckpoint(root,std::string(64,'b'),Mode::Resume));
    REQUIRE_THROWS(BlockCheckpoint(root,identity,Mode::Create));
    BlockCheckpoint recovered(root,identity,Mode::Resume);
    REQUIRE(recovered.load("f0_0",0,4)==data);
    REQUIRE_FALSE(recovered.load("f0_4",4,4));
}
TEST_CASE("checkpoint rejects malformed, corrupt and relocated payloads") {
    Directory d;const auto root=d.path/"store";const std::vector<float> data{1,2};
    {BlockCheckpoint s(root,identity,Mode::Create);s.save("core_0",0,data);}
    fs::copy_file(root/"core_0.block",root/"core_1.block");
    {BlockCheckpoint s(root,identity,Mode::Resume);REQUIRE_THROWS(s.load("core_1",0,2));}
    fs::remove(root/"core_1.block");
    {std::fstream f(root/"core_0.block",std::ios::binary|std::ios::in|std::ios::out);f.seekp(-1,std::ios::end);f.put('x');}
    {BlockCheckpoint s(root,identity,Mode::Resume);REQUIRE_THROWS(s.load("core_0",0,2));}
    fs::resize_file(root/"core_0.block",10);
    REQUIRE_THROWS(BlockCheckpoint(root,identity,Mode::Resume));
}
TEST_CASE("checkpoint enforces file types, bounds and pending crash recovery") {
    Directory d;const auto root=d.path/"store";
    const BlockCheckpoint::Limits limits{16,400,2};
    {
        BlockCheckpoint s(root,identity,Mode::Create,{},limits);
        for(const auto& k:{"../bad","","UPPER","a/b"}) REQUIRE_THROWS(s.save(k,0,std::vector<float>{1}));
        REQUIRE_THROWS(s.save("bad",0,std::vector<float>{std::numeric_limits<float>::infinity()}));
        REQUIRE_THROWS(s.save("bad",0,std::vector<float>{}));
        REQUIRE_THROWS(s.load("bad",0,UINT64_MAX));
        REQUIRE_THROWS(s.save("big",0,std::vector<float>(5)));
        s.save("a",0,std::vector<float>{1});s.save("b",1,std::vector<float>{2});
        REQUIRE_THROWS(s.save("c",2,std::vector<float>{3}));
    }
    {std::ofstream f(root/"crash.block.pending");f<<"partial";}
    {BlockCheckpoint s(root,identity,Mode::Resume,{},limits);REQUIRE_FALSE(fs::exists(root/"crash.block.pending"));}
    fs::remove(root/"a.block");fs::create_symlink(d.path/"outside",root/"a.block");
    REQUIRE_THROWS(BlockCheckpoint(root,identity,Mode::Resume));
    fs::remove(root/"a.block");fs::create_directory(root/"a.block");
    REQUIRE_THROWS(BlockCheckpoint(root,identity,Mode::Resume));
    fs::create_directory_symlink(root,d.path/"alias");
    REQUIRE_THROWS(BlockCheckpoint(d.path/"alias",identity,Mode::Resume));
    REQUIRE_THROWS(BlockCheckpoint(d.path/"absent",identity,Mode::Resume));
    REQUIRE_THROWS(BlockCheckpoint(d.path/"invalid","bad",Mode::Create));
}

TEST_CASE("failed block publication leaves committed work and poisons that owner") {
    Directory d;const auto root=d.path/"store";
    {
        BlockCheckpoint s(root,identity,Mode::Create);s.save("good",0,std::vector<float>{1});
        fs::create_directory(root/"new.block.pending");
        REQUIRE_THROWS(s.save("new",1,std::vector<float>{2}));
        REQUIRE_FALSE(fs::exists(root/"new.block"));
        REQUIRE_THROWS(s.load("good",0,1));
        REQUIRE_THROWS(s.save("later",2,std::vector<float>{3}));
    }
    fs::remove(root/"new.block.pending");
    BlockCheckpoint recovered(root,identity,Mode::Resume);
    REQUIRE(recovered.load("good",0,1)==std::vector<float>{1});
}

TEST_CASE("interruption before initial identity commit can recover only an empty store") {
    Directory d;const auto root=d.path/"store";fs::create_directory(root);
    {std::ofstream f(root/"identity.pending");f<<"partial first publication";}
    {
        BlockCheckpoint s(root,identity,Mode::Resume);
        REQUIRE_FALSE(fs::exists(root/"identity.pending"));
        s.save("good",0,std::vector<float>{1});
    }
    fs::remove(root/"identity");
    REQUIRE_THROWS(BlockCheckpoint(root,identity,Mode::Resume));
}
