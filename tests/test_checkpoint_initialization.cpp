#include "tools/app_support/operation_checkpoint.h"
#include "tools/app_support/joint_initialization.h"
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <unistd.h>
namespace fs=std::filesystem;
using voxmutatio::app_support::OperationCheckpointSession;
namespace {
struct Directory {
    fs::path path;
    Directory(){char p[]="/tmp/resonantia-initialization-XXXXXX";const auto r=mkdtemp(p);if(!r)throw std::runtime_error("mkdtemp");path=r;}
    ~Directory(){std::error_code e;fs::remove_all(path,e);}
};
const std::string identity(64,'a');
}
TEST_CASE("CLI block initialization resumes an empty root but never loses stage identity") {
    Directory d;const auto root=d.path/"blocks";fs::create_directory(root);
    {
        OperationCheckpointSession session(root,identity,true,"");
        const auto stage=session.stage("conversion");stage->save("f0_0",0,std::vector<float>{1,2});
    }
    {
        OperationCheckpointSession session(root,identity,true,"");
        REQUIRE(session.stage("conversion")->load("f0_0",0,2)==std::vector<float>{1,2});
    }
    fs::remove(root/"owner/identity");
    REQUIRE_THROWS(OperationCheckpointSession(root,identity,true,""));
    REQUIRE_FALSE(fs::exists(root/"owner/identity"));
    fs::remove_all(root/"owner");
    REQUIRE_THROWS(OperationCheckpointSession(root,identity,true,""));
    REQUIRE_FALSE(fs::exists(root/"owner"));
    REQUIRE_THROWS(OperationCheckpointSession(d.path/"missing",identity,true,""));
}
TEST_CASE("CLI session resumes an interrupted owner identity before any stage exists") {
    Directory d;const auto root=d.path/"blocks";fs::create_directories(root/"owner");
    std::ofstream(root/"owner/identity.pending")<<"incomplete";
    OperationCheckpointSession session(root,identity,true,"");
    REQUIRE(fs::is_regular_file(root/"owner/identity"));
    REQUIRE_FALSE(fs::exists(root/"owner/identity.pending"));
    REQUIRE_THROWS(OperationCheckpointSession(root,identity,true,""));
}
TEST_CASE("training initialization recovers partial writes only before a published state") {
    using namespace voxmutatio::app_support::joint_full;
    Directory d;
    std::ofstream(d.path/".lock")<<"";
    std::ofstream(d.path/"identity.txt.tmp.aBc123")<<"partial identity";
    REQUIRE_NOTHROW(validate_initial_training_directory(d.path));
    recover_initial_training_directory(d.path,identity);
    REQUIRE_FALSE(fs::exists(d.path/"identity.txt.tmp.aBc123"));
    REQUIRE_FALSE(fs::exists(d.path/"identity.txt"));
    std::ofstream(d.path/"identity.txt")<<identity<<'\n';
    const auto partial=d.path/"state_step_0000000000.safetensors.tmp.qWe123";
    std::ofstream(partial)<<"incomplete initial tensors";
    REQUIRE_THROWS(recover_initial_training_directory(d.path,std::string(64,'b')));
    REQUIRE(fs::exists(partial));
    recover_initial_training_directory(d.path,identity);
    REQUIRE_FALSE(fs::exists(partial));
    REQUIRE(fs::is_regular_file(d.path/"identity.txt"));
    std::ofstream(d.path/"state_step_0000000000.safetensors")<<"published state";
    REQUIRE_THROWS(validate_initial_training_directory(d.path));
    REQUIRE_THROWS(recover_initial_training_directory(d.path,identity));
    REQUIRE(fs::is_regular_file(d.path/"state_step_0000000000.safetensors"));
}
TEST_CASE("training initialization rejects unowned, corrupt, linked and excessive staging") {
    using namespace voxmutatio::app_support::joint_full;
    Directory d;const auto root=d.path/"state";fs::create_directory(root);
    for(const auto& name:{"unknown","state_step_0000000001.safetensors.tmp.qWe123","identity.txt.tmp.short"}) {
        std::ofstream(root/name)<<"preserve";
        REQUIRE_THROWS(recover_initial_training_directory(root,identity));
        REQUIRE(fs::exists(root/name));fs::remove(root/name);
    }
    const auto partial=root/"state_step_0000000000.safetensors.tmp.qWe123";
    std::ofstream(partial)<<"no committed identity";
    REQUIRE_THROWS(recover_initial_training_directory(root,identity));fs::remove(partial);
    std::ofstream(root/"identity.txt")<<"bad identity\n";
    REQUIRE_THROWS(validate_initial_training_directory(root));fs::remove(root/"identity.txt");
    std::ofstream(d.path/"outside")<<"preserve outside";
    fs::create_symlink(d.path/"outside",root/"identity.txt.tmp.qWe123");
    REQUIRE_THROWS(recover_initial_training_directory(root,identity));fs::remove(root/"identity.txt.tmp.qWe123");
    fs::create_hard_link(d.path/"outside",root/"identity.txt.tmp.qWe123");
    REQUIRE_THROWS(recover_initial_training_directory(root,identity));fs::remove(root/"identity.txt.tmp.qWe123");
    for(int i=0;i<17;++i)std::ofstream(root/("identity.txt.tmp."+std::to_string(100000+i)))<<"partial";
    REQUIRE_THROWS(recover_initial_training_directory(root,identity));
    REQUIRE(std::distance(fs::directory_iterator(root),fs::directory_iterator{})==17);
}
