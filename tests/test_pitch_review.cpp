#include "voxmutatio/f0/pitch_review.h"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
using namespace voxmutatio;
TEST_CASE("guided review finds transient octaves but does not flatten expression") {
    std::vector<float> raw(1401,200);for(int i=500;i<518;++i)raw[i]=100;
    auto curve=f0::prepare_pitch_curve(raw,3,0,{},2,10);
    auto issues=f0::find_pitch_issues(raw,curve,3,2,10);
    REQUIRE(issues.size()==1);REQUIRE(issues[0].suggestion.has_value());
    CHECK(issues[0].suggestion->start==3);CHECK(std::abs(issues[0].suggestion->end-3.18)<1e-9);CHECK(issues[0].suggestion->semitones==12);
    for(int i=0;i<1401;++i)raw[i]=200*std::pow(2.,.4*std::sin(i*.31)/12);
    curve=f0::prepare_pitch_curve(raw,3,0,{},2,10);CHECK(f0::find_pitch_issues(raw,curve,3,2,10).empty());
    for(int i=0;i<1401;++i)raw[i]=100*std::pow(2.,i/1400.);
    curve=f0::prepare_pitch_curve(raw,3,0,{},2,10);CHECK(f0::find_pitch_issues(raw,curve,3,2,10).empty());
    std::fill(raw.begin(),raw.end(),200);for(int i=500;i<900;++i)raw[i]=400;
    curve=f0::prepare_pitch_curve(raw,3,0,{},2,10);
    for(const auto& issue:f0::find_pitch_issues(raw,curve,3,2,10))CHECK_FALSE(issue.suggestion);
}
TEST_CASE("review respects UV and processed median boundaries") {
    std::vector<float> raw(1401,200);raw[499]=0;for(int i=501;i<518;++i)raw[i]=100;
    auto curve=f0::prepare_pitch_curve(raw,3,0,{},2,10);
    auto issues=f0::find_pitch_issues(raw,curve,3,2,10);
    REQUIRE(issues.size()==1);REQUIRE(issues[0].suggestion);CHECK(issues[0].suggestion->start==3);
    auto edits=std::vector<PitchEdit>{*issues[0].suggestion};
    auto fixed=f0::prepare_pitch_curve(raw,3,0,edits,2,10);CHECK(fixed[499]==curve[499]);CHECK(fixed[500]==200);
    CHECK(f0::find_pitch_issues(raw,fixed,3,2,10).empty());
    REQUIRE_THROWS(f0::prepare_pitch_curve(raw,3,0,{{PitchEditMode::kSmooth,2.98,3.1,0}},2,10));
    raw[506]=0;curve=f0::prepare_pitch_curve(raw,3,0,{},2,10);
    for(const auto& issue:f0::find_pitch_issues(raw,curve,3,2,10))CHECK_FALSE(issue.suggestion);
    raw[0]=std::numeric_limits<float>::quiet_NaN();REQUIRE_THROWS(f0::prepare_pitch_curve(raw,3,0,{},2,10));
    REQUIRE_THROWS(f0::prepare_pitch_curve(std::vector<float>(70000,200),3,0,{},2,10));
}
