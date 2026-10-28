#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "synthesizer/relative_attention.h"
#include "synthesizer/sampling.h"
#include "voxmutatio/synthesizer/synthesizer.h"
#include "voxmutatio/synthesizer/vits_ops.h"

namespace ops = voxmutatio::synthesizer::ops;

TEST_CASE("VITS convolutions preserve padding groups and tile boundaries", "[synth][conv][contract]") {
    struct Shape { int ci, co, length, kernel, stride, pad, dilation, groups; };
    for (const auto s : {Shape{16,24,1031,3,1,2,2,2}, Shape{16,16,41,5,3,4,2,1},
                         Shape{8,8,1,1,1,0,1,1}, Shape{1,1,7,3,1,1,1,1}}) {
        CAPTURE(s.ci, s.co, s.length, s.kernel, s.stride);
        std::vector<float> x(s.ci*s.length), w(s.co*(s.ci/s.groups)*s.kernel), bias(s.co);
        for (std::size_t i=0;i<x.size();++i) x[i]=std::sin(i*0.17f);
        for (std::size_t i=0;i<w.size();++i) w[i]=std::cos(i*0.13f)*0.1f;
        for (int i=0;i<s.co;++i) bias[i]=i*0.01f;
        const int length=(s.length+2*s.pad-s.dilation*(s.kernel-1)-1)/s.stride+1;
        for (const bool with_bias : {false,true}) {
            const auto actual=ops::conv1d(x.data(),s.ci,s.length,w.data(),with_bias?bias.data():nullptr,
                                         s.co,s.kernel,s.stride,s.pad,s.dilation,s.groups);
            REQUIRE(actual.size()==static_cast<std::size_t>(s.co*length));
            double error=0;
            for(int co=0;co<s.co;++co) for(int t=0;t<length;++t) {
                double expected=with_bias?bias[co]:0;
                for(int ci=0;ci<s.ci/s.groups;++ci) for(int k=0;k<s.kernel;++k) {
                    const int it=t*s.stride-s.pad+k*s.dilation;
                    if(it>=0 && it<s.length)
                        expected+=double(x[(co/(s.co/s.groups)*(s.ci/s.groups)+ci)*s.length+it])*
                                  w[(co*(s.ci/s.groups)+ci)*s.kernel+k];
                }
                REQUIRE(std::isfinite(actual[co*length+t]));
                error=std::max(error,std::abs(actual[co*length+t]-expected));
            }
            CHECK(error<2e-5);
        }
    }
}

TEST_CASE("VITS transpose convolution preserves stride holes and output tiles", "[synth][conv][contract]") {
    struct Shape { int ci, co, length, kernel, stride, pad; };
    for(const auto s : {Shape{16,16,1031,20,10,5},Shape{8,8,37,3,7,0},
                        Shape{8,8,41,5,1,3},Shape{1,1,7,4,2,1}}) {
        CAPTURE(s.ci,s.co,s.length,s.kernel,s.stride);
        std::vector<float> x(s.ci*s.length),w(s.ci*s.co*s.kernel),bias(s.co);
        for(std::size_t i=0;i<x.size();++i) x[i]=std::sin(i*0.17f);
        for(std::size_t i=0;i<w.size();++i) w[i]=std::cos(i*0.13f)*0.1f;
        for(int i=0;i<s.co;++i) bias[i]=i*0.01f;
        const int length=(s.length-1)*s.stride-2*s.pad+s.kernel;
        for(const bool with_bias:{false,true}) {
            const auto actual=ops::conv_transpose1d(x.data(),s.ci,s.length,w.data(),with_bias?bias.data():nullptr,
                                                   s.co,s.kernel,s.stride,s.pad);
            REQUIRE(actual.size()==static_cast<std::size_t>(s.co*length));
            double error=0;
            for(int co=0;co<s.co;++co) for(int t=0;t<length;++t) {
                double expected=with_bias?bias[co]:0;
                for(int k=0;k<s.kernel;++k) {
                    const int num=t+s.pad-k,it=num/s.stride;
                    if(num%s.stride || it<0 || it>=s.length) continue;
                    for(int ci=0;ci<s.ci;++ci) expected+=double(x[ci*s.length+it])*w[(ci*s.co+co)*s.kernel+k];
                }
                REQUIRE(std::isfinite(actual[co*length+t]));
                error=std::max(error,std::abs(actual[co*length+t]-expected));
            }
            CHECK(error<2e-5);
        }
    }
}

TEST_CASE("VITS relative attention matches explicit position equations",
          "[synth][attention][contract]") {
    constexpr int H = 2, D = 5, W = 10;
    for (const int T : {1, 7, 21, 33, 97, 257}) {
        CAPTURE(T);
        std::vector<float> q(H*D*T), k(q.size()), v(q.size());
        std::vector<float> rk((2*W+1)*D), rv(rk.size());
        for (std::size_t i=0; i<q.size(); ++i) {
            q[i] = std::sin(i * 0.17f); k[i] = std::cos(i * 0.13f);
            v[i] = std::sin(i * 0.31f);
        }
        for (std::size_t i=0; i<rk.size(); ++i) {
            rk[i] = std::cos(i * 0.11f); rv[i] = std::sin(i * 0.23f);
        }
        std::vector<float> expected(q.size());
        for(int h=0; h<H; ++h) for(int row=0; row<T; ++row) {
            std::vector<double> probability(T);
            for(int col=0; col<T; ++col) {
                double score=0;
                for(int d=0; d<D; ++d) {
                    const double query=q[(h*D+d)*T+row];
                    score += query*k[(h*D+d)*T+col];
                    if(std::abs(col-row)<=W) score += query*rk[(col-row+W)*D+d];
                }
                probability[col]=score/std::sqrt(static_cast<double>(D));
            }
            const double maximum=*std::max_element(probability.begin(),probability.end());
            double total=0;
            for(auto& value:probability) { value=std::exp(value-maximum); total+=value; }
            for(int d=0; d<D; ++d) {
                double value=0;
                for(int col=0; col<T; ++col) {
                    double content=v[(h*D+d)*T+col];
                    if(std::abs(col-row)<=W) content+=rv[(col-row+W)*D+d];
                    value+=probability[col]/total*content;
                }
                expected[(h*D+d)*T+row]=value;
            }
        }
        const auto actual=ops::relative_attention(q.data(),k.data(),v.data(),rk.data(),rv.data(),T,H,D,W);
        REQUIRE(actual.size()==expected.size());
        float maximum_error=0;
        for(std::size_t i=0;i<actual.size();++i) {
            REQUIRE(std::isfinite(actual[i]));
            maximum_error=std::max(maximum_error,std::abs(actual[i]-expected[i]));
        }
        CHECK(maximum_error<2e-5f);
    }
}

TEST_CASE("VITS attention rejects invalid shapes before device allocation",
          "[synth][attention][contract]") {
    const float x=1;
    REQUIRE_THROWS_AS(ops::relative_attention(nullptr,&x,&x,&x,&x,1,1,1,0),std::invalid_argument);
    REQUIRE_THROWS_AS(ops::relative_attention(&x,&x,&x,&x,&x,0,1,1,0),std::invalid_argument);
    REQUIRE_THROWS_AS(ops::relative_attention(&x,&x,&x,&x,&x,1,0,1,0),std::invalid_argument);
    REQUIRE_THROWS_AS(ops::relative_attention(&x,&x,&x,&x,&x,1,1,1,-1),std::invalid_argument);
    REQUIRE_THROWS_AS(ops::relative_attention(&x,&x,&x,&x,&x,50000,1,1,0),std::length_error);
    REQUIRE_THROWS_AS(ops::relative_attention(&x,&x,&x,&x,&x,1,1,2,1073741823),std::length_error);
}

TEST_CASE("Sampled synthesis has reproducible independent bounded Gaussian draws", "[synth][sampling][contract]") {
    namespace sample = voxmutatio::synthesizer::detail;
    CHECK(voxmutatio::VCConfig{}.synthesis_mode == voxmutatio::SynthesisMode::kSampled);
    CHECK(voxmutatio::synthesizer::SynthesizerConfig{}.synthesis_mode == voxmutatio::SynthesisMode::kSampled);
    const float golden[] = {0.8200101852f, 0.6542722583f, -0.2054895610f, -0.2005064487f,
                            2.2221922874f, 0.5865850449f, 1.4591230154f, -1.0386450291f};
    double sum=0, squares=0;
    constexpr int count=32768;
    for (int i=0; i<count; ++i) {
        const float value=sample::sampling_normal(0,sample::kPriorStream,i);
        REQUIRE(std::isfinite(value));
        if (i<8) CHECK(std::abs(value-golden[i])<2e-7f);
        CHECK(value==sample::sampling_normal(0,sample::kPriorStream,i));
        sum+=value; squares+=double(value)*value;
    }
    CHECK(std::abs(sum/count)<0.03);
    CHECK(std::abs(squares/count-1.0)<0.03);
    CHECK(sample::sampling_normal(0,sample::kPriorStream,0)!=sample::sampling_normal(1,sample::kPriorStream,0));
    CHECK(sample::sampling_normal(0,sample::kPriorStream,0)!=sample::sampling_normal(0,sample::kNsfStream,0));
    CHECK(std::isfinite(sample::sampling_normal(UINT64_MAX,sample::kNsfStream,UINT64_MAX)));
    std::vector<float> mean(192*5,0), logs(mean.size(),0), short_mean(192*3,0), short_logs(short_mean.size(),0);
    REQUIRE(sample::sample_prior(mean,logs,5,7));
    REQUIRE(sample::sample_prior(short_mean,short_logs,3,7));
    for(int c=0;c<192;++c) for(int t=0;t<3;++t) CHECK(mean[c*5+t]==short_mean[c*3+t]);
    std::fill(short_mean.begin(), short_mean.end(), 0);
    REQUIRE(sample::sample_prior(short_mean,short_logs,3,7,2));
    for(int c=0;c<192;++c) for(int t=0;t<3;++t) CHECK(mean[c*5+t+2]==short_mean[c*3+t]);
    constexpr auto limit=std::numeric_limits<std::uint64_t>::max()/400;
    CHECK(sample::valid_sampling_range(limit-1,1));
    CHECK_FALSE(sample::valid_sampling_range(limit,1));
    CHECK_FALSE(sample::valid_sampling_range(UINT64_MAX,1));
    CHECK_FALSE(sample::sample_prior(short_mean,short_logs,3,7,UINT64_MAX));
    logs[0]=1000;
    CHECK_FALSE(sample::sample_prior(mean,logs,5,7));
    logs[0]=std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(sample::sample_prior(mean,logs,5,7));
    CHECK_FALSE(sample::sample_prior(mean,{},5,7));
    CHECK_FALSE(sample::sample_prior(mean,logs,0,7));
}
