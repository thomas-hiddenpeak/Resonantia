#include "voxmutatio/f0/pitch_review.h"
#include <algorithm>
#include <array>
#include <stdexcept>
namespace voxmutatio::f0 {
namespace {
void validate(std::span<const float> raw,int filter,double pad,double duration) {
    if(raw.empty()||raw.size()>61100||!std::isfinite(pad)||pad<0||pad>5||!std::isfinite(duration)||duration<=0||duration>600||
       (filter!=0&&(filter<3||filter>31||filter%2!=1)))throw std::invalid_argument("invalid pitch review dimensions");
    for(float v:raw)if(!std::isfinite(v)||v<0||v>20000)throw std::invalid_argument("invalid pitch evidence");
}
double semitones(float a,float b){return 12*std::log2(static_cast<double>(a)/b);}
}
std::vector<float> prepare_pitch_curve(std::span<const float> raw,int filter,int shift,
    const std::vector<PitchEdit>& edits,double pad,double duration) {
    validate(raw,filter,pad,duration);
    if(shift<-48||shift>48||!valid_pitch_edits(edits))throw std::invalid_argument("invalid pitch review settings");
    const int radius=filter>=3?(filter|1)/2:0;
    const float factor=std::pow(2.0f,shift/12.0f);
    std::vector<float> values(raw.size());std::array<float,31> window{};
    for(std::size_t i=0;i<raw.size();++i) {
        float value=raw[i];
        if(radius) {
            for(int d=-radius;d<=radius;++d)window[d+radius]=raw[static_cast<std::size_t>(std::clamp<std::int64_t>(static_cast<std::int64_t>(i)+d,0,raw.size()-1))];
            std::sort(window.begin(),window.begin()+2*radius+1);value=window[radius];
        }
        values[i]=value>0?value*factor:value;
    }
    for(const auto& edit:edits) {
        if(edit.end>duration)throw std::invalid_argument("pitch edit exceeds input duration");
        const auto first=static_cast<std::size_t>(std::llround((edit.start+pad)*100));
        const auto last=static_cast<std::size_t>(std::llround((edit.end+pad)*100));
        if(last>=values.size()||last<=first)throw std::invalid_argument("pitch edit outside F0 timeline");
        if(edit.mode==PitchEditMode::kSmooth) {
            for(auto i=first;i<=last;++i)if(raw[i]<=0||values[i]<=0)throw std::invalid_argument("smooth pitch edit crosses unvoiced audio; narrow the interval");
            const double a=std::log(values[first]),b=std::log(values[last]);
            for(auto i=first+1;i<last;++i)values[i]=std::exp(a+(b-a)*static_cast<double>(i-first)/(last-first));
        }else {
            const float multiplier=std::pow(2.0f,edit.semitones/12.0);
            for(auto i=first;i<last;++i)if(raw[i]>0)values[i]*=multiplier;
        }
    }
    return values;
}
std::vector<PitchIssue> find_pitch_issues(std::span<const float> raw,std::span<const float> curve,
    int filter,double pad,double duration) {
    validate(raw,filter,pad,duration);
    if(curve.size()!=raw.size())throw std::invalid_argument("pitch curve size mismatch");
    for(float v:curve)if(!std::isfinite(v)||v<0)throw std::invalid_argument("invalid processed pitch");
    std::vector<PitchIssue> issues;const int radius=filter>=3?filter/2:0;
    const auto begin=std::max<std::size_t>(1,std::ceil(pad*100));
    const auto end=std::min<std::size_t>(raw.size(),std::floor((pad+duration)*100));
    for(auto i=begin;i+2<end&&issues.size()<64;++i) {
        if(raw[i-1]<=0||raw[i]<=0)continue;
        const double jump=semitones(raw[i],raw[i-1]);if(std::abs(jump)<7)continue;
        std::optional<PitchEdit> suggestion;auto until=i+1;
        if(std::abs(std::abs(jump)-12)<2) {
            for(auto j=i+2;j<std::min(end,i+36);++j) {
                if(raw[j-1]<=0||raw[j]<=0)break;
                const double exit=semitones(raw[j],raw[j-1]);
                if(jump*exit>=0||std::abs(std::abs(exit)-12)>=2||std::abs(semitones(raw[j],raw[i-1]))>3)continue;
                const int correction=jump<0?12:-12;const double factor=std::pow(2.,correction/12.);
                auto first=i,last=j;
                // Median filtering can move the first wrong frame across a UV
                // boundary. Expand only where the processed octave is also wrong.
                const auto wrong=[&](std::size_t k){return raw[k]>0&&curve[k]>0&&curve[j]>0&&
                    std::abs(12*std::log2(curve[k]*factor/curve[j]))<3;};
                while(first>begin&&i-first<static_cast<std::size_t>(radius)&&wrong(first-1))--first;
                while(last<end&&last-j<static_cast<std::size_t>(radius)&&wrong(last))++last;
                bool valid=true;for(auto k=first;k<last;++k)if(!wrong(k)){valid=false;break;}
                if(valid&&last-first>=2) {
                    suggestion=PitchEdit{PitchEditMode::kShift,first/100.-pad,last/100.-pad,static_cast<double>(correction)};
                    until=last;
                }else until=j;
                break;
            }
        }
        if(!suggestion) {
            // Already corrected / filtered out: no reason to prompt again.
            if(curve[i]<=0||curve[i-1]<=0||std::abs(semitones(curve[i],curve[i-1]))<7){i=until-1;continue;}
        }
        const auto start=suggestion?suggestion->start:i/100.-pad;
        const auto finish=suggestion?suggestion->end:std::min(duration,start+.05);
        issues.push_back({start,finish,suggestion?(suggestion->semitones>0?"短暂降低后恢复，疑似低了一个八度":"短暂升高后恢复，疑似高了一个八度"):"音高快速变化，可能是正常换音，请先试听",suggestion});
        i=until; // The return edge belongs to the same event.
    }
    return issues;
}
}
