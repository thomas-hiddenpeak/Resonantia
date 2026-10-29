// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#pragma once
#include <optional>
#include <array>
#include <charconv>
#include <stdexcept>
#include <ostream>
#include <string>
#include <string_view>
#include "voxmutatio/core/operation_progress.h"

namespace voxmutatio::app_support {
struct ProgressMessage {
    std::string phase, unit;
    std::uint64_t completed=0,total=0;
};
// Ignore ordinary logs; reject malformed recognized records. Parsing is bounded,
// locale independent and never searches an arbitrary log substring for a number.
inline bool valid_progress_phase(std::string_view phase) {
    return !phase.empty()&&phase.size()<=48&&phase.front()>='a'&&phase.front()<='z'&&
        phase.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-_")==phase.npos;
}
inline bool valid_progress_unit(std::string_view unit) {
    return unit=="unknown"||unit=="files"||unit=="chunks"||unit=="steps"||unit=="samples16k"||unit=="frames100hz";
}
inline std::optional<ProgressMessage> parse_progress_message(std::string_view line) {
    if(!line.starts_with("RESONANTIA_PROGRESS"))return std::nullopt;
    const auto invalid=[](){throw std::invalid_argument("invalid structured operation progress");};
    if(line.size()>192||line.find_first_of("\r\n\t")!=line.npos)invalid();
    std::array<std::string_view,6> words;
    for(std::size_t i=0;i<words.size();++i) {
        const auto space=line.find(' ');
        words[i]=line.substr(0,space);
        if(i+1==words.size()){if(space!=line.npos)invalid();}
        else {if(space==line.npos)invalid();line.remove_prefix(space+1);}
    }
    if(words[0]!="RESONANTIA_PROGRESS"||words[1]!="1"||!valid_progress_phase(words[2])||!valid_progress_unit(words[5]))invalid();
    const auto number=[&](std::string_view word) {
        std::uint64_t n=0;
        if(word.empty()||word.size()>20||word.find_first_not_of("0123456789")!=word.npos)invalid();
        const auto parsed=std::from_chars(word.data(),word.data()+word.size(),n);
        if(parsed.ec!=std::errc{}||parsed.ptr!=word.data()+word.size())invalid();return n;
    };
    ProgressMessage result{std::string(words[2]),std::string(words[5]),number(words[3]),number(words[4])};
    if(result.completed>result.total)invalid();return result;
}
inline void write_progress_message(std::ostream& out,const OperationProgress& p) {
    if(!valid_progress_phase(p.phase)||!valid_progress_unit(p.unit)||p.completed>p.total)
        throw std::invalid_argument("invalid operation progress");
    out<<"RESONANTIA_PROGRESS 1 "<<p.phase<<' '<<std::to_string(p.completed)<<' '<<std::to_string(p.total)<<' '<<p.unit<<std::endl;
    if(!out)throw std::runtime_error("operation progress write failed");
}
} // namespace voxmutatio::app_support
