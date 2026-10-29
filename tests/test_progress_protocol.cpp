#include "tools/app_support/progress_protocol.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace voxmutatio::app_support;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,"malformed progress accepted");}
int main()try {
    std::ostringstream out;write_progress_message(out,{"synthesis",16000,64000,"samples16k"});
    check(out.str()=="RESONANTIA_PROGRESS 1 synthesis 16000 64000 samples16k\n","wire contract");
    auto r=parse_progress_message(out.str().substr(0,out.str().size()-1));
    check(r&&r->phase=="synthesis"&&r->completed==16000&&r->total==64000&&r->unit=="samples16k","roundtrip");
    check(!parse_progress_message("joint-full step=3/8"),"human log interpreted as progress");
    check(!parse_progress_message("error: RESONANTIA_PROGRESS 1 train 3 8 steps"),"unanchored progress");
    for(const auto* line:{"RESONANTIA_PROGRESS 2 training 1 2 steps","RESONANTIA_PROGRESS 1 training -1 2 steps",
         "RESONANTIA_PROGRESS 1 training 3 2 steps","RESONANTIA_PROGRESS 1 training 1 0 steps",
         "RESONANTIA_PROGRESS 1 training 0 18446744073709551616 steps",
         "RESONANTIA_PROGRESS 1 training 0 2 invented","RESONANTIA_PROGRESS 1 ../x 0 2 files",
         "RESONANTIA_PROGRESS 1 training 0 2 steps extra","RESONANTIA_PROGRESS 1 training 0 2 steps\n"})
        rejects([&]{(void)parse_progress_message(line);});
    rejects([&]{(void)parse_progress_message("RESONANTIA_PROGRESS "+std::string(1000,'a'));});
    rejects([&]{write_progress_message(out,{"bad\nphase",0,1,"files"});});
    auto zero=parse_progress_message("RESONANTIA_PROGRESS 1 loading 0 0 unknown");check(zero&&zero->total==0,"unknown progress");
    std::cout<<"progress protocol contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
