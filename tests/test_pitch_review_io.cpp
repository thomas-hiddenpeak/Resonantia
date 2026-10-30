#include "tools/web/pitch_review.h"
#include "tools/app_support/operation_checkpoint.h"
#include "voxmutatio/io/audio_io.h"
#include "tools/web/web_service.h"
#include <sys/socket.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <fstream>
#include <unistd.h>
namespace fs=std::filesystem;
using namespace voxmutatio;
namespace review=resonantia::web::pitch_review;
struct ReviewFixture {
    fs::path root,work,program;VCConfig config;
    ReviewFixture() {
        char path[]="/tmp/resonantia-pitch-review-XXXXXX";const auto* dir=mkdtemp(path);if(!dir)throw std::runtime_error("mkdtemp");
        root=dir;work=root/"work";fs::create_directory(work);program=root/"vc_convert";
        for(const auto& p:{program,root/"hubert",root/"rmvpe",root/"generator"}){std::ofstream f(p);f<<"fixture bytes";}
        config.hubert_model_path=(root/"hubert").string();config.rmvpe_model_path=(root/"rmvpe").string();config.synthesizer_model_path=(root/"generator").string();config.filter_radius=3;
        const std::vector<float> audio(8*40000,.1f);if(!io::write_audio((work/"source.wav").string(),audio.data(),audio.size(),40000))throw std::runtime_error("write WAV");
        const auto identity=review::operation_identity(config,work/"source.wav",program);
        app_support::OperationCheckpointSession session(work/"conversion_blocks",identity,false,"");auto store=session.stage("conversion");
        std::vector<float> f0(1201,200);for(int i=300;i<315;++i)f0[i]=100;
        store->save("f0_0",0,std::span(f0).first(800));store->save("f0_800",800,std::span(f0).subspan(800));
        const std::vector<float> samples(160000,.1f);store->save("synth_0",0,samples);store->save("synth_64000",64000,samples);
    }
    ~ReviewFixture(){std::error_code e;fs::remove_all(root,e);}
};
TEST_CASE("historical pitch inspection and native excerpt are bounded and nonmutating") {
    ReviewFixture f;auto e=review::read_evidence(f.work);CHECK(e.duration==8);CHECK(e.pad==2);CHECK(e.raw.size()==1201);
    auto part=io::read_audio_excerpt((f.work/"source.wav").string(),.5,1.5);REQUIRE(part);CHECK(part->sample_rate==40000);CHECK(part->data.size()==40000);CHECK(part->source_format==SampleFormat::kInt16);
    CHECK_FALSE(io::read_audio_excerpt((f.work/"source.wav").string(),7,9));CHECK_FALSE(io::read_audio_excerpt((f.work/"source.wav").string(),0,16));
    CHECK_FALSE(io::read_audio_excerpt((f.work/"source.wav").string(),-1,1));
    REQUIRE(io::write_audio_excerpt((f.work/"source.wav").string(),(f.root/"excerpt.wav").string(),.5,1.5));
    auto saved=io::read_audio_excerpt((f.root/"excerpt.wav").string(),0,1);REQUIRE(saved);CHECK(saved->data==part->data);
    fs::create_symlink(f.work/"source.wav",f.root/"alias.wav");CHECK_FALSE(io::inspect_audio((f.root/"alias.wav").string()));
    fs::remove(f.work/"conversion_blocks/conversion/f0_800.block");REQUIRE_THROWS(review::read_evidence(f.work));
}
TEST_CASE("preview forks only dependency-equivalent blocks and resumes a partial seed") {
    ReviewFixture f;auto next=f.config;next.pitch_edits={{PitchEditMode::kShift,1,1.15,12}};
    const auto checksum=io::sha256_file(f.work/"conversion_blocks/conversion/synth_64000.block");
    auto seed=[&](const fs::path& target,const VCConfig& c,std::function<void()> boundary=[] {}) {
        return review::seed_preview(f.work,target,f.work/"source.wav",f.program,f.config,c,.5,1.5,"",boundary);
    };
    REQUIRE(seed(f.root/"preview",next)==1);
    auto id=review::operation_identity(next,f.work/"source.wav",f.program);
    {io::BlockCheckpoint store(f.root/"preview/conversion",core::sha256(id+":conversion"),io::BlockCheckpoint::Mode::ReadOnly);
     CHECK_FALSE(store.load("synth_0",0,160000));REQUIRE(store.load("synth_64000",64000,160000));REQUIRE(store.load("f0_800",800,401));}
    CHECK(io::sha256_file(f.work/"conversion_blocks/conversion/synth_64000.block")==checksum);
    REQUIRE_THROWS_AS(seed(f.root/"paused",next,[&]{if(fs::exists(f.root/"paused.seed"))throw io::OperationPaused();}),io::OperationPaused);
    CHECK_FALSE(fs::exists(f.root/"paused"));CHECK(fs::exists(f.root/"paused.seed"));REQUIRE(seed(f.root/"paused",next)==1);
    next.f0_up_key=1;REQUIRE_THROWS(seed(f.root/"different",next));next.f0_up_key=0;
    next.pitch_edits={{PitchEditMode::kShift,4,4.1,12}};REQUIRE_THROWS(seed(f.root/"outside",next));
    next.pitch_edits={{PitchEditMode::kShift,1,1.15,12}};
    {std::fstream data(f.work/"conversion_blocks/conversion/synth_64000.block",std::ios::binary|std::ios::in|std::ios::out);data.seekp(-1,std::ios::end);data.put('x');}
    REQUIRE_THROWS(seed(f.root/"corrupt",next));CHECK_FALSE(fs::exists(f.root/"corrupt"));
}
TEST_CASE("pitch report HTTP is read-only and missing evidence does not poison tasks") {
    ReviewFixture f;using namespace resonantia::web;
    Paths paths;paths.tasks=(f.root/"tasks").string();paths.runs=(f.root/"runs").string();paths.build=f.root.string();
    const std::string id="pitch-http-fixture";
    {TaskStore store(paths.tasks);TaskRecord r;r.id=id;r.kind="convert";r.state="succeeded";r.name="fixture.wav";
     r.parameters={{"recovery","audio-v1"},{"voice","base"},{"workflow","dry"},{"output_kind","vocal"},{"pitch_edits",""},{"f0_up_key","0"},{"filter_radius","3"},{"index_rate","0"},{"rms_mix_rate","0.5"},{"protect","0.5"},{"pp_separate","0"},{"pp_deharmony","0"},{"pp_dereverb","0"},{"pp_deecho","0"},{"pp_denoise","0"}};
     REQUIRE(store.save(r,0));fs::rename(f.work,store.task_directory(id)/"work");
     r.id="pitch-failed-fixture";r.revision=0;r.state="failed";r.parameters["source_task"]=id;r.parameters["pitch_edits"]="smooth:1:2";REQUIRE(store.save(r,0));}
    struct NoProcesses:ProcessRunner {ProcessResult run(const ProcessSpec&)const override{throw std::runtime_error("report attempted model execution");}};
    auto service=std::make_shared<WebService>(paths,std::make_shared<NoProcesses>());
    auto query=[&](const std::string& path,const std::string& requested="",const std::map<std::string,std::string>& fields=std::map<std::string,std::string>{}) {
        int fd[2];REQUIRE(socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0);Request r;r.method="GET";r.path=path;r.query="id="+(requested.empty()?id:requested);
        if(!fields.empty()) {r.method="POST";r.content_type="multipart/form-data; boundary=review";
            for(const auto& [key,value]:fields)r.body+="--review\r\nContent-Disposition: form-data; name=\""+key+"\"\r\n\r\n"+value+"\r\n";
            r.body+="--review--\r\n";}
        service->handle_request(fd[0],r);close(fd[0]);std::string wire;char buffer[4096];ssize_t n;
        while((n=read(fd[1],buffer,sizeof buffer))>0)wire.append(buffer,n);close(fd[1]);return wire;
    };
    auto report=query("/api/task/pitch");CHECK(report.find("200 OK")!=report.npos);CHECK(report.find("疑似低了一个八度")!=report.npos);
    CHECK(report.find("\"start\":1")!=report.npos);CHECK(query("/api/task/pitch","../invalid").find("400")!=std::string::npos);
    const auto fallback=query("/api/task/pitch","pitch-failed-fixture");
    CHECK(fallback.find("失败的修音没有应用")!=fallback.npos);CHECK(fallback.find("\"baseline_id\":\""+id+"\"")!=fallback.npos);
    const auto at=report.find("\"analysis_id\":\"");REQUIRE(at!=report.npos);
    std::map<std::string,std::string> fields{{"task_id","pitch-preview-fixture"},{"source_task",id},{"analysis_id",report.substr(at+15,64)},{"preview_start","0"},{"preview_end","2"},{"pitch_edits",""}};
    CHECK(query("/api/tasks/pitch-preview","",fields).find("没有需要试听")!=std::string::npos);
    fields["pitch_edits"]="shift:1:1.15:12";fields["preview_start"]="4";fields["preview_end"]="5";
    CHECK(query("/api/tasks/pitch-preview","",fields).find("当前片段")!=std::string::npos);
    fields["preview_start"]="0";fields["preview_end"]="2";fields["analysis_id"]=std::string(64,'0');
    CHECK(query("/api/tasks/pitch-preview","",fields).find("分析已变化")!=std::string::npos);
    fields["f0_up_key"]="1";CHECK(query("/api/tasks/pitch-preview","",fields).find("400")!=std::string::npos);
    fs::remove(fs::path(paths.tasks)/id/"work/conversion_blocks/conversion/f0_800.block");
    CHECK(query("/api/task/pitch").find("409")!=std::string::npos);
    CHECK(query("/api/task").find("200 OK")!=std::string::npos);
}
TEST_CASE("preview invalidates adjacent context and refuses more than two synthesis cores") {
    ReviewFixture f;auto next=f.config;next.pitch_edits={{PitchEditMode::kShift,3.9,4.05,12}};
    REQUIRE(review::seed_preview(f.work,f.root/"boundary",f.work/"source.wav",f.program,f.config,next,3,5,"",[]{})==2);
    CHECK_FALSE(fs::exists(f.root/"boundary/conversion/synth_0.block"));
    CHECK_FALSE(fs::exists(f.root/"boundary/conversion/synth_64000.block"));
    const auto dir=f.work/"conversion_blocks/conversion";
    fs::remove(dir/"synth_0.block");fs::remove(dir/"synth_64000.block");
    const auto identity=review::operation_identity(f.config,f.work/"source.wav",f.program);
    {io::BlockCheckpoint store(dir,core::sha256(identity+":conversion"),io::BlockCheckpoint::Mode::Resume);
     const std::vector<float> samples(80000,.1f);for(int offset:{0,32000,64000,96000})store.save("synth_"+std::to_string(offset),offset,samples);}
    REQUIRE_THROWS_WITH(review::seed_preview(f.work,f.root/"too-many",f.work/"source.wav",f.program,f.config,next,3,5,"",[]{}),"本次变化影响超过两个合成片段，请缩小修音范围");
    CHECK_FALSE(fs::exists(f.root/"too-many"));
}
