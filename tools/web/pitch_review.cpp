#include "tools/web/pitch_review.h"
#include "tools/app_support/operation_checkpoint.h"
#include "voxmutatio/io/audio_io.h"
#include "voxmutatio/io/file_digest.h"
#include <algorithm>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <map>
namespace resonantia::web::pitch_review {
namespace fs=std::filesystem;
using voxmutatio::io::BlockCheckpoint;
using Mode=BlockCheckpoint::Mode;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void directory(const fs::path& path){require(fs::is_directory(fs::symlink_status(path)),"修音缓存目录缺失或使用链接");}
fs::path input_path(const fs::path& work) {
    directory(work);auto input=work/"lead40.wav";
    if(!fs::exists(fs::symlink_status(input)))input=work/"source.wav";
    require(fs::is_regular_file(fs::symlink_status(input)),"原主唱输入未保留");return input;
}
Evidence read_evidence(const fs::path& work) {
    Evidence out;out.input=input_path(work);const auto info=voxmutatio::io::inspect_audio(out.input.string());
    require(info&&info->sample_rate>0&&info->frames>1&&info->frames<=600ULL*info->sample_rate,"音高检查需要有效的10分钟以内音频");
    out.samples16k=info->frames*16000/info->sample_rate;require(out.samples16k>1,"音频过短");
    out.duration=out.samples16k/16000.;const auto padding=std::min<std::uint64_t>(32000,out.samples16k-1);out.pad=padding/16000.;
    const auto root=work/"conversion_blocks";directory(root);directory(root/"owner");directory(root/"conversion");
    out.identity=BlockCheckpoint::stored_identity(root/"owner");
    BlockCheckpoint owner(root/"owner",out.identity,Mode::ReadOnly);
    BlockCheckpoint store(root/"conversion",voxmutatio::core::sha256(out.identity+":conversion"),Mode::ReadOnly);
    const auto count=(out.samples16k+2*padding)/160+1;
    require(count<=61100,"音高数据超过边界");out.raw.reserve(count);
    for(std::uint64_t i=0;i<count;i+=800) {
        const auto n=std::min<std::uint64_t>(800,count-i);
        auto block=store.load("f0_"+std::to_string(i),i,n);
        require(block.has_value(),"尚未保存完整音高，不能自动分析");
        out.raw.insert(out.raw.end(),block->begin(),block->end());
    }
    return out;
}
namespace {
std::string identity_with(const voxmutatio::VCConfig& c,const fs::path& input,const fs::path& program,
    const std::function<std::string(const fs::path&)>& digest) {
    voxmutatio::app_support::OperationIdentity id("conversion",digest(program));
    id.field("recipe",voxmutatio::app_support::conversion_recipe(c,0));id.field("input",digest(input));
    id.field("hubert",digest(c.hubert_model_path));id.field("generator",digest(c.synthesizer_model_path));
    id.field("pitch",digest(c.rmvpe_model_path));if(c.index_rate>0)id.field("index",digest(c.index_path));
    return id.finish();
}
}
std::string operation_identity(const voxmutatio::VCConfig& c,const fs::path& input,const fs::path& program) {
    return identity_with(c,input,program,[](const fs::path& file){return voxmutatio::io::sha256_file(file);});
}
std::size_t seed_preview(const fs::path& parent_work,const fs::path& destination,const fs::path& input,
    const fs::path& program,const voxmutatio::VCConfig& previous,const voxmutatio::VCConfig& next,
    double start,double end,const std::string& pause_file,const std::function<void()>& boundary) {
    require(std::isfinite(start)&&std::isfinite(end)&&start>=0&&end>start&&end-start<=15,"试听范围须在15秒内");
    auto a=previous,b=next;a.pitch_edits.clear();b.pitch_edits.clear();
    require(voxmutatio::app_support::conversion_recipe(a,0)==voxmutatio::app_support::conversion_recipe(b,0),"快速试听只支持局部修音，请先恢复基线参数");
    // Per-operation memo only: the child revalidates all dependencies on resume.
    std::map<fs::path,std::string> digests;
    const auto digest=[&](const fs::path& file)->std::string {
        boundary();auto [it,inserted]=digests.try_emplace(file);
        if(inserted)it->second=voxmutatio::io::sha256_file(file);return it->second;
    };
    for(const auto& pair:std::vector<std::pair<std::string,std::string>>{{a.hubert_model_path,b.hubert_model_path},{a.rmvpe_model_path,b.rmvpe_model_path},{a.synthesizer_model_path,b.synthesizer_model_path}})
        require(digest(pair.first)==digest(pair.second),"试听模型与基线不同");
    if(a.index_rate>0)require(digest(a.index_path)==digest(b.index_path),"试听索引与基线不同");
    const auto evidence=read_evidence(parent_work);require(end<=evidence.duration,"试听超出音频长度");
    require(digest(input)==digest(evidence.input),"试听主唱与基线不同");
    const auto old_id=identity_with(previous,evidence.input,program,digest),new_id=identity_with(next,input,program,digest);
    require(old_id==evidence.identity,"基线计算程序、素材或模型已变化，不能快速试听");
    const auto root=parent_work/"conversion_blocks";
    BlockCheckpoint owner(root/"owner",old_id,Mode::ReadOnly);
    BlockCheckpoint old(root/"conversion",voxmutatio::core::sha256(old_id+":conversion"),Mode::ReadOnly);
    auto old_curve=voxmutatio::f0::prepare_pitch_curve(evidence.raw,previous.filter_radius,previous.f0_up_key,previous.pitch_edits,evidence.pad,evidence.duration);
    auto new_curve=voxmutatio::f0::prepare_pitch_curve(evidence.raw,next.filter_radius,next.f0_up_key,next.pitch_edits,evidence.pad,evidence.duration);
    std::vector<std::size_t> changed;
    for(std::size_t i=0;i<old_curve.size();++i)if(old_curve[i]!=new_curve[i]) {
        const double t=i/100.-evidence.pad;require(t>=start-.011&&t<=end+.011,"请只预览当前片段内的修音变化");changed.push_back(i);
    }
    auto entries=old.entries();std::vector<BlockCheckpoint::Entry> cores;
    for(const auto& e:entries) {
        if(e.key.starts_with("synth_")){require(e.key=="synth_"+std::to_string(e.offset),"合成块位置无效");cores.push_back(e);}
        else require(e.key=="f0_"+std::to_string(e.offset)&&e.offset<evidence.raw.size()&&e.offset%800==0,"未知缓存块");
    }
    std::sort(cores.begin(),cores.end(),[](const auto& x,const auto& y){return x.offset<y.offset;});
    require(!cores.empty()&&cores.size()<=1024&&cores.front().offset==0,"基线缺少完整合成缓存");
    const auto pad=static_cast<std::uint64_t>(std::llround(evidence.pad*16000));
    std::vector<bool> invalidate;std::size_t recompute=0;
    for(std::size_t i=0;i<cores.size();++i) {
        const auto& c=cores[i];const auto end_sample=i+1<cores.size()?cores[i+1].offset:evidence.samples16k;
        require(end_sample>c.offset&&end_sample<=evidence.samples16k&&c.count==end_sample*40000/16000-c.offset*40000/16000,"合成缓存不连续或长度不匹配");
        const auto first=c.offset/160,last=std::min<std::uint64_t>(evidence.raw.size(),std::min(evidence.samples16k+2*pad,end_sample+2*pad+160)/160);
        const auto found=std::lower_bound(changed.begin(),changed.end(),first);
        const bool affected=found!=changed.end()&&*found<last;invalidate.push_back(affected);recompute+=affected;
        require(old.load(c.key,c.offset,c.count).has_value(),"基线合成块缺失");
    }
    require(recompute<=2,"本次变化影响超过两个合成片段，请缩小修音范围");
    if(fs::exists(fs::symlink_status(destination))) {
        // Resumed preview already owns a complete atomic seed. CLI validates its
        // identities and checksums before reuse; never re-seed over committed work.
        require(BlockCheckpoint::stored_identity(destination/"owner")==new_id,"试听缓存身份不匹配");return recompute;
    }
    const fs::path staging=destination.string()+".seed";
    {
        voxmutatio::app_support::OperationCheckpointSession session(staging,new_id,fs::exists(fs::symlink_status(staging)),pause_file);
        auto target=session.stage("conversion");
        for(std::size_t i=0;i<evidence.raw.size();i+=800) {
            boundary();target->save("f0_"+std::to_string(i),i,std::span(evidence.raw).subspan(i,std::min<std::size_t>(800,evidence.raw.size()-i)));
        }
        for(std::size_t i=0;i<cores.size();++i)if(!invalidate[i]) {
            boundary();auto data=old.load(cores[i].key,cores[i].offset,cores[i].count);require(data.has_value(),"基线合成块缺失");
            target->save(cores[i].key,cores[i].offset,*data);
        }
    }
    fs::rename(staging,destination);
    const int fd=::open(destination.parent_path().c_str(),O_RDONLY|O_CLOEXEC|O_DIRECTORY|O_NOFOLLOW);
    require(fd>=0,"试听保存点目录无法同步");
    const auto synced=::fsync(fd),closed=::close(fd);require(!synced&&!closed,"试听保存点目录同步失败");
    return recompute;
}
void write_excerpt(const fs::path& input,const fs::path& output,double start,double end) {
    require(voxmutatio::io::write_audio_excerpt(input.string(),output.string(),start,end),"试听需要有效范围内的单声道PCM16音频，保存失败");
}
}
