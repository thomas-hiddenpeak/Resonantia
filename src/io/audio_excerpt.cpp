#include "voxmutatio/io/audio_io.h"
#include "bounded_audio_io.h"
#include <cmath>
namespace voxmutatio::io {
std::optional<AudioFileInfo> inspect_audio(const std::string& path) {
    detail::WavChunkReader reader;if(!reader.open(path,true))return {};
    const auto m=reader.metadata();return AudioFileInfo{m.frame_count,static_cast<int>(m.sample_rate),m.channels};
}
std::optional<AudioBuffer> read_audio_excerpt(const std::string& path,double start,double end) {
    if(!std::isfinite(start)||!std::isfinite(end)||start<0||end<=start||end-start>15||end>600)return {};
    detail::WavChunkReader reader;if(!reader.open(path,true))return {};
    const auto m=reader.metadata();if(m.sample_rate<=0||end>static_cast<double>(m.frame_count)/m.sample_rate)return {};
    const auto first=static_cast<std::uint64_t>(std::llround(start*m.sample_rate));
    const auto last=static_cast<std::uint64_t>(std::llround(end*m.sample_rate));
    if(last<=first||last>m.frame_count||last-first>4'000'000)return {};
    AudioBuffer out;out.sample_rate=m.sample_rate;
    out.source_format=m.encoding==detail::WavEncoding::kPcm16?SampleFormat::kInt16:SampleFormat::kFloat32;
    if(!reader.read_mono_frames(first,static_cast<std::size_t>(last-first),out.data)||!reader.verify_unchanged())return {};
    return out;
}
bool write_audio_excerpt(const std::string& input,const std::string& output,double start,double end) {
    if(!std::isfinite(start)||!std::isfinite(end)||start<0||end<=start||end-start>15||end>600)return false;
    detail::WavChunkReader reader;if(!reader.open(input,true))return false;
    const auto m=reader.metadata();
    if(m.encoding!=detail::WavEncoding::kPcm16||m.channels!=1||!m.sample_rate||end>static_cast<double>(m.frame_count)/m.sample_rate)return false;
    const auto first=static_cast<std::uint64_t>(std::llround(start*m.sample_rate));
    const auto last=static_cast<std::uint64_t>(std::llround(end*m.sample_rate));
    if(last<=first||last>m.frame_count||last-first>4'000'000)return false;
    std::vector<float> samples;
    if(!reader.read_mono_frames(first,static_cast<std::size_t>(last-first),samples)||!reader.verify_unchanged())return false;
    detail::AtomicWavStreamWriter writer;
    return writer.open(output,m.sample_rate,samples.size())&&writer.append_decoded_pcm16(samples)&&writer.commit();
}
}
