// Copyright 2026 Resonantia Authors. Licensed under the MIT License.
#include "voxmutatio/training/joint_trainer.h"
#include "voxmutatio/training/mel_loss.h"
#include "voxmutatio/training/discriminator.h"
#include "voxmutatio/io/safetensors.h"
#include "export_weights.h"
#include "joint_model_contract.h"
#include "joint_nsf.h"
#include "joint_checkpoint.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace voxmutatio::training {
namespace ag = voxmutatio::autograd;
using ag::Tensor;
namespace {
constexpr int kMaxFrames=600;
void finite(const std::vector<float>& x) {
  for (float v:x) if (!std::isfinite(v)) throw std::invalid_argument("nonfinite joint input/parameter");
}
Tensor constant(float x, std::vector<int> shape) {
  std::size_t n=1; for(int d:shape) n*=d;
  return Tensor::from_host(std::vector<float>(n,x),std::move(shape));
}
Tensor cut_time(const Tensor& x,int start,int count) {
  int c=x.shape().at(0), t=x.shape().at(1);
  return ag::transpose2d(ag::slice_rows(ag::transpose2d(x,c,t),start,count,c),count,c);
}
Tensor broadcast_channel(const Tensor& condition,int channels,int frames) {
  return ag::matmul(ag::reshape(condition,{channels,1}),constant(1,{1,frames}));
}
Tensor mean(const Tensor& x) { return ag::scale(ag::sum(x),1.0f/x.numel()); }
Tensor square_mean(const Tensor& x) { return mean(ag::mul(x,x)); }
Tensor difference(const Tensor& a,const Tensor& b) { return ag::add(a,ag::scale(b,-1)); }
Tensor detach(const Tensor& x) { return Tensor::from_host(x.to_host(),x.shape()); }
float scalar(const Tensor& x) {
  float y=x.to_host().at(0);
  if(!std::isfinite(y)) throw std::runtime_error("nonfinite joint loss");
  return y;
}

struct Bank {
  std::map<std::string,Tensor> p;
  std::map<std::string,Tensor> weights;
  std::map<std::string,std::vector<int>> shapes;
  std::vector<detail::ExportTensor> preserved;
  std::vector<float> original_embedding;
  int speaker=0;
  void load(const std::string& path,bool generator,int sid) {
    io::SafetensorsLoader loader;
    if(!loader.load(path)) throw std::invalid_argument("cannot load joint checkpoint");
    const auto expected=detail::joint_manifest(generator);
    if(loader.tensor_names().size()!=expected.size())throw std::invalid_argument("unexpected joint parameter manifest");
    for(const auto& [name,shape]:expected) {
      const auto* t=loader.get_tensor(name);std::size_t count=1;for(int d:shape)count*=d;
      if(!t || t->dtype!="F32" || t->shape!=std::vector<int64_t>(shape.begin(),shape.end()) ||
          t->data_nbytes!=count*sizeof(float) || !loader.data(name))
        throw std::invalid_argument("invalid joint parameter layout: "+name);
    }
    speaker=sid;
    for(const auto& name:loader.tensor_names()) {
      const auto* t=loader.get_tensor(name);
      if(t->dtype!="F32" || t->data_nbytes%4) throw std::invalid_argument("joint checkpoint requires F32");
      std::vector<int> shape;
      for(auto d:t->shape) {
        if(d<=0 || d>std::numeric_limits<int>::max()) throw std::invalid_argument("invalid parameter shape");
        shape.push_back(d);
      }
      std::vector<float> data(t->data_nbytes/4);std::memcpy(data.data(),loader.data(name),t->data_nbytes);finite(data);
      if(name.ends_with(".weight_v")) {
        const std::size_t inner=data.size()/shape[0];
        for(int row=0;row<shape[0];++row) {
          double squared=0;for(std::size_t j=0;j<inner;++j){double v=data[row*inner+j];squared+=v*v;}
          const float represented=static_cast<float>(squared);
          if(!(represented>0) || !std::isfinite(represented))throw std::invalid_argument("unrepresentable joint weight norm: "+name);
        }
      }
      shapes[name]=shape;
      if(generator && name.starts_with("model.enc_q.")) preserved.push_back({name,shape,data});
      if(generator && name=="model.emb_g.weight") {
        if(shape!=std::vector<int>{109,256} || sid<0 || sid>=109)
          throw std::invalid_argument("joint speaker table must be [109,256]");
        original_embedding=data;
        data=std::vector<float>(data.begin()+sid*256,data.begin()+(sid+1)*256);
        shape={1,256};
      }
      p.emplace(name,Tensor::from_host(data,shape,true));
    }
  }
  Tensor get(const std::string& name) const { return p.at(name); }
  Tensor weight(const std::string& prefix) {
    auto it=weights.find(prefix); if(it!=weights.end()) return it->second;
    auto direct=p.find(prefix+".weight");
    if(direct!=p.end()) return direct->second;
    auto w=ag::weight_norm(get(prefix+".weight_v"),get(prefix+".weight_g"));
    weights[prefix]=w; return w;
  }
  std::vector<Tensor> params() const {
    std::vector<Tensor> out; for(const auto& [_,t]:p) out.push_back(t); return out;
  }
  Tensor conv(const Tensor& x,const std::string& prefix,int stride=1,int pad=-1,int dilation=1,int groups=1) {
    const auto w=weight(prefix); const auto s=w.shape();
    if(s.size()!=3 || x.shape().size()!=2) throw std::invalid_argument("joint conv shape");
    if(pad<0) pad=dilation*(s[2]-1)/2;
    const auto b=p.contains(prefix+".bias")?get(prefix+".bias"):Tensor{};
    return ag::conv1d(x,w,b,x.shape()[0],x.shape()[1],s[0],s[2],stride,pad,dilation,groups);
  }
  Tensor condition(const Tensor& g,const std::string& prefix) {
    auto w=weight(prefix);
    auto matrix=ag::reshape(w,{w.shape()[0],256});
    auto bias=get(prefix+".bias");
    return ag::linear(g,matrix,bias);
  }
};

Tensor wavenet(Bank& b,Tensor x,const Tensor& g,const std::string& prefix,int layers) {
  const int h=192,T=x.shape()[1]; Tensor output;
  auto condition=ag::reshape(b.condition(g,prefix+".cond_layer"),{layers*2*h,1});
  for(int i=0;i<layers;++i) {
    auto activation=b.conv(x,prefix+".in_layers."+std::to_string(i));
    auto c=ag::slice_rows(condition,i*2*h,2*h,1);
    activation=ag::add(activation,broadcast_channel(c,2*h,T));
    auto gates=ag::mul(ag::tanh_op(ag::slice_rows(activation,0,h,T)),
                       ag::sigmoid(ag::slice_rows(activation,h,h,T)));
    auto rs=b.conv(gates,prefix+".res_skip_layers."+std::to_string(i));
    if(i<layers-1) {
      x=ag::add(x,ag::slice_rows(rs,0,h,T));
      auto skip=ag::slice_rows(rs,h,h,T); output=output.n?ag::add(output,skip):skip;
    } else output=output.n?ag::add(output,rs):rs;
  }
  return output;
}
Tensor flow(Bank& b,Tensor x,const Tensor& g) {
  const int T=x.shape()[1];
  for(int i=0;i<4;++i) {
    const auto p="model.flow.flows."+std::to_string(i*2);
    auto x0=ag::slice_rows(x,0,96,T), x1=ag::slice_rows(x,96,96,T);
    auto h=wavenet(b,b.conv(x0,p+".pre"),g,p+".enc",3);
    x=ag::flip_rows(ag::concat_rows(x0,ag::add(x1,b.conv(h,p+".post")),T),192,T);
  }
  return x;
}
Tensor norm(Bank& b,const Tensor& x,const std::string& p) {
  const int T=x.shape()[1], C=x.shape()[0];
  return ag::transpose2d(ag::layer_norm(ag::transpose2d(x,C,T),b.get(p+".gamma"),b.get(p+".beta"),T,C),T,C);
}
std::pair<Tensor,Tensor> prior(Bank& b,const Tensor& phone,const std::vector<int>& pitch) {
  const int T=phone.shape()[0],D=96,W=21;
  auto x=ag::linear(phone,b.get("model.enc_p.emb_phone.weight"),b.get("model.enc_p.emb_phone.bias"));
  x=ag::leaky_relu(ag::scale(ag::add(x,ag::embedding(b.get("model.enc_p.emb_pitch.weight"),pitch,192)),std::sqrt(192.f)),.1f);
  x=ag::transpose2d(x,T,192);
  std::vector<int> to_abs(T*T),to_rel(T*W);
  std::vector<float> abs_mask(T*T),rel_mask(T*W);
  for(int i=0;i<T;++i) {
    for(int j=0;j<T;++j) { int d=j-i+10; bool ok=d>=0&&d<W; to_abs[i*T+j]=ok?i*W+d:0; abs_mask[i*T+j]=ok?1:0; }
    for(int d=0;d<W;++d) { int j=i+d-10; bool ok=j>=0&&j<T; to_rel[i*W+d]=ok?i*T+j:0; rel_mask[i*W+d]=ok?1:0; }
  }
  auto ma=Tensor::from_host(abs_mask,{T,T}), mr=Tensor::from_host(rel_mask,{T,W});
  for(int i=0;i<6;++i) {
    const std::string root="model.enc_p.encoder.";
    const auto p=root+"attn_layers."+std::to_string(i);
    auto q=b.conv(x,p+".conv_q"), k=b.conv(x,p+".conv_k"), v=b.conv(x,p+".conv_v"); Tensor all;
    auto rk=ag::reshape(b.get(p+".emb_rel_k"),{W,D});
    auto rv=ag::reshape(b.get(p+".emb_rel_v"),{W,D});
    for(int head=0;head<2;++head) {
      auto qh=ag::scale(ag::transpose2d(ag::slice_rows(q,head*D,D,T),D,T),1/std::sqrt(float(D)));
      auto kh=ag::slice_rows(k,head*D,D,T);
      auto vh=ag::transpose2d(ag::slice_rows(v,head*D,D,T),D,T);
      auto qr=ag::matmul(qh,ag::transpose2d(rk,W,D));
      auto rel=ag::reshape(ag::embedding(ag::reshape(qr,{T*W,1}),to_abs,1),{T,T});
      auto probability=ag::softmax_rows(ag::add(ag::matmul(qh,kh),ag::mul(rel,ma)),T,T);
      auto pr=ag::reshape(ag::embedding(ag::reshape(probability,{T*T,1}),to_rel,1),{T,W});
      auto out=ag::add(ag::matmul(probability,vh),ag::matmul(ag::mul(pr,mr),rv));
      out=ag::transpose2d(out,T,D); all=all.n?ag::concat_rows(all,out,T):out;
    }
    x=norm(b,ag::add(x,b.conv(all,p+".conv_o")),root+std::string("norm_layers_1.")+std::to_string(i));
    auto ff=root+std::string("ffn_layers.")+std::to_string(i);
    auto y=b.conv(ag::relu(b.conv(x,ff+".conv_1")),ff+".conv_2");
    x=norm(b,ag::add(x,y),root+std::string("norm_layers_2.")+std::to_string(i));
  }
  auto stats=b.conv(x,"model.enc_p.proj");
  return {ag::slice_rows(stats,0,192,T),ag::slice_rows(stats,192,192,T)};
}

Tensor decode(Bank& b,Tensor z,const Tensor& g,const std::vector<float>& f0,
              const std::vector<float>& noise) {
  // The standalone diagnostic recompilation may retain activation snapshots.
  // Production builds have no callback, extra copies or changed derivatives.
#ifndef VOXMUTATIO_JOINT_OBSERVE
#define VOXMUTATIO_JOINT_OBSERVE(label, tensor) ((void)0)
#define VOXMUTATIO_JOINT_LOCAL_OBSERVER
#endif
  const int T=f0.size(), L=T*400;
  auto sine=detail::joint_sine_source(f0,noise);
  auto excitation=Tensor::from_host(sine,{L,1});
  VOXMUTATIO_JOINT_OBSERVE("model.dec.m_source.l_linear.input",excitation);
  auto source=ag::linear(excitation,b.get("model.dec.m_source.l_linear.weight"),b.get("model.dec.m_source.l_linear.bias"));
  VOXMUTATIO_JOINT_OBSERVE("model.dec.m_source.l_linear.output",source);
  auto har=ag::reshape(ag::tanh_op(source),{1,L});
  VOXMUTATIO_JOINT_OBSERVE("model.dec.m_source.output",har);
  auto x=ag::add(b.conv(z,"model.dec.conv_pre"),broadcast_channel(b.condition(g,"model.dec.cond"),512,T));
  const int rates[]={10,10,2,2};
  for(int i=0;i<4;++i) {
    const auto prefix="model.dec.ups."+std::to_string(i);
    const auto w=b.weight(prefix); int C=x.shape()[0], len=x.shape()[1],K=w.shape()[2];
    auto activated=ag::leaky_relu(x,.1f);
    VOXMUTATIO_JOINT_OBSERVE(prefix+".activation",activated);
    x=ag::conv_transpose1d(activated,w,b.get(prefix+".bias"),C,len,C/2,K,rates[i],(K-rates[i])/2);
    VOXMUTATIO_JOINT_OBSERVE(prefix+".output",x);
    int stride=1; for(int j=i+1;j<4;++j) stride*=rates[j];
    auto n=b.conv(har,"model.dec.noise_convs."+std::to_string(i),stride,i==3?0:stride/2);
    VOXMUTATIO_JOINT_OBSERVE("model.dec.noise_convs."+std::to_string(i)+".output",n);
    auto xn=ag::add(x,n); Tensor sum;
    for(int block=0;block<3;++block) {
      auto y=xn; const auto p="model.dec.resblocks."+std::to_string(i*3+block);
      const int dil[]={1,3,5};
      for(int j=0;j<3;++j) {
        auto a=ag::leaky_relu(y,.1f);
        VOXMUTATIO_JOINT_OBSERVE(p+".convs1."+std::to_string(j)+".activation",a);
        auto h=b.conv(a,p+".convs1."+std::to_string(j),1,-1,dil[j]);
        auto h_activated=ag::leaky_relu(h,.1f);
        VOXMUTATIO_JOINT_OBSERVE(p+".convs2."+std::to_string(j)+".activation",h_activated);
        y=ag::add(y,b.conv(h_activated,p+".convs2."+std::to_string(j)));
      }
      sum=sum.n?ag::add(sum,y):y;
    }
    x=ag::scale(sum,1.f/3);
  }
  auto activated=ag::leaky_relu(x,.01f);
  VOXMUTATIO_JOINT_OBSERVE("model.dec.conv_post.activation",activated);
#ifdef VOXMUTATIO_JOINT_LOCAL_OBSERVER
#undef VOXMUTATIO_JOINT_OBSERVE
#undef VOXMUTATIO_JOINT_LOCAL_OBSERVER
#endif
  return ag::tanh_op(b.conv(activated,"model.dec.conv_post"));
}
std::vector<SubDiscResult> discriminate(Bank& b,const Tensor& audio) {
  std::vector<SubDiscResult> out;
  const int groups[]={1,4,16,64,256,1},strides[]={1,4,4,4,4,1};
  SubDiscResult s; auto x=audio;
  for(int i=0;i<6;++i) {
    x=ag::leaky_relu(b.conv(x,"model.discriminators.0.convs."+std::to_string(i),strides[i],-1,1,groups[i]),.1f);
    s.fmaps.push_back(x);
  }
  s.score=b.conv(x,"model.discriminators.0.conv_post"); s.fmaps.push_back(s.score); out.push_back(s);
  const int periods[]={2,3,5,7,11,17,23,37};
  for(int d=0;d<8;++d) {
    const int p=periods[d], L=audio.shape()[1], pad=(p-L%p)%p;
    auto h=pad?ag::reflect_pad_end(audio,L+pad):audio;
    int C=1,H=(L+pad)/p; SubDiscResult result;
    for(int i=0;i<6;++i) {
      const auto key="model.discriminators."+std::to_string(d+1)+(i<5?".convs."+std::to_string(i):".conv_post");
      const auto w=b.weight(key); int cout=w.shape()[0],kh=w.shape()[2],stride=i<4?3:1;
      h=ag::conv2d(h,w,b.get(key+".bias"),C,H,p,cout,kh,1,stride,1,(kh-1)/2,0);
      H=(H-1)/stride+1; C=cout;
      if(i<5) h=ag::leaky_relu(h,.1f);
      result.fmaps.push_back(h);
    }
    result.score=h; out.push_back(result);
  }
  return out;
}
Tensor adversarial(const std::vector<SubDiscResult>& fake) {
  Tensor sum;
  for(const auto& d:fake) {
    auto x=ag::add(d.score,constant(-1,d.score.shape())); auto l=square_mean(x);
    sum=sum.n?ag::add(sum,l):l;
  }
  return sum;
}
void validate_clip(const JointTrainingClip& c,const JointStepOptions& o) {
  const int T=c.frames;
  if(T<3 || T>kMaxFrames || o.crop_frames<8 || o.crop_frames>T || o.crop_start<0 ||
      o.crop_start>T-o.crop_frames || c.phone.size()!=static_cast<std::size_t>(T)*768 ||
      c.pitch.size()!=static_cast<std::size_t>(T) || c.f0.size()!=static_cast<std::size_t>(T) ||
      c.spectrum.size()!=static_cast<std::size_t>(T)*1025 || c.audio.size()!=static_cast<std::size_t>(T)*400 ||
      (!o.posterior_noise.empty() && o.posterior_noise.size()!=static_cast<std::size_t>(T)*192) ||
      (!o.nsf_noise.empty() && o.nsf_noise.size()!=static_cast<std::size_t>(o.crop_frames)*400))
    throw std::invalid_argument("joint clip/crop/noise dimensions exceed bounded contract");
  finite(c.phone); finite(c.f0); finite(c.spectrum); finite(c.audio); finite(o.posterior_noise); finite(o.nsf_noise);
  for(int p:c.pitch) if(p<1 || p>255) throw std::invalid_argument("joint coarse pitch out of range");
  for(float f:c.f0) if(f<0 || f>20000) throw std::invalid_argument("joint F0 out of range");
  for(float f:c.spectrum) if(f<0) throw std::invalid_argument("negative magnitude");
}
} // namespace

struct JointTrainer::Impl {
  Bank g,d;
  std::unique_ptr<ag::AdamW> go,dop;
  std::unique_ptr<MelLoss> mel;
  std::mt19937 rng;
  std::map<std::string,std::vector<float>> d_grad;
  std::vector<float> noise(std::size_t n) {
    std::normal_distribution<float> normal(0,1); std::vector<float> values(n);
    for(auto& v:values) v=normal(rng); return values;
  }
};
JointTrainer::JointTrainer()=default;
JointTrainer::~JointTrainer()=default;
JointTrainer::JointTrainer(JointTrainer&&) noexcept=default;
JointTrainer& JointTrainer::operator=(JointTrainer&&) noexcept=default;
bool JointTrainer::init(const std::string& generator,const std::string& discriminator,int speaker,float lr,std::uint32_t seed) {
  impl_.reset();
  try {
    if(speaker<0 || speaker>=109 || !std::isfinite(lr) || lr<=0) return false;
    auto p=std::make_unique<Impl>(); p->g.load(generator,true,speaker); p->d.load(discriminator,false,0);
    if(p->g.p.size()!=560) throw std::invalid_argument("unexpected joint generator tensor manifest");
    p->go=std::make_unique<ag::AdamW>(p->g.params(),lr,.8f,.99f,1e-9f,.01f);
    p->dop=std::make_unique<ag::AdamW>(p->d.params(),lr,.8f,.99f,1e-9f,.01f);
    MelSpecConfig mc; mc.n_fft=2048; mc.hop=400; mc.n_mels=125; mc.rvc_profile=true;
    p->mel=std::make_unique<MelLoss>(mc); p->rng.seed(seed); impl_=std::move(p); return true;
  } catch(const std::exception& e) { std::fprintf(stderr,"JointTrainer init: %s\n",e.what()); return false; }
}

JointStepResult JointTrainer::step(const JointTrainingClip& c,const JointStepOptions& o,JointStepTrace* trace) {
  if(!impl_) throw std::logic_error("JointTrainer is not initialized");
  validate_clip(c,o);
  try {
    auto& s=*impl_; s.g.weights.clear(); s.d.weights.clear(); s.d_grad.clear();
    const int T=c.frames,N=o.crop_frames,L=N*400;
    auto eps=o.posterior_noise.empty()?s.noise(T*192):o.posterior_noise;
    auto nsf=o.nsf_noise.empty()?s.noise(L):o.nsf_noise;
    auto embedding=s.g.get("model.emb_g.weight");
    auto [mp,lp]=prior(s.g,Tensor::from_host(c.phone,{T,768}),c.pitch);
    auto post=wavenet(s.g,s.g.conv(Tensor::from_host(c.spectrum,{1025,T}),"model.enc_q.pre"),embedding,"model.enc_q.enc",16);
    auto stats=s.g.conv(post,"model.enc_q.proj");
    auto mq=ag::slice_rows(stats,0,192,T),lq=ag::slice_rows(stats,192,192,T);
    auto zq=ag::add(mq,ag::mul(ag::exp_op(lq),Tensor::from_host(eps,{192,T})));
    auto zp=flow(s.g,zq,embedding);
    std::vector<float> f0(c.f0.begin()+o.crop_start,c.f0.begin()+o.crop_start+N);
    auto y=decode(s.g,cut_time(zq,o.crop_start,N),embedding,f0,nsf);
    std::vector<float> target(c.audio.begin()+o.crop_start*400,c.audio.begin()+(o.crop_start+N)*400);
    auto real=Tensor::from_host(target,{1,L});
    int tm=0; auto target_mel=s.mel->target_log_mel(c.audio.data(),T*400,tm);
    auto mt=ag::slice_rows(Tensor::from_host(target_mel,{tm,125}),o.crop_start,N,125);
    JointStepResult result;
    if(trace) {
      trace->prior_mean=mp.to_host();trace->prior_logs=lp.to_host(); trace->posterior_mean=mq.to_host();
      trace->posterior_logs=lq.to_host();trace->posterior_z=zq.to_host();trace->flow_z=zp.to_host();trace->audio=y.to_host();
    }
    // Complete D update on detached generated audio; G graph remains untouched.
    {
      auto dr=discriminate(s.d,real), df=discriminate(s.d,detach(y));
      if(trace) {
        trace->discriminator_real.clear();trace->discriminator_fake.clear();
        for(const auto& d:dr)for(const auto& f:d.fmaps)trace->discriminator_real.push_back(f.to_host());
        for(const auto& d:df)for(const auto& f:d.fmaps)trace->discriminator_fake.push_back(f.to_host());
      }
      auto loss=adversarial(dr);
      for(const auto& d:df) loss=ag::add(loss,square_mean(d.score));
      result.discriminator=scalar(loss); ag::backward(loss);
      s.dop->clip_grad_norm(std::numeric_limits<float>::max()); // finite check; reference has no norm clipping
      if(trace) for(const auto& [name,p]:s.d.p) s.d_grad[name]=p.grad_to_host();
      if(o.update) s.dop->step();
    }
    s.d.weights.clear();
    Tensor fm;
    {
      auto dr=discriminate(s.d,real);
      // Only constant real feature maps participate in feature matching.
      for(auto& d:dr) { for(auto& f:d.fmaps) f=detach(f); d.score=Tensor{}; }
      auto df=discriminate(s.d,y);
      for(std::size_t i=0;i<df.size();++i) for(std::size_t j=0;j<df[i].fmaps.size();++j) {
        auto term=mean(ag::abs_op(difference(df[i].fmaps[j],dr[i].fmaps[j])));
        fm=fm.n?ag::add(fm,term):term;
      }
      auto adv=adversarial(df);
      auto mel=s.mel->l1(s.mel->log_mel(y,L),mt,N);
      auto diff=difference(zp,mp);
      auto kl=ag::scale(ag::sum(ag::add(ag::add(difference(lp,lq),constant(-.5f,{192,T})),
                    ag::scale(ag::mul(ag::mul(diff,diff),ag::exp_op(ag::scale(lp,-2))),.5f))),1.f/T);
      auto loss=ag::add(ag::add(ag::scale(mel,45),kl),ag::add(ag::scale(fm,2),adv));
      result.generator=scalar(loss);result.mel=scalar(mel);result.kl=scalar(kl);
      result.feature=scalar(fm)*2;result.adversarial=scalar(adv);
      ag::backward(loss); s.go->clip_grad_norm(std::numeric_limits<float>::max());
      if(trace)trace->audio_gradient=y.grad_to_host();
      if(o.update) s.go->step();
    }
    s.g.weights.clear();s.d.weights.clear();return result;
  } catch(...) { impl_.reset();throw; }
}

bool JointTrainer::export_model(const std::string& output) const {
  if(!impl_) return false;
  try {
    std::vector<detail::ExportTensor> entries=impl_->g.preserved;
    for(const auto& [name,p]:impl_->g.p) {
      if(name.starts_with("model.enc_q.")) continue;
      auto data=p.to_host();
      if(name=="model.emb_g.weight") {
        auto all=impl_->g.original_embedding;
        std::copy(data.begin(),data.end(),all.begin()+impl_->g.speaker*256); data=std::move(all);
      }
      entries.push_back({name,impl_->g.shapes.at(name),std::move(data)});
    }
    return detail::write_safetensors_atomic(output,entries);
  } catch(const std::exception& e) { std::fprintf(stderr,"JointTrainer export: %s\n",e.what());return false; }
}

void JointTrainer::set_learning_rate(float lr) {
  if(!impl_)throw std::logic_error("JointTrainer is not initialized");
  if(!std::isfinite(lr) || lr<=0)throw std::invalid_argument("invalid joint learning rate");
  impl_->go->set_lr(lr);impl_->dop->set_lr(lr);
}
float JointTrainer::learning_rate() const {
  if(!impl_)throw std::logic_error("JointTrainer is not initialized");
  return impl_->go->lr();
}
int JointTrainer::step_count() const {
  if(!impl_)throw std::logic_error("JointTrainer is not initialized");
  return impl_->go->step_count();
}

bool JointTrainer::save_checkpoint(const std::string& output,const std::string& run_state) const {
  if(!impl_ || run_state.size()>detail::kJointCheckpointMaxRunState)return false;
  try {
    const auto& s=*impl_;
    if(s.go->step_count()!=s.dop->step_count() || s.go->lr()!=s.dop->lr())return false;
    std::ostringstream random;random.imbue(std::locale::classic());random<<s.rng;
    std::ostringstream meta;meta.imbue(std::locale::classic());
    meta<<"RESONANTIA_JOINT_CHECKPOINT 1\n"<<s.g.speaker<<' '<<s.go->step_count()<<' '
        <<std::setprecision(std::numeric_limits<float>::max_digits10)<<s.go->lr()<<'\n'
        <<std::quoted(random.str())<<'\n'<<std::quoted(run_state)<<'\n';
    std::vector<detail::CheckpointTensor> entries;
    for(bool disc:{false,true}) {
      const auto& bank=disc?s.d:s.g;const auto* optimizer=disc?s.dop.get():s.go.get();
      const std::string prefix=disc?"D.":"G.";std::size_t i=0;
      for(const auto& [name,p]:bank.p) {
        entries.push_back({prefix+"p."+name,p.shape(),[p]{return p.to_host();}});
        entries.push_back({prefix+"m."+name,p.shape(),[optimizer,i]{return optimizer->first_moment(i);}});
        entries.push_back({prefix+"v."+name,p.shape(),[optimizer,i]{return optimizer->second_moment(i);}});++i;
      }
    }
    for(const auto& p:s.g.preserved) {
      const auto* values=&p.data;
      entries.push_back({"original."+p.name,p.shape,[values]{return *values;}});
    }
    entries.push_back({"original.model.emb_g.weight",{109,256},[&s]{return s.g.original_embedding;}});
    detail::write_joint_checkpoint(output,entries,meta.str());return true;
  }catch(const std::exception& e){std::fprintf(stderr,"Joint checkpoint save: %s\n",e.what());return false;}
}

bool JointTrainer::load_checkpoint(const std::string& input,std::string& run_state) {
  if(input.empty() || input.find('\0')!=std::string::npos)return false;
  try {
    io::SafetensorsLoader loader;
    if(!loader.load_snapshot(input,detail::kJointCheckpointMaxBytes))return false;
    const auto* t=loader.get_tensor("state");
    if(!t || t->dtype!="F32" || t->shape.size()!=1 || t->shape[0]<=0 ||
        t->shape[0]>static_cast<int64_t>(detail::kJointCheckpointMaxText) ||
        t->shape[0]*sizeof(float)!=t->data_nbytes)return false;
    const auto encoded=detail::checkpoint_tensor(loader,"state",{static_cast<int>(t->shape[0])});
    std::string text;text.reserve(encoded.size());
    for(float byte:encoded) {
      if(byte<0 || byte>255 || std::trunc(byte)!=byte)return false;
      text.push_back(static_cast<char>(static_cast<unsigned char>(byte)));
    }
    std::istringstream meta(text);meta.imbue(std::locale::classic());
    std::string magic,rng_text,application;int version=0,speaker=-1,steps=-1;float lr=0;
    if(!(meta>>magic>>version>>speaker>>steps>>lr>>std::quoted(rng_text)>>std::quoted(application)) ||
        magic!="RESONANTIA_JOINT_CHECKPOINT" || version!=1 || speaker<0 || speaker>=109 ||
        steps<0 || steps==std::numeric_limits<int>::max() || !std::isfinite(lr) || lr<=0 ||
        application.size()>detail::kJointCheckpointMaxRunState || rng_text.size()>16384)return false;
    meta>>std::ws;if(!meta.eof())return false;
    std::mt19937 rng;std::istringstream random(rng_text);random.imbue(std::locale::classic());
    if(!(random>>rng))return false;random>>std::ws;if(!random.eof())return false;
    // Check the entire fixed manifest before any GPU allocation; no untrusted
    // dimensions can select a different model or increase the resource bound.
    std::map<std::string,std::vector<int>> expected;
    for(bool disc:{false,true})for(const auto& [name,shape]:detail::joint_manifest(!disc)) {
      const auto actual=(!disc && name=="model.emb_g.weight")?std::vector<int>{1,256}:shape;
      for(const char* kind:{"p.","m.","v."})expected[(disc?"D.":"G.")+std::string(kind)+name]=actual;
      if(!disc && name.starts_with("model.enc_q."))expected["original."+name]=shape;
    }
    expected["original.model.emb_g.weight"]={109,256};
    if(loader.tensor_names().size()!=expected.size()+1)return false;
    for(const auto& [name,shape]:expected)detail::checkpoint_tensor(loader,name,shape);
    auto staged=std::make_unique<Impl>();staged->rng=rng;staged->g.speaker=speaker;
    for(bool disc:{false,true}) {
      auto& bank=disc?staged->d:staged->g;
      const std::string prefix=disc?"D.":"G.";
      for(const auto& [name,shape]:detail::joint_manifest(!disc)) {
        const auto actual=(!disc && name=="model.emb_g.weight")?std::vector<int>{1,256}:shape;
        bank.shapes[name]=shape;
        bank.p.emplace(name,Tensor::from_host(detail::checkpoint_tensor(loader,prefix+"p."+name,actual),actual,true));
        if(!disc && name.starts_with("model.enc_q."))
          bank.preserved.push_back({name,shape,detail::checkpoint_tensor(loader,"original."+name,shape)});
      }
      auto optimizer=std::make_unique<ag::AdamW>(bank.params(),lr,.8f,.99f,1e-9f,.01f);
      std::vector<std::string> names;for(const auto& [name,_]:bank.p)names.push_back(name);
      optimizer->restore_state(steps,lr,[&](std::size_t i,bool second){
        const auto& name=names.at(i);
        return detail::checkpoint_tensor(loader,prefix+(second?"v.":"m.")+name,bank.p.at(name).shape());
      });
      (disc?staged->dop:staged->go)=std::move(optimizer);
    }
    staged->g.original_embedding=detail::checkpoint_tensor(loader,"original.model.emb_g.weight",{109,256});
    MelSpecConfig mc;mc.n_fft=2048;mc.hop=400;mc.n_mels=125;mc.rvc_profile=true;
    staged->mel=std::make_unique<MelLoss>(mc);
    impl_.swap(staged);run_state.swap(application);return true;
  }catch(const std::exception& e){std::fprintf(stderr,"Joint checkpoint restore: %s\n",e.what());return false;}
}
std::vector<std::string> JointTrainer::parameter_names(bool discriminator) const {
  if(!impl_) throw std::logic_error("JointTrainer is not initialized");
  std::vector<std::string> result;
  for(const auto& [name,_]:(discriminator?impl_->d:impl_->g).p) result.push_back(name);
  return result;
}
JointParameterSnapshot JointTrainer::inspect_parameter(const std::string& name,bool discriminator) const {
  if(!impl_) throw std::logic_error("JointTrainer is not initialized");
  const auto& bank=discriminator?impl_->d:impl_->g;
  auto it=bank.p.find(name);if(it==bank.p.end()) throw std::out_of_range("unknown joint parameter");
  const auto index=std::distance(bank.p.begin(),it); const auto& optimizer=discriminator?impl_->dop:impl_->go;
  auto grad=discriminator&&impl_->d_grad.contains(name)?impl_->d_grad.at(name):it->second.grad_to_host();
  return {it->second.shape(),it->second.to_host(),std::move(grad),optimizer->first_moment(index),optimizer->second_moment(index)};
}
} // namespace voxmutatio::training
