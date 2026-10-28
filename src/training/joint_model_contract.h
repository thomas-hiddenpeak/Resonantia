// Fixed v2/40k joint-training checkpoint contract, validated before device allocation.
#pragma once
#include <map>
#include <string>
#include <vector>

namespace voxmutatio::training::detail {
using JointManifest=std::map<std::string,std::vector<int>>;
inline JointManifest joint_manifest(bool generator) {
  JointManifest m;
  const auto conv=[&](const std::string& p,std::vector<int> shape,bool wn=false,bool bias=true,bool transpose=false) {
    m[p+(wn?".weight_v":".weight")]=shape;
    if(wn) {std::vector<int> scale(shape.size(),1);scale[0]=shape[0];m[p+".weight_g"]=scale;}
    if(bias)m[p+".bias"]={shape[transpose?1:0]};
  };
  const auto wn=[&](const std::string& p,int layers) {
    conv(p+".cond_layer",{layers*384,256,1},true);
    for(int i=0;i<layers;++i) {
      conv(p+".in_layers."+std::to_string(i),{384,192,5},true);
      conv(p+".res_skip_layers."+std::to_string(i),{i==layers-1?192:384,192,1},true);
    }
  };
  if(generator) {
    m["model.emb_g.weight"]={109,256};
    conv("model.enc_p.emb_phone",{192,768});m["model.enc_p.emb_pitch.weight"]={256,192};
    for(int i=0;i<6;++i) {
      const auto n=std::to_string(i),p="model.enc_p.encoder.attn_layers."+n;
      for(const char* q:{"q","k","v","o"})conv(p+".conv_"+q,{192,192,1});
      m[p+".emb_rel_k"]={1,21,96};m[p+".emb_rel_v"]={1,21,96};
      for(int j=1;j<=2;++j)for(const char* k:{"gamma","beta"})
        m["model.enc_p.encoder.norm_layers_"+std::to_string(j)+"."+n+"."+k]={192};
      conv("model.enc_p.encoder.ffn_layers."+n+".conv_1",{768,192,3});
      conv("model.enc_p.encoder.ffn_layers."+n+".conv_2",{192,768,3});
    }
    conv("model.enc_p.proj",{384,192,1});
    conv("model.enc_q.pre",{192,1025,1});wn("model.enc_q.enc",16);conv("model.enc_q.proj",{384,192,1});
    for(int i=0;i<4;++i) {
      const auto p="model.flow.flows."+std::to_string(i*2);
      conv(p+".pre",{192,96,1});wn(p+".enc",3);conv(p+".post",{96,192,1});
    }
    conv("model.dec.conv_pre",{512,192,7});conv("model.dec.cond",{512,256,1});
    conv("model.dec.conv_post",{1,32,7},false,false);conv("model.dec.m_source.l_linear",{1,1});
    const int up[]={16,16,4,4},noise[]={80,8,4,1},kernel[]={3,7,11};
    for(int i=0;i<4;++i) {
      const int ci=512>>i,co=ci/2;
      conv("model.dec.ups."+std::to_string(i),{ci,co,up[i]},true,true,true);
      conv("model.dec.noise_convs."+std::to_string(i),{co,1,noise[i]});
      for(int b=0;b<3;++b)for(int side=1;side<=2;++side)for(int j=0;j<3;++j)
        conv("model.dec.resblocks."+std::to_string(i*3+b)+".convs"+std::to_string(side)+"."+std::to_string(j),
             {co,co,kernel[b]},true);
    }
  } else {
    const int ci[]={1,16,64,256,1024,1024},co[]={16,64,256,1024,1024,1024};
    const int k[]={15,41,41,41,41,5},groups[]={1,4,16,64,256,1};
    for(int i=0;i<6;++i)conv("model.discriminators.0.convs."+std::to_string(i),{co[i],ci[i]/groups[i],k[i]},true);
    conv("model.discriminators.0.conv_post",{1,1024,3},true);
    const int pc[]={1,32,128,512,1024,1024};
    for(int d=1;d<=8;++d) {
      const auto p="model.discriminators."+std::to_string(d);
      for(int i=0;i<5;++i)conv(p+".convs."+std::to_string(i),{pc[i+1],pc[i],5,1},true);
      conv(p+".conv_post",{1,1024,3,1},true);
    }
  }
  return m;
}
} // namespace voxmutatio::training::detail
