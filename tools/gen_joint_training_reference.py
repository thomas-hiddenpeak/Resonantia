"""Independent real-audio RVC joint one-step fixture; never updates user models.

Keeps original weight-norm parameters, uses upstream enc_p/enc_q/Flow/decoder/D
and loss functions, and persists gradients and AdamW states for C++ comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file, save_file
from gen_vits_reference import CONFIG, load_bin, save_bin
from infer.module.models import SynthesizerTrnMs768NSFsid, MultiPeriodDiscriminatorV2
from train.mel_processing import spectrogram_torch, spec_to_mel_torch, mel_spectrogram_torch
from train.losses import discriminator_loss, generator_loss, feature_loss, kl_loss


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot(out, prefix, model, optimizer):
    named=dict(model.named_parameters())
    for kind in ('value', 'gradient', 'first', 'second'):
        result={}
        for name,p in named.items():
            if kind=='value': t=p.detach()
            elif kind=='gradient': t=p.grad if p.grad is not None else torch.zeros_like(p)
            else: t=optimizer.state[p]['exp_avg' if kind=='first' else 'exp_avg_sq']
            result['model.'+name]=t.detach().contiguous()
        save_file(result,str(out/(prefix+'_'+kind+'.safetensors')))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--replay-cpp',type=Path,help='Diagnostic only: freeze C++ audio and updated D to isolate backward error; never an acceptance reference')
    ap.add_argument('--resource-fixture',action='store_true',help='Only prepare a real 600-frame clip; no model update')
    args=ap.parse_args();out=args.output
    if out.exists() and any(out.iterdir()): raise FileExistsError('refusing to overwrite existing evidence')
    out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    ref=root/'runs/CyberTom_Pilot_HTK/experiment/content-stages'
    source=root/'runs/CyberTom_Pilot_HTK/experiment/singing-phrases/source_phrase.wav'
    assert digest(source)=='f75033aa168b7027a6adf194b0df34d82a934c1ece6dd7abb1200a67084f5538'
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261025)
    T,start,N,crop=128,200,32,48
    if args.resource_fixture: T,crop=600,0
    samples,sr=sf.read(source,dtype='float32'); assert sr==40000
    audio=torch.from_numpy(samples[start*400:(start+T)*400].copy()).reshape(1,-1)
    phone=torch.from_numpy(load_bin(ref/'phone.bin')[start:start+T].copy()).unsqueeze(0)
    f0=torch.from_numpy(load_bin(ref/'f0.bin')[start:start+T].copy()).reshape(1,-1)
    pitch=torch.from_numpy(load_bin(ref/'pitch.bin')[start:start+T].copy()).long().reshape(1,-1)
    spec=spectrogram_torch(audio,2048,40000,400,2048)
    eps=torch.randn(1,192,T);nsf=torch.randn(1,N*400,1)
    for name,data in [('phone',phone[0]),('pitch',pitch.reshape(-1,1)),('f0',f0.reshape(-1,1)),
                      ('spec',spec[0]),('target',audio.reshape(-1,1)),('eps',eps[0]),('nsf',nsf.reshape(-1,1))]:
        save_bin(out/(name+'.bin'),data.numpy())
    if args.resource_fixture:
        (out/'resource-fixture.json').write_text(json.dumps({'frames':T,'source_start_frame':start,
            'source_sha256':digest(source),'updates':0,'purpose':'Real six-second resource boundary only'},indent=2)+'\n')
        return
    gp=root/'models/pretrained_v2/pretrained_v2/f0G40k.safetensors'
    dp=gp.with_name('f0D40k.safetensors')
    config={**CONFIG,'spk_embed_dim':1}
    g=SynthesizerTrnMs768NSFsid(**config).float().train()
    state={k.removeprefix('model.'):v for k,v in load_file(str(gp)).items()}
    state['emb_g.weight']=state['emb_g.weight'][:1].clone();g.load_state_dict(state,strict=True);del state
    d=MultiPeriodDiscriminatorV2(use_spectral_norm=False).float().train()
    d.load_state_dict({k.removeprefix('model.'):v for k,v in load_file(str(dp)).items()},strict=True)
    go=torch.optim.AdamW(g.parameters(),lr=1e-4,betas=(.8,.99),eps=1e-9,weight_decay=.01,foreach=False)
    do=torch.optim.AdamW(d.parameters(),lr=1e-4,betas=(.8,.99),eps=1e-9,weight_decay=.01,foreach=False)
    def draw(x,*a,**kw):
        if tuple(x.shape)==tuple(eps.shape): return eps.to(x)
        if tuple(x.shape)==tuple(nsf.shape): return nsf.to(x)
        raise ValueError('unexpected random draw '+str(x.shape))
    print('START independent joint G forward',flush=True)
    with patch('torch.randn_like',draw):
        emb=g.emb_g(torch.tensor([0])).unsqueeze(-1)
        mp,lp,mask=g.enc_p(phone,pitch,torch.tensor([T]))
        z,mq,lq,_=g.enc_q(spec,torch.tensor([T]),g=emb)
        zp=g.flow(z,mask,g=emb)
        y=g.dec(z[:,:,crop:crop+N],f0[:,crop:crop+N],g=emb)
    for name,t in [('prior_mean',mp),('prior_logs',lp),('posterior_mean',mq),('posterior_logs',lq),
                   ('posterior_z',z),('flow_z',zp),('audio',y)]:
        save_bin(out/(name+'.bin'),t.detach()[0].numpy())
    if args.replay_cpp:
        cpp=torch.from_numpy(load_bin(args.replay_cpp/'audio.bin').copy()).reshape_as(y)
        y=y+(cpp-y).detach()
    target=audio[:,crop*400:(crop+N)*400].unsqueeze(1)
    target_mel=spec_to_mel_torch(spec,2048,125,40000,0,None)[:,:,crop:crop+N]
    print('START independent D backward/update',flush=True)
    dr,df,fr,ff=d(target,y.detach());ld,_,_=discriminator_loss(dr,df)
    for label,groups in [('real',fr),('fake',ff)]:
        for i,t in enumerate(t for group in groups for t in group):
            save_bin(out/f'disc_{label}_{i}.bin',t.detach().reshape(1,-1).numpy())
    del fr,ff
    do.zero_grad();ld.backward();do.step();snapshot(out,'D',d,do)
    print('START independent G backward/update',flush=True)
    if args.replay_cpp:
        d.load_state_dict({k.removeprefix('model.'):v for k,v in load_file(str(args.replay_cpp/'D_value.safetensors')).items()},strict=True)
    dr,df,fr,ff=d(target,y)
    generated_mel=mel_spectrogram_torch(y.squeeze(1),2048,125,40000,400,2048,0,None)
    lm=torch.nn.functional.l1_loss(target_mel,generated_mel)
    lk=kl_loss(zp,lq,mp,lp,mask);lf=feature_loss(fr,ff);la,_=generator_loss(df)
    lg=45*lm+lk+lf+la
    y.retain_grad()
    go.zero_grad();lg.backward();go.step();snapshot(out,'G',g,go)
    save_bin(out/'audio_gradient.bin',y.grad.detach().reshape(1,-1).numpy())
    result={'frames':T,'source_start_frame':start,'crop_start':crop,'crop_frames':N,
            'losses':{k:float(v.detach()) for k,v in [('d',ld),('g',lg),('mel',lm),('kl',lk),('fm',lf),('adv',la)]},
            'model_sha256':{str(p):digest(p) for p in (gp,dp)},'source_sha256':digest(source),
            'torch':torch.__version__,'threads':2,'device':'cpu','learning_rate':1e-4,
            'betas':[.8,.99],'eps':1e-9,'weight_decay':.01,'speaker':0,'updates':1,
            'limits':'Diagnostic replay, NOT an independent acceptance reference' if args.replay_cpp else 'One real short phrase numerical fixture; no naturalness or timbre acceptance',
            'replay_cpp':str(args.replay_cpp) if args.replay_cpp else None,
            'upstream_sha256':{n:digest(root/'tools/rvc_ref'/n) for n in
                ('infer/module/models.py','infer/module/attentions.py','train/losses.py','train/mel_processing.py')}}
    (out/'reference.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
