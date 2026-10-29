// Opt-in real HTTP/model integration. Six seconds, isolated task tree; no training
// or publication. Reuses verified separation/cleanup and compares revised PCM to
// a fresh converter call on the exact retained lead. Re-run with --continue only
// to inspect/reuse the same successful fixture stages after harness failures.
'use strict';
const fs=require('node:fs'),path=require('node:path'),net=require('node:net'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process'),{createHash}=require('node:crypto');
const [repoArg,buildArg,rootArg]=process.argv.slice(2),repo=path.resolve(repoArg),build=path.resolve(buildArg),root=path.resolve(rootArg);
const continuing=process.argv.includes('--continue'),delay=ms=>new Promise(r=>setTimeout(r,ms));
const hash=p=>createHash('sha256').update(fs.readFileSync(p)).digest('hex');
let server,exit,base;
async function run(program,args,log){const fd=fs.openSync(log,'a'),child=spawn(program,args,{stdio:['ignore',fd,fd]});fs.closeSync(fd);await new Promise((resolve,reject)=>{child.once('error',reject);child.once('exit',code=>code===0?resolve():reject(Error(program+' exited '+code)));});}
async function post(route,fields){const form=new FormData();for(const [k,v]of Object.entries(fields))form.append(k,String(v));const r=await fetch(base+route,{method:'POST',body:form});assert.equal(r.status,202,await r.text());}
async function get(id){const r=await fetch(base+'/api/task?id='+id);return r.status===404?null:r.json();}
async function complete(id,fields){let r=await get(id);if(!r)await post('/api/tasks/convert',{task_id:id,...fields});for(let n=0;n<3600;n++){r=await get(id);assert(r);if(r.state==='succeeded')return r;assert(!['failed','paused','interrupted'].includes(r.state),JSON.stringify(r));await delay(100);}throw Error('timed out '+id);}
const work=id=>path.join(root,'runs/.web_tasks',id,'work');
(async()=>{
 assert(continuing||!fs.existsSync(root),'use new isolated root');fs.mkdirSync(root,{recursive:true});
 const formal=path.join(repo,'runs/Tom_Cybersinger/model.safetensors'),formalSha=hash(formal);
 if(!continuing){
  fs.symlinkSync(path.join(repo,'models'),path.join(root,'models'),'dir');fs.symlinkSync(build,path.join(root,'build'),'dir');
  fs.mkdirSync(path.join(root,'runs/Tom_Cybersinger'),{recursive:true});
  for(const name of ['model.safetensors','model.index'])fs.copyFileSync(path.join(repo,'runs/Tom_Cybersinger',name),path.join(root,'runs/Tom_Cybersinger',name));
  const source=path.join(root,'runs/fixture/raw/source.wav');fs.mkdirSync(path.dirname(source),{recursive:true});
  await run('/usr/bin/ffmpeg',['-nostdin','-v','error','-i',path.join(repo,'tests/fixtures/speech_02.wav'),'-t','6','-ar','44100','-ac','2','-c:a','pcm_s16le',source],path.join(root,'prepare.log'));
 }
 const socket=net.createServer();await new Promise(r=>socket.listen(0,'127.0.0.1',r));const port=socket.address().port;await new Promise(r=>socket.close(r));base='http://127.0.0.1:'+port;
 const fd=fs.openSync(path.join(root,'server.log'),'a');server=spawn(path.join(build,'vc_serve'),['--repo',root,'--webroot',path.join(repo,'runs/Web_Task_Revisions_20261029/webui-staged'),'--host','127.0.0.1','--port',String(port)],{detached:true,stdio:['ignore',fd,fd]});fs.closeSync(fd);exit=new Promise(r=>server.once('exit',r));
 for(let i=0;i<100;i++){try{if((await fetch(base+'/api/tasks')).ok)break;}catch{}await delay(50);}
 const original=await complete('real-revision-parent',{session:'fixture',voice:'Tom_Cybersinger',workflow:'song',output_kind:'vocal',index_rate:.35,rms_mix_rate:0,protect:.5,filter_radius:3});
 const originalJson=JSON.stringify(original),originalResult=hash(path.join(work(original.id),'result.wav'));
 const start=Date.now();const revised=await complete('real-revision-pitch',{source_task:original.id,f0_up_key:-2});
 const revisionMs=Date.now()-start;assert.equal(revised.parameters.reused_stages,'separation,cleanup');assert.equal(revised.parameters.f0_up_key,'-2');
 for(const name of ['vocals','cleanup'])assert(!fs.existsSync(path.join(work(revised.id),name)),'repeated '+name);
 const reference=path.join(root,'reference-pitch.wav');
 if(!fs.existsSync(reference))await run(path.join(build,'vc_convert'),['--hubert',path.join(repo,'models/hubert_base/model.safetensors'),'--rmvpe',path.join(repo,'models/rmvpe.safetensors'),'--model',path.join(work(revised.id),'model.safetensors'),'--input',path.join(work(revised.id),'lead40.wav'),'--output',reference,'--version','v2','--speakers','109','--sr','40000','--pitch','-2','--rms-mix','0.000000','--protect','0.500000','--filter-radius','3','--index',path.join(work(revised.id),'model.index'),'--index-rate','0.350000'],path.join(root,'reference.log'));
 assert.equal(hash(reference),hash(path.join(work(revised.id),'result.wav')),'revised PCM differs from direct inference');
 const remixed=await complete('real-revision-mix',{source_task:revised.id,output_kind:'mix'});
 assert.equal(remixed.parameters.reused_stages,'separation,cleanup,conversion');assert(!fs.existsSync(path.join(work(remixed.id),'convert.log')),'remix ran converter');
 assert.equal(JSON.stringify(await get(original.id)),originalJson,'original metadata changed');assert.equal(hash(path.join(work(original.id),'result.wav')),originalResult,'original result changed');
 assert.equal(hash(formal),formalSha,'formal model changed');
 const evidence={passed:true,seconds:6,source:'tests/fixtures/speech_02.wav',actualGpuPipeline:true,pitchRevisionMs:revisionMs,pitchReuse:revised.parameters.reused_stages,outputReuse:remixed.parameters.reused_stages,directOutputByteIdentical:true,parentUnchanged:true,formalSha256:formalSha};
 fs.writeFileSync(path.join(root,'evidence.json'),JSON.stringify(evidence,null,2));console.log(JSON.stringify(evidence));
})().catch(e=>{console.error(e.stack);process.exitCode=1;}).finally(async()=>{if(server?.exitCode===null)process.kill(-server.pid,'SIGTERM');if(exit)await exit;});
