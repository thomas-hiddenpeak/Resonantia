// Opt-in native HTTP compatibility gate. Real six-second audio, CPU-only dry
// preprocessing; no model, training, or publication. Requires the two audited old
// vc_serve executables. Usage: node this.cjs REPO NEW_BUILD NEW_ROOT OLD1 OLD2
const fs=require('node:fs'),path=require('node:path'),net=require('node:net'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process'),{createHash}=require('node:crypto');
const [repo,build,root,...old]=process.argv.slice(2).map(p=>path.resolve(p));
const wait=ms=>new Promise(r=>setTimeout(r,ms)),hash=bytes=>createHash('sha256').update(bytes).digest('hex');
let server,exit,base;
async function stop(){if(server?.pid&&server.exitCode===null)process.kill(-server.pid,'SIGTERM');if(exit)await exit;server=null;}
async function start(binary,fixture){assert(fs.existsSync(binary),'missing server fixture: '+binary);const socket=net.createServer();await new Promise(r=>socket.listen(0,'127.0.0.1',r));const port=socket.address().port;await new Promise(r=>socket.close(r));base='http://127.0.0.1:'+port;const log=fs.openSync(path.join(fixture,'server.log'),'a');server=spawn(binary,['--repo',fixture,'--webroot',path.join(repo,'webui'),'--host','127.0.0.1','--port',String(port)],{detached:true,stdio:['ignore',log,log]});fs.closeSync(log);let startError;exit=new Promise(r=>{server.once('exit',r);server.once('error',e=>{startError=e;r();});});for(let n=0;n<100;n++){if(startError)throw startError;try{if((await fetch(base+'/api/tasks')).ok)return;}catch{}await wait(50);}throw Error('server startup failed');}
async function submit(id,fields){const form=new FormData();for(const [k,v]of Object.entries({task_id:id,...fields}))form.append(k,String(v));const response=await fetch(base+'/api/tasks/preprocess',{method:'POST',body:form});assert.equal(response.status,202,await response.text());for(let n=0;n<100;n++){const r=await(await fetch(base+'/api/task?id='+id)).json();if(r.state==='succeeded')return r;assert(!['failed','paused'].includes(r.state),JSON.stringify(r));await wait(50);}throw Error('CPU preprocessing timed out');}
(async()=>{
 assert(old.length===2&&!fs.existsSync(root));for(const binary of [...old,path.join(build,'vc_serve')])assert(fs.existsSync(binary),'missing server fixture: '+binary);fs.mkdirSync(root,{recursive:true});
 const source=path.join(root,'source.wav');const ff=spawn('/usr/bin/ffmpeg',['-nostdin','-v','error','-i',path.join(repo,'tests/fixtures/speech_02.wav'),'-t','6','-ar','40000','-ac','1','-c:a','pcm_s16le',source],{stdio:'inherit'});assert.equal(await new Promise(r=>ff.once('exit',r)),0);
 const changed=path.join(root,'updated-web');fs.copyFileSync(path.join(build,'vc_serve'),changed);fs.appendFileSync(changed,'X');fs.chmodSync(changed,0o755);
 const results=[];
 for(const [version,binary]of old.entries()){
  const fixture=path.join(root,'version-'+version);fs.mkdirSync(path.join(fixture,'runs/material/raw'),{recursive:true});fs.symlinkSync(path.join(repo,'models'),path.join(fixture,'models'),'dir');fs.symlinkSync(build,path.join(fixture,'build'),'dir');fs.copyFileSync(source,path.join(fixture,'runs/material/raw/source.wav'));
  await start(binary,fixture);const parent=await submit('legacy-parent',{session:'material',workflow:'dry',output_kind:'vocal'});const original=JSON.stringify(parent);
  const work=id=>path.join(fixture,'runs/.web_tasks',id,'work');const marker=fs.readFileSync(path.join(work(parent.id),'workflow.identity'),'utf8');
  await stop();await start(path.join(build,'vc_serve'),fixture);
  const revised=await submit('stable-revision',{source_task:parent.id});assert.equal(revised.parameters.reused_stages,'cleanup');assert.equal(JSON.stringify(await(await fetch(base+'/api/task?id='+parent.id)).json()),original.replace('"cleanup":true}', '"cleanup":true,"revise":true}'));
  assert.equal(fs.readFileSync(path.join(work(parent.id),'workflow.identity'),'utf8'),marker);assert.equal(hash(fs.readFileSync(path.join(work(revised.id),'result.wav'))),hash(fs.readFileSync(source)));
  await stop();await start(changed,fixture);const next=await submit('next-web-revision',{source_task:revised.id});assert.equal(next.parameters.reused_stages,'cleanup');assert.equal(hash(fs.readFileSync(path.join(work(next.id),'result.wav'))),hash(fs.readFileSync(source)));
  results.push({legacySha256:hash(fs.readFileSync(binary)),firstUpgrade:true,subsequentWebOnlyUpgrade:true,parentMarkerUnchanged:true});await stop();
 }
 const evidence={passed:true,realNativeHttp:true,seconds:6,modelExecutions:0,versions:results};fs.writeFileSync(path.join(root,'evidence.json'),JSON.stringify(evidence,null,2));console.log(JSON.stringify(evidence));
})().catch(e=>{console.error(e.stack);process.exitCode=1;}).finally(stop);
