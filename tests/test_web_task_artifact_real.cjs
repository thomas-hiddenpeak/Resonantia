// Reuse an isolated, completed real-browser fixture. Verifies result-digest
// publication and protected cleanup using a six-second CPU-only preparation;
// no voice synthesis, training or publication is requested.
const fs=require('node:fs'),path=require('node:path'),net=require('node:net'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process'),{createHash}=require('node:crypto');
const [repo,build,root]=process.argv.slice(2).map(p=>path.resolve(p));
const previous=JSON.parse(fs.readFileSync(path.join(root,'evidence.json'),'utf8'));assert(previous.passed&&previous.realBrowser);
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
let server,exit,base;
async function post(route,fields){const fd=new FormData();for(const [key,value] of Object.entries(fields))fd.append(key,String(value));return fetch(base+route,{method:'POST',body:fd});}
(async()=>{
  const socket=net.createServer();await new Promise(r=>socket.listen(0,'127.0.0.1',r));const port=socket.address().port;await new Promise(r=>socket.close(r));base='http://127.0.0.1:'+port;
  const log=fs.openSync(path.join(root,'artifact-server.log'),'a');server=spawn(path.join(build,'vc_serve'),['--repo',root,'--webroot',path.join(repo,'runs/Web_Task_Lifecycle_20261029/webui-staged'),'--host','127.0.0.1','--port',String(port)],{detached:true,stdio:['ignore',log,log]});fs.closeSync(log);exit=new Promise(r=>server.once('exit',r));
  for(let i=0;i<100;i++){try{if((await fetch(base+'/api/tasks')).ok)break;}catch{}await delay(50);}
  const session=fs.readdirSync(path.join(root,'runs')).find(name=>name.startsWith('convert_'));assert(session);
  const id='artifact-digest-1';assert.equal((await fetch(base+'/api/task?id='+id)).status,404);
  const fields={task_id:id,session,workflow:'dry',output_kind:'vocal'};
  const accepted=await post('/api/tasks/preprocess',fields);assert.equal(accepted.status,202,await accepted.text());
  let task;for(let n=0;n<200;n++){task=await (await fetch(base+'/api/task?id='+id)).json();if(task.state==='succeeded')break;assert(!['paused','failed'].includes(task.state),JSON.stringify(task));await delay(100);}
  assert.equal(task.state,'succeeded');assert(task.actions.cleanup);
  const result=Buffer.from(await (await fetch(base+task.result_url)).arrayBuffer());assert.equal(task.parameters.result_sha256,hash(result));
  const directory=path.join(root,'runs/.web_tasks',id);const source=fs.readFileSync(path.join(directory,'work/source.wav'));
  assert.equal((await post('/api/task/cleanup',{id,attempt:task.attempt})).status,200);
  assert.equal((await post('/api/task/cleanup',{id,attempt:task.attempt})).status,200);
  const cleaned=await (await fetch(base+'/api/task?id='+id)).json();assert.equal(cleaned.parameters.retention,'pruned-v1');assert.equal(cleaned.state,'succeeded');
  assert.equal(hash(Buffer.from(await (await fetch(base+task.result_url)).arrayBuffer())),hash(result));
  assert(source.equals(fs.readFileSync(path.join(directory,'work/source.wav'))));
  assert.equal((await post('/api/tasks/preprocess',fields)).status,409);
  assert.equal(hash(fs.readFileSync(path.join(repo,'runs/Tom_Cybersinger/model.safetensors'))),previous.formalSha256);
  const evidence={passed:true,resultBytes:result.length,resultSha256:hash(result),sourceRetained:true,idempotentCleanup:true,duplicateRejected:true,modelWorkRequested:false};
  fs.writeFileSync(path.join(root,'artifact-evidence.json'),JSON.stringify(evidence,null,2));console.log(JSON.stringify(evidence));
})().catch(e=>{console.error(e.stack);process.exitCode=1;}).finally(async()=>{if(server?.exitCode===null)process.kill(-server.pid,'SIGTERM');if(exit)await exit;});
