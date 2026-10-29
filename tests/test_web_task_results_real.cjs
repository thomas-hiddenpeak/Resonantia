// Real native HTTP/browser acceptance on an isolated copy of an already-completed
// song. No conversion, training, or mutation of the source task is performed.
'use strict';
const fs=require('node:fs'),path=require('node:path'),net=require('node:net'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process'),{createHash}=require('node:crypto');
const {chromium}=require(path.join(process.env.NODE_PATH||path.join(__dirname,'browser/node_modules'),'playwright-core'));
const [repoArg,buildArg,rootArg,id]=process.argv.slice(2),repo=path.resolve(repoArg),build=path.resolve(buildArg),root=path.resolve(rootArg);
assert(/^[A-Za-z0-9_-]{8,96}$/.test(id));assert(!fs.existsSync(root),'new isolated root required');
const source=path.join(repo,'runs/.web_tasks',id),target=path.join(root,'runs/.web_tasks',id);
const sha=bytes=>createHash('sha256').update(bytes).digest('hex'),hash=p=>sha(fs.readFileSync(p));
const delay=ms=>new Promise(r=>setTimeout(r,ms));let server,exit,base,browser,token='';
const before={record:hash(path.join(source,'record')),result:hash(path.join(source,'work/result.wav'))};
const marker=fs.readFileSync(path.join(source,'work/stage.conversion.done'),'utf8').trim().split('\n');
assert.equal(marker[0],'RESONANTIA_STAGE 1');const relative=JSON.parse(marker[3]);
assert(!path.isAbsolute(relative)&&relative.split('/').every(p=>p&&p!=='.'&&p!=='..'));
before.vocal=hash(path.join(source,'work',relative));assert.equal(before.vocal,marker[4]);assert.notEqual(before.result,before.vocal);
fs.mkdirSync(target,{recursive:true});
for(const name of ['record','work/source.wav','work/result.wav','work/workflow.identity','work/stage.conversion.done','work/'+relative]) {
  const from=path.join(source,name),to=path.join(target,name);assert(fs.lstatSync(from).isFile());
  fs.mkdirSync(path.dirname(to),{recursive:true});fs.copyFileSync(from,to);
}
async function start(auth='') {
  token=auth;const socket=net.createServer();await new Promise(r=>socket.listen(0,'127.0.0.1',r));const port=socket.address().port;await new Promise(r=>socket.close(r));base='http://127.0.0.1:'+port;
  const log=fs.openSync(path.join(root,'server.log'),'a');server=spawn(path.join(build,'vc_serve'),['--repo',root,'--webroot',path.join(repo,'webui'),'--host','127.0.0.1','--port',String(port),...(auth?['--auth-token',auth]:[])],{detached:true,stdio:['ignore',log,log]});fs.closeSync(log);
  exit=new Promise((resolve,reject)=>{server.once('error',reject);server.once('exit',resolve);});
  for(let n=0;n<200;n++){try{if((await api('/api/tasks')).ok)return;}catch{}assert(server.exitCode===null,'server exited');await delay(50);}throw Error('server did not start');
}
async function stop(){if(server?.exitCode===null)process.kill(-server.pid,'SIGTERM');if(exit)await exit;server=null;}
async function api(route,options={}){return fetch(base+route,{...options,headers:{...(token?{Authorization:'Bearer '+token}:{}),...options.headers}});}
async function post(route,fields){const body=new FormData();for(const [k,v]of Object.entries(fields))body.append(k,String(v));return api(route,{method:'POST',body});}
async function task(){const r=await api('/api/task?id='+id);assert.equal(r.status,200);return r.json();}
async function checkResults() {
  const r=await task();assert.equal(r.state,'succeeded');assert.deepEqual(r.outputs.map(o=>o.kind),['mix','vocal']);
  for(const kind of ['mix','vocal']) {
    const response=await api('/api/task/result?id='+id+'&variant='+kind);assert.equal(response.status,200);
    assert.equal(sha(Buffer.from(await response.arrayBuffer())),kind==='mix'?before.result:before.vocal);
    const range=await api('/api/task/result?id='+id+'&variant='+kind,{headers:{Range:'bytes=0-43'}});
    assert.equal(range.status,206);assert.equal((await range.arrayBuffer()).byteLength,44);
  }
  return r;
}
(async()=>{
  await start();let r=await checkResults();const originalName=r.name;
  browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH,headless:true,args:['--no-sandbox']});
  const context=await browser.newContext({viewport:{width:1280,height:1000}}),page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base);await page.evaluate(id=>resonantiaTasks.open(id),id);
  await page.locator('#taskVocalCard').waitFor({state:'visible'});
  await page.waitForFunction(()=>document.getElementById('taskAudio').readyState>=1&&document.getElementById('taskVocalAudio').readyState>=1);
  assert((await page.locator('#taskDownload').getAttribute('href')).includes('variant=mix'));
  assert((await page.locator('#taskVocalDownload').getAttribute('href')).includes('variant=vocal'));
  await page.locator('#taskRename').click();await page.locator('#taskRenameInput').fill('听审版 · 降八度');await page.locator('#taskRenameSave').click();
  await page.locator('#taskName').filter({hasText:'听审版 · 降八度'}).waitFor();
  await page.reload();await page.locator('#nav-tasks').click();await page.locator('#taskName').filter({hasText:'听审版 · 降八度'}).waitFor();
  await page.waitForFunction(()=>document.getElementById('taskAudio').duration>260&&document.getElementById('taskVocalAudio').duration>260);
  await page.screenshot({path:path.join(root,'desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.screenshot({path:path.join(root,'mobile.png'),fullPage:true});assert.deepEqual(errors,[]);
  await browser.close();browser=null;await stop();await start('fixture-results-token');
  assert.equal((await fetch(base+'/api/task?id='+id)).status,401);
  assert.equal((await fetch(base+'/api/task/result?id='+id+'&variant=vocal')).status,401);
  assert.equal((await fetch(base+'/api/task/rename',{method:'POST',body:''})).status,401);
  r=await checkResults();assert.equal(r.display_name,'听审版 · 降八度');assert.equal(r.name,originalName);
  assert.equal((await post('/api/task/rename',{id,name:'不应覆盖',expected_name:originalName})).status,409);
  assert.equal((await post('/api/task/cleanup',{id,attempt:r.attempt})).status,200);await checkResults();
  assert.equal(hash(path.join(source,'record')),before.record);assert.equal(hash(path.join(source,'work/result.wav')),before.result);assert.equal(hash(path.join(source,'work',relative)),before.vocal);
  const evidence={passed:true,realNativeHttp:true,realBrowser:true,sourceTask:id,renameSurvivesRestart:true,dualOutputsByteIdentical:true,bothRetainedAfterCleanup:true,authenticatedRange:true,sourceTaskUnchanged:true,modelExecutions:0};
  fs.writeFileSync(path.join(root,'evidence.json'),JSON.stringify(evidence,null,2));console.log(JSON.stringify(evidence));
})().catch(e=>{console.error(e.stack);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();await stop();});
