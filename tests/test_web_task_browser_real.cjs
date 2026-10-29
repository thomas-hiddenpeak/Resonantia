// Real native service + Chromium + real short speech. No routed/mock API,
// full-song conversion, long training, or voice-quality assertion.
const fs=require('node:fs'),path=require('node:path'),net=require('node:net');
const assert=require('node:assert/strict'),{spawn}=require('node:child_process');
const {createHash}=require('node:crypto'),{chromium}=require('playwright-core');
const [repo,build,ui,root]=process.argv.slice(2,6).map(p=>path.resolve(p));
const continueFixture=process.argv[6]==='--continue';
assert(repo&&build&&ui&&root);assert(continueFixture || !fs.existsSync(root),'fixture directory must be new');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
const hash=p=>createHash('sha256').update(fs.readFileSync(p)).digest('hex');
let server,browser,base,serverExit;
const errors=[];
async function run(program,args,log){const fd=fs.openSync(log,'a'),child=spawn(program,args,{stdio:['ignore',fd,fd]});fs.closeSync(fd);const timer=setTimeout(()=>child.kill('SIGTERM'),60000);try{await new Promise((resolve,reject)=>{child.once('error',reject);child.once('exit',code=>code===0?resolve():reject(new Error(program+' failed; '+log)));});}finally{clearTimeout(timer);}}
async function records(){const r=await fetch(base+'/api/tasks');assert.equal(r.status,200);return (await r.json()).tasks;}
async function waitFor(id,states,timeout=240000){const end=Date.now()+timeout;while(Date.now()<end){const r=(await records()).find(r=>r.id===id);if(r&&states.includes(r.state))return r;if(r?.state==='failed')throw new Error(JSON.stringify(r));await delay(500);}throw new Error('task timeout '+id);}
async function newPage(context){const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(base);return page;}
async function main(){
  if(!continueFixture){fs.mkdirSync(root,{recursive:true});fs.symlinkSync(path.join(repo,'models'),path.join(root,'models'),'dir');fs.symlinkSync(build,path.join(root,'build'),'dir');}
  const formal=path.join(repo,'runs/Tom_Cybersinger/model.safetensors'),before=hash(formal);
  const voice=path.join(root,'runs/Tom_Cybersinger');if(!continueFixture){fs.mkdirSync(voice,{recursive:true});fs.copyFileSync(formal,path.join(voice,'model.safetensors'));}assert.equal(hash(path.join(voice,'model.safetensors')),before);
  const source=path.join(root,'speech6.wav'),train=path.join(root,'training1.wav');
  if(!continueFixture)for(const [file,seconds] of [[source,6],[train,1.28]])await run('/usr/bin/ffmpeg',['-nostdin','-v','error','-i',path.join(repo,'tests/fixtures/speech_02.wav'),'-t',String(seconds),'-ar','40000','-ac','1','-c:a','pcm_s16le',file],path.join(root,'prepare.log'));
  const socket=net.createServer();await new Promise(r=>socket.listen(0,'127.0.0.1',r));const port=socket.address().port;await new Promise(r=>socket.close(r));base='http://127.0.0.1:'+port;
  const fd=fs.openSync(path.join(root,'server.log'),'a');server=spawn(path.join(build,'vc_serve'),['--repo',root,'--webroot',ui,'--host','127.0.0.1','--port',String(port)],{detached:true,stdio:['ignore',fd,fd]});fs.closeSync(fd);serverExit=new Promise(r=>server.once('exit',r));
  for(let i=0;i<100;i++){try{if((await fetch(base+'/api/tasks')).ok)break;}catch{}await delay(100);}
  browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH,headless:true,args:['--no-sandbox']});
  let context=await browser.newContext(),page=await newPage(context);
  let audio;
  if(!continueFixture){
  await page.waitForFunction(()=>document.querySelector('#voiceSelect').value==='Tom_Cybersinger'&&document.querySelector('#convPreset').options.length>0);
  await page.locator('#inputKind').selectOption('dry');
  await page.locator('#convFileInput').setInputFiles(source);await page.waitForFunction(()=>!document.querySelector('#btnConvert').disabled);
  await page.locator('#indexRate').evaluate(el=>{el.value='0';el.dispatchEvent(new Event('input',{bubbles:true}));});
  await page.locator('#btnConvert').click();await page.locator('#nav-tasks').waitFor();
  for(let i=0;i<100;i++){audio=(await records()).find(r=>r.kind==='convert');if(audio)break;await delay(100);}
  assert(audio);assert(['queued','running'].includes(audio.state),'fixture finished before page-close boundary');
  await page.locator('#taskName').filter({hasText:'speech6.wav'}).waitFor();
  await page.screenshot({path:path.join(root,'running-before-close.png'),fullPage:true});
  await context.close(); // Closes all tabs; service and model child must continue.
  audio=await waitFor(audio.id,['succeeded']);
  }else{audio=(await records()).find(r=>r.kind==='convert'&&r.state==='succeeded');assert(audio,'continuation needs the already-completed audio fixture');await context.close();}
  context=await browser.newContext();page=await newPage(context);await page.locator('#nav-tasks').click();
  await page.evaluate(id=>globalThis.resonantiaTasks.open(id),audio.id);
  await page.locator('#taskName').filter({hasText:'speech6.wav'}).waitFor();
  await page.locator('#taskStateLabel').filter({hasText:'已完成'}).waitFor();
  await page.locator('#taskListen').click();await page.waitForFunction(()=>document.querySelector('#taskAudio').readyState>=1);
  assert(Math.abs(await page.locator('#taskAudio').evaluate(el=>el.duration)-6)<0.05);
  await page.locator('#taskAudio').evaluate(el=>el.currentTime=2);await page.waitForFunction(()=>Math.abs(document.querySelector('#taskAudio').currentTime-2)<0.1);
  const range=await fetch(base+audio.result_url,{headers:{Range:'bytes=0-43'}});assert.equal(range.status,206);assert.equal((await range.arrayBuffer()).byteLength,44);
  await page.screenshot({path:path.join(root,'audio-after-reopen.png'),fullPage:true});
  assert.equal((await records()).filter(r=>r.kind==='convert').length,1,'reopening duplicated conversion');
  const trainingName=fs.existsSync(path.join(root,'runs/Browser_Training_Fixture/full_training'))?'Browser_Training_Fixture_2':'Browser_Training_Fixture';
  assert(!fs.existsSync(path.join(root,'runs',trainingName,'full_training')),'training continuation fixture already used');
  await page.locator('#nav-train').click();await page.locator('#voiceName').fill(trainingName);
  await page.locator('#trainFileInput').setInputFiles(train);await page.waitForFunction(()=>!document.querySelector('#btnTrain').disabled);
  await page.locator('#trainEpochs').evaluate(el=>el.closest('details').open=true);
  await page.locator('#recVad').evaluate(el=>el.closest('details').open=true);
  await page.locator('#trainEpochs').fill('1');await page.locator('#recVad').uncheck();
  await page.locator('#btnTrain').click();await page.locator('#confirmTraining').click();
  let training;
  for(let i=0;i<100;i++){training=(await records()).find(r=>r.kind==='training'&&r.name===trainingName);if(training)break;await delay(100);}
  assert(training);assert(['queued','running'].includes(training.state));
  await context.close();training=await waitFor(training.id,['succeeded']);
  context=await browser.newContext();page=await newPage(context);await page.locator('#nav-tasks').click();
  await page.evaluate(id=>globalThis.resonantiaTasks.open(id),training.id);await page.locator('#taskStateLabel').filter({hasText:'已完成'}).waitFor();
  await page.locator('#taskLogDetails summary').click();await page.locator('#taskLog').filter({hasText:'review required'}).waitFor();
  await page.locator('#taskLibrary').click();await page.locator('#trainingRun').selectOption(trainingName);
  assert(await page.locator('#trainingCandidate option').count()>0);
  await page.locator('#resumeRun').click();await page.locator('#taskName').filter({hasText:trainingName}).waitFor();
  const full=path.join(root,'runs',trainingName,'full_training');
  const log=fs.readFileSync(path.join(root,'runs/.web_tasks',training.id,'task.log'),'utf8');
  assert(log.includes('step=1/1'),'fixture must execute exactly one optimizer update');
  assert(fs.statSync(path.join(full,'result/model.safetensors')).size>1000000);
  assert(!fs.existsSync(path.join(root,'runs',trainingName,'model.safetensors')),'candidate was automatically published');
  assert.equal(hash(formal),before);assert.deepEqual(errors,[]);
  await page.screenshot({path:path.join(root,'training-after-reopen.png'),fullPage:true});
  const evidence={passed:true,realBrowser:true,realNativeService:true,conversionSeconds:6,trainingSteps:1,closedContextDuringBothJobs:true,freshContextRestoredBothResults:true,audioRangeAndSeeking:true,autoPublished:false,formalSha256:before};
  fs.writeFileSync(path.join(root,'evidence.json'),JSON.stringify(evidence,null,2));console.log(JSON.stringify(evidence));
}
main().catch(async e=>{
  console.error(e.stack);process.exitCode=1;
  if(base)try{for(const r of await records())if(r.actions.pause){const fd=new FormData();fd.append('id',r.id);fd.append('attempt',String(r.attempt));await fetch(base+'/api/task/pause',{method:'POST',body:fd});await waitFor(r.id,['paused','failed','succeeded'],60000);}}catch(cleanup){console.error('fixture safe-stop failed:',cleanup.message);}
}).finally(async()=>{await browser?.close();if(server?.exitCode===null)process.kill(-server.pid,'SIGTERM');if(serverExit)await serverExit;});
