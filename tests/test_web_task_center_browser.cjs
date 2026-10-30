// Browser lifecycle contract with server-owned fixtures; no model quality claim.
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const assert = require('node:assert/strict');
const dependencies=process.env.NODE_PATH || path.join(__dirname,'browser/node_modules');
const { chromium } = require(path.join(dependencies,'playwright-core'));
const ui = path.resolve(process.argv[2] || 'webui');
const output = path.resolve(process.argv[3] || 'build/task-browser');
fs.mkdirSync(output, { recursive: true });
const now = Date.now();
const records = new Map();
const record = (id, kind, state) => ({id, kind, name:kind === 'training' ? 'Training_fixture' : '单依纯 <img src=x onerror=alert(1)>.mp3',
  state, stage:kind === 'training' ? 'training' : 'converting', phase:'synthesis', attempt:1,
  created_ms:now, started_ms:now, updated_ms:now, finished_ms:0, progress_at_ms:now,
  completed:2, total:7, unit:'chunks', parameters:{recovery:kind === 'training' ? 'training-full-v1' : 'audio-v1'},
  actions:{pause:state === 'running', resume:state === 'paused'}, error:'', result_url:''});
records.set('browser-fixture-1', record('browser-fixture-1','convert','running'));
let submissions = 0, controls = [], browser, context, page, token = '', audioAuthorized = 0, dropBeforeAcceptance=false;
const submittedIds=[],submittedBodies=[];
const pcm = Buffer.alloc(32044); pcm.write('RIFF'); pcm.writeUInt32LE(pcm.length-8,4); pcm.write('WAVEfmt ',8);
pcm.writeUInt32LE(16,16);pcm.writeUInt16LE(1,20);pcm.writeUInt16LE(1,22);pcm.writeUInt32LE(16000,24);
pcm.writeUInt32LE(32000,28);pcm.writeUInt16LE(2,32);pcm.writeUInt16LE(16,34);pcm.write('data',36);pcm.writeUInt32LE(32000,40);
const types = {'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json'};
const server = http.createServer(async (req,res) => {
  const u = new URL(req.url, 'http://localhost');
  assert.ok(!req.url.includes('test-token'),'credential leaked to URL');
  if(u.pathname.startsWith('/api/') && token && req.headers.authorization !== 'Bearer '+token) return res.writeHead(401).end();
  const json = value => res.writeHead(200, {'Content-Type':'application/json'}).end(JSON.stringify(value));
  if (u.pathname === '/api/storage') return json({managed_bytes:1048576,available_bytes:107374182400,managed_limit_bytes:68719476736,task_limit_bytes:25769803776,reserve_bytes:8589934592,history_tasks:records.size,retained_tasks:records.size,retained_limit:256,history_limit:4096,reason:''});
  if (u.pathname === '/api/tasks') return json({tasks:[...records.values()],active_id:[...records.values()].find(r=>r.state==='running')?.id || ''});
  if (u.pathname === '/api/task') {const r=records.get(u.searchParams.get('id'));return r?json(r):res.writeHead(404).end();}
  if (u.pathname === '/api/task/log') return res.end('Model loaded\nRESONANTIA_PROGRESS 1 synthesis 2 7 chunks\n<script>bad()</script>');
  if (u.pathname === '/api/task/result') {if(token)audioAuthorized++;return res.writeHead(200,{'Content-Type':'audio/wav','Content-Length':pcm.length}).end(pcm);}
  if (u.pathname === '/api/task/rename') {
    let body='';for await(const part of req)body+=part;
    const field=k=>new RegExp('name="'+k+'"\\r\\n\\r\\n([^\\r]*)').exec(body)?.[1];
    const r=records.get(field('id')),name=field('name');
    if((r.display_name||r.name)!==field('expected_name')&&(r.display_name||r.name)!==name)return res.writeHead(409).end('名称已变化');
    r.display_name=name;return json(r);
  }
  if (u.pathname === '/api/tasks/convert') {
    let body=''; for await(const part of req)body+=part;
    const id=/name="task_id"\r\n\r\n([^\r]+)/.exec(body)?.[1]; assert.ok(id);
    submittedIds.push(id);submittedBodies.push(body);
    if(dropBeforeAcceptance)return res.writeHead(503,{'Content-Type':'application/json'}).end('{"error":"temporary unavailable before acceptance"}');
    // Chromium itself may retry a dropped keep-alive connection. The stable ID
    // must still admit exactly one job, just like the native server contract.
    if(records.has(id))return res.writeHead(409,{'Content-Type':'application/json'}).end('{"error":"task already exists"}');
    submissions++; records.set(id,record(id,'convert','running'));
    req.socket.destroy(); return; // Acceptance response lost after server persisted the task.
  }
  if (u.pathname === '/api/task/pause' || u.pathname === '/api/task/resume' || u.pathname === '/api/task/cleanup') {
    let body='';for await(const part of req)body+=part;
    const id=/name="id"\r\n\r\n([^\r]+)/.exec(body)?.[1];
    const attempt=Number(/name="attempt"\r\n\r\n([^\r]+)/.exec(body)?.[1]);
    const r=records.get(id); controls.push({id,attempt,path:u.pathname}); assert.equal(attempt,r.attempt);
    if(u.pathname.endsWith('cleanup')){r.parameters.retention='pruned-v1';r.actions.cleanup=false;return json({task_id:id,freed_bytes:1048576});}
    if(u.pathname.endsWith('pause')) {r.state='paused';r.actions={pause:false,resume:true};}
    else {r.attempt++;r.state='running';r.actions={pause:true,resume:false};}
    return json({task_id:id,attempt:r.attempt});
  }
  if(u.pathname==='/api/task/source')return res.writeHead(200,{'Content-Type':'audio/wav','Content-Length':pcm.length}).end(pcm);
  if(u.pathname==='/api/voices')return json({voices:['Tom_Cybersinger'],training_profiles:['joint-full','joint','gan','decoder']});
  if(u.pathname==='/api/train/runs')return json({runs:[]});
  if(u.pathname==='/api/train/status')return json({running:false});
  const file=u.pathname==='/'?'index.html':u.pathname.slice(1);
  if(!/^[a-z.-]+$/.test(file)||!fs.existsSync(path.join(ui,file)))return res.writeHead(404).end();
  res.writeHead(200,{'Content-Type':types[path.extname(file)]||'text/plain'});fs.createReadStream(path.join(ui,file)).pipe(res);
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base='http://127.0.0.1:'+server.address().port;
  browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH,headless:true,args:['--no-sandbox']});
  context=await browser.newContext();page=await context.newPage();
  const errors=[];context.on('page',p=>p.on('pageerror',e=>errors.push(e.message)));page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base);
  await page.locator('#nav-tasks').click({timeout:5000});
  await page.locator('#taskStorage').filter({hasText:'100.00 GiB'}).waitFor({timeout:5000});
  await page.getByRole('button',{name:/单依纯/}).click();
  await page.locator('#taskCounter').filter({hasText:'2 / 7'}).waitFor();
  assert.equal(await page.locator('#taskDetail img').count(),0);
  await page.locator('#taskRename').click({timeout:5000});
  await page.locator('#taskRenameInput').fill('我的翻唱 <img src=x>');
  await page.locator('#taskRefresh').click();
  assert.equal(await page.locator('#taskRenameInput').inputValue(),'我的翻唱 <img src=x>','polling overwrote rename draft');
  await page.locator('#taskRenameSave').click();
  await page.locator('#taskName').filter({hasText:'我的翻唱 <img src=x>'}).waitFor({timeout:5000});
  assert.equal(await page.locator('#taskName img').count(),0);assert.equal(submissions,0);
  await page.locator('#taskRename').click();await page.locator('#taskRenameInput').fill('尚未保存的草稿');
  records.get('browser-fixture-1').display_name='另一窗口的新名称';
  await page.locator('#taskRenameSave').click();await page.locator('#taskRenameMessage').filter({hasText:'取消编辑'}).waitFor({timeout:5000});
  assert.equal(await page.locator('#taskRenameInput').inputValue(),'尚未保存的草稿');assert.equal(records.get('browser-fixture-1').display_name,'另一窗口的新名称');
  await page.locator('#taskRenameCancel').click();await page.locator('#taskName').filter({hasText:'另一窗口的新名称'}).waitFor();
  records.get('browser-fixture-1').completed=4;
  await page.locator('#taskCounter').filter({hasText:'4 / 7'}).waitFor();
  await page.locator('#taskPause').click();
  await page.locator('#taskResume').waitFor({state:'visible'});
  await page.locator('#taskResume').click();
  await page.locator('#taskStateLabel').filter({hasText:'运行中'}).waitFor();
  assert.deepEqual(controls.map(x=>x.attempt),[1,1]);
  await context.setOffline(true);
  await page.locator('#taskConnection').filter({hasText:'连接中断'}).waitFor();
  assert.equal(await page.locator('#taskCounter').textContent(),'4 / 7 块');
  await context.setOffline(false); await page.locator('#taskRefresh').click();
  await page.close();
  Object.assign(records.get('browser-fixture-1'),{state:'succeeded',stage:'done',finished_ms:Date.now(),actions:{pause:false,resume:false,cleanup:true},result_url:'/api/task/result?id=browser-fixture-1'});
  records.get('browser-fixture-1').outputs=[{kind:'mix',label:'完整翻唱'},{kind:'vocal',label:'干净人声'}];
  page=await context.newPage();await page.goto(base);await page.locator('#nav-tasks').click();
  await page.locator('#taskStateLabel').filter({hasText:'已完成'}).waitFor();
  await page.locator('#taskVocalCard').waitFor({state:'visible',timeout:5000});
  await page.waitForFunction(()=>document.getElementById('taskAudio').readyState>=1&&document.getElementById('taskVocalAudio').readyState>=1);
  assert((await page.locator('#taskDownload').getAttribute('href')).includes('variant=mix'));
  assert((await page.locator('#taskVocalDownload').getAttribute('href')).includes('variant=vocal'));
  assert.equal(await page.locator('#taskListen').isVisible(),false,'new results should appear without a load-all step');
  await page.locator('#taskCleanup').click();await page.locator('#taskRetention').filter({hasText:'已清理'}).waitFor();
  assert.equal(records.get('browser-fixture-1').state,'succeeded');assert.ok(records.get('browser-fixture-1').result_url);
  assert.equal(submissions,0);
  await page.evaluate(async()=>{const fd=new FormData();fd.append('session','fixture');fd.append('f0_up_key','0');await globalThis.resonantiaTasks.submit('/api/tasks/convert',fd,'convert');});
  await page.locator('#taskStateLabel').filter({hasText:'运行中'}).waitFor();assert.equal(submissions,1);
  await page.reload();await page.locator('#nav-tasks').click();await page.locator('#taskStateLabel').filter({hasText:'运行中'}).waitFor();assert.equal(submissions,1);
  const training=record('training-fixture-1','training','running');records.set(training.id,training);
  await page.evaluate(id=>globalThis.resonantiaTasks.open(id),training.id);
  await page.locator('#taskName').filter({hasText:'Training_fixture'}).waitFor();await page.close();
  Object.assign(training,{state:'failed',error:'索引构建失败；已保存训练结果',actions:{pause:false,resume:true},finished_ms:Date.now()});
  page=await context.newPage();await page.goto(base);await page.locator('#nav-tasks').click();
  await page.locator('#taskError').filter({hasText:'索引构建失败'}).waitFor();
  await page.locator('#taskResume').waitFor({state:'visible'});
  await page.locator('#taskLogDetails summary').click();await page.locator('#taskLog').filter({hasText:'Model loaded'}).waitFor();
  assert.equal(await page.locator('#taskLog script').count(),0);
  for(const r of records.values())if(r.state==='running'){r.state='succeeded';r.actions={};}
  dropBeforeAcceptance=true;
  await page.evaluate(async()=>{const fd=new FormData();fd.append('session','preserved-session');await globalThis.resonantiaTasks.submit('/api/tasks/convert',fd,'convert');});
  const pendingId=submittedIds.at(-1);assert(!records.has(pendingId));
  await page.reload();await page.locator('#nav-tasks').click();await page.locator('#taskRetrySubmission').waitFor({state:'visible'});
  dropBeforeAcceptance=false;await page.locator('#taskRetrySubmission').click();
  await page.locator('#taskRetrySubmission').waitFor({state:'hidden'});
  assert(records.has(pendingId));assert.equal(submittedIds.at(-1),pendingId);assert.equal(submissions,2);
  let prompts=0;page.on('dialog',async dialog=>{prompts++;await dialog.accept('test-token-2');});
  token='test-token-2';
  await page.evaluate(()=>globalThis.resonantiaTasks.refresh());
  await page.locator('#taskConnection').filter({hasText:'需要访问令牌'}).waitFor();
  await page.waitForTimeout(2200);assert.equal(prompts,0,'background polling prompted for credentials');
  await page.locator('#taskRefresh').click();await page.locator('#taskConnection').filter({hasText:'已连接'}).waitFor();assert.equal(prompts,1);
  await page.evaluate(()=>globalThis.resonantiaTasks.open('browser-fixture-1'));
  await page.locator('#taskLoadPrimary').waitFor({state:'visible'});await page.locator('#taskLoadPrimary').click();
  await page.locator('#taskLoadVocal').click();
  await page.waitForFunction(()=>document.getElementById('taskAudio').src.startsWith('blob:') && document.getElementById('taskAudio').readyState>=1);
  assert.equal(audioAuthorized,2);
  for(const width of [320,390,768,1440]) {
    await page.setViewportSize({width,height:960});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'overflow at '+width);
    await page.screenshot({path:path.join(output,`tasks-${width}.png`),fullPage:true});
  }
  await page.addScriptTag({path:require.resolve(path.join(dependencies,'axe-core/axe.min.js'))});
  const accessibility=await page.evaluate(async()=>await axe.run(document.getElementById('panel-tasks'),{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa']}}));
  assert.deepEqual(accessibility.violations.map(x=>({id:x.id,nodes:x.nodes.map(n=>n.target)})),[]);
  for(const r of records.values())if(r.state==='running'){r.state='succeeded';r.actions={};}
  const parent=records.get('browser-fixture-1');parent.actions.revise=true;
  parent.parameters={...parent.parameters,voice:'Tom_Cybersinger',workflow:'song',output_kind:'mix',f0_up_key:'-3',index_rate:'0.23',rms_mix_rate:'0.1',protect:'0.4',filter_radius:'5',pitch_edits:'smooth:1:1.12',pp_dereverb:'0'};
  await page.locator('#taskRefresh').click();await page.locator('#taskRevise').click({timeout:5000});
  await page.locator('#revisionNotice').waitFor({state:'visible',timeout:5000});
  assert.equal(await page.locator('#pitchShift').inputValue(),'-3');assert.equal(await page.locator('#indexRate').inputValue(),'23');
  assert.equal(await page.locator('#inputKind').inputValue(),'song');assert.equal(await page.locator('#outputKind').inputValue(),'mix');
  assert.equal(await page.locator('.pitch-edit').count(),1);assert.equal(await page.locator('#inputKind').isDisabled(),true);
  assert(await page.locator('#addPitchEdit').isVisible(),'historical task must expose local pitch editing without generating another result');
  assert(!(await page.locator('#resultSection').isVisible()),'editing must not pretend the historical result was regenerated');
  assert.equal(await page.locator('#convFileInfo img').count(),0);
  await page.locator('#btnConvRemove').click();assert.equal(await page.locator('#inputKind').isDisabled(),false,'removing inherited source unlocks workflow');
  parent.parameters.f0_up_key='-30';parent.parameters.filter_radius='31';parent.parameters.index_rate='0.234567';
  await page.evaluate(id=>resonantiaTasks.open(id),parent.id);await page.locator('#taskRevise').click();await page.locator('#revisionNotice').waitFor({state:'visible'});
  assert.equal(await page.locator('#pitchShift').inputValue(),'-30','saved pitch silently clamped');assert.equal(await page.locator('#filterRadius').inputValue(),'31');assert.equal(Number(await page.locator('#indexRate').inputValue()),23.4567);
  await page.locator('#btnConvRemove').click();parent.parameters.f0_up_key='-3';parent.parameters.filter_radius='5';parent.parameters.index_rate='0.23';
  await page.evaluate(id=>resonantiaTasks.open(id),parent.id);await page.locator('#taskRevise').click();await page.locator('#revisionNotice').waitFor({state:'visible'});

  for(const width of [320,390,1440]){await page.setViewportSize({width,height:960});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));await page.screenshot({path:path.join(output,'revision-'+width+'.png'),fullPage:true});}
  const revisionAccessibility=await page.evaluate(async()=>await axe.run(document.getElementById('panel-convert'),{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa']}}));assert.deepEqual(revisionAccessibility.violations.map(x=>({id:x.id,nodes:x.nodes.map(n=>n.target)})),[]);
  await page.locator('#pitchShift').evaluate(el=>{for(let p=el.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;});await page.locator('#pitchShift').focus();await page.locator('#pitchShift').press('ArrowLeft');assert.equal(await page.locator('#pitchShift').inputValue(),'-4');await page.locator('#btnConvert').click();
  await page.locator('#taskStateLabel').filter({hasText:'运行中'}).waitFor();
  const revisionBody=submittedBodies.at(-1);assert(revisionBody.includes('name="source_task"\r\n\r\nbrowser-fixture-1'));
  assert(!revisionBody.includes('name="session"'));assert(revisionBody.includes('name="f0_up_key"\r\n\r\n-4'));
  assert.notEqual(submittedIds.at(-1),parent.id);assert.equal(parent.state,'succeeded');
  // Pitch editing remains reachable after reopening history, including a failed
  // revision. Opening an editor must be read-only; submitting makes a new task.
  for(const r of records.values())if(r.state==='running'){r.state='succeeded';r.actions={};}
  await page.evaluate(id=>resonantiaTasks.open(id),parent.id);
  await page.locator('#taskPitch').waitFor({state:'visible',timeout:5000});
  const beforePitch=submissions;
  await page.locator('#taskPitch').click();
  await page.waitForFunction(()=>document.activeElement?.id==='pitchTitle');
  assert(await page.locator('#addPitchEdit').isVisible());assert.equal(submissions,beforePitch);
  const failed=record('failed-pitch-fixture','convert','failed');
  failed.error='RESONANTIA_PROGRESS 1 f0 26314 26314 frames100hz\nConversion failed: smooth pitch edit crosses unvoiced audio; narrow the interval';
  failed.parameters={...parent.parameters,source_task:parent.id,f0_up_key:'-5',filter_radius:'3',pitch_edits:'smooth:42:44;smooth:45:46'};
  failed.actions={resume:true};records.set(failed.id,failed);
  await page.evaluate(id=>resonantiaTasks.open(id),failed.id);
  await page.locator('#taskName').waitFor({state:'visible'});
  await page.reload();await page.locator('#nav-tasks').click();
  await page.locator('#taskPitch').filter({hasText:'修改局部修音'}).waitFor({timeout:5000});
  assert(!(await page.locator('#taskError').textContent()).includes('RESONANTIA_PROGRESS'));
  assert((await page.locator('#taskPitchHint').textContent()).includes('相同设置'));
  assert(await page.locator('#taskResume').isDisabled(),'unchanged invalid smoothing must not encourage another doomed retry');
  // Missing ancestor is explicit and cannot silently fall back to an unrelated input.
  records.delete(parent.id);await page.locator('#taskPitch').click();
  await page.locator('#taskActionMessage').filter({hasText:'无法读取原成功版本'}).waitFor({timeout:5000});
  assert.equal(submissions,beforePitch);records.set(parent.id,parent);
  await page.locator('#taskPitch').click();
  await page.waitForFunction(()=>document.activeElement?.id==='pitchTitle');
  assert((await page.locator('#revisionDescription').textContent()).includes('失败版本'));
  assert.equal(await page.locator('#pitchShift').inputValue(),'-5','must preserve failed revision settings, not overwrite with ancestor settings');
  assert.equal(await page.locator('#filterRadius').inputValue(),'3');
  assert.equal(await page.locator('.pitch-edit').count(),2);
  assert.equal(await page.locator('.pitch-edit').first().locator('[data-role="start"]').inputValue(),'42');
  assert.equal(submissions,beforePitch);assert.equal(failed.state,'failed');
  for(const [i,start,end] of [[0,'43.36','43.54'],[1,'45.68','45.73']]) {
    const row=page.locator('.pitch-edit').nth(i);
    await row.locator('[data-role="mode"]').selectOption('shift');
    await row.locator('[data-role="start"]').fill(start);await row.locator('[data-role="end"]').fill(end);
    await row.locator('[data-role="shift"]').fill('12');
  }
  for(const width of [320,390,768,1440]) {
    await page.setViewportSize({width,height:960});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'pitch editor overflow at '+width);
    await page.locator('#pitchPanel').scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(output,'pitch-'+width+'.png'),fullPage:true});
    await page.locator('#pitchPanel').screenshot({path:path.join(output,'pitch-editor-'+width+'.png')});
  }
  await page.addScriptTag({path:require.resolve(path.join(dependencies,'axe-core/axe.min.js'))});
  const pitchAccessibility=await page.evaluate(async()=>await axe.run(document.getElementById('panel-convert'),{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa']}}));
  assert.deepEqual(pitchAccessibility.violations.map(x=>({id:x.id,nodes:x.nodes.map(n=>n.target)})),[]);
  await page.locator('#regeneratePitch').click();
  await page.locator('#taskStateLabel').filter({hasText:'运行中'}).waitFor();
  const pitchBody=submittedBodies.at(-1);
  assert(pitchBody.includes('name="source_task"\r\n\r\n'+parent.id));
  assert(pitchBody.includes('name="pitch_edits"\r\n\r\nshift:43.36:43.54:12;shift:45.68:45.73:12'));
  assert(pitchBody.includes('name="f0_up_key"\r\n\r\n-5'));assert(!pitchBody.includes('name="session"'));
  assert.notEqual(submittedIds.at(-1),failed.id);assert.equal(failed.state,'failed');assert.equal(parent.state,'succeeded');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,submissions,controls,closedPageCompletion:true,offlineRecovery:true,pitchHistoryAndFailedRevision:true}));
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(async()=>{await browser?.close();server.closeAllConnections();server.close();});
