// Opt-in browser integration: real HTML/JS and HTTP uploads; conversion is
// intercepted to test UI state without starting GPU work or publishing a model.
// Usage: NODE_PATH=<playwright-core parent> node this.cjs URL CHROME AUDIO OUTDIR
const {chromium}=require('playwright-core');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 const [base,executablePath,audio,out]=process.argv.slice(2);
 assert(base&&executablePath&&audio&&out);fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({executablePath,headless:true,args:['--no-sandbox']});
 try{
  const context=await browser.newContext({viewport:{width:1280,height:1000}});
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  const voices=await (await context.request.get(base+'/api/voices')).json();
  assert.deepEqual(voices.voices,['Tom_Cybersinger']);assert.deepEqual(voices.reviewed_voices,['Tom_Cybersinger']);
  await page.goto(base);await page.waitForFunction(()=>document.querySelector('#voiceSelect').value==='Tom_Cybersinger');
  await page.waitForFunction(()=>document.querySelector('#convPreset').value==='tom_cybersinger');
  assert.equal(await page.locator('#voiceSelect option').count(),1);
  assert.match(await page.locator('#voiceSelect option:checked').textContent(),/^Tom Cybersinger/);
  assert.equal(await page.locator('#formalRelease h2').textContent(),'Tom Cybersinger');
  assert.equal(await page.locator('#indexRate').inputValue(),'35');assert.equal(await page.locator('#rmsMixRate').inputValue(),'0');
  await page.locator('#indexRate').evaluate(el=>{el.closest('details').open=true;el.value='23';el.dispatchEvent(new Event('input',{bubbles:true}));});await page.locator('#btnRefreshVoices').click();await page.waitForTimeout(150);
  assert.equal(await page.locator('#indexRate').inputValue(),'23','refresh must preserve manual settings');
  const second=await context.newPage();await second.goto(base);
  const session1=await page.evaluate(()=>sessionStorage.getItem('resonantiaConvertSession'));
  const session2=await second.evaluate(()=>sessionStorage.getItem('resonantiaConvertSession'));
  assert(session1&&session2&&session1!==session2,'tabs must not share material');await second.close();
  const popupPromise=page.waitForEvent('popup');await page.evaluate(url=>window.open(url),base);
  const duplicate=await popupPromise;await duplicate.waitForLoadState();
  assert.notEqual(await duplicate.evaluate(()=>sessionStorage.getItem('resonantiaConvertSession')),session1,'opener-cloned storage must not share material');await duplicate.close();
  await page.locator('#convFileInput').setInputFiles(audio);await page.waitForFunction(()=>!document.querySelector('#btnConvert').disabled);
  await page.locator('#inputKind').selectOption('harmonies');assert.equal(await page.locator('#outputKind').inputValue(),'vocal');
  await page.locator('#inputKind').selectOption('song');await page.locator('#outputKind').selectOption('mix');
  // Local repairs are offered after the first listening pass.
  await page.route('**/api/convert',route=>route.fulfill({status:200,contentType:'audio/wav',body:fs.readFileSync(audio)}));
  await page.locator('#btnConvert').click();await page.locator('#resultSection').waitFor({state:'visible'});
  await page.unroute('**/api/convert');
  await page.locator('#addPitchEdit').evaluate(el=>el.closest('details').open=true);await page.locator('#addPitchEdit').click();
  await page.locator('[data-role="start"]').fill('1');await page.locator('[data-role="end"]').fill('1.12');
  let fulfill,requestBody='';const pending=new Promise(r=>fulfill=r);let intercepted=false;
  await page.route('**/api/convert',async route=>{intercepted=true;requestBody=route.request().postData();await pending;await route.fulfill({status:200,contentType:'audio/wav',body:fs.readFileSync(audio)});});
  await page.locator('#btnConvert').click();
  await page.waitForFunction(()=>document.querySelector('#convFileInput').disabled);
  for(let i=0;!intercepted&&i<100;i++)await page.waitForTimeout(20);
  assert(intercepted);assert(requestBody.includes('smooth:1:1.12'));assert(requestBody.includes('\r\n\r\nsong\r\n'));
  assert(requestBody.includes('\r\n\r\nTom_Cybersinger\r\n'),'conversion must use the renamed release');
  assert(await page.locator('#btnConvRemove').isDisabled());assert(await page.locator('#voiceSelect').isDisabled());
  fulfill();await page.locator('#resultSection').waitFor({state:'visible'});assert(!(await page.locator('#convFileInput').isDisabled()));
  await page.locator('[data-tab="train"]').click();
  assert.equal(await page.locator('#trainMode').inputValue(),'joint-full');assert.equal(await page.locator('#trainEpochs').inputValue(),'200');
  await page.locator('[data-tab="library"]').click();
  await page.locator('#trainingRun').selectOption('CyberTom_Full_20261026');
  assert.match(await page.locator('#trainingRun option:checked').textContent(),/^Tom Cybersinger/);
  assert.equal(await page.locator('#trainingCandidate option').count(),4);
  await page.locator('#refreshRunLog').evaluate(el=>el.closest('details').open=true);await page.locator('#refreshRunLog').click();
  await page.waitForFunction(()=>document.querySelector('#runLog').textContent.includes('Wrote index:'));
  await page.locator('#publishVoice').click();assert((await page.locator('#publishStatus').textContent()).includes('请先听审'));
  await page.locator('#previewCandidate').click();await page.locator('#candidateNotice').waitFor({state:'visible'});
  assert((await page.locator('#candidateNotice').textContent()).includes('200'));
  await page.locator('#exitCandidate').click();assert(!(await page.locator('#voiceSelect').isDisabled()));
  await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'mobile horizontal overflow');
  await page.locator('#convFileInput').setInputFiles({name:'different-input.wav',mimeType:'audio/wav',buffer:fs.readFileSync(audio)});
  await page.waitForFunction(()=>document.querySelector('.file-name').textContent==='different-input.wav' && !document.querySelector('#btnConvert').disabled);
  assert.equal(await page.locator('.pitch-edit').count(),0,'different input must clear corrections tied to previous timeline');
  assert.deepEqual(errors,[]);fs.writeFileSync(path.join(out,'browser-result.json'),JSON.stringify({passed:true,checks:['formal default','preset refresh','tab isolation','real upload','workflow fields','manual pitch fields','busy lock','result playback','training defaults','candidate discovery','history log','review confirmation','candidate mode','mobile layout','no JS errors']},null,2));
  console.log('Browser workflows passed; real HTTP upload, mocked conversion; no publication/training performed.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
