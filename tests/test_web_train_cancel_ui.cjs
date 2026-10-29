// Submission ownership migrated from legacy polling/cancel to the durable task
// controller. Attempt-bound pause/cancel is exercised by task-center Chromium.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(process.argv[2]||path.join(__dirname,'../webui'),'app.js'),'utf8');
new vm.Script(source);
const begin=source.indexOf('// ---------- Training job controls ----------'),end=source.indexOf('// ---------- Voices ----------',begin);
assert(begin>=0&&end>begin);
const button=source.slice(source.indexOf('let activeTrain = null;'),source.indexOf("$('voiceName').addEventListener"));
let release, reject, confirm=true, busy=false;
const elements=new Map(),calls=[];
const $=id=>{if(!elements.has(id))elements.set(id,{value:'1',checked:false,disabled:false,style:{},events:{},textContent:'',addEventListener(k,f){this.events[k]=f;}});return elements.get(id);};
$('voiceName').value='Voice_Test';$('trainMode').value='joint-full';$('trainEpochs').value='200';$('trainSeg').value='32';
const context=vm.createContext({$,API:'/api',FormData,trainCascade:{state:{ready:true}},
  resonantiaStudio:{render(){},confirmTraining:async()=>confirm},
  resonantiaTasks:{isBusy:()=>busy,submit:async(url,fd,kind)=>{calls.push({url,fd,kind});return new Promise((resolve,fail)=>{release=resolve;reject=fail;});}}
});
vm.runInContext(button+source.slice(begin,end),context);
const click=()=>$('btnTrain').events.click();const tick=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
(async()=>{
  confirm=false;await click();assert.equal(calls.length,0,'confirmation cancelled');
  confirm=true;busy=true;await click();assert.equal(calls.length,0,'known active task prevents submission');busy=false;
  const first=click();await tick();assert.equal(calls.length,1);assert($('btnTrain').disabled);
  await click();assert.equal(calls.length,1,'double click cannot submit twice');
  assert.equal(calls[0].url,'/api/train');assert.equal(calls[0].kind,'training');
  assert.equal(calls[0].fd.get('epochs'),'200');assert.equal(calls[0].fd.get('steps'),null);
  assert.equal(calls[0].fd.get('resume'),'0');release('stable-task');await first;
  assert.match($('trainStage').textContent,/任务中心/);assert.match($('trainStage').textContent,/关闭/);
  assert.equal(vm.runInContext('activeTrain',context),null,'page no longer owns accepted execution');
  $('trainResume').checked=true;$('recVad').checked=true;
  const rejected=click();await tick();assert.equal(calls[1].fd.get('resume'),'1');assert.equal(calls[1].fd.get('vad'),'0');
  reject(new Error('GPU busy'));await rejected;assert.match($('trainStage').textContent,/GPU busy/);
  assert.equal(vm.runInContext('activeTrain',context),null);assert.equal($('btnTrain').disabled,false);
  $('trainMode').value='decoder';$('trainSteps').value='7';
  const legacy=click();await tick();assert.equal(calls[2].fd.get('steps'),'7');assert.equal(calls[2].fd.get('resume'),null);release('legacy-task');await legacy;
  console.log('Training submission contract passed: confirmation, busy/double-click guard, fixed full/legacy fields, durable handoff and rejection.');
})().catch(e=>{console.error(e);process.exitCode=1;});
