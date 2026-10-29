// Production profile controls against a minimal DOM; no browser/server/model.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(process.argv[2] || path.join(__dirname,'../webui'),'app.js'),'utf8');
new vm.Script(source);
const start=source.indexOf('// ---------- Training profiles ----------');
const end=source.indexOf('// ---------- Presets',start);
assert(start>=0 && end>start,'joint profile UI controls missing');
const elements=new Map();
const $=id=>{
    if(!elements.has(id))elements.set(id,{value:'',hidden:false,disabled:false,max:'',events:{},
        addEventListener(name,fn){this.events[name]=fn;}});
    return elements.get(id);
};
const context=vm.createContext({$});vm.runInContext(source.slice(start,end),context);
const run=text=>vm.runInContext(text,context);
$('trainMode').value='gan';$('trainSteps').value='1200';$('trainSeg').value='32';$('trainEpochs').value='7';
run('configureTrainingProfiles(undefined)');
assert.equal($('trainJointOption').hidden,true);assert.equal($('trainJointOption').disabled,true);
assert.equal($('trainMode').value,'gan');assert.equal($('trainSteps').value,'1200');
run('configureTrainingProfiles(["decoder","gan","joint"])');
assert.equal($('trainJointOption').hidden,false);assert.equal($('trainJointOption').disabled,false);
assert.equal($('trainSteps').value,'1200','capability discovery must preserve user settings');
$('trainMode').value='joint';$('trainMode').events.change();
assert.equal(String($('trainSteps').value),'50');assert.equal(String($('trainSeg').value),'32');
assert.equal(String($('trainEpochs').value),'0');assert.equal(String($('trainSteps').max),'50');
assert.equal($('trainSeg').disabled,true);assert.equal($('trainEpochs').disabled,true);
assert.equal($('trainJointHint').hidden,false);
$('trainSteps').value='12';run('configureTrainingProfiles(["joint"])');
assert.equal($('trainSteps').value,'12','refresh must not restart the step budget');
run('configureTrainingProfiles(undefined)');
assert.equal($('trainMode').value,'gan');assert.equal($('trainSeg').disabled,false);
assert.equal($('trainEpochs').disabled,false);assert.equal(String($('trainSteps').max),'100000');
assert.equal($('trainJointHint').hidden,true);
run('configureTrainingProfiles(["joint", "joint-full"])');
assert.equal($('trainFullOption').hidden,false);
$('trainMode').value='joint-full';$('trainMode').events.change();
assert.equal(String($('trainEpochs').value),'200');assert.equal(String($('trainSeg').value),'32');
assert.equal($('trainSteps').disabled,true);assert.equal($('trainSeg').disabled,true);
assert.equal($('trainEpochs').disabled,false);assert.equal($('trainFullControls').hidden,false);
$('trainEpochs').value='100';run('configureTrainingProfiles(["joint-full"])');
assert.equal($('trainEpochs').value,'100','capability refresh must preserve epoch selection');
run('configureTrainingProfiles(["joint"])');
assert.equal($('trainMode').value,'gan');assert.equal($('trainFullControls').hidden,true);
assert.equal($('trainSteps').disabled,false);
assert.equal(String($('trainEpochs').value),'0','leaving full mode must not carry its long epoch budget into legacy training');
elements.set('trainJointOption',null);elements.set('trainJointHint',null);
elements.set('trainFullOption',null);elements.set('trainFullControls',null);elements.set('trainResume',null);
run('configureTrainingProfiles([]); applyTrainingMode(false)'); // Older cached HTML.
console.log('Joint UI passed: capability discovery, old-server fallback, bounded defaults, refresh and switching; mock DOM only.');

// Selecting a released voice restores its reviewed preset, while refreshing
// the voice list preserves subsequent user edits. Presets may load last.
const voiceStart=source.indexOf('// ---------- Voice presets ----------');
const voiceEnd=source.indexOf('const CONV_SESSION',voiceStart);
assert(voiceStart>=0 && voiceEnd>voiceStart,'voice preset selection missing');
const presetData=JSON.parse(fs.readFileSync(path.join(process.argv[2] || path.join(__dirname,'../webui'),'presets.json'),'utf8'));
assert.deepEqual(presetData.inference.filter(p=>p.voice).map(p=>p.voice),['Tom_Cybersinger'],
    'only the current formal voice should have a voice-specific preset');
const voiceElements=new Map();
const voiceGet=id=>{if(!voiceElements.has(id))voiceElements.set(id,{value:'',events:{},addEventListener(e,f){this.events[e]=f;}});return voiceElements.get(id);};
const applied=[];
const voiceContext=vm.createContext({$:voiceGet,PRESETS:{inference:[]},applyPreset:id=>applied.push(id)});
vm.runInContext(source.slice(voiceStart,voiceEnd),voiceContext);
const voiceRun=s=>vm.runInContext(s,voiceContext);
voiceGet('voiceSelect').value='Tom_Cybersinger';voiceRun('applyVoicePreset()');assert.equal(applied.length,0);
voiceContext.PRESETS=presetData;voiceRun('applyVoicePreset()');
assert.equal(applied.at(-1),'tom_cybersinger');assert.equal(voiceGet('convPreset').value,'tom_cybersinger');
const selected=presetData.inference.find(p=>p.id==='tom_cybersinger');
assert(selected);assert.equal(selected.index_rate,.35);assert.equal(selected.rms_mix_rate,0);
assert.equal(selected.protect,.5);assert.equal(selected.filter_radius,3);assert.equal(selected.f0_up_key,0);
voiceGet('convPreset').value='safe';const count=applied.length;voiceRun('applyVoicePreset()');
assert.equal(applied.length,count,'refresh must preserve manual edits');assert.equal(voiceGet('convPreset').value,'safe');
voiceGet('voiceSelect').value='old_voice';voiceGet('voiceSelect').events.change();assert.equal(applied.length,count);
voiceGet('voiceSelect').value='Tom_Cybersinger';voiceGet('voiceSelect').events.change();assert.equal(applied.length,count+1);
const presetsBlock=source.slice(source.indexOf('async function loadPresets()'),source.indexOf('function applyTier'));
const voicesBlock=source.slice(source.indexOf('async function loadVoices()'),source.indexOf("$('btnRefreshVoices')"));
assert(presetsBlock.includes('applyVoicePreset()'),'late preset response must apply voice preset');
assert(voicesBlock.includes('applyVoicePreset()'),'late voices response must apply voice preset');
console.log('Voice presets passed: parameters, asynchronous discovery, voice changes and preserving manual edits.');

// Presets and server capability discovery may finish in either order.
const tierBegin=source.indexOf('function applyTier('),tierEnd=source.indexOf('function applyPreset(',tierBegin);
vm.runInContext('let PRESETS={training:[{id:"full",mode:"joint-full",steps:50,seg:32,epochs:200,desc:"full"}]};'+source.slice(tierBegin,tierEnd),context);
run('configureTrainingProfiles(["decoder","gan"]);applyTier("full")');
assert.equal($('trainMode').value,'gan');assert.equal(String($('trainEpochs').value),'0');
assert.match($('trainTierDesc').textContent,/不支持/);
run('knownTrainingProfiles=null;applyTier("full");configureTrainingProfiles(["decoder","gan"])');
assert.equal($('trainMode').value,'gan');assert.match($('trainTierDesc').textContent,/不支持/);
run('configureTrainingProfiles(["decoder","gan","joint","joint-full"]);applyTier("full")');
assert.equal($('trainMode').value,'joint-full');assert.equal(String($('trainEpochs').value),'200');
console.log('Full-training presets respect capabilities in either discovery order.');
