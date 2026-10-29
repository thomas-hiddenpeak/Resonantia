// Production upload UI, with a minimal DOM and controllable HTTP responses.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = process.argv[2] || path.join(__dirname, '../webui');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
new vm.Script(source);
for (const id of ['trainFileInput', 'convFileInput']) {
    const accept = html.match(new RegExp(`id="${id}"[^>]*accept="([^"]+)"`));
    assert(accept, 'missing accept list');
    for (const ext of ['wav','mp3','m4a','aac','flac','ogg','opus','wma','aiff','aif'])
        assert(accept[1].includes('.'+ext), `${id} missing ${ext}`);
}
function element() {
    return {value:'voice', style:{}, events:{}, disabled:false, textContent:'', children:[],
        classList:{add(){}, remove(){}}, addEventListener(e,f){this.events[e]=f;},
        append(...v){this.children.push(...v);}, appendChild(v){this.children.push(v);},
        querySelector(){return this;}, removeAttribute(){}, load(){}, click(){}};
}
const tick = async () => {for(let i=0;i<30;i++) await Promise.resolve();};
const deferred = () => {let resolve; const promise=new Promise(r=>resolve=r);return {promise,resolve};};
function harness() {
    const elements=new Map(), calls=[], replies=[], alerts=[];
    const $=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id);};
    const cascade=()=>({state:{ready:false,busy:false},enable(){this.state.ready=true;this.state.busy=false;},setBusy(b){this.state.busy=b;}});
    const trainCascade=cascade(), convCascade=cascade();
    const context=vm.createContext({$, trainCascade,convCascade,API:'/api',CONV_SESSION:'__convert__',
        document:{createElement:element}, URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},
        setTimeout:()=>1,clearTimeout(){},setInterval:()=>1,clearInterval(){}, Audio:element,
        FormData:class{values=[];append(...args){this.values.push(args);}},
        apiFetch:async(url,options)=>{calls.push({url,options});assert(replies.length,'unexpected upload');return await replies.shift();},
        resonantiaTasks:{isBusy:()=>false,observe(){},get(){},async submit(url,body){calls.push({url,options:{body}});assert(replies.length,'unexpected task');const r=await replies.shift();if(!r.ok)throw new Error(r.error);return 'audio-fixture';}},
        auditionSnapshot:async()=>true,setConn(){},errText:async r=>r.error,alert:m=>alerts.push(m)});
    const shared=source.slice(source.indexOf('// ---------- Audio uploads ----------'),source.indexOf('// ---------- Training job controls ----------'));
    const convert=source.slice(source.indexOf('// ---------- Convert ----------'),source.indexOf('const bind ='));
    const conversion=source.slice(source.indexOf("let localConversion = null;"),source.indexOf('// initial'));
    vm.runInContext(shared+convert+conversion,context);
    return {$,calls,replies,alerts,trainCascade,convCascade,run:s=>vm.runInContext(s,context)};
}
(async()=>{
    const h=harness();
    for(const ext of ['WAV','MP3','m4a','AAC','flac','ogg','opus','wma','aiff','aif'])
        assert(h.run(`supportedAudio({name:'voice.${ext}'})`));
    assert(!h.run("supportedAudio({name:'voice.mp4'})"));
    h.run("setConvFile({name:'bad.txt',size:10})");
    assert.equal(h.calls.length,0);assert.equal(h.alerts.length,1);
    const pending=deferred();h.replies.push(pending.promise);
    h.run("setConvFile({name:'first.MP3',size:10})");await tick();
    assert.equal(h.$('btnConvert').disabled,true);assert.equal(h.convCascade.state.ready,false);
    const second=deferred();h.replies.push(second.promise);
    h.run("setConvFile({name:'second.flac',size:20})");await tick();
    assert.equal(h.calls.length,1,'uploads must serialize, so older material cannot overwrite newer');
    pending.resolve({ok:true});await tick();
    assert.equal(h.calls.length,2);assert.equal(h.$('btnConvert').disabled,true);
    second.resolve({ok:false,error:'damaged audio'});await tick();
    assert.equal(h.convCascade.state.ready,false);assert.equal(h.$('btnConvert').disabled,true);
    assert.match(h.$('convUploadStatus').textContent,/damaged audio/);
    h.replies.push({ok:true});await h.run('ensureConvMaterial()');
    assert.equal(h.convCascade.state.ready,true);assert.equal(h.$('btnConvert').disabled,false);
    const conversion=deferred();h.replies.push(conversion.promise);
    const converting=h.$('btnConvert').events.click();
    const beforeReplacement=h.calls.length;
    h.run("setConvFile({name:'replace.mp3',size:5})");await tick();
    h.$('btnConvRemove').events.click();
    assert.equal(h.calls.length,beforeReplacement,'conversion must lock source replacement/removal');
    assert.equal(h.run('convFile.name'),'second.flac');
    conversion.resolve({ok:false,error:'conversion error'});await converting;
    assert.equal(h.$('btnConvert').disabled,false,'conversion failure should release input controls');
    assert.equal(h.convCascade.state.busy,false);
    const removed=deferred();h.replies.push(removed.promise);h.run('ensureConvMaterial()');await tick();
    h.$('btnConvRemove').events.click();removed.resolve({ok:true});await tick();
    assert.equal(h.convCascade.state.ready,false);assert.equal(h.$('btnConvert').disabled,true);

    const t=harness(), upload=deferred();t.replies.push(upload.promise);
    t.run("addTrainFiles([{name:'a.mp3',size:1},{name:'b.WMA',size:1}])");await tick();
    assert.equal(t.calls[0].options.body.values.filter(v=>v[0]==='files').length,2);
    assert.equal(t.$('btnTrain').disabled,true);
    upload.resolve({ok:true});await tick();assert.equal(t.$('btnTrain').disabled,false);
    t.replies.push({ok:false,error:'broken'});
    t.run("addTrainFiles([{name:'c.aac',size:1}])");await tick();
    assert.equal(t.trainCascade.state.ready,false);assert.equal(t.$('btnTrain').disabled,true);
    assert.match(t.$('trainUploadStatus').textContent,/broken/);
    assert.equal(t.$('trainRetryUpload').style.display,'');
    t.replies.push({ok:true});await t.$('trainRetryUpload').onclick();
    assert.equal(t.$('btnTrain').disabled,false);
    t.$('voiceName').value='';await t.run('ensureTrainMaterial()');
    assert.equal(t.$('btnTrain').disabled,true);
    console.log('Web formats, serialized uploads, pending/failure/removal and retry states passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
