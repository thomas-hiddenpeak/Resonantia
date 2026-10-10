// Run manually with Node.js; no browser, network, server or model required.
// Evaluate the production training controls with a minimal DOM/HTTP adapter.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../webui/app.js'), 'utf8');
new vm.Script(source); // Syntax check the complete bundled script.
const begin = source.indexOf('// ---------- Training job controls ----------');
const end = source.indexOf('// ---------- Voices ----------', begin);
assert(begin >= 0 && end > begin);
const button = source.slice(source.indexOf('let activeTrain = null;'), source.indexOf("$('voiceName').addEventListener"));
const tick = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };
function deferred() { let resolve; const promise = new Promise(r => { resolve = r; }); return {promise, resolve}; }
function response(data, status=200) { return {ok: status < 400, status, json: async () => data, text: async () => JSON.stringify(data)}; }
function running(id='job-a', extra={}) { return {job_id:id, kind:'train', stage:'training', running:true, done:false, cancellable:true, cancel_requested:false, cancelled:false, error:false, ...extra}; }
function terminal(id='job-a', extra={}) { return running(id, {running:false, done:true, cancellable:false, stage:'done', ...extra}); }
function harness() {
    const elements = new Map(), calls=[], replies=[], timers=[];
    let voices=0;
    const $ = id => {
        if (!elements.has(id)) elements.set(id, {value:id === 'voiceName' ? 'voice' : '1', checked:false, disabled:false, style:{}, textContent:'', events:{}, addEventListener(event, fn) {this.events[event]=fn;}});
        return elements.get(id);
    };
    const context=vm.createContext({$, API:'/api', trainCascade:{state:{ready:true}}, FormData,
        apiFetch:async (url, options) => {calls.push({url, options}); assert(replies.length, 'unexpected fetch '+url); const r=replies.shift(); return typeof r==='function' ? r() : await r;},
        errText:async r => (await r.json()).error,
        setTimeout:fn => {timers.push(fn);}, loadVoices:() => {voices++;}});
    vm.runInContext(button+source.slice(begin,end),context);
    const click=id => $(id).events.click();
    const poll=async data => {replies.push(response(data)); assert(timers.length); timers.shift()(); await tick();};
    const start=async (id='job-a') => {replies.push(response({job_id:id}),response(running(id))); await click('btnTrain'); await tick();};
    return {$,calls,replies,timers,click,poll,start,context,get voices(){return voices;}};
}
(async()=>{
    const h=harness(); await h.start();
    assert.equal(h.$('btnTrain').disabled,true);assert.equal(h.$('btnCancelTrain').disabled,false);
    const ack=deferred();h.replies.push(ack.promise);const cancelling=h.click('btnCancelTrain');
    assert.equal(h.calls.at(-1).options.body.get('job_id'),'job-a');
    assert.equal(h.$('btnCancelTrain').disabled,true);
    await h.poll(running()); // Snapshot taken before the cancel request was accepted.
    assert.match(h.$('trainStage').textContent,/正在请求取消/);
    ack.resolve(response({job_id:'job-a',cancelled:false},202));await cancelling;
    assert.match(h.$('trainStage').textContent,/已请求取消/);assert.equal(h.$('btnTrain').disabled,true);
    await h.poll(running());assert.match(h.$('trainStage').textContent,/已请求取消/);
    vm.runInContext('updateTrainBtn()',h.context);assert.equal(h.$('btnTrain').disabled,true);
    await h.poll(terminal('job-a',{stage:'cancelled',cancelled:true,cancel_requested:true}));
    assert.match(h.$('trainStage').textContent,/已取消/);assert.equal(h.voices,0);assert.equal(h.$('btnTrain').disabled,false);

    const stale=harness();await stale.start();
    const late=deferred();stale.replies.push(late.promise);const cancelLate=stale.click('btnCancelTrain');
    const oldTimer=stale.timers[0];
    await stale.poll(terminal('job-a',{stage:'cancelled',cancelled:true}));await stale.start('job-b');
    const calls=stale.calls.length;oldTimer();await tick();assert.equal(stale.calls.length,calls);
    late.resolve(response({job_id:'job-a',cancelled:true}));await cancelLate;
    assert.equal(stale.$('trainStage').textContent,'training');assert.equal(stale.$('btnCancelTrain').disabled,false);

    const mismatch=harness();await mismatch.start();await mismatch.poll(running('other-job'));
    assert.match(mismatch.$('trainStage').textContent,/作业已变化/);assert.equal(mismatch.$('btnCancelTrain').disabled,true);assert.equal(mismatch.voices,0);
    const failed=harness();await failed.start();await failed.poll(terminal('job-a',{error:true,cancel_requested:true}));
    assert.match(failed.$('trainStage').textContent,/训练失败/);assert.equal(failed.voices,0);
    const done=harness();await done.start();done.replies.push(response({error:'too late'},409));await done.click('btnCancelTrain');
    assert.match(done.$('trainStage').textContent,/取消未确认/);assert.equal(done.$('btnTrain').disabled,true);
    await done.poll(running('job-a',{stage:'publishing',cancellable:false}));assert.equal(done.$('btnCancelTrain').disabled,true);
    await done.poll(terminal());assert.match(done.$('trainStage').textContent,/已生成声线/);assert.equal(done.voices,1);
    const network=harness();await network.start();network.replies.push(()=>{throw new Error('offline');});network.timers.shift()();await tick();
    assert.match(network.$('trainStage').textContent,/无法确认/);assert.equal(network.$('btnTrain').disabled,true);assert.equal(network.$('btnCancelTrain').disabled,true);
    await network.poll(terminal('job-a',{cancelled:true}));assert.equal(network.voices,0);
    console.log('UI controls passed: syntax, exact job ID, pending vs stopped, stale poll/ack, mismatch, error, publication conflict, retry after network loss; mocked DOM/fetch only.');
})().catch(e=>{console.error(e);process.exitCode=1;});
