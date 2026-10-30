// Read-only diagnostics and explicit, server-owned local auditions.
(() => {
  const q=id=>document.getElementById(id), audio=q('pitchReviewAudio');
  let generation=0,mediaGeneration=0,baseline=null,selectedRecord=null,report=null,index=0,range=[0,0],offset=0;
  let accepted=[],ignored=[],undo=[],previews=new Map(),previewHints=new Map(),previewErrors=new Map(),lookingUp=new Set(),urls=new Map(),submitting=false,mode='';
  const say=(id,text)=>{q(id).textContent=text;};
  const stamp=t=>`${Math.floor(t/60)}:${(t%60).toFixed(2).padStart(5,'0')}`;
  const validId=id=>/^[A-Za-z0-9_-]{8,96}$/.test(id||'');
  const edit=e=>`shift:${e.start}:${e.end}:${e.semitones}`;
  function parse(text) {return (text||'').split(';').filter(Boolean).map(text=>{const p=text.split(':');return {text,start:Number(p[1]),end:Number(p[2])};});}
  function combine(indices) {
    const additions=indices.map(i=>report.issues[i]?.suggestion).filter(Boolean);
    const rows=parse(report.baseline_edits).filter(e=>!additions.some(a=>a.start<=e.end&&a.end>=e.start));
    rows.push(...additions.map(e=>({text:edit(e),start:e.start,end:e.end})));rows.sort((a,b)=>a.start-b.start);
    if(rows.length>16||rows.some((e,i)=>i&&e.start<=rows[i-1].end))throw new Error('建议区间重叠或超过16处，请分批复核');
    return rows.map(e=>e.text).join(';');
  }
  function storageKey(){return 'resonantiaPitchDraft:'+baseline.id+':'+report.analysis_id;}
  function save(){if(!report)return;try {localStorage.setItem(storageKey(),JSON.stringify({accepted,ignored,selected:index,preview_ids:[...previewHints].slice(0,64)}));}catch{say('pitchDraftStatus','草稿暂未存到浏览器，请保留页面；已提交试听仍在任务中心。');}}
  function reset() {
    ++generation;++mediaGeneration;audio.pause();audio.removeAttribute('src');audio.load();for(const url of urls.values())URL.revokeObjectURL(url);
    urls.clear();baseline=null;report=null;selectedRecord=null;accepted=[];ignored=[];undo=[];index=0;
    previews.clear();previewHints.clear();previewErrors.clear();lookingUp.clear();mode='';
    q('pitchReviewBody').hidden=true;q('pitchReload').disabled=true;say('pitchReviewStatus','从任务中心打开已完成的翻唱，即可检查可疑音高。');
  }
  async function get(url) {const response=await apiFetch(url,{signal:AbortSignal.timeout(20000)});if(!response.ok){const error=new Error(await errText(response));error.status=response.status;throw error;}return response;}
  function validReport(data) {
    const curve=a=>Array.isArray(a)&&a.length>0&&a.length<=61100&&a.every(n=>Number.isFinite(n)&&n>=0&&n<=1280000);
    return validId(data.baseline_id)&&data.baseline_id===baseline.id&&/^[a-f0-9]{64}$/.test(data.analysis_id)&&data.duration>0&&data.duration<=600&&data.step===.01&&Number.isFinite(data.start_time)&&
      curve(data.raw)&&curve(data.processed)&&data.raw.length===data.processed.length&&Array.isArray(data.issues)&&data.issues.length<=64&&data.issues.every(e=>e.start>=0&&e.end>e.start&&e.end<=data.duration&&typeof e.reason==='string'&&(!e.suggestion||(e.suggestion.start>=0&&e.suggestion.end>e.suggestion.start&&e.suggestion.end-e.suggestion.start<=2&&e.suggestion.end<=data.duration&&[12,-12].includes(e.suggestion.semitones))));
  }
  function issueRange(i) {const e=report.issues[i];return [Math.max(0,e.start-2),Math.min(report.duration,e.end+3.5)];}
  function matchingIssue(r) {
    const p=r?.parameters;if(!report||!validId(r.id)||r.kind!=='convert'||p?.recovery!=='audio-pitch-preview-v1'||p.source_task!==baseline.id||p.analysis_id!==report.analysis_id)return -1;
    const start=Number(p.preview_start),end=Number(p.preview_end);
    if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end-start>15||end>report.duration+.000001)return -1;
    return report.issues.findIndex((issue,i)=>issue.suggestion&&p.pitch_edits===combine([i])&&start<=issueRange(i)[0]+.000001&&end>=issueRange(i)[1]-.000001);
  }
  function remember(r,preferred=false) {
    const i=matchingIssue(r);if(i<0)return -1;const old=previews.get(i);
    if(!preferred&&old&&old.id!==r.id&&Number(old.created_ms||0)>=Number(r.created_ms||0))return i;
    if(old?.id===r.id&&old.state===r.state&&old.attempt===r.attempt&&old.revision===r.revision&&old.updated_ms===r.updated_ms)return i;
    previews.set(i,r);previewHints.set(i,r.id);previewErrors.delete(i);save();renderList();controls();return i;
  }
  async function lookupPreview(i,id,token=generation) {
    lookingUp.add(i);controls();
    try {
      const r=await (await get(API+'/task?id='+encodeURIComponent(id))).json();if(token!==generation)return;
      if(r.id!==id||matchingIssue(r)!==i){previewHints.delete(i);previews.delete(i);throw new Error('保存的试听与当前疑点或分析不匹配，未采用。');}
      remember(r,true);
    }catch(e){if(token===generation){if(e.status===404){previewHints.delete(i);previews.delete(i);}previewErrors.set(i,'无法核实试听：'+e.message);}}
    finally{if(token===generation){lookingUp.delete(i);save();renderList();controls();}}
  }
  async function load(source,requested=source,{preview=null}={}) {
    reset();baseline=source;selectedRecord=requested;const token=generation;q('pitchReload').disabled=false;say('pitchReviewStatus','正在读取已保存的音高，不启动转换…');
    try {
      const data=await (await get(API+'/task/pitch?id='+encodeURIComponent(requested.id))).json();if(token!==generation)return;
      if(!validReport(data))throw new Error('音高报告格式或来源不匹配');report=data;
      try {
        const saved=JSON.parse(localStorage.getItem(storageKey())||localStorage.getItem('resonantiaPitchDraft:'+report.analysis_id)||'{}');
        const safe=(a,limit)=>Array.isArray(a)?[...new Set(a.filter(i=>Number.isInteger(i)&&i>=0&&i<data.issues.length))].slice(0,limit):[];
        accepted=safe(saved.accepted,16).filter(i=>data.issues[i].suggestion);undo=accepted.map((_,i)=>accepted.slice(0,i));ignored=safe(saved.ignored,64).filter(i=>!accepted.includes(i));
        index=safe([saved.selected],1)[0]||0;
        if(Array.isArray(saved.preview_ids))for(const pair of saved.preview_ids.slice(0,64))if(Array.isArray(pair)&&safe([pair[0]],1).length&&validId(pair[1]))previewHints.set(pair[0],pair[1]);
        if(accepted.length)resonantiaWorkflow.setPitchEdits(combine(accepted));
      }catch{accepted=[];ignored=[];undo=[];previewHints.clear();}
      q('pitchReviewBody').hidden=false;say('pitchReviewStatus',data.notice+(requested.id!==source.id?' 自动建议以成功版为起点；采用后会替换失败版的局部设置。':''));
      let previewIndex=-1;if(preview)previewIndex=remember(preview,true);
      if(previewIndex>=0)index=previewIndex;
      renderList();if(data.issues.length)select(index);else {say('pitchSelection','未找到明确疑点，可使用高级手动修音');controls();}
      draft();if(preview&&previewIndex<0){previewErrors.set(index,'该试听与当前单项建议不匹配（可能包含组合修正）；原试听仍在任务中心。');controls();}
      // Stored IDs never prove completion. Only the selected hint is fetched;
      // other history entries are checked on selection, not in a request burst.

    }catch(e){if(token===generation){report=null;q('pitchReviewBody').hidden=true;say('pitchReviewStatus','不能自动分析：'+e.message+'。仍可使用高级手动修音；没有启动转换。');}}
  }
  function reviewState(i) {
    if(accepted.includes(i))return ['accepted','已采用'];if(ignored.includes(i))return ['ignored','保留原音'];
    const r=previews.get(i);if(r?.state==='succeeded')return ['ready','试听已完成，待听审'];
    if(r)return ['pending',({queued:'试听等待中',running:'试听生成中',pause_requested:'正在暂停',paused:'试听已暂停',interrupted:'试听已中断',failed:'试听失败',cancelled:'试听已停止'})[r.state]||'查看试听状态'];
    return ['unreviewed',previewHints.has(i)?'待核实试听':report.issues[i].suggestion?'尚未试听建议':'需人工听审'];
  }
  function renderList() {
    if(!report)return;const focused=q('pitchIssues').contains(document.activeElement)?document.activeElement.dataset.issue:null;q('pitchIssues').replaceChildren();report.issues.forEach((issue,i)=>{const b=document.createElement('button'),[state,label]=reviewState(i);b.type='button';b.className='btn-small';b.dataset.issue=String(i);b.dataset.state=state;b.setAttribute('aria-pressed',String(index===i));b.textContent=`${i+1}. ${stamp(issue.start)} · ${issue.reason} · ${label}`;b.onclick=()=>select(i);q('pitchIssues').append(b);});
    if(focused!==null)q('pitchIssues').querySelector(`[data-issue="${focused}"]`)?.focus({preventScroll:true});
    say('pitchReviewSummary',`${report.issues.length} 处疑点 · 已采用 ${accepted.length} 处 · 保留原音 ${ignored.length} 处 · 待听审 ${report.issues.length-accepted.length-ignored.length} 处`);
  }
  function select(i) {
    if(!report?.issues[i])return;audio.pause();++mediaGeneration;mode='';index=i;const issue=report.issues[i];range=issueRange(i);save();
    say('pitchSelection',`第 ${i+1} 处 · ${stamp(range[0])}–${stamp(range[1])} · ${issue.reason}`);say('pitchReviewPlayback','点击试听可从这个位置开始，切换版本保持同一位置。');renderList();draw();controls();
    const hint=previewHints.get(i);if(hint&&!previews.has(i)&&!lookingUp.has(i))lookupPreview(i,hint);
  }
  function nextIssue() {if(!report)return -1;for(let n=1;n<report.issues.length;++n){const i=(index+n)%report.issues.length;if(!accepted.includes(i)&&!ignored.includes(i))return i;}return -1;}
  function matchesBaseline() {
    if(!baseline||!report)return false;const p=baseline.parameters;
    for(const [key,id,scale] of [['f0_up_key','pitchShift',1],['filter_radius','filterRadius',1],['index_rate','indexRate',100],['rms_mix_rate','rmsMixRate',100],['protect','protect',100]])if(Math.abs(Number(q(id).value)/scale-Number(p[key]))>1e-8)return false;
    if(q('voiceSelect').value!==p.voice||q('inputKind').value!==p.workflow||q('conversionDereverb').checked!==(p.pp_dereverb==='1'))return false;
    try{return resonantiaWorkflow.validate()===combine(accepted);}catch{return false;}
  }
  function controls() {
    const issue=report?.issues[index],preview=previews.get(index),hint=previewHints.get(index),ready=preview?.state==='succeeded'&&!!preview.result_url;
    const mismatch=!!report&&!matchesBaseline(),blocked=submitting||mismatch||globalThis.resonantiaTasks?.isBusy();
    q('pitchBaselineWarning').hidden=q('pitchRestore').hidden=!mismatch;
    q('pitchTry').disabled=!issue?.suggestion||submitting||mismatch||lookingUp.has(index)||(!hint&&blocked);
    say('pitchTry',preview?(ready?'查看已完成的试听':'查看已有试听任务'):hint?'重新核实已有试听':'生成这处修正试听');
    q('pitchAdopt').disabled=!issue?.suggestion||!ready||accepted.includes(index)||blocked;
    q('pitchIgnore').disabled=!issue||blocked;q('pitchUndo').disabled=!undo.length||blocked;
    q('pitchListenAfter').disabled=!ready;q('pitchListenBefore').disabled=q('pitchListenSource').disabled=!issue;
    q('pitchOpenPreview').hidden=!hint;q('pitchOpenPreview').disabled=false;q('pitchNext').disabled=nextIssue()<0;
    const state=preview?.state;
    say('pitchPreviewStatus',previewErrors.get(index)|| (lookingUp.has(index)?'正在核实已有试听，暂不提交新任务。':
      ready?'试听已完成。对比“修正前 / 修正后”，满意后采用，或保留原音。':
      state==='failed'?'试听失败，原音频保留。查看任务中的原因，再决定是否继续。':
      ['paused','interrupted','cancelled'].includes(state)?'试听尚未完成，可到原试听任务查看保存点和继续操作。':
      preview?'试听正在后台运行，关闭页面后仍可在任务中心查看进展。':
      hint?'已有试听编号，需核实后台状态后继续。':issue?.suggestion?'尚未生成这处修正试听。先听原版，再决定是否试听建议。':
      issue?'此处可能是正常换音，暂不提供自动修正。试听后可选择保留原音。':'选择一个疑点开始听审。'));
  }
  function draft() {say('pitchDraftStatus',accepted.length?`已采用 ${accepted.length} 处；点击“生成完整新版”才会应用到整首歌。`:'尚未采用修正。试听不会覆盖原结果。');controls();}
  function draw() {
    const svg=q('pitchCurve');svg.querySelectorAll('path,line,text,rect').forEach(e=>e.remove());const ns='http://www.w3.org/2000/svg';
    const add=(tag,attrs,text)=>{const e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text)e.textContent=text;svg.append(e);};
    const first=Math.max(0,Math.ceil((range[0]-report.start_time)*100)),last=Math.min(report.raw.length,Math.floor((range[1]-report.start_time)*100)+1);
    const values=[...report.raw.slice(first,last),...report.processed.slice(first,last)].filter(v=>v>0);if(!values.length)return;
    const low=Math.floor(Math.log2(Math.min(...values)))-1,high=Math.ceil(Math.log2(Math.max(...values)))+1;
    const x=t=>42+(t-range[0])/(range[1]-range[0])*700,y=v=>180-(Math.log2(v)-low)/(high-low)*160;
    for(let n=low;n<=high;++n){add('line',{x1:42,y1:y(2**n),x2:742,y2:y(2**n),stroke:'#d5dce5'});add('text',{x:3,y:y(2**n)+4,fill:'#3d4a60','font-size':11},Math.round(2**n)+'Hz');}
    const issue=report.issues[index];add('rect',{x:x(issue.start),y:12,width:Math.max(2,x(issue.end)-x(issue.start)),height:168,fill:'#ffe3b5',opacity:.6});
    for(const t of [range[0],(range[0]+range[1])/2,range[1]])add('text',{x:x(t),y:203,'text-anchor':t===range[1]?'end':'start',fill:'#3d4a60','font-size':12},stamp(t));
    for(const [a,color,proposal] of [[report.raw,'#6e7787',false],[report.processed,'#2156b5',false],[report.processed,'#b45c06',true]]) {
      if(proposal&&!issue.suggestion)continue;let d='',connected=false;
      for(let i=first;i<last;++i){const t=report.start_time+i*.01;let v=a[i];if(proposal){if(t<issue.suggestion.start||t>=issue.suggestion.end){connected=false;continue;}v*=2**(issue.suggestion.semitones/12);}if(!v||!report.raw[i]){connected=false;continue;}d+=(connected?'L':'M')+x(t).toFixed(2)+','+y(v).toFixed(2);connected=true;}
      add('path',{d,fill:'none',stroke:color,'stroke-width':proposal?2.5:1.5});
    }
  }
  async function mediaUrl(url,token) {
    if(!storedAuthToken())return url;if(urls.has(url))return urls.get(url);
    const r=await get(url),size=Number(r.headers.get('Content-Length')),limit=128*1024*1024;
    if(!Number.isSafeInteger(size)||size<=0||size>limit){await r.body?.cancel();throw new Error('音频超过128MiB载入上限');}
    const reader=r.body.getReader(),chunks=[];let n=0;try{for(;;){const item=await reader.read();if(item.done)break;n+=item.value.byteLength;if(n>size)throw new Error('音频长度不匹配');chunks.push(item.value);}}catch(e){await reader.cancel();throw e;}
    if(n!==size)throw new Error('音频传输不完整');if(token!==generation)return '';
    const blob=URL.createObjectURL(new Blob(chunks,{type:'audio/wav'}));urls.set(url,blob);return blob;
  }
  async function listen(kind) {
    if(!report?.issues[index])return;const token=generation,request=++mediaGeneration;
    const position=mode?Math.max(range[0],Math.min(range[1]-.01,audio.currentTime+offset)):range[0];audio.pause();
    const preview=previews.get(index);if(kind==='after'&&!preview)return;
    const nextOffset=kind==='after'?Number(preview.parameters.preview_start):0;
    const url=kind==='source'?`${API}/task/source?id=${baseline.id}`:`${API}/task/result?id=${kind==='after'?preview.id:baseline.id}&variant=vocal`;
    say('pitchReviewPlayback','正在载入已保存的音频…');
    try {const src=await mediaUrl(url,token);if(token!==generation||request!==mediaGeneration)return;
      audio.src=src;offset=nextOffset;mode=kind;audio.currentTime=Math.max(0,position-offset);await audio.play();
      say('pitchReviewPlayback',kind==='source'?'原始上传音频（可能含伴奏）':kind==='after'?'修正后 · 仅当前疑点的独立试听':'修正前 · 成功版本的干净人声');
    }catch(e){if(token===generation&&request===mediaGeneration)say('pitchReviewPlayback','试听未开始：'+e.message);}
  }
  audio.addEventListener('timeupdate',()=>{if(!mode)return;if(audio.currentTime+offset>=range[1]-.025){if(q('pitchLoop').checked)audio.currentTime=Math.max(0,range[0]-offset);else audio.pause();}});
  audio.addEventListener('ended',()=>{if(mode&&q('pitchLoop').checked){audio.currentTime=Math.max(0,range[0]-offset);audio.play().catch(()=>{});}});
  q('pitchListenSource').onclick=()=>listen('source');q('pitchListenBefore').onclick=()=>listen('before');q('pitchListenAfter').onclick=()=>listen('after');
  q('pitchTry').onclick=async()=>{
    if(!report?.issues[index]?.suggestion||submitting||lookingUp.has(index))return;
    const existing=previews.get(index),hint=previewHints.get(index);
    if(existing){resonantiaTasks.open(existing.id);return;}
    if(hint){await lookupPreview(index,hint);return;}
    if(resonantiaTasks.isBusy()||!matchesBaseline())return;const token=generation,chosen=index;previewErrors.delete(chosen);submitting=true;controls();audio.pause();
    try {const fd=new FormData();for(const [key,value]of Object.entries({source_task:baseline.id,analysis_id:report.analysis_id,pitch_edits:combine([chosen]),preview_start:range[0],preview_end:range[1]}))fd.append(key,String(value));
      const id=await resonantiaTasks.submit(API+'/tasks/pitch-preview',fd);
      if(token===generation){previewHints.set(chosen,id);save();say('pitchDraftStatus','试听已提交。完成后点击任务中的“返回修音，继续听审”即可对比、采用。');}
    }catch(e){if(token===generation)say('pitchDraftStatus','试听未确认：'+e.message+'。请在任务中心检查，避免重复提交。');}
    finally{submitting=false;controls();}
  };
  q('pitchOpenPreview').onclick=()=>{const id=previewHints.get(index);if(validId(id))resonantiaTasks.open(id);};
  q('pitchNext').onclick=()=>{const i=nextIssue();if(i>=0)select(i);};
  q('pitchAdopt').onclick=()=>{if(q('pitchAdopt').disabled||resonantiaTasks.isBusy()||!matchesBaseline())return;try{
    const next=[...accepted,index];resonantiaWorkflow.setPitchEdits(combine(next));undo.push([...accepted]);accepted=next;ignored=ignored.filter(i=>i!==index);save();renderList();draft();
  }catch(e){say('pitchDraftStatus',e.message);}};
  q('pitchIgnore').onclick=()=>{if(q('pitchIgnore').disabled||!matchesBaseline())return;try{
    if(accepted.includes(index)){const next=accepted.filter(i=>i!==index);resonantiaWorkflow.setPitchEdits(combine(next));undo.push([...accepted]);accepted=next;}
    if(!ignored.includes(index))ignored.push(index);save();renderList();draft();say('pitchDraftStatus','此处保留原音，未应用建议。可继续检查下一处。');
  }catch(e){say('pitchDraftStatus',e.message);}};
  q('pitchUndo').onclick=()=>{if(q('pitchUndo').disabled||!undo.length||!matchesBaseline())return;const next=undo.at(-1);try{resonantiaWorkflow.setPitchEdits(combine(next));undo.pop();accepted=next;ignored=ignored.filter(i=>!accepted.includes(i));save();renderList();draft();}catch(e){say('pitchDraftStatus',e.message);}};
  q('pitchRestore').onclick=async()=>{if(!baseline||resonantiaTasks.isBusy())return;try{await resonantiaReviseAudio(baseline,{focusPitch:true});}catch(e){say('pitchDraftStatus',e.message);}};
  for(const event of ['input','change'])q('panel-convert').addEventListener(event,()=>{if(report)controls();});
  q('pitchReload').onclick=()=>{if(baseline)load(baseline,selectedRecord);};
  resonantiaTasks.observe(r=>{if(report)remember(r);controls();});
  document.addEventListener('play',e=>{if(e.target instanceof HTMLMediaElement)document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause();});},true);
  document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>audio.pause()));
  globalThis.resonantiaPitchReview={load,reset};
})();
