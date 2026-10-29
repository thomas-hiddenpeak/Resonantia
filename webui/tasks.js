// Server-owned jobs. Browser storage is only a selection/submission hint.
(() => {
  const q = id => document.getElementById(id);
  const active = r => ['queued', 'running', 'pause_requested'].includes(r?.state);
  const states = {queued:'等待设备',running:'运行中',pause_requested:'正在安全暂停',paused:'已暂停',interrupted:'服务中断',failed:'失败',cancelled:'已停止',succeeded:'已完成'};
  const stages = {queued:'等待设备',preparing:'准备素材',checking:'核验素材与模型',preprocessing:'清理与切分素材',separating:'分离人声',cleanup:'去和声与混响',cleaning:'清理主唱','preparing-vocal':'准备独唱人声',converting:'转换声线',mixing:'合成伴奏',saving:'保存结果',training:'训练声线',indexing:'构建音色索引',publishing:'保存训练候选',done:'完成',error:'失败'};
  const phases = {identity:'核验素材与模型','loading-models':'载入模型',f0:'提取音高',synthesis:'合成人声',output:'保存音频',separating:'分离音频',cache:'准备训练特征',training:'更新训练参数',indexing:'构建音色索引',separation:'分离音频'};
  const units = {chunks:'块',samples16k:'采样（16 kHz）',frames100hz:'帧（每秒 100 帧）',blocks:'块',frames:'帧',samples:'采样',steps:'步',files:'份',windows:'窗'};
  const kinds = {convert:'翻唱',preprocess:'预处理',training:'训练',step:'预处理'};
  const title = r => r?.display_name || r?.name || r?.id || '';
  const validId = id => typeof id === 'string' && /^[A-Za-z0-9_-]{8,96}$/.test(id);
  function read(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } }
  function save(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* server remains authoritative */ } }
  let selected = read('resonantiaTaskSelection', '');
  if (!validId(selected)) selected = '';
  let pending = read('resonantiaPendingTasks', []);
  if (!Array.isArray(pending)) pending = [];
  pending = pending.filter(validId).slice(0,20);
  const savedRequests = read('resonantiaPendingRecipes', []);
  const requests = new Map((Array.isArray(savedRequests)?savedRequests:[]).slice(0,20).filter(r=>
    validId(r?.id) && ['/api/train','/api/tasks/convert','/api/tasks/preprocess'].includes(r.endpoint) &&
    Array.isArray(r.fields) && r.fields.length<=64 && r.fields.every(pair=>Array.isArray(pair)&&pair.length===2&&
      typeof pair[0]==='string'&&/^[a-z][a-z0-9_]{0,63}$/.test(pair[0])&&typeof pair[1]==='string'&&pair[1].length<=4096)
  ).map(r=>[r.id,r]));
  function forget(id) {pending=pending.filter(value=>value!==id);requests.delete(id);save('resonantiaPendingTasks',pending);save('resonantiaPendingRecipes',[...requests.values()]);}
  let rows = [], current = null, offset = 0, busy = false, refreshing = false, actionBusy = false, activeId = '', storageReason = '';
  let connected = false, timer, mediaId = '', mediaGeneration = 0, mediaSignature = '', renameTarget = null;
  const media = {
    primary:{audio:'taskAudio',download:'taskDownload',load:'taskLoadPrimary',message:'taskPrimaryMessage',url:'',kind:'',pending:false},
    vocal:{audio:'taskVocalAudio',download:'taskVocalDownload',load:'taskLoadVocal',message:'taskVocalMessage',url:'',kind:'vocal',pending:false}
  };
  const listeners = new Set();
  const text = (id, value) => { q(id).textContent = value; };
  const time = value => value ? new Date(value).toLocaleString() : '尚未记录';
  const gib = value => (value / 1073741824).toFixed(2) + ' GiB';
  function choose(id, navigate = false) {
    selected = id; save('resonantiaTaskSelection', id);
    if (navigate) q('nav-tasks').click();
    if (current?.id !== id) { current = null; clearMedia(); renameTarget=null;q('taskRenameForm').hidden=true;q('taskDetail').hidden = true; }
    refresh();
  }
  function clearMedia() {
    ++mediaGeneration; mediaId = '';mediaSignature='';q('taskResults').hidden=true;
    for(const slot of Object.values(media)) {
      q(slot.audio).pause();q(slot.audio).removeAttribute('src');q(slot.audio).load();q(slot.download).hidden=true;
      q(slot.download).removeAttribute('href');text(slot.message,'');q(slot.load).hidden=true;slot.pending=false;
      if(slot.url)URL.revokeObjectURL(slot.url);slot.url='';slot.kind='';
    }
  }
  function connection(ok, message) {
    connected = ok;
    text('taskConnection', message || (ok ? '已连接 · 状态来自后台' : '连接中断 · 保留上次状态，任务可能仍在运行'));
    q('taskConnection').dataset.connected = String(ok);
    setConn(ok);
  }
  async function request(url, options = {}, interactive = false) {
    const response = await apiFetch(url, { ...options, quietAuth:!interactive, signal:AbortSignal.timeout(15000) });
    if (!response.ok) { const error = new Error(await errText(response)); error.status = response.status; throw error; }
    return response;
  }
  function renderList() {
    const list = q('taskList'); list.replaceChildren();
    for (const r of rows) {
      const li = document.createElement('li'), button = document.createElement('button');
      button.type = 'button'; button.className = 'task-row'; button.setAttribute('aria-pressed', String(r.id === selected));
      const name = document.createElement('strong'), summary = document.createElement('span');
      name.textContent = title(r);
      summary.textContent = `${kinds[r.kind] || '任务'} · ${states[r.state] || r.state} · ${time(r.created_ms)}`;
      button.append(name, summary); button.onclick = () => choose(r.id); li.append(button); list.append(li);
    }
    q('taskEmpty').hidden = rows.length > 0;
    q('taskPrevious').disabled = offset === 0;
    q('taskNext').disabled = rows.length < 25;
    text('taskPage', `第 ${offset / 25 + 1} 页`);
  }
  function renderDetail() {
    const r = current; if (!r) return;
    q('taskDetail').hidden = false;
    text('taskName', title(r)); text('taskId', `${r.id} · 第 ${r.attempt} 次执行`);
    q('taskRename').disabled=!connected||actionBusy;
    q('taskRenameSave').disabled=!connected||actionBusy;
    text('taskStateLabel', states[r.state] || r.state);
    text('taskStage', `${stages[r.stage] || r.stage || '等待状态'}${r.phase ? ' · ' + (phases[r.phase] || r.phase) : ''}`);
    const hasCount = Number.isFinite(r.total) && r.total > 0 && Number.isFinite(r.completed) && r.completed >= 0;
    text('taskCounter', hasCount ? `${r.completed} / ${r.total} ${units[r.unit] || r.unit || ''}` : '此阶段暂无可计量进度');
    const bar = q('taskProgress'); bar.hidden = !hasCount; bar.max = hasCount ? r.total : 1; bar.value = hasCount ? Math.min(r.completed,r.total) : 0;
    const seconds = Math.max(0, Math.floor(((r.finished_ms || Date.now()) - r.created_ms) / 1000));
    text('taskTimes', `提交：${time(r.created_ms)} · 距提交 ${Math.floor(seconds/60)} 分 ${seconds%60} 秒（含等待与暂停）`);
    text('taskFreshness', `最近进展：${time(r.progress_at_ms)} · 状态记录：${time(r.updated_ms)}`);
    text('taskError', r.error || ''); q('taskError').hidden = !r.error;
    text('taskRecovery', r.parameters?.reused_stages ? '本版已复用：'+r.parameters.reused_stages.split(',').map(x=>({separation:'人声分离',cleanup:'主唱清理',conversion:'声线转换'})[x]||x).join('、')+'。其余阶段按本版设置执行。' : r.state === 'pause_requested' ? '暂停请求已送达，等待当前安全保存点；此时尚未释放设备。' :
      r.state === 'interrupted' ? '服务曾中断。继续会先核验已保存内容，从有效保存点恢复；未保存的计算需要重做。' :
      r.actions?.resume ? '继续使用本任务固定的素材与设置，复用已验证的保存点。若身份核验失败，请查看日志。' :
      active(r) ? '任务在服务器运行，可以关闭页面。重新打开“任务中心”即可查看。' :
      r.kind === 'training' && r.state === 'succeeded' ? '训练候选已保存。请到声线库听审，再决定是否发布。' : '状态与日志已保留。');
    q('taskPause').hidden = !r.actions?.pause; q('taskPause').disabled = !connected || actionBusy || r.state === 'pause_requested';
    q('taskResume').hidden = !r.actions?.resume; q('taskResume').disabled = !connected || actionBusy || busy;
    q('taskRevise').hidden = !r.actions?.revise; q('taskRevise').disabled = !connected || actionBusy || busy;
    q('taskParent').hidden = !validId(r.parameters?.source_task);
    q('taskParent').onclick = () => choose(r.parameters.source_task,true);
    q('taskCleanup').hidden = !r.actions?.cleanup;q('taskCleanup').disabled = !connected || actionBusy || busy;
    text('taskRetention',r.parameters?.retention === 'pruned-v1' ? '冗余副本已清理。原始素材、结果、日志与训练保存点仍保留。' : '不会自动删除历史。成功后可清理冗余副本，保留原始素材、结果、日志与训练保存点。');
    q('taskLegacyCancel').hidden = !(r.kind === 'training' && active(r) && r.parameters?.recovery !== 'training-full-v1');
    q('taskLegacyCancel').disabled = !connected || actionBusy || r.stage === 'publishing';
    q('taskLibrary').hidden = r.kind !== 'training';
    q('taskListen').hidden = !(r.state === 'succeeded' && r.result_url) || Array.isArray(r.outputs);
    q('taskListen').disabled = !connected;
    renderMedia(r);
    const recipe = q('taskRecipe'); recipe.replaceChildren();
    for (const [key,label] of [['voice','声线'],['workflow','输入处理'],['output_kind','输出'],['mode','训练方式'],['epochs','训练轮数'],['f0_up_key','音高偏移'],['index_rate','音色检索率'],['rms_mix_rate','动态保留']]) {
      if (r.parameters?.[key] === undefined) continue;
      const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent=label; dd.textContent=r.parameters[key]; recipe.append(dt,dd);
    }
  }
  function renderPending() {
    q('taskPending').hidden = !pending.length;
    q('taskRetrySubmission').hidden = !pending.some(id=>requests.has(id));
    text('taskPending', pending.length ? `有 ${pending.length} 次提交尚未确认。正在按原任务编号查询，请勿重复提交。${pending.join('、')}` : '');
  }
  async function refresh(interactive = false) {
    if (refreshing) return;
    refreshing = true; clearTimeout(timer);
    try {
      const data = await (await request(`${API}/tasks?offset=${offset}&limit=25`, {}, interactive)).json();
      rows = data.tasks;
      if (!Array.isArray(rows)) throw new Error('任务列表格式错误');
      activeId = data.active_id || rows.find(active)?.id || '';busy=!!activeId;
      q('taskActive').hidden=!activeId;text('taskActiveName',activeId ? '后台正在运行：'+(title(rows.find(r=>r.id===activeId)) || activeId) : '');
      for (const id of [...pending]) {
        try {
          const found = await (await request(`${API}/task?id=${encodeURIComponent(id)}`)).json();
          if (found.id === id) { forget(id); if (!selected) selected = id; }
        } catch(e) { if (e.status !== 404) throw e; }
      }
      save('resonantiaPendingTasks',pending); renderPending();
      if (!selected) selected = data.active_id || rows[0]?.id || '';
      const id = selected;
      if (id) {
        try {
          const r = await (await request(`${API}/task?id=${encodeURIComponent(id)}`)).json();
          if (selected === id) { current = r; save('resonantiaTaskSelection',id); }
        } catch(e) {
          if (e.status !== 404) throw e;
          if (selected === id) {current=null;q('taskDetail').hidden=true;}
        }
      }
      connection(true); renderList(); renderDetail();
      try {
        const usage=await (await request(API+'/storage')).json();storageReason=usage.reason || '';
        text('taskStorage',`任务文件 ${gib(usage.managed_bytes)} / ${gib(usage.managed_limit_bytes)} · 磁盘可用 ${gib(usage.available_bytes)}（预留 ${gib(usage.reserve_bytes)}） · 未清理任务 ${usage.retained_tasks}/${usage.retained_limit} · 历史 ${usage.history_tasks}/${usage.history_limit}`);
        text('taskStorageWarning',storageReason);q('taskStorageWarning').hidden=!storageReason;
      } catch(e) {storageReason='磁盘状态暂不可确认，请刷新核查';text('taskStorageWarning',storageReason+'：'+e.message);q('taskStorageWarning').hidden=false;}
      for (const r of new Map([...rows, ...(current ? [current] : [])].map(r=>[r.id,r])).values()) for (const listener of listeners) listener(r);
      if (current && q('taskLogDetails').open) await loadLog();
    } catch(e) {
      connection(false, e.status === 401 ? '需要访问令牌 · 点击刷新重新连接，后台任务状态暂不可确认' : '连接中断 · 保留上次状态，任务可能仍在运行');
      renderDetail();
    } finally {
      refreshing = false; globalThis.resonantiaStudio?.render();
      timer = setTimeout(refresh, busy || pending.length || !connected ? 2000 : 10000);
    }
  }
  async function loadLog() {
    const id = selected; if (!id) return;
    try { const response = await request(`${API}/task/log?id=${encodeURIComponent(id)}`); const log = await response.text();
      if (selected === id) text('taskLog', log || '暂无日志');
    } catch(e) { if (selected === id) text('taskLog', '日志暂不可读取：' + e.message); }
  }
  async function control(action) {
    if (!current || actionBusy) return;
    const r = current; actionBusy = true; renderDetail(); text('taskActionMessage','正在提交请求…');
    const fd = new FormData(); fd.append('id',r.id); fd.append('attempt',String(r.attempt));
    if (action === 'cancel') fd.append('job_id',r.id);
    try {
      const response=await request(action === 'cancel' ? API+'/train/cancel' : API+'/task/'+action,{method:'POST',body:fd},true);
      const result=await response.json();
      text('taskActionMessage',action === 'cleanup' ? `冗余副本已清理，释放 ${gib(result.freed_bytes)}。结果与保存点保留。` : action === 'resume' ? '继续请求已受理，正在读取后台状态。' : '停止请求已受理，等待后台安全退出。');
    } catch(e) { text('taskActionMessage','操作结果未确认：'+e.message+'。刷新状态后再决定。'); }
    finally {actionBusy=false;await refresh();renderDetail();}
  }
  const labels={mix:'完整翻唱',vocal:'干净人声',result:'处理后音频'};
  function downloadName(r,kind) {
    const name=title(r).replace(/\.[a-z0-9]{2,5}$/i,'').replace(/[<>:"/\\|?*\x00-\x1f]/g,'_').replace(/[. ]+$/g,'')||r.id;
    return `${name}_${labels[kind]}.wav`;
  }
  function renderMedia(r) {
    if(r.state!=='succeeded'||!r.result_url){if(mediaId)clearMedia();return;}
    let outputs=Array.isArray(r.outputs)?r.outputs.filter(o=>o&&Object.hasOwn(labels,o.kind)).slice(0,2):[];
    if(!outputs.length)outputs=[{kind:r.parameters?.output_kind==='mix'?'mix':r.kind==='convert'?'vocal':'result'}];
    const signature=r.id+':'+outputs.map(o=>o.kind).join(',');
    if(signature!==mediaSignature) {
      clearMedia();mediaId=r.id;mediaSignature=signature;q('taskResults').hidden=false;
      media.primary.kind=outputs[0].kind;
      media.vocal.kind=outputs.length>1&&outputs[1].kind==='vocal'?'vocal':'';
      q('taskVocalCard').hidden=!media.vocal.kind;
      text('taskPrimaryTitle',labels[media.primary.kind]);q('taskAudio').setAttribute('aria-label',labels[media.primary.kind]);
      text('taskPrimaryHint',media.primary.kind==='mix'?'转换后的人声与原伴奏合成。':media.primary.kind==='vocal'?'转换后的目标声线，不含伴奏。':'已保存的预处理结果。');
      for(const slot of Object.values(media))if(slot.kind) {
        q(slot.load).disabled=false;
        if(storedAuthToken()){q(slot.load).hidden=false;text(slot.message,'点击载入后可试听和下载。');}
        else setMedia(slot,`${API}/task/result?id=${encodeURIComponent(r.id)}&variant=${slot.kind}`,r);
      }
    }
    text('taskOutputsNotice',r.outputs_notice||'');q('taskOutputsNotice').hidden=!r.outputs_notice;
    for(const slot of Object.values(media))if(slot.kind)q(slot.download).download=downloadName(r,slot.kind);
  }
  function setMedia(slot,url,r) {
    q(slot.audio).src=url;q(slot.audio).hidden=false;q(slot.download).href=url;
    q(slot.download).download=downloadName(r,slot.kind);q(slot.download).hidden=false;q(slot.load).hidden=true;
    text(slot.message,'播放或下载不会重新转换。');
  }
  async function loadMedia(which) {
    const r=current,slot=media[which];if(!r?.result_url||!slot.kind||slot.pending)return;
    slot.pending=true;q(slot.load).disabled=true;
    const generation=mediaGeneration; text(slot.message,'正在载入已保存的音频…');
    try {
      // Only the task-ID endpoint is trusted; never follow a URL supplied by metadata.
      const url = `${API}/task/result?id=${encodeURIComponent(r.id)}&variant=${slot.kind}`;
      let mediaUrl = url;
      if (storedAuthToken()) {
        const response = await request(url, {}, true), limit = 256*1024*1024;
        const size = Number(response.headers.get('Content-Length'));
        if (!Number.isSafeInteger(size) || size <= 0 || size > limit) { await response.body?.cancel(); throw new Error('此音频超出页面的 256 MiB 载入上限'); }
        const reader=response.body.getReader(), chunks=[];let received=0;
        try { for (;;) {const {done,value}=await reader.read();if(done)break;received+=value.byteLength;if(received>limit)throw new Error('音频超出载入上限');chunks.push(value);} }
        catch(e) {await reader.cancel();throw e;}
        if (received !== size) throw new Error('音频传输未完整，请重试');
        if (generation !== mediaGeneration) return;
        if (slot.url) URL.revokeObjectURL(slot.url);
        slot.url = URL.createObjectURL(new Blob(chunks,{type:'audio/wav'})); mediaUrl=slot.url;
      }
      if (generation !== mediaGeneration || selected !== r.id) return;
      setMedia(slot,mediaUrl,r);
    } catch(e) {if(generation===mediaGeneration){text(slot.message,'音频载入失败：'+e.message);q(slot.load).hidden=false;}}
    finally {if(generation===mediaGeneration){slot.pending=false;q(slot.load).disabled=false;}}
  }
  async function listen() {const id=selected;await loadMedia('primary');if(selected===id)await loadMedia('vocal');}
  async function rename(event) {
    event.preventDefault();if(!renameTarget||actionBusy)return;
    const target={...renameTarget},name=q('taskRenameInput').value.trim();
    if(!name||new TextEncoder().encode(name).length>255){text('taskRenameMessage','请填写名称，最多 255 字节（约 85 个汉字）。');return;}
    const fd=new FormData();fd.append('id',target.id);fd.append('expected_name',target.expected);fd.append('name',name);
    actionBusy=true;renderDetail();text('taskRenameMessage','正在保存…');
    try {
      await request(API+'/task/rename',{method:'POST',body:fd},true);
      if(selected===target.id){q('taskRenameForm').hidden=true;renameTarget=null;text('taskActionMessage','名称已保存，任务进度与音频保持不变。');}
    }catch(e){if(selected===target.id)text('taskRenameMessage','名称未确认保存：'+e.message+(e.status===409?'。请取消编辑，核实最新名称后重试。':''));}
    finally {actionBusy=false;await refresh();renderDetail();}
  }
  async function transmit(id,endpoint,fd) {
    try {
      const response=await request(endpoint,{method:'POST',body:fd},true), accepted=await response.json();
      if ((accepted.task_id || accepted.job_id) !== id) throw new Error('服务器返回的任务编号不匹配，请核实历史');
      forget(id);
    } catch(e) {
      // A lost response never means a failed job. Only definite rejection retires the hint.
      if (e.status === 409) {
        try { await request(`${API}/task?id=${encodeURIComponent(id)}`); }
        catch (lookup) {
          if (lookup.status === 404) { forget(id);throw e; }
        }
      }
      if ((e.status >= 400 && e.status < 500 && e.status !== 409 && e.status !== 408) || e.status === 507) {
        forget(id);throw e;
      }
      text('taskActionMessage','提交响应未收到，正在按原任务编号核实；不会自动重复提交。');
    } finally {await refresh();renderPending();}
  }
  async function submit(endpoint, fd, kind) {
    if (pending.length >= 20) throw new Error('仍有提交待确认，请先到任务中心核实');
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    const id = 'web-' + [...bytes].map(x=>x.toString(16).padStart(2,'0')).join('');
    fd.set('task_id',id);
    const fields=[...fd.entries()];
    if(fields.length>64 || fields.some(([key,value])=>!/^[a-z][a-z0-9_]{0,63}$/.test(key)||typeof value!=='string'||value.length>4096))throw new Error('后台任务只接受已上传素材的有限设置');
    requests.set(id,{id,endpoint,fields});save('resonantiaPendingRecipes',[...requests.values()]);
    pending.push(id);save('resonantiaPendingTasks',pending);selected=id;save('resonantiaTaskSelection',id);
    busy=true; renderPending(); q('nav-tasks').click(); text('taskActionMessage','正在提交，接收结果将保留在任务中心。');
    await transmit(id,endpoint,fd);return id;
  }
  async function retrySubmission() {
    const id=pending.find(value=>requests.has(value));if(!id)return;
    q('taskRetrySubmission').disabled=true;
    try {
      try {await request(API+'/task?id='+encodeURIComponent(id),{},true);forget(id);choose(id,true);return;}
      catch(e){if(e.status!==404)throw e;}
      const recipe=requests.get(id),fd=new FormData();for(const [key,value] of recipe.fields)fd.append(key,value);fd.set('task_id',id);
      await transmit(id,recipe.endpoint,fd);
    }catch(e){text('taskPending','重试未受理：'+e.message);}
    finally {q('taskRetrySubmission').disabled=false;renderPending();}
  }
  globalThis.resonantiaTasks = {submit,open:id=>choose(id,true),refresh,isBusy:()=>busy || pending.length>0,
    blockingReason:()=>storageReason,
    observe:fn=>listeners.add(fn),get:id=>current?.id===id ? current : rows.find(r=>r.id===id)};
  document.addEventListener('DOMContentLoaded',()=>{
    q('taskRename').onclick=()=>{if(!current)return;renameTarget={id:current.id,expected:title(current)};q('taskRenameInput').value=title(current);q('taskRenameForm').hidden=false;text('taskRenameMessage','');q('taskRenameInput').focus();};
    q('taskRenameForm').onsubmit=rename;q('taskRenameCancel').onclick=()=>{renameTarget=null;q('taskRenameForm').hidden=true;q('taskRename').focus();};
    q('taskLoadPrimary').onclick=()=>loadMedia('primary');q('taskLoadVocal').onclick=()=>loadMedia('vocal');
    q('taskRefresh').onclick=()=>refresh(true);q('taskPause').onclick=()=>control('pause');q('taskResume').onclick=()=>control('resume');
    q('taskCleanup').onclick=()=>control('cleanup');q('taskActiveOpen').onclick=()=>choose(activeId,true);
    q('taskRetrySubmission').onclick=retrySubmission;
    q('taskRevise').onclick=async()=>{if(!current||busy||actionBusy)return;actionBusy=true;renderDetail();try{await globalThis.resonantiaReviseAudio(current);}catch(e){text('taskActionMessage','无法载入原任务：'+e.message);}finally{actionBusy=false;renderDetail();}};
    q('taskLegacyCancel').onclick=()=>control('cancel');q('taskListen').onclick=listen;q('taskLibrary').onclick=()=>q('nav-library').click();
    q('taskPrevious').onclick=()=>{offset=Math.max(0,offset-25);refresh();};q('taskNext').onclick=()=>{offset+=25;refresh();};
    q('taskLogDetails').addEventListener('toggle',()=>{if(q('taskLogDetails').open)loadLog();});
    for(const slot of Object.values(media)) {
      q(slot.audio).addEventListener('error',()=>{if(!slot.kind)return;text(slot.message,'音频读取失败，点击载入可重试；任务不会重新转换。');q(slot.load).hidden=false;});
      q(slot.audio).addEventListener('play',()=>{for(const other of Object.values(media))if(other!==slot)q(other.audio).pause();});
    }
    window.addEventListener('online',()=>refresh());window.addEventListener('offline',()=>{connection(false);renderDetail();});
    window.addEventListener('focus',()=>refresh());document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh();});
    renderPending();refresh();
  });
})();
