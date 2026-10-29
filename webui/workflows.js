// Additive product controls; existing training/cascade contracts stay in app.js.
(() => {
    let candidates = [], candidatePreview = null, locked = false;
    let previousWorkflow = 'song';
    const q = (id) => document.getElementById(id);
    const option = (value, label) => { const o = document.createElement('option'); o.value=value; o.textContent=label; return o; };
    const runLabel = name => name === 'CyberTom_Full_20261026' ? 'Tom Cybersinger · 正式训练（2026-10-26）' : name;
    function workflowChanged() {
        const mode = q('inputKind').value;
        q('workflowHint').textContent = {
            song:'原始歌曲 → 立体声分离人声 → 去和声保留主唱 → 声线转换 → 可合回原伴奏。比干声转换多两次分离，耗时会增加。',
            harmonies:'从原始上传的人声中去掉伴唱 / 和声，再转换主唱。输出为独唱人声。',
            dry:'直接转换原始上传的单人干声。保留自然轻重，避免不必要的清理。',
            manual:'使用下方逐步处理后的结果转换，仅输出人声；其他输入类型会重新使用原始上传文件。'
        }[mode];
        q('inputKind').disabled = locked || !!convSource;
        q('manualCleanup').querySelectorAll('button,select').forEach(el=>{if(convSource)el.disabled=true;});
        q('convResetPre').disabled=locked||!!convSource;
        q('outputKind').disabled = locked || mode !== 'song';
        if (mode !== 'song') q('outputKind').value = 'vocal';
        else if (previousWorkflow !== 'song') q('outputKind').value = 'mix';
        previousWorkflow = mode;
        q('manualCleanup').hidden = mode !== 'manual';
        q('conversionDereverb').disabled = locked || mode === 'manual';
        if (mode === 'manual') q('conversionDereverb').checked = false;
        globalThis.resonantiaStudio?.render();
    }
    q('inputKind').addEventListener('change', workflowChanged);
    function lock(value) {
        locked = value;
        for (const root of ['panel-convert']) q(root).querySelectorAll('input,select,button').forEach(el => {
            if (el.id !== 'btnConvert') el.disabled = value;
        });
        if (!value) {
            convCascade.setBusy(false);
            q('voiceSelect').disabled = !!candidatePreview;
            workflowChanged();
            q('pitchEdits').querySelectorAll('.pitch-edit').forEach(row=>{
                row.querySelector('[data-role="shift"]').disabled=row.querySelector('[data-role="mode"]').value!=='shift';
            });
        }
    }
    function pitchText() {
        const edits = [];
        let previous = -1;
        q('pitchEdits').querySelectorAll('.pitch-edit').forEach(row => {
            const get = role => row.querySelector('[data-role="'+role+'"]');
            const start = Number(get('start').value), end = Number(get('end').value), shift = Number(get('shift').value);
            if (!get('start').value || !get('end').value || !Number.isFinite(start) || !Number.isFinite(end) ||
                start < 0 || end > 600 || end-start < .02-1e-9 || end-start > 2 || start <= previous ||
                !Number.isFinite(shift) || Math.abs(shift)>24) throw new Error('请按时间排序填写修音区间：每段 0.02–2 秒，不重叠，偏移在 ±24 半音内');
            previous=end;
            edits.push(get('mode').value+':'+start+':'+end+(get('mode').value==='shift'?':'+shift:''));
        });
        return edits.join(';');
    }
    q('addPitchEdit').onclick = () => {
        if (locked || q('pitchEdits').children.length >= 16) return;
        const row=document.createElement('div'); row.className='pitch-edit';
        row.innerHTML='<label>方式<select data-role="mode"><option value="smooth">平滑跳音</option><option value="shift">半音偏移</option></select></label><label>开始（秒）<input data-role="start" type="number" min="0" max="600" step="0.01"></label><label>结束（秒）<input data-role="end" type="number" min="0" max="600" step="0.01"></label><label>偏移（半音）<input data-role="shift" type="number" min="-24" max="24" value="12" step="1" disabled></label><button class="btn-small" type="button">移除</button>';
        row.querySelector('select').onchange=()=>{ row.querySelector('[data-role="shift"]').disabled=row.querySelector('select').value!=='shift'; };
        row.querySelector('button').onclick=()=>{row.remove();globalThis.resonantiaStudio?.render();}; q('pitchEdits').appendChild(row);
        globalThis.resonantiaStudio?.render();
    };
    globalThis.resonantiaWorkflow = {
        lock,
        hasCandidate: () => !!candidatePreview,
        validate: pitchText,
        resetPitchEdits: () => q('pitchEdits').replaceChildren(),
        refreshRuns,
        restore(p) {
            candidatePreview=null;q('candidateNotice').hidden=true;q('exitCandidate').hidden=true;
            q('inputKind').value=p.workflow;workflowChanged();q('outputKind').value=p.output_kind;
            q('conversionDereverb').checked=p.pp_dereverb==='1';
            q('pitchEdits').replaceChildren();
            for(const text of (p.pitch_edits||'').split(';').filter(Boolean)) {
                q('addPitchEdit').onclick();const row=q('pitchEdits').lastElementChild,parts=text.split(':');
                for(const [i,role] of ['mode','start','end','shift'].entries())if(parts[i]!==undefined)row.querySelector('[data-role="'+role+'"]').value=parts[i];
                row.querySelector('select').dispatchEvent(new Event('change'));
            }
        },
        append(fd) {
            fd.append('workflow',q('inputKind').value); fd.append('output_kind',q('outputKind').value);
            if(!(convSource&&q('inputKind').value==='manual'))fd.append('pp_dereverb',q('conversionDereverb').checked?'1':'0');
            fd.append('pitch_edits',pitchText());
            if (candidatePreview) { fd.append('training_run',candidatePreview.run); fd.append('candidate',candidatePreview.file); }
        }
    };
    function selectedRun() { return candidates.find(r=>r.name===q('trainingRun').value); }
    function chooseRun() {
        const run=selectedRun(); q('trainingCandidate').innerHTML='';
        q('publishReviewed').checked=false; q('publishStatus').textContent='';
        if (!run) {
            q('runInfo').textContent='暂无完整训练记录。先到“训练声线”建立任务，完成后在这里听审候选。';
            q('resumeRun').disabled=true; q('previewCandidate').disabled=true; q('publishVoice').disabled=true;
            q('publishName').value=''; return;
        }
        run.candidates.forEach(file=>q('trainingCandidate').appendChild(option(file,'第 '+Number(file.slice(16,26))+' 轮')));
        q('trainingCandidate').value=run.candidates.at(-1)||'';
        q('runInfo').textContent=(run.result_ready?'训练结果已就绪，可试听候选或以新名称发布。':'训练结果尚未完整，请查看日志。')+(!run.result_ready?(run.resumable?' 已有可恢复检查点。':' 尚无恢复检查点。'):' 此任务已完成，无需恢复。');
        q('resumeRun').textContent=run.task_id?'查看原任务':'恢复训练'; q('resumeRun').disabled=!run.task_id && (!run.resumable || run.result_ready); q('publishName').value=(run.name==='CyberTom_Full_20261026'?'Tom_Cybersinger':run.name)+'_Released';
        q('previewCandidate').disabled=!run.candidates.length; q('publishVoice').disabled=!run.result_ready || !run.candidates.length;
    }
    async function refreshRuns() {
        try {
            const response=await apiFetch(API+'/train/runs'); if(!response.ok)throw new Error(await errText(response));
            const data=await response.json(), previous=q('trainingRun').value; candidates=data.runs;
            q('trainingRun').innerHTML=''; candidates.forEach(run=>q('trainingRun').appendChild(option(run.name,runLabel(run.name))));
            if (candidates.some(run=>run.name===previous)) q('trainingRun').value=previous;
            else if(candidates.length)q('trainingRun').value=candidates.at(-1).name;
            chooseRun();
        } catch(error) { q('runInfo').textContent='无法读取训练记录：'+error.message; }
    }
    q('refreshRuns').onclick=refreshRuns; q('trainingRun').onchange=chooseRun;
    q('refreshRunLog').onclick=async()=>{
        const name=q('trainingRun').value;if(!name)return;
        try {const run=selectedRun();if(run?.task_id){globalThis.resonantiaTasks?.open(run.task_id);return;}const response=await apiFetch(API+'/train/status?name='+encodeURIComponent(name));
            if(!response.ok)throw new Error(await errText(response));const result=await response.json();
            if(q('trainingRun').value===name)q('runLog').textContent=result.log||'暂无日志';
        }catch(error){q('runLog').textContent='日志读取失败：'+error.message;}
    };
    q('trainingCandidate').onchange=()=>{q('publishReviewed').checked=false;};
    q('resumeRun').onclick=()=>{
        const run=selectedRun(); if(run?.task_id){globalThis.resonantiaTasks?.open(run.task_id);return;} if(!run?.resumable || run.result_ready)return;
        q('voiceName').value=run.name; q('trainMode').value='joint-full'; applyTrainingMode();
        q('trainResume').checked=true; updateTrainBtn();
        document.querySelector('[data-tab="train"]').click();
        q('runInfo').textContent='已填入恢复设置。确认下方轮数与原训练一致，再点击开始训练；无需重新上传素材。';
    };
    q('previewCandidate').onclick=()=>{
        const run=selectedRun(), file=q('trainingCandidate').value; if(!run||!file||convRunning)return;
        candidatePreview={run:run.name,file}; q('candidateNotice').hidden=false; q('exitCandidate').hidden=false;
        q('candidateNotice').textContent='候选试听：'+runLabel(run.name)+' · 第 '+Number(file.slice(16,26))+' 轮（仅试听，不替换正式声线）';
        document.querySelector('[data-tab="convert"]').click(); q('voiceSelect').disabled=true;
        q('inputKind').value='dry';workflowChanged();
        const preset=PRESETS.inference.find(p=>p.id==='tom_cybersinger'); if(preset)applyPreset(preset.id);
        if(!run.result_ready){q('indexRate').value=0;q('indexRateValue').textContent='0.00';}
    };
    q('exitCandidate').onclick=()=>{
        if(convRunning)return;candidatePreview=null;q('candidateNotice').hidden=true;q('exitCandidate').hidden=true;
        q('voiceSelect').disabled=false; lastVoicePreset=null;applyVoicePreset();
        globalThis.resonantiaStudio?.render();
    };
    q('publishVoice').onclick=async()=>{
        const run=selectedRun(); if(!run)return;
        if(!q('publishReviewed').checked){q('publishStatus').textContent='请先听审此候选，再勾选确认。';return;}
        const data=new FormData();data.append('name',run.name);data.append('candidate',q('trainingCandidate').value);
        data.append('publish_name',q('publishName').value.trim());data.append('reviewed','1');q('publishVoice').disabled=true;
        try{
            const response=await apiFetch(API+'/train/publish',{method:'POST',body:data});if(!response.ok)throw new Error(await errText(response));
            const result=await response.json();await loadVoices();q('voiceSelect').value=result.published;lastVoicePreset=null;applyVoicePreset();
            q('publishStatus').textContent='已发布「'+result.published+'」，可在翻唱页选择。';q('publishReviewed').checked=false;
        }catch(error){q('publishStatus').textContent='发布失败：'+error.message;}
        finally{q('publishVoice').disabled=false;}
    };
    workflowChanged();refreshRuns();
})();
