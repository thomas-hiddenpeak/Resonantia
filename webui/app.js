// Resonantia WebUI — orchestrates training + inference via the vc_serve backend.
const API = '/api';
const $ = (id) => document.getElementById(id);

// API authentication is optional on the server. Keep a supplied bearer token only
// for this browser tab and retry the first 401 once, so the default open mode stays
// unchanged while authenticated deployments work without embedding a secret in URLs.
let authPromptPromise = null;
let memoryAuthToken = '';
function storedAuthToken() {
    try { return sessionStorage.getItem('resonantiaAuthToken') || memoryAuthToken; }
    catch (e) { return memoryAuthToken; }
}
function storeAuthToken(token) {
    memoryAuthToken = token;
    try {
        if (token) sessionStorage.setItem('resonantiaAuthToken', token);
        else sessionStorage.removeItem('resonantiaAuthToken');
    } catch (e) { /* storage may be unavailable; memory fallback is enough */ }
}
function requestAuthToken() {
    if (!authPromptPromise) {
        authPromptPromise = Promise.resolve()
            .then(() => window.prompt('此服务需要访问令牌：') || '')
            .finally(() => { authPromptPromise = null; });
    }
    return authPromptPromise;
}
async function apiFetch(url, options = {}) {
    const send = (token) => {
        const headers = new Headers(options.headers || {});
        if (token) headers.set('Authorization', 'Bearer ' + token);
        return fetch(url, { ...options, headers });
    };
    let token = storedAuthToken();
    let response = await send(token);
    if (response.status !== 401) return response;
    storeAuthToken('');
    token = await requestAuthToken();
    if (!token) return response;
    storeAuthToken(token);
    response = await send(token);
    if (response.status === 401) storeAuthToken('');
    return response;
}

// Extract a human-readable message from a non-OK response (JSON {error} or text).
async function errText(r) {
    const t = await r.text().catch(() => '');
    let m = t;
    if (t) { try { m = JSON.parse(t).error || t; } catch (e) { m = t; } }
    m = (m || ('服务器错误 ' + r.status)).trim();
    return m.length > 300 ? m.slice(-300) : m;
}
function setConn(online) {
    const b = $('connBanner'); if (b) b.style.display = online ? 'none' : 'block';
    globalThis.resonantiaStudio?.connection(online);
}

// ---------- Tabs ----------
document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
        btn.classList.add('active');
        $('panel-' + btn.dataset.tab).classList.add('active');
        globalThis.resonantiaStudio?.navigate(btn.dataset.tab);
        if (btn.dataset.tab === 'convert') loadVoices();
    });
});

// ---------- Training profiles ----------
let previousTrainingMode = null;
let knownTrainingProfiles = null;
function applyTrainingMode(reset = true) {
    const joint = $('trainMode').value === 'joint';
    const full = $('trainMode').value === 'joint-full';
    if (previousTrainingMode === 'joint-full' && !full) $('trainEpochs').value = '0';
    previousTrainingMode = $('trainMode').value;
    $('trainSteps').max = joint ? '50' : '100000';
    $('trainSteps').disabled = full;
    $('trainSeg').disabled = joint || full;
    $('trainEpochs').disabled = joint;
    const hint = $('trainJointHint'); if (hint) hint.hidden = !joint;
    const controls = $('trainFullControls'); if (controls) controls.hidden = !full;
    const resume = $('trainResume'); if (resume && !full) resume.checked = false;
    if (joint) {
        if (reset) $('trainSteps').value = '50';
        $('trainSeg').value = '32'; $('trainEpochs').value = '0';
    }
    if (full) {
        $('trainSeg').value = '32';
        if (reset) $('trainEpochs').value = '200';
    }
}
function configureTrainingProfiles(profiles) {
    knownTrainingProfiles = Array.isArray(profiles) ? profiles : [];
    const supported = Array.isArray(profiles) && profiles.includes('joint');
    const option = $('trainJointOption');
    if (option) { option.hidden = !supported; option.disabled = !supported; }
    const fullSupported = Array.isArray(profiles) && profiles.includes('joint-full');
    const fullOption = $('trainFullOption');
    if (fullOption) { fullOption.hidden = !fullSupported; fullOption.disabled = !fullSupported; }
    // Static assets may be newer than a running server. Expose the new mode
    // only after that server advertises it; preserve existing user settings.
    if ((!supported && $('trainMode').value === 'joint') || (!fullSupported && $('trainMode').value === 'joint-full')) {
        $('trainMode').value = 'gan'; applyTrainingMode(false);
        if ($('trainTierDesc')) $('trainTierDesc').textContent = '当前服务尚不支持所选训练档位，请更新服务；已切回旧版模式。';
    }
}
$('trainMode').addEventListener('change', () => applyTrainingMode());

// ---------- Presets with plain-language descriptions ----------
let PRESETS = { inference: [], training: [] };
async function loadPresets() {
    try { PRESETS = await (await fetch('presets.json')).json(); } catch (e) { return; }
    const ts = $('trainTier'); ts.innerHTML = '';
    PRESETS.training.forEach((t) => { const o = document.createElement('option'); o.value = t.id; o.textContent = t.name; ts.appendChild(o); });
    ts.onchange = () => applyTier(ts.value);
    if (PRESETS.training[1]) { ts.value = PRESETS.training[1].id; applyTier(ts.value); }
    const cs = $('convPreset'); cs.innerHTML = '';
    PRESETS.inference.forEach((p) => { const o = document.createElement('option'); o.value = p.id; o.textContent = p.name; cs.appendChild(o); });
    cs.onchange = () => applyPreset(cs.value);
    if (PRESETS.inference[1]) { cs.value = PRESETS.inference[1].id; applyPreset(cs.value); }
    applyVoicePreset();
}
function applyTier(id) {
    const t = PRESETS.training.find((x) => x.id === id); if (!t) return;
    $('trainTierDesc').textContent = t.desc;
    if (knownTrainingProfiles && !knownTrainingProfiles.includes(t.mode)) {
        $('trainTierDesc').textContent = '当前服务尚不支持此训练档位，请更新服务后使用。已切回旧版模式。';
        $('trainMode').value = 'gan'; applyTrainingMode(false); $('trainEpochs').value = '0'; return;
    }
    $('trainMode').value = t.mode; $('trainSteps').value = t.steps; $('trainSeg').value = t.seg;
    applyTrainingMode(false);
    if (t.epochs !== undefined) $('trainEpochs').value = t.epochs;
}
function applyPreset(id) {
    const p = PRESETS.inference.find((x) => x.id === id); if (!p) return;
    $('convPresetDesc').textContent = p.desc;
    const set = (el, val, out, fmt) => { $(el).value = val; if (out) $(out).textContent = fmt ? fmt(val) : val; };
    set('pitchShift', p.f0_up_key, 'pitchShiftValue');
    set('indexRate', Math.round(p.index_rate * 100), 'indexRateValue', () => p.index_rate.toFixed(2));
    set('rmsMixRate', Math.round(p.rms_mix_rate * 100), 'rmsMixRateValue', () => p.rms_mix_rate.toFixed(2));
    set('protect', Math.round(p.protect * 100), 'protectValue', () => p.protect.toFixed(2));
    set('filterRadius', p.filter_radius, 'filterRadiusValue');
    $('convParamSummary').textContent =
        `音色检索 ${p.index_rate.toFixed(2)} · 辅音保护 ${p.protect.toFixed(2)}` +
        ` · 合成音量占比 ${p.rms_mix_rate.toFixed(2)} · 平滑 ${p.filter_radius} · 变调 ${p.f0_up_key} 半音`;
    globalThis.resonantiaStudio?.render();
}
// ---------- Voice presets ----------
let lastVoicePreset = null;
function applyVoicePreset() {
    const voice = $('voiceSelect').value;
    // The voice list and presets load independently. Wait for preset data and
    // don't reset manual adjustments when the same voice list is refreshed.
    if (!PRESETS.inference.length || voice === lastVoicePreset) return;
    lastVoicePreset = voice;
    const preset = PRESETS.inference.find((p) => p.voice === voice);
    if (preset) { $('convPreset').value = preset.id; applyPreset(preset.id); }
}
$('voiceSelect').addEventListener('change', applyVoicePreset);

const CONV_SESSION = (() => {
    // A duplicated tab inherits sessionStorage, so allocate a fresh identity on
    // each page load. Uploads are explicitly reselected after reload anyway.
    let id;
    try { id = 'convert_' + crypto.randomUUID(); }
    catch { id = 'convert_' + Date.now().toString(36) + Math.random().toString(36).slice(2); }
    try {
        sessionStorage.setItem('resonantiaConvertSession', id);
    } catch { /* blocked storage does not prevent isolated uploads */ }
    return id;
})();

// ---------- Shared cascade preprocessing (upload -> apply stages, per-step audition) ----------
const STAGES = [
    { step: 'separate', icon: '🎤', name: '分离人声' },
    { step: 'deharmony', icon: '🎶', name: '去和声' },
    { step: 'dereverb', icon: '🌊', name: '去混响' },
    { step: 'deecho', icon: '🔊', name: '去回声' },
    { step: 'denoise', icon: '🔇', name: '去噪' },
];
// Installed models per stage (from /api/sep-models); drives the per-stage picker.
let SEP_MODELS = {};
function populateModelSelects() {
    document.querySelectorAll('.stage-model').forEach((sel) => {
        const opts = SEP_MODELS[sel.dataset.step] || [];
        const prev = sel.value;
        sel.innerHTML = '';
        opts.forEach((o) => {
            const op = document.createElement('option');
            op.value = o.name; op.textContent = o.label; sel.appendChild(op);
        });
        if (prev) sel.value = prev;
        sel.style.display = opts.length > 1 ? '' : 'none';  // only show when there's a choice
    });
}
async function loadSepModels() {
    try {
        const r = await apiFetch(`${API}/sep-models`);
        if (r.ok) { SEP_MODELS = await r.json(); populateModelSelects(); }
    } catch (e) { /* offline: pickers stay hidden, server uses defaults */ }
}
function pollJob(name, jobId) {
    return new Promise((resolve, reject) => {
        const tick = async () => {
            try {
                const r = await apiFetch(`${API}/train/status?name=${encodeURIComponent(name)}`);
                if (!r.ok) throw new Error(await errText(r));
                const s = await r.json();
                if (!jobId || s.job_id !== jobId || s.name !== name) throw new Error('处理任务已变化，请重新确认输入');
                setConn(true);
                if (s.running) setTimeout(tick, 1000); else resolve(s);
            } catch (e) { reject(e); }
        };
        tick();
    });
}
// Fetch the current work-set audio as a frozen blob snapshot for per-stage audition.
async function auditionSnapshot(name, audioEl, isCurrent = () => true) {
    try {
        const r = await apiFetch(`${API}/preview?name=${encodeURIComponent(name)}&t=${Date.now()}`);
        if (!r.ok) return false;
        const blob = await r.blob();
        if (!isCurrent()) return false;
        if (audioEl._url) URL.revokeObjectURL(audioEl._url);
        audioEl._url = URL.createObjectURL(blob);
        audioEl.src = audioEl._url; audioEl.style.display = 'block';
        return true;
    } catch (e) { return false; }
}
function Cascade(containerId, getName, resetBtnId) {
    const box = $(containerId);
    const state = { ready: false, busy: false, applied: [] };
    STAGES.forEach((s) => {
        const row = document.createElement('div'); row.className = 'stage'; row.dataset.step = s.step;
        const icon = document.createElement('span'); icon.className = 'stage-icon'; icon.textContent = s.icon;
        const nm = document.createElement('span'); nm.className = 'stage-name'; nm.textContent = s.name;
        const btn = document.createElement('button'); btn.className = 'btn-step'; btn.textContent = '应用'; btn.disabled = true;
        const st = document.createElement('span'); st.className = 'stage-status';
        const sel = document.createElement('select'); sel.className = 'stage-model'; sel.dataset.step = s.step; sel.style.display = 'none'; sel.title = '选择模型';
        const au = document.createElement('audio'); au.className = 'stage-audio'; au.controls = true; au.style.display = 'none';
        btn.onclick = () => apply(s, row, st, au);
        row.append(icon, nm, btn, st, sel, au);
        box.appendChild(row);
    });
    const orderEl = document.createElement('p'); orderEl.className = 'cascade-order'; box.appendChild(orderEl);
    function updateOrder() { orderEl.textContent = state.applied.length ? '实际处理顺序：' + state.applied.join(' → ') : ''; }
    function setBusy(b) {
        state.busy = b;
        box.querySelectorAll('.btn-step').forEach((x) => { x.disabled = b || !state.ready; });
        const rb = $(resetBtnId); if (rb) rb.disabled = b || !state.ready;
        globalThis.resonantiaStudio?.render();
    }
    function enable() {
        state.ready = true;
        box.querySelectorAll('.stage').forEach((r) => {
            r.classList.remove('applied');
            r.querySelector('.stage-status').textContent = '';
            const a = r.querySelector('.stage-audio'); a.style.display = 'none'; a.removeAttribute('src');
        });
        state.applied = []; updateOrder();
        setBusy(false);
    }
    async function apply(s, row, st, au) {
        const name = getName(); if (!name) { alert('请先完成上传'); return; }
        setBusy(true); st.textContent = `正在${s.name}…`;
        try {
            const fd = new FormData(); fd.append('name', name); fd.append('step', s.step);
            const sel = row.querySelector('.stage-model'); if (sel && sel.value) fd.append('model', sel.value);
            const r = await apiFetch(`${API}/step`, { method: 'POST', body: fd });
            if (!r.ok) throw new Error(await errText(r));
            const accepted = await r.json();
            const done = await pollJob(name, accepted.job_id);
            if (done.error) throw new Error('处理失败（见服务端日志）');
            state.applied.push(s.name);
            st.textContent = state.applied.length + '. 已应用 ✓'; row.classList.add('applied');
            updateOrder();
            await auditionSnapshot(name, au);
        } catch (e) { st.textContent = '失败: ' + e.message; }
        setBusy(false);
    }
    return { enable, setBusy, state };
}
const trainCascade = Cascade('trainCascade', () => $('voiceName').value.trim(), 'trainResetPre');
const convCascade = Cascade('convCascade', () => CONV_SESSION, 'convResetPre');
loadSepModels();  // populate per-stage model pickers from installed models

// ---------- Audio uploads ----------
const AUDIO_FORMATS = 'WAV、MP3、M4A/AAC、FLAC、OGG/Opus、WMA、AIFF/AIF';
function supportedAudio(file) {
    return /\.(wav|mp3|m4a|aac|flac|ogg|opus|wma|aiff|aif)$/i.test(file.name);
}
function invalidateMaterial(cascade, auditionId) {
    cascade.state.ready = false;
    cascade.setBusy(false);
    $(auditionId).style.display = 'none';
}

// ---------- Training ----------
let trainFiles = [];
const trainArea = $('trainUploadArea');
trainArea.addEventListener('click', () => $('trainFileInput').click());
trainArea.addEventListener('dragover', (e) => { e.preventDefault(); trainArea.classList.add('drag'); });
trainArea.addEventListener('dragleave', () => trainArea.classList.remove('drag'));
trainArea.addEventListener('drop', (e) => {
    e.preventDefault(); trainArea.classList.remove('drag');
    addTrainFiles(e.dataTransfer.files);
});
$('trainFileInput').addEventListener('change', (e) => addTrainFiles(e.target.files));

function addTrainFiles(list) {
    if (activeTrain || trainCascade.state.busy) return;
    const files = Array.from(list);
    const rejected = files.filter((f) => !supportedAudio(f));
    if (rejected.length) {
        alert('不支持的文件：' + rejected.map((f) => f.name).join('、') + '。支持 ' + AUDIO_FORMATS);
        return;
    }
    for (const f of files) { trainFiles.push(f); estimateDuration(f); }
    renderTrainFiles();
    ensureTrainMaterial();
}
function estimateDuration(f) {
    // Metadata only: do not expand a long compressed recording into browser PCM.
    // Browser codec support is optional; the server supplies the WAV preview.
    const audio = new Audio(), url = URL.createObjectURL(f);
    let done = false;
    const finish = () => {
        if (done) return;
        done = true; clearTimeout(timer);
        if (Number.isFinite(audio.duration) && audio.duration > 0) f._dur = audio.duration;
        audio.onloadedmetadata = audio.onerror = null;
        audio.removeAttribute('src'); audio.load(); URL.revokeObjectURL(url);
        updateDataHint();
    };
    const timer = setTimeout(finish, 10000);
    audio.preload = 'metadata'; audio.onloadedmetadata = audio.onerror = finish; audio.src = url;
}
function updateDataHint() {
    const sec = trainFiles.reduce((s, f) => s + (f._dur || 0), 0), m = sec / 60, el = $('trainDataHint');
    const unknown = trainFiles.some((f) => !f._dur);
    el.textContent = `已选择 ${trainFiles.length} 个文件，已识别时长约 ${m.toFixed(1)} 分钟` +
        (unknown ? '（部分文件时长未知，仍可上传试听）。' : m < 10 ? '，建议至少 10 分钟。' : '。请继续检查清晰度、混响与重叠人声。');
    el.style.color = m < 10 ? '' : 'var(--success-color)';
}
function renderTrainFiles() {
    const ul = $('trainFileList'); ul.innerHTML = '';
    trainFiles.forEach((f, i) => {
        const li = document.createElement('li');
        const span = document.createElement('span'); span.textContent = f.name;
        const btn = document.createElement('button'); btn.textContent = '✕';
        btn.setAttribute?.('aria-label', '移除训练素材 ' + f.name);
        btn.onclick = () => { if (activeTrain) return; trainFiles.splice(i, 1); renderTrainFiles(); ensureTrainMaterial(); };
        li.append(span, ' ', btn);
        ul.appendChild(li);
    });
    updateDataHint();
    updateTrainBtn();
}
let activeTrain = null;
function updateTrainBtn() {
    const name = $('voiceName').value.trim();
    const resume = $('trainMode').value === 'joint-full' && $('trainResume')?.checked;
    $('btnTrain').disabled = !!activeTrain || !((trainCascade.state.ready || resume) && /^[A-Za-z0-9_-]{1,64}$/.test(name));
    globalThis.resonantiaStudio?.render();
}
$('voiceName').addEventListener('input', () => { updateTrainBtn(); ensureTrainMaterial(); });
$('trainResume')?.addEventListener('change', updateTrainBtn);
$('trainMode').addEventListener('change', updateTrainBtn);

// Create/refresh the training work-set (raw+work) so the cascade can operate.
let trainMatSig = '';
let trainUploadGeneration = 0, trainUploadQueue = Promise.resolve();
function ensureTrainMaterial() {
    if ($('trainMode').value === 'joint-full' && $('trainResume')?.checked) return;
    const name = $('voiceName').value.trim();
    const files = [...trainFiles];
    const sig = name + '|' + files.map((f) => f.name + f.size + f.lastModified).join(',');
    if (sig === trainMatSig && trainCascade.state.ready) return;
    const generation = ++trainUploadGeneration;
    const current = () => generation === trainUploadGeneration;
    invalidateMaterial(trainCascade, 'trainOrigAudition'); updateTrainBtn();
    $('trainUploadStatus').textContent = '';
    $('trainRetryUpload').style.display = 'none';
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(name) || files.length === 0) return;
    $('trainUploadStatus').textContent = '正在上传并准备音频，请稍候…';
    // Serialize mutations: an older in-flight upload must finish before a newer
    // one can publish to the same server material session.
    trainUploadQueue = trainUploadQueue.catch(() => {}).then(async () => {
        if (!current()) return;
        const fd = new FormData(); fd.append('name', name);
        files.forEach((f) => fd.append('files', f, f.name));
        try {
            const r = await apiFetch(`${API}/material`, { method: 'POST', body: fd });
            setConn(true);
            if (!r.ok) throw new Error(await errText(r));
            if (!current()) return;
            const preview = await auditionSnapshot(name, $('trainOrigAudio'), current);
            if (!current()) return;
            setConn(true); trainMatSig = sig; trainCascade.enable();
            $('trainOrigAudition').style.display = preview ? 'flex' : 'none';
            $('trainUploadStatus').textContent = preview ? '上传完成，可试听或继续预处理。' : '上传完成，预览暂不可用。';
            updateTrainBtn();
        } catch (e) {
            if (!current()) return;
            if (e instanceof TypeError) setConn(false);
            $('trainUploadStatus').textContent = '上传失败: ' + e.message;
            $('trainRetryUpload').style.display = '';
            globalThis.resonantiaStudio?.render();
        }
    });
    return trainUploadQueue;
}
$('trainResetPre').onclick = () => { trainMatSig = ''; ensureTrainMaterial(); };
$('trainRetryUpload').onclick = () => ensureTrainMaterial();

// ---------- Training job controls ----------
async function restoreActiveTraining() {
    if (activeTrain) return;
    try {
        const r = await apiFetch(`${API}/train/status`);
        if (!r.ok) return;
        const s = await r.json();
        // A manual submission may have started while this request was pending.
        if (activeTrain || !s.running || s.kind !== 'train' || !s.job_id || !s.name) return;
        const job = { name: s.name, id: s.job_id, cancelSent: !!s.cancel_requested, cancelInFlight: false };
        activeTrain = job;
        updateTrainBtn();
        $('trainProgress').style.display = 'block';
        $('btnCancelTrain').disabled = true;
        $('trainStage').textContent = '正在恢复训练进度…';
        pollTrain(job);
    } catch (_) { /* A later page load can retry; never submit or cancel a job here. */ }
}

$('btnTrain').addEventListener('click', async () => {
    if (activeTrain) return;
    if (globalThis.resonantiaStudio && !(await globalThis.resonantiaStudio.confirmTraining())) return;
    if (activeTrain) return;
    const name = $('voiceName').value.trim();
    const job = { name, id: null, cancelSent: false, cancelInFlight: false };
    activeTrain = job;
    // Files were uploaded as the work-set in step 2; the cascade cleaned it.
    const fd = new FormData();
    fd.append('name', name);
    fd.append('mode', $('trainMode').value);
    const full = $('trainMode').value === 'joint-full';
    const resume = full && $('trainResume')?.checked;
    if (!full) fd.append('steps', $('trainSteps').value);
    if (full) fd.append('resume', resume ? '1' : '0');
    fd.append('seg', $('trainSeg').value);
    fd.append('epochs', $('trainEpochs').value);
    fd.append('vad', !resume && $('recVad').checked ? '1' : '0');
    updateTrainBtn();
    $('btnCancelTrain').disabled = true;
    $('trainProgress').style.display = 'block';
    $('trainStage').textContent = '提交中…';
    $('trainFill').style.width = '0%';
    try {
        const r = await apiFetch(`${API}/train`, { method: 'POST', body: fd });
        if (!r.ok) throw new Error(await errText(r));
        const accepted = await r.json();
        if (!accepted.job_id) throw new Error('服务器未返回作业标识，请核实服务状态');
        job.id = accepted.job_id;
        pollTrain(job);
    } catch (err) {
        if (activeTrain !== job) return;
        $('trainStage').textContent = '提交结果未确认: ' + err.message;
        activeTrain = null; updateTrainBtn();
    }
});

$('btnCancelTrain').addEventListener('click', async () => {
    const job = activeTrain;
    if (!job || !job.id || job.cancelSent || job.cancelInFlight) return;
    job.cancelInFlight = true;
    $('btnCancelTrain').disabled = true;
    $('trainStage').textContent = '正在请求取消…';
    const fd = new FormData(); fd.append('job_id', job.id);
    try {
        const r = await apiFetch(`${API}/train/cancel`, { method: 'POST', body: fd });
        if (!r.ok) throw new Error(await errText(r));
        const result = await r.json();
        if (activeTrain !== job) return;
        if (result.job_id !== job.id) throw new Error('取消响应的作业标识不匹配');
        job.cancelSent = true;
        $('trainStage').textContent = result.cancelled ? '已取消；已提交素材保留' : '已请求取消，等待当前阶段结束…';
        // Only the status poll retires the UI job. An acknowledgement does not
        // release the server's resources or permit this UI to start another job.
    } catch (err) {
        if (activeTrain !== job) return;
        $('trainStage').textContent = '取消未确认: ' + err.message;
    } finally {
        job.cancelInFlight = false;
    }
});

const STAGE_LABELS = {queued:'等待设备',preparing:'准备素材',preprocessing:'清理与切分素材',separating:'分离人声',training:'训练中',indexing:'构建音色检索索引',publishing:'保存候选',done:'完成',error:'失败',cancelled:'已停止'};
async function pollTrain(job) {
    if (activeTrain !== job) return;
    try {
        const r = await apiFetch(`${API}/train/status?name=${encodeURIComponent(job.name)}`);
        if (!r.ok) throw new Error(await errText(r));
        const s = await r.json();
        if (activeTrain !== job) return;
        if (s.job_id !== job.id || s.kind !== 'train') {
            $('trainStage').textContent = '当前作业已变化，无法确认本次训练结果，请核实服务状态';
            $('btnCancelTrain').disabled = true;
            activeTrain = null; updateTrainBtn(); return;
        }
        $('trainFill').style.width = s.done ? '100%' : '0%';
        $('trainLog').textContent = s.log || '';
        $('trainLog').scrollTop = $('trainLog').scrollHeight;
        $('btnCancelTrain').disabled = !s.cancellable || job.cancelSent || job.cancelInFlight || s.cancel_requested;
        if (s.running) {
            $('trainStage').textContent = s.cancel_requested || job.cancelSent ? '已请求取消，等待当前阶段结束…' :
                job.cancelInFlight ? '正在请求取消…' : s.stage === 'publishing' ? '正在保存训练结果，已无法取消…' : (STAGE_LABELS[s.stage] || s.stage);
            if (s.stage === 'training' && !s.cancel_requested && !job.cancelSent && !job.cancelInFlight) {
                const progress = [...(s.log || '').matchAll(/joint-full step=(\d+)\/(\d+) epoch=(\d+)/g)].pop();
                const cache = [...(s.log || '').matchAll(/Cache (\d+)\/(\d+) frames=/g)].pop();
                const preparing = (s.log || '').lastIndexOf('Preparing model/data identity');
                if (cache && Number(cache[2]) > 0 && (!progress || cache.index > progress.index)) {
                    $('trainStage').textContent = `准备训练特征：${cache[1]}/${cache[2]} 段`;
                    $('trainFill').style.width = Math.min(100, Math.floor(100 * Number(cache[1]) / Number(cache[2]))) + '%';
                } else if (preparing >= 0 && (!progress || preparing > progress.index)) {
                    $('trainStage').textContent = '准备训练：核验素材并载入模型…';
                    $('trainFill').style.width = '0%';
                } else if (progress && Number(progress[2]) > 0) {
                    $('trainStage').textContent = `训练中：${progress[1]}/${progress[2]} 步，已完成 ${progress[3]} 轮`;
                    $('trainFill').style.width = Math.min(100, Math.floor(100 * Number(progress[1]) / Number(progress[2]))) + '%';
                    const times = [...(s.log || '').matchAll(/train_seconds=([\d.]+)/g)].map(m=>Number(m[1])).filter(v=>Number.isFinite(v)&&v>0);
                    if (times.length) $('trainStage').textContent += ` · 最近计算耗时 ${(times.reduce((a,b)=>a+b,0)/times.length).toFixed(2)} 秒/步（不含准备与保存）`;
                }
            }
            setTimeout(() => pollTrain(job), 1500);
        } else if (s.done) {
            $('btnCancelTrain').disabled = true;
            activeTrain = null; updateTrainBtn();
            if (s.cancelled) $('trainStage').textContent = s.resumable ? '已停止；完整训练进度和固定素材已保留，可恢复' : '已取消；已提交素材保留';
            else if (s.error) $('trainStage').textContent = '训练失败，请查看日志';
            else if (s.review_required) {
                $('trainStage').textContent = '训练完成，候选已保存，等待听审；现有声线保持不变';
                globalThis.resonantiaWorkflow?.refreshRuns();
            }
            else { $('trainStage').textContent = '完成 ✓ 已生成声线「' + job.name + '」'; loadVoices(); }
        } else {
            $('trainStage').textContent = '等待作业状态…';
            setTimeout(() => pollTrain(job), 1500);
        }
        globalThis.resonantiaStudio?.trainingStatus(s);
    } catch (e) {
        if (activeTrain !== job) return;
        $('btnCancelTrain').disabled = true;
        $('trainStage').textContent = '暂时无法确认作业状态，正在重试…';
        setTimeout(() => pollTrain(job), 2500);
    }
}

// ---------- Voices ----------
async function loadVoices() {
    try {
        const response = await apiFetch(`${API}/voices`);
        if (!response.ok) throw new Error(await errText(response));
        const d = await response.json();
        configureTrainingProfiles(d.training_profiles);
        setConn(true);
        const sel = $('voiceSelect'), cur = sel.value, voices = d.voices || [];
        sel.innerHTML = '';
        if (voices.length === 0) {
            const o = document.createElement('option');
            o.value = ''; o.textContent = '无（请先在①训练声线创建）'; o.disabled = true; o.selected = true;
            sel.appendChild(o);
        } else {
            voices.forEach((v) => { const o = document.createElement('option'); o.value = v;
                o.textContent = v === 'Tom_Cybersinger' ? 'Tom Cybersinger · 正式版（200 轮，听审通过）' : v + ((d.reviewed_voices || []).includes(v) ? '（听审通过）' : '（历史声线）'); sel.appendChild(o); });
            if (cur && voices.includes(cur)) sel.value = cur;
            else if (voices.includes('Tom_Cybersinger')) sel.value = 'Tom_Cybersinger';
        }
        applyVoicePreset();
        if ($('formalRelease')) $('formalRelease').hidden = !voices.includes('Tom_Cybersinger');
        globalThis.resonantiaStudio?.voices(d);
    } catch (e) { setConn(false); }
}
$('btnRefreshVoices').addEventListener('click', loadVoices);

// ---------- Convert ----------
let convFile = null, convUrl = null;
let convRunning = false;
let convUploadGeneration = 0, convUploadQueue = Promise.resolve();
const convArea = $('convUploadArea');
convArea.addEventListener('click', () => $('convFileInput').click());
convArea.addEventListener('dragover', (e) => { e.preventDefault(); convArea.classList.add('drag'); });
convArea.addEventListener('dragleave', () => convArea.classList.remove('drag'));
convArea.addEventListener('drop', (e) => { e.preventDefault(); convArea.classList.remove('drag'); if (e.dataTransfer.files[0]) setConvFile(e.dataTransfer.files[0]); });
$('convFileInput').addEventListener('change', (e) => { if (e.target.files[0]) setConvFile(e.target.files[0]); });

function setConvFile(f) {
    if (convRunning || convCascade.state.busy) return;
    if (!supportedAudio(f)) { alert('请上传 ' + AUDIO_FORMATS); return; }
    convFile = f;
    globalThis.resonantiaStudio?.conversion('input');
    globalThis.resonantiaWorkflow?.resetPitchEdits();
    $('resultSection').style.display = 'none';
    $('convFileInfo').querySelector('.file-name').textContent = f.name;
    $('convFileInfo').style.display = 'flex';
    convArea.style.display = 'none';
    ensureConvMaterial();
}
$('btnConvRemove').addEventListener('click', () => {
    if (convRunning || convCascade.state.busy) return;
    ++convUploadGeneration;
    $('convUploadStatus').textContent = '';
    convFile = null; $('convFileInput').value = '';
    globalThis.resonantiaWorkflow?.resetPitchEdits();
    $('resultSection').style.display = 'none';
    $('convFileInfo').style.display = 'none'; convArea.style.display = 'block';
    $('convOrigAudition').style.display = 'none';
    convCascade.state.ready = false; convCascade.setBusy(false);
    $('btnConvert').disabled = true;
    globalThis.resonantiaStudio?.conversion('empty');
});
// Upload the input as a work-set (session) so the cascade + audition can operate.
function ensureConvMaterial() {
    if (!convFile || convRunning || convCascade.state.busy) return;
    const file = convFile, generation = ++convUploadGeneration;
    const current = () => generation === convUploadGeneration;
    invalidateMaterial(convCascade, 'convOrigAudition'); $('btnConvert').disabled = true;
    $('convUploadStatus').textContent = '正在上传并准备音频，请稍候…';
    globalThis.resonantiaStudio?.conversion('uploading');
    convUploadQueue = convUploadQueue.catch(() => {}).then(async () => {
        if (!current()) return;
        const fd = new FormData(); fd.append('name', CONV_SESSION); fd.append('files', file, file.name);
        fd.append('preserve_stereo', '1');
        try {
            const r = await apiFetch(`${API}/material`, { method: 'POST', body: fd });
            setConn(true);
            if (!r.ok) throw new Error(await errText(r));
            if (!current()) return;
            const preview = await auditionSnapshot(CONV_SESSION, $('convOrigAudio'), current);
            if (!current()) return;
            setConn(true); convCascade.enable();
            $('convOrigAudition').style.display = preview ? 'flex' : 'none';
            $('convUploadStatus').textContent = preview ? '上传完成，可试听或开始转换。' : '上传完成，预览暂不可用。';
            $('btnConvert').disabled = convRunning;
            globalThis.resonantiaStudio?.conversion('ready');
        } catch (e) {
            if (!current()) return;
            if (e instanceof TypeError) setConn(false);
            $('convUploadStatus').textContent = '上传失败: ' + e.message;
            globalThis.resonantiaStudio?.conversion('upload-error');
        }
    });
    return convUploadQueue;
}
$('convResetPre').onclick = () => ensureConvMaterial();

const bind = (id, out, f) => { const el = $(id); el.addEventListener('input', () => $(out).textContent = f(el.value)); };
bind('pitchShift', 'pitchShiftValue', (v) => v);
bind('indexRate', 'indexRateValue', (v) => (v / 100).toFixed(2));
bind('rmsMixRate', 'rmsMixRateValue', (v) => (v / 100).toFixed(2));
bind('protect', 'protectValue', (v) => (v / 100).toFixed(2));
bind('filterRadius', 'filterRadiusValue', (v) => v);

$('btnConvert').addEventListener('click', async () => {
    if (convRunning || convCascade.state.busy) return;
    if (!convCascade.state.ready) { alert('请先上传待转换音频'); return; }
    if (!$('voiceSelect').value && !globalThis.resonantiaWorkflow?.hasCandidate()) { alert('请先选择已发布声线或训练候选'); return; }
    try { globalThis.resonantiaWorkflow?.validate(); }
    catch (error) {
        if (globalThis.resonantiaStudio) globalThis.resonantiaStudio.validationError(error.message);
        else alert(error.message);
        return;
    }
    const inputName = convFile.name;
    const generation = convUploadGeneration;
    convRunning = true;
    convCascade.setBusy(true);
    globalThis.resonantiaWorkflow?.lock(true);
    $('btnConvert').disabled = true;
    $('convProgress').style.display = 'block';
    $('resultSection').style.display = 'none';
    globalThis.resonantiaStudio?.conversion('running');
    // Single blocking request -> honest indeterminate bar + elapsed time (no fake %).
    $('convFill').classList.add('indeterminate');
    const t0 = Date.now();
    const iv = setInterval(() => { $('convInfo').textContent = `转换中… 已用 ${Math.round((Date.now() - t0) / 1000)}s`; }, 500);
    try {
        const fd = new FormData();
        fd.append('session', CONV_SESSION);
        fd.append('voice', $('voiceSelect').value || 'base');
        fd.append('f0_up_key', $('pitchShift').value);
        fd.append('index_rate', $('indexRate').value / 100);
        fd.append('rms_mix_rate', $('rmsMixRate').value / 100);
        fd.append('protect', $('protect').value / 100);
        fd.append('filter_radius', $('filterRadius').value);
        globalThis.resonantiaWorkflow?.append(fd);
        const r = await apiFetch(`${API}/convert`, { method: 'POST', body: fd });
        setConn(true);
        clearInterval(iv);
        if (!r.ok) throw new Error(await errText(r));
        const blob = await r.blob();
        if (generation !== convUploadGeneration) throw new Error('输入已变化，请重新转换');
        if (convUrl) URL.revokeObjectURL(convUrl);
        convUrl = URL.createObjectURL(blob);
        $('convFill').classList.remove('indeterminate'); $('convFill').style.width = '100%'; $('convInfo').textContent = '完成！';
        setTimeout(() => {
            $('convProgress').style.display = 'none';
            $('resultSection').style.display = 'block';
            $('resultAudio').src = convUrl;
            $('btnDownload').onclick = () => { const a = document.createElement('a'); a.href = convUrl; a.download = 'converted_' + inputName.replace(/\.[^.]+$/, '.wav'); a.click(); };
            convRunning = false;
            convCascade.setBusy(false);
            globalThis.resonantiaWorkflow?.lock(false);
            $('btnConvert').disabled = !convCascade.state.ready;
            globalThis.resonantiaStudio?.conversion('done');
        }, 600);
    } catch (err) {
        clearInterval(iv);
        $('convFill').classList.remove('indeterminate'); $('convFill').style.width = '0%';
        $('convInfo').textContent = '转换失败: ' + err.message;
        convRunning = false;
        convCascade.setBusy(false);
        globalThis.resonantiaWorkflow?.lock(false);
        $('btnConvert').disabled = !convCascade.state.ready;
        globalThis.resonantiaStudio?.conversion('error');
    }
});

// initial
loadPresets();
loadVoices();
restoreActiveTraining();
