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
    const { quietAuth = false, ...fetchOptions } = options;
    const send = (token) => {
        const headers = new Headers(options.headers || {});
        if (token) headers.set('Authorization', 'Bearer ' + token);
        return fetch(url, { ...fetchOptions, headers });
    };
    let token = storedAuthToken();
    let response = await send(token);
    if (response.status !== 401) return response;
    const latest = storedAuthToken();
    if (latest && latest !== token) {
        response = await send(latest);
        if (response.status === 401 && storedAuthToken() === latest) storeAuthToken('');
        return response;
    }
    if (storedAuthToken() === token) storeAuthToken('');
    if (quietAuth) return response;
    token = await requestAuthToken();
    if (!token) return response;
    storeAuthToken(token);
    response = await send(token);
    if (response.status === 401 && storedAuthToken() === token) storeAuthToken('');
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
    const automatic = $('trainAutomaticCleanup'); if (automatic) automatic.hidden = !full;
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
    if (PRESETS.inference[1] && !convSource) { cs.value = PRESETS.inference[1].id; applyPreset(cs.value); }
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
                const response=await apiFetch(API+'/task?id='+encodeURIComponent(jobId),{quietAuth:true});
                if (!response.ok) throw new Error(await errText(response));
                const task=await response.json();
                if (task.id!==jobId || task.name!==name) {reject(new Error('处理任务标识不匹配'));return;}
                setConn(true);
                if (['queued','running','pause_requested'].includes(task.state)) setTimeout(tick,1500);
                else resolve({error:task.state!=='succeeded',state:task.state,detail:task.error});
            } catch(error) {setConn(false);setTimeout(tick,2500);}
        };tick();
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
        if (globalThis.resonantiaTasks?.isBusy()) return;
        const name = getName(); if (!name) { alert('请先完成上传'); return; }
        setBusy(true); st.textContent = `正在${s.name}…`;
        try {
            const fd = new FormData(); fd.append('name', name); fd.append('step', s.step);
            const sel = row.querySelector('.stage-model'); if (sel && sel.value) fd.append('model', sel.value);
            const r = await apiFetch(`${API}/step`, { method: 'POST', body: fd });
            if (!r.ok) throw new Error(await errText(r));
            const accepted = await r.json();
            globalThis.resonantiaTasks?.open(accepted.job_id);
            const done = await pollJob(name, accepted.job_id);
            if (done.error) throw new Error(done.detail || '任务未完成，请在任务中心查看记录');
            state.applied.push(s.name);
            st.textContent = state.applied.length + '. 已应用 ✓'; row.classList.add('applied');
            updateOrder();
            await auditionSnapshot(name, au);
        } catch (e) { st.textContent = '状态需核实: ' + e.message; globalThis.resonantiaTasks?.refresh(); }
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
$('btnTrain').addEventListener('click', async () => {
    if (activeTrain || globalThis.resonantiaTasks?.isBusy()) return;
    if (globalThis.resonantiaStudio && !(await globalThis.resonantiaStudio.confirmTraining())) return;
    if (activeTrain || globalThis.resonantiaTasks?.isBusy()) return;
    const name = $('voiceName').value.trim(), full = $('trainMode').value === 'joint-full';
    const resume = full && $('trainResume')?.checked;
    const fd = new FormData(); fd.append('name', name); fd.append('mode', $('trainMode').value);
    if (!full) fd.append('steps', $('trainSteps').value);
    if (full) fd.append('resume', resume ? '1' : '0');
    fd.append('seg', $('trainSeg').value); fd.append('epochs', $('trainEpochs').value);
    fd.append('vad', !resume && $('recVad').checked ? '1' : '0');
    for (const [field,id] of [['separate','trainAutoSeparate'],['deharmony','trainAutoDeharmony'],['dereverb','trainAutoDereverb']])
        fd.append(field,full && !resume && !trainCascade.state.applied?.length && $(id)?.checked ? '1' : '0');
    activeTrain = {name, id:null}; updateTrainBtn();
    $('trainProgress').style.display = 'block'; $('btnCancelTrain').hidden = true;
    $('trainStage').textContent = '正在提交，进度与结果将在任务中心保留。';
    try {
        await globalThis.resonantiaTasks.submit(API+'/train',fd,'training');
        $('trainStage').textContent = '请在任务中心查看状态，可以关闭本页面。';
    } catch(error) { $('trainStage').textContent = '提交未受理：'+error.message; }
    finally { activeTrain=null;updateTrainBtn(); }
});

// ---------- Voices ----------
async function loadVoices() {
    try {
        const response = await apiFetch(`${API}/voices`);
        if (!response.ok) throw new Error(await errText(response));
        const d = await response.json();
        configureTrainingProfiles(d.training_profiles);
        setConn(true);
        const sel = $('voiceSelect'), cur = sel.value, voices = [...(d.voices || [])];
        if(convSource?.parameters?.voice&&!voices.includes(convSource.parameters.voice))voices.push(convSource.parameters.voice);
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
let convSource = null;
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
    convSource = null; $('revisionNotice').hidden=true;
    convFile = f;globalThis.resonantiaWorkflow?.lock(false);
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
    convSource=null;$('revisionNotice').hidden=true;
    convFile = null; $('convFileInput').value = '';
    globalThis.resonantiaWorkflow?.resetPitchEdits();
    $('resultSection').style.display = 'none';
    $('convFileInfo').style.display = 'none'; convArea.style.display = 'block';
    $('convOrigAudition').style.display = 'none';
    convCascade.state.ready = false; convCascade.setBusy(false);globalThis.resonantiaWorkflow?.lock(false);
    $('btnConvert').disabled = true;
    globalThis.resonantiaStudio?.conversion('empty');
});
// Upload the input as a work-set (session) so the cascade + audition can operate.
function ensureConvMaterial() {
    if (!convFile || convSource || convRunning || convCascade.state.busy) return;
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

let localConversion = null;
function observeLocalConversion(record) {
    const local = localConversion;
    if (!local || record.id !== local.id || local.generation !== convUploadGeneration) return;
    const running = ['queued','running','pause_requested'].includes(record.state);
    convRunning = running; convCascade.setBusy(running); globalThis.resonantiaWorkflow?.lock(running);
    if (running) {
        $('convInfo').textContent='任务在后台运行。请到任务中心查看阶段、进度与日志。';
    } else if (record.state === 'succeeded' && !local.completed) {
        local.completed=true; $('convProgress').style.display='none';
        if(record.actions?.revise){convSource=record;showRevision();globalThis.resonantiaWorkflow?.lock(false);}
        if (!storedAuthToken()) {
            const url=API+'/task/result?id='+encodeURIComponent(record.id);
            $('resultSection').style.display='block';$('resultAudio').src=url;
            $('btnDownload').onclick=()=>{const link=document.createElement('a');link.href=url;link.download='converted_'+local.name.replace(/\.[^.]+$/,'.wav');link.click();};
            globalThis.resonantiaStudio?.conversion('done');
        } else {
            $('convProgress').style.display='block';$('convInfo').textContent='完成。请在任务中心载入试听与下载。';
        }
    } else if (record.state === 'failed') {
        $('convInfo').textContent='转换失败：'+record.error+'。请到任务中心查看日志或继续此任务。';
        globalThis.resonantiaStudio?.conversion('error');
    } else if (!local.completed) {
        $('convInfo').textContent='任务已停止运行，状态和保存点在任务中心保留。';
        globalThis.resonantiaStudio?.conversion('ready');
    }
    globalThis.resonantiaStudio?.render();
}
globalThis.resonantiaTasks?.observe(observeLocalConversion);

function showRevision() {
    $('revisionNotice').hidden=!convSource;
    if(convSource)$('revisionDescription').textContent='正在调整：'+convSource.name+'。原任务和结果保留，新版会单独保存。';
    $('revisionParent').onclick=()=>{if(convSource)globalThis.resonantiaTasks.open(convSource.id);};
}
globalThis.resonantiaReviseAudio = async function(record) {
    if(convRunning||convCascade.state.busy||globalThis.resonantiaTasks.isBusy())throw new Error('请等待当前处理完成');
    if(!record.actions?.revise||!['convert','preprocess'].includes(record.kind)||record.state!=='succeeded')throw new Error('此任务不能调整参数');
    const response=await apiFetch(API+'/task?id='+encodeURIComponent(record.id));
    if(!response.ok)throw new Error(await errText(response));
    const source=await response.json();if(source.id!==record.id||!source.actions?.revise)throw new Error('原任务状态已变化，请刷新');
    const generation=++convUploadGeneration;convSource=source;convFile={name:source.name};localConversion=null;
    $('convFileInput').value='';$('convFileInfo').querySelector('.file-name').textContent=source.name;
    $('convFileInfo').style.display='flex';convArea.style.display='none';$('resultSection').style.display='none';$('convProgress').style.display='none';
    $('nav-convert').click();await loadVoices();if(generation!==convUploadGeneration)return;
    const p=source.parameters;$('voiceSelect').value=p.voice;lastVoicePreset=p.voice;
    for(const [key,id,scale] of [['f0_up_key','pitchShift',1],['index_rate','indexRate',100],['rms_mix_rate','rmsMixRate',100],['protect','protect',100],['filter_radius','filterRadius',1]]) {
        const el=$(id),value=Number(p[key])*scale;
        if(id==='filterRadius'&&![...el.options].some(o=>o.value===String(value))){const option=document.createElement('option');option.value=String(value);option.textContent=value+'（原任务）';el.appendChild(option);}
        if(id==='pitchShift'){el.min='-48';el.max='48';}
        if(el.type==='range'&&scale===100)el.step=Number.isInteger(value)?'1':'any';
        el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));
    }
    globalThis.resonantiaWorkflow.restore(p);convCascade.state.applied=[];convCascade.state.ready=true;convCascade.setBusy(false);
    showRevision();globalThis.resonantiaWorkflow.lock(false);
    $('convUploadStatus').textContent='已载入原任务素材与设置，无需重新上传。';
    const audio=$('convOrigAudio');audio.pause();if(audio._url)URL.revokeObjectURL(audio._url);audio._url='';audio.removeAttribute('src');
    const url=API+'/task/source?id='+encodeURIComponent(source.id);
    let available=true;
    try {
        if(!storedAuthToken())audio.src=url;
        else {
            const r=await apiFetch(url);if(!r.ok)throw new Error(await errText(r));
            const size=Number(r.headers.get('Content-Length'));if(!Number.isSafeInteger(size)||size<=0||size>128*1024*1024){await r.body?.cancel();throw new Error('原音频超过页面试听上限');}
            const reader=r.body.getReader(),chunks=[];let total=0;
            for(;;){const {done,value}=await reader.read();if(done)break;total+=value.byteLength;if(total>size){await reader.cancel();throw new Error('音频长度不一致');}chunks.push(value);}
            if(total!==size)throw new Error('原音频下载不完整');
            if(generation!==convUploadGeneration)return;
            audio._url=URL.createObjectURL(new Blob(chunks,{type:'audio/wav'}));audio.src=audio._url;
        }
    }catch(error){available=false;$('convUploadStatus').textContent='设置已载入，原音频试听暂不可用：'+error.message;}
    if(generation!==convUploadGeneration)return;
    $('convOrigAudition').style.display=available?'flex':'none';audio.style.display='block';
    globalThis.resonantiaStudio?.conversion('ready');
};

$('btnConvert').addEventListener('click', async () => {
    if (convRunning || convCascade.state.busy || globalThis.resonantiaTasks?.isBusy()) return;
    if (!convCascade.state.ready) { alert('请先上传待转换音频'); return; }
    if (!$('voiceSelect').value && !globalThis.resonantiaWorkflow?.hasCandidate()) { alert('请先选择已发布声线或训练候选'); return; }
    try { globalThis.resonantiaWorkflow?.validate(); }
    catch(error) { globalThis.resonantiaStudio?.validationError(error.message); return; }
    const fd = new FormData(); fd.append(convSource?'source_task':'session',convSource?convSource.id:CONV_SESSION); fd.append('voice', $('voiceSelect').value || 'base');
    fd.append('f0_up_key', $('pitchShift').value); fd.append('index_rate', $('indexRate').value / 100);
    fd.append('rms_mix_rate', $('rmsMixRate').value / 100); fd.append('protect', $('protect').value / 100);
    fd.append('filter_radius', $('filterRadius').value); globalThis.resonantiaWorkflow?.append(fd);
    const local = {name:convFile.name,generation:convUploadGeneration,id:null,completed:false};
    localConversion=local;
    $('resultSection').style.display='none';
    convRunning=true;convCascade.setBusy(true);globalThis.resonantiaWorkflow?.lock(true);
    $('convProgress').style.display='block';$('convInfo').textContent='正在提交后台任务…';
    globalThis.resonantiaStudio?.conversion('running');
    try {
        local.id=await globalThis.resonantiaTasks.submit(API+'/tasks/convert',fd,'convert');
        $('convInfo').textContent='请在任务中心查看进度、试听和下载，可以关闭此页面。';
        globalThis.resonantiaStudio?.conversion('ready');
    } catch(error) {
        $('convInfo').textContent='提交未受理：'+error.message;globalThis.resonantiaStudio?.conversion('error');
    } finally {
        convRunning=false;convCascade.setBusy(false);globalThis.resonantiaWorkflow?.lock(false);
        $('btnConvert').disabled=!convCascade.state.ready;
        const record=globalThis.resonantiaTasks.get(local.id);if(record)observeLocalConversion(record);
        globalThis.resonantiaStudio?.render();
    }
});

$('btnPreprocess').addEventListener('click', async () => {
    if (convRunning || convCascade.state.busy || !convCascade.state.ready || globalThis.resonantiaTasks.isBusy()) return;
    const workflow=$('inputKind').value;
    if (workflow==='manual' || (workflow==='dry'&&!$('conversionDereverb').checked)) return;
    const fd=new FormData();fd.append(convSource?'source_task':'session',convSource?convSource.id:CONV_SESSION);fd.append('workflow',workflow);fd.append('output_kind','vocal');
    fd.append('pp_dereverb',$('conversionDereverb').checked?'1':'0');
    try {await globalThis.resonantiaTasks.submit(API+'/tasks/preprocess',fd,'preprocess');}
    catch(error){$('convProgress').style.display='block';$('convInfo').textContent='清理未受理：'+error.message;}
    finally {globalThis.resonantiaStudio?.render();}
});

// initial: task restoration is owned by tasks.js, including terminal history.
loadPresets();
loadVoices();
