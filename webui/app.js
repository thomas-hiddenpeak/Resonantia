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
        authPromptPromise = Promise.resolve(window.prompt('此服务需要访问令牌：') || '');
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
}

// ---------- Tabs ----------
document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
        btn.classList.add('active');
        $('panel-' + btn.dataset.tab).classList.add('active');
        if (btn.dataset.tab === 'convert') loadVoices();
    });
});

// ---------- Presets (AINDAW-style tiers with plain-language descriptions) ----------
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
}
function applyTier(id) {
    const t = PRESETS.training.find((x) => x.id === id); if (!t) return;
    $('trainTierDesc').textContent = t.desc;
    $('trainMode').value = t.mode; $('trainSteps').value = t.steps; $('trainSeg').value = t.seg;
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
        `当前参数 → F0:${p.f0method} · 检索:${p.index_rate.toFixed(2)} · 保护:${p.protect.toFixed(2)}` +
        ` · RMS:${p.rms_mix_rate.toFixed(2)} · 平滑:${p.filter_radius} · 变调:${p.f0_up_key}`;
}
const CONV_SESSION = '__convert__';

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
function pollJob(name) {
    return new Promise((resolve) => {
        const tick = async () => {
            try {
                const s = await (await apiFetch(`${API}/train/status?name=${encodeURIComponent(name)}`)).json();
                setConn(true);
                if (s.running) setTimeout(tick, 1000); else resolve(s);
            } catch (e) { setConn(false); setTimeout(tick, 1500); }
        };
        tick();
    });
}
// Fetch the current work-set audio as a frozen blob snapshot for per-stage audition.
async function auditionSnapshot(name, audioEl) {
    try {
        const r = await apiFetch(`${API}/preview?name=${encodeURIComponent(name)}&t=${Date.now()}`);
        if (!r.ok) return false;
        const blob = await r.blob();
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
            const done = await pollJob(name);
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
    for (const f of list) if (f.name.match(/\.wav$/i)) { trainFiles.push(f); estimateDuration(f); }
    renderTrainFiles();
    ensureTrainMaterial();
}
async function estimateDuration(f) {
    try {
        const buf = await f.arrayBuffer();
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const audio = await ctx.decodeAudioData(buf);
        f._dur = audio.duration; ctx.close(); updateDataHint();
    } catch (e) { /* undecodable header; ignore */ }
}
function updateDataHint() {
    const sec = trainFiles.reduce((s, f) => s + (f._dur || 0), 0), m = sec / 60, el = $('trainDataHint');
    el.textContent = `已上传约 ${m.toFixed(1)} 分钟` + (m < 10 ? '，建议 ≥ 10 分钟以获得更好效果。' : '，数据量充足 ✓');
    el.style.color = m < 10 ? '' : 'var(--success-color)';
}
function renderTrainFiles() {
    const ul = $('trainFileList'); ul.innerHTML = '';
    trainFiles.forEach((f, i) => {
        const li = document.createElement('li');
        const span = document.createElement('span'); span.textContent = f.name;
        const btn = document.createElement('button'); btn.textContent = '✕';
        btn.onclick = () => { trainFiles.splice(i, 1); renderTrainFiles(); ensureTrainMaterial(); };
        li.append(span, ' ', btn);
        ul.appendChild(li);
    });
    updateDataHint();
    updateTrainBtn();
}
let activeTrain = null;
function updateTrainBtn() {
    const name = $('voiceName').value.trim();
    $('btnTrain').disabled = !!activeTrain || !(trainCascade.state.ready && /^[A-Za-z0-9_-]+$/.test(name));
}
$('voiceName').addEventListener('input', () => { updateTrainBtn(); ensureTrainMaterial(); });

// Create/refresh the training work-set (raw+work) so the cascade can operate.
let trainMatSig = '';
async function ensureTrainMaterial() {
    const name = $('voiceName').value.trim();
    if (!/^[A-Za-z0-9_-]+$/.test(name) || trainFiles.length === 0) return;
    const sig = name + '|' + trainFiles.map((f) => f.name + f.size).join(',');
    if (sig === trainMatSig && trainCascade.state.ready) return;
    trainMatSig = sig;
    const fd = new FormData(); fd.append('name', name);
    trainFiles.forEach((f) => fd.append('files', f, f.name));
    try {
        const r = await apiFetch(`${API}/material`, { method: 'POST', body: fd });
        if (!r.ok) throw new Error(await errText(r));
        setConn(true);
        trainCascade.enable();
        await auditionSnapshot(name, $('trainOrigAudio')); $('trainOrigAudition').style.display = 'flex';
        updateDataHint(); updateTrainBtn();
    } catch (e) { setConn(false); $('trainDataHint').textContent = '上传失败: ' + e.message; }
}
$('trainResetPre').onclick = () => { trainMatSig = ''; ensureTrainMaterial(); };

// ---------- Training job controls ----------
$('btnTrain').addEventListener('click', async () => {
    if (activeTrain) return;
    const name = $('voiceName').value.trim();
    const job = { name, id: null, cancelSent: false, cancelInFlight: false };
    activeTrain = job;
    // Files were uploaded as the work-set in step 2; the cascade cleaned it.
    const fd = new FormData();
    fd.append('name', name);
    fd.append('mode', $('trainMode').value);
    fd.append('steps', $('trainSteps').value);
    fd.append('seg', $('trainSeg').value);
    fd.append('epochs', $('trainEpochs').value);
    fd.append('vad', $('recVad').checked ? '1' : '0');
    updateTrainBtn();
    $('btnCancelTrain').disabled = true;
    $('trainProgress').style.display = 'block';
    $('trainStage').textContent = '提交中…';
    $('trainFill').style.width = '5%';
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

const STAGES_PCT = { queued: 10, preparing: 15, preprocessing: 25, separating: 30, training: 55, indexing: 90, publishing: 95, done: 100, error: 100, cancelled: 100 };
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
        $('trainFill').style.width = (STAGES_PCT[s.stage] || 50) + '%';
        $('trainLog').textContent = s.log || '';
        $('trainLog').scrollTop = $('trainLog').scrollHeight;
        $('btnCancelTrain').disabled = !s.cancellable || job.cancelSent || job.cancelInFlight || s.cancel_requested;
        if (s.running) {
            $('trainStage').textContent = s.cancel_requested || job.cancelSent ? '已请求取消，等待当前阶段结束…' :
                job.cancelInFlight ? '正在请求取消…' : s.stage === 'publishing' ? '正在保存训练结果，已无法取消…' : s.stage;
            setTimeout(() => pollTrain(job), 1500);
        } else if (s.done) {
            $('btnCancelTrain').disabled = true;
            activeTrain = null; updateTrainBtn();
            if (s.cancelled) $('trainStage').textContent = '已取消；已提交素材保留';
            else if (s.error) $('trainStage').textContent = '训练失败，请查看日志';
            else { $('trainStage').textContent = '完成 ✓ 已生成声线「' + job.name + '」'; loadVoices(); }
        } else {
            $('trainStage').textContent = '等待作业状态…';
            setTimeout(() => pollTrain(job), 1500);
        }
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
        const d = await (await apiFetch(`${API}/voices`)).json();
        setConn(true);
        const sel = $('voiceSelect'), cur = sel.value, voices = d.voices || [];
        sel.innerHTML = '';
        if (voices.length === 0) {
            const o = document.createElement('option');
            o.value = ''; o.textContent = '无（请先在①训练声线创建）'; o.disabled = true; o.selected = true;
            sel.appendChild(o);
        } else {
            voices.forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o); });
            if (cur && voices.includes(cur)) sel.value = cur;
        }
    } catch (e) { setConn(false); }
}
$('btnRefreshVoices').addEventListener('click', loadVoices);

// ---------- Convert ----------
let convFile = null, convUrl = null;
const convArea = $('convUploadArea');
convArea.addEventListener('click', () => $('convFileInput').click());
convArea.addEventListener('dragover', (e) => { e.preventDefault(); convArea.classList.add('drag'); });
convArea.addEventListener('dragleave', () => convArea.classList.remove('drag'));
convArea.addEventListener('drop', (e) => { e.preventDefault(); convArea.classList.remove('drag'); if (e.dataTransfer.files[0]) setConvFile(e.dataTransfer.files[0]); });
$('convFileInput').addEventListener('change', (e) => { if (e.target.files[0]) setConvFile(e.target.files[0]); });

function setConvFile(f) {
    if (!f.name.match(/\.wav$/i)) { alert('请上传 WAV'); return; }
    convFile = f;
    $('convFileInfo').querySelector('.file-name').textContent = f.name;
    $('convFileInfo').style.display = 'flex';
    convArea.style.display = 'none';
    ensureConvMaterial();
}
$('btnConvRemove').addEventListener('click', () => {
    convFile = null; $('convFileInput').value = '';
    $('convFileInfo').style.display = 'none'; convArea.style.display = 'block';
    $('convOrigAudition').style.display = 'none';
    convCascade.state.ready = false; convCascade.setBusy(false);
    $('btnConvert').disabled = true;
});
// Upload the input as a work-set (session) so the cascade + audition can operate.
async function ensureConvMaterial() {
    if (!convFile) return;
    const fd = new FormData(); fd.append('name', CONV_SESSION); fd.append('files', convFile, convFile.name);
    try {
        const r = await apiFetch(`${API}/material`, { method: 'POST', body: fd });
        if (!r.ok) throw new Error(await errText(r));
        setConn(true);
        convCascade.enable();
        await auditionSnapshot(CONV_SESSION, $('convOrigAudio')); $('convOrigAudition').style.display = 'flex';
        $('btnConvert').disabled = false;
    } catch (e) { setConn(false); alert('上传失败: ' + e.message); }
}
$('convResetPre').onclick = () => ensureConvMaterial();

const bind = (id, out, f) => { const el = $(id); el.addEventListener('input', () => $(out).textContent = f(el.value)); };
bind('pitchShift', 'pitchShiftValue', (v) => v);
bind('indexRate', 'indexRateValue', (v) => (v / 100).toFixed(2));
bind('rmsMixRate', 'rmsMixRateValue', (v) => (v / 100).toFixed(2));
bind('protect', 'protectValue', (v) => (v / 100).toFixed(2));
bind('filterRadius', 'filterRadiusValue', (v) => v);

$('btnConvert').addEventListener('click', async () => {
    if (!convCascade.state.ready) { alert('请先上传待转换音频'); return; }
    if (!$('voiceSelect').value) { alert('请先在①训练声线创建目标声线'); return; }
    $('btnConvert').disabled = true;
    $('convProgress').style.display = 'block';
    $('resultSection').style.display = 'none';
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
        const r = await apiFetch(`${API}/convert`, { method: 'POST', body: fd });
        clearInterval(iv);
        if (!r.ok) throw new Error(await errText(r));
        const blob = await r.blob();
        if (convUrl) URL.revokeObjectURL(convUrl);
        convUrl = URL.createObjectURL(blob);
        $('convFill').classList.remove('indeterminate'); $('convFill').style.width = '100%'; $('convInfo').textContent = '完成！';
        setTimeout(() => {
            $('convProgress').style.display = 'none';
            $('resultSection').style.display = 'block';
            $('resultAudio').src = convUrl;
            $('btnDownload').onclick = () => { const a = document.createElement('a'); a.href = convUrl; a.download = 'converted_' + convFile.name.replace(/\.[^.]+$/, '.wav'); a.click(); };
            $('btnConvert').disabled = false;
        }, 600);
    } catch (err) {
        clearInterval(iv);
        $('convFill').classList.remove('indeterminate'); $('convFill').style.width = '0%';
        $('convInfo').textContent = '转换失败: ' + err.message;
        $('btnConvert').disabled = false;
    }
});

// initial
loadPresets();
loadVoices();
