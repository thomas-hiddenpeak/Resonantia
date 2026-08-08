// Resonantia WebUI — orchestrates training + inference via the vc_serve backend.
const API = '/api';
const $ = (id) => document.getElementById(id);

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
function pollJob(name) {
    return new Promise((resolve) => {
        const tick = async () => {
            try {
                const s = await (await fetch(`${API}/train/status?name=${encodeURIComponent(name)}`)).json();
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
        const r = await fetch(`${API}/preview?name=${encodeURIComponent(name)}&t=${Date.now()}`);
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
    const state = { ready: false, busy: false };
    STAGES.forEach((s) => {
        const row = document.createElement('div'); row.className = 'stage'; row.dataset.step = s.step;
        const icon = document.createElement('span'); icon.className = 'stage-icon'; icon.textContent = s.icon;
        const nm = document.createElement('span'); nm.className = 'stage-name'; nm.textContent = s.name;
        const btn = document.createElement('button'); btn.className = 'btn-step'; btn.textContent = '应用'; btn.disabled = true;
        const st = document.createElement('span'); st.className = 'stage-status';
        const au = document.createElement('audio'); au.className = 'stage-audio'; au.controls = true; au.style.display = 'none';
        btn.onclick = () => apply(s, row, st, au);
        row.append(icon, nm, btn, st, au);
        box.appendChild(row);
    });
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
        setBusy(false);
    }
    async function apply(s, row, st, au) {
        const name = getName(); if (!name) { alert('请先完成上传'); return; }
        setBusy(true); st.textContent = `正在${s.name}…`;
        try {
            const fd = new FormData(); fd.append('name', name); fd.append('step', s.step);
            const r = await fetch(`${API}/step`, { method: 'POST', body: fd });
            if (!r.ok) throw new Error(await errText(r));
            const done = await pollJob(name);
            if (done.error) throw new Error('处理失败（见服务端日志）');
            st.textContent = '已应用 ✓'; row.classList.add('applied');
            await auditionSnapshot(name, au);
        } catch (e) { st.textContent = '失败: ' + e.message; }
        setBusy(false);
    }
    return { enable, setBusy, state };
}
const trainCascade = Cascade('trainCascade', () => $('voiceName').value.trim(), 'trainResetPre');
const convCascade = Cascade('convCascade', () => CONV_SESSION, 'convResetPre');

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
    for (const f of list) if (f.name.match(/\.(wav|flac)$/i)) { trainFiles.push(f); estimateDuration(f); }
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
function updateTrainBtn() {
    const name = $('voiceName').value.trim();
    $('btnTrain').disabled = !(trainCascade.state.ready && /^[A-Za-z0-9_-]+$/.test(name));
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
        const r = await fetch(`${API}/material`, { method: 'POST', body: fd });
        if (!r.ok) throw new Error(await errText(r));
        setConn(true);
        trainCascade.enable();
        await auditionSnapshot(name, $('trainOrigAudio')); $('trainOrigAudition').style.display = 'flex';
        updateDataHint(); updateTrainBtn();
    } catch (e) { setConn(false); $('trainDataHint').textContent = '上传失败: ' + e.message; }
}
$('trainResetPre').onclick = () => { trainMatSig = ''; ensureTrainMaterial(); };

$('btnTrain').addEventListener('click', async () => {
    const name = $('voiceName').value.trim();
    // Files were uploaded as the work-set in step 2; the cascade cleaned it.
    const fd = new FormData();
    fd.append('name', name);
    fd.append('mode', $('trainMode').value);
    fd.append('steps', $('trainSteps').value);
    fd.append('seg', $('trainSeg').value);
    fd.append('epochs', $('trainEpochs').value);
    fd.append('vad', $('recVad').checked ? '1' : '0');
    $('btnTrain').disabled = true;
    $('trainProgress').style.display = 'block';
    $('trainStage').textContent = '提交中…';
    $('trainFill').style.width = '5%';
    try {
        const r = await fetch(`${API}/train`, { method: 'POST', body: fd });
        if (!r.ok) throw new Error(await errText(r));
        pollTrain(name);
    } catch (err) {
        $('trainStage').textContent = '失败: ' + err.message;
        $('btnTrain').disabled = false;
    }
});

const STAGES_PCT = { queued: 10, preprocessing: 25, separating: 30, training: 55, indexing: 90, done: 100, error: 100 };
async function pollTrain(name) {
    try {
        const r = await fetch(`${API}/train/status?name=${encodeURIComponent(name)}`);
        const s = await r.json();
        $('trainStage').textContent = s.stage;
        $('trainFill').style.width = (STAGES_PCT[s.stage] || 50) + '%';
        $('trainLog').textContent = s.log || '';
        $('trainLog').scrollTop = $('trainLog').scrollHeight;
        if (s.running || (!s.done && s.stage !== 'idle')) {
            setTimeout(() => pollTrain(name), 1500);
        } else {
            $('btnTrain').disabled = false;
            if (!s.error) { $('trainStage').textContent = '完成 ✓ 已生成声线「' + name + '」'; loadVoices(); }
        }
    } catch (e) {
        setTimeout(() => pollTrain(name), 2500);
    }
}

// ---------- Voices ----------
async function loadVoices() {
    try {
        const d = await (await fetch(`${API}/voices`)).json();
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
    if (!f.name.match(/\.(wav|flac)$/i)) { alert('请上传 WAV 或 FLAC'); return; }
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
        const r = await fetch(`${API}/material`, { method: 'POST', body: fd });
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
        const r = await fetch(`${API}/convert`, { method: 'POST', body: fd });
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
