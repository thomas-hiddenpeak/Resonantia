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
const REC_CHAINS = {
    songDefault: '带伴奏歌曲 → 预处理链:分离人声(MelBand-RoFormer) → 去混响(MelBand-RoFormer) → 切片 → 训练。✓ 均 SOTA 内置自动。 ✓ 去和声(卡拉OK)已内置。',
    vocalReverb: '纯人声(含混响) → 预处理链:去混响(MelBand-RoFormer,内置) → 切片 → 训练。✓',
    vocalClean: '纯人声(干净) → 预处理链:切片 → 训练。✓ 当前即可直接训练。'
};
function updateRecHint() {
    const isSong = $('recContent').value === 'song';
    $('reverbGroup').style.display = isSong ? 'none' : '';  // song implies reverb+harmony
    $('recTypeHint').textContent = isSong
        ? REC_CHAINS.songDefault
        : ($('recReverb').checked ? REC_CHAINS.vocalReverb : REC_CHAINS.vocalClean);
}
$('recContent').addEventListener('change', updateRecHint);
$('recReverb').addEventListener('change', updateRecHint);

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
        btn.onclick = () => { trainFiles.splice(i, 1); renderTrainFiles(); };
        li.append(span, ' ', btn);
        ul.appendChild(li);
    });
    updateDataHint();
    updateTrainBtn();
}
function updateTrainBtn() {
    const name = $('voiceName').value.trim();
    const ok = trainFiles.length > 0 && /^[A-Za-z0-9_-]+$/.test(name);
    $('btnTrain').disabled = !ok;
    if ($('btnUploadMaterial')) $('btnUploadMaterial').disabled = !ok;
}
$('voiceName').addEventListener('input', updateTrainBtn);

// ---------- Per-step preprocessing ----------
let appliedSteps = [];
const STEP_NAMES = { separate: '分离人声', deharmony: '去和声', dereverb: '去混响', deecho: '去回声', denoise: '去噪' };
function renderChips() {
    $('stepChips').textContent = appliedSteps.length ? appliedSteps.map((s) => STEP_NAMES[s]).join(' → ') : '（无）';
}
function setStepBusy(busy) {
    document.querySelectorAll('.btn-step').forEach((b) => b.disabled = busy);
    $('btnUploadMaterial').disabled = busy;
    $('btnPreview').disabled = busy;
}
if ($('btnUploadMaterial')) $('btnUploadMaterial').addEventListener('click', async () => {
    const name = $('voiceName').value.trim();
    const fd = new FormData();
    fd.append('name', name);
    trainFiles.forEach((f) => fd.append('files', f, f.name));
    $('btnUploadMaterial').disabled = true;
    $('btnUploadMaterial').textContent = '上传中…';
    try {
        const r = await fetch(`${API}/material`, { method: 'POST', body: fd });
        if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.error || r.status); }
        appliedSteps = []; renderChips();
        $('stepControls').style.display = 'block';
        $('btnUploadMaterial').textContent = '✓ 素材已上传（可重新上传覆盖）';
        $('btnUploadMaterial').disabled = false;
    } catch (err) {
        $('btnUploadMaterial').textContent = '① 上传素材（失败: ' + err.message + '）';
        $('btnUploadMaterial').disabled = false;
    }
});
document.querySelectorAll('.btn-step').forEach((btn) => btn.addEventListener('click', async () => {
    const name = $('voiceName').value.trim(), step = btn.dataset.step;
    const fd = new FormData(); fd.append('name', name); fd.append('step', step);
    setStepBusy(true);
    $('stepStatus').firstChild.textContent = `正在${STEP_NAMES[step]}… `;
    try {
        const r = await fetch(`${API}/step`, { method: 'POST', body: fd });
        if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.error || r.status); }
        await pollStep(name, step);
    } catch (err) {
        $('stepStatus').firstChild.textContent = '失败: ' + err.message + ' ';
        setStepBusy(false);
    }
}));
async function pollStep(name, step) {
    const r = await fetch(`${API}/train/status?name=${encodeURIComponent(name)}`);
    const s = await r.json();
    if (s.running) { setTimeout(() => pollStep(name, step), 1200); return; }
    setStepBusy(false);
    if (s.error) { $('stepStatus').firstChild.textContent = STEP_NAMES[step] + '失败（见日志）'; return; }
    appliedSteps.push(step); renderChips();
    $('stepStatus').firstChild.textContent = '已应用：';
}
if ($('btnPreview')) $('btnPreview').addEventListener('click', () => {
    const name = $('voiceName').value.trim(), a = $('previewAudio');
    a.src = `${API}/preview?name=${encodeURIComponent(name)}&t=${Date.now()}`;
    a.style.display = 'block'; a.play().catch(() => {});
});

$('btnTrain').addEventListener('click', async () => {
    const name = $('voiceName').value.trim();
    const fd = new FormData();
    fd.append('name', name);
    fd.append('mode', $('trainMode').value);
    fd.append('steps', $('trainSteps').value);
    fd.append('seg', $('trainSeg').value);
    fd.append('epochs', $('trainEpochs').value);
    fd.append('separate', $('recContent').value === 'song' ? '1' : '0');
    fd.append('dereverb', ($('recContent').value === 'song' || $('recReverb').checked) ? '1' : '0');
    fd.append('deecho', $('recEcho') && $('recEcho').checked ? '1' : '0');
    fd.append('deharmony', $('recContent').value === 'song' ? '1' : '0');
    fd.append('denoise', $('recDenoise') && $('recDenoise').checked ? '1' : '0');
    fd.append('vad', $('recVad') && $('recVad').checked ? '1' : '0');
    trainFiles.forEach((f) => fd.append('files', f, f.name));

    $('btnTrain').disabled = true;
    $('trainProgress').style.display = 'block';
    $('trainStage').textContent = '提交中…';
    $('trainFill').style.width = '5%';
    try {
        const r = await fetch(`${API}/train`, { method: 'POST', body: fd });
        if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.error || r.status); }
        pollTrain(name);
    } catch (err) {
        $('trainStage').textContent = '失败: ' + err.message;
        $('btnTrain').disabled = false;
    }
});

const STAGES = { queued: 10, preprocessing: 25, training: 55, indexing: 90, done: 100, error: 100 };
async function pollTrain(name) {
    try {
        const r = await fetch(`${API}/train/status?name=${encodeURIComponent(name)}`);
        const s = await r.json();
        $('trainStage').textContent = s.stage;
        $('trainFill').style.width = (STAGES[s.stage] || 50) + '%';
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
    $('btnConvert').disabled = false;
    if ($('btnPpPreview')) $('btnPpPreview').disabled = false;
}
$('btnConvRemove').addEventListener('click', () => {
    convFile = null; $('convFileInput').value = '';
    $('convFileInfo').style.display = 'none'; convArea.style.display = 'block';
    $('btnConvert').disabled = true;
    if ($('btnPpPreview')) $('btnPpPreview').disabled = true;
});

// ---------- Convert-input preprocessing (inference-stage) ----------
const ppFlags = () => ({
    pp_separate: $('ppSeparate') && $('ppSeparate').checked ? '1' : '0',
    pp_deharmony: $('ppDeharmony') && $('ppDeharmony').checked ? '1' : '0',
    pp_dereverb: $('ppDereverb') && $('ppDereverb').checked ? '1' : '0',
    pp_deecho: $('ppDeecho') && $('ppDeecho').checked ? '1' : '0',
    pp_denoise: $('ppDenoise') && $('ppDenoise').checked ? '1' : '0',
});
if ($('btnPpPreview')) $('btnPpPreview').addEventListener('click', async () => {
    if (!convFile) return;
    const btn = $('btnPpPreview'), a = $('ppPreviewAudio');
    const fl = ppFlags();
    if (fl.pp_separate + fl.pp_deharmony + fl.pp_dereverb + fl.pp_deecho + fl.pp_denoise === '00000') { alert('请先勾选至少一个预处理步骤'); return; }
    btn.disabled = true; btn.textContent = '处理中…（分离/去混响较慢，请稍候）';
    try {
        const fd = new FormData();
        fd.append('audio', convFile, convFile.name);
        Object.entries(fl).forEach(([k, v]) => fd.append(k, v));
        const r = await fetch(`${API}/preprocess`, { method: 'POST', body: fd });
        if (!r.ok) throw new Error(await errText(r));
        const blob = await r.blob();
        a.src = URL.createObjectURL(blob); a.style.display = 'block'; a.play().catch(() => {});
        btn.textContent = '▶ 试听预处理后音频';
    } catch (err) {
        btn.textContent = '试听失败: ' + err.message;
    }
    btn.disabled = false;
});

const bind = (id, out, f) => { const el = $(id); el.addEventListener('input', () => $(out).textContent = f(el.value)); };
bind('pitchShift', 'pitchShiftValue', (v) => v);
bind('indexRate', 'indexRateValue', (v) => (v / 100).toFixed(2));
bind('rmsMixRate', 'rmsMixRateValue', (v) => (v / 100).toFixed(2));
bind('protect', 'protectValue', (v) => (v / 100).toFixed(2));
bind('filterRadius', 'filterRadiusValue', (v) => v);

$('btnConvert').addEventListener('click', async () => {
    if (!convFile) return;
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
        fd.append('audio', convFile, convFile.name);
        fd.append('voice', $('voiceSelect').value || 'base');
        fd.append('f0_up_key', $('pitchShift').value);
        fd.append('index_rate', $('indexRate').value / 100);
        fd.append('rms_mix_rate', $('rmsMixRate').value / 100);
        fd.append('protect', $('protect').value / 100);
        fd.append('filter_radius', $('filterRadius').value);
        Object.entries(ppFlags()).forEach(([k, v]) => fd.append(k, v));
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
updateRecHint();
