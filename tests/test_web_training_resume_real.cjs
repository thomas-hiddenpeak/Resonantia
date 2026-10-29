// Real HTTP + native CUDA joint-full training. One real 1.28s clip, three
// optimizer steps in total; pause/restart/resume, no published voice changes.
// Usage: node test_web_training_resume_real.cjs REPO BUILD NEW_ISOLATED_ROOT
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const net = require('node:net');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const { createHash } = require('node:crypto');
const [repo, build, root] = process.argv.slice(2).map(p => path.resolve(p));
if (!repo || !build || !root) throw new Error('repo, build, new isolated root required');
const id = 'real-web-training', name = 'Task_Recovery_Fixture';
const crash = process.argv[5] === '--crash';
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const full = path.join(root, 'runs', name, 'full_training');
let server, serverExit, base;
let orphan;
function childIdentity(pid) {
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8');
    const fields = stat.slice(stat.lastIndexOf(')') + 2).split(' ');
    const command = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8');
    return { pid, ppid: Number(fields[1]), start: fields[19], state: fields[0], command };
  } catch (e) { if (e.code === 'ENOENT' || e.code === 'ESRCH') return null; throw e; }
}
function ownedWorker() {
  // This Jetson kernel does not expose task/<tid>/children. Inspect PPID plus
  // the unique fixture path; only this test uses process discovery.
  for (const pid of fs.readdirSync('/proc').filter(p => /^\d+$/.test(p))) {
    const candidate = childIdentity(Number(pid));
    if (candidate?.ppid === server.pid && path.basename(candidate.command.split('\0')[0]) === 'vc_train' && candidate.command.includes(full)) return candidate;
  }
  throw new Error('owned real trainer not found before simulated crash');
}
function orphanAlive() {
  const current = orphan && childIdentity(orphan.pid);
  return current && current.start === orphan.start && current.state !== 'Z';
}
async function digest(file) {
  const hash = createHash('sha256');
  for await (const chunk of fs.createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}
async function run(program, args, log) {
  const fd = fs.openSync(log, 'a');
  const child = spawn(program, args, { stdio: ['ignore', fd, fd] }); fs.closeSync(fd);
  const timer = setTimeout(() => child.kill('SIGTERM'), 60000);
  try { await new Promise((resolve, reject) => { child.once('error', reject); child.once('exit', (code, signal) => code === 0 ? resolve() : reject(new Error(`${program}: ${code}/${signal}`))); }); }
  finally { clearTimeout(timer); }
}
async function startServer() {
  const socket = net.createServer(); await new Promise(r => socket.listen(0, '127.0.0.1', r));
  const port = socket.address().port; await new Promise(r => socket.close(r)); base = `http://127.0.0.1:${port}`;
  const fd = fs.openSync(path.join(root, 'server.log'), 'a');
  server = spawn(path.join(build, 'vc_serve'), ['--repo', root, '--webroot', path.join(repo, 'webui'), '--host', '127.0.0.1', '--port', String(port)], { detached: true, stdio: ['ignore', fd, fd] });
  fs.closeSync(fd); serverExit = new Promise(resolve => server.once('exit', resolve));
  for (let i = 0; i < 100; ++i) {
    if (server.exitCode !== null || server.signalCode !== null) throw new Error('test server exited');
    try { if ((await fetch(base + '/api/tasks')).ok) return; } catch {}
    await delay(50);
  }
  throw new Error('test server unavailable');
}
async function stopServer() {
  if (!server) return;
  const child = server; server = null;
  if (child.exitCode === null && child.signalCode === null) {
    try { process.kill(-child.pid, 'SIGTERM'); } catch (e) { if (e.code !== 'ESRCH') throw e; }
  }
  await serverExit;
}
async function post(endpoint, fields) {
  const form = new FormData(); for (const [k, v] of Object.entries(fields)) form.append(k, String(v));
  const response = await fetch(base + endpoint, { method: 'POST', body: form });
  return { status: response.status, body: await response.json() };
}
async function task() {
  const response = await fetch(base + '/api/task?id=' + id); assert.equal(response.status, 200);
  return response.json();
}
async function until(predicate, timeout = 300000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const r = await task(); if (predicate(r)) return r;
    if (['failed', 'cancelled', 'succeeded'].includes(r.state)) throw new Error(JSON.stringify(r));
    await delay(100);
  }
  throw new Error('training fixture timed out');
}
function checkpointHeader(file) {
  const fd = fs.openSync(file, 'r');
  try {
    const prefix = Buffer.alloc(8); assert.equal(fs.readSync(fd, prefix, 0, 8, 0), 8);
    const length = Number(prefix.readBigUInt64LE()); assert(length > 0 && length < 16 * 1024 * 1024);
    const bytes = Buffer.alloc(length); assert.equal(fs.readSync(fd, bytes, 0, length, 8), length);
    return JSON.parse(bytes.toString());
  } finally { fs.closeSync(fd); }
}
async function main() {
  assert(!fs.existsSync(root), 'fixture root must be new'); fs.mkdirSync(root, { recursive: true });
  fs.symlinkSync(path.join(repo, 'models'), path.join(root, 'models'), 'dir');
  fs.symlinkSync(build, path.join(root, 'build'), 'dir');
  const raw = path.join(root, 'runs', name, 'raw'); fs.mkdirSync(raw, { recursive: true });
  const work = path.join(root, 'runs', name, 'work'); fs.mkdirSync(work);
  await run('/usr/bin/ffmpeg', ['-nostdin', '-v', 'error', '-i', path.join(repo, 'tests/fixtures/speech_librispeech.wav'), '-t', '1.28', '-ar', '40000', '-ac', '1', '-c:a', 'pcm_s16le', path.join(raw, 'clip.wav')], path.join(root, 'prepare.log'));
  fs.copyFileSync(path.join(raw, 'clip.wav'), path.join(work, 'clip.wav'));
  const sourceHash = await digest(path.join(raw, 'clip.wav'));
  const formal = path.join(repo, 'runs/Tom_Cybersinger/model.safetensors'), formalHash = await digest(formal);
  await startServer(); const firstPid = server.pid;
  const accepted = await post('/api/train', { task_id: id, name, mode: 'joint-full', epochs: 3, prepared: 1 });
  assert.equal(accepted.status, 200, JSON.stringify(accepted)); assert.equal(accepted.body.job_id, id);
  const running = await until(r => r.phase === 'training' && r.completed >= 1 && r.completed < 3);
  assert.equal(running.total, 3); assert(running.actions.pause);
  if (crash) {
    orphan = ownedWorker(); server.kill('SIGKILL'); await serverExit; server = null;
    await startServer();
    assert.equal((await task()).state, 'interrupted', 'crashed execution presented as active or complete');
    assert.equal(fs.readFileSync(path.join(root, 'runs/.web_tasks', id, 'control.1'), 'utf8'), '1', 'restart did not signal old safe-stop channel');
    const deadline = Date.now() + 60000;
    while (orphanAlive() && Date.now() < deadline) await delay(100);
    assert(!orphanAlive(), 'old GPU worker did not retire'); orphan = null;
  } else assert.equal((await post('/api/task/pause', { id, attempt: 1 })).status, 202);
  const paused = await until(r => r.state === (crash ? 'interrupted' : 'paused'));
  assert.equal(paused.result_url, ''); assert(paused.actions.resume);
  assert(!fs.existsSync(path.join(full, 'result')), 'paused task published a candidate');
  const checkpoints = fs.readdirSync(path.join(full, 'checkpoints')).filter(n => /^state_step_\d{10}\.safetensors$/.test(n)).sort();
  const saved = path.join(full, 'checkpoints', checkpoints.at(-1));
  const savedStep = Number(checkpoints.at(-1).slice(11, 21)); assert(savedStep >= (crash ? 0 : 1) && savedStep <= 3);
  const keys = Object.keys(checkpointHeader(saved)); assert(keys.includes('state'));
  assert(keys.some(k => k.startsWith('G.')) && keys.some(k => k.startsWith('D.')), 'checkpoint lacks complete training tensors');
  const savedHash = await digest(saved);
  if (!crash) { await stopServer(); await startServer(); }
  assert.notEqual(server.pid, firstPid);
  assert.equal((await task()).state, crash ? 'interrupted' : 'paused');
  const resumed = await post('/api/task/resume', { id, attempt: 1 }); assert.equal(resumed.status, 202, JSON.stringify(resumed));
  assert.equal((await post('/api/task/resume', { id, attempt: 1 })).status, 202);
  assert.equal((await post('/api/task/pause', { id, attempt: 1 })).status, 409);
  const complete = await until(r => r.state === 'succeeded'); assert.equal(complete.attempt, 2);
  const response = await fetch(base + '/api/task/log?id=' + id); assert.equal(response.status, 200);
  const log = await response.text(); fs.writeFileSync(path.join(root, 'retained.log'), log);
  assert(log.includes(`Resumed step=${savedStep}`)); assert(log.includes('cache_hits=1/1'));
  assert(log.includes(`steps_this_process=${3 - savedStep}`));
  assert(fs.statSync(path.join(full, 'result/model.safetensors')).size > 1000000);
  assert(fs.statSync(path.join(full, 'result/model.index')).size > 0);
  assert(!fs.existsSync(path.join(root, 'runs', name, 'model.safetensors')), 'candidate was auto-published');
  if (fs.existsSync(saved)) assert.equal(await digest(saved), savedHash, 'retained complete checkpoint was changed');
  assert.equal(await digest(path.join(raw, 'clip.wav')), sourceHash); assert.equal(await digest(formal), formalHash);
  const evidence = { passed: true, clipSeconds: 1.28, totalSteps: 3, lastObservedStep: running.completed, checkpointStep: savedStep, resumeStep: savedStep, cacheHits: '1/1', attempt: complete.attempt, realServerRestart: true, realServerCrash: crash, previousWorkerRetired: crash, formalSha256: formalHash };
  fs.writeFileSync(path.join(root, 'evidence.json'), JSON.stringify(evidence, null, 2)); console.log(JSON.stringify(evidence));
}
main().catch(async error => {
  console.error(error.stack); process.exitCode = 1;
  if (server) try { const r = await task(); if (r.actions.pause) { await post('/api/task/pause', { id, attempt: r.attempt }); await until(t => ['paused', 'failed', 'succeeded'].includes(t.state), 60000); } } catch (e) { console.error('fixture safe-stop:', e.message); }
}).finally(async () => {
  await stopServer();
  // On test failure only: never leave our isolated orphan computing. Verify
  // process start time before signalling a PID; no production process matches.
  if (orphanAlive()) process.kill(orphan.pid, 'SIGTERM');
});
