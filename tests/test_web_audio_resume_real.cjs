// Asset-enabled integration: real HTTP server/child processes, Tom G and 12s of
// real speech. No browser mocks, long training, or full-song conversion.
// Usage: node test_web_audio_resume_real.cjs REPO BUILD ISOLATED_ROOT
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const net = require('node:net');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const { createHash } = require('node:crypto');
const [repo, build, root] = process.argv.slice(2).map(p => path.resolve(p));
if (!repo || !build || !root) throw new Error('repo, build, isolated root required');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const digest = p => createHash('sha256').update(fs.readFileSync(p)).digest('hex');
async function run(program, args, log, timeout = 180000) {
  const fd = fs.openSync(log, 'a');
  const child = spawn(program, args, { stdio: ['ignore', fd, fd] }); fs.closeSync(fd);
  const timer = setTimeout(() => child.kill('SIGTERM'), timeout);
  try { await new Promise((resolve, reject) => { child.once('error', reject); child.once('exit', (code, signal) => code === 0 ? resolve() : reject(new Error(`${program} exited ${code}/${signal}; ${log}`))); }); }
  finally { clearTimeout(timer); }
}
async function freePort() { const s = net.createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }
let server, base, serverExit;
async function startServer() {
  const port = await freePort(); base = `http://127.0.0.1:${port}`;
  const fd = fs.openSync(path.join(root, 'server.log'), 'a');
  server = spawn(path.join(build, 'vc_serve'), ['--repo', root, '--webroot', path.join(repo, 'webui'), '--host', '127.0.0.1', '--port', String(port)], { detached: true, stdio: ['ignore', fd, fd] });
  fs.closeSync(fd); serverExit = new Promise(resolve => server.once('exit', resolve));
  for (let i = 0; i < 100; i++) {
    if (server.exitCode !== null) throw new Error('isolated server exited before ready');
    try { const r = await fetch(base + '/api/tasks'); if (r.ok) return; } catch {}
    await delay(50);
  }
  throw new Error('isolated server not ready');
}
async function stopServer() {
  if (!server) return;
  // Only the isolated fixture process group, never the production service.
  const child = server; server = null;
  if (child.exitCode === null) process.kill(-child.pid, 'SIGTERM');
  await serverExit;
}
async function post(endpoint, fields) {
  const form = new FormData(); for (const [k, v] of Object.entries(fields)) form.append(k, String(v));
  const r = await fetch(base + endpoint, { method: 'POST', body: form });
  return { status: r.status, body: await r.json() };
}
const id = 'real-web-resume-tom';
async function task() { const r = await fetch(base + '/api/task?id=' + id); assert.equal(r.status, 200); return r.json(); }
async function waitState(states, timeout = 180000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) { const r = await task(); if (states.includes(r.state)) return r; if (r.state === 'failed') throw new Error(JSON.stringify(r)); await delay(100); }
  throw new Error('task state timed out');
}
async function main() {
  assert(!fs.existsSync(root), 'isolated root must be new'); fs.mkdirSync(root, { recursive: true });
  fs.symlinkSync(path.join(repo, 'models'), path.join(root, 'models'), 'dir');
  fs.symlinkSync(build, path.join(root, 'build'), 'dir');
  const source = path.join(root, 'runs', 'fixture', 'raw', 'source.wav'); fs.mkdirSync(path.dirname(source), { recursive: true });
  const formal = path.join(repo, 'runs', 'Tom_Cybersinger', 'model.safetensors');
  const formalBefore = digest(formal);
  const model = path.join(root, 'runs', 'Tom_Cybersinger', 'model.safetensors'); fs.mkdirSync(path.dirname(model), { recursive: true }); fs.copyFileSync(formal, model);
  await run('/usr/bin/ffmpeg', ['-nostdin', '-v', 'error', '-i', path.join(repo, 'tests/fixtures/speech_02.wav'), '-t', '12', '-ar', '40000', '-ac', '1', '-c:a', 'pcm_s16le', source], path.join(root, 'prepare.log'));
  await startServer(); const firstPid = server.pid;
  const accepted = await post('/api/tasks/convert', { task_id: id, session: 'fixture', voice: 'Tom_Cybersinger', workflow: 'dry', output_kind: 'vocal', rms_mix_rate: 0, index_rate: 0, protect: 0.5, filter_radius: 3 });
  assert.equal(accepted.status, 202, JSON.stringify(accepted));
  const work = path.join(root, 'runs/.web_tasks', id, 'work');
  const blocks = path.join(work, 'conversion_blocks/conversion');
  const deadline = Date.now() + 180000;
  let saved = [];
  while (Date.now() < deadline) {
    if (fs.existsSync(blocks)) saved = fs.readdirSync(blocks).filter(n => /^f0_.*\.block$/.test(n));
    if (saved.length) break;
    const r = await task(); assert(!['failed', 'succeeded'].includes(r.state), JSON.stringify(r)); await delay(20);
  }
  assert(saved.length, 'no real completed F0 block');
  const preserved = new Map(saved.map(n => [n, digest(path.join(blocks, n))]));
  const pause = await post('/api/task/pause', { id, attempt: 1 }); assert.equal(pause.status, 202, JSON.stringify(pause));
  const paused = await waitState(['paused']); assert.equal(paused.attempt, 1); assert.equal(paused.actions.resume, true);
  assert.equal((await fetch(base + '/api/task/result?id=' + id)).status, 409);
  await stopServer(); await startServer(); assert.notEqual(server.pid, firstPid);
  assert.equal((await task()).state, 'paused', 'real server restart lost pause');
  const resume = await post('/api/task/resume', { id, attempt: 1 }); assert.equal(resume.status, 202, JSON.stringify(resume));
  assert.equal(resume.body.attempt, 2);
  assert.equal((await post('/api/task/resume', { id, attempt: 1 })).status, 202, 'lost resume acknowledgement is not idempotent');
  assert.equal((await post('/api/task/pause', { id, attempt: 1 })).status, 409, 'old attempt controlled resumed child');
  const complete = await waitState(['succeeded']); assert.equal(complete.attempt, 2);
  for (const [n, hash] of preserved) assert.equal(digest(path.join(blocks, n)), hash, 'completed real block was changed');
  const response = await fetch(base + complete.result_url); assert.equal(response.status, 200);
  const bytes = Buffer.from(await response.arrayBuffer()); fs.writeFileSync(path.join(root, 'resumed.wav'), bytes);
  await stopServer();
  await run(path.join(build, 'vc_convert'), ['--hubert', path.join(repo, 'models/hubert_base/model.safetensors'), '--rmvpe', path.join(repo, 'models/rmvpe.safetensors'), '--model', model,
    '--input', source, '--output', path.join(root, 'continuous.wav'), '--version', 'v2', '--speakers', '109', '--sr', '40000', '--pitch', '0', '--rms-mix', '0', '--protect', '0.5', '--filter-radius', '3'], path.join(root, 'continuous.log'));
  assert(bytes.equals(fs.readFileSync(path.join(root, 'continuous.wav'))), 'HTTP resumed output differs from continuous native conversion');
  assert.equal(digest(formal), formalBefore, 'formal voice was changed');
  const evidence = { passed: true, seconds: 12, firstPid, realRestart: true, preservedF0Blocks: preserved.size, attempt: complete.attempt, outputBytes: bytes.length, pcmWavIdentical: true, formalSha256: formalBefore };
  fs.writeFileSync(path.join(root, 'evidence.json'), JSON.stringify(evidence, null, 2)); console.log(JSON.stringify(evidence));
}
main().catch(async error => {
  console.error(error.stack);
  if (server) { try { const current = await task(); if (current.actions.pause) { await post('/api/task/pause', { id, attempt: current.attempt }); await waitState(['paused', 'succeeded', 'failed'], 60000); } } catch (cleanup) { console.error('fixture safe-stop failed:', cleanup.message); } }
  process.exitCode = 1;
}).finally(stopServer);
