// Recreate an interrupted initial write using the real one-clip/three-epoch
// training fixture's recipe and feature cache. Stop at the first safe boundary;
// this checks recovery and checkpoint persistence, not voice quality.
'use strict';
const fs = require('node:fs'), path = require('node:path');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const [repo, build, reference, root] = process.argv.slice(2).map(p => path.resolve(p));
if (!root) throw new Error('REPO BUILD REFERENCE_FULL_TRAINING NEW_ROOT required');
async function main() {
  assert(!fs.existsSync(root)); fs.mkdirSync(root, { recursive: true });
  const state = path.join(root, 'checkpoints'), clips = path.join(root, 'clips'), cache = path.join(root, 'cache');
  fs.mkdirSync(state); fs.cpSync(path.join(reference, 'clips'), clips, { recursive: true });
  assert.equal(fs.readdirSync(clips).filter(n => n.endsWith('.wav')).length, 1);
  fs.cpSync(path.join(reference, 'cache'), cache, { recursive: true });
  const identity = fs.readFileSync(path.join(reference, 'checkpoints/identity.txt'), 'utf8');
  assert.match(identity, /^[0-9a-f]{64}\n$/); fs.writeFileSync(path.join(state, 'identity.txt'), identity);
  const partial = path.join(state, 'state_step_0000000000.safetensors.tmp.qWe123');
  fs.writeFileSync(partial, 'interrupted initial checkpoint write');
  const control = path.join(root, 'control'); fs.writeFileSync(control, '0');
  const output = path.join(root, 'model.safetensors');
  const args = ['--gan', '--gan-profile', 'joint-full', '--epochs', '3', '--seg', '32',
    '--hubert', path.join(repo, 'models/hubert_base/model.safetensors'), '--rmvpe', path.join(repo, 'models/rmvpe.safetensors'),
    '--pretrained', path.join(repo, 'models/pretrained_v2/pretrained_v2/f0G40k.safetensors'),
    '--dmodel', path.join(repo, 'models/pretrained_v2/pretrained_v2/f0D40k.safetensors'),
    '--target', clips, '--cache-dir', cache, '--checkpoint-dir', state, '--out', output,
    '--resume-initial', '--cancel-file', control, '--progress'];
  const child = spawn(path.join(build, 'vc_train'), args, { stdio: ['ignore', 'pipe', 'pipe'] });
  let text = '', requested = false, timedOut = false;
  const receive = data => {
    text += data.toString();
    if (text.length > 1024 * 1024) { child.kill('SIGTERM'); throw new Error('unexpected log volume'); }
    if (!requested && text.includes('RESONANTIA_PROGRESS 1 training 0 3 steps')) {
      requested = true; fs.writeFileSync(control, '1');
    }
  };
  child.stdout.on('data', receive); child.stderr.on('data', receive);
  const timer = setTimeout(() => { timedOut = true; fs.writeFileSync(control, '1'); }, 180000);
  const hardLimit = setTimeout(() => child.kill('SIGTERM'), 240000);
  let exit;
  try { exit = await new Promise((resolve, reject) => { child.once('error', reject); child.once('exit', (code, signal) => resolve({ code, signal })); }); }
  finally { clearTimeout(timer); clearTimeout(hardLimit); fs.writeFileSync(path.join(root, 'train.log'), text); }
  assert(!timedOut); assert.equal(exit.code, 2, text); assert.equal(exit.signal, null); assert(requested);
  assert(!fs.existsSync(partial), 'incomplete initialization was not retired');
  assert(!fs.existsSync(output), 'safe stop exported a model'); assert(text.includes('cache_hits=1/1'));
  const stopped = text.match(/Joint-full stopped at complete step=(\d+)/); assert(stopped);
  const step = Number(stopped[1]); assert(step >= 0 && step <= 1, 'fixture exceeded one optimizer update');
  assert(fs.statSync(path.join(state, `state_step_${String(step).padStart(10, '0')}.safetensors`)).size > 1000000);
  assert.equal(fs.readFileSync(path.join(state, 'identity.txt'), 'utf8'), identity);
  const evidence = { passed: true, recoveredInitialWrite: true, cacheHits: '1/1', completedSteps: step, exitCode: exit.code, inferenceModelExported: false };
  fs.writeFileSync(path.join(root, 'evidence.json'), JSON.stringify(evidence, null, 2)); console.log(JSON.stringify(evidence));
}
main().catch(e => { console.error(e.stack); process.exitCode = 1; });
