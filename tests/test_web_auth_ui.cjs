// Exercise the production authentication functions without a browser or server.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../webui/app.js'), 'utf8');
const end = source.indexOf('// Extract a human-readable message');
assert.ok(end > 0);

function client(answers, {cached = '', unavailableStorage = false, open = false} = {}) {
  let accepted = 'correct', prompts = 0;
  const storage = new Map(cached ? [['resonantiaAuthToken', cached]] : []);
  const sent = [];
  const context = vm.createContext({
    Headers,
    window: {prompt: () => { ++prompts; return answers.shift() ?? null; }},
    sessionStorage: {
      getItem: k => { if (unavailableStorage) throw Error('storage'); return storage.get(k); },
      setItem: (k, v) => { if (unavailableStorage) throw Error('storage'); storage.set(k, v); },
      removeItem: k => { if (unavailableStorage) throw Error('storage'); storage.delete(k); },
    },
    fetch: async (url, options) => {
      const token = options.headers.get('Authorization');
      sent.push(token);
      return {status: open || token === `Bearer ${accepted}` ? 200 : 401};
    },
  });
  vm.runInContext(source.slice(0, end), context);
  return {
    request: () => vm.runInContext("apiFetch('/api/status')", context),
    prompts: () => prompts,
    rotate: () => { accepted = 'rotated'; },
    sent,
  };
}

(async () => {
  for (const first of ['wrong', '', null]) {
    const c = client([first, 'correct']);
    assert.equal((await c.request()).status, 401);
    assert.equal((await c.request()).status, 200, 'a later request must allow correcting the token');
    assert.equal(c.prompts(), 2);
  }
  const concurrent = client(['correct', 'rotated']);
  const replies = await Promise.all([concurrent.request(), concurrent.request()]);
  assert.deepEqual(replies.map(r => r.status), [200, 200]);
  assert.equal(concurrent.prompts(), 1, 'concurrent requests share the pending prompt');
  concurrent.rotate();
  assert.equal((await concurrent.request()).status, 200);
  assert.equal(concurrent.prompts(), 2, 'a completed prompt is not cached forever');

  const fallback = client(['correct'], {unavailableStorage: true});
  assert.equal((await fallback.request()).status, 200);
  assert.equal((await fallback.request()).status, 200);
  assert.equal(fallback.prompts(), 1);
  for (const options of [{cached: 'correct'}, {open: true}]) {
    const c = client([], options);
    assert.equal((await c.request()).status, 200);
    assert.equal(c.prompts(), 0);
  }
  console.log('Web authentication retry, concurrency, rotation and storage fallback passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
