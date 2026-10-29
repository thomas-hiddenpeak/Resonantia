// Standalone, host-only browser contract: static production UI + mock HTTP.
// The generated PCM silence tests playback controls, never voice/model quality.
const fs = require("node:fs"),
  path = require("node:path"),
  http = require("node:http");
const os = require("node:os"),
  { spawn } = require("node:child_process");
const root = path.resolve(__dirname, "..");
const dependencies =
  process.env.NODE_PATH || path.join(__dirname, "browser/node_modules");
const { chromium } = require(path.join(dependencies, "playwright-core"));
const out = path.resolve(
  process.argv[2] || path.join(root, "build/webui-audit"),
);
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "resonantia-ui-"));
const samples = 16000 * 2,
  pcm = Buffer.alloc(44 + samples * 2);
pcm.write("RIFF");
pcm.writeUInt32LE(pcm.length - 8, 4);
pcm.write("WAVEfmt ", 8);
pcm.writeUInt32LE(16, 16);
pcm.writeUInt16LE(1, 20);
pcm.writeUInt16LE(1, 22);
pcm.writeUInt32LE(16000, 24);
pcm.writeUInt32LE(32000, 28);
pcm.writeUInt16LE(2, 32);
pcm.writeUInt16LE(16, 34);
pcm.write("data", 36);
pcm.writeUInt32LE(samples * 2, 40);
const audio = path.join(temporary, "ui-playback-fixture.wav");
fs.writeFileSync(audio, pcm);
const types = {
  "index.html": "text/html; charset=utf-8",
  "styles.css": "text/css",
  "app.js": "text/javascript",
  "workflows.js": "text/javascript",
  "studio.js": "text/javascript",
  "tasks.js": "text/javascript",
  "presets.json": "application/json",
};
const server = http.createServer((req, res) => {
  const name = req.url === "/" ? "index.html" : req.url.slice(1);
  if (!Object.hasOwn(types, name)) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, { "Content-Type": types[name] });
  fs.createReadStream(path.join(process.argv[3] || path.join(root, "webui"), name)).pipe(res);
});
(async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const args = [
    path.join(__dirname, "test_web_studio_browser.cjs"),
    "http://127.0.0.1:" + server.address().port,
    process.env.CHROMIUM_PATH || chromium.executablePath(),
    audio,
    out,
  ];
  const status = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, {
      stdio: "inherit",
      env: { ...process.env, NODE_PATH: dependencies },
    });
    child.once("error", reject);
    child.once("exit", (code) => resolve(code ?? 1));
  });
  process.exitCode = status;
})()
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  })
  .finally(() => {
    server.close();
    fs.rmSync(temporary, { recursive: true, force: true });
  });
