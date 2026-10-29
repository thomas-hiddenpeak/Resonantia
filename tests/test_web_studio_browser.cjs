// Opt-in studio acceptance. Real production HTML/JS; mocked service responses
// and a real WAV fixture. Never starts GPU conversion, training or publication.
const { chromium } = require("playwright-core");
const fs = require("node:fs"),
  path = require("node:path"),
  assert = require("node:assert/strict");
(async () => {
  const [base, executablePath, audio, out] = process.argv.slice(2);
  assert(base && executablePath && audio && out);
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({
    executablePath,
    headless: true,
    args: ["--no-sandbox"],
  });
  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1000 },
      reducedMotion: "reduce",
    });
    const page = await context.newPage(),
      errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    const wav = fs.readFileSync(audio);
    let uploadFails = false,
      convertFails = false,
      convertCount = 0,
      preprocessCount = 0,
      trainCount = 0;
    let releaseConversion = null,
      holdConversion = false,
      lastBody = "";
    const voices = {
      voices: ["Tom_Cybersinger"],
      reviewed_voices: ["Tom_Cybersinger"],
      training_profiles: ["decoder", "gan", "joint", "joint-full"],
      training: { running: false },
    };
    const run = {
      name: "CyberTom_Full_20261026",
      resumable: true,
      result_ready: true,
      candidates: ["candidate_epoch_0000000200.safetensors"],
    };
    const taskRecords = new Map();
    let runs = [run],
      trainingStatus = {
        name: "",
        kind: "none",
        running: false,
        done: false,
        log: "Wrote index: fixture",
      };
    await page.route("**/api/**", async (route) => {
      const url = new URL(route.request().url()),
        p = url.pathname;
      const json = (body, status = 200) =>
        route.fulfill({
          status,
          contentType: "application/json",
          body: JSON.stringify(body),
        });
      if (p === '/api/tasks') return json({tasks:[...taskRecords.values()],active_id:[...taskRecords.values()].find(r=>r.state==='running')?.id || ''});
      if (p === '/api/storage') return json({managed_bytes:0,available_bytes:107374182400,managed_limit_bytes:68719476736,task_limit_bytes:25769803776,reserve_bytes:8589934592,retained_tasks:taskRecords.size,history_tasks:taskRecords.size,retained_limit:256,history_limit:4096,reason:''});
      if (p === '/api/task') return taskRecords.has(url.searchParams.get('id')) ? json(taskRecords.get(url.searchParams.get('id'))) : json({error:'missing'},404);
      if (p === '/api/task/log') return route.fulfill({status:200,body:'retained fixture log'});
      if (p === '/api/task/result') {
        const range=/^bytes=(\d+)-(\d*)$/.exec(route.request().headers().range || '');
        const start=range?Number(range[1]):0,end=range&&range[2]?Math.min(Number(range[2]),wav.length-1):wav.length-1;
        const headers={'Accept-Ranges':'bytes','Content-Length':String(end-start+1)};
        if(range)headers['Content-Range']=`bytes ${start}-${end}/${wav.length}`;
        return route.fulfill({status:range?206:200,contentType:'audio/wav',headers,body:wav.subarray(start,end+1)});
      }
      if (p === "/api/voices") return json(voices);
      if (p === "/api/sep-models") return json({});
      if (p === "/api/train/runs") return json({ runs });
      if (p === "/api/train/status") return json(trainingStatus);
      if (p === "/api/material")
        return json(
          uploadFails
            ? { error: "录音无法解码，请重新选择文件" }
            : { ok: true },
          uploadFails ? 400 : 200,
        );
      if (p === "/api/preview")
        return route.fulfill({
          status: 200,
          contentType: "audio/wav",
          body: wav,
        });
      if (p === "/api/tasks/convert" || p === '/api/tasks/preprocess') {
        const preprocessing=p==='/api/tasks/preprocess';
        if(preprocessing)preprocessCount++;else convertCount++;
        lastBody = route.request().postData();
        if (holdConversion) await new Promise((r) => (releaseConversion = r));
        if(convertFails)return json({error:'设备正在处理另一个任务'},409);
        const id=/name="task_id"\r\n\r\n([^\r]+)/.exec(lastBody)[1];
        taskRecords.set(id,{id,name:'试听.wav',kind:preprocessing?'preprocess':'convert',state:'succeeded',stage:'done',attempt:1,completed:1,total:1,unit:'files',created_ms:Date.now(),finished_ms:Date.now(),parameters:{},actions:{},result_url:'/api/task/result?id='+id});
        return json({task_id:id},202);
      }
      if (p === "/api/train") {
        trainCount++;
        assert.match(route.request().postData(),/name="dereverb"\r\n\r\n1\r\n/,'automatic cleanup recipe was not submitted with training');
        return json({ error: "测试拒绝启动训练" }, 409);
      }
      throw new Error("Unexpected API call: " + p);
    });
    await page.goto(base);
    await page.waitForFunction(
      () => document.querySelector("#voiceSelect").value === "Tom_Cybersinger",
    );
    assert.equal(
      await page.locator('[data-tab="library"]').count(),
      1,
      "business navigation must separate voice library from new training",
    );
    assert(await page.locator("#btnConvert").isDisabled());
    assert.match(
      await page.locator("#conversionReadiness").textContent(),
      /上传/,
    );
    assert(
      (await page.locator("#convUploadArea").getAttribute("role")) === "button",
      "upload must support keyboard",
    );
    await page.addScriptTag({ path: require.resolve("axe-core") });
    async function accessibility(label) {
      const violations = await page.evaluate(async () => {
        const r = await axe.run(document, {
          runOnly: {
            type: "tag",
            values: ["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"],
          },
        });
        return r.violations.map((v) => ({
          id: v.id,
          targets: v.nodes.map((n) => n.target),
        }));
      });
      assert.deepEqual(violations, [], label + " accessibility");
    }
    for (const tab of ["convert", "train", "library"]) {
      await page.locator('[data-tab="' + tab + '"]').click();
      await accessibility(tab);
    }
    await page.locator('[data-tab="convert"]').click();
    await page.locator('[data-tab="convert"]').focus();
    await page.keyboard.press("ArrowRight");
    assert.equal(
      await page.locator('[data-tab="train"]').getAttribute("aria-selected"),
      "true",
    );
    await page.keyboard.press("Home");
    const chooser = page.waitForEvent("filechooser");
    await page.locator("#convUploadArea").focus();
    await page.keyboard.press("Enter");
    await (await chooser).setFiles([]);
    await page.screenshot({
      path: path.join(out, "desktop-empty.png"),
      fullPage: true,
    });
    uploadFails = true;
    await page.locator("#convFileInput").setInputFiles(audio);
    await page.locator("#convRetryUpload").waitFor({ state: "visible" });
    assert(
      !(await page.locator("#connBanner").isVisible()),
      "HTTP validation error is not a service outage",
    );
    uploadFails = false;
    await page.locator("#convRetryUpload").click();
    await page.waitForFunction(
      () => !document.querySelector("#btnConvert").disabled,
    );
    assert.match(await page.locator("#routePlan").textContent(), /分离人声/);
    await page.locator("#inputKind").selectOption("dry");
    assert.equal(await page.locator("#outputKind").inputValue(), "vocal");
    assert(
      !(await page.locator("#routePlan").textContent()).includes("分离人声"),
    );
    await page.locator("#inputKind").selectOption("song");
    assert.equal(
      await page.locator("#outputKind").inputValue(),
      "mix",
      "returning to a song restores the mix default",
    );
    holdConversion = true;
    convertFails = true;
    await page.locator("#btnConvert").click();
    await page.waitForFunction(
      () => document.querySelector("#convFileInput").disabled,
    );
    for (let i = 0; !releaseConversion && i < 100; i++)
      await page.waitForTimeout(20);
    assert(releaseConversion);
    assert(await page.locator("#voiceSelect").isDisabled());
    await page.screenshot({
      path: path.join(out, "desktop-running.png"),
      fullPage: true,
    });
    releaseConversion();
    await page.waitForFunction(
      () => !document.querySelector("#btnConvert").disabled,
    );
    await page.locator('#nav-convert').click();
    assert.match(await page.locator("#convInfo").textContent(), /设备正在/);
    assert(!(await page.locator("#resultSection").isVisible()));
    holdConversion = false;
    convertFails = false;
    await page.locator("#btnConvert").click();
    await page.locator('#nav-convert').click();
    await page.locator("#resultSection").waitFor({ state: "visible" });
    assert.equal(convertCount, 2);
    assert(lastBody.includes("Tom_Cybersinger"));
    assert.match(
      await page.locator("#resultContext").textContent(),
      /Tom Cybersinger/,
    );
    await page.waitForFunction(
      () => document.querySelector("#resultAudio").readyState >= 1,
    );
    await page.locator("#resultAudio").evaluate((el) => (el.currentTime = 0.5));
    await page.waitForFunction(()=>Math.abs(document.getElementById('resultAudio').currentTime-0.5)<0.1);
    await page.locator("#compareOriginal").click();
    assert.equal(
      await page.locator("#compareOriginal").getAttribute("aria-pressed"),
      "true",
    );
    await page.waitForFunction(
      () =>
        Math.abs(document.querySelector("#comparisonAudio").currentTime - 0.5) <
        0.1,
    );
    await page.locator("#compareConverted").click();
    assert.equal(
      await page.locator("#compareConverted").getAttribute("aria-pressed"),
      "true",
    );
    await page.locator('#btnPreprocess').click();await page.locator('#taskStateLabel').filter({hasText:'已完成'}).waitFor();
    assert.equal(preprocessCount,1);assert.equal(convertCount,2,'cleanup must not synthesize a voice');
    await page.locator('#nav-convert').click();
    await page.locator("#inputKind").selectOption("dry");
    assert(await page.locator("#resultStale").isVisible());
    await page.locator("#addPitchEdit").evaluate(el=>el.closest('details').open=true);
    await page.locator("#addPitchEdit").click();
    await page.locator("#regeneratePitch").click();
    assert(await page.locator("#pitchError").isVisible());
    assert(await page.locator("#resultSection").isVisible(),'invalid correction must preserve the listening result');
    assert.equal(convertCount,2,'invalid correction must not reach the service');
    await page.locator('.pitch-edit button').click();
    await accessibility("result");
    await page.screenshot({
      path: path.join(out, "desktop-result.png"),
      fullPage: true,
    });
    await page.locator('[data-tab="library"]').click();
    await page.locator("#trainingRun").selectOption(run.name);
    assert(await page.locator("#resumeRun").isDisabled(),'a completed run does not need recovery');
    await page.locator("#previewCandidate").click();
    await page.locator("#candidateNotice").waitFor({ state: "visible" });
    assert.equal(
      await page.locator('[data-tab="convert"]').getAttribute("aria-selected"),
      "true",
    );
    await page.locator("#exitCandidate").click();
    await page.locator('[data-tab="train"]').click();
    assert(!(await page.locator("#trainingRun").isVisible()));
    await page.locator("#voiceName").fill("Studio_Test");
    await page.locator("#trainFileInput").setInputFiles(audio);
    await page.waitForFunction(
      () => !document.querySelector("#btnTrain").disabled,
    );
    await page.locator('#trainAutoDereverb').evaluate(el=>el.closest('details').open=true);
    await page.locator('#trainAutoDereverb').check();
    await page.locator("#btnTrain").click();
    await page.locator("#trainingConfirm").waitFor({ state: "visible" });
    await accessibility("training confirmation");
    assert.equal(trainCount, 0, "review must precede long training");
    await page.locator("#dismissTraining").click();
    assert.equal(trainCount, 0);
    await page.locator("#btnTrain").click();
    await page.locator("#confirmTraining").click();
    await page.waitForFunction(() =>
      document.querySelector("#trainStage").textContent.includes("测试拒绝"),
    );
    assert.equal(trainCount, 1);
    assert(!(await page.locator("#voiceName").isDisabled()));
    await page.screenshot({
      path: path.join(out, "desktop-training.png"),
      fullPage: true,
    });
    await page.locator('[data-tab="library"]').click();
    await page.screenshot({
      path: path.join(out, "desktop-library.png"),
      fullPage: true,
    });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('[data-tab="convert"]').click();
    await page.screenshot({
      path: path.join(out, "mobile.png"),
      fullPage: true,
    });
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      "mobile must not overflow",
    );
    for (const width of [320, 768, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      for (const tab of ["convert", "train", "library"]) {
        await page.locator('[data-tab="' + tab + '"]').click();
        assert(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
          tab + " overflow at " + width,
        );
      }
    }
    runs = [];
    voices.voices = [];
    voices.reviewed_voices = [];
    await page.locator("#refreshRuns").click();
    await page.waitForFunction(
      () => document.querySelector("#trainingRun").options.length === 0,
    );
    assert(await page.locator("#previewCandidate").isDisabled());
    assert(await page.locator("#publishVoice").isDisabled());
    assert(await page.locator("#resumeRun").isDisabled());
    await page.locator('[data-tab="convert"]').click();
    await page.waitForFunction(
      () => document.querySelector("#voiceSelect").value === "",
    );
    assert(await page.locator("#btnConvert").isDisabled());
    trainingStatus = {
      name: "Studio_Active",
      kind: "train",
      job_id: "active-test",
      running: true,
      done: false,
      cancellable: true,
      stage: "training",
      log: "joint-full step=20/100 epoch=2",
    };
    taskRecords.set('active-test',{id:'active-test',name:'Studio_Active',kind:'training',state:'running',stage:'training',phase:'training',attempt:1,completed:20,total:100,unit:'steps',created_ms:Date.now(),parameters:{recovery:'training-full-v1'},actions:{pause:true,resume:false}});
    await page.reload();
    await page.locator('#nav-tasks').click();
    await page.evaluate(()=>globalThis.resonantiaTasks.open('active-test'));
    await page.locator('#taskCounter').filter({hasText:'20 / 100'}).waitFor();
    assert(!(await page.locator('#taskPause').isDisabled()));
    await page.locator('[data-tab="train"]').click();
    assert(await page.locator("#voiceName").isDisabled());
    assert(await page.locator("#trainFileInput").isDisabled());
    assert(await page.locator("#btnTrain").isDisabled());

    assert.deepEqual(errors, []);
    fs.writeFileSync(
      path.join(out, "result.json"),
      JSON.stringify(
        {
          passed: true,
          convertCount,
          trainCount,
          mocked: true,
          checks: [
            "business navigation",
            "upload retry",
            "error honesty",
            "workflow plan",
            "inflight lock",
            "convert failure and retry",
            "result provenance",
            "comparison",
            "result settings changes",
            "invalid correction preserves result",
            "candidate navigation",
            "training confirmation",
            "keyboard navigation and upload",
            "empty voice and candidate lists",
            "active training restored and locked",
            "320/390/768/1440 layouts",
            "axe WCAG A/AA automated rules",
            "no JS errors",
          ],
        },
        null,
        2,
      ),
    );
    console.log(
      "Studio browser acceptance passed (HTTP/model operations mocked).",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
