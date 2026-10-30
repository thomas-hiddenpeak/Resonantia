// Task-oriented presentation of existing Web workflows. No model or API ownership.
(() => {
  const q = (id) => document.getElementById(id);
  const voiceLabel = (name) =>
    name === "Tom_Cybersinger" ? "Tom Cybersinger" : name;
  let conversionState = "empty",
    trainingLocked = false,
    resultContext = "",
    resultSettings = "",
    confirmPending = false;
  let online = q("connBanner").style.display === "none";
  const settingsKey = () =>
    JSON.stringify(
      [
        "voiceSelect",
        "inputKind",
        "outputKind",
        "pitchShift",
        "indexRate",
        "rmsMixRate",
        "protect",
        "filterRadius",
      ]
        .map((id) => q(id).value)
        .concat(
          q("conversionDereverb").checked,
          convCascade.state.applied,
          [...q("pitchEdits").querySelectorAll("input,select")].map(
            (el) => el.value,
          ),
        ),
    );
  const stateLabels = {
    empty: "等待音频",
    input: "已选择",
    uploading: "准备音频",
    ready: "可以开始",
    "upload-error": "上传失败",
    running: "处理中",
    done: "已完成",
    error: "需要重试",
  };
  function setText(id, text) {
    q(id).textContent = text;
  }
  function render() {
    const backgroundBusy = !!globalThis.resonantiaTasks?.isBusy();
    const storageReason = globalThis.resonantiaTasks?.blockingReason() || "";
    const ready = convCascade.state.ready && !!convFile;
    const candidate = globalThis.resonantiaWorkflow?.hasCandidate();
    const name = candidate
      ? "训练候选（待听审）"
      : voiceLabel(q("voiceSelect").value);
    setText("summaryInput", convFile?.name || "尚未上传");
    setText("summaryVoice", name || "尚无可用声线");
    setText("taskState", stateLabels[conversionState] || "等待音频");
    q("taskState").dataset.state = conversionState;
    setText(
      "taskTitle",
      convRunning
        ? "正在生成你的作品"
        : conversionState === "done"
          ? "这一版，已准备好"
          : "准备你的翻唱",
    );
    let reason = !convFile
      ? "上传一份音频后即可开始。"
      : !ready
        ? "等待音频准备完成，再开始生成。"
        : !name
          ? "还没有可用声线，请先完成训练与发布。"
          : "已准备就绪，使用当前设置生成。";
    if (conversionState === "upload-error")
      reason = "音频尚未准备好，请在上传区域重试。";
    if (convRunning) reason = "后台任务提交后可关闭页面，进展和结果在任务中心。";
    else if (backgroundBusy || activeTrain) reason = "后台任务正在使用设备，请到任务中心查看或安全暂停。";
    if (conversionState === "error")
      reason = "设置和输入已保留，可再次点击生成。";
    if (conversionState === "done")
      reason = "向下对照听审，也可以调整后重新生成。";
    if (convCascade.state.busy && !convRunning)
      reason = "正在清理人声，完成后可试听并继续生成。";
    if (storageReason && !convRunning) reason = storageReason;
    setText("conversionReadiness", reason);
    q("btnConvert").disabled =
      !!storageReason || backgroundBusy || convRunning || !!activeTrain || convCascade.state.busy || !ready || !name;
    q("regeneratePitch").disabled = q("btnConvert").disabled;
    const pitchLocked = backgroundBusy || convRunning || !!activeTrain || convCascade.state.busy || !ready;
    q("pitchEdits").querySelectorAll("input,select,button").forEach(el => {
      el.disabled = pitchLocked || (el.dataset.role === 'shift' && el.closest('.pitch-edit').querySelector('[data-role="mode"]').value !== 'shift');
    });
    q("addPitchEdit").disabled = pitchLocked || q("pitchEdits").children.length >= 16;
    const editCount = q("pitchEdits").children.length;
    setText("regeneratePitch", editCount ? `生成完整新版（${editCount} 处修正）` : "生成完整新版");
    setText("pitchContext", ready
      ? `${convSource?.display_name || convFile.name} · ${editCount ? '已添加 ' + editCount + ' 个修音区间，生成新版后生效。' : '尚未添加修音区间。'}`
      : "先上传音频，或在任务中心选择已有翻唱，点击“局部修音”。");
    q('btnPreprocess').disabled=backgroundBusy || !!storageReason || convRunning || convCascade.state.busy || !ready || q('inputKind').value==='manual' || (q('inputKind').value==='dry'&&!q('conversionDereverb').checked);
    q("resultStale").hidden =
      !resultSettings || resultSettings === settingsKey();
    if (conversionState === "done" && !q("resultStale").hidden) {
      setText("taskState", "设置已调整");
      setText("taskTitle", "重新生成，应用修改");
    }
    if (convCascade.state.busy && !convRunning) setText("taskState", "清理中");
    q("publishedVoices")
      .querySelectorAll("button")
      .forEach((el) => (el.disabled = convRunning));
    q("convUploadArea").setAttribute("aria-disabled", String(convRunning));
    q("convUploadArea").tabIndex = convRunning ? -1 : 0;
    const mode = q("inputKind").value,
      route = [];
    if (mode === "song") route.push("分离人声", "去和声，保留主唱");
    if (mode === "harmonies") route.push("去和声，保留主唱");
    if (mode === "manual")
      route.push(
        convCascade.state.applied.length ? "使用已清理的人声" : "使用原始输入",
      );
    if (q("conversionDereverb").checked && mode !== "manual")
      route.push("去混响");
    route.push("转换为目标声线");
    if (mode === "song" && q("outputKind").value === "mix")
      route.push("合回原伴奏");
    route.push("试听与下载");
    q("routePlan").replaceChildren(
      ...route.map((text) => {
        const li = document.createElement("li");
        li.textContent = text;
        return li;
      }),
    );
    q("convRetryUpload").hidden = conversionState !== "upload-error";
    q("convUploadStatus").classList.toggle(
      "error",
      conversionState === "upload-error",
    );
    q("convInfo").classList.toggle("error", conversionState === "error");
    const resume =
      q("trainMode").value === "joint-full" && q("trainResume").checked;
    setText("trainingNameSummary", q("voiceName").value.trim() || "尚未命名");
    setText(
      "trainingFilesSummary",
      resume
        ? "复用已固定的素材"
        : trainFiles.length
          ? trainFiles.length + " 份录音"
          : "尚未上传",
    );
    setText(
      "trainingPlanSummary",
      q("trainMode").value === "joint-full"
        ? q("trainEpochs").value + " 轮完整训练"
        : q("trainMode").value === "joint"
          ? q("trainSteps").value + " 步短试听"
          : "高级训练配方",
    );
    setText(
      "trainingReadiness",
      activeTrain
        ? "任务正在进行，配置已锁定。"
        : !/^[A-Za-z0-9_-]{1,64}$/.test(q("voiceName").value.trim())
          ? "先填写有效的任务名称。"
          : !(trainCascade.state.ready || resume)
            ? "上传并检查录音后即可开始。"
            : "下一步会请你核对本次训练计划。",
    );
    if ((!!activeTrain || backgroundBusy) !== trainingLocked) {
      trainingLocked = !!activeTrain || backgroundBusy;
      q("trainingForm")
        .querySelectorAll("input,select,button")
        .forEach((el) => {
          el.disabled = trainingLocked;
        });
      if (!trainingLocked) {
        applyTrainingMode(false);
        trainCascade.setBusy(false);
      }
    }
    if (trainingLocked)
      q("trainingForm")
        .querySelectorAll("input,select,button")
        .forEach((el) => (el.disabled = true));
    q("btnTrain").disabled =
      trainingLocked ||
      !!storageReason ||
      convRunning ||
      trainCascade.state.busy ||
      !/^[A-Za-z0-9_-]{1,64}$/.test(q("voiceName").value.trim()) ||
      !(trainCascade.state.ready || resume);
    q("trainUploadArea").setAttribute("aria-disabled", String(trainingLocked));
    q("trainUploadArea").tabIndex = trainingLocked ? -1 : 0;
    q("trainNavStatus").hidden = !activeTrain;
    const manualTraining=!!trainCascade.state.applied.length;
    for (const id of ['trainAutoSeparate','trainAutoDeharmony','trainAutoDereverb']) q(id).disabled=trainingLocked || manualTraining || resume;
    q('trainAutomaticHint').textContent=manualTraining?'已逐步处理素材，自动清理不重复执行；重置素材后可改用自动清理。':'只勾选素材确实需要的处理；单人干声可全部跳过。';
    q("trainUploadStatus").classList.toggle(
      "error",
      q("trainUploadStatus").textContent.startsWith("上传失败"),
    );
  }
  function navigate(tab) {
    const titles = {
      convert: "制作翻唱",
      train: "训练声线",
      library: "声线库",
      tasks: "任务中心",
    };
    document.querySelectorAll(".tab-btn").forEach((el) => {
      const active = el.dataset.tab === tab;
      el.setAttribute("aria-selected", String(active));
      el.tabIndex = active ? 0 : -1;
    });
    document
      .querySelectorAll(".tab-panel")
      .forEach((el) => (el.hidden = el.id !== "panel-" + tab));
    setText("workspaceLabel", titles[tab] || "工作区");
    render();
  }
  function connection(value) {
    online = value;
    setText("serviceLabel", value ? "服务已连接" : "连接中断");
    q("serviceDot").classList.toggle("online", value);
    q("serviceDot").classList.toggle("offline", !value);
  }
  function voices(data) {
    const box = q("publishedVoices");
    box.replaceChildren();
    for (const name of data.voices || []) {
      const card = document.createElement("article");
      card.className = "published-voice";
      const content = document.createElement("div"),
        title = document.createElement("h3"),
        description = document.createElement("p"),
        button = document.createElement("button");
      title.textContent = voiceLabel(name);
      description.textContent =
        name === "Tom_Cybersinger"
          ? "正式训练 · 200 轮 · 已通过听审"
          : (data.reviewed_voices || []).includes(name)
            ? "已听审发布"
            : "已发布 · 建议先试听";
      button.className = "btn-secondary";
      button.textContent = "用这条声线制作翻唱 →";
      button.disabled = convRunning;
      button.onclick = () => {
        if (convRunning) return;
        q("exitCandidate").click();
        document.querySelector('[data-tab="convert"]').click();
        q("voiceSelect").value = name;
        lastVoicePreset = null;
        applyVoicePreset();
        render();
      };
      content.append(title, description);
      card.append(content, button);
      box.append(card);
    }
    if (!box.children.length) {
      const p = document.createElement("p");
      p.className = "hint";
      p.textContent =
        "还没有已发布声线。先建立训练任务，听审候选后发布到这里。";
      box.append(p);
    }
    render();
  }
  function selectComparison(original) {
    if (
      q(original ? "compareOriginal" : "compareConverted").getAttribute(
        "aria-pressed",
      ) === "true"
    )
      return;
    const from = original ? q("resultAudio") : q("comparisonAudio"),
      to = original ? q("comparisonAudio") : q("resultAudio");
    const time = from.currentTime,
      playing = !from.paused;
    from.pause();
    from.hidden = true;
    to.hidden = false;
    if (Number.isFinite(time)) {
      const seek = () => {
        to.currentTime = Math.min(
          time,
          Number.isFinite(to.duration) ? to.duration : time,
        );
      };
      if (to.readyState >= 1) seek();
      else to.addEventListener("loadedmetadata", seek, { once: true });
    }
    q("compareOriginal").setAttribute("aria-pressed", String(original));
    q("compareConverted").setAttribute("aria-pressed", String(!original));
    if (playing) to.play().catch(() => {});
  }
  function conversion(state) {
    conversionState = state;
    if (state === "input" || state === "empty") {
      resultSettings = "";
      for (const id of ["resultAudio", "comparisonAudio"]) {
        q(id).pause();
        q(id).removeAttribute("src");
        q(id).load();
      }
      if (convUrl) {
        URL.revokeObjectURL(convUrl);
        convUrl = null;
      }
    }
    if (state === "running") {
      const name = globalThis.resonantiaWorkflow?.hasCandidate()
        ? "训练候选（待听审）"
        : voiceLabel(q("voiceSelect").value);
      resultContext =
        convFile.name +
        " · " +
        name +
        " · " +
        (q("outputKind").value === "mix" ? "带原伴奏" : "仅人声") +
        " · 变调 " +
        q("pitchShift").value +
        " 半音";
      resultSettings = settingsKey();
      document.querySelectorAll("audio").forEach((el) => el.pause());
      q("convInfo").textContent = "已提交，正在处理音频…";
    }
    if (state === "done") {
      setText("resultContext", resultContext);
      q("comparisonAudio").src = q("convOrigAudio").src;
      q("compareOriginal").disabled = !q("convOrigAudio").src;
      q("comparisonAudio").hidden = true;
      q("resultAudio").hidden = false;
      q("compareOriginal").setAttribute("aria-pressed", "false");
      q("compareConverted").setAttribute("aria-pressed", "true");
      q("resultSection").focus({ preventScroll: true });
      q("resultSection").scrollIntoView({ behavior: "smooth", block: "start" });
    }
    render();
  }
  function confirmTraining() {
    if (confirmPending || q("btnTrain").disabled || activeTrain)
      return Promise.resolve(false);
    confirmPending = true;
    render();
    q("trainingConfirmSummary").textContent =
      q("trainingNameSummary").textContent +
      " · " +
      q("trainingFilesSummary").textContent +
      " · " +
      q("trainingPlanSummary").textContent +
      "。" +
      (trainCascade.state.applied.length
        ? "已应用：" + trainCascade.state.applied.join(" → ") + "。"
        : "未额外清理素材。");
    const dialog = q("trainingConfirm");
    dialog.returnValue = "";
    dialog.showModal();
    q("dismissTraining").focus();
    return new Promise((resolve) =>
      dialog.addEventListener(
        "close",
        () => {
          confirmPending = false;
          resolve(dialog.returnValue === "confirm");
        },
        { once: true },
      ),
    );
  }
  function trainingStatus(status) {
    q("reviewCompleted").hidden = !(
      status.done &&
      !status.error &&
      !status.cancelled &&
      status.review_required
    );
    render();
  }
  function validationError(message) {
    q("pitchError").textContent = message;
    q("pitchError").hidden = false;
    q("pitchError").closest("details").open = true;
    q("pitchError").scrollIntoView({ behavior: "smooth", block: "center" });
    q("pitchEdits").querySelector("input")?.focus({ preventScroll: true });
  }
  function openPitch() {
    q("pitchPanel").open = true;
    q("pitchTitle").focus({preventScroll:true});
    q("pitchPanel").scrollIntoView({behavior:"smooth",block:"start"});
  }
  globalThis.resonantiaStudio = {
    render,
    navigate,
    connection,
    voices,
    conversion,
    confirmTraining,
    trainingStatus,
    validationError,
    openPitch,
  };
  q("convRetryUpload").onclick = () => ensureConvMaterial();
  q("retryConnection").onclick = () => {
    loadVoices();
    loadSepModels();
  };
  q("retryConnectionBanner").onclick = q("retryConnection").onclick;
  q("regeneratePitch").onclick = () => {
    if (!q("btnConvert").disabled) q("btnConvert").click();
  };
  q("resultPitch").onclick = openPitch;
  q("pitchEdits").addEventListener("input", () => {
    q("pitchError").hidden = true;
  });
  q("reviewCompleted").onclick = () =>
    document.querySelector('[data-tab="library"]').click();
  q("compareOriginal").onclick = () => selectComparison(true);
  q("compareConverted").onclick = () => selectComparison(false);
  for (const kind of ["conv", "train"])
    q(kind + "UploadArea").addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        if (q(kind + "UploadArea").getAttribute("aria-disabled") !== "true")
          q(kind + "FileInput").click();
      }
    });
  const nav = [...document.querySelectorAll(".tab-btn")];
  nav.forEach((el, index) =>
    el.addEventListener("keydown", (event) => {
      let next;
      if (["ArrowDown", "ArrowRight"].includes(event.key))
        next = (index + 1) % nav.length;
      if (["ArrowUp", "ArrowLeft"].includes(event.key))
        next = (index + nav.length - 1) % nav.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = nav.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        nav[next].click();
        nav[next].focus();
      }
    }),
  );
  document.querySelector(".brand").onclick = (event) => {
    event.preventDefault();
    document.querySelector('[data-tab="convert"]').click();
  };
  for (const id of [
    "inputKind",
    "outputKind",
    "conversionDereverb",
    "voiceSelect",
    "trainTier",
    "trainMode",
    "trainResume",
    "voiceName",
    "trainEpochs",
    "trainSteps",
    "pitchShift",
    "indexRate",
    "rmsMixRate",
    "protect",
    "filterRadius",
    "pitchEdits",
  ]) {
    q(id).addEventListener("change", render);
    q(id).addEventListener("input", render);
  }
  document.addEventListener(
    "play",
    (event) => {
      if (event.target.tagName === "AUDIO")
        document.querySelectorAll("audio").forEach((el) => {
          if (el !== event.target) el.pause();
        });
    },
    true,
  );
  // Existing scripts may have completed discovery before this presentation layer.
  connection(online);
  navigate("convert");
  loadVoices();
})();
