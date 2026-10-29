# Tom Cybersinger 正式声线与 Web 工作流交付

本次交付使用已通过维护者听审的200轮正式模型，不重新训练。已接受的完整翻唱为
`runs/CyberTom_Full_20261026/full-cover-20261029-repaired/`，251.16秒。
新工作流仍需对新歌曲听审，不将同一歌曲的成功外推为所有音域/语言的质量保证。

## 使用

内网入口：<http://192.168.0.122:8080/>。刷新页面后默认进入翻唱页，选择
**Tom Cybersinger · 正式版（200轮，听审通过）**，目前为本机唯一已发布声线。

1. 上传歌曲或录音，支持已有WAV、MP3、M4A/AAC、FLAC、OGG/Opus、WMA、AIFF/AIF。
2. 完整歌曲选“完整歌曲”，程序保留立体声做“分离人声→去和声→转换”，可输出带原伴奏的翻唱。
   已有全人声但含和声选第二项；纯干声选第三项。自行分步处理可逐步试听。
3. 去混响按素材选用，默认不叠加；完整训练的Tom Cybersinger素材曾经去混响并听审通过，
   这不意味着每份推理输入都必须再次去混响。
4. 正式参数为原调、检索0.35、音量占比0（跟随原唱）、保护0.5、平滑窗口3。
   不使用压缩器；必要时全段固定降增益以防削波。
5. 如仍有局部跳音，在高级修音中按上传音频的秒数添加短区间，修改后重新试听。
   不预置本歌的时间点，也不自动拉平滑音。每段0.02–2秒、最多16段，输入最长10分钟。

清理路径最长10分钟/输入128MiB；Web上传及解码后单文件限64MiB，批量解码128MiB。
旧同步音频响应限64MiB；新后台任务结果按块提供，最多1GiB，支持Range试听/下载。
立体声比单声道占用更多空间，较长素材需分文件处理。库原生WAV仍限PCM16/float32，
其他编码格式通过服务器FFmpeg解码。完整歌曲比纯干声多两次模型分离，耗时更长，任务中心显示真实阶段、计数及更新时间。

训练页默认完整联合200轮；50步是素材/流程短试听。保留历史任务和50/100/150/200轮候选，
支持查看日志、填写恢复设置、指定候选试听与听审后发布新名称。恢复须同时具备固定素材与检查点，
完成训练不会自动覆盖旧声线。进度条只表示当前阶段；每步计算耗时来自实际训练日志。
粤语与串烧无需按语言排除，优先检查同一歌手、重叠人声、混响和录音质量；歌唱默认不启用语音VAD切分。

## 模型与质量证据

- 声线目录：`runs/Tom_Cybersinger/`；发布来源
  `runs/CyberTom_Full_20261026/full_training/checkpoints/candidate_epoch_0000000200.safetensors`。
  训练来源目录保留原名，界面中的训练记录显示为Tom Cybersinger。
- G+D更新81,200步，200轮，406个片段，总有效时长1077.43秒；8份原录音均有使用。
- 模型SHA256：`50c48edca6b271cb0741c55658bd268680bb06d73acd0a4807b0154d3e50f44b`。
- Index SHA256：`2a340010641cc59c38cb7fb28f19299c489ea176f70558f6574642eeaaf42852`。
- 真实28秒立体声去和声与已听审参考相对L2为`5.341701836414359e-08`，最大绝对误差`1.7881393432617188e-07`。
- 新CLI在相同已分离主唱输入上的局部修音与已接受28秒修正版1,120,000样本逐一一致，最大误差和相对L2均为0。
- 真实Web的6秒完整歌曲链路返回44.1k立体声、264600帧，峰值0.949951，无削波/非有限值；
  服务记录`Device: Orin`与HuBERT/F0/Index/Synth时间。该检查验证真实链路，不替代新样本听审。
- 真实Chromium桌面/390px手机宽度检查通过：默认声线、参数保留、标签页隔离、真实上传、
  处理中锁定、局部修音字段、候选/日志/确认发布、结果播放与无横向溢出。浏览器用例的转换响应
  被拦截，不启动模型；上述真实Web调用单独执行，不混为同一证据。

所有本地证据位于 `runs/CyberTom_Web_Release_20261029/evidence/`，音频未加入Git。

## 验证命令与交付状态

本次构建/测试使用Jetson Orin NX、CUDA13.2，已有真实模型与fixtures，不启动新的长训练。

```bash
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j 2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^(test_audio_io|test_bounded_pipeline_dsp|test_cli_contract|test_web_audio_upload|test_web_contract|test_web_train_cancel|test_web_transaction_failure)$' --output-on-failure
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG --output-on-failure
node tests/test_web_auth_ui.cjs
node tests/test_web_audio_ui.cjs
node tests/test_web_joint_ui.cjs
node tests/test_web_train_cancel_ui.cjs
bash scripts/check_sdd.sh
```

浏览器使用临时安装的Node22/playwright-core和既有Chromium，不加入产品运行依赖：

```bash
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules /tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_workflows_browser.cjs http://127.0.0.1:8081 /home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell runs/CyberTom_Full_20261026/full-cover-20261028/tail-diagnosis-20261029/lead-control/lead_source.wav runs/CyberTom_Web_Release_20261029/evidence/browser
```

本轮完整构建通过。完整CTest首轮64/66通过，2项失败；修复测试素材和隔离额外负载后
按原门槛复查2/2通过，最终66项均已有通过结果，无CTest跳过项。最后Web错误分类/UI增量
18/18通过，独立host11/11通过。以下是复查和最终增量验证命令：

```bash
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG --rerun-failed --output-on-failure -V
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L web --output-on-failure
cmake -S cmake/host-tests -B /tmp/resonantia-web-release-host -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/resonantia-web-release-host -j 2
ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure
```

主机服务已部署并重启，PID1999975，状态active。入口、静态文件、voices与train/runs接口
全部HTTP200；运行中的二进制与测试构建SHA相同。浏览器用上述相同命令改用
`http://192.168.0.122:8080`、输出目录`evidence/browser-deployed`再次通过，包含
不安全HTTP源下的会话隔离、复制标签页和更换输入清空旧修音区间。浏览器脚本早期曾因
未展开高级控件、匹配错误的日志文本以及重选同一文件不产生change而失败；修正测试操作后通过，
未用这些失败冒充产品通过。

## 保留与回退

首次部署时保留旧声线；同日维护者后续要求清除，已按下节记录删除旧版模型/索引。
原录音、正式训练checkpoint与阶段候选继续保留。正式发布通过HTTP候选发布接口完成，
使用已有三件套事务与显式听审确认；重复发布相同名称返回409。
部署前保留原二进制；回滚只恢复二进制/静态页面，不回滚或删除用户素材。
自动去和声已整合，歌词特定校正仍为用户选择的局部修音。尚未执行跨歌曲/语言/音域完整质量矩阵，
训练期间按用户要求停止了连续资源采样，未测得整个81,200步的持续资源峰值。

本轮完整回归首轮发现：旧HTTP替身只提供WAV魔数，新流程需要解析完整PCM；已把替身
改为完整的微型PCM16样本，保留原有进程失败/回滚/重试断言，host 11/11复跑通过。
另一个整机`cudaMemGetInfo`资源读数门槛首次失败（短505032704、长935022592字节），
当时存在host编译负载；停止额外负载后复查原256MiB门槛通过，短651341824、长855691264字节，
差194.8828125MiB；进程RSS短1342480、长1339032KiB。整机读数受其他工作影响，
本次未降低阈值，也未把整机读数当作进程独占分配量。
这两个初次失败与后续结果分别保留日志，不能把首轮写成零失败。

回退目录为 `runs/CyberTom_Web_Release_20261029/rollback/`，保留部署前6个二进制和服务单元。
兼容旧界面的备用副本来自之前2026-10-26发布时留存的页面；它是可用的旧界面，
不是本轮编辑前逐字相同的页面快照。回退时通过服务webroot指向副本，不覆盖工作区源码。

## 同日后续：统一名称并清除旧声线

维护者明确要求仅保留正式版并命名为 **Tom Cybersinger**。发布目录/API安全标识使用
`Tom_Cybersinger`；页面标题、声线选项、预设和训练记录显示名称一致。原训练run保留原名，
发布marker仍追溯到它，恢复点、406份固定素材、四个阶段候选和已接受的翻唱均保留。
模型及索引的SHA256与上文完全一致；转换参数与去和声流程不变。

已实际删除以下8个旧发布声线的模型/索引，以及它们目录内的5个旧候选/试验模型副本：
`CyberTom`、`CyberTom_20261026`、`CyberTom_HQ_20261024`、`CyberTom_Pilot`、
`CyberTom_Pilot_Decoder`、`CyberTom_Pilot_HTK`、`CyberTom_Pilot_LowLR`、
`CyberTom_Pilot_OutputFix`。合计21个文件、2133.416748MiB；不是仅在前端隐藏。
旧试验的日志、音频和诊断数据保留为证据，通用预训练资产未清理。

改名前核验服务空闲，迁移正式发布三件套后清理旧权重并重启。服务当前PID2005146，
内网入口HTTP200，`/api/voices`的`voices`和`reviewed_voices`均只含`Tom_Cybersinger`。
改名后的命令和结果：

```bash
node tests/test_web_joint_ui.cjs
node runs/Tom_Cybersinger_Web_20261029/evidence/rename-release.cjs
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L web --output-on-failure
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules /tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_workflows_browser.cjs http://192.168.0.122:8080 /home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell runs/CyberTom_Full_20261026/full-cover-20261028/tail-diagnosis-20261029/lead-control/lead_source.wav runs/Tom_Cybersinger_Web_20261029/evidence/browser
bash scripts/check_sdd.sh
git diff --check
```

唯一声线专用预设断言先因旧名称失败，修改后通过；18项Web CTest全部通过、0失败/跳过，
16.00秒。实际内网Chromium检查通过：唯一声线、显示名称、默认参数、转换请求使用新标识、
正式训练来源仍可查看，以及已有桌面/手机工作流。浏览器转换响应被拦截；本次未运行模型
转换或训练，也未重跑完整66项GPU测试。此次只改静态UI与发布文件，无C++构建变更。
本地删除清单、前后API、模型SHA、首轮失败与通过日志在
`runs/Tom_Cybersinger_Web_20261029/evidence/`。迁移脚本是一次性操作记录，重复执行会因
目标已存在而拒绝；旧声线已删除，旧页面回退副本不能恢复它们。

## 同日后续：任务中心与关闭页面后恢复

新版已部署到同一内网地址，刷新页面即可使用。转换、清理和训练成功提交后由后台持有，关闭页面不停止；
重开“任务中心”可查看当前或历史任务、真实阶段/计数、设置、日志以及试听下载结果。网络断开时明确
显示状态未知；恢复连接继续读取同一任务，不自动重复提交。需要分离或清理的翻唱可先选“只清理人声，先听效果”。

支持暂停的任务使用“安全暂停”，到分块或完整训练保存点才停；再点“继续此任务”。服务重启后显示中断，
必须明确继续，最多恢复到已保存点。完整joint-full训练可续跑；旧训练模式和独立手动步骤没有同等恢复承诺。
旧服务完成/失败时没有保存的转换，不能凭空补回任务历史。

成功任务可“清理冗余副本”，保留原素材、音频、日志、训练保存点、候选和正式模型。任务中心展示磁盘占用
及保护线；空间不足时阻止新计算，并在可恢复任务安全点暂停。部署验证历史中的“部署验证_6秒干声.wav”
只是CPU文件处理与页面恢复验收，没有运行模型。正式Tom文件未修改。

最新发布、测试、首次失败和回退证据以[任务生命周期验收](../specs/003-webui/task-lifecycle-20261029.md)为准，
上文PID和66项测试属于此前正式声线发布。

## 完成任务后调整参数

在“任务中心”选中已完成的音频任务，点击“调整参数再生成”。页面带回原素材、声线和设置，
无需重新上传；修改后提交为独立新版，原任务和音频保留，可用“查看原版”返回对照。
仅改变音高、检索率、保护、动态或局部修音会复用已验证的人声分离与清理；只切换人声/混音输出
可复用已转换的人声。改变去混响设置会重做清理及后续转换。任务详情和日志列出实际复用阶段。
身份核验仍需时间；已清理的副本、模型/程序变化或损坏的阶段不能当作可复用缓存。
本入口用于音频任务，完整训练仍按其固定配方与检查点恢复，不允许直接改变旧训练配方。

## 重命名与两份音频结果

任务详情标题旁点击“重命名”，填写名称后保存。名称仅用于显示，训练目录、声线名、素材及参数
保持；可在运行、暂停或完成后修改，关闭页面或服务重启后仍保留。其他窗口已改名时会提示冲突。

歌曲默认选择“完整翻唱 + 干净人声”，完成后直接显示两个播放器，各自提供WAV下载；干净人声
是转换后的目标声线，不含伴奏。两份音频不会同时播放。显式选择仅人声，或输入本来无伴奏时，
只显示人声。启用访问令牌的服务需点击各播放器的载入按钮。

混音任务的“清理冗余副本”保留这两份音频。此前任务只要仍有完成人声即可直接读取，已经删除
的人声会显示缺失提示，不会重新占用设备计算。详情见[验收记录](../specs/003-webui/task-results-20261029.md)。
