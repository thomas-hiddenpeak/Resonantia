# WebUI 工作室：任务流程与交互验收

## 目标与设计依据

维护者要求重新设计业务交互与视觉，完成自审后推送GitHub。本轮沿用已验证的HTTP、
C++/CUDA与模型能力，不改音频算法、不重新训练、不进行完整歌曲转换。

设计基于人的任务而不是后端接口：识别优于回忆、就近反馈、渐进展示、成本较高操作前
核对、结果可比较。依据[Nielsen可用性原则](https://www.nngroup.com/articles/ten-usability-heuristics/)
及[WCAG 2.2](https://www.w3.org/TR/WCAG22/)的键盘、焦点、标签、错误识别与点击目标要求。
这是具体设计依据，不宣称仅凭自动检查就通过完整无障碍认证或专业用户研究。

## 业务路径与验收要求

1. **制作翻唱**：先上传，再确认完整歌曲/含和声人声/单人干声，选择声线及输出；概览显示
   将要执行的处理链，正式声线默认采用已听审参数。清理模型和数值参数收在高级设置。
2. **生成与恢复**：上传/处理/失败/成功各有明确状态；失败原地可重试，业务错误不误报
   服务离线；处理中防止重复提交及更换素材。无后端阶段数据时只显示等待时间，不伪造进度。
3. **听审闭环**：结果显示来源、使用的声线和设置；原始/转换声音可在相同播放位置切换，
   两份音频不同时播放。下载入口明确；局部音高修正位于听审之后，换输入清除旧校正。
4. **训练声线**：命名→上传与检查素材→按需清理并听审→训练配方→确认启动；训练中锁定
   配置，提供真实进度、可展开日志及边界取消。刷新可跟踪仍活动的任务，不自动开始训练。
5. **声线库**：正式已发布声线与训练候选分区，训练记录→候选试听→听审确认→发布。
   未选记录、无候选、无恢复点时正确禁用入口；候选试听明确标记，退出恢复正式参数。
6. **视觉与可达性**：完整桌面工作区、手机单列，统一字阶/间距/颜色/图标，主要按钮足够大；
   标注必填/格式和禁用原因，键盘可选文件/导航/完成关键流程，可见焦点、减少动态效果。

## 实施与验证边界

静态HTML/CSS/JS保持无构建依赖；现有app.js负责HTTP与任务所有权，workflow controls负责
处理选项与候选，新增界面状态层负责任务概览、导航、确认与对照播放。数据仍由现有接口
返回，不增加模拟历史、固定成功率或无依据的GPU/剩余时间数字。

先保存旧页面用于在开发期间维持8080服务；8081用于独立浏览器验证。验收后才切换正式
入口。既有模型、索引、训练记录、素材与其他未提交工作不被重置。

验收证据须包含：实际DOM/截图、正常与失败/恢复路径、键盘及手机布局、既有Web测试、
SDD检查、审计发现及修复、最终提交内容审查与GitHub准确提交的CI结果。自动化浏览器
可以拦截模型请求验证界面，但必须标记，不能冒充真实音频质量验证。

## 自审与最终记录

已完成界面与本地流程审计。发现并修复：

| 审计发现 | 修复及证据 |
|---|---|
| 新建训练与候选发布混在同一长页 | 拆分创作/训练/声线库，浏览器验证导航与候选返回 |
| 业务校验失败误报服务离线，缺少原地重试 | 明确上传错误与连接状态，保留输入并重试；模拟400及409均验证 |
| 长训练立即开始，运行中仍可改素材 | 摘要确认后才发送请求，活动任务锁定，刷新恢复状态；浏览器与Node取消门禁 |
| 改过参数后容易把旧结果当成新结果 | 结果保留生成时背景，设置变化提示重新生成，A/B保留播放位置且互斥播放 |
| 无效修音会清空已有结果 | 请求前验证并聚焦修音区，已有结果保留；浏览器验证请求次数不增加 |
| 空候选仍可操作、完成记录仍诱导恢复 | 空状态禁用、已完成任务不提供无效恢复；浏览器覆盖 |
| 初版六处小字对比度不足 | 调整文字色后axe三页、结果、确认弹窗规则检查通过 |
| 部分入口只能鼠标点击，手机可能缺少重连 | 键盘上传/页签、可见焦点、错误区重连，320/390/768/1440宽度无横向溢出 |

视觉采用统一的暖白背景、深绿色动作色、克制的字阶和图标；桌面固定任务概览、手机
单列。已实际查看创作、结果、训练、声线库与手机截图，不以DOM断言代替视觉核查。
局部修音放在听审结果中并提供就近重新生成入口；默认参数、Tom Cybersinger与去和声链路保留。

### 验证命令与边界

```bash
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j 2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L web --output-on-failure
node tests/test_web_auth_ui.cjs
node tests/test_web_audio_ui.cjs
node tests/test_web_joint_ui.cjs
node tests/test_web_train_cancel_ui.cjs
bash scripts/check_sdd.sh
git diff --staged --check
```

完整已有目标构建成功，Web CTest18/18通过；发布暂存内容用`git checkout-index`展开到
干净副本后独立配置/构建host并运行CTest，11/11通过、0失败/skip，0.59秒。干净副本
也独立通过浏览器门禁，确认不依赖未提交源文件。源码、测试、文档一同发布；模型、
音频及本地诊断工具/产物未暂存。前置后端实现未在此次UI重设计中更改，既有真实模型
数值、完整训练和听审证据见PROJECT_STATE；不冒称重新执行全66项模型测试或声音听审。

新增浏览器门禁的可复现host入口（Node20+，仅测试依赖）：

```bash
npm ci --prefix tests/browser --ignore-scripts --no-audit --no-fund
tests/browser/node_modules/.bin/playwright-core install --with-deps chromium
node tests/run_web_studio_browser.cjs
```

独立runner只提供生产静态文件，API响应受控，并生成两秒PCM静音用于播放控件；不启动
任何模型。设备本地以既有Node22.23.4/Chromium执行相同测试；额外对真实8081/8080
执行`test_web_workflows_browser.cjs`，使用真实28秒主唱WAV验证HTTP上传/候选/日志，
转换响应仍明确拦截。没有新长训练或完整歌曲推理。

首个流程红测因缺少声线库导航失败；首次axe发现对比度问题，修复后通过。早期格式检查
发现三行尾随空格，格式化后消除。失败和通过记录均保留在
`runs/WebUI_Studio_20261029/evidence/`，未删除失败或降低门槛。新增CI作业
`web-studio-browser`自动执行四项Node门禁与浏览器/axe验收，GPU作业仍保持原手动边界。

### 部署

已在无活动任务时切回正式静态目录并重启，服务PID2020827，入口仍为
<http://192.168.0.122:8080/>。回退页面为
`runs/WebUI_Studio_20261029/baseline/webui/`；回退只改服务webroot，不涉及任何素材或模型。
完整删除清单与正式声线改名见[前一轮发布记录](WEBUI_RELEASE_20261029.md)。
GitHub推送与CI以最终交付中的提交链接及该提交的运行结果为准；本文件记录本地验收证据。

### 中文文件名上传修复（2026-10-29）

用户反馈`单依纯 – 我管不住我自己.mp3`上传报格式不支持。旧`safe_filename`删掉
中文和特殊横线，使结果只剩`.mp3`，随后被文件系统扩展名检查拒绝，尚未调用FFmpeg。
已保留合法UTF-8字符，拒绝非法编码、路径分隔符、NUL及超过255字节的名称；ASCII
过滤规则保持。不同中文文件不会再折叠成相同名称；归一化为WAV后的真正重名仍整批
拒绝，并保留原材料。四个上传入口共用修复。

新增回归覆盖中文、日文、重音字母、emoji、非法/截断/过长UTF-8、路径/字节上限、
不同中文名称和跨格式真正重名；真实MP3以ASCII及用户报告的文件名上传后，断言解码
WAV逐字节一致。模型进程在契约测试中使用替身，codec测试仅调用FFmpeg。

```bash
cmake --build /tmp/resonantia-web-release-host --target test_web_audio_upload -j 2
ctest --test-dir /tmp/resonantia-web-release-host -R '^test_web_audio_upload$' --output-on-failure
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j 2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L web --output-on-failure
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules \
  /tmp/resonantia-web-release-browser/node_modules/node/bin/node \
  runs/Unicode_Upload_Fix_20261029/upload-browser.cjs
bash scripts/check_sdd.sh
git diff --check
```

修复前focused测试因中文名返回400失败，生产HTTP一秒MP3同名复现400；修复后focused
1/1通过。完整已有目标构建成功，最终Web **18/18通过，0失败/skip，16.03秒**，含
12组真实codec及同名MP3解码比较。未重新运行全部66项模型测试。

确认没有活动任务/子进程后，原子替换`build/vc_serve`并重启服务，PID2034441；实际
运行二进制SHA与测试构建一致。内网Chromium以报告的文件名上传一秒真实人声MP3，
HTTP200，页面显示原名及“上传完成”，WAV预览时长1秒、生成按钮可用、无JS错误。
浏览器探针阻止所有除material之外的非GET API，未请求转换、清理、训练或发布。
此验证复现的是文件名问题，没有取得用户整首MP3，不能宣称其完整内容已验证。
正式声线API仍只返回Tom Cybersinger。

证据、浏览器脚本和回退二进制分别位于本地
`runs/Unicode_Upload_Fix_20261029/evidence/`、`upload-browser.cjs`及`backup/vc_serve`；
恢复备份二进制并重启即可回退，不涉及模型或上传素材；以上为当次部署的历史记录。

### 中文文件名修复的独立提交验证

在任务管理提交 `9b709be` 基础上单独纳入本修复的实现、测试和文档，共8个文件；
42个诊断工具和8份CyberTom原始录音仍留在本地。2026-10-29重新验证命令：

```sh
cmake --build /tmp/resonantia-web-release-host --target test_web_audio_upload --parallel 2
ctest --test-dir /tmp/resonantia-web-release-host -R '^test_web_audio_upload$' --output-on-failure
/tmp/resonantia-web-release-host/test_web_audio_upload /usr/bin/ffmpeg /home/rm01/Resonantia/tests/fixtures/speech_00.wav
cmake --build /tmp/resonantia-web-release-host --parallel 2
ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure --parallel 1 --no-tests=error
bash scripts/check_sdd.sh
git diff --cached --check
```

增量host构建成功；focused **1/1，0.52秒**，完整host契约 **19/19，4.57秒**，均0失败/skip。
真实短人声12组codec和同一MP3的ASCII/中文文件名解码一致性通过；fixture摘要与
[来源记录](../tests/fixtures/README.md#real-human-speech)一致。SDD32项通过。
证据位于 `runs/Git_Unicode_Upload_20261029/evidence/`，这次未重跑GPU模型或生产浏览器测试，
也未重新部署服务。远端CI结果以该独立提交对应的GitHub Actions记录为准。
