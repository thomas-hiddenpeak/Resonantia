# Resonantia 项目状态

**状态日期：2026-10-31（正式200轮模型已更名Tom Cybersinger，8个旧声线已清理；修正版完整翻唱听审通过，正式声线及去和声工作流已部署Web，通用生产质量仍未验证）**

本文件是当前实现与验证结果的**描述性快照**，不是替代 feature spec 的规范来源。
规格意图和需求归属见 [`specs/README.md`](README.md) 及各 feature spec；可执行边界以
`docs/DESIGN.md`、CLI `--help`、当前 CMake/CTest 注册和实现中的输入校验为准。
未来目标、历史研究结论和配置结构中的预留字段都不等于已支持能力。

## 验证基线

### 2026-10-31 FLAC上传与长立体声文件边界（已部署）

翻唱上传解码上限由旧mono的64MiB修正为双声道128MiB/600秒；原始文件64MiB、
默认mono解码64MiB及每批新解码产物128MiB保持。大PCM16预览以独立快照分块发送，
float32清理预览转PCM16副本。同步修正内部44.1k双声道float32读取上限为256MiB，
保留600秒与其他格式128MiB限制，避免下一清理阶段及混音拒绝长歌曲的中间文件。

两套构建成功；host **21/21（6.37秒）**、asset-free contract **45/45（36.63秒）**，
0失败/skip；I/O/上传/真实codec重点门禁 **3/3（7.86秒）**，4项Node UI通过。
真实人声循环7分钟FLAC解码74,088,078字节，完整预览字节一致；实际无模型CLI重读
7分钟float32，旧程序失败、新程序成功且文件逐字节一致。不宣称整曲模型质量通过。

04:22:27 CST更新`vc_serve`和`vc_preprocess`并重启，内网/本机HTTP200，PID3025559。
Tom正式模型/索引、训练及转换程序SHA未变，7个历史任务状态保留，无活动任务。
旧清理CLI缓存仍严格校验程序摘要，不绕过兼容门禁；旧结果可播放，重跑可能需新任务。
用户实际《山丘.flac》尚未取得，未跑训练/整曲转换或远端CI。命令、红测、边界和备份见
[解码上限修复记录](003-webui/decoded-upload-limits-20261031.md)。

### 2026-10-30 修音页面流程补齐（已部署）

短试听可从任务中心直接返回对应疑点，显示采用/保留原音/待听审计数与真实试听状态；
支持下一处、采用后撤销及“生成完整新版”修正数量。同一浏览器恢复选中位置和采用草稿，
试听即使不在当前历史页也按ID核对后找回；完成/暂停试听复用原任务，不重复提交。
过期分析、配方不符或来源缺失拒绝采用，错误不能被当作新建试听的依据。

host **21/21（5.06秒）**、asset-free contract **45/45（38.35秒）**，0失败/skip；
四项Node UI契约、任务中心/工作室/引导式Chromium通过；
真实8秒既有任务只读联调通过：返回、音频播放、采用/重开恢复/撤销、390px布局，0写请求、
0页面异常。17:46:46 CST仅替换静态页面，服务PID2600879及模型/程序摘要不变；内网实际
《富士山下》任务20疑点、原生播放、失败父版恢复通过，0写请求/页面异常。
本轮没有新生成模型音频或训练；部署与完整命令见
[页面流程验收](003-webui/pitch-review-workflow-20261030.md)。

后续Git发布前，两套构建成功；host **21/21（6.15秒）**、contract **45/45（36.69秒）**，
0失败/skip。修音浏览器runner补齐标准依赖查找并接入远端CI，3个Chromium runner及
4项Node UI在不设置NODE_PATH时通过。诊断工具、录音和模型仍不进入Git，完整命令见上述记录。

### 2026-10-30 引导式音高检查与局部试听（已部署）

成功转换可只读分析音高，标记疑似短八度变化；曲线/同位置循环试听/建议预览/采用撤销
已接入，手动修音收于高级区。预览有独立后台记录，可关闭页面、暂停及重启恢复，最多
重算两个受影响原合成核心，保留全部输入上下文；不会改写原任务或自动运行整曲。

最终host **21/21（5.81秒）**、asset-free contract **45/45（35.11秒）**，0失败/skip；
四项Node UI、工作室/任务中心/引导式Chromium与axe规则通过。8秒真实模型预览与独立
重算的320000个PCM样本一致，暂停后重启恢复成功；完整《富士山下》副本仅重算1/9核心，
33个F0块及其余8核心payload不变，41–49秒试听保留。原父任务332份文件与事前副本一致。

Web于01:02:15 CST更新，内网只读浏览器确认43.36/45.68疑点及失败父版恢复，0写请求/异常。
计算CLI、正式模型/索引保持原SHA；没有生产转换、长训练、全部model/slow回归或远端CI。
维护者对41–49秒修正对照反馈“确实改善了”；仅确认该片段改善，不宣称通用错音判断
或全曲音质已通过。完整命令、初次失败及证据见
[引导式修音实施记录](003-webui/guided-pitch-review-20261029.md)。

### 2026-10-29 历史任务局部修音入口（已部署）

局部修音原嵌在临时结果区，历史任务派生后随结果区隐藏。现改为独立可见编辑区，任务中心
提供直接入口；平滑跨无声帧的失败版本可恢复自己的设置，以原成功版本为来源创建新版。
进入编辑器不提交任务，原版/失败版均保留；已知错误显示中文并阻止不改设置的直接继续。

旧页面浏览器红测复现；最终任务中心与工作室 Chromium 验证、4项 UI 契约通过，含失败
版本刷新恢复、来源缺失、320/390/768/1440布局与axe。host **19/19，4.81秒，0失败/skip**。
生产内网只读浏览器验证实际《富士山下》成功/失败两版入口与参数恢复，0写请求、21份
任务/音频/模型/程序摘要不变。只替换静态页面，服务PID2339081保持；未新运行转换/训练，
不宣称自动错音检测、曲线编辑或局部重生成。详见[本轮验收](003-webui/pitch-editor-20261029.md)。

### 2026-10-29 任务重命名与双音频结果（已部署）

任务中心可修改显示名称，原训练目录/声线标识/参数和进度保持；完整训练身份明确排除显示字段。
默认歌曲完成后直接显示完整翻唱和转换后无伴奏人声，分别播放/下载；旧任务已有的人声可直接
访问，清理混音任务副本不再删除独有人声。已删除的人声显示缺失，不自动重算。

最终完整构建、host **19/19（7.80秒）**、contract **44/44（35.59秒）**通过，0失败/skip。
Chromium验证名称草稿/冲突、双播放器、鉴权、4档布局与axe；原生HTTP/浏览器在实际已完成歌曲
副本验证改名重启保留、两份音频字节一致、Range及清理保留。没有执行新模型转换或训练。
首次完整回归暴露显示字段被计入训练身份，修复后按原门槛通过，详见
[本轮交付记录](003-webui/task-results-20261029.md)。

19:08:34 CST更新为PID2339081/active。原生内网浏览器看到两个完整时长播放器和重命名入口，
只读验证、写请求0；两份生产任务记录/原素材/结果/人声、正式Tom模型及四个计算CLI摘要不变。

### 2026-10-29 跨Web更新的任务身份修复（已部署）

用户对完成歌曲创建`-12`半音新版时，被`task dependencies or recipe changed`拒绝。
旧Web身份隐式绑定服务可执行文件，仅更新Web也会使原任务不匹配；此前同版本测试及只读页面
验收漏掉跨部署执行。现以显式Web编排版本绑定完整依赖，原生计算CLI仍绑定实际程序摘要。
仅兼容已审计的09:27/14:04两个旧Web摘要，重算全部依赖后精确匹配，原标记不改写。
音频、训练外层与嵌套切片身份一并修正，训练数值和检查点格式未改变。

完整目标构建、当前树contract **44/44（41.08秒）**通过，0失败/skip；独立host此前同一逻辑
**19/19（4.61秒）**通过。两个旧服务的真实6秒CPU预处理跨版本迁移及再次更新通过；实际失败
任务副本保留`-12`并导入分离/清理，在模型调用前停止，未重跑整曲或训练。
15:18:25 CST部署为PID2302330/active，内网入口和任务查询正常，原记录/源/结果/正式模型及旧
workflow标记不变，失败任务可由用户继续。范围、首次失败及回退见
[修复验收](003-webui/task-revisions-20261029.md#跨web更新的身份修复2026-10-29)。

### 2026-10-29 完成音频调整参数（已部署）

任务中心新增“调整参数再生成”：恢复原素材和设置，建立独立新版；根据实际变化复用已校验的分离、
清理和转换阶段，旧记录/音频不变。来源任务不是共享可变缓存，新版独立持有导入产物。
仅更换人声/混音输出不重复模型计算，改清理设置重做清理及转换；坏身份/缓存拒绝，不放宽数值门槛。
完整构建、host19/19（5.33秒）、contract44/44（35.92秒）及Chromium恢复/布局/axe通过。
真实6秒变调新版与同主唱独立推理WAV逐字节一致；仅改混音复用全部模型阶段。
14:04:18 CST内网服务更新为PID2291009/active，计算CLI未变化，正式Tom SHA不变。
原生内网浏览器在用户261.576984秒完成歌曲上验证原素材/参数可恢复，0写请求；未重跑整曲或训练。
首次失败、精确命令和回退见[参数修订验收](003-webui/task-revisions-20261029.md)。

### 2026-10-29 后台长任务改进（已部署并完成本轮验收）

旧歌曲261.577秒任务最终报“伴奏合成失败：音频长度或格式不匹配”，临时产物已被旧服务清理，
不能恢复或记作成功。44100→40000→16000→40000→44100整数舍入产生的3帧差已建立回归并修复；
真正截断仍拒绝。没有重跑该整首歌曲。

Web已更新为后台持久任务：转换、预处理和训练在关闭页面后继续，任务中心可恢复历史状态、
真实阶段/计数、日志、设置和结果。提交响应丢失沿原ID核实，暂停/继续带执行代次；服务重启后
标为中断并由用户明确继续，不保证保住尚未提交的更新。完整joint-full训练支持检查点恢复；
旧decoder/gan/joint及独立手动step保留历史，但不宣称等价的分块/完整续训能力。

已纳管任务使用64GiB总预算、24GiB单任务和8GiB磁盘预留；到保护线后在安全点暂停，不是硬配额。
成功任务可清理冗余副本，保留原始素材、结果、日志、完整检查点、候选和正式模型。新音频结果保存
SHA256，清理前复核；旧无摘要记录仍可查看/下载，不开放清理。清理不删除历史ID。

最终完整构建通过，host **19/19（4.41秒）**、contract **44/44（35.09秒）**、codec
**1/1（2.88秒）**通过，0失败/skip；部署后实际页面再验UI **4/4（1.09秒）**。
真实模型门禁包括12秒转换暂停/服务重启/继续后WAV逐字节一致、3步训练从step2继续，
异常退出从已提交step0恢复至step3，以及初始化中断恢复。最新接入磁盘保护的真实6秒转换与
1步训练，在浏览器关闭后完成，新窗口找回音频/候选；结果摘要保护另以6秒CPU预处理验收。
Chromium替身门禁覆盖断网、鉴权、响应丢失、失败终态、同ID重试、4档布局与axe。

部署时间2026-10-29 09:27:37 CST，用户服务resonantia-web.service PID2258518/active。
内网入口、任务/磁盘/声线API正常；原生内网6秒CPU预处理经关闭整个浏览器context后重新打开，
可找回音频并Range206读取。正式Tom模型SHA不变，未新增长训练或整曲模型转换。
测试中发现的候选缺失、待确认配方误拒绝f0、清理未校验结果等问题均有红/绿记录；早先一次上传
并发资源门禁失败根因未确认，后续原门槛顺序回归通过。详细命令、首轮失败、证据、回退位置见
[实施清单](003-webui/task-lifecycle-20261029.md)。当前注册的全部测试并未在本增量全部重跑。

### 2026-10-29 中文文件名上传修复

`单依纯 – 我管不住我自己.mp3`因文件名过滤删掉所有非ASCII字符，只剩`.mp3`，
在解码前被误判无扩展名。已改为保留合法UTF-8并维持路径、编码、255字节及归一化
重名校验。修复前focused测试及生产HTTP均复现400；修复后完整目标构建成功，
Web CTest **18/18通过，0失败/skip，16.03秒**，含真实短人声12组codec及同名MP3。
真实内网浏览器使用该文件名的一秒MP3上传返回200、WAV预览一秒、原名显示正确。
服务已更新并重启，PID2034441。没有取得用户整首原文件，未执行模型转换或训练，
不将同名短音频验证视作整首文件验收。详细命令、回退和证据见
[工作室交付记录](../docs/WEBUI_STUDIO_20261029.md#中文文件名上传修复2026-10-29)。

### 2026-10-29 工作室业务流程与视觉重设计

维护者要求按实际业务重设计并自审后推送。Web现为制作翻唱、训练声线、声线库三条路径；
先上传与确认输入类型，任务概览显示处理链，生成后进行原声/结果同位置听审。训练启动
需摘要确认，活动任务锁定配置；候选试听与发布独立管理。正式Tom Cybersinger模型和
参数、已验证的去和声处理保持。依据、审计发现、修复、精确命令与边界见
[工作室交互记录](../docs/WEBUI_STUDIO_20261029.md)。

已有完整目标构建成功，Web18/18通过；暂存发布内容的干净副本host构建及11/11门禁通过，
同一副本浏览器验收通过。三页/结果/确认弹窗的axe规则、键盘、失败重试、设置与结果区分、
非法修音保留结果、空列表、活动训练恢复、320/390/768/1440布局均验证。API替身不执行
模型，额外真实HTTP上传/候选浏览器另验，不冒称新增声音质量或重跑此前66项模型测试。

新版已部署至原内网入口，PID2020827；开发期旧页面保留为回退副本。
新增`web-studio-browser` CI门禁，精确远端提交的结果以本轮交付链接为准。

### 2026-10-29 正式声线与Web工作流交付

正式声线现为`Tom_Cybersinger`，显示名Tom Cybersinger；模型/Index SHA与
`CyberTom_Full_20261026`的epoch200训练结果一致。按维护者后续要求，8个旧声线的
模型/索引及其目录内候选副本已实际删除（21个文件，共2133.416748MiB）。
Web默认选择正式声线及听审参数，加入歌曲/含和声人声/干声工作流、立体声去和声、伴奏混音、
可选去混响、人工局部音高、完整训练默认值、历史/恢复/候选试听/确认发布。正式训练
素材、checkpoint、候选和原录音保留。训练阶段进度与计算秒/步取自真实计数，不提供伪整体百分比。

完整66项CTest首轮64通过、2失败；HTTP失败替身改为完整PCM后通过，整机设备余量门槛
在移除额外编译负载后按原256MiB标准复查通过（长短差194.8828125MiB）。2/2复查通过，
最后Web错误分类与UI增量再验18/18，独立host11/11；没有降低数值或内存门槛。
真实28秒清理相对L2为5.34e-8；相同主唱输入的局部修音与已听审结果逐采样一致。
真实6秒HTTP歌曲混音等长/有限/无削波，日志确认Orin GPU。实际内网地址上的Chromium
桌面/手机布局、独立标签页、上传、参数锁定、候选与试听/发布入口通过；浏览器转换按钮
测试使用拦截响应，真实模型HTTP证据另行保存，未混淆两者。

声线改名后服务已重启，MainPID2005146，`http://192.168.0.122:8080/`仍为入口，
voices只返回`Tom_Cybersinger`且保持听审标记。
完整命令、首轮失败、复查、模型身份、边界和回退位置见
[Web交付记录](../docs/WEBUI_RELEASE_20261029.md)；本地证据根为
`runs/CyberTom_Web_Release_20261029/evidence/`。未追加长训练或完整歌曲转换。

后续改名与旧版清理另验：唯一声线专用预设先红后绿，Web CTest **18/18通过、0失败/skip，
16.00秒**，实际内网浏览器验证唯一声线、名称、请求标识及正式训练历史通过。证据为
`runs/Tom_Cybersinger_Web_20261029/evidence/`；本次未重新构建C++、未执行模型转换/训练，
也未将此前完整66项结果称为改名后重跑结果。

### 2026-10-29正式训练完成与局部翻唱修正

`CyberTom_Full_20261026`已完成200轮/81200次更新，四个阶段候选、最终model/Index及
step81200恢复点存在；同一Web作业done/error=false。日志逐步计数及loss有限性核查通过，
但独立持续资源采样已按维护者要求停止，因此不能宣称测得全程资源峰值。精确范围、计时、
证据路径及未完成门禁见[完整训练执行记录](002-voice-training/full-training-20261026.md#2026-10-29完成核查与翻唱听审)。

epoch200短翻唱听审通过；原完整翻唱因2:30后的音高/音量突变未获接受。尾段先分离主唱
后转换获“大幅度改善”，再对“无/要”两处共17个F0帧进行局部调整，维护者反馈“两处都
改善了”。基准重放逐样本一致，局部音频等长/有限/无削波；这是具体样本的改善证据，
不是全曲或通用质量通过。251.16秒完整修正版已生成，前149.80秒与已听审人声段206.8..217.0秒
逐样本保留，后半段采用主唱分离与三处短淡化；随后维护者确认完整版本听审通过，Web交付见上。
该离线诊断阶段没有新的生产C++/CUDA变更或CTest运行；下文66/66
是开训前的实际回归记录，不冒称本轮重新执行。

以下2026-10-26的运行中、预计完成时间及未执行验收等描述保留为当日历史记录，不能覆盖
本节最新状态；F6集中验收仍部分完成，F7新Web声线交付现已完成。

### 2026-10-26完整训练目标启动

维护者已批准能力补齐，并要求按计划推进200轮正式训练。目标及验收见
[完整训练计划](002-voice-training/full-training-20261026.md)，架构迁移见
[ADR-0009](../docs/adr/0009-resumable-joint-training.md)。纳入全部粤语/串烧候选素材，
先验证去混响；性能在训练中记录。正式Web任务现已启动，运行进展见本节末与执行记录。
既有62项完整回归属于上一版，不替代新实现的测试证据。

库内恢复增量：先记录AdamW和JointTrainer缺少API的编译红测，之后实际恢复测试暴露
U8状态与既有F32-only loader不兼容。修复为F32精确字节编码，没有扩大loader接受范围。
`cmake --build /tmp/resonantia-performance-20261024.uemhTG --target test_joint_training_models test_joint_training -j2`
成功；随后`ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^(test_joint_resume|test_joint_training|test_joint_numerical|test_autograd)$' --output-on-failure`
**4/4通过、0失败、0 skip，51.29秒**，日志`runs/CyberTom_Full_20261026/evidence/resume-green-v3.log`。
包含真实G/D与语音的保存/销毁/恢复/下一步、全参数和moment、随机状态及推理保留值验证，
未知版本/负variance/截断拒绝且原对象保留；原严格独立单步继续通过。
后续CLI/Web接线与完整CTest结果见下文；正式训练的运行状态另外记录。

素材听审已获明确接受：“去混响有改善，音色保留，可以采用”。8份原始录音经现有Web
dereverb_roformer处理完成，源文件hash保持；《劲歌金曲》的Node客户端300秒header超时，
服务器继续计算，输出inode保留后按完整WAV帧数/hash核验，其余请求HTTP200。
M4A的AAC容器padding以及44.1k/40k多次整数重采样须与实际解码帧数区分，失败核查记录保留。
固定清单406段/1077.43秒，4份留出85.95065秒且相邻2秒不训练；22份不足0.6秒的碎片剔除，
没有音量归一化/压缩。配置为200轮、已验证batch1、81200次更新，详见执行计划与
`runs/CyberTom_Full_20261026/{configuration.json,dataset/manifest.json,dataset/verification.json}`。

F2默认门禁：SHA-256已知向量、schedule/裁剪RNG恢复、缓存原子往返/损坏拒绝及CLI契约
**4/4通过，6.84秒**（`f2-host-green.log`）；Web新旧模式/取消/候选保留与恢复路由通过
`test_web_train_cancel`，UI四项通过（`f2-web-green.log`、`f2-ui-green-v2.log`）。
真实资产focused **5/6通过，64.72秒**（`f2-real-tests.log`）：原joint数值、库内resume、
autograd及两个缓存/状态测试通过，新增整作业CLI因HuBERT目录未转成权重文件失败，已修正。
之后独立新启两次训练的四步比较仍失败（`f2-integrated-green.log`，日志名不代表通过）：
判别器一个moment tensor相对L2约0.01727，门槛0.001；差异在取消之前已出现。
该对照混合了不同训练前缀与恢复行为；新增以同一第5轮checkpoint分叉、比较第6次更新
的隔离验证，另验实际取消/恢复与数据变化拒绝。阈值和原独立golden不变；当前仍在验证。
隔离首轮仍在单个`conv_k.bias`上记录RMS约6.53e-6（`f2-resume-isolation.log`）。
核对原`test_joint_numerical`后发现新增CLI测试按单tensor施加了原本逐模块的RMS门槛；
现按相同G模块和各D子判别器计算门禁，并继续输出单tensor偏差诊断，不承诺逐tensor/bitwise重放。
同时把库内resume新增测试的D聚合拆回各子判别器，非零moments统一使用原相对L2门槛，
取消其误用的微小参考RMS特例。批准的模块参数RMS1e-6、非零moment相对L2 1e-3未放宽；
原独立单步/梯度参考与测试不变，修正后的恢复门禁需重新执行。

复验`ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^(test_joint_resume|test_joint_full_cli)$' --output-on-failure`
**2/2通过、0失败/skip，368.25秒**（`f2-authoritative-gates.log`）。包含同一第5步状态
在连续运行与新进程恢复后的第6步逐模块参数/矩比较、严格非零moment门槛、完整随机状态，
另有实际协作取消/恢复、数据变更拒绝和旧输出保留。新整作业test实际296.36秒，
其操作包含五次真实进程的权重身份校验/恢复，将功能测试timeout从300调至480秒，
没有调整数值门槛。单tensor RMS仍作为诊断，不承诺跨进程bitwise或逐tensor误差门槛。
CLI源码此后补充缓存准备结束时的音频身份复核和阶段日志，并避免最终候选重复写出；
Web模式切换增加回归，离开完整训练清除其epoch预算，防止带入旧GAN。
`cmake --build /tmp/resonantia-performance-20261024.uemhTG -j2`完整构建成功；
`ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG --output-on-failure`
**66/66通过、0失败、0 skip、退出0，1006.47秒**；原始输出与详细测试日志存档为
`evidence/full-ctest.log`、`full-ctest-LastTest.log`、`full-ctest-result.json`。
之后仅补充UI刷新重新跟踪活动训练，先以缺少函数失败，再运行
`ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_web_.*_ui$' --output-on-failure`
**4/4通过、0失败/skip，1.03秒**（`ui-restore-red.log`/`ui-restore-green.log`）。
覆盖无上传时跟踪/取消正确id、非train/终态忽略、网络失败及迟到状态不覆盖新作业。
这是mock DOM/fetch验证，不冒称浏览器视觉验收；全量之后未修改C++/CUDA。
六个CLI已从验证构建原子安装，服务PID由221611变为288927，内网主页及新profile验证通过。
旧CyberTom/CyberTom_20261026模型/Index SHA前后不变；回退材料与全部hash在
`evidence/deployment.json`。正式Web任务`104144f937c65cdaedf25365847bde7b-1`已接受，
name=`CyberTom_Full_20261026`，mode=joint-full、epochs200、seg32、prepared1、vad0。
服务固定clips逐份406/406与数据清单hash一致；输入不重复清理/切片。
406份冻结缓存已完成、state0恢复点已写入，正式更新持续运行，尚未完成200轮。
离线只读观察器由用户systemd服务运行，
每5秒记录GPU负载/频率、进程RSS/HWM、MemAvailable及进度到`performance/`，
不启动/停止训练。日志在`train.log`；不以进程已启动推断音质或长期稳定性通过。
开始运行后发现旧UI在特征准备时仍显示固定55%；仅修改JS为核验/缓存段数/更新步数分别
显示，新阶段不被追加日志中的旧step覆盖。对应红测为“核验”缺失；相同4项UI复验
**4/4通过、1.14秒、0失败/skip**（`ui-cache-progress-red.log`/`ui-cache-progress-green.log`），
内网实际JS hash与验证源码一致，没有重启服务或中断训练。
`evidence/performance-report.cjs`只读取本次日志和采样，输出阶段耗时/分位数/资源实测；
首轮解析漏掉字段名f0的数字，报告生成失败并保留`performance-parser-red.log`，修正后通过。
此报告不含参考项目速度对比或完成时间估计，也不代表GPU核函数隔离计时。
首25步全部loss有限：训练计算均值2.82836秒、缓存读取均值0.0508123秒（两者合计中1.7648%）；
一次性准备1085.21秒，初始checkpoint保存9.10907秒。训练阶段13个GPU采样点负载
中位98.6%，进程HWM约4428MiB，系统MemAvailable最低约7305MiB；这是启动局部样本，
不证明全部200轮稳定或与原项目相比更快。完整数值见`evidence/performance-first-updates.json`。
重新核查398组同源训练/留出区间及2秒guard无重叠，整首天地未进入训练；五个验收输入
hash不变（`evaluation-input-audit.json`）。20个候选转换及25份ASR清单已准备，未执行。
第1轮406次更新实际完成并进入第2轮：核对初始checkpoint记录的唯一排列与每次日志frame/crop，
全部匹配；step/epoch/clip计数一致，loss有限，第406步LR=9.99874974e-5符合FP32衰减。
首轮计算平均2.65517秒/步、缓存读0.0520827秒/步，累计1077.9995+21.1456秒；
`epoch1-schedule-audit.json`和`performance-after-epoch1.json`记录证据与范围。
离线验收脚本`tools/evaluate_cybertom_audio.py`已准备，四个纯DSP单测通过，缺输出时预检失败
且不创建结果目录；未运行真实候选RMVPE/ASR分析，不将合成DSP测试当作音质证据。

维护者随后要求暂停持续复查、按当前速度估时后再检查。最近200步含缓存平均2.6677556秒，
654步时剩余59.688小时，预计2026-10-28 15:26（北京时间）完成更新，18:00安排设备本地
一次性只读检查并保存`scheduled-review-result.json`。这是线性估计，含明确误差边界，非完成保证。
独立5秒采样服务已按请求停止（uv收到TERM退出143，非训练失败），C++训练进程288995及Web
仍运行，原生日志保留。代理持续目标暂缓；本会话没有聊天定时唤醒工具，本地timer不保证聊天回复。
配置/语法检查及timer已启用、下次触发时间证据见`evidence/scheduled-review-setup.json`。

### 2026-10-26 CyberTom保守动态版交付

维护者对`listen_centered_tom15_rms0.wav`回复“音色保住了，动态也可以接受”。
此前两份联合训练短音频的自然度/厚度已获接受、Tom音色改善；100ms全频增强方案
被否决并保留失败记录，运行时没有加入该方案。这里是具体CyberTom短样本的接受，
不把Spec 002的通用生产质量、长训练、resume等未满足需求改成Complete。

修复后的完整构建成功，串行CTest **62/62通过、0失败、0 skip，693.80秒，退出0**；
详见`joint-entry/dynamics-20261026/full-ctest.log`、`full-last-test.log`与命令JSON。
该全量结束后补充声线预设自动选择，4项Node UI **4/4通过，1.01秒**；
红测为缺少声线预设处理块，绿测验证参数、异步加载次序、切换声线及刷新不重置手动编辑。
完整CTest存档在00:45:55完成，UI复验在00:46:51结束，原全量详细日志保留完整。

- 发布声线名`CyberTom_20261026`，从独立Web47步产物复制model/index/clips及来源清单；
  前置单步1+取消2+主训练47合计50步，此后没有追加训练。
- Web预设`CyberTom · 保守动态`在选择该声线时自动带入index0.35、RMS0、protect0.5、
  filter3、pitch0；sampled/seed0继承CLI默认。刷新同一声线不覆盖用户调整。
  同时修正旧预设文案把RMS1错误称作“保留源动态”的问题，旧预设数值未变。
- 服务无活动任务、子进程或HTTP连接时备份六个CLI并停止/替换/重启。运行PID从
  3692644变为221611，service active，运行中exe及六个CLI SHA-256与已验证构建一致。
  unit、端口、绑定地址不变；根build只安装CLI，其余旧静态库/测试仍不作为当前验证依据。
- 内网地址`http://192.168.0.122:8080/`，主页、JS、CSS、预设、voices、status均HTTP200。
  声线列表和自动预设均核对；原`runs/CyberTom`模型和Index前后hash不变。
  备份与回退材料在`build/.joint-release-20261026.ktox2kyy/`，未触发回滚。
  精确部署命令与新旧hash在`joint-entry/deployment.json`及该目录的脚本文本。
- 正式Web以相同输入及交付预设完成15秒短转换，HTTP200、26.071秒、40k、无NaN/Inf或截幅。
  与C++保守预览配平前音频的相关性0.9999999624、相对L2 2.75664e-4，差异来自诊断中
  对中间PCM多量化一次；原预定门限0.999/1e-3通过。听审预览另施加一个固定整体增益
  供等音量比较，正式Web保留源音量。记录`dynamics-20261026/production-check.json`。
- 正式Web输出再经相同离线Whisper-small无prompt识别，文本与原音及之前的联合训练
  版本一致；`delivered-asr-results.json`记录约7.99秒ASR耗时。原音本身存在识别错误，
  不将一致文本当作人工转录准确率或音色分数。
- 独立127.0.0.1:8081验证服务已在空闲时结束，正式8080保持active。

最终`scripts/check_sdd.sh`通过32项、`git diff --check`通过。源代码和文档未作新的Git提交或推送。

验证命令（构建目录同上）：

```sh
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG --output-on-failure
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_web_.*_ui$' --output-on-failure
python3 /tmp/resonantia-deploy-joint-20261026.py
uv run --project tools --no-sync python /tmp/resonantia-joint-production-check-20261026.py
```

### 2026-10-26 动态保持与RMS对齐修复

联合训练接线后的完整构建和串行CTest **62/62通过、0失败、0 skip，728.01秒，退出0**。
这是本节RMS修复前的基线，日志为`joint-entry/full-ctest.log`/`full-last-test.log`，
包含35 contract、28 integration（1项重叠），codec ON、slow Web deadline OFF。
维护者先接受两份留出样本自然度/厚度及Tom音色改善，随后补充“声音缺少了动态，感觉上了压缩器”。
该阶段曾因质量问题暂缓部署；后续保守版听审及实际交付见上节。

证据在`runs/CyberTom_Joint_20261025/experiment/joint-entry/dynamics-20261026/`：

- 同一两段<=15秒输入、同模型/Index/seed，仅将RMS设为1取回未匹配的模型输出。
  《需要人陪》50ms活动区间P90–P10从原音19.42dB缩为模型11.30dB，原mix0.5恢复到13.37dB；
  Tom对应12.03/9.91/9.93dB。指标不是感知分数，不能把模型变化全归因于输出RMS。
- 原RMS实现采用向前窗口、尾窗变分母和端点插值；独立上游采用居中补零、完整窗口分母、
  half-pixel插值。默认仍为1秒/0.5秒hop；已同步修正bounded与buffer路径，不增加公共选项。
- 新独立算子测试修复前 **22个断言失败**（66中44通过），修复后通过；既有全波形PCM规则
  回归同步使用新居中契约。冻结常量由`tools/gen_rms_operator_reference.py`提取原上游函数，
  不是拿C++输出作golden。原联合训练的23个文件未改。
- 使用完全相同的C++ filtered source与原始模型PCM，对两段真实音频、mix0/0.5独立比对。
  旧相对L2为0.0421–0.1446；修复后为3.90e-6–5.24e-6，最大PCM差1个量化单位，
  相关性均>0.99999999998。记录`centered-real-alignment.json`；这只证明RMS运算对齐。
- 离线100ms全频包络使《需要人陪》的上述强弱差恢复至19.15dB、Tom至12.01dB；
  预览只做固定整体增益配平，无压缩/限幅、无训练。维护者反馈“动态改善了，但是音色变差
  和有齿音”，随后明确Tom整段不好、翻唱好得多；**该方案未通过、未接入运行时**。
- Tom短窗增益变化在活动帧中P95为每10ms 0.54dB，最大1.53dB；包络相关的最佳整体lag为0。
  高频候选帧的分析不足以证明所有齿音都被放大，不将猜测当根因。记录
  `timing-and-modulation.json`、`sibilance-gain-analysis.json`及`human-feedback.json`。
- 已送一份保守C++试听`listen_centered_tom15_rms0.wav`：只修正1秒包络并选mix0，
  固定整体增益配平；随后听审获接受，见上节。没有追加训练，本轮训练总数仍50步。

完整构建成功，RMS/CLI/pipeline/E2E focused 4项通过，28.84秒；完整回归见上节。
主要复验命令（RMS探针只做CPU后处理，不训练或重跑模型）：

```sh
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R 'test_bounded_pipeline_dsp|test_cli_contract|test_pipeline_contract|test_e2e_conversion' --output-on-failure
uv run --project tools --no-sync python tools/probe_joint_dynamics.py runs/CyberTom_Joint_20261025/experiment/joint-entry/dynamics-20261026
```

### 2026-10-25 联合训练CLI/Web与限定短训练

证据目录`runs/CyberTom_Joint_20261025/experiment/joint-entry/`；下节61项全量是接线前
基线。新增`--gan-profile joint`显式入口、Web `mode=joint`、`training_profiles`能力声明和
旧服务兼容UI。默认旧GAN/decoder保持原样，joint限制50步、crop32、epochs0、sampled、
decay1；逐clip重算冻结前端，40k/0.32..6秒输入在CUDA前校验，不缓存数据集频谱/prior。
CLI频谱使用同一私有helper并对原冻结真实频谱回归，未更新golden。

- 先失败：CLI不认识profile；40k/长度预检3例旧路径进入模型加载；新UI无能力处理块，
  随后旧缓存HTML兼容失败。Web逆转本次接线的独立控制二进制因未声明joint失败。
  Web新测试初稿因Request不可复制/移动的编译失败保留但不计缺陷红测。
- 完整构建成功，相关CTest **22/22通过、0失败、0 skip，31.81秒**，包含新的
  `test_web_joint_ui`和原严格整步门禁；日志`full-build.log`、`focused-ctest.log`。
- 真实CLI单步：`one-step/monitor.json`命令退出0；训练含前端4.861秒，50ms采样RSS峰值
  4,722,972 KiB、系统最低可用9,525,428 KiB、GPU峰值999/1000。单步输出独立保存。
- 真实GPU取消：`real-cancel/result.json`。第1步完成后写取消标记时第2步已开始，
  完整第2步后退出2，10.091秒，旧输出保留，未发信号。此短例不证明所有取消时延边界。
- 独立Web `127.0.0.1:8081`，使用修复后的CLI和独立runs，通过`/api/material`上传已有
  3段普通话干声共60秒，`/api/train`完成47步，训练188.649秒、平均4.01382秒/步。
  前置1步+取消2步+47步合计50步，没有继续训练。3首是《千万次地问》《我是一只鱼》
  《有没有一首歌会让你想起我》，原始上传hash与旧manifest核对；未用两首指定粤语歌。
- `web-terminal.json`记录done/error=false，21个clips、2979条检索向量及模型一并提交。
  `export-verification.json`验证560 tensor形状/dtype/有限值；457个推理tensor变化：
  decoder243、enc_p113、Flow100、embedding1；其他speaker行及原enc_q保持逐值一致。
  模型SHA-256 `f2e677da201c3e784c3b1b5503f6ef643837908717a89cfc5465a17b26982754`。
- Web监视从训练中途开始，系统最低可用9,577,016 KiB、GPU峰值999/1000；直接子进程
  RSS采样未匹配到CLI，原报告0不能作为零内存或峰值。保留原始记录，修正解释在
  `web-resources-corrected.json`；不以此替代前述独立单步/6秒资源资格化。
- 两份留出短音频经同一Web `/api/convert`成功返回40k WAV：Tom普通话15秒，
  《需要人陪》完整唱句10.7秒，检索率0.35、pitch0、rms/protect0.5、filter3、sampled/0。
  HTTP墙钟28.824/43.069秒，均有限且无PCM截幅，已送聊天听审；没有整曲转换。
  请求、来源和hash在`listening/conversion-results.json`；导出检查脚本最初误用无前缀
  embedding key而失败，改为实际`model.*`后通过，未修改模型文件。

离线Whisper-small固定revision、CPU FP32、beam1、无文本prompt的ASR已完成（`listening/asr-results.json`）。
Tom原音及转换音识别文本完全相同；《需要人陪》原音本身含识别错误，转换音也有若干词替换。
没有独立人工转录，未把ASR间一致率当成准确率或音色指标。
2026-10-26维护者听审回复：“自然度和厚度可接受，Tom 音色也有改善”，原话及两份样本范围
记录于`listening/human-feedback.json`。此接受仅覆盖上述两个留出短样本，不外推为整曲、
长训练或任意说话人泛化质量。后续将这份已接受模型接入现有Web，不追加训练。
本阶段62项CTest结果及后续动态听审问题见上节。正式8080服务未重启，正式CyberTom未替换，
未作Git提交或推送。短训练执行及数值通过仍不等于FR-014自然度/厚度/Tom相似度验收。

主要命令（精确训练argv/HTTP参数见上述JSON，不应为复验而再次消耗50步预算）：

```sh
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R 'test_web_|test_training_cancel|test_cli_contract|test_joint_' --output-on-failure
uv run --project tools --no-sync python /tmp/resonantia-joint-listen-20261025.py
uv run --project tools --no-sync python tools/probe_cybertom_asr.py runs/CyberTom_Joint_20261025/experiment/joint-entry/listening/asr-manifest.json runs/CyberTom_Joint_20261025/experiment/joint-entry/listening/asr-results.json
```

### 2026-10-25 联合训练前向修复：严格单步通过

本轮证据在`runs/CyberTom_Joint_20261025/experiment/forward-repair/`。下文的
`cpp-step-v5`、`strict-followup/`和`numerical-repair/`失败属于此前版本，保留原始记录。
当前树没有替换冻结参考、放宽阈值、改变AdamW epsilon或注入参考激活符号。

- CPU/aarch64、2线程参考在单组1D输入大于20480元素时使用oneDNN/ACL通道后置
  计算，kernel tap在外、输入通道在内、bias起始；大输入转置卷积tap倒序。CUDA
  tile实现这些顺序，period判别器宽度1卷积逐列保持相同规则。
- 小输入普通卷积按连续640项块内FMA、块间单独相加，bias起始；小输入转置卷积
  仍按通道归约、重叠叠加、最后bias。参考NVPL 0.3.0的640块宽经真实入口输出验证，
  未复制其实现代码。说话人单向量条件采用四路FMA及相邻两路先合并。
- 真实decoder六层的固定输入前向现全部逐值一致，原输入/g/v/bias梯度门禁继续通过。
  decoder入口16,384个输出、两组条件6,656个输出也与独立CPU结果逐值一致。
- NSF host计算被编译器合并FMA，与上游分开的张量运算产生相位误差。将相位、音调
  幅度、随机幅度乘积显式物化为FP32后，真实12,800点音源RMS误差由`1.76349e-7`
  降至`1.27391e-9`；该问题是最后使严格单步失败的一项已定位原因。生产实现使用
  同一私有helper，独立sin函数的末位差异未通过替换golden消除。
- 先失败证据：5个真实大输入层逐值检查失败；说话人条件禁用修复的控制二进制
  7/7断言失败；小卷积15/16失败；period卷积27/32失败；NSF 6个采样点失败。
  修复后的相关测试全部通过。新增条件测试最初误用2D linear布局，已按实际RVC
  Conv1d调用修正，其原始11个失败不计缺陷证据；原23文件整步参考从未更改。

相关4项CTest **4通过、0失败、0 skip**，8.58秒（`excitation-tests.log`）：
`test_joint_training`、`test_autograd`、`test_joint_numerical`、`test_joint_kernel_gradients`。
其中原16,477断言的整图数值门禁现在通过，包括各模块梯度、参数更新及moments。

资源复验：真实T=600、crop=32，首/中/尾各两次前向反向完成，0次optimizer更新；
首步3.077秒，后续2.555–2.604秒。采样RSS峰值5,067,952 KiB，系统最低可用
8,935,756 KiB，GPU负载峰值999/1000。固定累加顺序比上一版约1.5秒的稳定步慢，
这是精确对齐的性能代价，不是GPU未启用。50ms采样仅覆盖进程RSS/系统共享RAM，
不能当作独立VRAM峰值或长期稳定性证据。导出560 tensor与基础模型逐值一致；
原23文件size/SHA-256一致（`identity-check.json`）。

完整构建成功，串行CTest **61/61通过、0失败、0 skip，673.40秒，退出0**。
包括旧decoder/GAN、实际格式解码、Web、E2E/分块及有界资源门禁；slow Web deadline
配置仍OFF，未计入。完整日志为`full-ctest.log`/`full-last-test.log`。
最终独立导出探针再次通过原比较器：音频相对L2 `7.02319e-7`、相关性
`0.999999999999774`；各模块梯度最大相对L2 `8.31857e-5`、moments最大
`8.31797e-5`、参数更新最大RMS `8.86684e-7`。0步与1步导出归属检查通过。
命令为`tools/probe_joint_training.cpp`上述构建产物以reference目录运行至`final-step/`，
随后执行`tools/compare_joint_training.py reference final-step --base <原G>`；
精确结果在`final-step/comparison.json`，机器摘要为`verification-summary.json`。
`scripts/check_sdd.sh`与`git diff --check`通过。
此阶段尚无CLI/Web joint profile和新听审语音；后续接线与短训练见上节。
数值一致性不是自然度/声音厚度/Tom相似度验收，FR-014仍未满足。

本轮主要验证命令（仓库根目录）：

```sh
cmake --build /tmp/resonantia-performance-20261024.uemhTG --target test_joint_training test_autograd test_joint_training_models -j2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^(test_joint_training|test_autograd|test_joint_kernel_gradients|test_joint_numerical)$' --output-on-failure
c++ -O3 -std=c++20 -Iinclude -Isrc -I/usr/local/cuda-13.2/targets/sbsa-linux/include tools/probe_joint_training.cpp /tmp/resonantia-performance-20261024.uemhTG/libvoxmutatio.a -L/usr/local/cuda-13.2/lib64 -Wl,-rpath,/usr/local/cuda-13.2/lib64 -lcudart -lcublas -lcufft -lcublasLt -lculibos -lpthread -ldl -lrt -o /tmp/resonantia-joint-probe-forward-repair
uv run --project tools --no-sync python tools/probe_joint_resources.py runs/CyberTom_Joint_20261025/experiment/forward-repair/memory /tmp/resonantia-joint-probe-forward-repair models/pretrained_v2/pretrained_v2/f0G40k.safetensors models/pretrained_v2/pretrained_v2/f0D40k.safetensors runs/CyberTom_Joint_20261025/experiment/resource-fixture runs/CyberTom_Joint_20261025/experiment/forward-repair/memory --memory
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG --output-on-failure
```

### 2026-10-25 联合训练实施与单步门禁

维护者明确批准ADR-0008：先单步和内存验证，再最多50步短训练，不跑长训练，
产出的试听继续送聊天听审。ADR已Accepted；以下是实施证据，不是音质验收。

- 新增库内`JointTrainer`，对完整有界clip训练enc_p、enc_q、Flow、decoder和选定
  speaker行；保留g/v weight norm与显式条件梯度。HuBERT/RMVPE仍是冻结输入。
  新RVC Mel配置为40k/2048/400/125，旧配置默认保留；每步刷新posterior/NSF噪声。
- 推理导出保留560 tensor schema，更新enc_p/Flow/decoder/选定speaker，保留原始enc_q
  及其他speaker行；不是checkpoint/resume。0步逐值一致和1步导出归属检查通过。
- 补齐同字节数但错误轴顺序的模型拒绝测试：`enc_q.pre.weight`由`[192,1025,1]`
  改成`[192,1,1025]`曾被接受，现按完整G/D manifest在设备分配前验证；失败重初始化
  不得再step或导出。失败3断言与修复后通过记录在`joint-model-{red,green}.log`。
- 修复abs在0处应为0、Leaky ReLU在0处应取负侧斜率的导数；新测试先复现2断言失败，
  再通过。autograd取消`--use_fast_math`，保持FP32计算，推理组件保留原编译设置。
- 独立上游CPU FP32、2线程、真实1.28秒唱句、固定噪声、32帧decoder crop进行一次
  D/G更新。前向latent相对L2约`1e-6`、音频约`1e-5`量级通过；固定真实音频的RVC Mel
  梯度相对L2 `2.4122e-5`通过。参考及全部输入/参数/梯度/moment哈希见
  `tests/fixtures/joint_training_manifest.json`，禁止以C++重放结果替换独立参考。
- **严格单步门禁仍失败**：`test_joint_numerical`已注册并实际失败，保留原阈值。
  精确数学配置的独立探针`cpp-step-v5`中decoder/embedding/posterior梯度相对L2分别
  约`0.003935/0.003547/0.002344`；部分一步参数和moments也未达标。
  导出正确、梯度非零、loss接近均不替代该门禁。
- 已定位其中一项判别器偏差：上游period-5 conv3的真实人声激活，FP32为
  `+5.07018e-8`，FP64为`-3.32551e-9`，落到Leaky ReLU不同分支。仅上游自身
  FP32/FP64梯度相对L2即`0.00140302`，与C++该模块约`0.00140248`接近。
  这是数值敏感性证据，不足以解释全部G偏差，也不豁免已批准的0.001门槛。
- 真实600帧/6秒输入、32帧crop，首/中/尾各两次前向反向、**0次optimizer更新**：
  后5次约1.47–1.50秒/次，进程采样RSS峰值5,075,020 KiB，系统最低可用7,794,672 KiB，
  GPU负载峰值998/1000；未见连续增长，非法601帧/非有限音频/尾部越界均拒绝。
  Jetson为共享RAM，记录的是50ms采样RSS与系统可用内存，不冒称独立VRAM精确峰值；
  该检查只资格化已测的32帧crop，不能推广为任意decoder长度或数据集规模已验证。

证据目录：`runs/CyberTom_Joint_20261025/experiment/`。构建目录
`/tmp/resonantia-performance-20261024.uemhTG`，Release/SM87，model/codec tests ON，
slow Web deadline test OFF。关键命令（从仓库根目录运行）：

```sh
uv run --project tools --no-sync python tools/gen_joint_training_reference.py <new-reference-directory>
uv run --project tools --no-sync python tools/compare_joint_training.py <reference> <cpp-output> --base models/pretrained_v2/pretrained_v2/f0G40k.safetensors
uv run --project tools --no-sync python tools/probe_joint_numerical_sensitivity.py <reference> <new-report.json>
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j 2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_joint_model_contract$' --output-on-failure
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_joint_numerical$' --output-on-failure
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG --output-on-failure
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_joint_kernel_gradients$' -V
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_bounded_file_resources$' -V
```

本阶段CLI/Web尚未接入joint profile，最多50步Pilot尚未启动，未替换正式CyberTom、
未重启服务、未进行整曲转换。数值fixture仅0.32秒，不作为完整唱句试听。FR-014仍未满足。
全量构建已成功。完整60项CTest运行58通过、2失败、0 skip，耗时636.97秒：
`test_joint_numerical`为上述已知严格门禁；`test_bounded_file_resources`在并行CPU参考计算
期间因共享内存增长差值超限失败（短/长1,078,702,080/1,562,714,112 bytes，允许差268,435,456）。
该资源测试单独重跑仍失败（121.16秒，2断言失败）：短/长RSS为1,342,028/1,458,384 KiB，
设备可用量推导的增长为240,533,504/1,501,761,536 bytes。
因此不能简单归因于并行参考计算，也不能计作通过；尚未定位是实际工作集差异还是
共享内存/进程测量问题，未调整其64 MiB host/256 MiB device阈值。新补充的
`test_joint_kernel_gradients`另行运行通过（6个真实尺寸层、246断言，0.37秒），
输入/原始g/v/bias梯度最大相对L2约`2.392e-6`，保留整图门禁失败。
固定decoder输入及输出梯度时，上游自身FP32/FP64梯度差`0.00157502`，
C++相对上游FP32差`0.00238405`；仍不足以宣称全部偏差都已解释。
本轮后续增加的第61项测试已独立执行，不能把两次运行包装成单次61/61全通过。

最终库构建再次运行同一600帧/32帧crop资源探针（`memory-final/`）：6次前向/反向成功，
0次optimizer更新；首步1.93秒、其后1.49–1.57秒，采样RSS峰值5,067,700 KiB，
系统最低可用7,712,556 KiB，GPU负载峰值998/1000。推理导出560 tensor逐值与原始G
相同，未进行训练发布。该短训练资源检查与上面的有界文件转换资源失败是不同门禁。
`scripts/check_sdd.sh`通过（32 required files），`git diff --check`通过。

维护者随后明确选择“保持原门槛，继续定位”；不批准跳过失败进入50步实验。
后续排查保留原参考、原阈值和所有失败证据，不把分层测试或诊断重放视为整图通过。

#### 保持原门槛后的定位

本轮新增证据保存在同一实验目录的`strict-followup/`。重新逐文件核对冻结fixture的
23项size/SHA-256，全部一致；没有替换参考、屏蔽失败测试或启动50步训练。

- 解码器77个Leaky ReLU位置逐层记录。固定C++ latent与输入梯度时，有13个接近零的
  元素发生正负分支分歧；保持上游前向不变、只在诊断反向重放这些符号后，decoder
  梯度相对L2由`0.002383998`降到`2.65450e-6`。独立C++ decoder探针与完整C++步的
  decoder梯度差仅`2.92015e-7`，确认探针重现了原路径。生产构建的观察宏为空操作。
- 完整G反向对照固定C++音频及更新后的D，再重放decoder反向符号；使用上游原生
  posterior时共有18个分歧元素。decoder/enc_q/emb_g梯度相对L2分别从
  `0.00269177/0.00140198/0.00244782`降到`5.11430e-6/4.20569e-6/5.27627e-6`；
  enc_p/Flow为`3.29909e-5/8.40120e-7`。这些是有干预的归因实验，**不是原门禁通过**。
- D的9个子模块固定相同音频分别反向，period 3/5/17各发现一个分支分歧。period-5
  梯度相对L2由`0.00140281`降到`2.72026e-6`；诊断符号重放后9模块最大误差
  `2.79642e-6`。与上游FP32/FP64控制一起，证据指向近零激活的导数分支敏感性，
  而不是仅凭最终loss接近推测反向正确。上述G/D定位探针均为0次optimizer更新。
- 独立CPU AdamW使用**完全相同的C++梯度**重算一步，14模块的参数/两个moment共
  42项比较通过；参数最大绝对RMS`2.67183e-8`、moment最大相对L2`9.70571e-7`。
  原enc_p参数更新平方误差的`96.2919%`集中在6个attention key bias，其梯度L2仅
  `1.11e-8`至`1.51e-7`，符号/舍入差被`eps=1e-9`的AdamW放大。没有在生产代码中
  清零这些梯度、修改epsilon或注入参考符号；原更新门禁仍须独立通过。
- 有界推理资源继续拆分：显式`cudaMalloc`峰值短/长为133,693,440/136,314,880 bytes，
  9,060/16,994次成功分配均配对释放，未出现observer容量溢出。它只覆盖显式分配，
  不能代表驱动或共享系统RAM总峰值。该次原CTest仍失败：RSS
  1,340,104/1,457,440 KiB，共享设备可用量推导的增长699,752,448/1,661,669,376 bytes。
- glibc大块分配探针定位主机活跃分配峰值在VITS残差卷积，短/长为
  549,312,912/557,749,456 bytes。此前已用完的`xc/up/xsrc`仍与残差大数组同时存活；
  现提前释放这些stage输入及已消费latent，保持公式、累加顺序、随机源、输出和接口。
  原64 MiB host/256 MiB device阈值不变。修改前带该观察器的一次CTest通过
  （RSS 1,454,236/1,451,020 KiB），表明全局内存/驻留测量存在运行间波动；不能把
  偶发通过或单次采样低峰值解释为此前失败已经解决。

缓冲生命周期修改后，同一glibc观察器的短/长活跃分配峰值为
394,063,232/400,555,216 bytes，分别减少155,249,680/157,194,240 bytes（约28%）。
带观察器的原资源CTest通过（135.86秒；RSS 1,345,148/1,341,832 KiB，设备增长
786,771,968/750,796,800 bytes）；无插桩完整测试结果另行记录，不能用该诊断运行替代。
全量构建成功；先执行的8项相关CTest全部通过、0 skip（21.17秒）：autograd、joint
contract/model/kernel、synth ops/failure、VITS alignment/debug。新增5个Python诊断
脚本语法解析通过；其真实模型执行结果已分别保存在JSON和日志中。

最终无插桩完整CTest：**61项，60通过、1失败、0 skip，674.46秒，退出码8**。
唯一失败为`test_joint_numerical`（16,477断言中19失败），仍保留原梯度、参数及moment
门槛。旧decoder/GAN、VITS/RMVPE/FCPE/HuBERT/WavLM、E2E、两种有界分块和Web
相关已配置测试均通过；slow Web deadline选项仍未启用，不能计为已执行。
`test_bounded_file_resources`无插桩通过（131.45秒）：短/长RSS
1,343,304/1,343,408 KiB，设备可用量推导的增长762,388,480/799,739,904 bytes，
均在原64 MiB/256 MiB差值门槛内。生产窗口planner对两个原始输入长度均产生最大
256,000-sample F0窗口（`f0-window-shapes.log`），未通过缩短处理范围降低资源。
完整输出保存在`strict-followup/full-ctest.log`及`full-last-test.log`；原失败记录保留。

本轮构建及验证命令（工作目录为仓库根目录）：

```sh
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j 2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^(test_vits_alignment|test_vits_gen_debug|test_synth_ops|test_synth_cuda_failure|test_joint_training|test_joint_model_contract|test_joint_kernel_gradients|test_autograd)$' --output-on-failure
c++ -shared -fPIC -O2 -std=c++20 -I/usr/local/cuda-13.2/targets/sbsa-linux/include tools/probe_cuda_allocations.cpp -ldl -lpthread -o /tmp/resonantia-cuda-allocations-20261025.so
c++ -shared -fPIC -O2 -std=c++20 tools/probe_host_allocations.cpp -ldl -lpthread -o /tmp/resonantia-host-allocations-20261025.so
env LD_PRELOAD=/tmp/resonantia-host-allocations-20261025.so RESONANTIA_HOST_ALLOCATION_TRACE=/home/rm01/Resonantia/runs/CyberTom_Joint_20261025/experiment/strict-followup/host-after-lifetime ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_bounded_file_resources$' -V
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG --output-on-failure
scripts/check_sdd.sh
git diff --check
```

定位工具、用途边界和重现命令见`tests/fixtures/README.md`。当前仍需解决冻结参考下
近零分支与参数更新的整步对齐；归因完成不等于生产数值修复完成。未接入joint CLI/Web，
未发布新的CyberTom，没有生成适合听审的新样本。

#### FP32算子修复与回归（续）

本轮继续落实运行时代码修复，证据目录为
`runs/CyberTom_Joint_20261025/experiment/numerical-repair/`。未放宽ADR-0008原门禁，
未替换独立参考，未清零小梯度或注入参考反向符号，未启动50步/长训练或发布声线。

- weight norm由混合double/FP32归约改为与冻结参考相同的四路FP32顺序，反向按
  `a*dw-b*v`分别舍入。真实三个权重矩阵的范数与CPU参考逐值一致。
- Softmax的分母/反向点积改为四路归约，归一化采用倒数相乘，修复长行抵消误差。
  原先失败的`enc_p`参数更新检查在此修改后通过，但这不代表其他模块通过。
- 分组卷积在乘积归约后加入bias；转置卷积先做通道归约、再叠加重叠tap、最后加入
  bias，避免小bias在大数相消前被舍掉。转置卷积通道归约使用CUDA共享内存tile和
  固定FMA顺序；新增workspace索引溢出拒绝，临时columns不由反向图长期持有。
- 新增四组`test_autograd`回归覆盖上述差异及尾向量、重叠输出、512输入通道。
  原始日志保留：weight norm长度1025/4097共5038个输出断言失败；Softmax反向
  97个失败；bias顺序2个失败；宽转置卷积11个失败。修复后的这些检查全部通过。
  weight norm首次新测试中4/17长度的两个scale常量误由Tensor倒数乘法生成，随后
  按原生weight norm的标量除法修正；首次日志另有20个由此导致的失败，不计入缺陷
  复现。这只更正新增小型单元测试，原23文件整步参考未变。构建中一次workspace
  声明位置错误已修复；失败构建日志仍保留。

全量构建成功。最终相关7项CTest运行6通过、1失败、0 skip（51.38秒，退出码8）：
autograd、joint contract/model/kernel、旧decoder和GAN通过，`test_joint_numerical`
仍失败。整步失败断言由19降为12；当前decoder/embedding/enc_q梯度相对L2约
`0.003194/0.002005/0.001764`，仍高于`0.001`；decoder/enc_q参数RMS约
`5.623e-6/4.559e-6`，仍高于`1e-6`。D的所有梯度比较通过，但period-2参数更新
RMS约`1.719e-6`仍失败；moment等其余失败保留于日志。不能把减少失败数量解释为
已经满足训练或音质验收。

新构建重新执行6秒真实输入（T=600、crop=32；首/中/尾各两次）的资源探针，全部
前向/反向完成，0次optimizer更新；首步2.102秒，后续1.530–1.572秒。采样进程RSS
峰值5,069,772 KiB，最低系统可用8,797,524 KiB，GPU负载峰值998/1000。该采样包含
导出过程，不能当作独立VRAM峰值；不据6次运行推断长期稳定性。导出的560个tensor
逐值与基础模型一致；重新核对原23个fixture的size/SHA-256全部一致。

最终串行完整CTest：**61项，60通过、1失败、0 skip，651.31秒，退出码8**。
唯一失败仍为`test_joint_numerical`（16,477断言中12失败）。旧decoder/GAN、Web、
实际音频格式、前端、合成和E2E/分块门禁通过；slow Web deadline配置未启用，未计入。
有界资源测试139.47秒通过，短/长RSS为1,344,892/1,341,284 KiB，设备可用量推导的
增长为782,962,688/833,495,040 bytes，原64 MiB host/256 MiB device门槛未变。
完整输出及详情保存在`full-ctest.log`、`full-last-test.log`；`scripts/check_sdd.sh`
与`git diff --check`通过。一份误与focused尾段重叠启动的全量运行已主动中断
（退出130），日志`full-ctest-overlap-aborted.log`保留，不计验收。
Web服务未重启，正式模型未替换，没有生成可听审的新语音。剩余严格单步一致性
失败仍阻塞CLI/Web joint接线、最多50步短训练和后续音质验收；可用性目标尚未达成。

本轮主要命令（仓库根目录）：

```sh
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j 2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^(test_autograd|test_joint_training|test_joint_model_contract|test_joint_kernel_gradients|test_joint_numerical|test_generator_trainer|test_gan)$' --output-on-failure
c++ -O3 -std=c++20 -Iinclude -Isrc -I/usr/local/cuda-13.2/targets/sbsa-linux/include tools/probe_joint_training.cpp /tmp/resonantia-performance-20261024.uemhTG/libvoxmutatio.a -L/usr/local/cuda-13.2/lib64 -Wl,-rpath,/usr/local/cuda-13.2/lib64 -lcudart -lcublas -lcufft -lcublasLt -lculibos -lpthread -ldl -lrt -o /tmp/resonantia-joint-probe-numerical-repair
uv run --project tools --no-sync python tools/probe_joint_resources.py runs/CyberTom_Joint_20261025/experiment/numerical-repair/memory /tmp/resonantia-joint-probe-numerical-repair models/pretrained_v2/pretrained_v2/f0G40k.safetensors models/pretrained_v2/pretrained_v2/f0D40k.safetensors runs/CyberTom_Joint_20261025/experiment/resource-fixture runs/CyberTom_Joint_20261025/experiment/numerical-repair/memory --memory
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG --output-on-failure
scripts/check_sdd.sh
git diff --check
```

### 2026-10-24 CyberTom 重新训练与听审

用户否决旧 CyberTom 的电音效果后，先用三首普通话干声各20秒做流程试训。
`CyberTom_Pilot`（GAN 100步）、`CyberTom_Pilot_Decoder`（decoder 50步）均经
Web material/train 接口完成；`CyberTom_Pilot_LowLR`复用同一21片段，用同一
`vc_train` CLI执行decoder 50步、学习率`1e-5`。后者使用CLI是因为Web尚无
学习率字段；前两者使用默认`2e-4`。各实验的命令、输入清单、模型/Index哈希、
转换参数和原始ASR结果保存在相应`runs/<name>/experiment/`（本地忽略资产）。

离线ASR辅助脚本为`tools/probe_cybertom_asr.py`，通过既有tools uv环境运行，
不进入产品运行时。使用本地`openai/whisper-small`固定revision
`973afd24965f72e36ca33b3055d56a652f456b4d`，CPU/FP32、greedy、固定语言；
真实LibriSpeech文本只在解码完成后计算WER，不作为提示词。输入限制为30秒以内。
中文清唱只有原音频ASR对照，没有人工逐字标注，不能据此宣称真实CER为零。

12.485秒英文、检索率0.35的WER：GAN短训31.25%（10/32）、decoder短训25%
（8/32）、低学习率短训0%（0/32）；原音频3.125%（1/32）。低学习率另两段
5.855/4.815秒英文各有一个`MISTER`/`Mr`差异，与原音频一致。**人工听审优先：
用户认为50步decoder内容不可辨、低学习率版虽能听出内容但仍基本是电音，
因此三版均未通过声线音质验收。** ASR不能替代可懂度听审、音色相似度或伪影评价。

用户随后要求使用完整原始素材按Web高音质档重新训练。独立作业
`CyberTom_HQ_20261024`使用六首歌，排除《别来无恙》《劲歌金曲》，并从
《天地在我心》留出8–23秒供试听。Web归一化、trim/3秒切片后得到212片段、
630.02355秒，40kHz；无削波样本。配置为GAN 2800步、segment 40帧、默认
学习率`2e-4`、seed 0，无额外分离或降噪。2800步已完成，优化耗时3214.441521秒；
训练导出560个有限tensor，Index为31,291×768。作业`done`只表示产物发布成功。
**人工听审明确否决`hq_mandarin15_index035.wav`：“全是电音”，本轮训练质量失败。**
旧正式`CyberTom`及所有原始素材保留，失败实验没有替换正式声线；“高音质”是现有
预设名称，不是验收结论。`training-result.json`已记录`quality_accepted=false`。

F0修正后，同一HQ模型12.485秒英文的WER为：检索率0.35时56.25%（18/32），
关闭检索37.5%（12/32），Web balanced参数40.625%（13/32）。旧F0行为、检索率0.35
时为31.25%（10/32）。另外5.855/4.815秒为11.76%（2/17）/60%（6/10）。
因此**不能把F0修正报告为HQ可懂度改善**；本轮多个样本未达到预定WER≤20%的辅助门限。
中文仅有原音频ASR对照，无人工逐字标注，不能报告真实CER。详细参数、哈希、音频和
结果位于`runs/CyberTom_HQ_20261024/experiment/`（本地忽略资产）。

另按用户要求通过Web测试工作区《需要人陪》FLAC的45–65秒片段：
`/api/preprocess`的vocal_roformer分离耗时29.16秒，`/api/convert`对同一20秒干声
分别使用HQ及低学习率模型，检索率0.35、pitch=0、RMS=0.5、protect=0.5、filter=3，
耗时42.80/36.07秒。三份干声均完成本地Whisper检查；HQ的ASR改写多于低学习率版，
但原始分离人声的ASR也有错字。**人工反馈：“原始分离人声清楚，两版转换都明显电音”。**
该片段说明失真主要在分离之后出现，不能据此定位到某一个变声模块；两版均不合格。
未执行整首251.16秒转换。命令、输出、ASR和听审保存在`experiment/cover/`。

### 2026-10-24 F0 默认行为修正及验证

同一预训练模型、真实语音、相同特征的离线参考对照中，仅将无声帧F0插值的C版本
被用户指出明显电音；噪声开关A/B差别不大。训练使用原始零F0，推理原默认则将其填满。
`VCConfig::interp_unvoiced`默认改为`false`，显式`true`保留旧行为；median filter仍
独立生效。此修正与训练失败并存，不代表已解决所有电音或音色问题。

构建目录`/tmp/resonantia-performance-20261024.uemhTG`，Orin NX/SM87、CUDA 13.2、FP32：

```bash
cmake --build /tmp/resonantia-performance-20261024.uemhTG --parallel 2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_bounded_pipeline_dsp$' --output-on-failure
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L contract -LE 'integration|slow' --parallel 1 --output-on-failure --no-tests=error
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^(test_e2e_conversion|test_cli_bounded_callers|test_long_audio_chunking|test_bounded_file_fcpe_v1)$' --parallel 1 --output-on-failure --no-tests=error
```

新增DSP回归在旧默认下失败（1 case/4 assertions），修正后1/1通过。完整目标构建成功，
32/32 contract通过（24.76秒），4/4真实资产integration通过（312.71秒），均无失败/跳过。
两组共36个distinct tests；当前共注册57个，另外21个本轮未执行，不能称57/57全量通过。
真实E2E波形相关性0.999716；RMVPE/v2 bounded/legacy频谱相关性0.99246934，
FCPE/v1为0.99985907。数值对齐不等于人工音质合格。
日志与命令保存在`experiment/f0-*.log`、`f0-verification-results.json`。

验证后已替换Web调用的`build/vc_convert`及`build/vc_batch`，回滚副本位于
`build/.f0-default-20261024.nz00aajq/previous/`，部署哈希见`experiment/f0-deployment.json`。
服务每次转换启动CLI，因此此次无需重启；内网首页、voices和训练状态HTTP均为200。

### 2026-10-24 生成器输出激活缺陷与修复

对完整HQ权重做同输入、无检索、无随机噪声的独立参考对照，C++/PyTorch波形相关性
只有0.995365（门限>0.999）。分层检查显示prior和Flow的相对L2约为4.7e-7/3.2e-7；
即使给C++ decoder直接输入参考latent，输出偏差仍存在。最终定位到`conv_post`前
LeakyReLU：参考`F.leaky_relu(x)`默认负半轴系数0.01，而C++推理及autograd训练
两处均误用了其他层的0.1。两处均已修正，其他激活保持原值。

旧预训练模型输出相关性0.999716仍高于原门限，掩盖了约4.1235%的相对L2误差。
在`test_vits_alignment`与`test_generator_trainer`增加独立真实音频参考的相对L2
`<1e-3`后，两项均在旧实现失败（CTest退出8）；修复后分别为0.000364068和
0.000364192，两项通过。相同HQ权重修复后的C++/参考波形相关性为
0.9999999094，PCM16相对L2为0.00066294（也低于0.001）。数值缺陷已复现并修复，
**尚不能认定它解释了全部主观电音，旧权重是在错误实现上优化的，音质仍未通过。**

最小修复为`src/synthesizer/synthesizer.cpp`与`src/training/generator_trainer.cpp`
最后激活的两个系数；未改变格式/API，未启动新一轮长训练。兼容边界见
[Spec 001 §2.2](001-voice-conversion-engine/spec.md#22-生成器输出激活对齐修正2026-10-24)。
前述F0修复的验证记录是历史阶段；本次代码修改后重新构建并验证：

```bash
cmake --build /tmp/resonantia-performance-20261024.uemhTG --target test_vits_alignment test_generator_trainer --parallel 2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^(test_vits_alignment|test_generator_trainer)$' --parallel 1 --output-on-failure --no-tests=error
cmake --build /tmp/resonantia-performance-20261024.uemhTG --parallel 2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L contract -LE 'integration|slow' --parallel 1 --output-on-failure --no-tests=error
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^(test_gan|test_vits_gen_debug|test_e2e_conversion|test_cli_bounded_callers|test_long_audio_chunking|test_bounded_file_fcpe_v1)$' --parallel 1 --output-on-failure --no-tests=error
```

全部目标构建成功；focused 2/2（22.37秒）、contract 32/32（24.69秒）、其余相关
integration 6/6（317.67秒）均通过，0失败/跳过；共40个distinct tests，另外17个
注册测试本次未运行，**不是57/57全量结果**。`scripts/check_sdd.sh`及`git diff --check`
另行验证。日志、前后对照和命令证据保存在
`runs/CyberTom_HQ_20261024/experiment/export-alignment/`。

2026-10-24 08:32（Asia/Shanghai）在无模型作业时部署`vc_convert`、`vc_batch`、
`vc_train`；原程序保存在`build/.decoder-output-20261024.KdCNGG/previous/`。
Web继续按请求启动新CLI，无需重启常驻服务；内网首页、voices、训练状态HTTP为200。
原正式CyberTom模型/Index SHA-256复核未变。部署前后各二进制哈希见
`experiment/export-alignment/deployment.json`。

离线辅助工具`tools/probe_cybertom_posterior.py`使用固定真实英文、原始F0和未微调模型
重建5.84秒语音，绕过HuBERT/prior/Flow/Index；完整posterior均值/采样与40帧posterior
拼接（均整段decode）的ASR各2/17错误，原音频1/17。分段posterior的latent相对L2变化
约0.216，但这不等于可听伪影或训练根因。用户听审确认`source.wav`无电音、`posterior_full_mean.wav`尾段有电音，
基础重建仍未通过自然度验收，原听审记录保存在`experiment/posterior/human-review.json`。
`tools/probe_cybertom_reference.py`与`tools/probe_cybertom_synth.cpp`保存了独立参考及
导出模型逐层对拍入口；这些是离线诊断，不构成新的产品运行依赖或通用质量保证。

旧HQ权重通过修正后的Web再转同一15秒中文（检索率0.35，其余参数不变）耗时33.79秒，
`experiment/export-alignment/hq_mandarin15_outputfix.wav`的ASR仍明显偏离原音频，
不能据此恢复HQ质量资格，且该文件尚无人工听审。随后独立建立
`CyberTom_Pilot_OutputFix`：复用原21片段及Index、decoder 50步、学习率1e-5、segment 40、
seed 0，从原始预训练权重开始，使用修正后的训练器，耗时64.10秒。Web没有学习率字段，
因此此对照使用同一C++ CLI训练，再通过Web转换验收；不是一次完整HQ重训。
导出560个有限tensor、239个decoder tensor变化；模型SHA-256为
`57a257243649af4c4db016353db482f488e11c3071a60917f20d784d0589f58c`。
数据/命令/二进制哈希/输出保存在`runs/CyberTom_Pilot_OutputFix/experiment/`，
正式CyberTom及原素材不变。该声线仅为独立试验，已被人工自然度听审否决。
该新模型经Web生成12.485秒英文和15秒中文（均检索率0.35），分别耗时23.45/26.96秒。
固定Whisper英文WER=0/32，中文ASR与原始干声ASR文本相同；满足本轮短样本ASR辅助门，
但中文没有人工逐字标注，不能称真实CER=0。音频无削波，`quality-result.json`保持
`quality_accepted=false`，用户随后确认两份输出均“整段有电音”，状态更新为
`failed human review`。ASR可懂不等于自然度合格；本轮未再启动长训练或整首翻唱。

### 2026-10-24 RMVPE错误频率刻度修正

根据用户“原音频无电音、posterior重建尾段有电音、OutputFix两版整段电音”的反馈，
进一步发现`src/f0/rmvpe.cpp`与旧`tools/gen_rmvpe_reference.py`共同错用Slaney频率刻度。
独立上游RVC前端明确为HTK刻度、Slaney面积归一化；旧对拍只能证明两份错误实现一致。
同一真实LibriSpeech row-0上，新旧358个共同有声帧的音高偏差中位数558.94 cents。

修复前针对直接上游生成的`rmvpe_htk_*`运行`test_rmvpe_alignment`失败（exit 8）：
salience RMS 0.0596794，366个参考有声帧无一满足0.5 Hz，平均/最大误差32.1378/109.032 Hz。
修正后salience RMS为1.61189e-7，F0平均/最大误差2.04493e-5/0.000106812 Hz，
全586帧UV一致。新`e2e_htk_audio.bin`完整233600样本波形通过既有相关性门禁；
旧VITS fixture不改写，作为固定输入算术测试保留，不再把其旧F0当音高准确性证据。
Bounded profile原门限不变：1757帧UV一致率0.999431，mean/p95为0.015771/0.0460739 Hz，
边界max 0.0607681 Hz，最大输入217040 samples，小于256000上限。

新增默认滤波器系数测试；第一轮暴露8000 Hz端点的浮点反变换残差，因此端点以1e-12
绝对容差检查零，其余系数1e-8；未放宽任何既有模型误差门限。完整build通过，
`ctest -L contract --output-on-failure` 33/33通过（32无模型+1 WavLM资产边界），25.96秒。
其余24项非contract门禁全部通过（597.37秒）；两组不重叠，当前57/57 CTest通过、
0失败/CTest跳过，合计623.33秒。完整控制台结果和命令/退出码见`verification-results.json`；
并行执行注册清单查询覆盖了CTest的`LastTest.log`，故不宣称保留了本次所有case的详细输出，
核心数值门禁详细输出仍保存在此前focused的`green-detail.log`。

正确F0、原预训练G的两份5.84秒对照：posterior直接重建ASR为1/17，与原音频同水平；
完整基础模型转换为2/17。用户听审：“都有改善,但是也还略有电音”。因此本修复已证明
会改善听感，仍不能宣称自然度合格；不将结果归因为所有电音的唯一原因。原有HQ/低学习率/
OutputFix模型均使用过错误F0标签，保持未通过质量验收。新的50步HTK训练对照`CyberTom_Pilot_HTK`已完成（63.93秒），使用同21片段/原始预训练G/
学习率1e-5/seed0，从修正前端重新提取F0；不是完整重训。560个tensor有限，
239个decoder tensor变化，其余与预训练G完全一致。模型SHA-256为
`1e80ae60ae469d1fea2c0c6b70949d31a0b6b399c9b9d1027c1b25e28c89f5cb`，
试验目录`runs/CyberTom_Pilot_HTK/experiment/`记录数据哈希、命令、导出校验及质量状态，
该独立声线已接入Web。相同Web参数（检索0.35、pitch0、RMS/protect0.5、filter3）
生成12.485秒英文/15秒中文/6秒翻唱，用时25.24/28.59/15.01秒。英文ASR为0/32，
中文ASR与此前原干声代理文本相同；6秒翻唱原片与输出的ASR均不足以判定可懂度，
不计为质量通过。均无削波。用户随后听审确认“有改善,但还是有电音”，
因此HTK短训版仍未通过自然度验收，记录更新为`failed human review`、
`quality_accepted=false`。不能把音高修正或ASR通过视为电音完全解决。

2026-10-24 09:30 CST，`vc_convert`/`vc_batch`/`vc_train`整组部署并重启Web（PID 3423900），
内网根页面、voices与training status均HTTP200；原正式CyberTom模型/Index哈希未变。
备份为`build/.rmvpe-htk-20261024.d332z30d/previous/`，二进制哈希与回滚依据见
`experiment/rmvpe-htk/deployment.json`。

命令、输入/输出/上游源代码哈希、red/green日志、ASR与听审记录位于
`runs/CyberTom_HQ_20261024/experiment/rmvpe-htk/`；新fixture provenance位于
`tests/fixtures/rmvpe_htk_manifest.json`，复现说明见[fixtures README](../tests/fixtures/README.md)。

### 2026-10-24 HTK修正后残余电音：随机成分对照

HTK短训仍被用户否决后，检查发现当前`Synthesizer::infer`只使用prior均值，
`sine_source`把NSF excitation noise设为零；上游正常推理使用prior noise scale
0.66666以及有声0.003/无声0.1÷3的NSF噪声。既有数值门禁使用noise=0的参考，
因此只证明该确定性路径一致，不能证明其与正常随机合成的听感等价。
这目前是质量假设，不是已证明的残余电音根因。

扩展离线`tools/probe_cybertom_reference.py`，固定HTK短训模型、真实LibriSpeech
5.84秒、正确HTK F0、speaker0、无Index/前端高通等额外处理，生成四种控制：
全关、仅NSF、仅prior、两者恢复。prior/NSF使用独立固定seed 20261024/20261025，
保留相同噪声抽样以隔离变量；FLOAT WAV不做自动增益或削波。四组Whisper均2/17错误，
均无削波，ASR不能据此选择更自然的版本。给用户的A/B分别是全关/两者恢复，用户确认“B 明显更自然、少电音”。
该结论支持恢复采样合成，但不代表最终CyberTom音质/相似度已经通过。

用当前库重新编译独立C++诊断程序，与A对拍：233600样本，相关性
0.9999999249253975、相对L2 0.0004624262414481486（含C++输出PCM16量化），
满足既有>0.999/<1e-3门限。说明A代表当前确定性合成实现；B仍为离线参考实验。
本轮仅改离线工具及听审/状态记录，没有修改或部署运行时，也没有新一轮训练；
前节57/57是上一轮运行时修正的验证记录，本轮没有重复运行整套CTest。
命令、退出码、输入/模型哈希和产物见
`runs/CyberTom_Pilot_HTK/experiment/noise-ablation/diagnosis.json`。

### 2026-10-24 采样恢复的运行时修正与听审

用户确认B明显更自然后，C++默认恢复prior/NSF采样，保留显式mean对照及固定seed。
独立上游网络使用真实语音/HTK F0和明确的同一高斯输入：修复前相关性0.954476、
相对L2 0.298519，修正后相关性约1、相对L2 0.000177476；原mean参考继续通过。
训练前处理同样使用配置模式，仍按clip缓存，不宣称逐优化step重新抽样。

首轮扩大回归发现分段与整段的频谱相关性0.9724196971，未达到原>0.98门限。
当时每个chunk从局部索引0重新使用噪声，后改为传入补边音频的绝对100 Hz帧起点。
保留原五参数infer（起点0），新增带起点的重载，并在访问输入前拒绝派生索引溢出。
固定seed相同绝对帧的抽样现在一致；focused真实长音频测试回升到0.9872并通过，
没有降低噪声幅度或误差门限。首轮旧二进制测试在发现失败后中断，未运行项不计通过。
新CLI用正式素材`CyberTom_Pilot_Decoder/clips/song_03_0000.wav`跑2步decoder导出验证，
耗时5.188秒，560个tensor全部有限、239个decoder tensor变化、其余与原预训练G一致。
mel-L1为0.9082→0.9229，仅验证真实训练/导出可运行，不作为收敛或音质证据，产物不发布。
同一HTK模型和5.84秒输入的C++短试听已提供给用户，ASR为3/17（此前离线B为2/17）；
新增一处识别差异。用户随后听审确认C++输出“与 B 接近，明显好于旧版”，
证明该英文短样本的相对自然度改善也出现在实际C++路径；中文、翻唱和目标音色尚待验收。
最终构建`/tmp/resonantia-performance-20261024.uemhTG`：Release/SM87/FP32，model与codec
测试ON、slow Web OFF。`cmake --build /tmp/resonantia-performance-20261024.uemhTG -j 3`
成功；`ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L contract --output-on-failure`
为33/33（30.77秒），同命令改为`-LE contract`为24/24（641.99秒）。合计57/57、
0 fail/CTest skip，两个集合不重叠；本轮完整console与每项LastTest细节均已归档。
额外host-only的`test_vc_train_entry`重建及CTest 1/1通过，确认公共类型依赖未破坏host入口。
独立sampled训练decoder相对L2为0.000177602；bounded与legacy频谱相关性为
RMVPE/v2 0.99945777、FCPE/v1 0.99985054。17.565/35.130秒resource gate：
峰值RSS 1391760/1449500 KiB，设备增长839426048/870170624 bytes，原门限均通过。

10:28 CST整组切换`vc_convert/vc_batch/vc_train`并重启Web（PID 3692644），根页面、voices、
training status均HTTP200，`CyberTom_Pilot_HTK`可选；正式CyberTom模型/Index哈希未变。
备份`build/.sampled-synth-20261024.j6scaixn/previous/`，具体新旧二进制哈希见
`sampled-runtime/deployment.json`。同HTK模型及Web参数（检索0.35、pitch0、RMS/protect0.5、
filter3）生成12.485秒英文/15秒中文/6秒翻唱，用时25.12/28.59/15.40秒，均无削波。
英文ASR为0/32；中文句子主体与旧对照一致，末尾从“默默的死”变为“默默的思念”，
没有人工ground truth，不能宣称CER=0；6秒翻唱的ASR仍不具验收判别力。
用户随后听审明确确认两份Web中文/翻唱“两份仍有明显电音”，本轮歌声质量失败，
`quality_accepted=false`。英文短样本的相对改善仍成立，但不能推广为歌声或Tom音色通过。
离线`tools/probe_cybertom_singing.py`用独立上游posterior直接编码真实干声频谱，
绕过HuBERT/prior/Flow/Index，固定posterior与NSF抽样，再比较原预训练decoder与HTK
微调decoder。两份6秒重建ASR无判别力，用户指出“人声正好被打断”，因此该轮听审
无效，不能据此归因decoder或内容编码器。后改为从已有分离干声中按停顿选取
原歌47.5–58.2秒的10.7秒片段，输入/裁剪样本索引及哈希保存在`experiment/singing-phrases/`。
新的原声、基础posterior重建及CyberTom posterior重建均为10.7秒、无削波，ASR可识别
相近的两句歌词但有字词差异；没有人工文本，不计算真实CER。该片段的RMVPE与YIN
有声帧音高中位差8.74音分，YIN仅为交叉检查，不是ground truth。用户随后确认基础重建
电音较明显、CyberTom重建较弱但仍存在，仍未通过自然度验收。两份原对照未匹配响度，
不能把主观程度差直接解释为decoder训练的独立收益。绕过HuBERT/prior/Flow/Index仍有
伪影，说明这些环节不是该重建电音出现的必要条件；尚未证明唯一根因，也不排除转换
路径存在额外问题。后续只新增离线诊断工具并在真实素材上
执行成功，没有改动57项已验证的运行时或再部署服务。
没有运行完整歌曲或再做长时间训练。

命令和失败/通过日志保留于
`runs/CyberTom_Pilot_HTK/experiment/sampled-runtime/`，其中`pre-offset-*`为分段修复前记录。

### 2026-10-25 完整唱句重建的音高提取器对照

继续使用原歌47.5–58.2秒的同一10.7秒真实干声，新增离线工具
`tools/probe_cybertom_pitch_control.py`。复用已存的posterior latent、HTK pilot decoder、
speaker0及NSF seed20261025；先重现用户已听过的RMVPE版本，再仅把F0换为既有
`torchfcpe==0.0.4`的bundled模型（已审计哈希、restricted loader、threshold0.006、
local_argmax、保留无声帧）。没有重新训练、运行完整歌曲、修改运行时或部署Web。

真实baseline逐样本复现，相对L2=0；FCPE/RMVPE共同有声929帧的绝对差中位数
1.395音分、P95为16.061音分；有声/无声判定有36/1070帧不同。这不是音高ground truth，
也不证明更换提取器改善了电音。两份原始重建均无削波；另生成RMS匹配试听，
只做固定增益（FCPE为0.999101，RMVPE为1），不施加降噪、平滑或压缩。
本地Whisper对新FCPE试听的转录与既有RMVPE试听相同，不能据此判定电音改善；
没有人工歌词ground truth，不计算真实CER。用户随后确认“A/B差别不大”，仍有可听见
的电音；后补充主要是音高抖动/修音感与沙哑/颗粒感的结合。换提取器未解决该片段。

首次工具执行因FCPE采样率访问器名称写错而退出1；修正为实际`get_model_sr`后执行
退出0，两份重建各约12.3/12.2秒。失败日志、成功命令、输入/模型/代码/产物哈希和
听审状态保存在`runs/CyberTom_Pilot_HTK/experiment/singing-pitch-control/`。
本轮只验证离线诊断及文档，未重跑此前57项运行时CTest；不能将其计为本轮新测试通过。

### 2026-10-25 重建与训练路径数值排查

用户要求继续定位且不做长训练后，新增离线
`tools/probe_cybertom_reconstruction_audit.py`，沿用同一10.7秒真实干声与原模型。
记录在`runs/CyberTom_Pilot_HTK/experiment/reconstruction-audit/`：

- 原始预训练`.pth`以restricted loader读取，560个tensor与safetensors转为FP32后
  逐项完全相同。原生40k输入无重采样；独立复数STFT/重叠相加往返相对L2
  `1.096e-7`，最大绝对误差`1.192e-7`；频谱幅值与上游实现相对L2 `3.416e-8`。
- 缓存posterior latent重现相对L2=0；NSF纯基频正弦对float64相位积分的相对L2
  `1.533e-5`、最大绝对差`5.301e-5`。这些检查没有显示大的权重转换、频谱或
  相位数值错误，不是完整音质通过，也不排除模型本身的重建限制。
- 确认训练流程与上游有差异：当前CLI先裁40帧频谱再运行posterior/Flow，上游先
  编码完整clip再裁decoder latent。同一真实干声的分段posterior均值相对L2
  `0.226205`；保持同一高斯抽样后latent相对L2 `0.172933`。这是上下文差异的
  实测影响，不是已证实的电音根因，也不能解释未经微调的基础重建电音。
- 当前CLI mel为FFT1024/hop256/80带、无reflect补边、log floor `1e-12`；本地
  上游40k训练配置为2048/400/125带、有补边、log floor `2e-6`。posterior频谱
  的小功率floor也不同，省略floor使该片段posterior均值相对差`0.003621`。
  这些是训练资格仍缺少独立对齐门禁的依据，不能从优化loss下降推断听感合格。

另以`tools/probe_cybertom_mel.cpp`链接当前库，在同一真实输入/重建的0、1、10秒处
各取0.4秒，不做任何优化步；`tools/probe_cybertom_mel_reference.py`使用相同现有
几何与clamp的独立FFT，分离“数值精度”与“训练配置差异”。FP32对拍：mel-L1
绝对差最大`5.854e-5`；梯度相对L2为0.001255/0.011805/0.036079、方向余弦至少
0.999349；FP64交叉检查结果近似。没有设立新的数值合格门限，也没有把这些误差
归因为整段电音。编译/执行与数据见`experiment/mel-audit/`。

周期激励的负对照保持posterior和NSF随机抽样，仅将纯正弦分量设零：原baseline
重现相对L2=0；去周期版本虽仍被ASR识别，但FCPE对输出/原声共同有声帧的音高差
中位数升至782.90音分（baseline为2.14），因此该方向失败，不作为改善或发布候选。
原始音量下降，匹配试听需4.346倍增益；记录保留于`experiment/excitation-control/`。
再次说明ASR不能替代音高保持与自然度验收。

本轮只增加诊断工具/证据/文档，没有修改运行时、训练模型、Web配置或启动长训练；
唯一根因仍未确定。按用户描述的“抖动＋颗粒感”，随后在`experiment/texture-control/`
保留周期音高，分别仅关闭posterior采样或仅关闭有声NSF噪声（无声噪声保持0.1/3）。
baseline逐样本重现，三个版本RMS匹配、无削波；两项输出相对原声的FCPE共同有声
帧音高差中位数均约2音分，但posterior均值版的有声边界偏差更多：P95为63.90音分、
36帧超过100音分；有声NSF噪声关闭版P95为29.78音分、8帧超过100音分，与baseline
29.84音分/8帧接近。这仍不是音高ground truth或听感判定。用户随后确认baseline和
无有声NSF噪声版相对更少抖动/颗粒感，posterior均值版的修饰音/抖动明显；进一步确认
前两份“虽然有瑕疵但是自然度可以接受”。因此保留posterior采样，不采纳均值模式，
也没有证据优先选择关闭有声NSF噪声。`texture-control/accepted-reference.json`固定
两份音频哈希及用户验收范围：仅该10.7秒直接重建自然度通过，不代表实际VC、Tom音色
或其他输入通过。没有修改生产配置或发布新模型。

### 2026-10-25 训练上下文对齐诊断与实际Web转换对照

自然度参考被用户接受后，不再以消除所有轻微瑕疵为目标。新增离线
`tools/probe_cybertom_context.py`及`tools/probe_cybertom_context.cpp`，检查训练切片的
可修正范围：posterior的16个kernel5层需单侧32帧，四个Flow coupling各增加6帧，
合计单侧56帧。保持完整clip的同一高斯抽样，测试首尾/中间、8/40/200帧核心及
0/32/56帧上下文的18种组合。上游参考18组完成，C++/CUDA全部18组满足独立参考
相对L2 `<1e-4`断言，最大`1.72903e-6`；56帧上下文的6组对完整clip最大
`1.50758e-6`，原无上下文的40帧核心最大差`0.240948`。32帧对当前权重也很接近，
但不能以小误差替代56帧的结构依赖上界或保证训练后仍成立。

本次只调用已有posterior/Flow前向，没有反向、优化器step、音频解码或权重导出；
因此只证明有界上下文可恢复前向对齐，尚未修正训练CLI，也未验证训练收益。
最多312输入帧，C++探针0.623秒、进程峰值RSS740100 KiB；不是完整训练内存门禁。
首次用`/usr/bin/time`包装时因工具不存在返回127、未执行探针，随后通过Python的
subprocess/resource记录成功结果，失败日志保留。命令、代码/资产与结果在
`runs/CyberTom_Pilot_HTK/experiment/context-control/`。

随后通过当前Web `/api/convert`运行同一句10.7秒完整唱句两次，均使用
`CyberTom_Pilot_HTK`、pitch0、RMS/protect0.5、filter3，唯一参数差为检索率0/0.35。
两次均HTTP200，耗时18.73/19.13秒，输出各428000个40k样本、无削波；另外保存
RMS匹配试听（固定增益0.683507/0.731264），原始HTTP输出不改写。
该对照用于把已接受的直接重建与实际转换区分，并判断Index影响；不能把两个流程
之间的所有差异直接归因于某一个模块。没有重新训练、整曲转换、服务重启或部署。
本地Whisper对无Index版的转录与已接受的重建参考一致；0.35版出现额外字词差异，
尚无人工歌词ground truth，不计算真实CER或据此断定电音。FCPE对原声/输出共同
有声帧的音高差中位数分别2.05/1.98音分，P95约21.6/21.4音分，仅为交叉检查。
结束后服务idle，原模型及Index哈希不变。
结果、参数和哈希见`experiment/web-phrases/`。用户随后回复“两个版本,都差点意思”，
两份实际Web转换均未达到已接受的posterior重建自然度标准；不能把剩余问题只归因于
Index，也不能以ASR转录一致替代听审。目标音色仍未验收，未发布新模型。

### 2026-10-25 同唱句内容编码、prior与合成逐段对齐

新增离线`tools/probe_cybertom_content.py/.cpp`，使用同一10.7秒真实完整唱句、
已接受重建的原始RMVPE F0和Pilot权重，绕过Web高通、补边、Index及RMS处理。
两套实现共用scipy多相降采样后的16k输入，HuBERT layer12输出534帧，经nearest
2倍重复得到1068帧；输出10.68秒，只省去末端静音的20毫秒。prior与Flow检查共用
同一参考phone输入，随机抽样固定seed0；不能把本探针称为Web端到端一致性测试。

C++/CUDA与独立Transformers/RVC参考的HuBERT特征RMS误差`1.22730e-6`，
prior mean/logs相对L2分别`4.89297e-7`/`2.37289e-7`，Flow输出`4.86918e-7`；
最终音频相对L2`1.05712e-4`、相关性`0.9999999944`。全部满足探针断言
（特征RMS `<1e-4`，三个latent阶段相对L2 `<1e-4`，音频相对L2 `<1e-3`
且相关性 `>0.999`）；这只排查该短句上的明显实现偏差，不证明自然度或音色通过。

另生成一份保持已接受重建NSF seed20261025的参考转换试听，prior seed20261024、
采样尺度0.66666，RMS匹配增益0.642969、无削波。本地Whisper转录与重建参考一致，
仍无人工歌词ground truth。用户随后反馈“抖动和颗粒感明显减轻，但音色丢失比较严重”，
并明确同时指声音变薄/发空及不像Tom；本样本整体未通过。多个前处理、上下文和随机
条件同时不同于Web，改善不能直接归因于某一项。结果和哈希在
`runs/CyberTom_Pilot_HTK/experiment/content-stages/`。

`tools/probe_cybertom_content_weights.py`另从固定官方资产版本
`lj1995/VoiceConversionWebUI@e6d0c1a17da07c33557852f9dfa2bd44cc75737d`
核对legacy `hubert_base.pt`（SHA256 `f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96`）。
显式映射名称后，213个推理tensor与HF bin、部署safetensors逐项完全相等；仅跳过
不参与推理的`label_embs_concat`。16k采样率、不做waveform归一化、12层/768维、
LayerNorm顺序配置也匹配，排除了本地转换错用另一套内容权重的怀疑；没有执行
legacy fairseq前向，不能称为该实现的完整数值等价证明。加载使用weights_only，
仅为旧Dictionary元数据登记无行为的数据占位类，不导入fairseq代码。
首次curl下载因HTTP/2中断返回92，随后hf_hub_download成功并通过固定SHA校验；
首轮名称映射因新旧weight_norm命名不同失败，补齐显式映射后断言通过。
失败与成功结果保存在`experiment/content-compatibility/`。

本轮没有训练、部署、重启服务或整曲转换；只增加离线探针、结果和文档，未重跑
运行时完整CTest，不把此前57/57结果计作本轮执行。

### 2026-10-25 Pilot音色适配审计与待批准训练方案

确认最近使用的是60秒/21片段、50步、LR `1e-5`的decoder-only诊断Pilot，并非已验收
的正式声线。新增`tools/probe_cybertom_adaptation.py`以同一句真实唱段、相同latent和
NSF seed仅替换基础/Pilot decoder：Pilot逐样本重现上一份参考试听（相对L2=0），
输出与基础模型相关性0.778811、相对L2差0.648938，说明训练确实影响了输出，不能
把音色失败说成“训练无效/没有加载微调权重”。该指标不是感知音色相似度。

560个checkpoint tensor中，decoder有239/243改变，enc_p/Flow/enc_q/emb_g共321个
均未改变。还原weight-norm后的有效decoder kernel/bias相对L2改变量为0.00295129；
原始g/v重参数化距离0.67736不代表有效权重改变67.7%，两种统计分开保留。固定
embedding和prior不证明一定学不会目标声音：可训练decoder/Flow及条件偏置仍可适配。
当前GAN optimizer范围、提前裁40帧、mel配置和每clip固定NSF缓存与上游存在差异，
但没有把任何一项当作已证实的唯一音质根因。

命令、哈希和通过的重现/权重检查见
`runs/CyberTom_Pilot_HTK/experiment/adaptation-audit/`。本轮optimizer step=0，
没有新训练/部署/服务重启；只生成两份10.68秒离线分析音频，未作为改善候选要求听审。

[ADR-0008](../docs/adr/0008-joint-voice-training-alignment.md)已整理成Proposed方案：
新增显式选择的有界联合训练，纳入prior/说话人条件梯度，修正上下文/损失/随机输入
流程，扩展原子G导出；先做0/1步独立数值与资源门禁，再最多50步小验证。尚未获批
或实施，不授权长训练，也不替换正式CyberTom；当前FR-014仍未满足。

### 2026-10-24 Web 常见音频格式上传与部署

维护者明确回复“同意服务器端 FFmpeg 方案（推荐）”后实施ADR-0007。
material、带上传training、直接preprocess/convert共用request staging解码，支持
WAV、MP3、M4A(AAC/ALAC)、AAC、FLAC、OGG(Vorbis/Opus)、Opus、WMA、AIFF/AIF；
WAV字节保留，其他输入归一化PCM16/44.1kHz/mono。扩展名忽略大小写、归一化重名拒绝；
失败批次保留旧raw/work/marker。新增64MiB/文件、128MiB/解码批次、120秒/文件、
2GiB子进程地址空间与单decoder限制。模型CLI没有新增格式或Python依赖。
UI支持两处选择和拖放，串行上传避免旧请求晚提交覆盖新素材；等待和失败期间禁用操作，
过期响应不能恢复按钮，训练上传可重试。浏览器仅读metadata估算时长。

本机最初安装到NVIDIA `7:8.0.1-nvidia1`，能力清单显示缺MP3/AAC/WMA等codec；
随后使用 `sudo -n apt-get install -y --allow-downgrades ffmpeg=7:6.1.1-3ubuntu5`
安装Ubuntu完整音频构建，另增加libavdevice60/libavfilter9/libpostproc57。
`/usr/bin/timeout`、`/usr/bin/prlimit`已存在；后续系统升级仍需核验实际codec能力。

最终构建Release/SM87、CUDA13.2；**32/32 asset-free contracts通过，24.80秒，
0失败/跳过；独立无CUDA host子集11/11通过，0.57秒**。额外codec integration通过，
2.87秒；12组编码均取真实`speech_00.wav`的首1秒，实际解码长度0.975238–1.04490秒，
符合1±0.15秒断言；生产校验确认完整PCM16/44.1kHz/mono，测试确认非静音。
编码包括PCM、MP3、AAC、ALAC、FLAC、Vorbis、Opus、WMA、AIFF；还验证错扩展名、
播放列表/损坏字节及仅视频无音轨拒绝。mock解码器覆盖四入口失败、成功但无效输出、
每文件/累计超限、decoder busy、缺依赖、WAV兼容、失败清理；真实timeout wrapper用
0.1秒睡眠fixture验证退出与回收，未等待生产120秒限额或做内存耗尽压力测试。

先红后绿记录：旧实现的 `/tmp/test_web_audio_upload_red` 返回
`MP3 material must normalize successfully; got 400`；旧UI的
`node tests/test_web_audio_ui.cjs /tmp/resonantia-audio-ui-20261024` 返回
`trainFileInput missing mp3`。复查新边界又复现并修复尾随空格WAV文件名走错decoder、
旧转换完成时重新启用正在上传按钮的问题，新增用例最终通过。

关键构建与测试命令：

```bash
cmake -S . -B /tmp/resonantia-performance-20261024.uemhTG \
  -DRESONANTIA_ENABLE_AUDIO_CODEC_TESTS=ON
cmake --build /tmp/resonantia-performance-20261024.uemhTG --parallel 2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG \
  -R '^test_web_audio_(upload|codecs)$' --output-on-failure
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG \
  -L contract -LE 'integration|slow' --parallel 1 --output-on-failure --no-tests=error
cmake -S cmake/host-tests -B /tmp/resonantia-audio-host-20261024 \
  -DCMAKE_BUILD_TYPE=Release \
  -DFETCHCONTENT_SOURCE_DIR_CATCH2=/home/rm01/Resonantia/build/_deps/catch2-src
cmake --build /tmp/resonantia-audio-host-20261024 --parallel 2
ctest --test-dir /tmp/resonantia-audio-host-20261024 \
  --output-on-failure --parallel 1 --no-tests=error
scripts/check_sdd.sh
git diff --check
```

配置共注册57项，本轮执行32项默认契约和1项codec integration，独立host11项是重叠子集。
其余24项模型integration及可选slow deadline未执行，不能称57/57全量通过。
没有真实模型转换、重新训练或声音相似度验收；新格式支持不代表修复音色质量。
隔离临时repo的真实 `vc_serve` + HTTP MP3上传/preview均200，返回88278-byte WAV，
状态idle；没有模型文件/CLI，测试结束删除临时repo。

部署脚本 `/tmp/resonantia-deploy-audio-20261024.sh` 在服务无作业、子进程及活动HTTP
连接时备份并同时替换`build/vc_serve`与UI，原unit/模型CLI/端口保持不变。
回滚材料、脚本与验证日志保存在`build/.web-audio-20261024.zFoECZ/`。
服务PID `1456763`、active/running；运行中exe与独立构建逐字节一致。
主页、JS/CSS、voices/status经本机LAN IP `http://192.168.0.122:8080/` 均HTTP200，
CyberTom可选、无运行作业。正式CyberTom模型/Index及`__convert__`现有文件SHA-256
全部保持不变。部署后仅GET检查，上传试听由用户自行执行。

### 2026-10-24 Web 切换到性能修复构建

用户明确授权切换并重启服务，转换试听由用户自行上传执行。部署前确认没有 running
训练/step，也没有服务子进程；把已通过下节验证的独立构建中 `vc_serve`、`vc_convert`、
`vc_train`、`vc_preprocess`、`build_index` 五个可执行文件 staged 到仓库 `build/`，停止
服务后逐项 rename 安装并重新启动。服务 unit、`--repo`、webroot、绑定地址和端口未改。
旧可执行物、前后状态及 SHA-256 清单保存在
`build/.web-release-20261024.EJuYEg/previous` 及其父目录；失败回滚未触发。

`resonantia-web.service` 新 PID 为 `1441776`（旧 PID `1394939`），状态
`active/running`。运行中的 `/proc/1441776/exe` 以及部署的五个 CLI 均与独立构建逐字节
一致。CyberTom 正式模型、Index SHA-256 复核通过；未执行上传、预处理、训练或转换。

通过内网 IP `http://192.168.0.122:8080/` 的 GET 检查：主页、`/app.js`、`/styles.css`、
`/api/voices`、`/api/train/status` 全部返回 HTTP 200；声线列表包含 `CyberTom`，作业
`running=false`。这些是本机经 LAN IP 的可达性与部署检查，不是新的音频 E2E 结果。
检查使用 `curl --noproxy '*' --fail --silent --show-error --max-time 5`，响应头及正文
保存在上述 deployment 目录。SDD 检查通过（32 required files）。

根 `build/` 本轮只部署上述可执行物，其余旧静态库/测试二进制未重建；性能修复的编译和
CTest 证据仍以 `/tmp/resonantia-performance-20261024.uemhTG` 为准。下节旧版本对照命令
是切换前的历史命令；切换后若复核旧转换，应使用 deployment `previous/vc_convert`，
不能再把当前 `build/vc_convert` 当作旧基线。

### 2026-10-24 CyberTom 离线性能修复

基于 `06d2ed3` 的未提交工作树，独立 Release/SM87 构建
`/tmp/resonantia-performance-20261024.uemhTG`；Orin NX、aarch64、8 CPU、CUDA 13.2.86、
GCC 13.3。模型和原素材未替换；性能验证阶段没有重新训练正式 CyberTom，也没有重启/
切换 Web。用户随后授权的 Web 部署和当前运行状态见上一节。

实现范围：CPU exact Index 最多 4 路 query 并行；enc_p attention 改 GPU；大通道 VITS
卷积采用有界 im2col/SGEMM、transpose gather；梯度范数 block reduction；256 MiB
dead-graph device buffer cache、固定 host segments 复用；移除 GAN 无用 decoder trainer
与重复 enc_p/inverse-Flow；CLI feature/GAN 耗时、Web 成功转换日志保留。
公开模型、Index、WAV 格式和推理参数均保持原契约，未引入 persistent GPU runtime。

**最终验证：全部目标构建成功；30/30 asset-free contract、6/6 受影响真实资产 integration
通过，0 失败、0 跳过，分别 24.56 s 和 47.33 s。** 6 项为 `test_vits_alignment`、
`test_vits_gen_debug`、`test_index_retrieval`、`test_mel_loss`、`test_generator_trainer`、
`test_gan`。真实 VITS m_p RMS `6.98772e-08`、Flow RMS `1.90472e-07`、audio correlation
`0.999716 > 0.999`；没有放宽门限。decoder/GAN 使用测试自有的短片段、短训练及临时导出。
SDD 检查和 `git diff --check` 通过。

完整配置注册 54 项，本轮执行其中 36 项，**不能称 54/54 全量通过**。按用户要求不执行
整段音频转换或长训练；未跑的 18 项是 `test_build_index_contract`、
`test_cli_bounded_callers`、`test_pipeline_contract`、`test_hubert_inference`、
`test_rmvpe_alignment`、`test_e2e_conversion`、`test_long_audio_chunking`、
`test_bounded_file_resources`、`test_bounded_file_fcpe_v1`、`test_bounded_file_wavlm_preflight`、
`test_fcpe_alignment`、`test_wavlm_alignment`、`test_vits_stream_characterization` 和五项
`test_separation_{stft,spec,umx,roformer,vad}`。slow body deadline 默认未注册。未跑不是 skip
或 pass；未修改的 encoder/F0/separation 本轮不重复进行完整资产验收。

短片段端到端对照使用同一 CyberTom 模型、61639×768 Index、同一 1 s 人声摘录、相同 CLI
参数和相同 Nsight `cuda,nvtx / sample=none` 设置。以下是单次配对观测，非 p95 或实时 SLA；
bounded 路径仍包含上下文 padding，不能按 1 s 结果线性推算整首歌曲耗时。

| 1 s 转换阶段 | 修复前 ms | 修复后 ms |
|---|---:|---:|
| HuBERT | 863.821 | 855.559 |
| F0 | 1144.96 | 1299.10 |
| Index | 6932.65 | 1782.50 |
| Synth | 5889.90 | 2290.12 |
| 转换总计（不含模型 init） | 14842.30 | 6238.21 |

总计约 **2.38 倍速度，耗时减少 58%**。两份输出均 40000 samples/40 kHz，相关性
`0.9999999998646125`、RMS 差 `1.220703125e-06`。这证明本次短样本数值保持，不证明
Tom 音色相似度或消除机器人感。GPU 明确参与：修复后的 Nsight 记录 13042 个 kernel，
kernel 时间总和 1496.99 ms；3083 次 memcpy、5.613 GB、设备 copy 时间 616.96 ms。
kernel/copy 时间不能再与包含等待的 CUDA API wall time 相加。

独立组件 microbenchmark 使用相同 3 s 真实训练 clip 和两次 0.4 s GAN step：

| 组件 | 修复前 ms | 修复后 ms | 说明 |
|---|---:|---:|---|
| exact Index 50 queries | 2326.39 | 612.60 | 各 3 次中位数；加权输出逐字节一致 |
| enc_p 298 frames | 887.37 | 91.43 | CPU attention 改 GPU |
| decoder 100 frames | 1698.72 | 778.13 | FP32 tiled convolutions |
| GAN step 1 | 1759.27 | 1488.09 | 包含首次分配；不是长期训练均值 |
| GAN step 2 | 1633.58 | 1203.88 | CUDA malloc/free 各 3030 → 1573 |
| step 2 梯度范数 kernels 合计 | 322.8 | 14.87 | 405 次 kernel 调用；每 block 一次 atomic |

另外执行真实 `vc_train --gan --steps 1 --seg 40 --seed 20261024`，单独导出到诊断目录。
日志显示 `latent=0.000`、prior=92.144 ms、训练步 1394.547 ms，首步 loss 与旧 CLI 的四位
小数输出一致。560 个导出 tensors 同 shape、全部有限；前后导出 RMS 差 `2.1129e-08`、
最大绝对差 `9.2007e-05`，浮点重排和归约修复不承诺训练权重 bitwise 一致。

先失败证据与开发中修正：旧梯度归约在超过 `2^24` 个单位梯度时漏累加，新回归失败；
旧 Web 成功路径没有阶段日志，新 HTTP 回归失败。缓存初版会把无梯度参数变成零梯度，
新增 weight-decay 回归复现 `2.0 → 1.98`，现已改为 backward 才取 grad buffer。
新 CUDA 文件最初缺少 `math_constants.h`，及卷积 fault counter 仍按旧 kernel 计数，
均已修正并重新通过最终构建/测试；原门限未删除，fault coverage 额外覆盖 GEMM 边界。

当前剩余开销：Index 仍是 CPU 精确检索；HuBERT 和 RMVPE 仍有 CPU 前后处理、布局转换；
VITS 权重/activation 的 host-device 往返及每次调用的 cuBLAS handle/临时分配仍存在。
该版本仍非实时。训练 cache 的 256 MiB 只限定空闲 buffers，不是总训练显存上限；
没有验证长期训练稳定性、完整长音频资源门禁、主观音色或常驻 GPU 后端。

原诊断记录在 `runs/CyberTom/performance-20261024.5HY9L7`；修复后的日志、Nsight/SQLite/CSV、
`verified-metrics.json`、benchmark 源码和 1 s WAV 归档于
`runs/CyberTom/performance-fixed-20261024.uemhTG`（均为本地忽略资产）。正式 G SHA-256
`2b8c876116be20e06e30ccfe7d37d5385156f0fccbd0b566f94410cd9e37dae2`、Index SHA-256
`443bd9ec3c978d70f7e14546c13675e392f91fb9b0a5015c3e4b73c3af6ecd82` 未变。

```bash
PERF_BUILD=/tmp/resonantia-performance-20261024.uemhTG
cmake -S . -B "$PERF_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DVOXMUTATIO_CUDA_ARCHITECTURES=87 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DVOXMUTATIO_ENABLE_MODEL_TESTS=ON \
  -DFETCHCONTENT_SOURCE_DIR_CATCH2=/home/rm01/Resonantia/build/_deps/catch2-src \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
nice -n 10 cmake --build "$PERF_BUILD" --parallel 2
ctest --test-dir "$PERF_BUILD" -L contract -LE 'integration|slow' \
  --parallel 1 --output-on-failure --no-tests=error
ctest --test-dir "$PERF_BUILD" \
  -R '^(test_vits_alignment|test_vits_gen_debug|test_index_retrieval|test_mel_loss|test_generator_trainer|test_gan)$' \
  --parallel 1 --output-on-failure --no-tests=error
# 本地脚本只使用 1 s WAV、3 s clip 和两次 0.4 s GAN step：
bash "$PERF_BUILD/verify-short.sh"
# 同配置旧 build 对照，输出另存，未触及正式模型或 Web material：
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --duration=60 --kill=sigterm --stats=false --output="$PERF_BUILD/infer-baseline-matched" \
  build/vc_convert --hubert models/hubert_base/model.safetensors --rmvpe models/rmvpe.safetensors \
  --model runs/CyberTom/model.safetensors \
  --input runs/CyberTom/performance-20261024.5HY9L7/input-1s.wav \
  --output "$PERF_BUILD/baseline-matched-1s.wav" --version v2 --speakers 109 --sr 40000 \
  --pitch 0 --rms-mix 0.4 --protect 0.33 --filter-radius 3 \
  --index runs/CyberTom/model.index --index-rate 0.75
cmp "$PERF_BUILD/index-before.f32" "$PERF_BUILD/index-after.f32"
bash scripts/check_sdd.sh
git diff --check
```

### 2026-10-24 推送前合并远端 CI 更新

推送 `d20dbe0` 时发现远端新增 `5f7fe83`、`c65eec7`，采用 merge 保留双方历史。
合并 README/状态记录时保留八项修复和远端托管 CI/取消测试；更新当前测试数量与 CI 边界。
工作流和独立 host 构建入口与 `c65eec7` 一致，自动 CI 仍只运行仓库检查及 10 项 host 子集；
未手动派发设备作业或配置 runner。以下为合并工作树的本地验证，不代替推送后的 GitHub CI。

- 根构建重新 configure、全部目标构建成功；默认 asset-free contract **29/29 通过，
  0 失败/跳过，23.19 秒**，包含远端新增的协议与最终发布门禁以及本次修复回归。
- 独立无 CUDA host 构建成功，**10/10 通过，0 失败/跳过，0.29 秒**。
- Node 可用、slow=OFF 时根构建默认29项、资产齐全53 distinct（30 contract / 24 integration /
  1 overlap）；slow=ON 共54项。缺 Node 减2项。注册清单已核对，本次合并未重跑24项
  integration及1项slow；下节52/52全量证据属于 `d20dbe0`，不报告合并树54/54通过。
- SDD 32项、shell语法和差异格式检查通过。本机日志目录为
  `/tmp/resonantia-push-merge-20261024.0j9cec3p`。

```bash
cmake -S cmake/host-tests -B /tmp/resonantia-push-merge-20261024.0j9cec3p/host \
  -DCMAKE_BUILD_TYPE=Release \
  -DFETCHCONTENT_SOURCE_DIR_CATCH2=/home/rm01/Resonantia/build/_deps/catch2-src \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
nice -n 10 cmake --build /tmp/resonantia-push-merge-20261024.0j9cec3p/host --parallel 1
nice -n 10 ctest --test-dir /tmp/resonantia-push-merge-20261024.0j9cec3p/host \
  --parallel 1 --output-on-failure --no-tests=error
# 根构建沿用下节相同 configure 参数，重新配置和构建后执行：
nice -n 10 cmake --build /tmp/resonantia-fixes-20261024.wmLxjK --parallel 1
nice -n 10 ctest --test-dir /tmp/resonantia-fixes-20261024.wmLxjK \
  -L contract -LE 'integration|slow' --parallel 1 --output-on-failure --no-tests=error
bash scripts/check_sdd.sh
bash -n scripts/*.sh
git diff --check
```

### 2026-10-24 八项审计修复（d20dbe0）

基于干净的 `main@7082b96`，修复完整 GAN 导出漏写 Flow、批处理同名覆盖、host 频谱奇数
padding 越界、sqrt/log 裁剪梯度、MelLoss 配置与形状校验、AdamW 非有限 beta、Web token
重试，以及 WavLM 完整长度/RMSE 门禁。批处理保留 flat 输出布局，冲突在 CUDA/模型加载及
所有输出写入前返回非零；GAN 的 decoder + Flow 只做一次原子发布，不扩展 resume。

回归证据：P1 三项新门禁在旧实现 **0/3 通过**，修复后 **3/3 通过，30.88 秒**；
autograd/MelLoss 两项新门禁在修复前 **0/2 通过**；认证重试新测试在旧 JS 退出 1，
修复后认证与既有取消 UI 测试均退出 0。WavLM 使用隔离目录内的真实参考变体（少一帧、
全元素偏移 0.0005），不改原资产：旧二进制两项均退出 0；新二进制分别因完整 shape
断言与 RMSE 断言失败（均退出 1，偏移用例 RMSE `0.0004999977`）。

修复后 focused contract **5/5，4.41 秒**，真实模型 focused（WavLM、generator、GAN）
**3/3，48.47 秒**，默认 asset-free contract **27/27，23.45 秒**，均无失败或跳过。
对 `posterior_encoder.cpp`、`mel_loss.cpp` 和新 training contract 做 host ASan/UBSan
插桩，**2 cases / 97 assertions 通过**；`detect_leaks=0`，不声称覆盖 CUDA device 内存或泄漏。

本轮最终验证：全新 Release/SM87 configure + 全目标 build 成功；完整 CTest
**52/52 通过、0 失败、0 跳过，1409.86 秒**，包含全部 24 项真实资产 integration 和
122.06 秒的真实 Web body deadline 门禁。实际 label 为 29 contract / 24 integration /
1 overlap；不是 53 项。WavLM 完整 224256 元素的 RMSE 为 `1.19736e-6`，门槛 `<1e-4`。
详细日志未发现缺资产或内部 skip，52 个执行名称与注册清单逐项一致；本轮 20 个变更的
代码/构建/测试文件 SHA-256 在测试前后完全相同。SDD **32 项通过**，`git diff --check` 通过。

复现构建/测试命令（Orin / aarch64 / SM87，Release；GNU 13.3.0、CUDA 13.2.86、
CMake 3.28.3；现有真实模型与 LibriSpeech/reference 资产，Node.js 18.19.1）：

```bash
cmake -S . -B /tmp/resonantia-fixes-20261024.wmLxjK \
  -DCMAKE_BUILD_TYPE=Release -DVOXMUTATIO_CUDA_ARCHITECTURES=87 \
  -DVOXMUTATIO_ENABLE_MODEL_TESTS=ON -DVOXMUTATIO_ENABLE_SLOW_WEB_TESTS=ON \
  -DFETCHCONTENT_SOURCE_DIR_CATCH2=/home/rm01/Resonantia/build/_deps/catch2-src \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
nice -n 10 cmake --build /tmp/resonantia-fixes-20261024.wmLxjK --parallel 1
nice -n 10 ctest --test-dir /tmp/resonantia-fixes-20261024.wmLxjK \
  -R '^(test_autograd|test_training_contract|test_cli_contract|test_web_auth_ui|test_web_train_cancel_ui)$' \
  --parallel 1 --output-on-failure --no-tests=error
nice -n 10 ctest --test-dir /tmp/resonantia-fixes-20261024.wmLxjK \
  -R '^(test_gan|test_generator_trainer|test_wavlm_alignment)$' \
  --parallel 1 --output-on-failure --no-tests=error
nice -n 10 ctest --test-dir /tmp/resonantia-fixes-20261024.wmLxjK \
  -L contract -LE 'integration|slow' --parallel 1 --output-on-failure --no-tests=error
nice -n 10 ctest --test-dir /tmp/resonantia-fixes-20261024.wmLxjK \
  --parallel 1 --output-on-failure --no-tests=error
bash scripts/check_sdd.sh
git diff --check
```

host sanitizer 编译/运行命令（其余库使用上述 Release build）：

```bash
c++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined \
  -I include -I /usr/local/cuda/include -I build/_deps/catch2-src/src \
  -I /tmp/resonantia-fixes-20261024.wmLxjK/_deps/catch2-build/generated-includes \
  tests/test_training_contract.cpp src/training/posterior_encoder.cpp src/training/mel_loss.cpp \
  /tmp/resonantia-fixes-20261024.wmLxjK/libvoxmutatio.a \
  /tmp/resonantia-fixes-20261024.wmLxjK/_deps/catch2-build/src/libCatch2Main.a \
  /tmp/resonantia-fixes-20261024.wmLxjK/_deps/catch2-build/src/libCatch2.a \
  -L /usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -lcublas -lcufft -pthread \
  -o /tmp/resonantia-fixes-20261024.wmLxjK/training-contract-sanitized
ASAN_OPTIONS=detect_leaks=0 /tmp/resonantia-fixes-20261024.wmLxjK/training-contract-sanitized
```

本机证据目录 `/tmp/resonantia-fixes-20261024.wmLxjK` 保存 configure/build、red/green、
focused/contract/full CTest、sanitizer、WavLM before/after 日志及测试注册清单。
WavLM 负对照的工作目录为该目录下 `wavlm-controls/{truncated,error}/run`；分别执行
`wavlm-before` 与本轮 `test_wavlm_alignment`，原始模型/fixture 只读引用。

该修复提交在 Node 可用且 slow=OFF 时默认 asset-free 为 **27 项**，完整资产为 **51 distinct**
（28 contract-labelled、24 integration-labelled、1 overlap）；slow=ON 再加 1 项真实
120 秒 body deadline，总计 **52**。缺 Node 时两个 UI 测试明确不注册，不算通过；
因此默认/full 数量分别减 2。以下日期化记录保留各自历史数量及当时授权边界。

未扩大证据：训练音质/长期稳定性、resume、真实 GPU 取消和最终发布窗口、任意时长资源
保证、非 SM87 平台及实时产品仍未验收；Spec 006 目标声线 model/Index 资格资产仍缺失。

### 历史验证记录

- 2026-10-10，发布后对`5f7fe83`进行host完整性核验：此前10/6差异中两个Catch2用例
  可直接运行于host，取消测试的纯协议部分也不需要CUDA。现复用这两个用例，并将已有协议/
  完整fake迭代/自然helper断言原样拆出；真实参数解析及CLI预取消仍保留完整CUDA构建路径。
  另为原先未确定性命中的最终三产物发布窗口增加私有显式屏障，复用已有取消测试fixture，
  检查发布中409、reservation保留、完整收尾和重试。host注册10项；与原10项不是同一集合。
  对应[验收计划§9](003-webui/acceptance-plan.md#9-托管host覆盖完整性与最终发布窗口)列明逐项差异。
  工作流及设备手动/专属/就绪门禁保持不变；没有设备访问、GPU训练、受限探针或runner设置。
  当前增量的实际host结果以对应SHA的GitHub CI日志为准，设备验证仍未运行。

- 2026-10-10，后续授权将安全发布边界落实为CI：push/PR自动作业只使用GitHub托管
  `ubuntu-24.04`运行仓库检查与6项现有Web/CLI host子集。新增独立`cmake/host-tests`入口，
  不启用CUDA，也不修改根CMake或原测试源码。设备作业需要manual/main/显式勾选/就绪变量
  四重条件，固定`resonantia-orinnx-dedicated`专属标签，无任意标签输入或通用ARM64兜底。
  发布前工作流静态检查与24种事件组合通过，host CMake配置/编译图与6项CTest注册核验通过；
  Mac没有执行Linux测试，实际托管host结果应以对应提交的CI日志为准。设备测试未运行，
  未配置runner/变量/凭据，未访问设备。该改动替代下条“仅本地提交”的历史发布边界，
  不新增GPU、模型、训练或受限探针证据。设备检查路径和原精确排除项继续保留。

- 2026-10-10，基于本地/远端一致的`main@0955500`，新增`test_web_job_ownership`普通host门禁：
  上传自然校验拒绝/worker先于响应退休各配合正常响应/普通异常，**4/4确定性交错通过**。
  B活动时旧A的training/step finish各重复两次，B状态/id不变、C仍409；B结束后C可重试，
  旧素材/模型保留、staging清理、FD无增量。两个临时负对照分别移除generation保护后均按预期
  失败。接缝仅在私有测试目标编译，普通服务对象/库/`vc_serve`重编译哈希与增量前相同。
  明确选择10项host回归**10/10通过、0失败/skip，3.43秒**，当前注册24项、其余14项未运行；
  没有GPU测试、真实训练、rename/pthread拦截或进程注入。编译/测试并发1、nice10，结束时
  可用内存13752MiB、swap0。详细场景、选择清单和证据边界见[验收计划§8](003-webui/acceptance-plan.md#8-abc所有权与重复finish后续普通host验证)。
  本轮补齐等价所有权语义，未发现新的生产缺陷；原受限探针未重跑，也未新增独立审查结论。
  现有CI的通用`self-hosted,linux,ARM64`标签不能证明只路由OrinNX，已查询仓库runner列表为0；
  此增量仅本地提交，不推送、不改workflow/runner，专属设备CI仅另存方案。

- 2026-10-10，已审查的 211 文件修正版提交为 `a08b61c`。发布前追加的 CI 调整仅隔离
  `test_web_transaction_failure`：保留源码，以精确名称从所有会进入设备作业的 CI
  事件中排除，保留仓库检查及其余 contract，CTest 串行执行。该用例的 `rename`
  符号拦截仍不在本轮执行授权内，**未运行、不计通过或 skip**；历史结果不能替代
  当前故障注入覆盖。工作流排除项不修改测试源码或生产实现。真实模型/GPU 取消验收、
  数值等价、取消时延及最终发布确定性窗口仍未验证，未启动训练。

- 2026-10-10 在 OrinNX / aarch64 / Jetson Orin NX / GNU 13.3.0 / CUDA 13.2.86 /
  CMake 3.28.3 上，基于 `main@3a8ba14` 的 multipart 工作区修改完成先失败后通过验证：
  新增真实 HTTP 测试在修复前触发 13 个失败断言（重复/伪装 boundary、畸形参数被接受
  并替换素材，以及二进制 delimiter 前缀导致合法上传失败）；修复后 focused Web CTest
  1/1 通过。补充 quoted escape/closing-prefix 用例后，默认 asset-free contract
  **15/15 通过、0 失败、0 CTest skip，11.40 秒**，其中 `test_web_contract` 为 0.47 秒。
  测试使用临时素材、回环 HTTP 和假 CLI；没有加载语音模型，没有重跑下述 39 项资产套件，
  不增加模型、音质或 realtime 验收证据。独立 build 为
  `/tmp/resonantia-web-20261010.ycTBxI`；全部编译/测试并发为 1、使用 `nice -n 10`，
  构建期间采样可用内存超过 14,000 MiB、swap 为 0。可复现命令：

  ```bash
  cmake -S . -B /tmp/resonantia-web-20261010.ycTBxI \
    -DCMAKE_BUILD_TYPE=Release -DVOXMUTATIO_CUDA_ARCHITECTURES=87 \
    -DVOXMUTATIO_ENABLE_MODEL_TESTS=OFF \
    -DFETCHCONTENT_SOURCE_DIR_CATCH2=/home/rm01/Resonantia/build/_deps/catch2-src \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON
  nice -n 10 cmake --build /tmp/resonantia-web-20261010.ycTBxI --parallel 1
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ctest \
    --test-dir /tmp/resonantia-web-20261010.ycTBxI \
    --output-on-failure --label-regex contract --parallel 1
  ```

- 同日第二组 Web 增量沿用上述环境与 build，修复 material raw/work/marker 事务及
  服务内发布/preview 竞态。素材测试修复前为 **7 个失败断言**；并发复现中 256 次上传
  成功但 1024 次 preview 出现 **61 次失败**。修复后并发计数为 **0/0 失败**，加入
  marker symlink/目录拒绝、训练上传 fresh raw/cleanup 和慢 preview socket 不阻塞发布后，
  focused Web **1/1 通过、0.86 秒**。最后默认 asset-free contract
  **15/15 通过、0 失败、0 CTest skip，11.28 秒**（Web 0.81 秒）。使用同一组串行
  低优先级 build/CTest 命令，不运行模型资产集成或真实训练；代码仍为未提交工作区修改。
  multipart 第一组已有独立 host Web 审查；第二组随后独立审查发现诊断分配异常可使
  backup 析构删除原素材，修复及复验见下条，不能沿用上一阶段通过结论掩盖此问题。

- 同日独立审查发现 material/step/training 的事务诊断在 rollback 前分配内存，单次
  `bad_alloc` 可使 `TempDir` 删除最后旧副本。OrinNX 新增隔离的
  `test_web_transaction_failure` 修复前确认三种事务均丢旧产物；修复采用默认保留备份的
  无分配 noexcept guard，仅成功 commit/完整 rollback 后允许清理，并延后错误文本构造。
  focused Web + exception gate **2/2 通过，0.87 秒**。独立审查包的 AArch64 复验中，
  baseline 和上一快照均按预期触发保留断言失败，修复后 **23 项故障断言、6 项确定性
  并发用例通过**；单次分配异常保留的 raw/work/steps 在 backup 中，而非原位。
  最后默认 asset-free contract **16/16 通过、0 失败、0 CTest skip，11.35 秒**。
  探针/Web 门禁只使用独立临时目录/假 CLI；默认套件仍含既有底层 CUDA 测试。随后独立审查回传：固定源码包 v2 的 host Web、
  23 项故障断言、三种事务单次分配回归与 6 项确定性并发均通过；这不代表独立重跑
  全部 16 项设备 contract。异常保证可恢复备份，不保证自动恢复原路径/持续可用。

- 同日 header 增量以异常安全修复源码包 v2 为独立基线，保留该快照。真实 HTTP 红测试
  有 **29 个失败断言**：23 类拒绝输入未能正确拒绝/保留旧素材，另有 6 类合法写法
  未正确处理。逐行 part header 与唯一 Content-Disposition 参数解析后，23 类拒绝输入、
  8 类合法写法及跨 part 重复字段通过；原 raw/work/marker 保留、回滚、preview 并发用例
  均保留。focused Web **1/1 通过，0.88 秒**；最终默认 contract
  **16/16 通过、0 失败、0 CTest skip，11.43 秒**。没有修改 v2 的事务实现或新异常门禁，
  新 header 代码尚待独立审查；不增加模型/音质/realtime 验收证据。

- 同日 Web 作业生命周期增量针对 P1 失败恢复缺口：step 的两处 promise 分配与 train
  的 job name 复制异常发生在 slot 预留后、原 catch 外，导致无 worker 但 running 永久 true。
  修复前隔离门禁复现 **3 处 stuck reservation**，重试均 409；扩展启动初始化的异常处理后，
  **step 8/8、train 9/9 单次故障通过，stuck=0**，启动前错误返回 503 后能重试；
  3/5 个响应异常场景仍保持已接手 worker 的 running，假 CLI 退出后为 done/error，
  原 raw/work 保留且 staging 清理。最终 focused Web **3/3，0.95 秒**；默认 asset-free
  contract **17/17、0 失败、0 CTest skip，11.49 秒**。构建/CTest 仍为上述串行低优先级
  命令，新增 `test_web_job_lifecycle` 不加载模型。该固定 v4 增量随后完成部分独立复验，范围见下条。
- 后续请求资源门禁在原生产实现上 **1/1 通过，0.89 秒**，确认缺的是自动证据：
  128 MiB+1 声明提前 413；两份 128 MiB 声明占满预留后 503；EOF/RST 后恢复容量，
  accepted socket 关闭，33 次串行截断 EOF 可持续复用连接；job/material/CLI 记录不变。
  新用例最多 3 个客户端，body 最大 215 bytes，无真实素材、模型训练或长超时等待。
- 已取回并只读核验部分独立审查证据：Library
  `libfile_68d884c3ed948191951d3da71c8387c6` v0，ZIP SHA-256
  `9b327e92fae588711d51deaed6e76155adf76d2973040e301d66d4ef7750de3d`。
  host GCC 14.2/C++20 已复现 v3 三处失败及 v4 的 17 点通过；还复现既有 train 上传
  失败分支先 finish、错误响应抛异常后再次 finish，误清除 B 作业并接受 C 的竞态。
  此分支早于 v4；GPU 执行仍有 mutex 串行保护。按报告正常修复：提交 catch 仅覆盖
  发布，释放与响应放在 catch 外，只释放一次。本轮现有 Web 三项 **3/3，0.92 秒**，
  默认 **17/17、0 失败、0 CTest skip，11.45 秒**；SDD 32 项通过。
  审查任务后续被工具安全检查中断，不能称完整审查通过。仅阅读已完成日志/源码，
  未执行下载包探针、未重试或绕过被阻止动作；修复后的精确 A/B/C 并发场景尚未复跑，
  已有三项门禁通过不能代替这个独立复验。v4 固定快照继续保留。

- header 固定源码 v3 已收到独立审查通过结论：复现 v2 的 29 处失败及 v3 绿测试；
  额外 23 类拒绝均保留旧素材、5 类合法语法通过；ASan/UBSan 下 10 万种有种子变异
  和边界检查通过，未发现新增可操作缺陷。LeakSanitizer 因 ptrace 环境不可用。
  该审查对设备 16-test/SDD 只审阅日志，没有独立重跑，不能据此声称设备全套独立通过。
  v3 的代码/证据快照继续保留；上述结果不覆盖后续生命周期增量。

- 同日完成两个保持生产行为的验收项。新增门禁保持生产行为：`test_web_contract` 在未完成请求头的第0/3/6秒滴流，
  慢header共53 bytes、最多2客户端；10050 ms后400与EOF，socket关闭，job/CLI不变、
  后续状态请求成功。观测窗9–13秒，整个用例<15秒；120秒body deadline未运行。
  `test_web_process_contract` 直接调用原runner，验证空argv/EINVAL、缺失程序/ENOENT、
  日志为自建目录/EISDIR、自有短命子进程自行SIGTERM的分类与输出回收；4/4失败分类和
  4/4后续正常调用通过。整组34 ms、输出178 bytes、父子peak RSS上界6912 KiB，FD不增，
  无存活或僵尸直接子进程、临时目录清除。最多1个helper，每次自然失败<2秒，CTest上限10秒。
  初版测试因设备内核不提供`/proc/.../children`在用例前失败，改用只读proc父PID元数据后通过；
  这是测试兼容性修正，不是产品红绿缺陷修复。不抢waitpid所有权、不引入interposition。
  这些证据不覆盖Web对信号/wait/log I/O结果的消费、底层wait/I/O故障、后代进程组或取消。

  最终默认asset-free contract **18/18、0失败、0 CTest skip，21.94秒**；SDD32项通过。
  生产源码与v5相同，全部串行/nice 10，未运行120秒body、真实模型/训练或取消/进程组实验；
  默认套件仍包含原有底层CUDA门禁。新增测试尚无独立审查结论。

- 后续独立D2验收：独立D2门禁已在OrinNX通过：第0秒完整header和部分body，第20/40/60/80/100秒各补1字节；
  **120708 ms**后服务端400与EOF，整个CTest用例120.74秒（总墙钟120.75秒），小于130秒。
  声明body为**1024 bytes**，实际body为**216 bytes**；包含就绪/状态查询的全部请求共**570 bytes**，
  不应把声明量当作实际传输。最多2客户端，未提前完成body、半关闭或RST；accepted socket关闭，
  后续状态查询200、job/raw/work/steps/CLI记录不变、无staging，自有监听器已回收。
  中途父子进程RSS采样7720 KiB（单点采样，不是峰值上界）。本轮对原生产实现直接通过，未改生产逻辑。

  长用例用默认OFF的`VOXMUTATIO_ENABLE_SLOW_WEB_TESTS`单独注册`test_web_body_deadline`，
  复用测试程序的`--body-deadline-only`模式；label为contract/web/host/slow，CTest TIMEOUT=130。
  不放入原60秒Web门禁；该D2验收后已恢复OFF，当时默认18项、快速回归18/18、0失败/skip、21.71秒。
  完整资产配置42项的口径同样以slow选项OFF为准，ON额外注册1项（未据此重跑模型套件）。


- 后续P2：P2消费端门禁`test_web_result_contract`已通过：通过现有`ProcessRunner`依赖接口模拟
  `kSignaled/SIGTERM`、`kWaitFailed/ECHILD`、`kExited/exit_code=0/io_error=EIO`，覆盖preprocess、
  convert输入预处理、convert主处理、step、训练preprocess/train/index共7个路径；**21/21失败
  与21/21正常重试通过，58 ms**。替身在失败前仍写出有效格式的小产物，避免缺少输出掩盖结果忽略问题。
  同步响应500且无成功音频；后台最终running=false/done=true/error=true；旧raw/work/steps/clips/
  model/index不变，下游不执行；等待worker共享所有权结束后检查最终staging/临时目录清理，重试成功发布。
  主fixture使用唯一临时目录；同步handler自建的唯一临时目录也逐项确认移除。无FD增量，真实子进程0，
  fixture清理前文件合计3504 bytes、进程peak RSS3712 KiB；请求≤4KiB、响应≤8KiB，CTest硬上限5秒。
  初版测试有一处initializer括号编译错误，修正后首次执行即通过；没有产品红测试，生产文件与v7逐字节一致。
  P2结束时默认快速回归为**19/19，0失败/skip，21.78秒**；长门禁仍默认OFF，本轮未重跑未改动的D2。
  本项只验证Web对模拟结果的消费，不制造或证明OS信号/wait/I/O故障；无interposition或故障探针。


### 已授权训练取消增量

训练取消增量（WEB-FR-011）已按2026-10-10 12:41:53 UTC用户授权实现，基线固定为审查源码v8、证据v9。
`POST /api/train/cancel`只接受单个`job_id`；首个有效请求202、重复请求200，过期id/step/已完成或
进入最终发布的作业409，无作业404，非法输入400；继承现有Bearer鉴权。接受与状态响应含作业id，
状态区分`cancel_requested`和`cancelled`。排队worker经条件变量唤醒；活动CLI自然返回后才在阶段
检查点停止，绝不把flag当作已停止。请求/worker按代号持有训练预留，临时目录及GPU锁收尾后才发布终态。
已开始的raw/work提交完成并保留；最终clips/model/index发布开始后拒绝过晚取消并完成提交/回滚。
真实CLI失败即使同时请求取消仍为error。前端按钮跟随确切作业，取消终态不显示“已生成声线”。

验证：旧实现红测因缺job_id失败；新增取消门禁覆盖排队中GPU仍被他人持有、三个CLI阶段、重复与
过期请求、未收尾保持busy、重试、已提交上传保留、8次完成/取消竞争、取消期间失败和step排除。
同一门禁检查旧产物、临时目录与FD；仅用现有ProcessRunner替身，没有真实模型或子进程。
真实HTTP门禁增加method/Allow、Bearer与非法取消参数；Node.js 18.19.1手动运行
`node tests/test_web_train_cancel_ui.cjs`，通过完整app.js语法及mock DOM/fetch的状态竞态检查，
不代表浏览器视觉验收。首次UI替身遗漏cancel_requested字段，补齐契约字段后通过，保留失败日志。
固定v9默认无模型回归 **20/20、0失败、0 CTest skip，22.00秒**，全部串行/nice 10；
取消专用门禁0.05秒，原启动17点及结果消费21失败/21重试回归通过。长body门禁保持OFF，未重跑既有D2。

限制：CLI没有训练循环内协作停止接口，等待当前CLI自然结束的时延不设上限；无timeout、信号、
强杀、step/同步任务取消或resume。精确旧A/B/C探针、预留前filesystem异常terminate观察、真实OS
wait/I/O及真实模型/音质缺口仍保留。不把Spec003整体标为Complete。正式ADR审查状态仍为Proposed，实施授权及回滚见
[ADR-0006](../docs/adr/0006-training-cancellation-ownership.md)。

后续独立审查回传（2026-10-10，固定源码v9、27项哈希一致）：未发现已确认的训练取消缺陷；
host取消、mock UI及额外重复提交/无ID取消/重复取消/新作业恢复检查通过。该结论不覆盖以下新测试接缝。
原8次最终发布竞态只接受两种合法结果，不保证命中“发布已开始”窗口；丢失初次提交响应或刷新页面
后无法恢复跟踪仍是既有限制，本次未修复或扩展会话恢复。

素材提交中取消的独立补充：`test_web_train_cancel_commit`在专属测试目标重编译Web支持源码，
只对该目标定义`RESONANTIA_TEST_MATERIAL_COMMIT_GATE`，于raw安装完成、work安装之前通过条件
变量暂停。现有事务测试依赖rename拦截，本补充不使用它、不拦截系统调用、不生成OS故障。
普通Web对象和`vc_serve`重建后SHA256与v9逐字节一致，生产库无暂停符号；没有运行时/API改变。
测试先在无调用的v9上因未进入gate失败（这是观测接缝红测，不是产品缺陷），随后控制组与取消组
**1/1通过，0.01秒**：中途取消202/重复200、仍running且新train/step/material均409；释放gate后完整
提交raw/work、退役旧marker，再发布cancelled；旧模型保留、后续CLI调用为零；终态重复取消200，
旧id无法取消下一作业，下一worker在自然返回前不释放预留，正常重试成功。仅tiny临时素材及既有
ProcessRunner替身，FD无增长、staging/backup清理。首次测试编译误复制不可复制Request，修正字段构造后运行红测。
本补充串行回归执行**20/20通过、0失败、21.67秒**；当前注册21项，主动排除内部拦截rename的
`test_web_transaction_failure`一项（未运行，不算通过或CTest skip），遵守本轮禁止系统调用拦截的约束。
原取消及新提交用例focused 2/2通过（0.06秒），不重复运行无变化的UI/长body门禁。
这一补充没有发现需修复的产品缺陷；新接缝尚无独立审查，不扩展到最终发布窗口的确定性暂停、
崩溃/回滚异常或真实训练。固定Library源码v9/证据v10/计划v4保持不变，另存本轮增量和日志。


- 2026-10-10 当时 CMake 在资产齐全配置下注册 **45 个 distinct tests**：22 个带 `contract` label、
  24 个带 `integration` label；`test_bounded_file_wavlm_preflight` 同时带两个 label，因此
  22+24 不是 distinct 总数。2026-08-12 的 fresh Release/SM87 build
  `/tmp/resonantia-nonrt-final.a8O1Ay` 已执行全部 39 项：**39/39 通过、0 失败**，
  `Total Test time (real) = 1297.94 sec`。关键耗时为 `test_cli_bounded_callers`
  75.89 秒、`test_long_audio_chunking` 438.00 秒、`test_bounded_file_resources`
  326.92 秒、`test_bounded_file_fcpe_v1` 249.34 秒。
- 2026-08-12 在 Jetson Orin NX Engineering Reference Developer Kit（SM87）、CUDA
  13.2.78、CMake 3.28.3 上，`/tmp/resonantia-audit-final.VbIWW3` 的 30/30（11 contract、
  19 integration、356.41 秒）是本轮加固前的历史基线，不证明当前修改。最终仍使用：

  ```bash
  cmake -S . -B <build-dir> -DCMAKE_BUILD_TYPE=Release \
    -DVOXMUTATIO_CUDA_ARCHITECTURES=87 \
    -DVOXMUTATIO_ENABLE_MODEL_TESTS=ON
  cmake --build <build-dir> --parallel 2
  ctest --test-dir <build-dir> --output-on-failure
  ```
- 模型权重和数值参考 fixture 不入 Git。缺少资产的集成测试由 CMake 不注册，不能把
  “未执行”解释成通过。
- bounded file focused resource gate 在 Orin/CC 8.7/Release、相同 HuBERT/RMVPE/v2 config
  下比较 17.565/35.130 秒真实语音：peak RSS `1342624`/`1354844 KiB`（增加 `12220 KiB`，
  允许 `65536 KiB`），device growth `892334080`/`851812352` bytes，RMVPE F0 hard input
  `256000` samples。只有两个时长点，临时磁盘仍为 `O(duration)`，且没有独立冻结的
  click/transient 指标；该 focused 结果已由上述 39/39 全量包含执行，但仍不能
  外推为任意时长的经验保证。
- 当前没有可跨机器复用的产品吞吐、延迟或显存基准；standalone exact-Index probe 只有
  synthetic/available-feature stage 诊断数据，不是产品 backend、支持上限或 realtime SLO。
- Spec 006 的 [R0 asset manifest](006-realtime-voice-conversion/r0-assets.json) 已对 7 个 present
  文件、2 个本地 conversion inputs、3 个 generation scripts 和 `tools/uv.lock` 完成
  path/bytes/SHA-256 核验；FCPE official package/checkpoint provenance、locked project 与
  same-filesystem staged + rollback 事务式发布已固定。qualification gate 现在只因
  provenance-tracked 目标声线 v2/40 kHz
  model，以及与同一份有授权 corpus 匹配的 Flat Index 缺失而按设计失败，不能解释为
  realtime 资产已就绪。
- 本轮 SDD bridge 已把 ADR-0004/0005 纳入 required/link contract；2026-08-12 执行
  `bash scripts/check_sdd.sh`，32 个 required files/link 全部通过；`git diff --check` 退出 0
  且无输出。该文档门禁不替代 C++ build/CTest。
- ADR-0001 R0–R5 已实施：12 个内部 OBJECT 组件聚合为公共 `voxmutatio` static facade；
  四个主要 CLI 分别使用内部 command-scoped app-support target；`vc_serve` 使用内部 Web
  support 模块。既有 target 名、安装面和公共库 API 未因重构扩大。
- ADR-0004 R1–R5 已实施：additive bounded file API 保留 legacy owning APIs，HuBERT
  `vc_convert`/`vc_batch` 使用 bounded path，WavLM 在匹配资格资产出现前保留 legacy。
- ADR-0005 已实施：Flat Index active bytes 来自 private unlinked filesystem snapshot；
  `build_index` 确定排序并原子发布。copy fallback 的临时容量成本仍是部署边界。

## 当前能力矩阵

| 模块 | 当前状态 | 事实边界 |
|---|---|---|
| WAV I/O / 重采样 | 已实现 | WAV 读写、声道下混、线性重采样；不解码 FLAC/MP3 |
| safetensors | 已实现 | 通用消费者使用 file-backed mapping；VITS checkpoint 以 256 MiB cap 复制到共享 read-only anonymous snapshot；PyTorch 权重仅可离线转换 |
| HuBERT | 已实现、默认 | v1 输出 256D，v2 输出 768D |
| WavLM | 已实现、可选 | 需要与 WavLM 特征匹配训练的 synthesizer；不是默认 RVC 路径 |
| RMVPE | 已实现、默认 | 完整运行时 tensor manifest、有限值/输入边界和失败重初始化均有门禁；持久共享只读 checkpoint 映射；算子临时 device buffer 由 RAII 管理 |
| FCPE | 已实现、可选 | 完整 tensor manifest、weight-norm 安全物化、有限值/输入边界和失败重初始化均有门禁；R0 已固定精确来源与 same-filesystem staged + rollback 事务式发布，不据此承诺实时性能 |
| Flat Index | 已实现 | private unlinked filesystem snapshot 上 mmap exact weighted k-NN；优先 FICLONE、否则需约一份 Index 临时 filesystem 容量；非零 rate fail-closed；standalone GPU probe 非产品 backend |
| F0 VITS v1/v2 | 已实现 | CUDA FP32；v1/256D、v2/768D；完整 manifest/weight-norm；VITS-only 256 MiB anonymous snapshot + path identity；所有公开推理/debug 对尺寸、finite 与内部异常 fail-closed；当前仅 40 kHz，拒绝 non-F0 |
| 离线变声 pipeline | 已实现 | HuBERT file callers 使用 bounded spool/window/atomic writer；直接 E2E matrix 为 RMVPE/v2 + FCPE/v1，另有两种 F0 window profiles。WavLM、legacy file/buffer 仍为 owning/unbounded；不是 realtime |
| 训练 | 实验性、部分完成 | `vc_train` 有 decoder/GAN 路径；旧 `FineTuneTrainer` 不支持 |
| Web/CLI 编排 | 实现存在、验收部分 | `vc_serve` 是唯一受支持 Web 入口；`/api/convert` 的正数 `index_rate` 缺 voice Index 时返回 400 且不启动 CLI，零值显式走 no-index；默认 `0.0.0.0` 且无认证；旧嵌入式 `WebServer` 已 deprecated/fail-fast；非实时服务 |
| 声源分离 | runner 实现完成、实验性 | UMX、MelBand-RoFormer、Silero VAD；完整已用权重及公开 finite 边界有资产门禁，质量仍未产品化 |
| 端到端实时变声 | Spec 006 approved / R0 部分完成，产品未实现 | 已有 legacy `infer_stream` characterization、模式感知报告 validator、asset manifest、离线 Index fail-closed 与独立 Index 诊断 probe；仍没有 persistent runtime、session、`vc_realtime` 或 ALSA 产品路径 |

## 离线变声支持契约

| 维度 | 支持 | 明确不支持或会拒绝 |
|---|---|---|
| 设备 | CUDA | CPU 推理 |
| 精度 | FP32 | FP16 |
| 文件格式 | WAV 输入、WAV 输出 | FLAC、MP3 等 |
| 内容编码 | HuBERT（默认）、WavLM（可选） | 与 synthesizer 特征空间不匹配的组合 |
| 文件资源路径 | HuBERT CLI/batch 使用 bounded；WavLM 使用 legacy | 把 WavLM 静默换为 HuBERT；把 legacy API 宣称为 duration-independent |
| 模型 | F0 v1（256D）、F0 v2（768D） | non-F0 |
| F0 | RMVPE（默认）、FCPE（可选） | PM |
| 输出采样率 | 当前兼容模型的原生 40 kHz | 32/48 kHz 等其他模型率；独立 `target_sample_rate` 转换 |
| 变声参数 | pitch、index rate、RMS mix、protect | formant shift |

字段存在不表示功能存在：`device=cpu`、`use_half_precision=true`、
`F0Method::kPm`、非零 `formant_shift`、非 40 kHz `model_sample_rate`、非模型原生 `target_sample_rate`
和非 WAV 输出都必须在初始化或参数更新时失败。

## 规格状态

| Spec | 状态 | 说明 |
|---|---|---|
| 001 Voice Conversion Engine | bounded HuBERT file path 已实现 / 规格验证部分完成 | R1–R5、Index/VITS snapshot、代表性 CUDA failure gates 已纳入；真实 CLI callers 已有资产门禁，本轮全量见验证基线；`vc_probe` 仍无产品验收 |
| 002 Voice Training | 实验性 / 部分完成 | 可运行的 decoder/GAN 实验路径；无生产质量与 resume 保证 |
| 003 WebUI | 实现存在 / 验证部分完成 | 默认开放、可选 token、严格参数、正数 Index rate fail-closed、显式 no-index、step work+marker/训练三产物事务、FD 隔离与四类 CLI 失败传播有 HTTP 门禁；资源/恶意输入矩阵仍缺 |
| 004 Source Separation | runner 实现完成 / 实验性 | 模型集成、CLI model/task 拒绝和 Web step 事务有门禁；五任务质量/资产覆盖仍不完整 |
| 006 Realtime Voice Conversion | Approved / R0 evidence 部分完成 / 产品未实现 | ALSA、v2/HuBERT/FCPE/40k + 有效 Index、250/350 ms SLO、drop-oldest 已批准；独立 probe 只证明 exact retrieval 的诊断可行性。目标 model+matching Index/licensed corpus 缺失，产品 realtime/质量/SLO 仍未资格化 |

## Web P1 验收复盘（2026-10-10）

下列结论按原 Spec 003 子项归档；已闭合的是列出的范围，整份 Web 仍为部分验收。

| 原计划/契约 | 已有可复验证据 | 未闭合或明确边界 |
|---|---|---|
| FR-004 / §5 上传语法与资源 | boundary/part header、16 KiB field / 128 parts、拒绝保留旧素材；header v3 独立审查通过 | header/file 与完整大 body、同时连接上限仍缺；120秒body已独立可选通过；停止解析器边缘扩展 |
| §5 请求资源释放 | 声明 body 超限提前拒绝；256 MiB 预留耗尽及 EOF/RST 后恢复、socket 关闭；33 次串行 EOF 名额复用；job/material/CLI 不变 | EOF/RST门禁只传最多215 bytes；另有D2真实body超时（216 bytes）socket/后续可用性证据；完整大body、同时32连接拒绝、精确RSS仍缺 |
| FR-007 素材事务 | raw/work/marker 一起替换；常规失败回滚、非常规 marker 拒绝；material/step/train 分配异常保留备份；23 项独立故障断言 | 更多清理/回滚组合未穷尽；异常备份可能需人工恢复，无外部写者或崩溃原子性保证 |
| FR-007 预览一致性 | 256 上传/1024 预览、六项暂停 rename 的成功/回滚窗口；8 MiB 慢预览不阻塞提交 | 锁内文件 I/O 无硬延迟保证；大音频 streaming 尚未设计 |
| FR-008 / FR-010 临时资源与终态 | HTTP 假 CLI 非零退出、空产物、step/训练发布失败及旧产物保留；默认门禁验证 staging 清理 | 自然launch/exec/signal runner分类、回收及再次调用已有门禁；P2指定三类结果×7消费路径已通过；底层真实wait/I/O和清理失败诊断仍缺 |
| P1 生命周期 / FR-006、010 | 初始化失败已有17点门禁；既有重复finish修复后，新增普通host A/B/C四种交错、旧training/step完成重放及两个负对照通过 | 原受限探针未重跑，未新增独立审查；未覆盖全局/worker OOM；预留前filesystem分配曾触发标准库terminate，待单独调查；训练取消阶段边界见FR-011；timeout/process-group仍未实施，不启动真实训练验证 |

## 下一步

1. **P1：训练产品化**——补齐持久化 checkpoint/optimizer state、断点续训、大数据集
   稳定性、失败恢复和人工/客观音色质量验证；决定是否替换或删除旧
   `FineTuneTrainer` 门面。
2. **P1：分离验证与发布**——明确权重来源、许可证、资产清单和真实数据质量基准。
3. **P1：Web 本轮指定轻量硬化收口**——上传解析、材料/步骤/训练事务与预览、作业启动17点、
   EOF/RST请求资源门禁，以及D1/D2/P1/P2指定验收均已有证据；边界与集中闭合清单见
   [Web验收计划§6](003-webui/acceptance-plan.md#6-本轮收口与用户决策)。不把整份Spec003标为完成。
   v5重复finish窄修复现有[§8普通host等价所有权门禁](003-webui/acceptance-plan.md#8-abc所有权与重复finish后续普通host验证)；
   原受限A/B/C探针未重跑，预留前filesystem分配terminate仍待调查。更广OS/压力未验证项不自动扩展为追加实验。
   用户已批准并完成训练阶段及完整迭代协作取消编排（FR-011/TR-FR-015），有host证据；真实GPU取消未验收，timeout与强制终止未实施。
   所有权增量的后续安全发布已授权：push只触发托管host检查，设备保持手动/就绪/专属标签门禁。
   后续设备执行仍需先核实runner唯一归属并单独授权；不扩展到真实训练、部署或已中止探针。
4. **实时能力暂缓**——Spec 006/ADR-0002 保持 Approved / R0 evidence 部分完成 / 产品未实现；
   本轮不接 persistent runtime、session 或 ALSA。恢复时仍需有授权 corpus、匹配 target
   model+Index 及 Index-enabled 质量/SLO 证据。
5. **持续加固**——保持 contract 与 asset-gated integration 两层测试，继续覆盖输入
   上限、原子产物、回滚与服务暴露边界。

### 2026-10-10 后续授权：完整迭代协作取消

CLI可选作业文件令牌、退出2、完整GAN D/G与decoder迭代边界已接入Web；source/lease等待写操作结束、
runner自然reap和staging清理后才释放预留，实际错误优先。详见[DESIGN](../docs/DESIGN.md)。
保留固定v9和素材提交测试快照，另存本增量；仅host验证与编译，不推断GPU停止/音质或resume完成。
