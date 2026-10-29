# Web长任务改进实施清单（2026-10-29）

状态：本轮已部署并完成验收。依据[WEB-FR-014](spec.md#2026-10-29-后台长任务web-fr-014)和
[ADR-0010](../../docs/adr/0010-durable-web-task-lifecycle.md)。完成边界与证据见文末；历史分批记录保留当时状态。

## 完成标准与顺序

- [x] 只读监控用户当前转换结束，取得实际失败反馈；未重启服务或启动额外模型任务。
- [x] 复现并修复44.1k→40k→16k→40k→44.1k整数舍入导致的混音失败；真实长度不匹配仍拒绝（host回归通过，已部署）。
- [x] 有界持久任务存储、状态、列表、控制、结果/日志API；关闭页面不影响后台所有权。
- [x] 转换/清理工作移至独立任务，固定原输入/参数/模型身份，保存阶段产物，失败可复用。
- [x] 训练接入任务记录与历史；保留已有取消/完整checkpoint数值语义，恢复活动和终态。
- [x] 统一任务中心与训练/翻唱入口：真实阶段、进展更新时间、计数、日志、后台继续提示。
- [x] 长阶段结构化进度、安全暂停与分块恢复；阶段复用与分块恢复分别验收。
- [x] 服务重启、响应丢失、断线/重连、暂停/继续、失败重试、磁盘失败与历史清理门禁。
- [x] 真实短片段和joint-full少量步恢复验证；完整相关回归/自审/文档检查。
- [x] 无活动任务时部署，真实内网关闭/重开验证；不额外训练正式声线或跑整曲。

## 本轮已有证据

本地目录`runs/Web_Task_Lifecycle_20261029/evidence/`：
- `baseline.patch`：开始修改前的未提交中文文件名修复，必须保留。
- `current-task.json`：实际用户转换失败观察；后端未保存最终耗时，不补造成功记录。
- 源音频44100Hz/11535545帧，经现有整数重采样链预计回混11535542帧，差3帧，旧容差2帧。
  该数学重现不能替代已删除输出的直接测量；另用真实I/O重采样替身建立host回归。
- `mix-red-test.log`：44103帧短素材复现同一混音500；`mix-green-test.log`：原门槛兼容、
  精确舍入链接受、额外400帧截断拒绝，1/1通过0.53秒。没有执行模型推理。
- `store-red-build.log`：新任务存储尚未实现时链接失败；`host-focused-test.log`：任务存储与
  上传/混音2/2通过0.56秒。持久化、重开读取、旧修订拒绝、容量/路径/坏格式、启动标记interrupted
  已有host门禁；尚未接入Web请求和真实后台任务，不能宣称页面恢复已实现。
- 完整构建首轮因新增混音测试缺少`voxmutatio`头搜索依赖失败；补齐目标显式依赖后完整构建通过。
  `web-increment1-test.log`：Web **19/19通过，0失败/skip，16.18秒**，含原有真实codec门禁和新任务
  存储。日志`build-increment1.log`保留首次构建失败，`build-increment1-recheck.log`保留复查。
  当前生产PID2034441仍运行此前版本，尚未部署本轮混音/任务存储代码。

## 授权边界

沿用已讨论监控/恢复方案，覆盖推理、训练与清理。保留Tom Cybersinger及全部正式训练数据。
不新增长训练或整曲模型推理测试。当前正在运行用户任务必须先自然结束；部署前再次检查。
没有通过的恢复层级必须如实报告；不以浏览器mock、瞬时GPU利用率或计时动画代替真实执行证据。

## 第一批持久查询接口契约

- 生产服务设置`Paths.tasks=runs/.web_tasks`并独占存储；嵌入式旧测试留空保持原接口。
- `GET /api/tasks?offset=0&limit=25`按创建时间倒序，limit 1..100、offset 0..4096；
  返回`tasks/offset/limit`。`GET /api/task?id=...`返回单条；`GET /api/task/log?id=...`
  返回最多64KiB日志尾部。非法参数400，不存在404，方法错误405，持久存储失败503。
- 记录包含稳定ID、类型、名称、状态、实际阶段、错误、毫秒UTC时间、真实计数和固定参数；
  没有实际计数时total=0，不生成估算百分比。旧训练响应job_id同时作为初次任务ID。
- 训练/素材清理在取得任务所有权时先写记录，阶段更新在开始阶段前提交，结束保存日志快照后
  提交终态；页面断开不影响已启动worker。旧`/api/train/status`与取消身份契约保留。
- 服务重启把旧活动任务标为interrupted，保留阶段和计数，不能自动启动训练。记录失败时
  禁止接收新的后台工作并返回存储不可用；不把未写入的成功状态伪装为已保存。
- 此批不代表继续/重试、分块断点、推理后台化或页面重连已经完成；后续逐项验收。

## 第二批后台音频接口契约

- `POST /api/tasks/convert`复用旧转换参数及上传/session输入；新增必填稳定`task_id`
  （8..96字节ASCII字母数字、下划线、连字符）。`POST /api/tasks/preprocess`使用同一输入/
  清理配方，只输出清理后的音频。新接口202返回task_id，旧同步接口仍返回WAV。
- 相同ID已经存在时409并带task_id；客户端应查询该ID，不产生第二次模型执行。该增量以
  拒绝重复实现防重，不把冲突请求冒充新任务接受。控制的执行代次仍待后续实现。
- 接受前复制输入和动态G/索引到任务私有work目录、保存参数；工作线程只拥有值和持久路径，
  不持有Request/Part/socket引用。GPU锁由执行线程自己取得/释放。提交连接断开不会撤销工作。
- 阶段为separating、cleaning、preparing-vocal、converting、mixing、saving；由实际代码
  到达该阶段时保存，不生成模型内百分比。现阶段没有阶段内计数/暂停/继续入口。
- 成功结果固定`work/result.wav`，文件及work目录sync后提交终态；GET单任务返回result_url。
  `GET /api/task/result?id=...`只供已成功音频，未完成/失败409，文件丢失410；下载失败不改任务状态。
  按64KiB流式读取，单个byte Range支持闭区间/开放尾区间/后缀，206或416；重复Range头400。
  结果上限1GiB，上传限制保持原样。旧同步音频响应仍为64MiB上限。
- 失败保留任务目录、阶段产物、日志与错误；这仍不等于可安全重用阶段或分块续跑。
  基础HuBERT/RMVPE、分离权重、二进制身份校验，阶段完成标记与哈希校验、磁盘预算/历史清理，
  都仍待完成；当前不得开放“继续”按钮。
- 每个训练/清理任务使用独立live.log，结束时保存最多64KiB快照，不受下次同名训练影响。

## 后端增量验证记录

本轮所有后台生命周期试验使用隔离目录和合成ProcessRunner，不执行模型或CUDA；它们验证任务
所有权、HTTP、文件保留和故障处理，不作为音质或真实模型恢复验收。

- `api-red-test.log`：新增任务查询返回404（未实现）使测试失败；`api-green-test.log`首轮通过。
- `async-red-test.log`：后台转换入口未实现返回404；`async-green-test.log`：任务、旧结果契约、
  上传/混音3/3通过0.74秒。没有用mock结果冒充真实翻唱完成。
- `range-red-test.log`：Range仍返回完整200使测试失败；`range-green-test.log`：206/416、后缀/
  开放范围通过。`api-failure-test.log`：70MiB流式下载、下载断线、丢失结果410、终态日志保存失败
  返回503且保留音频、重启标interrupted，通过0.30秒。日志写失败由占用目标路径重现，**非真实
  磁盘满/断电试验**，这些更广故障仍待验收。
- 最终完整构建`cmake --build /tmp/resonantia-performance-20261024.uemhTG -j 2`通过，日志
  `build-increment2-final.log`无编译警告/错误。`ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L web --output-on-failure`
  **20/20通过，0失败/skip，17.06秒**，记录`web-increment2-final.log`，包含真实codec测试与所有旧Web门禁。
- `bash scripts/check_sdd.sh`通过32项路径要求；`git diff --check`通过。尚未运行新的真实音频
  后台模型/暂停恢复或新版浏览器关闭重开试验。

后续先补真实执行进度、依赖/阶段身份与控制状态机，再用隔离静态目录完成页面接入；生产webroot
当前就是工作区`webui`，编辑前须避免把未验收页面即时暴露给旧服务。全目标仍在推进，不勾完整任务。

host最终复查：`cmake --build /tmp/resonantia-web-release-host -j 2`成功，随后
`ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure` **13/13通过，0失败/skip，
1.27秒**；日志`host-increment2-final-build.log`与`host-increment2-final-test.log`。
最后检查生产服务仍为PID2034441、active，本增量没有替换或重启生产，没有新模型转换/训练进程。

## 第三批真实进度契约（实施中）

领域库提供同步可选ProgressObserver：phase、completed、total、unit；total=0时completed=0，
代表没有可量化总量。回调借用字符串只在调用期间有效，不重入模型；异常沿原调用的失败契约传播。
它不读取任务/HTTP，也不把进度回调本身称作可恢复安全点。

CLI显式`--progress`输出有界独立行`RESONANTIA_PROGRESS 1 <phase> <completed> <total> <unit>`，
每次flush。unit限定unknown、files、chunks、steps、samples16k、frames100hz。没有该参数保持原输出行为。
Web的ProcessRunner在自身线程观察有界日志行；回调失败必须继续排空并回收子进程、报告I/O失败，
不能把观察失败变成子进程泄漏或假成功。普通日志不以正则猜测为精确进度。

Web记录保留总阶段stage，新增phase表示CLI内部阶段。实际边界更新completed/total/unit/
progress_at_ms；轮询本身不修改进展时间。不伪造整个流水线的总百分比。
持久更新最多约每秒一次，阶段变化/阶段完成必须保存；原HTTP兼容接口的日志解析暂保留。

## 真实进度增量证据（2026-10-29，未部署）

- `progress-red-build.log`：协议接口无实现时链接失败；`observer-red-test.log`：ProcessRunner
  未观察过滤日志而失败。首次实现暴露“回调失败后不再保存子程序日志”，保留
  `observer-green-test.log`失败记录；分离观察错误和日志I/O错误后，
  `observer-green-recheck-test.log` **2/2通过0.05秒**，同时验证回收子进程和FD不泄漏。
- `progress-api-red-test.log`：后台未保存真实子程序计数而失败；随后任务与存储
  **2/2通过0.35秒**（`progress-api-green-test.log`）。TaskRecord新增phase；stage变化清除旧
  子阶段计数。轮询不制造时间进度，CLI观察按约1秒节流，子阶段变化/完成立即保存。
- 完整构建`progress-native-build.log`通过。指定CLI/协议/任务/进程及现有joint-full真实门禁
  **5/5通过298.41秒**（`progress-focused-test.log`）。测试选择中写了可执行目标名
  `test_vc_convert_bounded_contract`，其CTest名称实际为`test_cli_bounded_callers`，因此这项
  **未运行**，不计为通过；后续新增的短真实进度测试单独覆盖本增量转换路径。
- 其中`test_joint_full_cli`用1段1.28秒LibriSpeech，6步基准、同一第5步checkpoint继续、
  安全停止与继续对照；原模型/optimizer/epoch/RNG数值门槛全部通过（291.18秒）。该次没有打开
  `--progress`，证明旧训练语义未回归，**不是训练进度协议或Web暂停恢复的真实验收**。
- `test_operation_progress`采用`tests/fixtures/speech_02.wav`前6秒及真实HuBERT/RMVPE/G v2/
  karaoke Roformer，比较CLI不开/打开`--progress`：转换WAV逐字节一致；去和声浮点相对L2
  **5.53623e-08**、最大绝对误差**8.9407e-08**，门槛分别`1e-6`/`2e-6`；实际完成的F0/
  合成计数与至少2个分离块均验证，**1/1通过39.76秒**（`progress-real-recheck-test.log`）。
  分离iSTFT已有float atomicAdd，末位非确定性有实现证据，不要求跨运行float WAV逐字节相等。
  首轮选speech_00但它仅5.855秒，在模型执行前被6秒素材门槛拒绝；保留`progress-real-test.log`。
- 上述测试均使用独立临时目录，没有改变正式模型或启动整曲处理。正式服务和页面尚未切换。

## 下一批分块恢复实现约束（设计，尚未实现）

沿用ADR-0010，不用进程挂起冒充持久恢复。拟将有界block存储放入io层；模型层接收领域checkpoint
与可选停止回调，不识别Web任务ID/HTTP或产品目录。首次和继续调用必须显式区分。

1. 身份固定：输入字节、模型/索引、前端模型、分离权重及i64/config、完整数值配方、CLI实现身份
   都纳入摘要。Web参数和动态模型副本已保留，但缺少的基础依赖身份仍必须补齐。继续前先校验，
   不匹配或损坏必须失败，不能静默用新模型拼旧块。SHA-256复用训练已验证算法，并向正确依赖层
   收拢；io/core不得反向引用tools/training。
2. 块写入使用staging/fsync/rename/目录sync，一块包含版本、身份、绝对输入位置、浮点元素数和
   payload摘要。拒绝坏key、符号链接、非有限数据、截断/超长、错位置/形状和摘要；记录/文件数、
   单块和累计磁盘量均有界。目录flock单独占有。未提交staging不能当完成块，已提交块不覆盖。
3. F0按现有全局帧格和上下文缓存窗口的裁剪后输出；恢复先验证并重放已完成窗口，仍在完整F0后
   做全局平滑/音高编辑。不能独立对每块插值或丢失无声/尾音上下文。
4. 合成按原QuietCorePlanner重放切点，缓存每个core的裁剪/补齐后40k输出；已完成core跳过
   HuBERT/检索/VITS计算，保留原始绝对采样frame offset和seed。输入快照/便宜CPU全局滤波可
   重新生成，但不能把模型块重跑称作续跑。最后仍做原全局RMS与原子WAV发布。
5. Roformer缓存各窗口的完整推理输出。恢复按原顺序重放已完成块的窗口累加，重建out/wsum，
   再算未完成块；保持首尾窗、50%重叠、原累加顺序和归一化，避免每块重写全长累加器的I/O成本。
6. 停止仅在已提交块边界响应并释放GPU；模型/记录/I/O错误优先，不能误报paused。CLI受控暂停
   使用独立退出状态与明确checkpoint存在性；Web稳定任务ID另加执行代次，迟到/重复控制不能
   作用到后续执行。请求离开从不发送停止。存储故障时仍须允许安全停止已有工作。
7. 完成阶段保存经过校验的产物摘要；后续混音失败可只重新混音。未完整提交阶段可借native块
   checkpoint重建，不把残留WAV头或单纯文件存在当完成。训练仅joint-full提供完整状态继续，
   旧模式明确不支持等价恢复；训练预处理/缓存/索引也需要实质保留或安全重建边界。
8. 必须验证真实短音频连续/暂停继续的长度、拼接边界、数值差异及坏身份拒绝；分离至少覆盖两块，
   转换测试用短输入和足够小的测试分块配置，不能为验证而跑整曲。Web恢复还须验证重启、关闭页面
   期间终态、丢响应、失败重试及磁盘限制/清理。界面仅展示已实现并通过门禁的动作。

真实进度增量最终相关回归：
`ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L web --output-on-failure`
**21/21通过、0失败/skip、17.15秒**（`progress-web-test.log`）；
`ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^(test_bounded_pipeline_dsp|test_joint_run_state)$' --output-on-failure`
**2/2通过、0失败/skip、0.16秒**（`progress-dsp-test.log`）。
记录更新后再次检查生产仍PID2034441/active，无模型子任务；这轮没有部署生产。

独立host配置最终构建与全部测试也通过：
`cmake --build /tmp/resonantia-web-release-host -j 2`，
`ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure`，
**15/15通过、0失败/skip、1.24秒**；证据`progress-host-final-build.log`及`progress-host-final-test.log`。

## 分块保存与native/CLI恢复增量（2026-10-29，未部署）

上节设计的1–6项已在native及CLI层落地，**不是Web安全恢复完成**：

- `io::BlockCheckpoint`独占目录，版本化身份及不可覆盖块，SHA-256绑定key/offset/count/数据；
  缺块可计算，损坏/错形状/错身份拒绝。单块32MiB，每store4096块/2GiB；最多8194项有界扫描。
  失败写入不覆盖旧块，当前owner失效直至重开；只有没有任何块的目录允许修复初始identity提交
  中断。symlink/hardlink/非regular/非有限值/截断/超界均拒绝。真实断电/真实ENOSPC尚未模拟。
- core共用SHA-256，io共用稳定文件散列；原训练hash入口保留其符号链接接受语义。没有tools向
  io/core的反向依赖。原训练状态/缓存known-answer门禁3项（含新block）通过。
- bounded转换保存原全局F0裁剪窗口及原QuietCorePlanner绝对合成core，继续重建临时spool，
  全局F0修整与RMS仍在原位置。Roformer保存完整模型窗口，按原顺序重建overlap-add累加器，
  复用完成块时不运行相应模型。库可选checkpoint由调用者绑定全部身份，空指针走旧行为。
- `vc_convert/vc_preprocess --checkpoint-dir <dir> [--resume] [--pause-file <path>]`显式启用。
  CLI绑定输入字节、完整数值配方、实际模型/索引/分离json+i64/VAD（如启用）、当前运行的二进制
  摘要；模型加载前拒绝身份变化。每个预处理文件/阶段独立store，owner锁覆盖整个CLI会话。
  checkpoint配方不包含输出目录和临时pause通道；继续可以换输出目的地，但数值身份不可变。
- CLI退出0完成/2安全暂停/1错误。新Web尚未传这些选项或提供控制入口；进程退出2不能先当普通
  失败写入后再让前端猜测。下一批必须接入执行代次及pause_requested/paused/interrupted/resume。
- Roformer重建后的波形/切片仍由原输出流程写出；VAD和切片没有细粒度checkpoint。训练预处理
  阶段复用、完整产物哈希、任务级磁盘预算、服务重启及前端仍未完成。
- 严格二进制摘要意味着程序升级后旧块不能直接跨实现继续。Web须清楚报告该边界，不能默默重跑
  或删除旧恢复点；部署/回滚必须保留资产并按版本核对。

本地证据均在`runs/Web_Task_Lifecycle_20261029/evidence/`：

1. `checkpoint-red-build.log`：新store只有契约、没有实现时链接失败。
   首实现的三元表达式拼写导致`checkpoint-green-build.log`构建失败，修正后
   `checkpoint-green-recheck-test.log`1/1通过。`checkpoint-init-red-test.log`实际失败复现
   “初始identity提交中断不可恢复”，修正后纳入最终host门禁。
2. `checkpoint-native-red-build.log`：新增入口尚未实现时签名不一致失败。
   随后`checkpoint-native-green-build.log`暴露独立joint_run_state目标缺include搜索路径；补齐后
   `checkpoint-native-green-recheck-build.log`通过。
3. `ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^(test_block_checkpoint|test_joint_run_state|test_joint_cache)$' --output-on-failure`
   **3/3通过0.10秒**，日志`checkpoint-native-contract-test.log`。
4. `ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_operation_checkpoint$' -V`
   **1/1通过68.95秒**，日志`checkpoint-native-real-test.log`。真实speech_02前6秒、真实HuBERT/
   RMVPE/Gv2，1秒合成core配置使短样本跨多个块；暂停后重开复用至少F0+合成两块，最终PCM16 WAV
   与连续版本逐字节相同。全缓存回放Hubert/index/synth耗时计数均0，输出一致。真实karaoke Roformer
   至少两窗，第一窗后暂停，继续准确复用1窗；相对L2 **5.17645e-08**、max_abs **8.9407e-08**，
   门槛`1e-6/2e-6`；全缓存overlap-add回放逐float一致。暂停保留已有目标内容。
5. `checkpoint-cli-red.log`保留旧CLI缺少恢复选项的失败探针。新测试首次在重新配置CMake之前
   请求新目标失败（`checkpoint-cli-build.log`），随后缺少测试头include导致
   `checkpoint-cli-recheck-build.log`失败；补齐后`checkpoint-cli-final-build.log`通过。
6. `ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_cli_contract$' --output-on-failure`
   **1/1通过6.86秒**。`ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_checkpoint_cli$' -V`
   **1/1通过180.57秒**；日志`checkpoint-cli-contract-test.log`/`checkpoint-cli-real-test.log`。
   独立CLI进程用真实6秒录音：安全pause退出2、重开继续、修改pitch/min-rms身份拒绝、重复new拒绝、
   坏控制通道退出1，目标结果保留；转换PCM一致，立体声去和声relative L2 **5.6661e-08**、max_abs
   **1.19209e-07**（同门槛）。180.57秒是多次模型散列及完整对照/失败用例总和，不能当单次推理耗时。
7. 最后`cmake --build /tmp/resonantia-web-release-host -j 2`及
   `ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure`：**16/16通过，0失败/skip，1.36秒**。
   日志`checkpoint-host-final-build.log`/`checkpoint-host-final-test.log`，覆盖新增写入失败和初始化中断。
8. `cmake --build /tmp/resonantia-performance-20261024.uemhTG -j 2`完整构建通过，无捕获编译警告/错误
   （`checkpoint-full-build.log`）；`ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L contract --output-on-failure`
   **41/41通过，0失败/skip，32.47秒**（`checkpoint-contract-test.log`）。另运行唯一不在contract标签内
   的Web codec：`ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_web_audio_codecs$' --output-on-failure`
   **1/1通过2.94秒**（`checkpoint-codecs-test.log`）。当前注册73项，**未宣称本轮全73项通过**。

没有改变正式Tom模型、运行长训练或整曲转换，没有部署服务/UI。下一步直接接Web控制、阶段完成
校验、训练恢复和隔离页面，不重复已通过的音质实验；最终仍需真实Web/浏览器恢复与完整发布门禁。

最终文档检查：`bash scripts/check_sdd.sh`32项通过，`git diff --check`通过。生产仍PID2034441/active，
没有模型子任务；正式G SHA-256仍为`50c48edca6b271cb0741c55658bd268680bb06d73acd0a4807b0154d3e50f44b`。

## Web控制、阶段复用及完整训练持久编排增量

此节更新上面的历史“下一步”范围，仍未部署，不代表整项FR-014完成。

实现：

- TaskRecord format2保留attempt，读取format1默认为1。pause/resume绑定ID与attempt，重复继续
  返回同一代次，旧暂停不能影响新执行。控制写入失败或子进程普通错误不能变成paused。
- TaskArtifacts绑定工作流、阶段名和输出摘要，fsync完成后原子提交标记。混音失败重试跳过已验证
  分离/清理/转换；任意阶段输出或依赖改变拒绝复用。音频输入/G/索引属于任务固定副本。
- joint-full固定素材和G/D副本；full_training关联原task，切片集合、G、Index分别保存/核验。
  清理CLI保留Roformer块；训练恢复完整G/D/optimizer/RNG/checkpoint。索引失败不重复训练，候选
  保存使用副本，不发布正式声线。旧profile维持原取消语义和明确的恢复限制。
- 所有Web GPU入口持有可被原生子进程继承的flock，runner父进程死亡后不允许另一GPU执行。
  新服务对interrupted任务的原控制文件请求安全停止，仍显示interrupted；通道缺失不新建、损坏
  不重写。仅从已提交检查点继续，异常退出可能损失未提交更新；不是强杀或自动恢复训练。

红/绿证据均在`runs/Web_Task_Lifecycle_20261029/evidence/`：

| 日志前缀 | 事实 |
|---|---|
| `control-attempt-red-*` | 实际失败：执行代次未保存 |
| `control-artifacts-red-build` / `control-storage-green-*` | 阶段存储实现前链接失败；store/artifact 2/2通过0.04秒 |
| `control-api-red-*` / `control-api-green-recheck-*` | 旧API缺代次/控制参数失败；1/1通过0.65秒；中间构建缺field第三参数已修正 |
| `control-stage-binding-red-test` / `control-failures-*` | 改名标记可误复用的失败已修复；2/2通过0.94秒，含混音复用、错模型身份、失败优先及记录写入失败 |
| `control-orphan-red-*` / `control-orphan-green-*` | runner死亡错误释放GPU所有权的真实子进程反例；修复后进程/tasks 2/2通过1.03秒 |
| `training-control-red-*` / `training-control-green-*` | 训练未使用提交稳定ID的实际失败；完整训练暂停/重启/继续及索引失败复用1/1通过1.21秒 |
| `training-control-boundaries-*` | 补验前端身份变化拒绝GPU执行、原材料被替换后仍用固定切片、预处理暂停/重开；1/1通过2.17秒 |
| `training-orphan-red-recheck-*` / `training-orphan-green-*` | 旧训练进程通道未被新服务通知的失败；修复后1/1通过2.13秒。首个red fixture ID过短，已纠正并保留日志 |

构建/契约命令（下列增量与随后startup收尾增量分开记录，不声称重跑全部模型测试）：

```bash
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j 3
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L contract --output-on-failure
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_web_audio_codecs$' --output-on-failure
cmake --build /tmp/resonantia-web-release-host -j 3
ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure
```

`control-complete-*`：完整构建通过，contract **42/42、32.66秒**，host **17/17、2.12秒**。
训练编排增量`training-control-full-build.log`完整构建通过，`training-control-contract-test.log`
**42/42、32.82秒**，codecs **1/1、2.96秒**，host-final **17/17、2.95秒**，均0失败/skip。
这些广泛门禁先于startup旧通道收尾补充，后续最终复查另记。

真实HTTP资产门禁（Node22.23.4、Orin；仅隔离测试服务）：

```bash
/tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_audio_resume_real.cjs /home/rm01/Resonantia /tmp/resonantia-performance-20261024.uemhTG /home/rm01/Resonantia/runs/Web_Task_Lifecycle_20261029/real-web-control-2
/tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_training_resume_real.cjs /home/rm01/Resonantia /tmp/resonantia-performance-20261024.uemhTG /home/rm01/Resonantia/runs/Web_Task_Lifecycle_20261029/real-web-training-2
```

- `control-real-web-test.log`和加入资源锁后的`control-real-web-lease-test.log`均通过：speech_02
  前12秒+正式Tom G，真实HTTP提交，保存首个F0块后暂停、重启服务、从attempt2继续；旧块摘要
  不变，下载960044字节WAV与连续native版本逐字节一致。并验证重复resume不重开、旧pause409。
- `training-control-real-recheck-test.log`通过：speech_librispeech前1.28秒PCM16，joint-full
  epochs3/单切片共3步，在step2保存后暂停、真实重启服务、继续step2，仅执行剩余step3；缓存
  `cache_hits=1/1`，完整checkpoint含G/D/state，候选G/Index留存，原素材及正式Tom SHA不变。
  本门禁验证真实编排与计数；训练数值对齐仍由既有`test_joint_full_cli`门禁证明。
- 首个训练脚本用FFmpeg pcm_f32le生成WAVE_FORMAT_EXTENSIBLE，native输入检查拒绝，详见
  `training-control-real-test.log`。fixture改为Web实际接受的PCM16后重跑通过，没有放宽输入校验。

剩余：任务容量/磁盘保留与清理、统一任务页面（含鉴权媒体、关闭/重开、断网恢复）、手动工作流
的页面接入与兼容、最终发布/内网验收。后台普通重启门禁不替代异常退出门禁，后者续记。生产UI
仍直接读取workspace webui，开发页面必须在私有副本暂存，不能提前改线上页面。

继续实施时需先收紧的已知边界：CLI block root已建但owner尚未建立的初始化中断、joint-full identity已写但
首个完整checkpoint尚未提交的中断仍需独立故障恢复测试。不能把当前成功的安全点测试泛化成任意
初始化时刻均可恢复。UI私有副本位于`runs/Web_Task_Lifecycle_20261029/webui-staged/`，当前仅复制
旧页面，尚未实现任务中心。

### 异常退出验证与收尾

`tests/test_web_training_resume_real.cjs ... <new-root> --crash`在真实训练报告第1步后只终止隔离
服务父进程，重新启动同目录测试服务；新服务保留interrupted、请求旧控制通道停止，等待已验证
归属/启动时间的旧训练进程退出，然后显式继续。生产代码不扫描或强杀PID，进程发现只在测试脚本。

实际通过命令：

```bash
/tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_training_resume_real.cjs /home/rm01/Resonantia /tmp/resonantia-performance-20261024.uemhTG /home/rm01/Resonantia/runs/Web_Task_Lifecycle_20261029/real-web-training-crash-3 --crash
```

`training-orphan-real-final-test.log`通过：最后观察step1，恢复点step0，重启从step0继续至step3，
cache1/1，attempt2，旧进程退出，候选G/Index完成，正式Tom SHA不变。本次意外退出没有保留
尚未提交的更新，**不是零损失恢复**。没有改动训练数值或optimizer顺序。

前两次crash测试脚本失败均在模拟崩溃之前：本机内核不提供`task/<tid>/children`，随后进程
识别误用真实build路径而非服务argv中的隔离build符号路径。保留`training-orphan-real-test.log`
及`training-orphan-real-recheck-test.log`；每次失败都通过正常暂停收尾，未留下训练进程。
脚本改为校验PPID、可执行名、唯一fixture路径和启动时间后第三次通过；没有修改产品门槛。

startup收尾增量最后完整构建通过（`training-orphan-full-build.log`），同前述构建/CTest命令：
contract **42/42、34.06秒**，codec **1/1、3.01秒**，host **17/17、2.61秒**，0失败/skip。
`training-orphan-sdd.log`32项通过。正式服务仍PID2034441/active，内网status200且idle，无遗留
测试模型进程。后续容量错误分类修复另记；不能把这些先前结果称作后续树重跑结果。

### 容量与迟到控制补充

- `task-capacity-red-test.log`在256条记录时复现新提交503；修复为`TaskCapacityError`，不触发
  全局storage_failed。转换/清理/step/训练准入返回409，查询保持200。首轮focused2/2通过4.64秒；
  补训练分支后`task-capacity-training-red-test.log`实际复现漏分类503，再修正该catch。
  `task-capacity-final-focused-test.log`2/2通过3.38秒、host17/17通过4.93秒。
- 当时完整contract为**41通过/1失败，37.19秒**，`test_web_contract`失败于两份128MiB未完成
  上传的256MiB预留断言，未得到预期503。没有修改实现、断言或2秒等待；相同二进制单独复查
  **1/1通过11.59秒**（`task-capacity-http-recheck.log`）。根因未确认，保留原失败，不以重跑掩盖。
- `training-stale-control-red-test.log`复现旧无attempt取消请求误停已经继续的训练。新代次现在
  拒绝该旧接口，使用带attempt的任务控制；训练提交ID限定8..64字符，与旧控制边界一致。
  `training-stale-control-green-test.log`1/1通过4.31秒；未降低任何数值或资源门槛。
- 最终顺序验证日志使用`training-task-final-*`前缀，先完整构建和host构建，再依次运行host、
  full contract、codec及SDD，避免资源门禁与另一套测试同时占用机器。完整构建通过，host
  **17/17、3.93秒**，contract **42/42、33.77秒**，codec **1/1、2.96秒**，均0失败/skip；
  SDD32项和`git diff --check`通过。首次上传资源门禁失败仍保留，根因未确认。

真实模型恢复门禁在最后容量/旧取消隔离补充之前完成；这两个补充使用host/HTTP契约验证，没有
再跑一轮模型或长训练。当前注册74项，并非全部74项均在本增量重跑。源码与私有页面副本保留，
生产仍是旧服务和旧页面，下一步继续初始化中断、磁盘保留/清理及任务页面接入。

## 初始化、任务中心与磁盘增量（当时未部署）

当前源码与隔离页面已补齐以下增量；生产仍为PID2034441，不将隔离验收写成已部署。

- 操作checkpoint在刚创建目录就中断时可安全补建；已有块却缺失owner/identity仍拒绝。
  `init-session-red-test.log`复现缺owner失败，`init-training-red-build.log`记录新恢复辅助函数尚未
  实现的链接失败。host初始化门禁通过；完整构建后的`init-cli-green-test.log`为2/2、7.17秒。
- 新`vc_train --resume-initial`只接受joint-full且无完整checkpoint，身份核验后清理已知state0
  临时写入。`test_initial_training_resume_real.cjs`复用真实clip/缓存，恢复后1步安全停止、退出2、
  cache_hits=1/1、未导出推理G（`init-training-real-test.log`）。
- 私有`webui-staged/`新增任务中心、稳定提交ID、attempt控制、分页、真实阶段/计数/日志、
  历史音频Range试听下载、后台继续提示、自动重连和quiet鉴权。浏览器storage仅保存选中/待确认ID，
  服务记录是事实来源。旧同步转换入口仍兼容，新页面使用后台提交。
- `task-ui-red.log`缺任务页失败后，`task-ui-green-2/3.log`通过关闭期间完成、离线重连、提交响应
  丢失、训练关闭期间失败、令牌重连、恶意文件名/日志纯文本、4档布局和axe规则。首个green脚本
  未考虑Chromium连接丢失后的原ID传输重试，替身补齐真实服务已有的重复ID409行为；只接纳一份任务。
- `task-auth-race-red.log`复现迟到401清除刚更新的令牌；修复后green通过，后台轮询不反复弹认证框。
- `/api/tasks`增加分页外active_id；训练历史回到原task_id，host API red/green均留存。
  原声/结果同位置对照保留；浏览器替身补齐真实Range206后`studio-tasks-2.log`通过。
- `test_web_task_browser_real.cjs`使用真实vc_serve/Chromium、6秒speech_02和1.28秒训练clip。
  关闭整个浏览器context后转换和1步训练继续完成；新context恢复结果、Range/拖动试听、候选入口。
  首次训练界面高级设置未展开导致脚本失败；继续已有fixture而没有重跑转换。
  第二次暴露真实缺陷：不足周期保存间隔时最终G未进入候选列表。新增Web最终轮候选原子副本与
  SHA校验，`final-candidate-red-test.log`失败后green为1/1、3.35秒。第三次脚本误选历史中最新的
  训练任务而非音频任务，未启动新模型；修正明确ID后`task-browser-real-4.log`最终通过。
  整个系列仅做一次6秒转换、两次各1步训练；正式模型SHA保持
  `50c48edca6b271cb0741c55658bd268680bb06d73acd0a4807b0154d3e50f44b`。
- `task_storage.*`新增有界磁盘统计/冗余副本清理；storage-red链接失败后green1/1通过。
  retention-red复现清理后仍占满准入槽，green保留历史ID且允许新任务。storage-api增量3/3通过，
  storage-boundary为1/1、3.04秒：超限安全暂停、拒绝新任务/继续507、历史仍200、释放空间后
  显式继续且不重复已完成转换。临时发布并发测试首个单文件版本未复现，多文件版本red2复现
  ENOENT被误判故障；修复后green1/1、0.11秒。没有删除唯一检查点、原素材或正式声线。
- `task-ui-storage-red.log`缺磁盘显示失败，green验证清理后结果保留及原任务ID/attempt；新增独立
  清理试听入口、完整训练自动清理选项。自动清理UI红记录与后续green分开保留。

核心命令（绝对build路径与本地证据根同上）：

```sh
cmake --build /tmp/resonantia-web-release-host --target test_checkpoint_initialization test_web_tasks test_web_task_store test_task_storage -j2
ctest --test-dir /tmp/resonantia-web-release-host -R '^(test_checkpoint_initialization|test_web_tasks|test_web_task_store|test_task_storage)$' --output-on-failure
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j2
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell /tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_task_center_browser.cjs runs/Web_Task_Lifecycle_20261029/webui-staged runs/Web_Task_Lifecycle_20261029/evidence/task-ui-storage-green
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell /tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_task_browser_real.cjs /home/rm01/Resonantia /tmp/resonantia-performance-20261024.uemhTG /home/rm01/Resonantia/runs/Web_Task_Lifecycle_20261029/webui-staged /home/rm01/Resonantia/runs/Web_Task_Lifecycle_20261029/real-browser-1 --continue
```

单项日志记录实际执行的focused组合；上列合并focused命令是最终重验入口，不能冒称该组合已跑。
磁盘保护接入后尚需最终真实短任务/页面回归、完整host/contract及生产切换；目标保持未完成。

## 最终门禁与生产切换（2026-10-29 09:27:37 CST）

上述各批记录保留当时的失败/未完成状态；本节与顶部清单是本轮最终状态，不把旧测试算成新树重跑。

### 收尾发现与修复

- 第一次完整回归3个UI测试误读仍在线的旧页面，41通过/3失败；新增
  `RESONANTIA_WEB_TEST_UI_DIR`显式指定隔离待发布页面，UI4/4和contract44/44通过。
- 最终真实浏览器首轮在提交前发现待确认配方字段正则不接受`f0`，尚未开始GPU工作。
  `task-recipe-field-red.log`复现，允许数字后的`task-ui-final2.log`通过（包括响应丢失、接收前503、
  重开后原ID重试）。`task-browser-real-final-2.log`随后通过真实6秒转换和1步训练，关闭整个
  context期间完成，新context恢复结果、Range拖动和最终候选。两次短任务未自动发布模型。
- 清理最后补上结果完整性门槛：`cleanup-integrity-red-test.log`复现结果损坏后仍允许删除副本；
  新成功音频保存SHA256，清理前匹配，缺摘要旧记录不开放清理但保留查询/下载。
  `task-artifact-real-final.log`用已有隔离fixture追加6秒CPU预处理：保存摘要正确，重复清理幂等，
  原输入与结果保持，重复提交409，没有运行模型。该验证在最终源码重新构建后完成。

### 最终测试记录

| 记录 | 结果及范围 |
|---|---|
| `cleanup-integrity-full-build.log`、`cleanup-integrity-host-build.log` | 完整目标及host目标构建通过 |
| `cleanup-integrity-host-tests.log` | 19/19，4.41秒，0失败/skip |
| `cleanup-integrity-contract-tests.log` | 44/44，35.09秒，0失败/skip |
| `task-release-codec-final.log` | 1/1，2.88秒，真实codec；后续只改结果摘要/清理保护，未重跑codec |
| `task-ui-final2.log`、`studio-final.log` | Chromium任务中心与工作室通过；API/模型操作替身，不是音质验收 |
| `task-browser-real-final-2.log` | 真实原生后台和GPU，6秒转换+1步训练，关闭/重开通过；正式模型SHA不变 |
| `task-artifact-real-final.log` | 最后摘要保护的6秒CPU真实产物/清理校验通过 |
| `deployed-ui-contract.log` | 配置回实际webui后4/4，1.09秒，0失败/skip |
| `production-browser.log` | 内网真实上传/页面、候选历史及布局通过；转换被拦截，不运行模型或发布 |
| `production-task-smoke.log`、`.json` | 内网原生API，无替身；6秒CPU干声任务，关闭/新建context后结果恢复，Range206 |

当前完整构建注册76项；本增量没有重跑全部76项，也没有用跳过或替身结果宣称真实音质通过。
此前一次上传并发预算门禁失败的根因仍未确认，原记录保留；最终同门槛串行contract通过。

实际最后执行命令（日志目录为`runs/Web_Task_Lifecycle_20261029/evidence/`）：

```sh
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j2
cmake --build /tmp/resonantia-web-release-host -j2
ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure -j1
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L contract --no-tests=error --output-on-failure -j1
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_web_audio_codecs$' --output-on-failure
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell /tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_task_center_browser.cjs runs/Web_Task_Lifecycle_20261029/webui-staged runs/Web_Task_Lifecycle_20261029/evidence/task-ui-final2
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell /tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/run_web_studio_browser.cjs runs/Web_Task_Lifecycle_20261029/evidence/studio-final runs/Web_Task_Lifecycle_20261029/webui-staged
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell /tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_task_browser_real.cjs /home/rm01/Resonantia /tmp/resonantia-performance-20261024.uemhTG /home/rm01/Resonantia/runs/Web_Task_Lifecycle_20261029/webui-staged /home/rm01/Resonantia/runs/Web_Task_Lifecycle_20261029/real-browser-final-2
/tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_task_artifact_real.cjs /home/rm01/Resonantia /tmp/resonantia-performance-20261024.uemhTG /home/rm01/Resonantia/runs/Web_Task_Lifecycle_20261029/real-browser-final-2
cmake -S . -B /tmp/resonantia-performance-20261024.uemhTG -DRESONANTIA_WEB_TEST_UI_DIR=/home/rm01/Resonantia/webui
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R 'test_web_.*ui' --no-tests=error --output-on-failure -j1
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules /tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_workflows_browser.cjs http://192.168.0.122:8080 /home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell /home/rm01/Resonantia/runs/Web_Task_Lifecycle_20261029/real-browser-final-2/speech6.wav /home/rm01/Resonantia/runs/Web_Task_Lifecycle_20261029/evidence/production-browser
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules /tmp/resonantia-web-release-browser/node_modules/node/bin/node runs/Web_Task_Lifecycle_20261029/evidence/production-task-smoke.cjs
```

### 部署、数据与回退

部署前确认旧PID2034441无活动任务和计算子进程。旧二进制、页面及用户服务配置备份在
`runs/Web_Task_Lifecycle_20261029/deployment-backup/`，带SHA清单。停止用户服务后替换五个
已构建CLI（vc_serve/vc_train/vc_convert/vc_preprocess/build_index）及页面，重新启动：
用户服务`resonantia-web.service`为active、PID2258518，启动时间09:27:37 CST。
内网`http://192.168.0.122:8080/`、voices/tasks/storage接口均正常。
`deployment-manifest.json`核验全部部署文件与测试版本SHA一致，正式Tom仍为既有SHA，且唯一发布声线。

线上历史中的“部署验证_6秒干声.wav”为CPU-only文件处理验收（原生HTTP未拦截），不代表清理模型
或音色测试；关闭页面后重新打开可找回6秒音频并下载。所有模型/训练生命周期验收位于隔离目录。
本轮未跑新的整曲转换、长训练或候选发布；此前失败整曲的已删除临时输出没有恢复。

如需回退，先确认没有活动任务，再停止用户服务、从备份恢复上述五个二进制及原页面，然后启动。
不要删除`runs/.web_tasks`、素材、检查点或正式模型；旧页面不读取任务中心，但新版恢复后仍可查询
这些记录。旧版本不承诺操作新任务恢复格式，应保留新版本以继续已保存任务。ADR-0010仍为Proposed
正式审查状态，实施授权沿用维护者明确指令。

最终文档门禁：`bash scripts/check_sdd.sh`通过32项必需文件/链接检查，`git diff --check`无错误，
记录`final-deployed-sdd.log`与`final-deployed-diff-check.log`。本轮未提交或推送Git，既有用户修改和资产保留。
