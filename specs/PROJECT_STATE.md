# Resonantia 项目状态

**状态日期：2026-10-24（八项审计修复；本轮验证记录见下文）**

本文件是当前实现与验证结果的**描述性快照**，不是替代 feature spec 的规范来源。
规格意图和需求归属见 [`specs/README.md`](README.md) 及各 feature spec；可执行边界以
`docs/DESIGN.md`、CLI `--help`、当前 CMake/CTest 注册和实现中的输入校验为准。
未来目标、历史研究结论和配置结构中的预留字段都不等于已支持能力。

## 验证基线

### 2026-10-24 八项审计修复

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

当前 Node 可用且 slow=OFF 时默认 asset-free 为 **27 项**，完整资产为 **51 distinct**
（28 contract-labelled、24 integration-labelled、1 overlap）；slow=ON 再加 1 项真实
120 秒 body deadline，总计 **52**。缺 Node 时两个 UI 测试明确不注册，不算通过；
因此默认/full 数量分别减 2。以下日期化记录保留各自历史数量及当时授权边界。

未扩大证据：训练音质/长期稳定性、resume、真实 GPU 取消和最终发布窗口、任意时长资源
保证、非 SM87 平台及实时产品仍未验收；Spec 006 目标声线 model/Index 资格资产仍缺失。

### 历史验证记录

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
   本次所有权增量保持本地提交；下一步先审定专属OrinNX CI路由和runner归属，再决定安全推送。
   不自动扩展到真实训练、部署或已中止探针，也不配置/注册runner。
4. **实时能力暂缓**——Spec 006/ADR-0002 保持 Approved / R0 evidence 部分完成 / 产品未实现；
   本轮不接 persistent runtime、session 或 ALSA。恢复时仍需有授权 corpus、匹配 target
   model+Index 及 Index-enabled 质量/SLO 证据。
5. **持续加固**——保持 contract 与 asset-gated integration 两层测试，继续覆盖输入
   上限、原子产物、回滚与服务暴露边界。

### 2026-10-10 后续授权：完整迭代协作取消

CLI可选作业文件令牌、退出2、完整GAN D/G与decoder迭代边界已接入Web；source/lease等待写操作结束、
runner自然reap和staging清理后才释放预留，实际错误优先。详见[DESIGN](../docs/DESIGN.md)。
保留固定v9和素材提交测试快照，另存本增量；仅host验证与编译，不推断GPU停止/音质或resume完成。
