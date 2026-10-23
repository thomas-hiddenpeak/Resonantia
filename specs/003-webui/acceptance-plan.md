# Web P1 剩余项：最小设计与验收规划

**2026-10-10 · D1/D2/P1/P2指定验收已完成；后续训练取消已获用户授权，范围及结果见§4。** 对应
[PROJECT_STATE P1](../PROJECT_STATE.md#下一步)、[Spec 003](spec.md) FR-005/006/008/010 与 §5。
原硬化生产代码基线为固定审查源码 **v5**（ZIP SHA-256
`3c31dc07847785f51f42041b23c67f79b93ad54b11d008eb9cf2993c05ae5c5f`）。
D2测试从固定源码v6（ZIP SHA-256 `c483c89d8c4527106b6c74ba33218704882fd3ca0b63fd7a387caf02cd86482b`）追加。
原规划轮仅新增本文和状态导航；随后按用户授权补充D1/P1与独立D2测试及CMake注册，生产源码与deadline保持v5不变。
规划中的新门禁均串行、仅自有临时进程/loopback；合计RSS观察预算256MiB、无模型GPU作业，
数据与时间预算见各表。预算是实验约束，未实施系统资源限制；P1已测父子peak RSS上界6912 KiB，D1未单独采样RSS。

## 1. 已验证范围与剩余项性质

取消增量前设备基线：默认 **19/19，21.78 秒**；SDD 32 项。D1所在Web门禁10.94秒；P1门禁0.04秒。
上传解析、事务/备份、预览一致性、启动失败的17点门禁已记录在原计划。
新增请求门禁验证 128 MiB+1 声明提前413、256 MiB总预留耗尽/EOF与RST后恢复、
accepted socket关闭、33次串行EOF名额复用；最多3客户端，实际body最大215 bytes。
另已通过真实10秒header超时；仍不证明完整大上传、同时32连接拒绝或整体RSS上界。

| 剩余项 | 当前实现事实 | 下一步性质 |
|---|---|---|
| 慢客户端读取 | header 10秒、body 120秒绝对deadline；每次recv使用剩余时间，碎片到达不续期；失败走400及RAII释放 | **D1/D2已补**；更改时长、改408、滑动idle timeout或允许无限滴流均是行为变更 |
| 进程失败 | `ProcessResult`区分exit/signal/launch/exec/wait与log I/O；非零CLI已有HTTP门禁 | **P1自然故障与P2指定消费矩阵已补**；真实wait/I/O注入另行限定，不能以系统资源耗尽造故障 |
| 作业timeout/cancel | 已有训练job_id、取消端点与阶段检查点；CLI没有循环内stop接口 | **训练取消已授权实现**，见§4；timeout、强杀、更多任务范围未实施 |

v4部分审查确认旧代码重复finish能清除后来作业的预留；v5已缩小catch范围、现有门禁通过。
v5只读复核已核验源码与18项manifest，支持该窄修复：失败release/响应在commit catch外，
外层HTTP catch只回错误、不改作业状态；成功启动后的响应也在startup catch外。
GPU执行仍受mutex串行保护。**精确A/B/C修复后动态复验仍未完成**；静态结论与已有日志不能
补成动态通过。不重跑被中断/未核验探针，也不以新包装绕过原检查。

## 2. 慢客户端：优先黑盒验收，保持现有行为

当前读取deadline始于服务worker进入对应读取阶段；body deadline在header校验和预留后建立。
超时/截断映射400；`Request`释放body预留，`ConnectionGuard`关闭连接并归还名额。
请求体未完整到达时没有作业。15秒socket发送timeout是单次阻塞发送限制，**不是完整响应的
绝对总时限**；慢响应总时限/streaming另属设计，不混入本轮读取验收。

| 候选门禁 | 最小输入与断言 | 预算/证据限制 |
|---|---|---|
| D1：生产10秒header | 独立loopback服务；未完成header在第0/3/6秒各发送少量字节；约10秒后400并关闭，不能延到最后一字节后再10秒；随后正常请求可用 | 1个慢客户端+1个顺序探测；输入≤4KiB；用例硬截止15秒，计时容差明确记录，不当产品SLO |
| D2：生产120秒body | 完整header声明小body，持续少量滴流且始终不足声明长度；约120秒后400；预留/accepted socket释放、job与素材不变，后续请求可用 | 独立可选门禁，≤2客户端、body≤4KiB、硬截止130秒；不可塞入当前60秒Web测试；低CPU但需约2分钟墙钟时间 |
| D3：快速逻辑测试（可选） | 若后续确需快门禁，单独评审私有测试时钟/短预算依赖，验证碎片不重置deadline与清理路径 | 不增对外运行开关；短预算只证明逻辑，不能冒充真实10/120秒验证；当前不引入此改动 |

D1/D2均已完成；D2独立记录如下。失败必须区分服务端deadline、
测试端提前断开与测试自身超时，不能把客户端主动关闭当作服务超时证据。

独立D2门禁已在OrinNX通过：第0秒完整header和部分body，第20/40/60/80/100秒各补1字节；
**120708 ms**后服务端400与EOF，整个CTest用例120.74秒（总墙钟120.75秒），小于130秒。
声明body为**1024 bytes**，实际body为**216 bytes**；包含就绪/状态查询的全部请求共**570 bytes**，
不应把声明量当作实际传输。最多2客户端，未提前完成body、半关闭或RST；accepted socket关闭，
后续状态查询200、job/raw/work/steps/CLI记录不变、无staging，自有监听器已回收。
中途父子进程RSS采样7720 KiB（单点采样，不是峰值上界）。本轮对原生产实现直接通过，未改生产逻辑。

长用例用默认OFF的`VOXMUTATIO_ENABLE_SLOW_WEB_TESTS`单独注册`test_web_body_deadline`，
复用测试程序的`--body-deadline-only`模式；label为contract/web/host/slow，CTest TIMEOUT=130。
不放入原60秒Web门禁；该D2验收后已恢复OFF，当时默认18项、快速回归18/18、0失败/skip、21.71秒。
完整资产配置42项的口径同样以slow选项OFF为准，ON额外注册1项（未据此重跑模型套件）。

复现独立D2（现有build，模型测试保持OFF）：
```bash
cmake -S . -B <build-dir> -DVOXMUTATIO_ENABLE_SLOW_WEB_TESTS=ON
nice -n 10 cmake --build <build-dir> --target test_web_contract --parallel 1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ctest --test-dir <build-dir> \
  --verbose --tests-regex '^test_web_body_deadline$' --parallel 1
cmake -S . -B <build-dir> -DVOXMUTATIO_ENABLE_SLOW_WEB_TESTS=OFF
```
本门禁没有直接测量body预留计数/总容量恢复；只证明该超时路径的socket释放与后续请求可用，
不能扩大为完整资源证明。此前EOF/RST的容量门禁仍保留其独立范围。

## 3. 进程故障：先自然失败和消费端契约

[PosixProcessRunner](../../tools/web/workflow_support.cpp)以argv启动直接子进程，合并/过滤输出，
最后waitpid；`success()`要求正常退出0且无I/O错误。保留三层证据，不能混称“进程故障全覆盖”。

| 层/候选门禁 | 最小方案与期望 | 范围与预算 |
|---|---|---|
| P1：真实自然失败 | 空argv→LaunchFailed/EINVAL；临时目录内不存在的可执行文件→ExecFailed/ENOENT；log路径指向自建目录→LaunchFailed；自有假子进程自行raise(SIGTERM)→Signaled/SIGTERM | 直接调用现有runner，无模型；最多1个受测试管理子进程；每例≤2秒、整组≤10秒，输出≤8KiB、临时数据≤1MiB；确认FD/子进程回收与后续正常调用 |
| P2：Web结果传播 | 用已有`ProcessRunner`接口的测试实现返回WaitFailed或`io_error=EIO`，验证同步调用报错、后台job进入error、禁止发布且能重试 | 不创建OS故障、不启动真实CLI；最多1个假作业，整组≤5秒；**只证明结果消费，不证明真实wait/I/O路径** |
| P3：底层异常组合 | 真实wait/read/write/close失败、EINTR恢复与异常回收先列条件和精确预期，再设计独立受限测试 | 仍缺证据；不抢其他线程的waitpid、不改服务全局SIGCHLD、不填满磁盘/内存/进程表，不执行未核验的所有权探针 |

新增门禁保持生产行为：`test_web_contract` 在未完成请求头的第0/3/6秒滴流，
慢header共53 bytes、最多2客户端；10050 ms后400与EOF，socket关闭，job/CLI不变、
后续状态请求成功。观测窗9–13秒，整个用例<15秒；后续D2结果见第2节。
`test_web_process_contract` 直接调用原runner，验证空argv/EINVAL、缺失程序/ENOENT、
日志为自建目录/EISDIR、自有短命子进程自行SIGTERM的分类与输出回收；4/4失败分类和
4/4后续正常调用通过。整组34 ms、输出178 bytes、父子peak RSS上界6912 KiB，FD不增，
无存活或僵尸直接子进程、临时目录清除。最多1个helper，每次自然失败<2秒，CTest上限10秒。
初版测试因设备内核不提供`/proc/.../children`在用例前失败，改用只读proc父PID元数据后通过；
这是测试兼容性修正，不是产品红绿缺陷修复。不抢waitpid所有权、不引入interposition。
这些证据不覆盖Web对信号/wait/log I/O结果的消费、底层wait/I/O故障、后代进程组或取消。

当前日志按单行64KiB缓冲，但文件总量无轮转上限；这是另一个既有设计项。当前先读管道到EOF，
假如后代仍持有写端，直接子进程退出也未必结束读取；此为实现结构所示风险，尚未补动态验收。

### P2指定矩阵验收（已完成）

P2消费端门禁`test_web_result_contract`已通过：通过现有`ProcessRunner`依赖接口模拟
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

沿用现有`WebService(Paths, shared_ptr<const ProcessRunner>)`测试替身，不创建真实子进程。
代表结果为`kSignaled/SIGTERM`、`kWaitFailed/ECHILD`和`kExited/exit_code=0/io_error=EIO`；
最后一项须防止“正常退出但日志I/O失败”被误当成功。

- 同步preprocess/convert：替身即使留下有效输出也必须500，不能返回音频成功；convert预处理
  失败须短路，主convert失败亦覆盖；后续正常调用恢复，临时目录最终清理。
- 后台step/train：接受后最终running=false/done=true/error=true，旧work/steps及已发布voice
  保留；训练preprocess/train/index各阶段失败均禁止发布、不执行下游阶段，最终staging清理。
- 每次失败后正常重试通过；最多1个假worker、串行、整组目标≤5秒。通过响应/状态、调用记录
  和临时文件断言；不把finish瞬间当作清理已完成。

仅证明结果消费，不证明真实wait/I/O故障或进程组所有权。不复用分配故障入口、不抢waitpid、
该P2不使用interposition或实现cancel/强制终止；随后授权的训练取消见§4，精确A/B/C动态复验仍独立标记未完成。

## 4. 已授权训练取消：阶段边界、所有权与验证

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
[ADR-0006](../../docs/adr/0006-training-cancellation-ownership.md)。

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


## 5. 后续范围

本次后续明确授权了CLI完整迭代协作停止；当前只有编排层host证据，真实GPU取消尚未验收。
自动超时、checkpoint/resume和其他作业范围均不是已完成能力，不据本增量自动扩大。浏览器断线不自动取消；HTTP上传未完成仍只释放请求资源。
任何后续终止策略必须保持作业所有权、当前产物与用户素材语义，不猜测训练时限。
旧D1/D2/P1/P2范围和精确A/B/C未完成事实继续保留，不重跑已中止探针。

## 6. 本轮收口与用户决策

本轮既定轻量硬化范围收口

已闭合的是以下明确验收范围，不是整份Spec003或所有Web故障：
1. 上传解析：multipart boundary、part header/Content-Disposition及16KiB field/128 parts边界、拒绝保留旧素材。
2. 素材/步骤/训练事务与预览：常规回滚、异常时旧副本保留、发布/preview一致性及慢preview不阻塞发布。
   非常规失败可能保留backup待人工恢复，不承诺外部写者或进程崩溃时多次rename原子性。
3. 作业启动：step8/train9共17个已观测初始化分配点的预留释放、重试与活动worker不提前释放。
4. 请求资源：128MiB+1声明拒绝、256MiB预留耗尽及EOF/RST恢复、socket释放及33次串行连接复用。
5. 绝对读取期限：D1真实10秒header、D2独立120秒body滴流不续期，超时关闭与后续请求可用。
   D2没有直接证明超时后的body预留计数归零；已有EOF/RST容量证据不能代替这一点。
6. 进程与消费端：P1自然launch/exec/signal分类及回收；P2三类模拟结果×七个路径，21失败/21重试、
   同步500、后台error终态、旧产物保留、停止下游和最终清理。

仍保留的已有缺口（不伪装为完成，也不自动追加实验）：
- 上传提交失败重复finish的原受限A/B/C探针未重跑；后续§8已补齐普通host等价所有权门禁，
  不将其称为原OS故障探针或独立审查通过。
- 预留前filesystem分配异常触发标准库terminate的既有观察未单独调查。
- P2不证明OS层wait/I/O故障生成与恢复；完整大上传、极端并发、更多异常组合、真实模型/质量等
  仍是未验证范围。它们不改变本轮已批准D1/D2/P1/P2验收完成的事实。

2026-10-10 12:41:53 UTC用户已选择先做训练取消，§4替代原候选中的待选择状态；无须重复审批。
本增量完成代码、接口、UI及小型替身验证并保存快照。后续循环内stop等能力需单独明确需求，
不把本次授权扩展成强杀、自动超时、模型实验或部署。仍保留上述证据缺口。
家族史优先事项与本项目独立；Thor/DeusRidet维持暂停。

## 7. Later authorized complete-iteration cancellation

Current protocol, cleanup ordering and limits: [DESIGN](../../docs/DESIGN.md).
Baseline v9 and the independently checked material-commit increment stay immutable.
The first red host test required real vc_train to exit 2 before CUDA when pre-cancelled;
the old CLI rejected the unknown option. New host tests cover token parsing/ownership,
complete fake steps and error precedence; a natural host helper exercises actual runner
reap/log drain. Web doubles cover delivery, pending reservation, exit2/error classification,
stale-id/token isolation and retry; existing material-commit tests remain.
No real model training or GPU acceptance is implied. Run bounded host tests serially;
exclude syscall-interposition, signal-using server/process tests and GPU labels in this
increment. Keep explicit selected/passed/excluded counts with the delivery evidence.

## 8. A/B/C所有权与重复finish：后续普通host验证

基线为`main@0955500`。既有`test_web_job_lifecycle`覆盖初始化分配点和活动worker响应异常，
此前没有“worker已退休但旧接受响应仍等待，新B已运行”的确定性门禁，也没有旧generation
完成回调重复到达时B状态不变的直接断言。本增量只补这些证据，不改变公共API或生产行为。

`test_web_job_ownership`以测试目标私有宏、两个显式响应接缝和ProcessRunner替身覆盖4/4场景：

| A的路径 | 正常响应 | 普通C++响应异常 |
|---|---|---|
| 上传因目录型marker自然校验拒绝 | 响应/staging收尾前B/C均409，收尾后B可接受 | 异常展开释放A后B可接受，无A worker |
| A worker先于延迟接受响应退休 | B可接受，A响应仍返回A id | A异常展开不清除B |

每种场景均在B活动期间重复两次重放A generation的training及step finish，检查B完整状态
不变、C仍409；B自然结束后C可重试。旧raw/work与已发布model/index不变，无staging/FD泄漏。
每个屏障最长2秒、请求≤4KiB/响应≤8KiB、CTest上限15秒；首个preprocess替身返回错误，
不启动子进程、模型、GPU或训练，不使用rename/pthread拦截或进程注入。

无响应接缝时测试按预期因屏障未观测而失败；接缝就绪后4/4通过，0.03秒。
另在自有临时源码中分别移除training和step的generation保护，两个负对照均因旧finish修改B
而失败，仓库源码保持不变。宏关闭的生产对象/库/`vc_serve`哈希保持一致且没有回调符号。
这是等价所有权语义的普通host验证，不证明原OS故障生成，也不构成新的独立审查。

同一OrinNX独立build串行、低优先级运行以下10项，**10/10通过、0失败、0skip，3.43秒**：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ctest \
  --test-dir /tmp/resonantia-web-20261010.ycTBxI --output-on-failure --parallel 1 \
  -R '^(test_bounded_audio_io|test_cli_contract|test_web_job_lifecycle|test_web_job_ownership|test_web_result_contract|test_web_train_cancel|test_web_train_cancel_commit|test_training_cancel|test_vc_train_entry|test_legacy_webserver_contract)$'
```

24项已注册测试中其余14项未运行：10项GPU用例、`test_web_contract`、
`test_web_transaction_failure`、`test_web_process_contract`和`test_realtime_benchmark_report`。
未运行不计skip/通过，未扩大已有独立审查结论；原受限探针、真实GPU取消、最终发布窗口、
预留前filesystem terminate及更广OS/OOM边界仍未闭合。本增量不修改CI或其他测试，
保持本地提交，待专属OrinNX路由方案单独获批并核实后再考虑推送。

## 9. 托管host覆盖完整性与最终发布窗口

对`main@5f7fe83`的有界核验发现，此前OrinNX选择的10项和GitHub的6项并非全由平台决定。
六项一致的是`test_web_job_lifecycle`、`test_web_job_ownership`、`test_web_result_contract`、
`test_web_train_cancel`、`test_web_train_cancel_commit`和`test_vc_train_entry`。

| 本地额外测试 | 原托管CI缺少原因 | 本次处理与保留边界 |
|---|---|---|
| `test_bounded_audio_io` | 子集未接入Catch2；实现本身无需CUDA | 复用原测试和实现，加入host入口；共享现有固定Catch2依赖策略 |
| `test_legacy_webserver_contract` | 同上，原根构建粗粒度链接CUDA facade | 直接复用旧API实现与原测试，加入host入口，无socket服务启动 |
| `test_training_cancel` | 同一文件混合纯协议、参数解析和真实CUDA链接CLI的预取消入口 | 纯令牌/完整迭代/错误优先/路径复用/自然helper回收断言原样移至`test_training_cancel_protocol`，两种构建均注册；原名保留参数解析与真实CLI预取消，不用替身冒充真实CLI |
| `test_cli_contract` | 需要五个实际生产CLI；其应用实现链接CUDA | 保留完整构建入口。是构建/链接依赖，不代表这些参数检查必须执行GPU；不为此安装CUDA或替换真实CLI |

因此当前host入口为10项：原6项、上述两个直接复用测试、拆出的协议门禁，以及新增的
`test_web_train_cancel_publish`。数量与原10项恰好相同但集合不同；不得等同两套证据。
完整根CTest增加协议和发布两个注册，保留原CLI测试及其他用例；本轮未运行完整设备套件。

最终发布待项的依据：旧`race_case`允许取消先到或发布先到，两种结果都通过，不能保证
命中`publishing`。新私有目标复用同一`test_web_train_cancel.cpp`、fake runner和有界gate，
在三产物事务安装clips后、model/index安装前暂停。对照与取消两种case均确认新clips已安装、
旧三产物仍在backup、running为true、cancellable为false且新作业409；取消case连续两次
返回409且job状态不变。释放后必须完整发布三产物、done且非cancelled，源素材不变，
staging/backup与FD清理，槽位可重试、旧id不能匹配新作业。

回调只存在于`RESONANTIA_TEST_TRAINING_PUBLISH_GATE`私有编译，原生产宏关闭代码保持一致。
每个gate最长2秒，CTest上限15秒；普通fixture文件移动与fake worker，无系统调用拦截、
进程注入、模型或GPU。该门禁验证取消截止点和正常发布收尾，不新增崩溃原子性、OS故障、
rollback组合或真实GPU证据。运行结果以对应SHA的托管CI日志为准，设备job继续明确未运行。
