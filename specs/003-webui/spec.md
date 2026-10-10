# Feature Specification: WebUI and HTTP Orchestration

**Feature**: `003-webui`
**Created from implementation**: 2026-08-09
**Updated**: 2026-10-10
**Status**: Implementation present; acceptance partially verified; deployment/training experimental

## 1. Scope

`vc_serve` 是一个纯 C++、HTTP/1.1 over POSIX sockets 的单机编排服务。它提供无构建
步骤的 HTML/CSS/JavaScript WebUI，并通过子进程调用现有 CLI：

```text
browser -> vc_serve -> vc_preprocess / vc_train / build_index / vc_convert
```

服务不是独立模型后端，不在进程内常驻加载模型，也不是实时音频服务器。训练端点只
暴露 Spec 002 的实验性 decoder/GAN 路径，不能把 Web 可点击解释为训练已产品化。

## 2. Startup Contract

```text
vc_serve [--host IPv4] [--port 8080] [--webroot PATH]
         [--repo PATH] [--auth-token TOKEN]
```

| Option | Default / behavior |
|---|---|
| `--host` | **`0.0.0.0`**，监听全部 IPv4 接口 |
| `--port` | `8080`，合法范围 1..65535 |
| `--webroot` | `webui`，启动时解析为存在的 canonical directory |
| `--repo` | 默认取 canonical webroot 的父目录 |
| `--auth-token` | 默认空，即 **API 不认证**；非空时要求 `Authorization: Bearer ...` |

默认配置是为了本机/受信网络便捷使用的“开放”模式。服务在
`0.0.0.0` 且 token 为空时打印警告，但仍会启动。公网或不受信网络部署必须显式增加
token，并在外层配置 TLS、网络 ACL/防火墙和可信反向代理。

Bearer token 只保护 `/api/` 路径；静态 Web 资源仍可公开读取。当前服务没有用户、
角色、session cookie 或租户隔离。

## 3. As-built API

| Method and path | Behavior |
|---|---|
| `GET /` and static paths | 从 canonical webroot 提供 WebUI 文件 |
| `GET /api/voices` | 枚举 base/已导出声线及当前训练状态 |
| `GET /api/sep-models` | 按任务返回已安装分离模型 |
| `POST /api/material` | 上传一组 WAV，并 staged 替换 raw/work material |
| `POST /api/step` | 对 work set 执行一次 separation/de-reverb/de-echo/de-noise/de-harmony |
| `GET /api/preview?name=...` | 返回 work set 的首个 WAV |
| `POST /api/preprocess` | 对单个 WAV 执行可选预处理并返回 WAV preview |
| `POST /api/train` | 启动单个 decoder 或 GAN 实验训练任务 |
| `GET /api/train/status` | 返回 job_id/kind/state、stage/running/done/error、cancel_requested/cancelled/cancellable/cancel_boundary 与受限 log tail |
| `POST /api/train/cancel` | 单个 job_id 定位训练作业；202接受请求、200重复请求，阶段/完整训练迭代边界取消，发布开始后409 |
| `POST /api/convert` | 对上传或 material session 的 WAV 运行离线转换，返回 WAV；正数 `index_rate` 要求所选声线存在 Index，零值显式 no-index |

未知 API 返回 404；已知 API 的方法不匹配返回 405 并带对应 `Allow`；非 API 静态路径
的非 GET 请求返回 405/`Allow: GET`。这些方法语义由 `test_web_contract` 真实 socket
门禁覆盖。上传和输出均为 WAV-only。

## 4. Functional Requirements

- **FR-001 Static UI**：无 Node/npm 构建依赖，静态 HTML/CSS/JS 可由 `vc_serve` 提供。
- **FR-002 Configurable bind**：允许配置 IPv4 host/port，默认 `0.0.0.0:8080`。
- **FR-003 Optional auth**：token 非空时，对所有 `/api/` 请求执行 constant-time Bearer
  token 比较；默认不启用认证。
- **FR-004 WAV-only upload**：文件名扩展名必须为 `.wav`，内容必须具有 RIFF/WAVE
  signature；重复或不安全文件名失败。
- **FR-005 CLI orchestration**：GPU 操作调用已构建 CLI，不在 HTTP worker 内嵌模型。
  子进程以参数向量直接启动，不拼接 shell 命令；日志过滤不得改变真实退出状态；监听与
  客户端 socket 必须 close-on-exec，不能被长寿命 CLI 继承。
- **FR-006 GPU serialization**：训练、step、预处理和转换共享单个 GPU mutex，避免
  同时执行 GPU 作业。预处理/转换和 `/api/step` 在抢锁失败时立即返回 409；step worker
  只有在完成 GPU reservation 后才向请求方报告接受。
- **FR-007 Transactional material**：上传先进入唯一 staged 目录，再 rename 提交；提交
  失败尝试恢复旧 raw/work 与 `steps.txt`；成功替换素材时必须同时使旧 step marker 失效，
  marker 为目录或 symlink 等非常规文件时在移动旧素材前拒绝，无法完整恢复时保留备份并记录路径。
  服务内 material/train 上传及 step 发布与 preview 读取互斥；已有素材的 preview 在替换期间
  必须返回完整的旧代或新代 WAV，不能读到 rename 中间态。此约束不涵盖外部进程修改。
  raw/work/marker、step work/marker 与训练三产物事务一旦移动 live 产物，任何后续异常
  （包括文件系统错误后的一次诊断分配失败）都必须恢复旧产物或保留可恢复备份；只有明确
  成功提交或完整回滚才允许销毁备份。异常可能留下部分 live 与保留备份，保证可恢复，
  不保证自动恢复或不中断可用；服务诊断须记录保留路径，恢复方式见 `docs/DESIGN.md`。
- **FR-008 Temporary cleanup**：一次性预处理/转换使用 `mkdtemp` 唯一路径并由 RAII
  清理。
- **FR-009 Path containment**：静态路径拒绝绝对路径、`.`/`..`，解析 canonical path
  后确认仍位于 canonical webroot；越界 symlink 不得读取。
- **FR-010 Error honesty**：CLI 非零退出、信号终止、启动/等待/日志 I/O 失败、空输出或
  参数非法必须返回错误，不报告成功。preprocess/step/train 产出的音频目录必须至少包含
  一个非空常规 RIFF/WAVE 文件。训练的 clips/model/index 在同目录 staging 中生成，只有
  preprocess、train、index 均成功且三类产物有效时才成组发布并进入 `done`；失败保留旧
  clips/model/index，首次失败不得暴露部分声线。所有显式 boolean field 只接受 `0`/`1`；
  material 的 raw/work 副本只有在 `steps.txt` 记录了成功 step 且 work 含有效 WAV 时才视为
  已预处理，否则训练必须从 raw 执行请求的 cleanup flags。step 的新 work 与更新后的
  `steps.txt` marker 必须作为同一事务发布；marker staging/commit 失败不得留下新 work。
  `/api/convert` 的 `index_rate > 0` 若找不到所选 base/custom voice 的非空常规 Index 文件，
  必须在启动 `vc_convert` 前返回 400；只有正数 rate 配合该文件才传递
  `--index/--index-rate`，`index_rate=0` 必须显式省略这两个 flags。
  step 握手状态或训练作业元数据的初始化在预留 job slot 后失败时，必须返回启动错误并
  释放该 slot，不能让后续请求永久 busy；worker 已接手后，响应构造失败不能提前释放
  仍运行的作业。失败请求只可释放自己尚持有的 reservation：释放后的错误响应异常
  不得再次清除随后请求已取得的 slot。此约束不引入取消或 timeout 策略。

## Training cancellation increment (WEB-FR-011)

User-approved scope (2026-10-10 12:41:53 UTC): training only; queued cancellation
responds immediately and wakes the waiting worker; running cancellation waits for a
safe stage boundary; no forced process termination; an input-material commit already
started completes. Ownership, API additions, migration and rollback are recorded in
[ADR-0006](../../docs/adr/0006-training-cancellation-ownership.md).

The later explicitly authorized extension adds vc_train's optional job-specific
--cancel-file channel and exit status 2. Initialization/clip and complete training
iteration boundaries cooperate; preprocess/index still finish naturally. The runner
never signals a child for cancellation. cancel_requested must not be presented as stopped.
Only requested, normally exited vc_train status 2 with no other process/channel error
is cancellation; other nonzero statuses remain errors. The channel is closed before
staging cleanup and reservation release. cancel_boundary is stage_or_iteration.
Full semantics and host-only verification limits are in [DESIGN](../../docs/DESIGN.md).

The following records the original v9 stage-only increment and its separate tests; its
no-loop-interface limitation is superseded only by the later bounded extension above.
Only the current job owner may publish cancelled and release its reservation after
resource cleanup. New/stale requests must not clear another job's reservation.
A final model/artifact publication is non-cancellable once started and a late cancel
must explicitly report too late; committed uploaded material and prior voice outputs
must not be silently removed. No automatic timeout, step cancellation or resume is added.

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



## 5. Request and Resource Boundaries

当前实现的硬边界：

| Boundary | Limit |
|---|---|
| request headers | 64 KiB |
| request body | 128 MiB |
| aggregate in-flight body reservation | 256 MiB |
| one uploaded file | 64 MiB |
| one buffered audio response | 64 MiB；preview/preprocess/convert 超过时返回 413 |
| one non-file multipart field | 16 KiB |
| multipart parts | 128 |
| listen backlog / connection cap basis | 32 |
| auth token | 4096 bytes |
| voice/material name | 64 safe `[A-Za-z0-9_-]` characters |
| header read deadline | 10 s |
| body read deadline | 120 s |
| base/send socket timeout | 15 s；receive 另受 10 s header / 120 s body deadline 约束 |

请求拒绝 transfer encoding，要求 POST/PUT 带唯一合法 Content-Length，并限制并发请求体
的总预留内存。

multipart 的 `Content-Type` 必须包含唯一、非空、最长 200 bytes 的可打印 ASCII `boundary`
参数；参数名不区分大小写，支持 token 或 quoted-string 值。重复 boundary（即使值相同）、
未闭合引号/转义、参数尾部垃圾或缺少真实 boundary 均返回 400；其他 quoted parameter
中的 `boundary=` 文本不能选择分隔符。只有完整 delimiter line 才结束 part，WAV 二进制
数据中不构成分隔符行的前缀必须原样保存。分隔符不匹配、缺 closing delimiter 或 closing
delimiter 之后有额外数据均返回 400；字段/part 数超过既有上限返回 413。解析失败时不得
替换现有 material 的 raw/work 或 step marker。

multipart part header 必须逐行解析：每个 part 有唯一的 `Content-Disposition: form-data`，
并带一个 `name` 参数；`filename` 至多一次且可省略。省略/空 filename 按既有字段路径处理，
不因无关头出现 filename 文本而变成文件；material 请求没有真实文件时仍返回既有 400。
头字段/参数名称不区分大小写，值保留大小写。支持 MIME token、quoted-string/quoted-pair、
合法折行（只去掉 CRLF、保留空白）和 token 间注释；格式正确的无关头及扩展参数忽略。
同一个头的重复 `name`/`filename` 或重复 Content-Disposition 即使值相同也返回 400，
这是服务的歧义拒绝策略；不同 parts 重复字段名仍允许，不改变已有字段选择规则。
缺少真正 disposition/name、非 form-data 类型、畸形头行、孤立续行、裸 CR/LF、NUL/DEL、
未闭合引号/转义/注释或参数尾部垃圾返回 400。`filename*` 按
[RFC 7578 §4.2](https://www.rfc-editor.org/rfc/rfc7578.html#section-4.2) 不支持。
参数与 MIME 折行/注释语法参考 [RFC 2183 §2](https://www.rfc-editor.org/rfc/rfc2183.html#section-2)、
[RFC 2045 §5.1](https://www.rfc-editor.org/rfc/rfc2045.html#section-5.1) 与
[RFC 5322 §2.2.3](https://www.rfc-editor.org/rfc/rfc5322.html#section-2.2.3)；不据此宣称完整 MIME 兼容。
上述拒绝发生在 upload staging 前，必须保留旧 raw/work/marker，不启动模型 CLI。

## 6. Security Boundary and Non-goals

已实现的输入安全边界不能替代部署安全设施。当前明确不提供：

- TLS/HTTPS、证书管理或可信代理头处理；
- 默认启用的认证、用户数据库、授权角色、多租户数据隔离或审计日志；
- CSRF/session 体系、速率限制或分布式防滥用；
- 容器/OS 沙箱；CLI 子进程仍拥有服务进程的文件与设备权限；
- WebSocket、麦克风采集或实时音频 streaming；
- 非 WAV 上传或输出。

因此默认 `0.0.0.0` + 空 token 只适合受信环境。面向公网时必须由部署者补齐 TLS、
身份认证、网络限制、进程权限和持久数据保护。

## 7. Verification

默认 CTest 注册 `test_web_contract`，启动真实 `vc_serve`、临时 repo 与假 CLI，覆盖静态
root、默认开放、Bearer auth、404/405、step GPU busy、发布失败保留 live work、成功发布
及 staging/backup 清理；还覆盖 step work+marker 的成功与部分发布回滚、
preprocess/step/train/index/convert 的非零退出传播，以及训练成功
但 clips/model/index 为空或无有效 WAV 时不得报告 `done`、失败重训保留旧三产物、新声线
失败不暴露部分产物、unknown voice 不回退到 base、空/非 WAV/超限音频响应失败，以及长寿命
CLI 不继承 HTTP socket。boolean 严格语法、训练 raw+flags / step-marked work 两条 source
分支，以及 base/custom voice 在正数 rate 缺 Index 时 400 且不启动 CLI、零 rate 显式
no-index、非空 Index+正数 rate 的 argv 传播，也由真实 HTTP 与假 CLI argv 门禁。路径穿越和
越界 symlink 仍只有既有手工 smoke 证据，
以下资源与恶意输入门禁也未全部完成，因此不能据此宣称整份 Web 规格 Complete：

- CLI 参数和非法目录必须非零退出。
- 静态 root smoke 返回 200；`../` 和越界 symlink smoke 返回 403。
- token 开启后缺失/错误 Bearer 返回 401，正确 token 才能访问 API。
- 超大 header/body/file/field、重复 header、transfer encoding 和 multipart 边界错误均被
  拒绝。
- staged material 提交、失败回滚、GPU busy 和临时目录清理均应有回归覆盖。
- `/api/step` 已统一为 busy 409；step staged 结果提交失败保留或恢复 live work。
- Web 发起的转换最终仍受 Spec 001 的 CUDA FP32、HuBERT 默认、F0、WAV-only 和
  当前 40 kHz model-rate 输出契约约束。

`test_web_contract` 还覆盖上述 multipart boundary/quoted-string 拒绝路径、二进制 delimiter
前缀原样上传、16 KiB field 与 128 parts 的上限及超限，以及拒绝时保留旧 raw/work/marker。
另有 material 第二次 rename 失败恢复 raw/work/marker、首次失败不残留半成品、marker 目录
与 symlink 拒绝、带上传的训练清除旧 marker 并从 fresh raw 执行 cleanup 的门禁。真实 HTTP
双客户端以 256 次交替上传与 1024 次预览验证整代字节；慢预览停止读取 8 MiB body 时，
并发素材发布须在 2 秒内完成。该 2 秒仅是测试进度阈值，不是产品延迟 SLO。
`test_web_transaction_failure` 使用独立进程和单次 allocation failure，验证 material、step、
training 在失败 publish rename 后仍保留全部旧产物。另有独立审查探针经 OrinNX 复验：
21 项普通提交/回滚故障断言与新增 2 项分配异常断言，以及 material/train/step 各成功/回滚
共 6 项暂停 rename 的确定性并发用例通过；探针不是默认 CTest 的替代品。
测试只使用临时素材和假 CLI，不构成真实语音模型质量验证。

multipart part header 门禁覆盖 23 类拒绝输入、8 类合法写法与跨 part 重复字段；拒绝时
检查原 raw/work/marker 字节不变及无残留 staging，合法参数不被扩展信息污染。

`test_web_job_lifecycle` 在隔离进程中只对请求线程、job reservation 后逐次注入一次分配
失败；内部只读 test access 用于精确选择该窗口，所有行为断言均通过请求/状态响应。
step 的 8 个与 train 的 9 个观测分配点均覆盖：未启动 worker 的失败返回 503、running
解除、下一次请求可接受，假 CLI 失败进入 done/error 并清理 staging、保留原素材；已接手
的 worker 在响应构造失败后仍保持 running，直到其完成。本门禁不注入 worker 内部故障、
不测全局 OOM，也不替代已有 GPU busy 和真实 HTTP 门禁。

请求资源门禁现覆盖不完整 HTTP body：仅从 header 即拒绝 `Content-Length=128 MiB+1`
为 413；两个 128 MiB 声明占满 256 MiB 预留后，第三个非空 body 请求为 503。EOF 截断
返回 400，RST 断开亦释放 body 预留；之后可接受另一完整 128 MiB 声明，并在其截断时
返回 400。整个用例最多三个客户端连接、每个实际 body 最多 215 bytes，不传输大文件。
它还验证 EOF/RST 后 accepted socket 全部关闭、33 个串行截断 EOF 不耗尽连接名额，
以及旧 raw/work/marker、job state、CLI 调用记录不变且没有 staging。MIME 内容本身
合法但 HTTP 长度未满足，仍不能提前进入训练工作流。现有生产实现已通过此门禁，
本组补的是证据，不能把这项通过描述为修复了上传泄漏。

train 上传提交失败的 error response 已移到提交 catch 之外，避免先 finish、响应分配
失败后又 finish，误清除后来作业的 reservation。部分独立审查已有 v4 旧代码的 A/B/C
竞态复现日志；当前修复运行了已有 Web/事务/生命周期门禁，未重跑该专用并发探针。
GPU mutex 始终串行保护执行，该缺陷是作业状态与准入错误，不是已证明的 GPU 并行。

新增门禁保持生产行为：`test_web_contract` 在未完成请求头的第0/3/6秒滴流，
慢header共53 bytes、最多2客户端；10050 ms后400与EOF，socket关闭，job/CLI不变、
后续状态请求成功。观测窗9–13秒，整个用例<15秒；独立D2结果见下文。
`test_web_process_contract` 直接调用原runner，验证空argv/EINVAL、缺失程序/ENOENT、
日志为自建目录/EISDIR、自有短命子进程自行SIGTERM的分类与输出回收；4/4失败分类和
4/4后续正常调用通过。整组34 ms、输出178 bytes、父子peak RSS上界6912 KiB，FD不增，
无存活或僵尸直接子进程、临时目录清除。最多1个helper，每次自然失败<2秒，CTest上限10秒。
初版测试因设备内核不提供`/proc/.../children`在用例前失败，改用只读proc父PID元数据后通过；
这是测试兼容性修正，不是产品红绿缺陷修复。不抢waitpid所有权、不引入interposition。
这些证据不覆盖Web对信号/wait/log I/O结果的消费、底层wait/I/O故障、后代进程组或取消。

独立D2门禁已在OrinNX通过：第0秒完整header和部分body，第20/40/60/80/100秒各补1字节；
**120708 ms**后服务端400与EOF，整个CTest用例120.74秒（总墙钟120.75秒），小于130秒。
声明body为**1024 bytes**，实际body为**216 bytes**；包含就绪/状态查询的全部请求共**570 bytes**，
不应把声明量当作实际传输。最多2客户端，未提前完成body、半关闭或RST；accepted socket关闭，
后续状态查询200、job/raw/work/steps/CLI记录不变、无staging，自有监听器已回收。
中途父子进程RSS采样7720 KiB（单点采样，不是峰值上界）。本轮对原生产实现直接通过，未改生产逻辑。

长用例用默认OFF的`VOXMUTATIO_ENABLE_SLOW_WEB_TESTS`单独注册`test_web_body_deadline`，
复用测试程序的`--body-deadline-only`模式；label为contract/web/host/slow，CTest TIMEOUT=130。
不放入原60秒Web门禁；该D2验收后已恢复OFF，当时默认18项、快速回归18/18、0失败/skip、21.71秒。
当时完整资产配置42项的口径以slow选项OFF为准（新增P2后为43项），ON额外注册1项（未据此重跑模型套件）。

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

仍缺自动门禁：header/file 大小、完整大 body 传输、同时达到连接上限、
更多 rollback/清理异常组合、路径穿越/越界 symlink，
以及底层真实wait/I/O故障矩阵；P2指定消费矩阵已补，不代表OS层全覆盖。

## 8. Success Criteria

- 已配置的受信环境中，用户可通过 Web 完成 WAV material、可选预处理、实验训练状态
  查看和离线 WAV 转换。
- 启用 token 后 API 不允许匿名访问。
- 路径穿越和越界 symlink 不暴露 webroot 外文件。
- 服务不会因并行 GPU 请求静默超额占用显存。
- 文档和启动日志清楚说明：默认开放、训练实验性、非实时、无 TLS。

## 9. Legacy public C++ API migration

本 Spec 的 Web 产品契约只由 `vc_serve` 与内部 `tools/web/` 实现。历史公共头
`include/voxmutatio/webui/server.h` 中的嵌入式 `WebServer` 从未监听 socket，也不实现本
Spec 的鉴权、资源、事务或路由契约；它不是 `vc_serve` 的库形式，不能再作为受支持服务
入口。维护者于 2026-08-11 批准
[ADR-0003](../../docs/adr/0003-retire-legacy-embedded-webserver.md) 的退役路径。

- **WEB-MIG-001 Compatibility window**：旧声明和静态库符号至少保留一个文档化兼容窗口，
  公共头必须仍可由最小消费者包含、实例化、移动和析构；旧服务器入口标记 deprecated。
- **WEB-MIG-002 Honest failure**：为维持源码兼容，`init()` 仍保存配置并返回成功，路由注册
  与停止方法保留签名；`start()` 必须立即抛出 `std::logic_error`，错误信息明确引导使用
  `vc_serve`，不得打开 socket、进入忙循环、打印已启动或返回伪造 API 成功。失败后
  `is_running()` 仍为 false。
- **WEB-MIG-003 Sole supported entry**：受支持调用方应把 `vc_serve` 作为独立进程/服务管理，
  并使用本 Spec 第 2–6 节的 HTTP 契约。此次迁移不改变 `vc_serve` 的 flags、默认开放策略、
  路由、状态码、认证或资源上限，也不新增嵌入式 adapter。
- **WEB-MIG-004 Removal**：旧类只可在后续明确宣布的不兼容版本删除；删除前必须重新搜索
  调用方、更新迁移通知并同步 public API/CMake/SDD。当前没有承诺具体删除日期。

回滚触发条件是公共头消费者无法编译/链接、`vc_serve` 现有 HTTP 契约回退或默认门禁注册
减少。回滚应恢复兼容声明/符号并修复迁移层，不能恢复占用 CPU 的忙循环，也不涉及任何
用户模型、音频或训练数据迁移。
