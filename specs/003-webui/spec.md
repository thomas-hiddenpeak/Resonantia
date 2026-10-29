# Feature Specification: WebUI and HTTP Orchestration

**Feature**: `003-webui`
**Created from implementation**: 2026-08-09
**Updated**: 2026-10-29
**Status**: Implementation present; acceptance partially verified; deployment/training experimental

## 1. Scope

`vc_serve` 是一个纯 C++、HTTP/1.1 over POSIX sockets 的单机编排服务。它提供无构建
步骤的 HTML/CSS/JavaScript WebUI，并通过子进程调用现有 CLI：

```text
browser -> vc_serve -> vc_preprocess / vc_train / build_index / vc_convert
```

服务不是独立模型后端，不在进程内常驻加载模型，也不是实时音频服务器。训练端点只
暴露 Spec 002 的实验性 decoder/GAN及显式joint路径，不能把 Web 可点击解释为训练已产品化。

## 2. Startup Contract

```text
vc_serve [--host IPv4] [--port 8080] [--webroot PATH]
         [--repo PATH] [--auth-token TOKEN] [--ffmpeg /absolute/path]
```

| Option | Default / behavior |
|---|---|
| `--host` | **`0.0.0.0`**，监听全部 IPv4 接口 |
| `--port` | `8080`，合法范围 1..65535 |
| `--webroot` | `webui`，启动时解析为存在的 canonical directory |
| `--repo` | 默认取 canonical webroot 的父目录 |
| `--auth-token` | 默认空，即 **API 不认证**；非空时要求 `Authorization: Bearer ...` |
| `--ffmpeg` | `/usr/bin/ffmpeg`，必须为绝对路径；缺依赖时仍可启动并接受 WAV |

默认配置是为了本机/受信网络便捷使用的“开放”模式。服务在
`0.0.0.0` 且 token 为空时打印警告，但仍会启动。公网或不受信网络部署必须显式增加
token，并在外层配置 TLS、网络 ACL/防火墙和可信反向代理。

Bearer token 只保护 `/api/` 路径；静态 Web 资源仍可公开读取。当前服务没有用户、
角色、session cookie 或租户隔离。

## 3. As-built API

| Method and path | Behavior |
|---|---|
| `GET /` and static paths | 从 canonical webroot 提供 WebUI 文件 |
| `GET /api/voices` | 枚举 base/已导出声线、当前训练状态及`training_profiles`（decoder/gan/joint/joint-full） |
| `GET /api/sep-models` | 按任务返回已安装分离模型 |
| `POST /api/material` | 上传一组支持的音频，归一化为 WAV 后 staged 替换 raw/work material |
| `POST /api/step` | 对 work set 执行一次 separation/de-reverb/de-echo/de-noise/de-harmony |
| `GET /api/preview?name=...` | 返回 work set 的首个 WAV |
| `POST /api/preprocess` | 对单个支持的音频归一化后执行可选预处理并返回 WAV preview |
| `POST /api/train` | 启动单个decoder/GAN/joint/joint-full训练任务；joint默认50步、seg32、epochs0；joint-full独立契约见下文 |
| `GET /api/train/status` | 返回 job_id/kind/state、stage/running/done/error、cancel_requested/cancelled/cancellable/cancel_boundary 与受限 log tail |
| `POST /api/train/cancel` | 单个 job_id 定位训练作业；202接受请求、200重复请求，阶段/完整训练迭代边界取消，发布开始后409 |
| `POST /api/convert` | 对上传或 material session 的 WAV 运行离线转换，返回 WAV；正数 `index_rate` 要求所选声线存在 Index，零值显式 no-index |

未知 API 返回 404；已知 API 的方法不匹配返回 405 并带对应 `Allow`；非 API 静态路径
的非 GET 请求返回 405/`Allow: GET`。这些方法语义由 `test_web_contract` 真实 socket
门禁覆盖。上传支持 FR-012 的格式清单；内部素材、模型 CLI 输入及输出仍为 WAV。

joint映射到`vc_train --gan --gan-profile joint`，复用原取消和model/index/clips事务发布。
UI仅在`training_profiles`明确声明joint时显示选项；不声明的旧服务继续隐藏，刷新能力
不覆盖用户已填参数。joint选中时步数上限50，seg32/epochs0固定；详细输入边界见Spec 002。

`presets.json`的推理预设可用可选`voice`精确匹配一个声线名。用户选择该声线时自动
应用对应预设；异步声线/预设加载顺序不影响结果，同一声线刷新不覆盖用户后续手动调整。
未匹配声线保持当前参数。正式声线`Tom_Cybersinger`匹配`tom_cybersinger`：index0.35、RMS0、
protect0.5、filter3、pitch0。RMS0跟随原唱强弱，1保留模型音量；这是参数说明，不是
任意输入质量保证。是否已部署及听审范围见PROJECT_STATE。

## 4. Functional Requirements

- **FR-001 Static UI**：无 Node/npm 构建依赖，静态 HTML/CSS/JS 可由 `vc_serve` 提供。
- **FR-002 Configurable bind**：允许配置 IPv4 host/port，默认 `0.0.0.0:8080`。
- **FR-003 Optional auth**：token 非空时，对所有 `/api/` 请求执行 constant-time Bearer
  token 比较；默认不启用认证。浏览器只共享尚未完成的 token 提示，首次输错/取消或
  token 轮换后，后续请求必须允许重新输入；已完成的提示结果不得永久缓存。
  `test_web_auth_ui` 以生产 JS 验证重试、并发提示共享、轮换及 sessionStorage 失败回退；
  Node.js 18+ 仅用于可选的测试运行，不增加 WebUI 构建或服务运行依赖。
- **FR-004 Upload validation**：扩展名按 FR-012 白名单校验，不区分大小写；WAV 保持原有
  RIFF/WAVE signature 校验及字节，新格式必须完整解码后提交；不安全文件名或归一化为
  `.wav` 后重名的整次上传失败（如 `song.mp3` 与 `song.flac`）。
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
  清理。成功转换在清理前把 CLI 日志末尾最多 2048 bytes（包含 HuBERT/F0/Index/Synth
  timing）写入服务日志，并记录子进程至结果读回的 wall time；不改变 WAV 响应或新增
  持久任务/进度 API。该 wall time 不包含此前输入预处理，也不是纯 GPU kernel 时间。
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



## 2026-10-26完整联合训练增量（实施中）

沿用[ADR-0009](../../docs/adr/0009-resumable-joint-training.md)的已授权迁移。
`mode=joint-full`显式选择完整联合训练，默认200轮、crop32，单GPU batch1。
旧decoder/gan/joint接口与发布行为保持；新模式的结果留在声线目录的
`full_training/result/`，仅表示训练产物完成，听审接受前不加入声线列表或替换现有声线。

- 新作业固定一份`full_training/clips/`；缓存、恢复点和阶段候选保存在相邻目录，
  与一次性staging分离。取消、失败及服务重启后保留。一个声线名称仅容纳一个完整作业，
  重开完整训练使用新名称；不能静默覆盖已有完整状态。
- `resume=1`仅用于本模式，复用固定clips及最新完整恢复点，禁止同时上传或改变清理选项。
  若预处理已完成但尚无恢复点，允许从第0步启动并在日志中明确说明。
  数据/配方身份由CLI再次校验，不能用推理G代替恢复点。
- `prepared=1`是本模式的可选API参数：已有work/raw是人工审核的40k、0.32..6秒切片，
  原样固定，不重切、不加清理；与VAD及所有清理选项互斥，CLI在CUDA前验证每片元信息。
  默认`prepared=0`沿用Web预处理。此入口供固定训练/留出集，浏览器默认仍自动切片。
- 参数中显式boolean仍只接受0/1；`steps`不用于joint-full，轮次1..2000。

页面载入时读取现有`/api/train/status`，仅对仍在运行且有name/job_id的train任务恢复进度
轮询和取消按钮，不要求重新上传文件；不改变用户已填名称，不恢复step或历史终态。
异步状态返回不能覆盖页面内已经发起的新作业。此功能只恢复浏览器跟踪，不自动开始、
取消或恢复服务端训练；服务进程重启后的续训仍须显式`resume=1`。
训练进度区分模型/数据核验、冻结前端缓存段数和联合训练步数/轮次；缓存准备阶段不得
显示旧的固定55%训练进度。恢复作业追加日志时，使用较新的阶段，避免显示前次运行的步数。
  状态返回`resumable`和`review_required`，长训练日志包含epoch/step/LR和阶段耗时。
  UI区分待听审结果与已发布声线。GPU串行、作业归属、取消安全点及失败诚实契约不变。

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

### Multi-format upload (WEB-FR-012)

2026-10-24 维护者明确批准服务器 FFmpeg 方案，见
[ADR-0007](../../docs/adr/0007-web-audio-upload-decoding.md)。material、带文件 training、
preprocess、convert 共享同一归一化入口，支持 WAV、MP3、M4A（AAC/ALAC）、AAC、FLAC、
OGG（Vorbis/Opus）、Opus、WMA、AIFF/AIF。WAV 字节保持不变；其他格式在 request staging
中由 CPU FFmpeg 解码成 PCM16、44.1 kHz、mono WAV；模型输出继续遵守 40 kHz WAV 契约。
原输入可能有损；下混/重采样/量化不承诺保存原始多声道或高位深。

文件名保留合法UTF-8字符（含中文、日文、带重音字母和emoji），不能删空非ASCII
basename而误判扩展名或制造批内重名。ASCII空格/标点仍沿用现有过滤规则；拒绝
路径分隔符、NUL、非法UTF-8及超过255字节的原始basename。扩展名不区分大小写，
替换为`.wav`后的真正重名仍使整批失败并保留旧素材。

仅允许对应 demuxer 和本地 file 协议，禁用 MOV 外部引用，选取首个音轨，丢弃视频、字幕、
数据和 metadata。以 argv 启动 `/usr/bin/timeout`、`/usr/bin/prlimit`、配置的 FFmpeg；
不使用 shell。每实例最多一个解码批次，解码线程2、filter线程1、地址空间2 GiB、单次
分配64 MiB、每文件120秒 deadline，TERM后2秒未退出则KILL。此限制不适用于模型进程。
新解码产物每文件最多64 MiB、每批累计128 MiB（约12.7/25.4分钟 PCM16单声道），超限
整批413；FFmpeg截断停止的输出不得被接纳。原HTTP body128 MiB/inflight256 MiB仍生效。
错误为400（格式/损坏/无音轨/超时）、409（decoder busy）、413（解码超限）、503（依赖
不可启动），磁盘staging错误500。失败不得提交素材或调用模型；已存在素材及marker保留。

UI 两个选择/拖拽入口使用相同白名单；训练支持多文件。上传和准备期间禁用预处理与主
操作按钮，失败显示原因；同一入口按序上传，过期响应不能重新启用旧素材。时长仅用
浏览器metadata估算，浏览器不认识的格式仍可由服务器解码为WAV试听。

迁移：保留现有WAV路径与数据，部署完整音频codec的FFmpeg及coreutils/util-linux；
独立构建和短音频验证后备份vc_serve及UI，在无活动任务时一起切换并重启。回滚恢复
二进制与UI，新增素材已是标准WAV，无需迁移模型或原文件。部署验证仅GET，由用户试听。
`test_web_audio_upload`覆盖四入口、UTF-8文件名/边界/归一化重名、失败事务、大小/并发/超时；`test_web_audio_ui`覆盖界面；
`RESONANTIA_ENABLE_AUDIO_CODEC_TESTS=ON`注册真实短人声的`test_web_audio_codecs`。

## 6. Security Boundary and Non-goals

已实现的输入安全边界不能替代部署安全设施。当前明确不提供：

- TLS/HTTPS、证书管理或可信代理头处理；
- 默认启用的认证、用户数据库、授权角色、多租户数据隔离或审计日志；
- CSRF/session 体系、速率限制或分布式防滥用；
- 容器/OS 沙箱；CLI 子进程仍拥有服务进程的文件与设备权限；
- WebSocket、麦克风采集或实时音频 streaming；
- FR-012 白名单外的输入、视频/播放列表/DRM媒体及非 WAV 输出。

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

- 已配置的受信环境中，用户可通过 Web 完成音频上传并归一化为 WAV material、可选预处理、实验训练状态
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

## 2026-10-29 正式声线与翻唱工作流（已实现，限定资产验证）

用户已听审通过 CyberTom 200 epoch / 81,200 step 正式训练及修正版完整翻唱。
新增能力沿用现有 C++ CLI、单 GPU 串行 HTTP 服务与 runs 目录，不改变模型格式。

- 发布须显式确认听审，完整训练候选可先试听；训练结束不等于发布。已有声线不得被覆盖。
- 完整歌曲使用立体声人声分离 → 立体声去和声 → 主唱转换；干声可跳过清理。
  去混响仍可选，不能默认为所有素材有益。歌曲可输出带原伴奏的翻唱，使用固定增益避免削波，
  不使用压缩器；分离残留与局部错音仍须听审。
- 转换会话隔离、处理中锁定输入；处理失败不展示旧结果为新结果。兼容旧 HTTP 字段；
  新字段缺省保留旧行为。FFmpeg 立体声解码仅对显式请求开启，保留原资源限制。
- 训练界面默认完整联合训练 200 epoch；短试验最多 50 step，显示其用途。
  展示实际阶段、轮次/步数与日志；提供历史任务、断点恢复、候选试听及发布入口。
- 人工局部修音以输入音频的秒为坐标，只支持短区间平滑或半音偏移，默认关闭，
  不允许跨无声区插音；不将本次歌词特定校正宣称为自动通用修复。
- 新能力须通过接口错误路径、前端状态、真实模型短片段及完整回归验证后部署。

兼容与回滚：发布操作保留已有声线/训练素材/完整训练状态；旧 CLI 单声道预处理默认不变。
新增立体声处理仅适用于完整文件、不剪切时间轴；回滚可使用原服务二进制和静态页面，
保留正式模型供后续恢复。实际验证记录见 PROJECT_STATE。

同日维护者另行明确要求清除历史声线并改名：本机仅保留正式发布的 **Tom Cybersinger**，
内部安全名称为`Tom_Cybersinger`，8个旧声线的模型/索引及其目录内的候选模型副本实际删除，
不以界面隐藏代替清理。原正式训练任务`CyberTom_Full_20261026`保留素材、checkpoint、
阶段候选和日志；它是训练来源，不再作为第二个发布声线。正式模型与索引字节、听审参数不变。
声线列表、默认预设、正式版说明及训练记录显示名称同步；这次删除不改变以后显式训练/发布的能力。

### 新增 HTTP 字段与端点

| 接口 | 新增契约 |
|---|---|
| `GET /api/train/runs` | 发现含 `full_training` 的任务，列出规范命名的阶段候选、`result_ready` 与 `resumable`；恢复须同时存在固定clips与非空checkpoint |
| `POST /api/train/publish` | multipart `name,candidate,publish_name,reviewed=1`；仅接受该run的正规候选文件与完成索引。400非法/未听审，404缺资产，409忙/目标已有model、index或clips，500发布失败。复用三件套回滚事务，review记录随clips发布；不覆盖现有声线 |
| `GET /api/voices` | 保留 `voices` 字符串数组，新增 `reviewed_voices` 供区分听审发布与历史声线 |
| `POST /api/material` | 可选 `preserve_stereo=1`，非WAV音频用FFmpeg解码到44.1k双声道PCM16；缺省仍为mono，WAV保持原内容 |
| `POST /api/step` | 原返回对象新增 `job_id`，浏览器只认领相同任务；清理显式调用立体声完整文件路径 |
| `POST /api/convert` | `workflow=manual|song|harmonies|dry`（缺省manual）、`output_kind=vocal|mix`（缺省vocal）。mix仅限song。manual取session/work，其他取session/raw，锁定GPU后固定输入快照；pp_*也适用于session输入 |
| `POST /api/convert` | 可选 `training_run,candidate` 试听未发布候选；检索率>0仍须完成索引。可选 `pitch_edits` 沿用Spec001的严格语法与边界 |

song先独立生成全人声，随后去和声；干声可选去混响。RoFormer之间用双声道float32 WAV，
清理结束后转mono40k PCM16再进入模型，与已听审交付路径相同。需要时对主唱作一次固定
降增益防止PCM16削波。混音为 `原立体声 − 全人声 + 转换主唱`，使用全曲单一增益使峰值≤0.95，
不使用压缩/限制器，也不加入原伴唱。库仍只读WAV，压缩格式解码依赖已批准的FFmpeg方案。
清理输入上限600秒/128MiB，上传、解码与HTTP响应原上限不扩大；超大float32清理预览可
生成PCM16试听副本，原处理结果不变，副本仍须≤64MiB。

界面默认完整歌曲与混音输出，默认Tom Cybersinger及听审参数；若未安装正式声线，不显示
已发布说明。训练只推荐完整联合200轮与最多50步短试验，旧配方保留高级入口。
每标签页独立material会话；处理中锁定输入与参数，不接受旧job完成冒充当前步骤。
进度条表示当前阶段真实计数，未知进度不提供伪百分比；训练展示日志实测计算秒/步，
不等于完整墙钟时间或GPU利用率。候选试听与发布必须由使用者确认，不自动晋升。

## 2026-10-29 工作室交互（WEB-FR-013）

维护者要求按实际业务重设计UI并自审后发布。前端仍为无构建HTML/CSS/JS；不改变HTTP、
GPU所有权、存储格式或音频算法。设计、验收清单与证据见
[工作室交互记录](../../docs/WEBUI_STUDIO_20261029.md)。

- 三个独立入口：制作翻唱、训练声线、声线库。翻唱先上传并确认输入类型，再选择声线，
  最后生成和听审；新训练按任务名称、录音检查/清理、训练计划、启动确认排序。
- 处理概览从当前设置生成，清楚区分将要执行的步骤和真正执行状态；没有后端计数时
  只展示等待时间。不得以模拟阶段、GPU利用率或估算百分比冒充执行证据。
- 上传失败可以原位重试；HTTP业务错误不等于服务离线；连接中断可重连。处理中锁定
  相应输入，活动训练阻止新的生成，训练启动须确认摘要；训练阶段仍使用原取消契约。
- 生成结果记录其输入、声线和参数背景；修改设置明确提示结果仍是上一版。原始与转换
  音频可同位置切换且不重叠播放。局部修音放在结果听审后，非法区间在请求前拒绝并保留结果。
- 声线库分开已发布声线与训练候选；空记录/无候选/无恢复点时禁用相应操作，候选试听
  明确标记未发布。发布保留原显式听审确认，正式声线Tom Cybersinger及参数不变。
- 提供键盘上传与页签导航、可见焦点、状态/错误文本、桌面双列与手机单列布局；可选
  浏览器门禁验证320/390/768/1440宽度及WCAG相关自动规则，不把自动检查称为完整认证。

迁移仅替换静态页面；开发期间正式服务使用冻结页面，验收后切换。回退使用同一服务的
旧页面，不改模型/Index/素材。现有Node和Web门禁继续保留，独立浏览器HTTP替身明确标记。

## 2026-10-29 后台长任务（WEB-FR-014）

维护者授权当前任务结束后补齐转换、清理和训练的后台任务生命周期。实施前决策、兼容和回滚
见[ADR-0010](../../docs/adr/0010-durable-web-task-lifecycle.md)，实际状态见
[实施清单](task-lifecycle-20261029.md)。本节为新增要求，尚未实现时不得用旧FR-008/013计时降级
宣称已满足。旧同步接口保持兼容；新界面须使用独立任务而非整次长HTTP响应。

- 关闭页面、刷新或断网不得取消后台工作；重开可发现活动任务及关闭期间发生的成功/失败终态。
- 转换/清理/训练均有稳定ID、任务列表、参数摘要、真实阶段、阶段内计数、耗时、最近进展时间、
  有界日志、明确错误、可执行控制和结果访问。训练进度与GPU利用率不混用。
- 任务参数/输入固定，完成结果和有效阶段产物持久保存；下载失败不能导致重跑计算。
  混音失败可从已生成主唱重试；只有阶段产物校验通过才能复用，正常舍入误差不能判成截断。
- 网络错误表示客户端暂不能确认状态，不擅自标记后台失败或自动重复提交。
- 暂停/继续在安全点执行；关闭页面不是暂停。joint-full沿用完整checkpoint，其他profile的
  恢复能力据实声明。转换长阶段分块恢复须保留时间轴/拼接/采样语义并单独通过数值门禁。
- 服务重启后仍可查询历史，旧活动任务标为interrupted并给出受支持的恢复操作；不得伪装仍运行
  或无条件重开计算。版本/输入/参数/模型/索引身份不匹配或检查点损坏时拒绝恢复。
- 元数据、日志、列表和文件传输有界；任务清理不得删除活动任务、唯一检查点、训练来源或正式模型。

验收至少覆盖真实页面关闭/重开、关闭期间终态、断网/重连、提交响应丢失、服务重启、暂停与完成
竞争、失败阶段重试、资源/鉴权/路径失败及真实短片段恢复。按阶段执行迁移，生产仅在无活动任务
且门禁通过后切换；程序回退保留新任务产物和原模型，旧版不支持的记录明确保留待新版本恢复。

### 任务控制与完整训练恢复接口

- `POST /api/task/pause`、`POST /api/task/resume`接收`id`和`attempt`。暂停202表示请求已接受，
  必须查询到`paused`才表示计算资源已释放；发布阶段拒绝暂停。重复继续使用同一旧attempt返回
  同一新attempt，迟到暂停返回409。失败优先于暂停，持久化失败返回不可确认而非伪终态。
- `GET /api/task`及列表返回`attempt`和`actions.pause/resume`。转换/预处理按已验证块及完成阶段
  继续；joint-full按完整训练checkpoint继续，索引失败可复用已验证G。未启用恢复的历史记录及
  decoder/gan/joint不宣称等价续训。暂停索引须等待当前索引阶段完成，其结果随后保留。
- `POST /api/train`可提交稳定`task_id`；启用任务存储时joint-full固定输入及G/D副本，绑定前端、
  清理模型、参数和程序身份。`full_training/web-task`关联原任务；这类训练须通过原任务继续，
  旧`resume=1`不能另建执行绕过归属。未管理的旧完整训练可按原resume入口显式导入一次。
  训练task_id为8..64个安全字符，与旧训练控制的ID上限一致；已继续的完整训练拒绝旧的、没有
  attempt的`/api/train/cancel`，防止旧页面取消请求误停新代次，使用任务控制暂停。
- 训练清理使用独立CLI分块检查点；固定切片集合、完成G和完成Index分别校验。训练结果仍进入
  `full_training/result`待听审，不自动发布或修改正式声线。旧取消保留cancelled语义，完整训练
  保留的checkpoint可通过任务继续；其他训练profile仍遵守原取消契约。
- GPU子进程继承跨进程资源锁。服务异常退出而子进程尚存时，新计算拒绝占用同一GPU，历史查询
  仍可用。启动时对原任务/attempt控制文件请求安全停止，不强杀旧计算，不自动重试，不把
  请求停止伪装成paused。若旧进程因I/O失败提前退出，仅能从此前完整checkpoint继续，未提交
  更新可能重算；不以服务PID消失推断计算已经停止。

状态/阶段、分块接口、真实浏览器及生产切换的证据见[实施清单](task-lifecycle-20261029.md)。
后端单项测试不能代替真实浏览器验收或整项FR-014完成证据。

任务记录容量上限属于新任务准入限制：达到上限返回409，已有任务查询/日志/结果继续可用，
不等同于I/O或格式损坏。磁盘预算与安全清理策略仍单独验收。

列表另返回独立于分页的`active_id`；训练历史返回关联`task_id`，页面通过原任务查看和继续。
`GET /api/storage`提供任务文件大小合计、文件系统可用空间、保护线及记录数量；
`POST /api/task/cleanup`使用`id/attempt`，仅在无活动执行时清理成功任务的冗余副本，重复请求
不会删除额外内容，结果仍可下载。超出磁盘准入保护线返回507，磁盘核查失败503；均不破坏历史读取。

### 磁盘保留与清理实施边界

- 默认最多256条未清理任务、4096条历史ID。清理不删除任务ID或终态，旧提交不能因清理重新执行。
- 仅成功任务可以显式清理冗余副本；运行、暂停、中断和失败任务保留所有恢复内容。清理保留原输入、
  完成音频、日志、训练素材、完整训练检查点、候选和正式模型，不自动按时间删除这些资产。
  新音频成功记录保存结果SHA256，清理前复核；旧记录无摘要时仍能查询/下载，但禁止清理副本。
- 工作任务管理的总文件预算64GiB，单任务24GiB，文件系统预留8GiB；历史未纳管的训练/正式资产
  不计入任务预算，但影响实际可用空间。新提交和继续前检查；运行中在阶段/结构化进展边界检查，
  达到保护线请求安全暂停。该限制是边界保护，不是文件系统硬配额，不保证阶段临时写入零超量。
- 页面展示实际占用、剩余空间、保护线和清理范围。扫描、身份或I/O异常应拒绝新计算且明确显示，
  不能删除历史来掩盖错误；历史读取和已有结果下载仍独立可用。
- 格式迁移沿用v2记录，以仅允许成功任务使用的保留标志记录清理结果；清理期间中断可重复执行，
  不删除未知路径。磁盘与清理门禁完成前不得宣称此节已实现或部署。

### WEB-FR-014：完成任务调整参数（2026-10-29）

已实现并部署；测试、首次失败与回退见[参数修订验收](task-revisions-20261029.md)。

维护者反馈已完成翻唱调整参数必须重做全部流程。本增量在现有任务/阶段快照机制内补齐派生版本，
不修改训练数值、CLI或存储格式，不建立共享可变缓存。

- 任务中心对成功的转换/预处理任务提供“调整参数再生成”，还原原素材和设置，无需重新上传。
  旧ID、状态和结果不变，提交使用新稳定ID，记录`source_task`供回看旧版；关闭页面恢复规则不变。
- `POST /api/tasks/convert`或`/api/tasks/preprocess`可带`source_task`，与上传/session互斥。
  未给出的音频参数沿用父任务；只接受已成功、具备audio-v1身份的音频任务，非法ID400、缺失404、
  非成功/不支持409。新ID重复仍409，查询原新ID核实，不重复计算。
- 读取父任务须核验原始素材、保存的配方及程序/模型依赖；阶段只从已完成且SHA正确的产物导入。
  分离只依赖相同歌曲输入/分离路径；清理另依赖全部清理设置；转换另依赖声线/索引/音高/检索/
  动态/保护/平滑/局部修音。仅改变输出人声/混音时可复用转换；改变清理设置后重做清理及转换。
  不导入旧转换块检查点，不把改变配方视为原任务续跑。当前版本身份不兼容/坏产物明确拒绝，不混用。
- 新任务独立复制所需已验证阶段文件并写自己的完成标记，之后恢复不依赖父任务目录。原输入别名
  仍保持原输入，避免无意额外重采样/归一化。没有可复用产物时明确重算，不假报命中。
- 已清理冗余模型的父任务仅在当前同名模型能验证原身份时使用；若转换输出已清理，原纯人声最终
  结果可在摘要相符时复用，否则重算转换。原素材/正式模型/唯一结果不删除。
- `GET /api/task/source?id=...`提供成功音频任务的原输入（与结果相同鉴权、Range与有界读取）。
  页面恢复原设置并提示创建新版，保留父任务导航；手动逐步清理不能操作另一个浏览器会话的素材。

迁移是API可选字段和任务parameters的兼容扩展，旧任务仍可读，旧客户端行为不变；已有whole-workflow
身份只用于校验父任务，派生时按明确依赖选择完成阶段，生成新身份。先隔离页面/host/真实短音频门禁，
空闲时更新服务；回退恢复程序和页面并保留全部任务，旧界面没有派生入口。ADR-0010补充同一边界。

### 2026-10-29 Web更新后的身份兼容修复

实际旧任务表明，Web层误用了隐式绑定`/proc/self/exe`的CLI操作身份；仅更新Web服务就使保存的
完整身份不匹配，前次同一二进制的短音频测试未覆盖这个场景。修复采用显式Web编排协议版本，
继续核验全部配方、输入、模型、原生计算CLI及阶段产物SHA；原生CLI块身份仍绑定实际计算程序。

旧身份仅兼容已核对数值编排相同的两个已部署Web版本（2026-10-29 09:27与14:04），按其确定
可执行SHA重新计算完整旧摘要，必须与保存值完全相符；未知旧版本或任何依赖变化仍拒绝。
兼容不改写原身份、阶段标记或旧记录，后续标记沿用实际匹配的身份。新任务使用稳定Web协议身份。
训练Web编排与切片清单有同样隐式绑定，需一致修正嵌套摘要，不能只修翻唱。训练数值/检查点
格式和流程不变。兼容策略细化见ADR-0010；回退不删除数据，新协议任务需新服务继续。

上述约束先于实施记录；现已部署。跨服务二进制、两个旧生产者、嵌套训练身份、依赖漂移拒绝及
实际失败任务副本的门禁结果见[参数修订验收](task-revisions-20261029.md#跨web更新的身份修复2026-10-29)。

### WEB-FR-014：任务显示名称与双音频结果（2026-10-29）

维护者要求任务可重命名，默认同时呈现完整翻唱及干净人声。此处干净人声指转换后的目标声线、
不含伴奏；歌曲默认mix已生成两份，复用现有结果及conversion阶段，不重复模型计算。

- `POST /api/task/rename`接收`id/name/expected_name`。名称为1..255字节合法UTF-8、去首尾ASCII
  空格后非空且不含控制字符。200返回任务；非法输入400、不存在404、他人已改名409、保存失败503。
  比较原显示名，重复提交相同新名幂等；允许活动/终态及训练任务改名，不用进度revision充当名称锁。
  显示名称存于既有parameters的display_name，原name/目录/训练声线/配方/身份/进度均保持。
- GET任务返回display_name；详情附outputs及不可用提示。结果接口增加可选`variant=mix|vocal|result`，
  省略保持原行为。无对应结果404、尚未完成409、未知variant400。沿用鉴权、Range和有界流式发送。
  额外人声须有有效完成标记，读取时校验SHA，缺失/损坏不伪装为成功，也不使其他任务查询失效。
- 成功歌曲默认展示“完整翻唱”“干净人声”两个独立播放/下载入口，无需先点载入。鉴权音频按需
  点击载入并保持单份256MiB、双份512MiB上限；普通页面用原生流式音频。无伴奏输入只显示人声。
  旧任务若人声已经删除则明确提示，不重新转换。用户明确选择仅人声时不额外生成伴奏版。
- 清理冗余副本必须保留混音任务独有的人声；仅人声任务仍可移除与最终结果重复的output.wav。
  不改变已有WAV、模型数值、CLI及Web编排身份，不新增数据库或文件格式。

迁移为既有元数据的可选显示字段和只读结果查询扩展，旧记录/客户端仍可读；回退保留所有文件，
旧程序不显示新名称且旧清理规则可能删除独有人声，回退期间不得执行混音任务清理。
