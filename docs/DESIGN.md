# Resonantia 架构设计与实现边界

**状态日期：2026-10-24（八项审计修复；产品范围不变）。** 本文记录当前实现，不把目标设计写成已经支持的能力。可执行入口和测试方法以仓库当前 `CMakeLists.txt`、CLI `--help` 与 CTest 注册为准。

本文是 SDD 的“当前实现视图”，不是独立的需求来源。规范性需求、状态证据和历史材料的权威层级及阅读路由见 [`../specs/README.md`](../specs/README.md)；需求到代码和测试的映射见 [`../specs/TRACEABILITY.md`](../specs/TRACEABILITY.md)。若本文与功能规格或可执行证据冲突，必须按 SDD 漂移处理并在同一变更中校准，不能静默选择其中一份。

## 1. 产品范围

Resonantia 将 RVC 风格的声音转换推理链迁移到 C++20/CUDA，当前产品与验证基线是
Jetson Orin 系列（SM87；本地验证机为 Orin NX）。运行时不依赖 Python/PyTorch；Python
工具只负责权重转换、下载和生成离线参考数据。

当前主链：

```text
WAV -> 重采样至 16 kHz -> HuBERT -> 可选特征检索 -> RMVPE/F0 处理
    -> F0 VITS v1/v2 -> WAV
```

Web 服务是 CLI 编排层，不是独立的模型执行后端。训练代码和声源分离代码已存在，但仍按实验性能力管理。
端到端实时变声当前没有实现；相关目标只存在于
[`Spec 006`](../specs/006-realtime-voice-conversion/spec.md) 与
[`ADR-0002`](adr/0002-persistent-streaming-runtime-and-local-audio-boundary.md) 的已批准目标中。
ADR-0002 于 2026-08-10 `Accepted`，授权从 R0 证据开始分阶段实施，但 persistent runtime、
streaming session、`vc_realtime` 与 ALSA adapter 均尚未进入本设计的 as-built 架构。
R0 已加入 legacy `infer_stream` characterization、mode-aware `BenchmarkReport` 构造期
validator、machine-readable asset manifest、离线 Pipeline Index fail-closed 修复与独立
exact-Index 诊断 probe；这些都是证据/契约 scaffold，不是 realtime 产品入口。

批准的首个 realtime qualified profile 为 v2/HuBERT/FCPE/40 kHz，并要求有效 Flat Index 与
`index_rate > 0`；no-index 仅是显式诊断/兼容模式或启动时预授权的粘滞 overload fallback，
不得静默旁路。主验收使用 Index-enabled 10-minute run（p95 `<= 250 ms`、p99 `<= 350 ms`、
0 fallback）。R0 已测 standalone CPU exact reference 与 persistent-GPU exact retrieval 的
synthetic/available-feature 诊断可行性，但仍无代表性目标声线 model+Index pair、完整 pipeline
realtime latency/memory 或质量 A/B baseline。probe 不选择产品 backend/支持上限；`index_rate`、
rows 上限、backend 和 fallback trigger 必须由 target evidence 决定，不能视为现状。

## 2. 明确的运行时契约

| 维度 | 支持 | 不支持/会拒绝 |
|---|---|---|
| 设备 | CUDA | CPU 推理 |
| 精度 | FP32 | FP16 |
| 音频文件 | WAV | FLAC、MP3 |
| RVC 模型 | F0 v1（256D）、F0 v2（768D） | non-F0 |
| 内容编码 | HuBERT（默认）、WavLM（需匹配合成器） | 不兼容特征空间的模型组合 |
| F0 提取 | RMVPE（默认）、FCPE（可选） | PM 等未实现方法 |
| 输出采样率 | 当前兼容模型的原生 40 kHz | 32/48 kHz 等其他模型率；独立 target sample rate 转换 |
| 变声参数 | pitch、index rate、RMS mix、protect | formant shift |

配置结构保留部分未来字段，不能据此推断能力已经实现；初始化必须对未实现的非默认配置失败。

## 3. 模块状态

### `core/`

共享类型、设备探测和 CUDA 内存设施。产品执行路径依赖 CUDA；`Device` 类型能描述 CPU 不等于模型具有 CPU 后端。

### `io/`

实现 WAV 解析、同目录 staging + atomic rename 写入、声道下混、线性重采样和
safetensors 加载。WAV 发布在完整写入、`fsync` 和 close 后提交；失败保留旧目标并清理
staging。private bounded seam 另提供 checked `pread` WAV reader、mode-0600 raw spool 与
incremental atomic WAV writer；源文件身份漂移、short read、长度/字节数溢出均 fail-closed。
通用 `SafetensorsLoader::load` 仍是 file-backed mapping；只有 VITS consumer 选择下述 anonymous
snapshot。音频 API 目前不提供 FLAC/MP3 解码器。

### `content/`

- HuBERT：推理主链默认使用，v1 输出 256D，v2 输出 768D。
- WavLM：有独立实现、可选 pipeline/CLI 分发和本地对拍测试；必须配合在 WavLM 特征空间训练的合成器，不能把 stock ContentVec 模型视为兼容。

### `f0/`

- RMVPE：推理主链默认使用；初始化会完整校验运行时使用的 tensor，并持久共享只读、
  已验证的 checkpoint 映射。失败重初始化会清空旧状态；算子层的临时 device buffer 由
  `core::CudaBuffer` 持有，CUDA copy/计算抛错时沿栈释放。
- FCPE：有独立实现、可选 pipeline/CLI 分发和本地对拍测试；初始化完整校验 tensor，
  事务式物化 weight-norm 输出投影，失败重初始化会清空旧状态。R0 已固定
  `torchfcpe==0.0.4` package/checkpoint provenance、locked tool project 与 same-filesystem
  staged + rollback 事务式发布；不是默认 F0 方法，也未据此证明实时性能。

### `index/`

产品实现按 [ADR-0005](adr/0005-immutable-flat-index-snapshot.md) 把源 Flat Index 加载为
private、已 unlink 的 filesystem-backed snapshot，再 mmap 执行原有 host exact weighted k-NN；
不依赖 FAISS，也不会把整份索引复制到显存。loader 以 nonblocking fd 拒绝 FIFO/非 regular
输入，校验 header/精确字节数/有限值，优先在源目录 `O_TMPFILE` + `FICLONE`，否则在已 unlink
临时文件上以 1 MiB buffer 复制和复核。copy fallback 需要约 `O(Index bytes)` 的临时 filesystem
容量；tmpfs 可计入内存/swap，ENOSPC/权限/快照失败使 required Index fail-closed。成功后源路径
replacement/truncate/overwrite 不会改变 active bytes 或触发 SIGBUS。`build_index` 对 WAV 路径
确定排序，并以同目录 staging、file `fsync`/close、rename 原子发布，父目录 fsync best-effort。
standalone R0 probe 的 persistent-GPU exact retrieval 仍只用于诊断/对拍，不是产品 backend，
也不定义 qualified rows/rate/batch 上限。

### `synthesizer/`

实现 F0 VITS v1/v2 的 TextEncoder、Flow、NSF/HiFi-GAN 生成器。初始化逐项校验所有运行时
已用 tensor 的 F32 dtype、精确 shape/nbytes、可读对齐和有限值；weight-norm 的 `g/v` 还要求
每个 `v` row 非退化，并以 double 次序验证重建结果有限且可表示为 FP32。Synthesizer
checkpoint 上限为 256 MiB；初始化从单一 `O_CLOEXEC|O_NONBLOCK` 只读 fd 复制到
`MAP_ANONYMOUS` mapping，复制前后复验 fd 的 device/inode/size/mtime/ctime，随后 `mprotect`
为 read-only。Synthesizer copies 共享这份 immutable snapshot，成功初始化会让约 checkpoint
文件大小常驻 host memory；源文件后续 truncate 不会破坏 active bytes。初始化同时把调用方
路径固定为 absolute + lexically-normal 路径（不解析 symlink），公开调用若发现该路径身份已
改变或被替换仍返回空结果，且不受后续 current-working-directory 改变影响。该 anonymous
snapshot 是 VITS 专用选择，不代表 RMVPE/FCPE/WavLM 等消费者都复制权重。`infer`/`infer_stream` 及公开
`debug_*` 在读取调用方数组前检查各自最宽中间尺寸乘积，拒绝对应输入的 NaN/Inf，并在返回前
拒绝非有限的可观察输出；`infer` 还显式约束 attention 的 `2*T*T`。VITS 算子的临时 device buffer 由
`core::CudaBuffer` 管理。Synthesizer 的公开 init/infer/infer_stream/debug 边界捕获 host
allocation、model/I/O、CUDA、标准及未知内部异常并返回既定空/false 结果。legacy
`infer_stream` 仍实际执行完整 `infer` 后裁剪，repeated call
会重置 NSF phase；characterization 已冻结该事实，但它不是 stateful session。non-F0
合成器不是当前能力。

### `pipeline/`

`vc_batch` 先冻结输入列表并检查 flat 输出映射；不同输入映射到同一
`output-dir/<stem>.wav` 时在 CUDA/模型加载和任何输出写入之前返回冲突错误。
这覆盖递归目录同名和扩展名大小写冲突，保留原有非冲突文件的输出布局。

编排 HuBERT/WavLM、RMVPE/FCPE、可选 Index 和 Synthesizer，负责 protect、RMS、分块与拼接。
仅接受 CUDA FP32/F0/40 kHz 契约。`index_rate > 0` 时，Index 缺失、feature width 不匹配或
lookup failure 均显式失败，不静默旁路检索。

`convert_file_bounded` 是同步 seekable-WAV file-to-file API：稳定 16 kHz source spool、全局
double forward/reverse high-pass、一次一个 quiet core、独立 F0 windows、raw synth/RMS spool
和 incremental atomic writer 都不在内存保留全段音频。对 checked 16 kHz samples `C/Q/P` 与
source `N>=2`：

```text
Qeff = max(Q, 160)
Peff = min(max(P, 160), N - 1)
Ceff = min(max(C, 160), 640000 - (2*Qeff + 2*Peff + 160))
core <= Ceff + 2*Qeff
content/Index/VITS <= Ceff + 2*Qeff + 2*Peff + 160
F0 input <= 128000 + 2*F
```

RMVPE/FCPE 的单侧 `F` 分别为 64000/32000，所以 F0 hard input 分别为 256000/192000，和
content segment bound 独立。caller policy 将支持的 HuBERT v1/v2 + RMVPE/FCPE 配置送入
bounded 实现；直接 bounded E2E matrix 是 RMVPE/v2 与 FCPE/v1，另有两种 F0 的独立 window
profile，因此没有把未直接执行的另外两个 Cartesian pairing 伪称为独立 E2E 用例。
`vc_convert`/`vc_batch` 对 HuBERT 使用该 API，对 WavLM 使用 legacy `convert_file`，因为仓库
没有匹配的 WavLM-trained VITS 资产可完成 bounded qualification。`convert_buffer` 和旧
`convert_file` 仍返回完整 owning audio、执行 legacy whole-clip F0，working set 随时长增长。
可恢复的 model/I/O/CUDA/runtime 异常以 `VCResult` 失败返回；持续 `std::bad_alloc` 在连结果
都无法构造时可传播。同一 pipeline instance 不是 thread-safe。

focused Orin/CC 8.7 Release resource gate 对 17.565/35.130 秒真实语音记录 peak RSS
`1342624`/`1354844 KiB` 与 device growth `892334080`/`851812352` bytes；这是两个时长点，
必须与 hard bounds 联合解释。临时磁盘仍为 `O(duration)`，且当前没有独立冻结的
click/transient 边界指标；这些限制不能被 `>0.98` phase-invariant spectrum gate 掩盖。

### `training/`

GAN 推理导出把 trained Flow 与 decoder 的权重合并到同一个 safetensors staging，再一次
rename 发布；WaveNet/decoder 扣除初始化时折叠的 speaker condition，fused weight norm
还原为 g/v（零行用 g=0、非零单位 v 表示）。失败保留旧文件；这仍不保存 enc_q/D/optimizer
训练状态，不支持 resume。模块内的 `export_weights.h` 只定义私有 staging 类型及转换，
不新增公共方法、持久化格式或运行依赖。

默认 training contract 覆盖 host/GPU 频谱在奇数 padding 差下的帧数一致性，以及 MelLoss
正参数、有限频段、尺寸乘积、输入/loss shape 校验。autograd 的 sqrt/log 裁剪分支及端点
采用零梯度，AdamW beta 和学习率更新拒绝非有限值；真实 GAN 导出由模型门禁复验。

存在 CUDA autograd、AdamW、mel loss、生成器微调、后验编码器、flow 和多周期判别器/GAN 路径。`vc_train` 可执行这些实验性训练路径，但尚无生产训练稳定性、数据规模或音质保证。公共 `FineTuneTrainer` 门面尚未实现，必须返回不支持，不能用空循环伪装成功。

### `separation/`

包含 STFT/iSTFT、Open-Unmix、MelBand-RoFormer 和 Silero VAD 相关实现。模型和数值参考资产不进 Git，因此相应测试只在本地资产齐全时注册。该模块不是 `vc_convert` 默认主链的一部分。
三个模型 loader 在 device upload 前校验所有已用 tensor 的 F32 dtype、精确 shape/nbytes 和
有限值；公开音频入口拒绝 NaN/Inf，可观察 host 输出在返回前复验有限值。该边界不等于已对
每个 CUDA 中间 activation 做全设备扫描，也不构成生产音质保证。

### WebUI / `vc_serve`

浏览器认证只共享尚未完成的 token 提示 Promise，结束后清除；首次输错/取消或服务端
token 轮换后可再次输入。token 仍限于 tab 的 sessionStorage/内存；默认开放模式不提示。
Node 的 mock fetch 测试覆盖并发提示、后续重试、轮换及 storage 不可用，不等同于浏览器视觉验收。

前端是无构建步骤的 HTML/CSS/JavaScript。`vc_serve` 使用 POSIX socket 提供静态文件与
API，并以子进程调用 CLI。内部进程适配器用 argv 直接启动程序，不经过 shell；它合并并
有界过滤 stdout/stderr，同时独立保留进程的退出码、终止信号和日志 I/O 错误。训练工作流
逐阶段检查 preprocess、train 与 index 结果；clips、model、index 先写入同目录 staging，
只有有效 WAV clips 与非空 model/index 齐全时才成组发布并报告 `done`，失败保留旧产物。
material 上传后的 raw/work 副本不代表已经完成 step；只有非空 `steps.txt` marker 与有效 work
WAV 同时存在时训练才复用 work，否则从 raw 开始并执行请求的 separation/cleanup flags。
material/train 上传把 raw/work 替换与旧 `steps.txt` 退役纳入同一个 backup/rollback 事务；
非常规 marker（包括 symlink）在移动原素材前拒绝。material、step、training 的备份均在
移动 live 产物前启用无分配、noexcept retention guard；只有成功提交或完整回滚才允许
TempDir 清理。错误文本在回滚之后构造；中途异常保留备份路径及字节供恢复，不能只依赖
HTTP 层 catch。异常保护保证可恢复副本，不保证原路径已经恢复或持续可用；不自动恢复。
运维从服务 stderr 中的 `transaction incomplete; backup retained at ...` 定位备份，先暂停
对应素材的写入并确认相关作业结束，逐项核对 live 与 backup 中的 raw/work/marker（或对应
step/训练产物）；先将保留字节复制到独立恢复目录、确认属于同一旧代且内容完整，再由
维护者安排恢复。不得盲目覆盖 live 或删除唯一备份。默认 `test_web_transaction_failure`
隔离进程在失败 rename 后注入一次
分配失败，验证三种事务的旧产物在 live 或 backup 中完整保留。一个实例内的 material mutex 同步
material/train/step 发布与 preview 的文件选择、校验及最多 64 MiB 的完整读取。写路径先取得
GPU mutex 再取得 material mutex，preview 只取后者；不在 CLI 执行或 socket 发送期间持有
material mutex，异常由 RAII 解锁。锁内包含文件读取和提交/回滚及 backup 清理，因此没有
硬性时间上限；它不保证外部进程修改或进程崩溃时多次 rename 的原子性。
监听 socket 与连接 socket 均为 close-on-exec，CLI 不继承 HTTP 文件描述符。监听地址由
`--host` 配置，默认 `0.0.0.0`；可选
`--auth-token` 保护 API。服务限制请求头、请求体、上传大小和并发数；已知 API 的错误
方法返回 405/`Allow`，未知 API 返回 404。交互式 step 在 GPU busy 时立即 409，成功结果
把 work 与 `steps.txt` marker 作为一组经 staging/backup 发布，任一提交失败都保留旧 work
和 marker。内部由 `tools/web/http_transport.*` 负责
socket、HTTP、鉴权和资源 reservation，`workflow_support.*` 负责文件/进程适配器，
`web_service.*` 负责路由与工作流；job/GPU 状态归单个 `WebService` 实例所有。公网部署
仍需要反向代理 TLS 和网络访问控制。step 的 promise/future 握手与 shared ownership
准备放在启动异常处理范围内；train 的 job metadata 初始化独立捕获失败，并把
shared ownership 准备纳入原 worker 启动处理。未交给 worker 的初始化异常释放预留并
返回 503，已接手 worker 的响应生成保持在启动 catch 外，避免响应失败错误地清除活动
作业。默认 `test_web_job_lifecycle` 使用隔离的请求线程单次分配故障与假 CLI 验证释放、
重试、终态及 staging 清理；内部 friend 只供测试读取 running 来定位注入窗口，没有
生产故障回调或新运行时依赖。
train 上传发布的 try/catch 只包住锁与提交；是否失败在 catch 后统一判定，只调用一次
finish，再构造错误响应。释放后响应异常直接交给 HTTP 错误边界，不重新释放可能已归
后续请求的共享 slot。没有引入新的 job token/调度架构，也不改变 GPU mutex 串行执行。

Request 的 body reservation 与 ConnectionGuard 的 socket/连接计数仍由原 RAII 释放。
新增真实 HTTP 门禁只用短 body 和较大的声明长度验证 128 MiB 上限与 256 MiB 总预留、
EOF/RST 后恢复及 accepted socket 关闭；33 次串行截断 EOF 验证连接名额可复用。
最多三个客户端、实际 body 不超过 215 bytes；不是吞吐/压力、完整大上传或 slow-client
超时验收，不需改变现有 10/120 秒 deadline。
另有真实10秒header deadline门禁：第0/3/6秒滴流不续期，400后关闭socket、job与CLI不变，
后续请求可用；53 bytes慢header、最多2客户端、用例<15秒。独立D2随后验证120秒body绝对deadline：声明1024 bytes、实际216 bytes，
0/20/40/60/80/100秒滴流仍在120708 ms后400/EOF；socket关闭、后续查询可用。
其CTest上限130秒，通过默认OFF的VOXMUTATIO_ENABLE_SLOW_WEB_TESTS单独注册，
不改变当时默认18项快速套件或生产逻辑。
`test_web_process_contract` 覆盖原runner的空argv、缺失程序、日志打开失败及自有子进程
自行SIGTERM，检查结果分类、输出、FD/直接子进程回收及下一次成功调用。
该host门禁串行、最多1个helper、CTest上限10秒，不改变生产runner、等待/取消或进程组行为。

新增host门禁`test_web_result_contract`只使用既有ProcessRunner依赖测试替身，通过socketpair
观察实际HTTP响应writer；模拟signal、wait失败、exit0+io_error三类结果，覆盖7路径的失败/正常重试。
即使替身产物格式有效也须同步500或后台error，保留旧产物、不执行下游，等待worker结束后清理。
不新增生产seam、不启动真实子进程；默认快速套件现19项，该门禁CTest上限5秒。
所有 Web boolean field 只接受 `0` 或 `1`；显式空值
或其他字符串返回 400。preview/preprocess/convert 的音频响应当前在发送前
完整缓冲，硬上限为 64 MiB；超过时明确返回 413，不会以 200 返回空 body。
`/api/convert` 只有在 `index_rate > 0` 且选定声线存在非空 Index 时才向 `vc_convert`
传递 `--index/--index-rate`；正数 rate 缺 Index 会在启动 CLI 前返回 400，`index_rate=0`
则显式不传这两个 flags，即使该声线目录中已有 Index 也保持 no-index 模式。

multipart transport 按参数/quoted-string 语法提取唯一 boundary，拒绝重复 boundary、未闭合
引号/转义及尾部垃圾，不从其他参数的值中查找 boundary。完整 delimiter line 才终止 part，
二进制 WAV 内相似前缀保持原字节。边界解析失败先返回 400、字段或 part 数超限返回
413，material handler 不进入 staging/commit；真实 HTTP contract 验证旧 raw/work 与
step marker 保留，并覆盖 16 KiB field / 128 parts 的准确上限。

part header 逐行验证并展开 MIME 续行，仅从唯一 Content-Disposition 的 form-data 参数中
取得 name/filename；无关头、参数后缀、quoted text 或注释不能提供属性。合法 token/quoted
值、转义、折行和注释得到语法处理；重复 disposition/name/filename 按歧义拒绝策略返回
400，畸形头/参数及 filename* 同样拒绝。跨 part 重复字段名保持允许。扫描受原有 64 KiB
part header 上限约束，注释深度迭代计数、不递归；没有增加运行时依赖。

历史公共头 `voxmutatio/webui/server.h` 的嵌入式 `WebServer` 从未实现监听器，也不是上述
服务的库形式。按 ADR-0003，它只在一个迁移窗口内保留 deprecated 源码/静态链接表面；
`start()` 明确 fail-fast 并引导调用方迁移到 `vc_serve`，不再忙循环或返回伪造 API 成功。

### 训练取消与迁移（WEB-FR-011）

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
[ADR-0006](adr/0006-training-cancellation-ownership.md)。

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


## 4. 权重与数据

运行时权重格式是 safetensors。原始 PyTorch `.pth` 使用一次性 Python 工具离线转换；不要在 C++ 服务中加载 pickle。

模型权重和数值参考 fixture 体积较大且被 `.gitignore` 排除。由此形成两层测试：

1. 默认 `contract`：不依赖这些本地资产，CI 构建并执行；除低层契约外还包含真实 CLI
   子进程与真实 HTTP socket 测试。
2. 可选 `integration`：配置 `VOXMUTATIO_ENABLE_MODEL_TESTS=ON`；只有所需资产存在的用例才进入 CTest。

测试内部“找不到文件就 return”不是可信的通过证据，因此 CMake 在注册前检查资产；WavLM
还在运行时检查完整参考 shape/payload、双方有限值及全元素 RMSE `<1e-4`。
Node.js 18+ 可用、slow=OFF 时默认27项、资产齐全51 distinct（28 contract-labelled、24
integration-labelled、1 overlap）；slow=ON 加1项真实 body deadline，共52项。缺 Node
时 UI 认证/取消两项明确不注册，数量减2；Node 只用于测试，不是 WebUI 构建/运行依赖。
当前执行结果见 `specs/PROJECT_STATE.md`。2026-08-12 的 fresh Release/SM87
build `/tmp/resonantia-nonrt-final.a8O1Ay` 已实际执行并通过 39/39（0 失败，
`Total Test time (real) = 1297.94 sec`）。加固前 Release/SM87 30/30（11 contract +
19 integration，356.41 秒）只保留为历史基线。新增门禁包括 bounded
I/O/DSP/resource/F0/caller、immutable Index/build-index、VITS snapshot 和代表性 CUDA
failure injection；注册或 focused pass 本身不是全量证据，该历史工作树 39/39
是对它们的独立收口。

CUDA fault seam 只编入私有测试 target，覆盖 RMVPE/VITS 代表性 allocation、H2D/D2H copy、
pre-launch、observable synchronization，以及 Synthesizer public fail-closed 边界；production
不导出 seam 符号，也不据此宣称每个 cuBLAS/cuFFT/CUDA 调用或真实系统 OOM 已被注入。

## 5. 构建原则

- CMake 3.24+，C++20/CUDA20。
- `VOXMUTATIO_CUDA_ARCHITECTURES` 决定目标架构，默认 87；不再额外硬编码一份冲突的 `-gencode`。
- Catch2 优先使用系统包，否则下载带 SHA-256 校验的固定源码归档。
- CI 只在 `main` push/手动触发时使用自托管 ARM64 CUDA runner，避免在自托管机器执行 fork PR 的任意代码。

运行时源码使用显式清单，按 core、I/O、CUDA ops、content、F0、Index、synthesizer、
separation、autograd、training、pipeline 和 deprecated legacy WebUI compatibility surface
组成内部 OBJECT
targets；公共 `voxmutatio` 仍聚合为一个兼容 static archive。CLI 不再内联全部解析与执行：
四个主要命令各自经内部 `resonantia_vc_*_app_support` 的 Options/parser/runner seam 装配，
`vc_serve` 经 `resonantia_web_support` 装配。上述内部 target 不新增安装或公共 API。

仓库当前没有 `install()`、导出 package config 或稳定的预编译 SDK 发行面；成熟消费方式是
源码树 CLI，嵌入方只能以 `add_subdirectory`/随源码重编译方式使用 `voxmutatio` facade。

## 6. 验证口径

历史本地对齐阈值仍可作为集成测试目标：

- HuBERT 特征 RMS 误差 `< 1e-4`
- RMVPE F0 绝对误差 `< 0.5 Hz`
- VITS 输出相关性 `> 0.999`
- 端到端输出相关性 `> 0.99`

只有在对应模型、输入、参考文件均存在且测试实际执行时，才能报告这些结果。默认 CI 通过仅说明可复现契约测试通过，不等价于全模型数值对齐已经在该次提交复验。


### 完整迭代协作取消（2026-10-10 后续授权，TR-FR-015 / WEB-FR-011）

本增量以固定源码v9和已复验的素材提交测试增量为基线。用户明确授权CLI编排/Web集成，
不扩展到训练数值、CUDA kernels、自动超时、强杀或checkpoint/resume。此前“CLI无循环内接口”
是v9历史边界；当前增加可选`vc_train --cancel-file <path>`，不带参数时不打开/轮询任何文件。
参数不可为空或重复。Web在每个唯一staging目录独占创建0600的一字节常规文件，初始`0`、请求后`1`；
CLI以CLOEXEC/NOFOLLOW/NONBLOCK只读打开一次，拒绝符号链接、非常规文件、长度/内容异常，
不重新打开替换路径。该文件不是认证凭据，也不持久保存任务。

CLI退出语义：正常完成0，错误1，协作取消2。首次检查在CUDA初始化之前；后续检查在已完成的
初始化阶段/clip边界、每个完整迭代之前和最终导出之前。GAN的D/G整对更新和原LR处理完整执行，
decoder的forward/backward/optimizer整步完整执行；共用私有`run_training_iterations`只包住编排，
不拆分数值步骤。当前步骤报告的错误先于并发取消。最后一步完成后也检查，取消不开始新导出。
导出已开始则完成：直接CLI此时可能正常返回0并替换自己的`--out`；Web的`--out`始终是staging，
后续checkpoint会丢弃该结果并保留已发布模型。最终三产物发布已开始仍409且完整收尾。

Web只有在对应job确实请求取消、vc_train正常退出2且没有system/log/signal异常时确认cancelled；
不更改`ProcessResult::success()`或其他CLI的非零退出语义。专属source/lease的request与close互斥，
close等已开始写操作结束，关闭后旧shared_ptr不能写或重建路径。lease在runner自然排空日志/reap之后
关闭并解绑，再清理staging，最后原owner释放GPU锁/作业预留。注册令牌时重查本代取消，覆盖注册前竞争。
取消写失败返回500且保持预留；source记录首个写/close错误，最终error优先。新作业从新id、新token开始。
`cancel_boundary`更新为`stage_or_iteration`；收到请求仍不等于已经停止。

验证分层：`test_training_cancel`覆盖默认/参数/异常token、完整fake step、最后一步取消、错误优先、
关闭/旧路径隔离，以及真实CLI的预取消入口（隐藏GPU、输入不存在）。同一测试的微型host helper使用
共享loop helper，由真实PosixProcessRunner自然排空/reap；这不是GPU训练。`test_web_train_cancel`
沿用ProcessRunner替身检查token送达、重复请求、runner返回前busy、旧id/新token隔离、退出2未请求即错误、
并发system/log/普通错误优先、清理及重试。`test_web_train_cancel_commit`保留素材中途提交控制/取消门禁。

证据限制：host编排与编译成功不证明真实GPU循环的取消延迟、资源峰值或数值等价；没有启动真实训练。
单次初始化/clip/step/export或底层驱动卡住时仍无停止时限。token写/close的OS故障分支未做系统调用拦截验证；
既有GPU析构的cudaFree错误仅日志记录、TempDir清理失败及进程异常guard的旧行为不在本增量修复范围。
取消丢弃进度，没有resume；未改变页面刷新/提交响应丢失后的跟踪限制，也未确定性命中最终发布窗口。
正式ADR仍Proposed，Spec003仍部分验收。当时原A/B/C和其他既有未验证项保持未完成；
后续普通host等价所有权验证见下文，原受限探针仍未重跑。

本轮验证结果：首个真实CLI预取消红测失败（旧参数不支持）；实现后3个focused门禁通过，0.25秒。
补充自然host helper后，明确选择8项串行回归8/8通过、0失败/skip，3.38秒；当前注册22项，
未运行10项GPU、`test_web_contract`/`test_web_process_contract`（使用信号）、
`test_web_transaction_failure`（rename拦截）及无关`test_realtime_benchmark_report`，不称全量通过。
SDD32项、git diff --check、Node mock UI门禁通过；vc_train/vc_serve已编译，未部署。

### CLI普通异常退出边界：独立审查修复（2026-10-10）

独立审查发现原`tools/vc_train.cpp`没有catch边界：GAN finite-loss、optimizer/CUDA的普通C++异常
可外逸到terminate；此前“所有错误退出1/普通栈展开”的描述过宽。Web把异常进程归error本身正确。
现由main的try覆盖options/parse/run，catch标准及未知异常，普通栈展开完成后返回1；正常返回0、
协作取消2保持。错误诊断用noexcept小函数并捕获诊断自身异常，诊断失败仍返回1，绝不转为成功或取消。
未改变run_training_iterations、训练器、核函数或Web流程。

`test_vc_train_entry`把相同生产main链接到host应用替身，替身抛标准/未知/解析/body异常；
测试专属terminate handler使用_Exit(86)观察旧缺陷，避免SIGABRT/崩溃转储，没有生产测试hook。
红测旧入口返回86而非1；修复后8个case验证标准/未知/parse/body、诊断失败、正常0/取消2，
marker断言普通栈对象恰好展开一次。取消后body抛异常仍1。加既有CLI及取消门禁3/3通过、3.18秒。
这不是在真实CUDA上制造异常；不覆盖noexcept析构异常、信号/崩溃、硬件/驱动致命故障、
静态析构失败或所有GPU错误后的释放保证。原209文件审查包保持固定，本修复另存增量。

### 作业所有权与延迟响应：普通host门禁（2026-10-10）

`test_web_job_ownership`补齐此前A/B/C所有权与重复finish的动态证据。它以私有宏
`RESONANTIA_TEST_JOB_OWNERSHIP`单独编译Web实现，在训练接受响应和上传拒绝响应前设置
显式有界屏障；普通服务不定义此宏，没有回调或运行时开关。测试friend仅重放旧generation
的training/step完成回调，不改对象布局。普通`web_service.cpp.o`、支持库和`vc_serve`
重编译后SHA-256与本增量前相同，宏关闭对象中没有测试回调符号。

四种确定性交错为“自然上传校验拒绝/worker先于响应结束”分别配合“正常响应/普通C++响应异常”：
前者使用目录型`steps.txt`，在素材移动前被既有校验拒绝；A响应及staging收尾期间B/C均409。
后者让A的替身worker自然结束后接受B，再释放A的延迟响应；正常响应仍报告A的id，异常展开
不修改B。B的替身worker保持活动时，重复两次重放A的training和step finish，B的状态/id
保持不变且C仍409；B自然完成后C可重试。全程核对旧raw/work/model/index、staging清理和FD数。
替身在首个preprocess阶段返回错误，不执行任何CLI、模型、CUDA或训练；屏障等待上限2秒，
CTest上限15秒，没有rename/pthread拦截、进程注入或信号操作。

缺少响应接缝的测试先因无法观测屏障而失败，这仅验证观测能力，不代表发现生产缺陷。
随后4/4交错通过；临时源码负对照分别移除training/step的generation检查，均按预期检测到
旧finish修改B。负对照不修改仓库实现。本轮10项明确选择的host回归10/10通过、0失败/skip，
3.43秒；当前注册24项，其余14项未运行。该证据闭合所列等价所有权语义，但不是原受限
OS故障探针的复跑或新的独立审查；真实OS异常生成、全局/worker OOM、预留前filesystem
terminate、真实GPU取消和最终三产物发布窗口仍未验证。复现范围与命令见验收计划§8。
