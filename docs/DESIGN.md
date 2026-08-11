# Resonantia 架构设计与实现边界

**状态日期：2026-08-12。** 本文记录当前实现，不把目标设计写成已经支持的能力。可执行入口和测试方法以仓库当前 `CMakeLists.txt`、CLI `--help` 与 CTest 注册为准。

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

存在 CUDA autograd、AdamW、mel loss、生成器微调、后验编码器、flow 和多周期判别器/GAN 路径。`vc_train` 可执行这些实验性训练路径，但尚无生产训练稳定性、数据规模或音质保证。公共 `FineTuneTrainer` 门面尚未实现，必须返回不支持，不能用空循环伪装成功。

### `separation/`

包含 STFT/iSTFT、Open-Unmix、MelBand-RoFormer 和 Silero VAD 相关实现。模型和数值参考资产不进 Git，因此相应测试只在本地资产齐全时注册。该模块不是 `vc_convert` 默认主链的一部分。
三个模型 loader 在 device upload 前校验所有已用 tensor 的 F32 dtype、精确 shape/nbytes 和
有限值；公开音频入口拒绝 NaN/Inf，可观察 host 输出在返回前复验有限值。该边界不等于已对
每个 CUDA 中间 activation 做全设备扫描，也不构成生产音质保证。

### WebUI / `vc_serve`

前端是无构建步骤的 HTML/CSS/JavaScript。`vc_serve` 使用 POSIX socket 提供静态文件与
API，并以子进程调用 CLI。内部进程适配器用 argv 直接启动程序，不经过 shell；它合并并
有界过滤 stdout/stderr，同时独立保留进程的退出码、终止信号和日志 I/O 错误。训练工作流
逐阶段检查 preprocess、train 与 index 结果；clips、model、index 先写入同目录 staging，
只有有效 WAV clips 与非空 model/index 齐全时才成组发布并报告 `done`，失败保留旧产物。
material 上传后的 raw/work 副本不代表已经完成 step；只有非空 `steps.txt` marker 与有效 work
WAV 同时存在时训练才复用 work，否则从 raw 开始并执行请求的 separation/cleanup flags。
监听 socket 与连接 socket 均为 close-on-exec，CLI 不继承 HTTP 文件描述符。监听地址由
`--host` 配置，默认 `0.0.0.0`；可选
`--auth-token` 保护 API。服务限制请求头、请求体、上传大小和并发数；已知 API 的错误
方法返回 405/`Allow`，未知 API 返回 404。交互式 step 在 GPU busy 时立即 409，成功结果
把 work 与 `steps.txt` marker 作为一组经 staging/backup 发布，任一提交失败都保留旧 work
和 marker。内部由 `tools/web/http_transport.*` 负责
socket、HTTP、鉴权和资源 reservation，`workflow_support.*` 负责文件/进程适配器，
`web_service.*` 负责路由与工作流；job/GPU 状态归单个 `WebService` 实例所有。公网部署
仍需要反向代理 TLS 和网络访问控制。所有 Web boolean field 只接受 `0` 或 `1`；显式空值
或其他字符串返回 400。preview/preprocess/convert 的音频响应当前在发送前
完整缓冲，硬上限为 64 MiB；超过时明确返回 413，不会以 200 返回空 body。
`/api/convert` 只有在 `index_rate > 0` 且选定声线存在非空 Index 时才向 `vc_convert`
传递 `--index/--index-rate`；正数 rate 缺 Index 会在启动 CLI 前返回 400，`index_rate=0`
则显式不传这两个 flags，即使该声线目录中已有 Index 也保持 no-index 模式。

历史公共头 `voxmutatio/webui/server.h` 的嵌入式 `WebServer` 从未实现监听器，也不是上述
服务的库形式。按 ADR-0003，它只在一个迁移窗口内保留 deprecated 源码/静态链接表面；
`start()` 明确 fail-fast 并引导调用方迁移到 `vc_serve`，不再忙循环或返回伪造 API 成功。

## 4. 权重与数据

运行时权重格式是 safetensors。原始 PyTorch `.pth` 使用一次性 Python 工具离线转换；不要在 C++ 服务中加载 pickle。

模型权重和数值参考 fixture 体积较大且被 `.gitignore` 排除。由此形成两层测试：

1. 默认 `contract`：不依赖这些本地资产，CI 构建并执行；除低层契约外还包含真实 CLI
   子进程与真实 HTTP socket 测试。
2. 可选 `integration`：配置 `VOXMUTATIO_ENABLE_MODEL_TESTS=ON`；只有所需资产存在的用例才进入 CTest。

测试内部“找不到文件就 return”不是可信的通过证据，因此 CMake 在注册前检查资产。当前
资产齐全配置注册 39 个 distinct tests：16 contract-labelled、24 integration-labelled；
`test_bounded_file_wavlm_preflight` 同时属于两类。2026-08-12 的 fresh Release/SM87
build `/tmp/resonantia-nonrt-final.a8O1Ay` 已实际执行并通过 39/39（0 失败，
`Total Test time (real) = 1297.94 sec`）。加固前 Release/SM87 30/30（11 contract +
19 integration，356.41 秒）只保留为历史基线。新增门禁包括 bounded
I/O/DSP/resource/F0/caller、immutable Index/build-index、VITS snapshot 和代表性 CUDA
failure injection；注册或 focused pass 本身不是全量证据，本次当前工作树 39/39
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
