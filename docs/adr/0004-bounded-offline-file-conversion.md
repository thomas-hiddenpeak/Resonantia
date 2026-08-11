# ADR-0004: 有界离线文件转换与兼容 buffer 路径

**Status**: Accepted
**Date**: 2026-08-12
**Accepted**: 2026-08-12（维护者批准 additive API、保留旧 API 的分阶段方案；R1–R4
已实现并通过各自 focused 门禁，R5 已按“HuBERT bounded / WavLM legacy”策略切换 caller；
当前树的 Release/SM87 最终全量 CTest 已 39/39 通过，该实证与 ADR 状态分开记录）
**Owners**: Resonantia maintainers
**Governing specs**: [Spec 001](../../specs/001-voice-conversion-engine/spec.md)、
[ADR-0001](0001-modular-runtime-and-orchestration-boundaries.md)、
[重构章程](../REFACTORING_GUIDE.md)

## Context and decision drivers

迁移前，Spec 001 的离线长音频路径只约束 HuBERT、Index 与 VITS 的单次调用。旧
`convert_file` 先由 `io::read_audio` 读取、解码、下混并重采样整份 WAV，再调用
`convert_buffer`；后者又复制 16 kHz 输入。`convert_long` 同时保留整段高通输入、RMS
参考、reflect-pad、F0/UV、切点能量与完整 40 kHz 输出。RMVPE/FCPE 也对整段 padded
clip 一次推理，因此其 CPU activation、CUDA allocation 与临时输出均随时长增长。

2026-08-12 在本地 Release/SM87 完整资产环境运行迁移前约 18 秒
`test_long_audio_chunking`，测试通过且得到 `spec_corr=0.9819`；以 `/proc/<pid>/status`
每秒采样得到进程峰值 RSS `2527780 KiB`。该单点只冻结当前资源问题的可复现基线，
不是未来上限，也不能证明内存随时长的具体斜率。

`AudioBuffer` 与 `VCResult::audio` 都是 owning `std::vector<float>`。因此接收完整
`AudioBuffer` 并返回完整 `AudioBuffer` 的 `convert_buffer` 不可能承诺与时长无关的 host
内存；现有调用方还可观察 `convert_file` 成功返回的完整 `VCResult::audio`。直接清空其中
任一结果会破坏现有源码契约。另一方面，CLI/批处理只检查 file conversion 的成功、错误和
timing，并不消费返回音频，所以可以先通过 additive file-to-file API 迁移实际离线工作流。

本决策只处理 Spec 001 的 seekable WAV 离线转换。它不创建 session、队列、实时线程、
ALSA 适配器或低延迟承诺，也不修改 Spec 006/ADR-0002。

## Decision

### 1. Additive file-to-file contract

新增公共方法：

```cpp
VCResult convert_file_bounded(const std::string& input_path,
                              const std::string& output_path,
                              int speaker_id = 0);
```

该名称明确表示离线资源契约，不把它描述成 realtime/streaming session。成功时它原子提交
目标 WAV，保留 `success`、错误与四阶段 timing；`VCResult::audio.data` 必须为空，
`audio.sample_rate` 记录模型原生采样率。调用方以目标文件和 `success` 判断 file delivery。
输入仍为 seekable WAV，输出仍为当前 40 kHz PCM16 WAV，其他格式和现有不支持配置继续失败。

现有 `convert_buffer` 保持签名、完整返回音频、全段 F0 和当前长音频拼接语义；它被明确记录为
owning/buffered API，不获得恒定内存承诺。现有 `convert_file` 在至少一个明确迁移窗口内保持
当前签名、完整返回音频和数值行为。CLI 与 batch 切到新入口且门禁通过后，旧 file API 才可
标记 deprecated。是否以后为旧入口增加硬上限、改为空结果或删除，均需新的显式批准；本 ADR
不选择任意秒数作为兼容上限。

### 2. Private bounded WAV seam

在 `io` 组件内加入非公共 RAII seam：

- seekable WAV reader 只解析一次 RIFF/fmt/data 元数据，以 checked `pread` 分段解码、下混，
  并暴露 native frame metadata/range；后续 planner 以全文件 sample index 锚定线性重采样
  相位。open 和每次 read 前后都以 fd `fstat` 核对
  device/inode/size/mtime/ctime，short read 或身份/长度变化立即失败；单次 encoded/float buffer
  受分析窗口上限约束；
- unique temporary spool 以 mode `0600` 创建，拥有明确文件描述符、路径与清理责任，用于
  必须跨 pass 保存的数据；
- atomic incremental WAV writer 在目标同目录创建唯一 staging，先写占位 header，逐块量化并
  append；WAV staging 沿用现有 writer 的 mode `0666` + caller umask 兼容语义。结束时 checked
  backpatch 长度与文件 `fsync`/close 必须先成功，随后 rename 是不可回滚的 runtime commit
  point；rename 后父目录 `fsync` 仅 best-effort，不能因其失败把已提交结果改报失败。本 ADR
  保证运行时原子发布和 rename 前失败保留旧目标，不宣称目录 namespace 的完整 crash
  durability；输入与输出为同一路径时也只在全部读取和写入成功后替换。

这些类型不进入 `include/voxmutatio/` 的稳定公共面。pipeline 只依赖该窄 io seam，不复制 WAV
解析或文件事务代码。临时磁盘和最终文件仍为 `O(duration)`；承诺的是进程拥有的 host/GPU
working set 与时长解耦，而不是零临时存储。

### 3. One-core-at-a-time offline planner

第一阶段把源 WAV 分段 decode/downmix，并以全文件相位线性重采样到一份稳定 16 kHz raw-float
spool；完成前后复核源 fd 身份与 size，之后所有 pass 只读该不可变 spool，不再重复读取可能被
替换或 truncate 的源路径。以 16 kHz 全局 sample index 计算输出长度、nominal cut 与 resampling
mapping，所有乘加和 byte count 在 narrowing 前检查。

令 `C`、`Q`、`P` 分别为三个配置秒数经 checked conversion 得到的 16 kHz sample 数，`N`
为稳定 source sample 数，`L=640000` 为当前 content 输入上限。生产实现使用：

```text
Qeff = max(Q, 160)
Peff = min(max(P, 160), N - 1)
Ceff = min(max(C, 160), L - (2*Qeff + 2*Peff + 160))
```

`N < 2`、固定开销达到 `L` 或 `Ceff < 160` 均 fail-closed。quiet-point 查询只保留有限窗口；
planner 一次只保留前一切点和下一核心段，
不保存随时长增长的 audio、energy 或 bounds vector。每个核心段必须取得进展且
`core_samples <= Ceff + 2Qeff`；交给 content/Index/VITS 的 16 kHz segment 必须
`<= Ceff + 2Qeff + 2Peff + 160`。无法满足时 fail-closed，不能生成超限段。

现有 Butterworth high-pass 是递归 IIR，任意有限 halo 都不能保证与全段 `filtfilt` 等价；
因此必须用 double 精度的全局 forward spool 与 reverse spool 保持递归状态，而不是逐窗口
近似滤波。VITS 的每块 raw float output 先顺序写 spool；完整 output length 和 source/output
RMS envelope records 已知后，再以有界顺序 pass 应用全局 RMS 并写 PCM staging。RMS records
也顺序落盘和消费，不在内存保存与时长成比例的数组。最后才 backpatch/fsync/rename/目录
fsync，保持绝对输出位置、补零和 runtime atomic publish 语义。

### 4. Bounded F0 windows

RMVPE/FCPE 当前公共 `infer(audio, num_samples)` 会按完整输入建立 CPU/CUDA 中间量。RMVPE
含双向 GRU；FCPE input stack 的 GroupNorm 在时间维参与 group statistics。两者都不存在一个
可证明与全段推理逐样本等价的有限 halo。新 file API 因此明确采用 profile-specific F0 context
的数值迁移：每个 window 只生成 raw F0/UV，把 crop 后核心帧按全局 160-sample lattice 顺序写
spool；全部 raw frames 完成后，再以有界、必要时多 pass 的全局处理完成跨任意长 unvoiced run
的 interpolation、有限半径 median filter 和 pitch shift。不得先在各 window 内插值/滤波再 crop，
否则会把人为 window edge 当成音频边界。

令 `F` 为经 profile 资格化的单侧 F0 context samples。F0 planner 使用独立的固定 raw core，
不把 content 的 `Ceff/Qeff/Peff` 叠加到 F0 上界；单次 F0 输入必须
`<= 128000 + 2F`，所有 crop 必须落在同一全局 160 lattice。因此 RMVPE 的当前硬上界为
`256000` samples，FCPE 为 `192000` samples。HuBERT、Index 与 VITS 沿用逐核心段执行和
绝对输出位置拼接。bounded API 当前只资格化 HuBERT profiles；WavLM bounded 调用在 payload
读取和 staging/spool 创建前 fail-fast，而 `vc_convert`/`vc_batch` 对 WavLM 明确保留旧
`convert_file` 路径，不能把 WavLM 静默改成 HuBERT。

单次 F0 输入、host activation、device buffer 与 synth piece 的上限只由已验证配置、模型和
固定 halo 决定，不得依赖总文件时长。Index mmap/模型权重可依模型或索引大小常驻，但不能因
处理更多音频块而复制或累计。每块结束必须释放 operation-owned scratch；不得把全部 F0/UV
或全部 float output 留在 vector 中。

由于上述 BiGRU/GroupNorm 全局依赖，即使 context overlap 正确，窗口化 F0 也不保证与旧全段
结果逐样本相同。这是 additive API 的显式数值迁移，不改写 `convert_buffer` reference。真实
语音门禁必须分别资格化 RMVPE/FCPE 的 profile context，并比较新 file output 与旧
buffer/chunk output 的采样率、长度、有限值、相位不变频谱和块边界连续性；不得以只检查
shape 或 WAV 可读代替，也不得把 `<0.5 Hz` 宪章门槛写成未经测量即可保证。

维护者于 2026-08-12 冻结首轮实现 profile：raw F0 core 为 `128000` 个 16 kHz samples
（8 秒）；RMVPE 单侧 context 为 `64000` samples（4 秒），FCPE 单侧 context 为 `32000`
samples（2 秒）。资格化必须把 crop 后 window 结果与同一真实语音的 whole-clip reference
对齐到相同全局 160-sample lattice，并分别满足：双方均 voiced 帧的 mean absolute F0 error
`<0.5 Hz`、p95 absolute F0 error `<0.5 Hz`、全帧 UV agreement `>0.95`、内部 window
边界双方均 voiced 帧的 max absolute F0 error `<0.5 Hz`。全局 max F0 error 只作诊断，不是
通过/失败门槛。

这些 profile 与指标最初来自 2026-08-12 本地真实语音、真实模型资产的探索性测量；随后已在
最终 bounded 实现上分别重跑 RMVPE/FCPE 数值门禁，并完成短/长资源 gate，才进入 R5 caller
切换。该事实不把两个时长点扩大成任意时长实测保证。

### 5. Failure and concurrency semantics

新入口在读取大 payload 前检查 initialized state、speaker、路径、WAV header、输入/输出长度与
配置。它保持现有同步、单操作调用语义；本 ADR 不承诺同一 pipeline 实例并发安全，也不引入
共享 mutable scratch。任一 decode/resample/F0/Index/VITS/spool/write、rename 前 file sync/
close 或 rename 失败均返回 `success=false`、非空错误，清理临时文件并保留旧目标；不得把部分
staging 当作结果。rename 成功后 runtime publish 已提交，父目录 best-effort fsync 失败不改报
失败，也不声称 crash-consistent namespace durability。

## Alternatives considered

- **直接改变 `convert_file` 成功时让 `VCResult::audio` 为空**：能最快降低 CLI 峰值，但会破坏
  已有调用方可观察行为。除非维护者另外明确批准不兼容迁移，否则拒绝。
- **让 `convert_buffer` 变成 borrowed chunk callback 或空结果**：与 owning input/result 契约
  不兼容，也会影响现有数值 reference；拒绝。未来若需要 callback sink，应另加 API。
- **旧 `convert_file` 立即增加任意固定秒数/字节上限**：能避免极端 OOM，却在没有产品输入
  上限依据时拒绝以前可表达的文件；本迁移窗口内拒绝，后续需单独批准和错误契约。
- **只在返回前清空整段 vector，或只把输出 writer 改成 append**：峰值已经发生，F0 和输入
  仍随时长增长，不能满足资源契约；拒绝。
- **把全段 F0 写入 host/disk 后仍一次调用 RMVPE/FCPE**：输出存储可转移，但模型 activation
  与 device workspace 仍无界；拒绝。
- **复用 Spec 006 persistent runtime/session**：会把 offline 文件事务与 realtime 队列、
  backpressure、ALSA/SLO 混在一起；范围和证据不同，拒绝。
- **完全重写 F0 为可精确分解的 stateful 网络**：可能改善边界，但改变模型算法和权重兼容，
  不是本轮最小迁移；保留为独立研究。

## Consequences

HuBERT CLI/batch 的长 WAV 转换已把 process-owned host 与 F0 CPU/GPU 峰值约束到配置窗口和
模型/Index 大小，不再由总文件时长决定；atomic output、失败保留和 buffer reference 仍可回归。
外部 library consumer 可逐步选择新 API，旧源码不会在同一阶段被强制改写。

代价是增加 seekable reader、spool、incremental writer、多 pass DSP 和清理故障面；临时磁盘
占用与 I/O 随时长增长。新 file API 的 F0 在窗口边界不保证与全段逐样本相同，必须用真实
语音和现有门槛资格化 context/crop，不能把预期差异写成无条件“等价”。旧 buffer/file API
在迁移窗口内仍可能因超长输入耗尽资源，文档和调用方必须明确这一点。

## Migration and rollback

1. **R0 / RED（已完成）**：冻结 current `convert_buffer` 长音频长度/频谱 reference、atomic output 与
   现有 RSS 诊断；加入要求 additive `convert_file_bounded` 的失败测试。本阶段不改核心实现。
2. **R1 / private I/O seam（已完成）**：加入 bounded reader、raw-float/raw-double spool 与 incremental
   atomic writer；先以无模型 I/O 测试覆盖 fd identity/truncate、overflow、同路径、
   append/backpatch、file fsync、rename commit point、best-effort directory fsync 与清理。
   reader/writer/spool 不依赖 CUDA。
3. **R2 / bounded planner（已完成）**：一次处理一个 core，强制 segment hard bound，完成 stable 16 kHz
   spool、全局 double high-pass、raw synth/RMS spool 与 transactional PCM commit。
4. **R3 / bounded F0（已完成）**：按冻结 profile 生成/crop raw F0/UV，再做有界全局处理；在最终实现上
   用真实 RMVPE 与 FCPE 重跑冻结数值指标。
5. **R4 / additive API（已完成）**：实现新入口并保持旧两入口不变；验证成功结果不拥有音频、目标文件
   完整、失败保留旧文件，且 short/long 输入的峰值差只在预先记录的 allocator/driver slack 内。
6. **R5 / callers（已完成）**：`vc_convert`、`vc_batch` 的 HuBERT v1/v2 profiles 通过共享
   policy 进入 bounded API；WavLM 因没有匹配的 WavLM-trained VITS 资格资产而保留 legacy
   `convert_file`。Web 继续通过 CLI 获得相同行为。旧 `convert_file` 尚未删除，也不因 caller
   切换而改变 owning 结果。
7. **R6 / SDD（已完成）**：同步 DESIGN、TRACEABILITY、PROJECT_STATE、API/CLI 文档和
   focused 验证记录；Release/SM87 合入门禁已执行当前注册的 39 项并全部通过。

每阶段都可回滚到仍使用旧 `convert_file` 的 CLI；新 API 没有持久格式，回滚无需转换用户
音频。出现旧 `convert_buffer` reference 回退、输出长度/频谱门槛失败、峰值随时长增长、临时
文件泄漏、旧目标被破坏或 CTest 注册减少时停止切换并回滚该阶段。不得删除用户输入或用
hard reset 恢复工作树。

## Verification and SDD impact

- **RED/API**：R0 时 `test_long_audio_chunking` 先要求新 additive 方法存在并按预期失败；当前
  tree 已由成功返回空 owning result、完整原子 WAV 和旧 API compatibility 用例把红灯转绿。
- **数值**：同一份真实人声、同一模型/config 下，新 file output 与旧 long-buffer reference
  保持 40 kHz、有限、非空和既有长度门槛；相位不变 magnitude-spectrogram correlation 不得
  低于现有 `0.98` 门槛。RMVPE/FCPE 还必须在最终实现上分别重跑冻结 profile 的
  both-voiced mean/p95、UV agreement 和内部边界 voiced max 门禁；global max 只记录诊断。
- **资源**：测试记录 window/core/F0 最大输入的推导值；隔离进程比较相同配置的短/长文件
  峰值 RSS，并记录 host/device 峰值、硬件、命令和 allocator/driver slack。单个 18 秒 RSS
  数字不是完成证据；至少需要多时长样本证明峰值不随总时长增长。
- **I/O/failure**：默认无资产测试覆盖 WAV 分段边界、算术、staging/backpatch、异常清理和
  失败保留；模型 integration 覆盖多块成功、Index fail-closed 与真实输出。
- **兼容**：旧 `convert_buffer`、旧 `convert_file`、CLI flags/退出码、WAV 格式、CTest 名/
  label/asset gate 不得在 additive 阶段回退。
- 2026-08-12 的 focused resource gate 在 Orin、compute capability 8.7、Release、相同
  HuBERT/RMVPE/v2 config 下比较 17.565 秒与 35.130 秒真实语音：peak RSS 分别为
  `1342624`/`1354844 KiB`（增加 `12220 KiB`，门限 `65536 KiB`），device growth 分别为
  `892334080`/`851812352` bytes（长样本未增加），并记录 RMVPE 单次 F0 硬上界
  `256000` samples。该证据只有两个时长点；它与代码硬上界共同支持当前 qualified profile，
  不能外推为任意时长或 WavLM 的实测证明。测试尚无独立冻结的 click/transient 边界指标。
- 各阶段实施后同步 Spec 001、`docs/DESIGN.md`、`specs/TRACEABILITY.md`、
  `specs/PROJECT_STATE.md` 与公开 API/CLI 文档；focused 门禁不能替代全量。
- 2026-08-12 的 fresh Release/SM87 build `/tmp/resonantia-nonrt-final.a8O1Ay` 已执行
  39/39（0 失败，`Total Test time (real) = 1297.94 sec`），完成当前树的独立全量收口。
