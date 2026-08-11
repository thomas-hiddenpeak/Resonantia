# Research Record: Realtime Voice Conversion

**Date**: 2026-08-10
**Status**: R0 evidence partially implemented for an approved design; no realtime capability is implemented
**Related**: [Spec](spec.md), [Plan](plan.md), [Data model](data-model.md),
[R0 asset manifest](r0-assets.json),
[ADR-0002](../../docs/adr/0002-persistent-streaming-runtime-and-local-audio-boundary.md)

## 1. Evidence boundary

编号治理说明：`005` 已在项目历史中 retired，仓库校验明确禁止重建；spec 编号一经使用
不得复用。因此本提案分配为 `006-realtime-voice-conversion`，所有活动交叉引用均使用
`006`。该记录仅解释编号分配，不恢复或覆盖 retired 005 的历史内容。

本记录描述 2026-08-10 current tree 的检查、legacy VITS characterization、资产复现、离线
E2E 诊断和 standalone exact-Index feasibility probe。它们不替代正式 session benchmark，也
不把名字包含 `stream`/`real-time` 的 helper 或模型论文特性当作产品证据。

维护者同日批准推进实时能力，并明确指出放弃 Index 会损害产品竞争力。因此 RT-DEC-002
已修订为 Index-enabled qualified profile；no-index 仅是显式诊断/兼容模式或启动时预授权的
粘滞 overload fallback。standalone probe 已提供 synthetic/available-feature exact retrieval
的 latency/memory 诊断，但该产品决定不改变目标资产缺口：当前仍没有代表性目标声线
model+Index 的 full-pipeline realtime latency/memory，或 Index-enabled 对 no-index 的质量基线。

当前事实来源：公共 headers、`src/`、CLI/Web 实现、CMake/CTest 注册，以及本记录中的实际
命令。Spec 001/003 明确把当前产品定义为离线 WAV/HTTP CLI 编排；Project State 也把整链
实时变声列为规划项。

## 2. Current API and state findings

### Pipeline and synthesizer

- `VoiceConversionPipeline` 只有 `convert_file` 和 `convert_buffer`；没有 session、push/pull、
  reset/drain、sequence/timestamp 或 backpressure contract。
- `Synthesizer::infer_stream` 调用完整 `infer(features, frames, ...)`，随后按 `skip_head` 和
  `return_length` 从完整 output 拷贝窗口。它没有跨块 cache/phase/workspace。
- `test_vits_stream_characterization` 已冻结：40 kHz 下每 frame 400 samples、full infer 后按
  window 精确 crop、非法窗口拒绝、重复调用逐样本一致，以及每次调用 NSF phase 从头重置。
  它仍没有连续 session、跨块 state、边界质量、延迟或 repeated-call memory assertion。

因此 `infer_stream` 是局部 full-infer crop helper，不是 realtime session；公共 header 已同步
这一事实。FCPE 名称或模型特性也不构成 end-to-end latency guarantee。

### Whole-buffer future dependencies

`convert_buffer` 当前包含：

- 5th-order Butterworth zero-phase `filtfilt` 和两端 reflect padding；
- 整个 buffer 的 F0、使用未来 voiced frame 的 unvoiced interpolation 和对称 median filter；
- 约 1 s window/0.5 s hop 的全段 RMS envelope；
- HuBERT bidirectional/global self-attention；
- RMVPE bidirectional GRU；
- 每次 VITS generator 调用从局部 frame 0 累计 NSF harmonic phase。

直接循环调用 `convert_buffer` 会改变边界、future context、相位和输出长度；长音频 silence
chunking 只约束 HuBERT/Index/VITS 的单次 working set；F0 仍整段执行，host 内存与 F0
CPU/GPU working set 都随时长增长，也不是低延迟接口。

### Runtime loading and CUDA execution

- HuBERT 与 WavLM 每次 infer 仍重新创建 safetensors loader 并 mmap/解析文件；RMVPE 已持久
  共享初始化时完整验证的只读映射，但其 CUDA activation 仍随整段 F0 帧数增长。
- Synthesizer 以 process-global static map 按路径缓存 mmap loader；生命周期不是 session-owned，
  路径集合也没有上限。
- content CUDA helpers 为单个 matmul/conv/norm 反复分配 device buffers、复制 host weights/
  activations、同步拷回 host，并创建/销毁 handles。
- RMVPE/VITS CUDA ops 同样对大量单算子执行 raw `cudaMalloc/cudaMemcpy/cudaFree`。
- 没有 per-session CUDA stream/event、device-resident complete graph 或 profile-bounded workspace。
- `CudaFlatIndex` 名称包含 CUDA，但 current exact search 在 host mmap 上做
  `O(queries × database × dim)` 距离和 partial sort。它的语义可作为 exact reference，但
  latency 随 rows/query batch 增长；在 R0 测量并设定支持上限前，不能宣称它已满足 realtime。

这些事实说明首个工程阶段必须是包含 Index 基线与 owner 的 persistent runtime，而不是先接
麦克风或 WebSocket；也不能通过删除 Index 来绕开质量目标。

### Existing timing surface

`VCResult` 只含 HuBERT、F0、Index、Synth 和 total 的单次 wall timing，Pipeline 使用
`high_resolution_clock`。它没有 monotonic capture timestamp、queue wait、algorithmic
lookahead、device buffer、jitter、drop/xrun 或 p95/p99，不能用于 realtime SLO。

## 3. Diagnostic current-tree measurement

命令：

```bash
cd tests
test_start_ns=$(date +%s%N)
/tmp/resonantia-final.udNc8h/test_e2e_conversion \
  "End-to-end voice conversion (real audio)"
test_end_ns=$(date +%s%N)
```

环境与资产：

| Field | Value |
|---|---|
| machine | NVIDIA Jetson Orin NX Engineering Reference Developer Kit, SM87 |
| OS | Linux 6.8.12-1021-tegra, aarch64 |
| power mode | MAXN (`nvpmodel -q` reported mode 0) |
| clocks | not locked; `jetson_clocks --show` required root and was not run |
| initial sampled thermal zone | approximately 57.5°C |
| build | existing Release current-tree build `/tmp/resonantia-final.udNc8h` |
| input | `speech_librispeech.wav`, 93,680 samples @ 16 kHz = 5.855 s |
| models | HuBERT + RMVPE + F0 VITS v2, 40 kHz output |

Observed result：

| Metric | Value |
|---|---:|
| HuBERT | 1,219.19 ms |
| RMVPE/F0 | 2,142.12 ms |
| Synth | 13,145.1 ms |
| Pipeline total | 16,507.2 ms |
| process wall | 16.775 s |
| Pipeline RTF | 2.819 |
| wall/input duration | 2.865 |
| output | 233,600 samples @ 40 kHz |
| E2E waveform correlation | 0.999716 |
| test result | passed, 467,208 assertions in one test case |

该诊断没有固定或报告代表性目标声线 Index 的 hash、rows、dim、`k`、`index_rate` 和 query
batch，表中也没有可归因的 Index stage timing。因此它既不能证明 Index-enabled realtime，
也不能用于选择 no-index；R0 必须补独立 baseline。

该运行证明 current path 在这次环境中不能持续 realtime；它**不**是正式 SLO baseline，原因是：

- 未锁 clocks、没有完整 warm-up/p50/p95/p99；
- 只有一个短 utterance，未记录功耗/频率/温度时间序列；
- 没有 queue、audio device 或 capture-to-playback latency；
- 没有 peak CUDA/system memory sampling；
- current call 包含 host/device allocation、copy 和 loader overhead，尚未区分优化后下界。

另一次 `test_fcpe_alignment` 实际通过（9 assertions，wall 6.622 s），但该测试先单独计算 mel，
再由 `infer` 再计算一次 mel，因此不能从总时间推导 FCPE 单次 latency，也不能支持“FCPE
整链 realtime”的结论。

### Standalone exact-Index feasibility probe

仓库新增 `benchmarks/realtime_index_benchmark.cu`（SHA-256
`5e3e041d7907adbfc1826072cd16ff37fa534cc0e2ff440be772852d84f17e38`）。它不接 CMake、
CTest 或产品代码，输出明确标记 `diagnostic_only=true`、`product_acceptance=false`。probe 在
初始化时常驻 database/norm/workspace/queries/output，计时调用不做 runner-owned allocation，
并独立对拍 top-k distance、归一化 weight、纯 retrieved vector 和 blended output，避免用低
`index_rate` 掩盖检索错误。构建与 10k 命令见 [`benchmarks/README.md`](../../benchmarks/README.md)。

同一台 Orin NX、CUDA 13.2、MAXN、未锁 clocks 的 2026-08-10 复跑结果：

| Synthetic shape | GPU p50 / p99 | synchronized wall p99 | CPU exact reference total | Persistent device bytes | Exact check |
|---|---:|---:|---:|---:|---:|
| 10k × 768, nq=1, k=8 | 1.003 / 1.009 ms | 1.018 ms | 7.570 ms | 35,003,652 | 8/8 |
| 100k × 768, nq=1, k=8 | 4.166 / 4.194 ms | 4.202 ms | 75.488 ms | 312,203,652 | 8/8 |
| 100k × 768, nq=8, k=8 | 15.190 / 15.212 ms total | 17.676 ms total | 606.118 ms total | 315,069,088 | 64/64 |

每项 warm-up 10 次、计时 100 次，FP32 pedantic SGEMM + deterministic exact top-k；数值门槛
均通过。CPU reference 在 GPU timing 后执行，只用于 exact semantics 对拍，不是 current product
的正式延迟 runner。结果说明 100k rows（约 33 分钟、50 fps 的特征规模）在持久 GPU 方案上
具有低毫秒级独立检索可行性，因此没有理由为性能预先放弃 Index；但它仍是 synthetic、无模型
竞争、无 target-speaker quality 的 stage 下界。current `CudaFlatIndex` 产品实现仍是 host mmap
scan，本次没有切换 backend，也没有据此设定 qualified rows/rate/batch 上限。

独立 probe 使用自己的低层 JSON schema，不调用 `BenchmarkReport` validator；后者目前只是
完整 realtime report 的构造期字段/模式门禁，尚无 session producer 或稳定序列化。两类证据
不得合并为“10-minute SLO 已报告”。

## 4. Product-boundary alternatives

### A. Deterministic stream library only

优点：最容易测试跨块 state/backpressure，不依赖物理设备。缺点：没有完整用户工作流，不能
单独满足“端到端 realtime”。保留为所有方案的 R2 prerequisite。

### B. Local ALSA full-duplex CLI — recommended

优点：Jetson/Linux 原生、低额外协议开销、部署边界清晰；ALSA 可以提供 hardware timestamp、
period/buffer/xrun 状态。当前环境已有 ALSA 1.2.11 headers/library、`arecord`/`aplay`、capture/
playback devices，当前用户属于 `audio` group。

风险：物理 route/format 支持需要 loopback 或明确设备验收；不能在 audio critical thread 上
运行模型。建议以 deterministic replay 先验证 runtime，再接 ALSA adapter。

### C. WebSocket/browser streaming — defer

当前 `vc_serve` 是 HTTP/1.1 + CLI subprocess orchestration，不在进程内常驻模型。WebSocket
会引入 binary framing、connection/session state、network jitter/backpressure、browser
AudioWorklet、认证、origin/TLS 和 GPU 多连接语义。把它塞入现有 `/api/convert` 会同时改变
Spec 003 的 deployment 与 GPU ownership，应另立后续 ADR。

### D. Per-chunk `vc_convert` subprocess or `convert_buffer`

模型重复加载、future-dependent DSP、phase reset 和不可控 queue latency 使其不满足本规格，
拒绝。

## 5. Approved model profile and retrieval modes

### Qualified competitive profile — approved

v2 HuBERT + FCPE + F0 VITS 40 kHz，正常质量路径启用有效 Flat Index 且
`index_rate > 0`：

- 有 current-tree 模型与数值 fixtures；v2 E2E 有真实 reference；
- FCPE checkpoint 明显小于 RMVPE，并且模型结构更适合有限 receptive-field window；
- Index 是 RVC 目标音色检索质量的一部分，不能为了达成 latency 静默关闭；
- 固定模型组合和经 R0 资格化的一个 Index envelope，才能给出可复现 workspace、latency
  与 quality contract。

该 profile 已获产品/架构批准，但仍未通过实现门禁。FCPE current implementation 仍有 host
mel、per-op allocation/copy；Flat Index 产品路径当前是 host exact scan。独立 GPU probe 只
证明 stage 诊断可行性，二者都尚未在 qualified full pipeline 中证明实时，且本文不预先选择
`index_rate`、rows 上限或 host/device 产品检索算法。

### Explicit no-index diagnostic/compatibility mode

no-index 只允许由用户在启动时明确选择，或通过启动配置预先授权 sustained-overload 后的
一次性粘滞降级。缺失/损坏/超上限 Index 不得自动触发 fallback；默认按启动错误或运行时
error/overload 契约处理。任何 fallback 都记录 transition reason、sequence、generation、
bypassed hops 和单独 latency/quality 结果，同一 session 不自动恢复 Index。

no-index 可帮助区分“模型核心过慢”与“检索 stage 过慢”，并提供有限兼容性，但不能用于
替代 Index-enabled 的 nominal SLO、质量验收或竞争力声明。

### Deferred profiles

- RMVPE：current default/offline quality path，但双向 GRU、361 MB checkpoint 和完整-window
  implementation 增加状态/延迟/内存风险。
- WavLM：需要匹配训练的 synthesizer，不能用 stock RVC v2 自动替换。
- v1：可能减少 encoder layers/feature width，但当前最高级 E2E reference 与 proposed
  product profile 是 v2；需要单独 benchmark/quality acceptance。
- 多 Index/profile 动态切换：会扩大 rows/dim/`k`、内存和调度矩阵；首版只资格化 R0 选定的
  envelope，不允许运行中更换 Index。

## 6. Chunk and continuity research

现有模型不是 causal graph。首版兼容路径应比较有限 matrix，而不是把所有参数公开为用户
flags：

```text
input hop:       20 ms aligned multiples
context window:  finite candidates chosen from real-asset benchmark
lookahead:       explicit and included in end-to-end latency
emit:            center frames only
crossfade:       bounded and no more than half the emitted region
```

需要分别解决：

- HuBERT global attention：用 sliding context recompute + center crop，不能宣称 KV cache/causal；
- FCPE：保存 mel/receptive context、F0/UV、median/protect state；
- VITS：text/flow/generator 的上下文 crop，以及 NSF phase/overlap tail 连续；
- input/output resampler：保存 fractional phase，不逐块四舍五入；
- output accounting：按 absolute sequence/frame placement，避免 chunk jitter 累积 time drift。

候选选择必须同时报告 quality、compute p99、algorithmic latency 和 workspace；只比较平均 kernel
time 会偏向大窗口且隐藏 queue/device 延迟。

## 7. Approved latency and overload policy

已批准的 nominal SLO：

- capture-to-playback p95 `<= 250 ms`；
- capture-to-playback p99 `<= 350 ms`；
- compute p99 `<=` emitted audio duration；
- 5 s warm-up 后 10 min run：0 drop、0 xrun、0 fatal、0 non-finite。

这是批准目标，不是 current result。主验收必须使用 Index-enabled profile 且 0 fallback；
no-index run 必须分开。如果未来基于证据选择 500 ms relaxed 档，需另行修改批准契约，不能
在实现中自行放宽。

批准 overload 为 bounded drop-oldest whole-hop：它限制陈旧语音延迟并保持最新输入。所有
drop、sequence gap、concealment 和 xrun 必须计数。Index 失败/超预算也必须可观测；默认不
旁路 Index，继续按 error/overload 契约处理。只有启动时明确授权的 fallback policy 才能在
R0 固化条件下粘滞切到 no-index，并记录 Index deadline/fallback/bypass。nominal benchmark
不允许依靠 drop 或 fallback 达标；二者只在独立 overload test 验证。

## 8. Existing real assets

### Audio

`tools/download_test_audio.py` 从 Hugging Face 的
`hf-internal-testing/librispeech_asr_dummy` clean validation parquet 取前三个真实 LibriSpeech
样本。2026-08-10 审计将下载脚本和实际 source 固定到 revision
`5be91486e11a2d616f4ec5db8d3fd248585ac07a`、parquet SHA-256
`4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603`，并从固定 revision
重新生成前三行；输出 bytes/SHA-256 与本地 WAV 完全一致。完整 row ID、speaker/chapter、
transcript、license caveat 和生成脚本/lock hashes 见 [R0 manifest](r0-assets.json)。

| File | Duration | SHA-256 |
|---|---:|---|
| `speech_00.wav` / `speech_librispeech.wav` | 5.855 s | `799f78ed4beb4de7ceae3a809262d4ce242394342ccd1d58cef7d49dbc2def46` |
| `speech_01.wav` | 4.815 s | `473b1bb573e8f7cdcd639c6e318036236190107b66bfe6892286121a92d464c1` |
| `speech_02.wav` | 12.485 s | `5f4d8f7331acffc3d2efbe0f764b9eb3447911dc17f9843e38e1083eb1685b9c` |

三条真实语音可用于 quality 和短 stream integration。10-minute endurance 可以重复真实语音
来验证资源稳定性，但重复数据不能扩大音质/说话人代表性结论。上游 LibriSpeech 官方
dataset card 标注 CC BY 4.0；dummy mirror 本身没有重复 license metadata，manifest 保留该
caveat 而不把两者混写成同一份 license 声明。

### Selected model assets

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `models/hubert_base/model.safetensors` | 378,298,232 | `c6001424ebec530e5a66afcb848b28443aec068a7acb93884fa1b4f69f24e483` |
| `models/fcpe/model.safetensors` | 43,339,596 | `6590a66a231e10ff3d52f8d90a6fb3df603ce76ccb149c93ae6f77ae62ec7167` |
| `models/pretrained_v2/pretrained_v2/f0G40k.safetensors` | 145,897,024 | `8792f5dc13bcbf8a5bf0a0890823ec0d5e5cb1102374a3f3a30bf59cf680a2da` |

比较资产：RMVPE `361,977,092` bytes，SHA-256
`820e94253031f7709724372dc7990576a444c486645120fbd93b01954d1ec704`。

[R0 manifest](r0-assets.json) 还记录了 HuBERT/VITS 固定 HF revision、原始 checkpoint
bytes/SHA-256、repo license metadata、转换脚本 hash 与当前 output hash。FCPE 已固定为
`torchfcpe==0.0.4` PyPI wheel（SHA-256 `f042c463...`）内 bundled checkpoint（SHA-256
`b9aeaeb6...`）；exact wheel METADATA/LICENSE 为 MIT。使用锁定工具环境构造模型并写出的
safetensors 与本地文件 byte-identical。表中的 `f0G40k` 是 generic pretrained
architecture/reference checkpoint，不是已训练且 provenance-tracked 的目标声线模型。

### Index evidence gap / asset sourcing

当前仓库实现和离线测试证明 Flat Index 文件边界与 exact weighted retrieval，但实际文件盘点
没有发现任何真实 `.index`，也没有与之匹配的 provenance-tracked 目标声线 v2/40 kHz model；
Web contract 临时生成的假 `model.index` 只验证事务，不是质量资产。
[R0 manifest](r0-assets.json) 因此将 `target_speaker_vits_v2_model` 与
`target_speaker_flat_index` 都标为 `missing` 和 qualification blocker。

截至 2026-08-10，对官方/原始来源和可固定 revision 的公开 RVC 资产进行的只读筛查没有找到
一个同时满足以下条件的现成包：F0 RVC v2、12-layer HuBERT 768D、40 kHz；匹配的
target-speaker model + Index；可回溯 corpus 来源/许可和适用于 voice-model/VC 的授权或同意；
以及与本项目 raw Flat 文件格式兼容。下表只记录淘汰证据，任何一项都不是已批准资产：

| Candidate and fixed revision | Rejection evidence |
|---|---|
| [`TheOriginalBox/NTTS-AI-Voice-RVC@400e77e`](https://huggingface.co/TheOriginalBox/NTTS-AI-Voice-RVC/tree/400e77ee7f26c87f6b3048b28c62ed0216f06c55) | 有 PTH/Index/`total_fea.npy`，但 card 未记录 corpus/source/voice consent、RVC version 或 sample rate；对 `total_fea.npy` 仅做 header range 检查得到 shape `(49511, 256)`，不是 v2 768D。 |
| [`SpyTechLabs/Hisanimations-RVCv2@bde5467`](https://huggingface.co/SpyTechLabs/Hisanimations-RVCv2/tree/bde54673a2188a941d1a855497f5d7e487b46fab) | card 把声音归因于第三方 YouTube 创作者，但没有 corpus 文件清单/许可或说话人同意，也未证明精确 40 kHz profile。 |
| [`1358Adrian/so-vits-svc-rvc-models@eb96fb1`](https://huggingface.co/1358Adrian/so-vits-svc-rvc-models/tree/eb96fb1579d7721dfecb56c917d8ba52860400c5) | 部分文件名声称 v2/40k 并带 Index，但来源包含商业角色、VTuber 和 voicebank，没有底层 corpus 许可及声音主体同意；仓库 license tag 不能代替这些权利证据。 |
| [`elgohary47/Sheikh-Refaat-RVC-v1@462dd20`](https://huggingface.co/elgohary47/Sheikh-Refaat-RVC-v1/tree/462dd209e26c4707250c439023e97b58739ba6dc) + [`sheikh_refaat_dataset@c1e8d12`](https://huggingface.co/datasets/elgohary47/sheikh_refaat_dataset/tree/c1e8d129f4140f5ea7b320424feec4ffdf5a2eb0) | model/FAISS Index/dataset 都存在，但 repo 名称为 v1 而 card 声称 v2；“1940s radio archives”没有 archive URL、recording ID、权利人或 speaker/estate 授权。dataset 的 MIT tag 不证明底层录音权利。 |

官方 RVC v2 用 12-layer/768D HuBERT，而官方 Index 是通过 FAISS
`index_factory(..., "IVF...,Flat")` 生成的 serialized IVF+Flat 文件；参见
[RVC v2 README](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md)、
[training instructions](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Instructions-and-tips-for-RVC-training)
和 [FAISS tuning record](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/faiss-tuning-TIPS/0a8c54fd97ef511c25e9807cdcd4f5536dc543ce)。
本项目 `CudaFlatIndex` 则要求 little-endian
`[u64 rows][u64 dim][rows * dim * float32]`，文件大小必须精确为
`16 + rows * dim * 4`。因此公开 `added_IVF*_Flat*.index` 不得直接冒充或载入 R0 raw Flat；
应使用固定 HuBERT 从同一目标语料/training split 按稳定顺序重建，并重新生成 hash。

[VCTK Corpus 0.92 官方固定记录](https://datashare.ed.ac.uk/handle/10283/3443)（DOI
`10.7488/ds/2645`）是 research-only 派生候选：可选择一名无已知记录问题的 speaker、仅用一个 mic
通道，再在本地训练 target model 并重建 Index。它当前不是选定或已批准资产；
[VCTK README](https://datashare.ed.ac.uk/bitstreams/bb7edd96-5d96-4c0e-8989-1e45597e7b72/download)
记录 CC BY 4.0、110 speakers、约 400 sentences/speaker 和 48 kHz audio，但
[VCTK license](https://datashare.ed.ac.uk/bitstreams/956a1688-0b59-428c-8a2f-10837433dde3/download)
明确说明 privacy、publicity 和 personality rights 不在该许可授权范围内。因此该候选只能在
保留此 caveat 并完成预期用途权利评审后用于本地研究；不能由 CC BY 4.0 推导出
目标声线的公开发布、商用或 personality rights 已清除。

若自建 R0 资产，最小可审计交付集为：

- 单说话人、低噪、dry/mono 的固定 corpus 和不重叠 train/validation/test split；逐文件记录
  path/bytes/SHA-256/duration/rate/channels、immutable source revision/DOI、license/attribution、
  rights holder、processing history 和适用于 ML training/VC/派生模型/预期分发的同意或授权；
- F0 RVC v2/768D/40 kHz target-speaker checkpoint，记录 trainer repository commit、环境 lock、
  command/config/seed/epochs、speaker ID、原始 PTH 与 runtime safetensors SHA-256，并验证必需 tensor
  names/shapes/dtypes/finiteness；
- 使用相同 corpus training split 和固定 HuBERT hash 产生的 raw 768D Flat Index，记录稳定的
  file/frame order、builder commit/command、rows/dim/bytes/SHA-256，并验证精确文件大小；
- 一份 pair-binding manifest，绑定 corpus/model/Index/HuBERT hashes、speaker ID、`k`、
  `index_rate`、query batch 和 qualified rows envelope，防止 mix-and-match；
- 与训练语料隔离的真实 holdout/reference，以同一 model/profile 执行 Index-enabled 对 no-index
  的可归因质量 A/B，并报告 target-speaker similarity、latency 和 memory。

在上述证据入库且通过 R0 校验前，用户关于 Index 竞争力的判断是产品方向，不是缺失
benchmark 的替代品；`target_speaker_vits_v2_model` 和 `target_speaker_flat_index` 必须继续
保持 `missing`/qualification blocker，不得将任一研究候选表述为 approved。

## 9. Asset-governance gaps

2026-08-10 已完成的治理工作：

- `tests/fixtures/README.md` 已改为实际 `.bin`/WAV inventory，并记录固定真实 utterance；
- `models/README.md` 已改为 current paths，并区分 runtime safetensors 与一次性 `.pth` 输入；
- [machine-readable R0 manifest](r0-assets.json) 已记录所有首版 present 资产的 path/bytes/hash、
  source/license/generation evidence，并由 `sha256sum --check --strict` 全部复核；
- 音频前三行已从固定 dataset revision 重新生成并逐字节匹配，HuBERT/VITS 原始 checkpoint
  已与固定官方 HF revision 的 remote bytes/hash 对上。

仍然阻塞或未验证：

- 没有已授权 corpus 及其匹配的代表性目标声线 v2/40 kHz model+Flat Index pair，
  Index-enabled quality/latency baseline 不可运行；
- `download_test_audio.py` 已固定 manifest revision；`download_models.py::download_hubert` 的
  重复嵌套路径已由 mocked clean-driver reproducer 修复。当前本地 source/output hashes 已固定，
  但历史转换包版本未保存，且本轮未重新生成大型模型输出；
- 既有非 R0 reference/separation/VAD assets 尚无统一完整 provenance，普通 Git checkout 也不会
  自动获得被忽略的大型 fixtures/models；
- 尚无 streaming-specific reference、chunk/timestamp manifest、boundary metric baseline 或 ALSA
  loopback fixture。

FCPE provenance 已不再是 qualification blocker：`tools/pyproject.toml`/`uv.lock` 固定
`torchfcpe==0.0.4` 及 wheel hash，`export_fcpe.py` 校验 bundled checkpoint hash、强制
weights-only、在同一文件系统的 staging 中生成，并用进程内 rollback 发布；任一生成或发布
失败保持旧资产。
锁定环境重放的 model.safetensors 与本地 SHA-256 `6590a66a...` 逐字节相同。现有
`fcpe_ref_f0.bin` 与本次锁定环境重算仅有最大约 `1.38e-4 Hz` 浮点差，但历史 reference
生成环境没有完整保存，因此它仍按既有数值门禁使用，不被伪称 bit-exact provenance replay。

缺失资产应导致 benchmark/integration 不注册或明确 unverified，不能 return-success 或计入通过。

## 10. Maintainer decisions recorded

维护者于 2026-08-10 明确允许推进，并对 Index 竞争力提出约束，四项决定记录为：

1. RT-DEC-001：本机 ALSA boundary + deterministic WAV replay；
2. RT-DEC-002：v2/HuBERT/FCPE/40 kHz，Index-enabled 是 qualified 正常质量路径；no-index
   仅为显式诊断/兼容或预授权 overload fallback；
3. RT-DEC-003：capture-to-playback p95 250 ms/p99 350 ms，compute p99 不超过 emitted duration；
4. RT-DEC-004：bounded drop-oldest whole-hop，silence/crossfade concealment 与单调计数。

尚未由维护者或证据选择的量包括 `index_rate`、Index rows 上限/执行算法、fallback trigger、
window/workspace/ALSA period 和 Index 质量 A/B 门槛。它们由 R0/R3 的真实资产 matrix 固化，
不是首版任意用户选项，也不得由实现者凭空填写。
