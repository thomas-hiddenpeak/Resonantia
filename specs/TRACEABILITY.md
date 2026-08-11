# Resonantia 需求追踪矩阵

**核对日期：2026-08-11。** 本表描述当前仓库中的实现与 CTest 证据；本次 current-tree
完整资产运行记录如下。状态词定义见 [specs/README.md](./README.md#状态词汇)。

## 证据口径

- 默认门禁由 `VOXMUTATIO_BUILD_TESTS=ON` 注册，当前包括 core、audio I/O、content
  kernels、F0 基础、Index 边界、pipeline smoke、autograd、真实 CLI 进程、真实 HTTP
  socket 契约、legacy Web API 和 realtime 报告 validator，共 11 项。
- 资产门禁仅在 `VOXMUTATIO_ENABLE_MODEL_TESTS=ON` 时构建；每个测试还要通过
  `cmake/ResonantiaTests.cmake` 的资产存在检查，当前完整资产环境注册 19 项。缺资产时
  “不注册”不是“通过”。
- 7 个 CUDA contract 与全部 19 个 integration 共用 `RESOURCE_LOCK voxmutatio_gpu`；CLI、
  HTTP、legacy Web API 与 realtime 报告 validator 四个 host contract 不占 GPU lock。当前 CI 在普通 PR 只做
  仓库策略检查；ARM64/SM87 CUDA build 与默认 contract CTest 只在 `main` push 或手动触发。
- `/tmp/resonantia-audit-final.VbIWW3` 的 Release/SM87、model-tests-enabled 完整运行记录为
  **30/30 passed**（11 contract + 19 integration、0 failures，real 335.05 秒）。这只证明当前
  已注册 suite；未注册资产、standalone probe 与尚不存在的 realtime 产品验收不能计作通过。

## VC 推理与转换

| ID / 约束 | 当前实现位置 | CTest / 其他证据 | 状态与缺口 |
|---|---|---|---|
| VC-FR-001 WAV I/O | `io/audio_io.*`、`pipeline::convert_file` | `test_audio_io`（默认）；`test_e2e_conversion`（资产门禁） | **已实现·默认门禁**；FLAC/MP3 不支持 |
| VC-FR-002 Resampling | `io::resample_linear`、`pipeline/pipeline.cpp` 的 16 kHz 前端与 40 kHz 模型率输出 | `test_audio_io` resample（默认）；e2e（资产门禁） | **已实现·默认门禁**；当前 synthesizer 仅兼容 40 kHz，不提供其他模型率或独立目标输出采样率 |
| VC-FR-003 HuBERT default | `content/hubert_encoder.*`、pipeline 默认 encoder、v1/v2 feature dim | `test_hubert_inference`、`test_pipeline_contract`（资产门禁） | **已实现·可选门禁** |
| VC-FR-004 Optional WavLM | `content/wavlm_encoder.*`、pipeline encoder 分发 | `test_wavlm_alignment`（资产门禁） | **部分**：实现与对齐门禁存在；是否匹配 synthesizer 由调用方/模型资产保证，stock RVC 不可无条件替换 |
| VC-FR-005 F0 | `f0/rmvpe*`、`f0/fcpe.*`、pipeline F0 分发；完整 tensor manifest、staged init、公开输入 finite/size 边界 | `test_f0`（默认基础/坏 checkpoint）；RMVPE/FCPE alignment（资产门禁，含失败重初始化与输入边界） | **已实现·默认及可选门禁**；PM 明确不支持 |
| VC-FR-006 Index | `index/cuda_flat_index.*`；`pipeline/pipeline.cpp` 的 required-rate/feature-width/lookup-failure 边界 | `test_index`（默认边界/移动）；`test_index_retrieval` 与 pipeline 错维/缺失 Index 契约（资产门禁） | **已实现·默认门禁**；真实特征检索为可选门禁；非零 rate 不再静默旁路缺失、错维或失败的 Index |
| VC-FR-007 Synthesizer | `synthesizer/synthesizer.*`、`vits_ops.cu` | `test_vits_alignment`、`test_vits_gen_debug`、pipeline synthesizer contract（资产门禁） | **已实现·可选门禁**；F0 v1/v2，non-F0 拒绝 |
| VC-FR-008 Pipeline | `pipeline/pipeline.*`：内容/F0/Index/protect/RMS/分块拼接 | `test_pipeline_smoke`（默认）；pipeline contract、e2e、long audio（资产门禁） | **部分**：HuBERT/Index/VITS 按块执行；F0 仍整段执行，host 与 F0 CPU/GPU working set 随时长增长；约 18 秒 fixture 不等于真实 5 分钟资源实测 |
| VC-FR-009 Numerical safety | 公共入口边界、`core/cuda_buffer.h`、各模块 checked loaders | audio/index/content/autograd 默认边界；pipeline/model rollback 为资产门禁 | **部分**：核心边界与 RAII 有测试；并非每条 CUDA kernel/library 失败都有故障注入 |
| VC-FR-010 Output | pipeline/40 kHz model-rate `AudioBuffer`、`io::write_audio`、`vc_convert` 输出 | `test_audio_io` 原子替换/失败保留（默认）；e2e 有限/非空输出（资产门禁） | **已实现·默认门禁**：只支持当前 40 kHz model-rate WAV；同目录 staging 经 fsync/close 后 rename，失败保留旧目标 |
| VC-FR-011 CLI | `tools/app_support/cli_parse.h`、各 `vc_*_app.*` Options/parser/runner、薄 main 与现有 CMake target | `test_cli_contract` 真实进程非法语法/退出码（默认） | **部分**：严格解析与失败契约有默认门禁；真实模型 CLI 成功 WAV 仍缺资产门禁 |
| VC 初始化失败不得保留旧模型 | `pipeline/pipeline.cpp::init` staging/commit | `test_pipeline_contract` failed reinitialization（资产门禁） | **已实现·可选门禁** |
| VC-FR-012 Auxiliary CLIs | `vc_batch`、`vc_probe` CMake targets | `test_cli_contract` 覆盖 `vc_batch` 数值失败；无 `vc_probe` 行为门禁 | **部分**：`vc_probe` 仍是骨架；不得计入 Spec 001 完成证据 |

## 训练

| ID / 约束 | 当前实现位置 | CTest / 其他证据 | 状态与缺口 |
|---|---|---|---|
| TR-FR-001 Autograd | `autograd/tensor.h`、`src/autograd/autograd.cu` | `test_autograd`（默认 gradcheck/边界） | **已实现·默认门禁**；覆盖度随公开算子维护 |
| TR-FR-002 AdamW/裁剪/非有限防护 | `autograd::AdamW`、`training/gan_trainer.cpp` | optimizer 凸收敛（默认）；GAN 集成（资产门禁） | **已实现·默认门禁**；生产超参稳定性未知 |
| TR-FR-003 Differentiable STFT/mel | `training/mel_loss.*`、`autograd` frame/math ops | `test_autograd` primitives（默认）；`test_mel_loss`（真实语音资产门禁） | **已实现·可选门禁** |
| TR-FR-004 Trainable decoder | `training/generator_trainer.*` | `test_generator_trainer` alignment/fine-tune/export（资产门禁） | **已实现·可选门禁** |
| TR-FR-005 Posterior/Flow/Discriminator | `training/posterior_encoder.*`、`discriminator.*` | `test_gan` enc_q/flow/discriminator/rollback（资产门禁） | **已实现·可选门禁** |
| TR-FR-006 GAN losses/alternating step | `training/gan_trainer.*` | `test_gan` fine-tune/full（资产门禁） | **已实现·可选门禁**；长期稳定性未知 |
| TR-FR-007 WAV/data baseline | `tools/app_support/vc_train_app.cpp` 的发现、前端与固定片段采样；`vc_preprocess_app.cpp` 可独立准备片段；对应 `tools/vc_*.cpp` 为薄 main | `test_cli_contract` 严格数值/退出码（默认）；无独立 filelist/cache/batching 产品 CTest | **部分**：训练路径与严格参数边界存在；完整数据产品仍缺 |
| TR-FR-008 Progress | `vc_train_app.cpp` runner 的文本 step/loss 输出 | 无机器可读协议 CTest | **部分**：稳定结构化协议未定义 |
| TR-FR-009 Inference G export | `GeneratorTrainer::export_model`、`vc_train` 临时文件 + rename | generator zero-step/训练后 round-trip（资产门禁） | **已实现·可选门禁** |
| TR-FR-010 Independent Index build | `tools/build_index.cpp` | index contract/retrieval；无训练事务集成测试 | **部分**：工具独立可构建，未与 G 导出原子绑定 |
| TR-FR-011 Checkpoint resume | 无 G/D/optimizer/epoch/RNG 持久化 | 无 | **不支持** |
| TR-FR-012 Legacy `FineTuneTrainer` | `training/fine_tuning.cpp` | `test_core` 仅验证畸形 dataset | **不支持**：门面必须失败 |
| TR-FR-013 Distributed/FP16/multi-GPU/from-scratch | 无 | 无 | **不支持** |
| TR-FR-014 Production voice quality | 无当前可复现客观/盲听门禁 | 无 | **未知/未验证**；不得从短步 loss 推断音质或相似度 |
| 训练初始化事务与失败后不可用 | Generator、PosteriorEncoder、Flow、Discriminator、GANTrainer reset/guards | generator/GAN rollback cases（资产门禁） | **已实现·可选门禁** |

## 声源分离与预处理

| ID / 约束 | 当前实现位置 | CTest / 其他证据 | 状态与缺口 |
|---|---|---|---|
| SEP-FR-001 STFT/iSTFT | `separation/stft.*`，training 复用 STFT | `test_separation_stft`、`test_separation_spec`（可选 fixture 门禁） | **已实现·可选门禁** |
| SEP-FR-002 UMX runner | `separation/separator.cu` | `test_separation_umx`（模型 + reference 资产） | **已实现·可选门禁** |
| SEP-FR-003 RoFormer runner | `separation/roformer.cu` | `test_separation_roformer`（模型 + reference 资产） | **已实现·可选门禁**；当前本地 44.1 kHz 模型运行时 clamp 到约 4 秒 chunk，通用配置的内存上界与 scratch 释放仍缺独立资源回归门禁 |
| SEP-FR-004 Task coverage | local separation weights/JSON、registry、preprocess stage flags | CTest 只注册 dereverb RoFormer reference；没有五任务逐项门禁 | **部分**：五类代码/本地资产路径存在，不等于五类均由当前 CTest 证明 |
| SEP-FR-005 Long audio | `Roformer::forward_chunked`、config chunk clamp、overlap-add | 当前 separation CTest 没有专门强制多 chunk 的资源/一致性用例 | **部分** |
| SEP-FR-006 VAD | `separation/vad.cu` | `test_separation_vad`（模型 + reference 资产） | **已实现·可选门禁** |
| SEP-FR-007 Registry | `separation/sep_registry.*`、drop-in JSON metadata、`tools/app_support/vc_preprocess_app.cpp` strict resolution | `test_cli_contract` unknown/default/task mismatch（默认） | **已实现·默认门禁**：发现与 CLI 拒绝契约有进程级证据；真实五任务权重执行仍分别受资产约束 |
| SEP-FR-008 CLI/Web | `vc_preprocess_app.cpp` 的 `--*`/`--*-model` runner、`WebService` 的 `/api/sep-models` 与 `/api/step` | `test_cli_contract` + `test_web_contract`（默认） | **部分**：CLI model/task 与 step busy/rollback/success 有门禁；`/api/sep-models` 多模型响应及五任务端到端资产覆盖仍缺 |
| SEP-FR-009 Numerical safety | UMX/RoFormer/VAD 完整已用 tensor 的 F32/shape/nbytes/finite loader；公开 input/output finite boundary；`vc_preprocess` strict parser/派生范围 | CLI 边界（默认）；UMX 错 shape、RoFormer/VAD 非有限权重、三模型 NaN/Inf 输入及有效 reference（资产门禁） | **部分**：已用权重与可观察输入/输出 fail-closed；任意恶意模型 config 的资源上界和所有 CUDA 中间值故障注入仍缺完整门禁 |
| SEP-FR-010 Runtime purity | C++/CUDA runners、safetensors loader；Python converter 仅在 `tools/` | CMake runtime targets不查找 Python；无独立 purity CTest | **部分**：静态证据存在，缺自动依赖审计 |

## Web 服务

| ID / 约束 | 当前实现位置 | CTest / 其他证据 | 状态与缺口 |
|---|---|---|---|
| WEB-FR-001 Static UI | `webui/`、`tools/web/workflow_support.*::serve_static`、CMake `vc_serve` | `test_web_contract` 真实 server/static root（默认） | **已实现·默认门禁** |
| WEB-FR-002 Configurable bind | 薄 `vc_serve` startup config + `HttpTransport`，默认 `0.0.0.0:8080` | `test_web_contract` loopback/动态端口；help/默认值代码（默认） | **已实现·默认门禁**；默认全接口由静态契约保留，测试不暴露外网 |
| WEB-FR-003 Optional auth | `HttpTransport` 的 token config、constant-time Bearer compare 与 `/api/` precheck | `test_web_contract` anonymous/wrong/correct token + public static（默认） | **已实现·默认门禁** |
| WEB-FR-004 WAV-only upload | filename 白名单、`.wav`、RIFF/WAVE signature、duplicate 检查 | 无 multipart/upload CTest | **部分** |
| WEB-FR-005 CLI orchestration | `WebService` + argv-based `PosixProcessRunner` 调用 preprocess/train/index/convert；不经过 shell；HTTP sockets close-on-exec | `test_web_contract` 使用长寿命假 CLI 验证 FD 隔离，并以假 preprocess/train/index/convert 覆盖全部工作流（默认） | **已实现·默认门禁** |
| WEB-FR-006 GPU serialization | `WebService::gpu_mutex_`；convert/preprocess/step 使用 `try_lock`，step reservation 后才返回接受 | `test_web_contract` preprocess 持锁时 step 立即 409（默认） | **已实现·默认门禁**；训练排队语义保持后台作业契约 |
| WEB-FR-007 Transactional material | staged raw/work、rename commit、backup rollback | 无故障注入 CTest | **部分** |
| WEB-FR-008 Temporary cleanup | `TempDir`/`mkdtemp` RAII | `test_web_contract` step/训练事务成功与 rollback 后检查产物/backup；一次性 convert/preprocess 未故障注入 | **部分**：`TempDir` 析构清理失败仍未上报 |
| WEB-FR-009 Path containment | component 拒绝、canonical webroot containment | 无 traversal/symlink CTest | **部分** |
| WEB-FR-010 Error honesty | `ProcessResult` 精确保留退出/信号/launch/wait/log I/O 状态；严格 boolean；有效 WAV/voice；step work+marker 及训练 clips/model/index staging + rollback 发布 | `test_web_contract` 404/405/Allow、busy、step work+marker commit/rollback、四类 CLI 非零退出、invalid boolean、raw+flags/work-marker argv、空/非 WAV/超限输出、unknown voice、旧训练三产物回滚、新声线无 partial artifact、成功链（默认） | **部分**：已消除 shell filter/index/旧产物/unknown voice/ignored cleanup flag/marker 假成功；信号终止与 spawn/wait 故障注入仍缺默认门禁 |
| WEB resource boundaries | `HttpTransport`：64 KiB header、128 MiB body、256 MiB inflight、64 MiB file/缓冲音频响应、16 KiB field、128 parts、32 connections、deadlines | 超限 preview/preprocess/convert 为真实 HTTP 默认门禁；其余常量与拒绝分支尚无 boundary/slow-client CTest | **部分** |
| WEB 部署边界 | Spec 003：无 TLS、无默认认证、无用户/角色/多租户/rate limiting/WebSocket | README/Spec 003 | **不支持**直接不可信公网部署或实时音频 |
| WEB-MIG-001..004 Legacy public C++ API | `include/voxmutatio/webui/server.h`、`src/webui/server.cpp` deprecated compatibility surface；真实服务仍为 `vc_serve` | `test_legacy_webserver_contract`（默认）验证最小消费者生命周期与 `start()` fail-fast；`test_web_contract` 验证真实服务未回退 | **已实现·默认门禁**：保留一个迁移窗口，后续不兼容版本才可删除；不提供嵌入式 HTTP server |

## 实时声音转换（已批准、R0 证据部分完成、产品未实现）

本节追踪 [Spec 006](./006-realtime-voice-conversion/spec.md) 的已批准需求；当前没有 realtime
runtime、session、ALSA CLI 或产品 benchmark runner。ADR-0002 于 2026-08-10 `Accepted`：
本机 ALSA、Index-enabled qualified profile、250/350 ms SLO 与 bounded drop-oldest 已决。
R0 已有 legacy characterization、构造期报告 validator、asset manifest、离线 Index
fail-closed 和 standalone exact-retrieval 诊断 probe；这些 scaffold 不等于产品路径或性能验收。

| ID / 约束 | 提案目标位置 | CTest / 其他证据 | 状态与缺口 |
|---|---|---|---|
| RT-FR-001 Stateful session | 已批准 `streaming` component/session 目标 | legacy `test_vits_stream_characterization` 只冻结 helper 现状 | **不支持**：当前只有 whole-buffer pipeline 与 full-infer crop helper |
| RT-FR-002 Transactional initialization | 已批准 runtime/session/Index staged init | 无 | **不支持**；从 R0 证据阶段开始实施 |
| RT-FR-003 Persistent CUDA runtime | 已批准 device-resident weights/stream/handles/workspace + session-owned Index | 006 research 有 standalone persistent-GPU exact-Index 诊断 probe | **不支持**：probe 非产品 runtime/backend；当前模型路径仍重复 loader/allocation/copy，target Index backend/上限未资格化 |
| RT-FR-004 Bounded queues | 已批准 fixed-capacity SPSC rings 目标 | 无 | **不支持** |
| RT-FR-005 Real-time thread boundary | 已批准 ALSA adapter + GPU worker ownership 目标 | 无 | **不支持** |
| RT-FR-006 Frame alignment | 已批准 20 ms emit / 10 ms F0-VITS alignment 目标 | `test_vits_stream_characterization` 固定 legacy 40 kHz 下 400 samples/frame 与 crop | **不支持**：只刻画旧 helper，未实现 session frame accounting |
| RT-FR-007 Explicit context | 已批准 finite sliding window/lookahead/crop/crossfade 目标 | 无 | **不支持** |
| RT-FR-008 Cross-chunk state | 已批准 resampler/filter/F0/RMS/NSF/overlap session state 目标 | characterization 证明 legacy repeated call bit-exact 且每次 NSF phase reset | **不支持**：该事实正说明 helper 没有跨块 NSF/session state |
| RT-FR-009 Qualified profile | 已批准 v2/HuBERT/FCPE/40 kHz + 有效 Flat Index/`index_rate > 0`；显式 no-index 仅诊断/兼容或预授权粘滞 fallback | 离线 Pipeline 对非零 rate 的缺失/错维/lookup failure fail-closed；manifest 已固定 FCPE provenance，仍缺 target model + matching Index/licensed corpus | **不支持**；standalone probe 不替代 qualified target，rate/rows/backend 待真实 pair |
| RT-FR-010 ALSA adapter | 已批准 `vc_realtime` app-support/ALSA boundary | 环境探测仅记录 ALSA 可用 | **不支持**；RT-DEC-001 已决但尚无实现 |
| RT-FR-011 Backpressure | 已批准 drop-oldest/counters/concealment；Index 默认不旁路，显式预授权 fallback 才可粘滞降级 | 无 | **不支持**；RT-DEC-004 已决，fallback trigger 待 R0 |
| RT-FR-012 Error semantics | 已批准 lifecycle/error categories 目标 | 无 | **不支持** |
| RT-FR-013 Runtime updates | 已批准 immutable parameter snapshots 目标 | 无 | **不支持** |
| RT-FR-014 Observability | 已批准 monotonic/CUDA/ALSA stage metrics + Index latency/mode/query/bypass/fallback/hash/shape 参数 | `test_realtime_benchmark_report` 门控 mode-aware 构造期字段一致性 | **部分**：没有 producer、稳定 serialization、完整 outcome population 或 session metrics |
| RT-FR-015 Latency and stability | 已批准 Index-enabled 10-minute benchmark + 250/350 ms SLO，要求 0 fallback | current pipeline diagnostic RTF≈2.82；standalone probe 记录 synthetic/available-feature exact retrieval latency/memory | **不支持**：无产品 runner/代表性 target/full pipeline；probe 不是 SLO/backend/上限 |
| RT-FR-016 Numerical and continuity safety | 已批准 real-asset stream/continuity、Index exact retrieval 与 Index/no-index 质量 A/B gates | standalone probe 与 CPU exact reference 对拍；legacy crop/phase characterization；离线 module/E2E/Index 门禁 | **不支持**：没有 target-speaker 质量 A/B、stream continuity 或 boundary gate |
| RT-FR-017 Memory evidence | 已批准 fixed workspace + endurance memory report 目标 | 无 realtime resource gate | **不支持** |
| RT-FR-018 Offline compatibility | 既有 Spec 001/API/CLI/CTest 在未来迁移中保持 | current-tree 30/30（11 contract + 19 integration、0 fail，335.05 秒）；尚未发生 realtime 产品迁移 | **提案约束**：R0 scaffold 加入后现有 suite 未回退，后续仍须逐阶段复验 |

## I/O、输入边界与安全

| ID / 约束 | 当前实现位置 | CTest / 其他证据 | 状态与缺口 |
|---|---|---|---|
| IO-001：WAV 读写、未知 chunk、声道下混 | `io/audio_io.*` | `test_audio_io`（默认） | **已实现·默认门禁** |
| IO-002：重采样参数边界 | `io/audio_io.*` | `test_audio_io`（默认） | **已实现·默认门禁**；质量为线性重采样，未声明高保真指标 |
| IO-003：FLAC/MP3 | 无解码实现；pipeline 仅接受 `output_format=wav` | pipeline contract（资产门禁）、支持矩阵 | **不支持** |
| IO-004：safetensors shape/dtype/offset/payload | `io/safetensors.*`、`training/checked_weights.h` | `test_audio_io` 默认边界；训练 rollback 资产门禁 | **已实现·默认门禁**（通用解析边界） |
| IO-005：Index header、维度、范围、k、move | `index/cuda_flat_index.*` | `test_index`（默认） | **已实现·默认门禁** |
| SEC-001：CUDA allocation/copy 失败不可继续 | `core/cuda_buffer.h` 与模块错误检查 | autograd/content kernel contract 覆盖部分参数边界 | **部分**：核心 RAII 失败会抛错；不是每条 CUDA kernel 路径都有独立故障注入 |
| SEC-002：初始化失败不保留半加载状态 | pipeline、RMVPE/FCPE 与训练模型 staging/reset/initialized guards | F0 默认坏 checkpoint；F0 alignment、pipeline/training rollback tests（资产门禁） | **已实现·默认及可选门禁** |

## 构建与验证

| ID / 约束 | 当前实现位置 | 验证 | 状态与缺口 |
|---|---|---|---|
| BUILD-001：CMake 3.24、C++20/CUDA20、显式 CUDA 架构 | 根 `CMakeLists.txt`，默认 SM87 cache | configure/build 命令；可信 ARM64 CI | **已实现**；非 SM87 产品支持未知 |
| BUILD-002：核心库与 CLI 目标 | 显式 source list + 12 个内部 OBJECT 组件；`voxmutatio` static compatibility facade；四个按命令隔离的 `resonantia_vc_*_app_support` 与 `resonantia_web_support`；既有 CLI targets | fresh build、CMake graph、静态库符号集合与 CLI/HTTP contract | **已实现**；内部 target 不构成新公共 ABI |
| BUILD-003：可复现默认 contract suite | `cmake/ResonantiaTests.cmake`；优先系统 Catch2 3.4，缺失时使用固定 3.4.0 archive + SHA-256 | CI `ctest --label-regex contract` | **已实现·默认门禁**；跨机器的系统包补丁版本可能不同 |
| BUILD-004：模型/fixture 测试不得假通过 | `VOXMUTATIO_ENABLE_MODEL_TESTS`、每测试 `ASSETS` 注册检查 | configure 输出缺失资产原因 | **已实现** |
| BUILD-005：GPU 测试串行与 timeout | CTest `RESOURCE_LOCK voxmutatio_gpu`、每测试 timeout | CTest properties | **已实现** |
| BUILD-006：PR 与自托管 runner 安全边界 | `.github/workflows/ci.yml` | PR 只跑 repository-policy；`main` push/manual 才跑 GPU | **已实现**；普通 PR 没有编译门禁，这是显式覆盖缺口 |
| BUILD-007：SDD/Codex bridge | 根 `AGENTS.md`、`scripts/check_sdd.sh`、SDD 索引与 ADR 模板 | repository-policy 在 PR/push 运行 SDD contract；本地 bridge smoke | **已实现**；模块重构由 Accepted ADR-0001 追踪；ADR-0002 R0 仅部分完成，qualification 仍被 target model + matching Index/licensed corpus 阻塞 |

## 更新规则

1. 行为、默认值、支持矩阵、CMake 注册或测试资产改变时，同一变更更新对应行。
2. 从“部分”升级前，必须同时给出实现位置和已注册门禁；可选资产门禁不能升级为默认门禁。
3. 删除或改名需求时保留旧 ID，标记 superseded，并链接替代 ID；不要复用 ID。
4. 测试失败时状态不得保持“已验证”；记录失败边界或降级为“部分/未知”。
5. 性能、音质和稳定性结论不写入本表，除非仓库保存了环境、输入、命令和结果。
