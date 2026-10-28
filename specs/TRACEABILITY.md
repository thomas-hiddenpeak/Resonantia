# Resonantia 需求追踪矩阵

**核对日期：2026-10-26（完整训练编排与恢复）。** 本表描述当前仓库中的实现与 CTest 证据；当前运行记录见 [PROJECT_STATE](./PROJECT_STATE.md)。状态词定义见
[specs/README.md](./README.md#状态词汇)。

## 证据口径

- 默认门禁由 `VOXMUTATIO_BUILD_TESTS=ON` 注册，Node.js 18+ 可用、slow=OFF 的 asset-free 配置包括 core、
  audio/bounded I/O、content kernels、F0、immutable Index、pipeline/bounded DSP、autograd、
  VITS attention/卷积数值、两个私有 CUDA failure targets、training 数值/配置、真实 CLI/HTTP、Web 事务与作业启动单次分配故障、自然进程失败/回收、Web模拟结果消费、UI 认证/取消、legacy Web API 和 realtime 报告 validator。
- 资产门禁仅在 `VOXMUTATIO_ENABLE_MODEL_TESTS=ON` 时构建；每个测试还要通过
  `cmake/ResonantiaTests.cmake` 的资产存在检查。本次model=ON/codec=ON/slow=OFF注册
  **66个distinct tests**：37 contract、30 integration，`test_bounded_file_wavlm_preflight`
  同时属于两类，不能相加当作总数。66/66本地执行通过；注册JSON与完整日志见PROJECT_STATE。
  缺资产时“不注册”不是“通过”。slow默认OFF；ON额外注册130秒上限的
  `test_web_body_deadline`（本轮未运行）；真实FFmpeg codec门禁为单独可选项。
  缺Node时四个UI测试明确不注册；不影响无Node的静态UI构建/运行。
- GPU 测试共用 `RESOURCE_LOCK voxmutatio_gpu`；bounded I/O、CLI、HTTP、legacy Web API 与
  realtime 报告 validator 等 host contract 不占 GPU lock。当前 push/PR CI 自动运行托管仓库检查与独立 11 项 host 子集；
  ARM64/SM87 设备作业要求 main 手动 opt-in、就绪变量及专属 OrinNX 标签。
- 2026-08-12 的 fresh Release/SM87 build `/tmp/resonantia-nonrt-final.a8O1Ay` 已执行
  39/39（0 失败，`Total Test time (real) = 1297.94 sec`）。关键耗时为
  `test_cli_bounded_callers` 75.89 秒、`test_long_audio_chunking` 438.00 秒、
  `test_bounded_file_resources` 326.92 秒、`test_bounded_file_fcpe_v1` 249.34 秒。
  `/tmp/resonantia-audit-final.VbIWW3` 的加固前 30/30（11 contract + 19 integration、
  356.41 秒）只保留为历史基线。未注册资产、focused pass、standalone probe 与尚不存在的
  realtime 产品验收仍不能单独计作全量通过。

## VC 推理与转换

2026-10-24 性能实现的补充追踪（运行结果见 [PROJECT_STATE](PROJECT_STATE.md)）：

| 既有约束 | 本次实现 | 回归证据 |
|---|---|---|
| VC-FR-006 精确检索 | query 级最多 4 CPU workers；保留 scalar distance/tie | `test_index` 多 query/重复向量/NaN；真实 CyberTom Index 前后逐字节对照 |
| VC-FR-007/009 数值及资源 | GPU relative attention；16 MiB tiled convolution scratch；局部 cuBLAS RAII | `test_synth_ops` explicit-position/CPU convolution reference；VITS 对齐；CUDA 每个可注入 attention/transpose 边界及 public conv_pre failure |
| TR-FR-001/002 图/优化器 | 256 MiB bounded buffer reuse；block 梯度范数归约 | `test_autograd` 大梯度数组、NaN/Inf、图释放/复用、无梯度 weight-decay；真实 GAN/decoder/mel 门禁 |
| TR-FR-007/008 前处理/进度 | GAN 去重复 prior/latent；host segments 复用；CLI 分阶段文本耗时 | 真实单步 CLI 与短 GAN microbenchmark；不宣称机器进度协议 |
| WEB-FR-008 诊断及清理 | 清理前向服务日志保留有界转换日志 tail | `test_web_contract` 成功请求后仍能读取 HuBERT/Synth 耗时 |

| ID / 约束 | 当前实现位置 | CTest / 其他证据 | 状态与缺口 |
|---|---|---|---|
| VC-FR-001 WAV I/O | `io/audio_io.*`、private `io/bounded_audio_io.*`、pipeline file APIs | `test_audio_io`、`test_bounded_audio_io`（默认）；`test_e2e_conversion`（资产门禁） | **已实现·默认门禁**；seekable bounded reader/atomic streaming writer 有独立门禁，FLAC/MP3 不支持 |
| VC-FR-002 Resampling | `io::resample_linear`、`bounded_pipeline.cpp` 的全局相位 16 kHz spool 与 40 kHz 模型率输出 | `test_audio_io` + `test_bounded_pipeline_dsp`（默认）；e2e（资产门禁） | **已实现·默认门禁**；当前 synthesizer 仅兼容 40 kHz，不提供其他模型率或独立目标输出采样率 |
| VC-FR-003 HuBERT default | `content/hubert_encoder.*`、pipeline 默认 encoder、v1/v2 feature dim | `test_hubert_inference`、`test_pipeline_contract`（资产门禁） | **已实现·可选门禁** |
| VC-FR-004 Optional WavLM | `content/wavlm_encoder.*`、pipeline encoder 分发、caller file policy | `test_wavlm_alignment` 完整参考 shape/payload、有限值与全元素 RMSE <1e-4；bounded preflight（资产门禁） | **部分**：encoder 实现与对齐门禁存在；没有匹配的 WavLM-trained VITS bounded 资格资产，故 CLI/batch 明确走 legacy 而非静默换 HuBERT |
| VC-FR-005 F0 | `f0/rmvpe*`、`f0/fcpe.*`、bounded F0 window/spool；完整 manifest、staged init、finite/size 边界；ops 使用 `CudaBuffer` | `test_f0`、`test_cuda_failure`（默认）；RMVPE/FCPE alignment + final window profiles（资产门禁） | **已实现·默认及可选门禁**；PM 不支持；bounded raw core 128000，RMVPE/FCPE 单侧 context 64000/32000，硬输入独立为 256000/192000；故障注入是代表性路径，不是每条 CUDA 调用 |
| VC-FR-006 Index | `index/cuda_flat_index.*` private unlinked filesystem snapshot；pipeline required-rate/width/lookup fail-closed | `test_index` FIFO/source-mutation/exact（默认）；`test_index_retrieval`（资产门禁） | **已实现·默认及可选门禁**；优先 `FICLONE`、否则 1 MiB bounded copy/compare，需要 `O(Index)` 临时 filesystem 容量；fallback 分支未被强制注入，ENOSPC/权限失败显式拒绝，非零 rate 不旁路 |
| VC-FR-007 Synthesizer | `synthesizer/synthesizer.*`：完整 manifest/weight-norm；256 MiB VITS-only read-only anonymous snapshot + absolute path identity；公开 infer/debug finite/size/exception fail-closed；VITS ops RAII | `test_audio_io` snapshot/FIFO/ABA/read-only（默认）；`test_pipeline_smoke` VITS truncate/path replacement/CWD/finite/size（默认）；`test_synth_cuda_failure`（默认）；VITS/pipeline model tests（资产门禁） | **已实现·默认及可选门禁**；source truncate 不再造成 active mapping SIGBUS，路径身份变化仍拒绝调用；成功 init 约增加 checkpoint-size host residency；其他模型仍 file-backed |
| VC-FR-008 Pipeline | `pipeline/pipeline.*`、`bounded_pipeline.*`、shared caller policy | pipeline/bounded DSP（默认）；pipeline/e2e/long/F0/resource/CLI caller（资产门禁） | **已实现·默认及可选门禁**：HuBERT callers bounded；direct E2E matrix 为 RMVPE/v2 + FCPE/v1，另有两种 F0 profiles；effective core=`Ceff+2Qeff`、content=`Ceff+2Qeff+2Peff+160`；17.565/35.130 秒 RSS 1342624/1354844 KiB、device 892334080/851812352 B。只有两个时长点且无独立 click threshold；legacy buffer/file 与 WavLM 仍随时长增长 |
| VC-FR-008 RMS对齐增量 | bounded及buffer的居中补零窗口、固定分母、half-pixel插值；Spec 001 §2.5 | `test_bounded_pipeline_dsp`独立上游常量/PCM；两份真实音频各mix0/0.5探针；当前62项全量 | **已实现·验证通过**：修复前22断言失败；真实最大1 PCM差、相对L2<5.3e-6；保守Tom15动态/音色听审通过并部署，100ms离线预览被否决、未接入 |
| VC-FR-009 Numerical safety | 公共 checked boundaries、`CudaBuffer`、VITS public exception boundary、RMVPE/VITS private fault seams | `test_cuda_failure`、`test_synth_cuda_failure` 及模块默认边界；真实模型 rollback/alignment 为资产门禁 | **部分**：代表性 allocation、H2D/D2H、pre-launch、observable sync 和 host-allocation failure 已注入且 seam 不进入 production；并非每条 CUDA/cuBLAS/cuFFT 或真实系统 OOM 都被注入，Pipeline 持续 `bad_alloc` 可传播 |
| VC-FR-010 Output | pipeline/40 kHz model-rate `AudioBuffer`、`io::write_audio`、`vc_convert` 输出 | `test_audio_io` 原子替换/失败保留（默认）；e2e 有限/非空输出（资产门禁） | **已实现·默认门禁**：只支持当前 40 kHz model-rate WAV；同目录 staging 经 fsync/close 后 rename，失败保留旧目标 |
| VC-FR-011 CLI | `tools/app_support/cli_parse.h`、`file_conversion_policy.h`、各 app-support/薄 main | `test_cli_contract`（默认）；`test_cli_bounded_callers` 真实 HuBERT/RMVPE/v2/speech（资产门禁） | **部分**：严格解析、批处理输出冲突在模型/写入前拒绝、失败和 `vc_convert`/单文件 `vc_batch` bounded caller 已有对应门禁，本轮结果见 PROJECT_STATE；WavLM 明确 legacy，`vc_probe` 仍无行为验收 |
| VC 初始化失败不得保留旧模型 | `pipeline/pipeline.cpp::init` staging/commit | `test_pipeline_contract` failed reinitialization（资产门禁） | **已实现·可选门禁** |
| VC-FR-012 Auxiliary CLIs | `vc_batch`、`vc_probe` CMake targets | `test_cli_contract` 覆盖 batch 数值失败/递归同名冲突及旧输出保留；`test_cli_bounded_callers` 覆盖单文件成功；无 `vc_probe` 行为门禁 | **部分**：`vc_probe` 仍是骨架；不得计入 Spec 001 完成证据 |
| VC-MIG-001..006 Bounded file migration | additive API + private bounded I/O/planner/F0/RMS + shared HuBERT caller policy；legacy API 保留 | bounded I/O/DSP、RMVPE/FCPE profiles、long/resource/WavLM preflight/CLI caller | **已实现·默认及可选门禁**：R1–R5 已实施并经 2026-08-12 历史工作树 39/39 收口；当前执行见 PROJECT_STATE；临时 disk `O(duration)`、WavLM legacy、只有两个资源时长点且无独立 click threshold 仍是明示证据边界 |

VC-FR-005/008 的2026-10-24默认F0修正：`VCConfig::interp_unvoiced=false`，
bounded生产与`test_bounded_pipeline_dsp`共用private处理入口；真实
`test_e2e_conversion`验证默认raw-F0而非测试专用override。显式旧模式保留，
兼容边界见[Spec 001 §2.1](001-voice-conversion-engine/spec.md#21-f0-默认行为修正与兼容2026-10-24)。
当前执行结果与人工听审见PROJECT_STATE；不能从ASR正确外推声线质量合格。

VC-FR-007/VC-MIG-006的2026-10-25资源复验曾失败；VITS现在在进入残差分支前释放
已消费的latent、上一层激活、上采样结果和噪声投影。数值与资源门禁保持原阈值；
有插桩分配归因和无插桩CTest分别记录在PROJECT_STATE，历史通过不替代当前证据。

VC-FR-007/TR-FR-004的2026-10-24输出激活修正：推理`generator`与训练
`GeneratorTrainer::decode`在`conv_post`前使用参考值`0.01`，其他激活仍为`0.1`。
`test_vits_alignment`与`test_generator_trainer`增加独立真实音频参考的相对L2门禁；
旧实现两项均失败，修正后通过。部署/旧权重兼容及实际执行记录见PROJECT_STATE与
[Spec 001 §2.2](001-voice-conversion-engine/spec.md#22-生成器输出激活对齐修正2026-10-24)。

VC-FR-005/008/TR-FR-007的2026-10-24 RMVPE修正：`f0/rmvpe_frontend.h`使用HTK刻度
与Slaney面积归一化，训练/推理共享。`test_f0`冻结独立滤波器系数，
`test_rmvpe_alignment`改用直接调用上游生成的`rmvpe_htk_*`，校验全帧UV和最大/平均F0；
`test_e2e_conversion`使用新的`e2e_htk_audio.bin`并检查完整长度。
旧Slaney参考的通过记录不能证明官方RMVPE准确性；旧VITS固定输入对拍仍有效。
前后失败/通过、bounded profile与人工听审详见PROJECT_STATE，兼容策略见
[Spec 001 §2.3](001-voice-conversion-engine/spec.md#23-rmvpe-前端与独立参考修正2026-10-24)。

## 训练

VC-FR-007/008、TR-FR-004/007的2026-10-24采样修正：`synthesizer/sampling.h`、
Synthesizer及pipeline配置、convert/batch/train解析恢复prior/NSF采样，固定seed可复现。
`test_synth_ops`验证冻结抽样、分流/seed变化、绝对帧切片一致及非有限/索引溢出拒绝；
`test_pipeline_smoke`验证非法mode、采样索引溢出及有限权重导致的采样指数溢出失败；
`test_cli_contract`验证模式、完整uint64种子边界及非法值。
`test_vits_alignment`、`test_generator_trainer`、`test_e2e_conversion`分别比较sampled
和显式mean的独立真实音频参考；前者包含重复调用、不同seed和UV source对照。
原bounded/legacy误差、资源及GAN门禁仍适用，不因采样放宽。训练clip缓存的限制、
部署和旧权重重新听审见[Spec 001 §2.4](001-voice-conversion-engine/spec.md#24-采样合成恢复与兼容2026-10-24)
及PROJECT_STATE。

| ID / 约束 | 当前实现位置 | CTest / 其他证据 | 状态与缺口 |
|---|---|---|---|
| TR-FR-001 Autograd | `autograd/tensor.h`、`src/autograd/autograd.cu` | `test_autograd`（默认 gradcheck/边界，含 sqrt/log 裁剪端点、weight norm/Softmax FP32归约、卷积/单向量条件FP32顺序、period列边界及workspace越界） | **已实现·默认门禁**；独立算子回归不替代joint整步梯度门禁 |
| TR-FR-002 AdamW/裁剪/非有限防护 | `autograd::AdamW`、`training/gan_trainer.cpp` | optimizer 凸收敛、非有限 beta 拒绝、无效学习率更新保留旧值（默认）；GAN 集成（资产门禁） | **已实现·默认门禁**；生产超参稳定性未知 |
| TR-FR-003 Differentiable STFT/mel | `training/mel_loss.*`、`autograd` frame/math ops；新增显式RVC配置 | `test_autograd` primitives/零点导数 与 `test_training_contract` 配置/shape/尺寸/host-GPU 奇数 padding（默认）；`test_mel_loss`（真实语音资产门禁）；RVC真实音频独立梯度探针见PROJECT_STATE | **已实现·默认及可选门禁**；RVC配置不改变旧默认 |
| TR-FR-004 Trainable decoder | `training/generator_trainer.*`；joint路径保留原始g/v参数及显式说话人条件 | `test_generator_trainer`；新增`test_joint_kernel_gradients`真实尺寸六层固定输入梯度、`test_joint_numerical`整步门禁（资产门禁） | 旧路径**已实现·可选门禁**；joint **部分**：六层逐值前向/梯度及整步严格梯度/更新通过，短训练与音质未验收 |
| TR-FR-005 Posterior/Flow/Discriminator | `training/posterior_encoder.*`、`discriminator.*`；joint完整clip与可训练prior/embedding图 | `test_gan`；`test_joint_model_contract`完整manifest/rollback、`test_joint_numerical`（资产门禁） | 旧路径**已实现·可选门禁**；joint **部分**：严格单步通过，显式CLI/Web接入及限定47步完成；音质未验收 |
| TR-FR-006 GAN losses/alternating step | `training/gan_trainer.*`；新增库内`JointTrainer`完整clip/prior/选定embedding路径 | `test_gan` fine-tune/full；`test_joint_training`（含NSF分项舍入/边界）、`test_joint_model_contract`、`test_joint_numerical`（后两者资产门禁） | 旧路径**已实现·可选门禁**；joint **部分**：严格单步/导出与600帧输入资源复验通过，显式CLI/Web限定短训练完成；长期稳定性未知 |
| TR-FR-007 WAV/data baseline | `vc_train_app.cpp`与`joint_full_training.cpp`；full使用有界逐片冻结前端缓存、SHA身份、epoch排列与裁剪RNG | `test_cli_contract`预检；`test_joint_cache`往返/损坏；`test_joint_run_state`排列/恢复；`test_joint_full_cli`真实音频 | **部分**：CLI/Web full缓存/调度已验证，406段正式数据已固定并启动；batch1，不支持并行大batch |
| TR-FR-008 Progress | `vc_train_app.cpp` runner 的文本 step/loss 输出 | 无机器可读协议 CTest | **部分**：稳定结构化协议未定义 |
| TR-FR-009 Inference G export | `GeneratorTrainer::export_model`、`GANTrainer::export_model`；新增`JointTrainer::export_model`包含enc_p/Flow/decoder/选定embedding、保留原始enc_q及其他speaker行 | generator/GAN资产门禁；joint数值测试内0步逐值一致、1步训练参数归属和失败保留检查 | 旧路径**已实现·可选门禁**；joint **部分**：0/1步导出及整体数值门禁通过；不是resume checkpoint |
| TR-FR-010 Independent Index build | `tools/build_index.cpp`：确定路径排序、same-directory unique staging、fsync/close + rename | `test_build_index_contract` 真实 HuBERT/语音确定性/旧 inode/失败清理（资产门禁）；无训练事务集成 | **部分**：工具独立构建与发布已加固，仍未与 G 导出原子绑定 |
| TR-FR-011 Checkpoint resume | `JointTrainer::save_checkpoint/load_checkpoint`、`joint_checkpoint.h`、`AdamW::restore_state`；`joint_full_training.cpp`与Web完整作业编排 | `test_autograd`原子恢复/下一步；`test_joint_training`写入失败保留；`test_joint_resume`真实G/D/RNG/optimizer；`test_joint_run_state`、`test_joint_cache`、`test_joint_full_cli`及Web host/Node门禁 | **已实现·可选门禁**：真实同一状态下一步、取消/恢复、数据变更拒绝与66项全量通过；非bitwise重放、异常退出回到最近保存点，失败与统计边界见PROJECT_STATE；[ADR-0009](../docs/adr/0009-resumable-joint-training.md) |
| TR-FR-012 Legacy `FineTuneTrainer` | `training/fine_tuning.cpp` | `test_core` 仅验证畸形 dataset | **不支持**：门面必须失败 |
| TR-FR-013 Distributed/FP16/multi-GPU/from-scratch | 无 | 无 | **不支持** |
| TR-FR-014 Production voice quality | 实验性decoder/GAN；无通用生产质量保证 | 2026-10-24/25 CyberTom听审；上下文18组、同唱句独立对齐、213个内容权重身份与50步Pilot适配审计见PROJECT_STATE；不是默认CTest | **部分**：两份posterior重建自然度获接受；两份Web转换未通过；参考转换抖动/颗粒减轻但声音变薄且不像Tom，未通过。[ADR-0008](../docs/adr/0008-joint-voice-training-alignment.md)已Accepted，库内严格单步/资源复验通过，限定短训练共50步完成，2026-10-26两份留出短音频的自然度/厚度获接受、Tom音色改善；100ms预览Tom音色/齿音未通过；保守1秒/mix0 Tom15随后获接受，已交付CyberTom_20261026；仍未满足通用生产质量FR-014 |
| TR-FR-015 Cooperative cancellation | `app_support/training_cancel.*` + `vc_train_app.cpp`，可选token/完整迭代/export gate/退出2 | `test_training_cancel_protocol`：原样拆出的host协议、fake step与自然helper；`test_training_cancel`保留参数解析/实际CLI预取消 | **编排已实现**；新增joint真实GPU取消：完整迭代后退出2，旧文件保留、未发信号；仅2步短例，通用资源峰值/延迟未验收 |
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

WEB-FR-012 的服务器FFmpeg依赖已获维护者明确批准，见
[ADR-0007](../docs/adr/0007-web-audio-upload-decoding.md)。上传归一化已实现；
WAV字节兼容路径保留，新增格式不会改变模型CLI或输出WAV契约。

| ID / 约束 | 当前实现位置 | CTest / 其他证据 | 状态与缺口 |
|---|---|---|---|
| WEB-FR-001 Static UI | `webui/`、`tools/web/workflow_support.*::serve_static`、CMake `vc_serve` | `test_web_contract` 真实 server/static root（默认） | **已实现·默认门禁** |
| WEB-FR-002 Configurable bind | 薄 `vc_serve` startup config + `HttpTransport`，默认 `0.0.0.0:8080` | `test_web_contract` loopback/动态端口；help/默认值代码（默认） | **已实现·默认门禁**；默认全接口由静态契约保留，测试不暴露外网 |
| WEB-FR-003 Optional auth | `HttpTransport` 的 token config、constant-time Bearer compare 与 `/api/` precheck | `test_web_contract` anonymous/wrong/correct token + public static；`test_web_auth_ui` 重试/并发/轮换/storage 回退（Node 可用时默认） | **已实现·默认门禁** |
| WEB-FR-004 Upload validation | filename/extension白名单、WAV的RIFF/WAVE signature、归一化`.wav`后duplicate检查；transport唯一boundary/完整delimiter line | `test_web_contract` MIME/二进制WAV；`test_web_audio_upload`跨格式重名/空/不支持格式 | **已实现·默认门禁**；新增非WAV内容交由FR-012完整解码校验 |
| WEB-FR-005 CLI orchestration | `WebService` + argv-based `PosixProcessRunner` 调用 preprocess/train/index/convert；显式joint映射至`--gan-profile joint`；capability gated UI | `test_web_contract`工作流/FD；`test_web_train_cancel` joint参数/失败/取消/三产物；`test_web_joint_ui`旧服务与刷新兼容（Node可用时） | **已实现·默认门禁**；独立Web真实47步完成，音质未验收 |
| WEB-FR-006 GPU serialization | `WebService::gpu_mutex_`；convert/preprocess/step 使用 `try_lock`，step reservation 后才返回接受 | `test_web_contract` preprocess持锁时step立即409；`test_web_job_lifecycle`初始化失败释放slot/重试/活动worker保留；`test_web_job_ownership`四种A/B/C交错、旧training/step finish重放和两个负对照 | **已实现·默认门禁**；GPU mutex串行不变；重复finish修复的等价所有权语义已有普通host动态证据，原受限OS探针未重跑、未新增独立审查 |
| WEB-FR-007 Transactional material | staged raw/work + marker 退役、rename commit/backup rollback；material mutex 同步服务内发布与 preview | `test_web_contract`：第二次 rename 失败恢复旧三项/首次失败无半成品，marker 目录与 symlink 拒绝；训练上传清 marker 并执行 cleanup；256 次上传/1024 次预览完整字节，8 MiB 慢接收仍可发布 | **部分**：另有默认 `test_web_transaction_failure` 覆盖三种事务单次分配异常保留旧产物；独立探针覆盖 rollback 自身失败保留备份及六项暂停 rename 并发；外部修改/崩溃原子性不保证 |
| WEB-FR-008 Temporary cleanup | `TempDir`/`mkdtemp` RAII | `test_web_contract` step/训练事务成功与 rollback 后检查产物/backup；一次性 convert/preprocess 未故障注入 | **部分**：`TempDir` 析构清理失败仍未上报 |
| WEB-FR-009 Path containment | component 拒绝、canonical webroot containment | 无 traversal/symlink CTest | **部分** |
| WEB-FR-010 Error honesty | `ProcessResult` 精确保留退出/信号/launch/wait/log I/O 状态；严格 boolean；有效 WAV/voice；step work+marker 及训练 clips/model/index staging + rollback 发布；convert 正数 `index_rate` 缺非空 Index 时在启动 CLI 前 400，零 rate 不传 Index flags | `test_web_contract` 404/405/Allow、busy、step work+marker commit/rollback、四类 CLI 非零退出、invalid boolean、raw+flags/work-marker argv、空/非 WAV/超限输出、unknown voice、base/custom 缺 Index fail-closed、zero-rate no-index、非空 Index+正数 rate argv、旧训练三产物回滚、新声线无 partial artifact、成功链（默认） | **部分**：另有 `test_web_job_lifecycle` 的 17 个 post-reservation 单次分配故障、启动失败 503/可重试、假 CLI error 终态和 staging 清理；已消除 shell filter/index/旧产物/unknown voice/ignored cleanup flag/marker 假成功；`test_web_process_contract` 已覆盖runner自然launch/exec/signal结果与回收；另有`test_web_result_contract`的signal/wait/exit0+I/O三类模拟结果×7消费路径（21失败+21正常重试）；底层真实wait/I/O仍缺 |
| WEB-FR-011 Training cancellation | service-instance+generation job_id、训练owner、可唤醒等待、阶段/完整迭代协议、取消API与UI；页面载入重新跟踪活动train | `test_web_train_cancel`阶段/id/错误竞争及full恢复路由；`test_web_train_cancel_commit/publish`；UI刷新/迟到响应竞态 | **编排已实现·host证据**；真实CLI取消/恢复有资产门禁，Web中间层使用runner替身；无timeout/强杀、服务端作业自动恢复或崩溃原子性保证；ADR Proposed |
| WEB-FR-012 Audio normalization | 四入口`normalize_uploads`、FFmpeg CPU/argv、demuxer/file白名单、PCM16/44.1kHz/mono、64/128MiB、120s/2GiB、decoder mutex；UI序列化上传与状态 | `test_web_audio_upload`四入口/失败保留旧代/超限/busy/缺依赖/超时wrapper回收；`test_web_audio_ui`格式及等待/失败/过期/重试；可选`test_web_audio_codecs`真实1秒人声12组codec与无音轨/错格式 | **已实现·host与codec证据**；WAV原样兼容；codec完整性依赖部署FFmpeg，非完整声线转换或听感验收 |
| WEB resource boundaries | `HttpTransport`：64 KiB header、128 MiB body、256 MiB inflight、64 MiB file/缓冲音频响应、16 KiB field、128 parts、32 connections、deadlines | 超限 preview/preprocess/convert、16 KiB field/128 parts 上限及超限、multipart boundary/quoted-string/framing 为真实 HTTP 默认门禁；128 MiB+1 声明提前 413、两份 128 MiB 声明占满预留后 503、EOF/RST 后容量恢复和 socket 关闭、33 串行 EOF 连接复用及无 job/material 副作用 | **部分**：最多3客户端、body215 bytes；另有53 bytes/最多2客户端真实10秒header门禁，400关闭且job/CLI不变；完整大body、header/file上限、同时连接上限仍缺；120秒body已有独立opt-in门禁（声明1024/实传216 bytes，120708 ms，400/EOF及后续200） |
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
| RT-FR-018 Offline compatibility | 既有 Spec 001/API/CLI/CTest 在未来迁移中保持 | 加固前 30/30 为历史；2026-08-12 历史 39 distinct（16 contract-labelled、24 integration-labelled、1 overlap）已全量通过；尚未发生 realtime 产品迁移 | **提案约束**：本轮仅加固离线，不改变 realtime 实现状态；未来迁移仍须对对应待合入工作树 fresh 复验 |

## I/O、输入边界与安全

| ID / 约束 | 当前实现位置 | CTest / 其他证据 | 状态与缺口 |
|---|---|---|---|
| IO-001：WAV 读写、未知 chunk、声道下混 | `io/audio_io.*` | `test_audio_io`（默认） | **已实现·默认门禁** |
| IO-002：重采样参数边界 | `io/audio_io.*` | `test_audio_io`（默认） | **已实现·默认门禁**；质量为线性重采样，未声明高保真指标 |
| IO-003：FLAC/MP3 | 无解码实现；pipeline 仅接受 `output_format=wav` | pipeline contract（资产门禁）、支持矩阵 | **不支持** |
| IO-004：safetensors shape/dtype/offset/payload | `io/safetensors.*`、`training/checked_weights.h`；VITS 选择 capped anonymous snapshot | `test_audio_io` parse/snapshot/FIFO/ABA/truncate/read-only（默认）；训练 rollback 资产门禁 | **已实现·默认门禁**：通用 loader 仍 file-backed；VITS-only snapshot 上限 256 MiB，单 fd 身份前后复验 |
| IO-005：Index header、维度、范围、k、move/snapshot | `index/cuda_flat_index.*`、ADR-0005 | `test_index` FIFO/truncate/overwrite/exact/move（默认）；真实 retrieval（资产门禁） | **已实现·默认及可选门禁**；private unlinked filesystem snapshot 保持 exact mmap 热路径；copy capacity/未强制注入 reflink fallback 明示 |
| SEC-001：CUDA allocation/copy 失败不可继续 | `CudaBuffer`、RMVPE/VITS ops checked launches、test-only fault macro | `test_cuda_failure` + `test_synth_cuda_failure`（默认）；真实模型正常路径（资产门禁） | **部分**：代表性 allocation/H2D/D2H/pre-launch/sync/public boundary 已注入并验证 RAII；不是所有 kernel/library/OOM 路径 |
| SEC-002：初始化失败不保留半加载状态 | pipeline、RMVPE/FCPE、VITS 与训练 staged/reset guards；VITS anonymous snapshot/path identity；Index snapshot transaction | F0/VITS/Index 默认坏输入与 replacement；pipeline/training rollback（资产门禁） | **已实现·默认及可选门禁** |

## 构建与验证

| ID / 约束 | 当前实现位置 | 验证 | 状态与缺口 |
|---|---|---|---|
| BUILD-001：CMake 3.24、C++20/CUDA20、显式 CUDA 架构 | 根 `CMakeLists.txt`，默认 SM87 cache | configure/build命令；专属OrinNX手动CI路径（本增量未运行） | **已实现**；非SM87产品支持未知 |
| BUILD-002：核心库与 CLI 目标 | 显式 source list + 12 个内部 OBJECT 组件；`voxmutatio` static compatibility facade；四个按命令隔离的 `resonantia_vc_*_app_support` 与 `resonantia_web_support`；既有 CLI targets | fresh build、CMake graph、静态库符号集合与 CLI/HTTP contract | **已实现**；内部 target 不构成新公共 ABI |
| BUILD-003：可复现默认 contract suite | `cmake/ResonantiaTests.cmake`；固定 Catch2 3.4.0 fallback | 当前本机Node/模型/fixture齐全、codec ON、slow OFF：62 distinct / 35 contract / 28 integration / 1 overlap | **已实现·默认及可选门禁**；各修复阶段精确命令/结果见 [PROJECT_STATE](./PROJECT_STATE.md)；缺Node的4项UI门禁和缺资产未注册项不可计通过 |
| BUILD-004：模型/fixture 测试不得假通过 | `VOXMUTATIO_ENABLE_MODEL_TESTS`、每测试 `ASSETS` 注册检查 | configure 输出缺失资产原因 | **已实现** |
| BUILD-005：GPU 测试串行与 timeout | CTest `RESOURCE_LOCK voxmutatio_gpu`、每测试 timeout | CTest properties | **已实现** |
| BUILD-006：自动检查与自托管runner安全边界 | `.github/workflows/ci.yml` | push/PR只用托管Ubuntu；设备需manual/main/opt-in/ready且固定专属标签；24种事件组合静态验证 | **已实现**；标签为配置约定，runner身份/唯一归属未核实，设备测试未运行；无通用ARM64兜底 |
| BUILD-007：SDD/Codex bridge | 根 `AGENTS.md`、`scripts/check_sdd.sh`、SDD 索引与 ADR 模板 | repository-policy + 本地 32-required-file/link contract | **已实现**；ADR-0004/0005 追踪本轮离线迁移；ADR-0002/Spec006 实现状态未改变 |
| BUILD-008：无CUDA的托管host子集 | `cmake/host-tests/CMakeLists.txt`，10项Web/CLI/IO门禁；共享`ResonantiaCatch2.cmake`依赖策略 | 独立C++20构建与串行CTest；完整性差异见验收计划§9，运行结果见对应SHA的host-contract-tests日志 | **限定子集**；真实CLI的两项仍需完整CUDA链接；无模型/GPU，受限rename探针不执行 |

## 更新规则

1. 行为、默认值、支持矩阵、CMake 注册或测试资产改变时，同一变更更新对应行。
2. 从“部分”升级前，必须同时给出实现位置和已注册门禁；可选资产门禁不能升级为默认门禁。
3. 删除或改名需求时保留旧 ID，标记 superseded，并链接替代 ID；不要复用 ID。
4. 测试失败时状态不得保持“已验证”；记录失败边界或降级为“部分/未知”。
5. 性能、音质和稳定性结论不写入本表，除非仓库保存了环境、输入、命令和结果。

### WEB-FR-011 / TR-FR-015 后续增量

`WebService::TrainingCancelLease`绑定本代token；完整迭代协作取消只识别对应vc_train的正常退出2，
真实process/channel错误优先。`test_web_train_cancel`新增token送达/忙状态/旧id隔离/exit2消费；
`test_training_cancel`覆盖host共享loop与实际预CUDA入口。协议与限制见[DESIGN](../docs/DESIGN.md)。
旧v9“无CLI循环内接口”历史结论已由本编排增量替代；真实GPU验证、resume与既有独立缺口仍未完成。

独立审查修复：`tools/vc_train.cpp`补普通C++异常catch边界，诊断异常吞掉后仍返回1；
`test_vc_train_entry`用同一main+host应用替身验证8种退出/栈展开/诊断情形（旧入口红测86，
修复后8/8 case；CLI相关3/3门禁，3.18秒）。此证据不覆盖noexcept析构/进程崩溃/真实GPU致命错误。

## 2026-10-29 CyberTom Web发布增量

| 契约 | 实现 | 验证 |
|---|---|---|
| Spec004 立体声清理 | io/audio_io、vc_preprocess `--preserve-stereo` | test_audio_io、test_cli_contract；28秒真实karaoke结果对已听审参考相对L2 5.34e-8 |
| Spec001 人工局部音高 | core/pitch_edit.h、bounded_pipeline、vc_convert | test_bounded_pipeline_dsp、test_cli_contract；已接受28秒修正版逐采样一致 |
| Spec003 正式发布/候选 | web_service、workflows.js | test_web_audio_upload、test_web_train_cancel；真实正式发布文件校验 |
| Spec003 歌曲工作流/输入锁定 | web_service、app.js、workflows.js | test_web_audio_upload、test_web_audio_ui.cjs；真实6秒HTTP歌曲混音；可选test_web_workflows_browser.cjs |
| Spec003 完整训练界面 | presets.json、app.js、workflows.js | test_web_joint_ui.cjs、test_web_train_cancel_ui.cjs、真实浏览器候选/日志/默认值 |
| Spec003 正式声线改名与旧版清理 | Tom_Cybersinger发布目录、app.js、presets.json、workflows.js | test_web_joint_ui.cjs唯一专用预设；可选浏览器单声线/名称/请求字段检查；实际删除清单及SHA见发布记录 |
| WEB-FR-013 工作室业务交互 | index.html/styles.css、studio.js、app.js、workflows.js | test_web_studio_browser.cjs与独立host runner；既有Node/Web门禁；审计与部署见WEBUI_STUDIO_20261029 |

完整命令、资产边界及本次结果见[发布记录](../docs/WEBUI_RELEASE_20261029.md)。
