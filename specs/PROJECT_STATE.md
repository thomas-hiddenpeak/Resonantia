# Resonantia 项目状态

**状态日期：2026-08-11**

本文件是当前实现与验证结果的**描述性快照**，不是替代 feature spec 的规范来源。
规格意图和需求归属见 [`specs/README.md`](README.md) 及各 feature spec；可执行边界以
`docs/DESIGN.md`、CLI `--help`、当前 CMake/CTest 注册和实现中的输入校验为准。
未来目标、历史研究结论和配置结构中的预留字段都不等于已支持能力。

## 验证基线

- 2026-08-11 在 Jetson Orin NX Engineering Reference Developer Kit（SM87）、
  CUDA 13.2.78、CMake 3.28.3 上，以 `/tmp/resonantia-audit-final.VbIWW3` 完成 current-tree
  out-of-tree configure/build；clean build 为 **213/213** steps，终审修正后受影响目标
  重新构建并由 `ninja: no work to do` 复核同步。默认无模型资产的 contract 门为 **11/11**，
  其中 7 项使用 GPU lock，另有真实 CLI 进程、真实 HTTP socket、legacy Web API 与 realtime
  报告 validator 四项 host contract。
- 启用 `VOXMUTATIO_ENABLE_MODEL_TESTS=ON` 且 `models/` 与 `tests/fixtures/` 中 19 个
  integration 用例声明的资产齐全时，current-tree 完整验证为 **30/30 CTest 通过**
  （11 个 contract、19 个 integration、0 失败，real 335.05 秒）。复现命令为：

  ```bash
  cmake -S . -B <build-dir> -DCMAKE_BUILD_TYPE=Release \
    -DVOXMUTATIO_CUDA_ARCHITECTURES=87 \
    -DVOXMUTATIO_ENABLE_MODEL_TESTS=ON
  cmake --build <build-dir> --parallel 2
  ctest --test-dir <build-dir> --output-on-failure
  ```
- 模型权重和数值参考 fixture 不入 Git。缺少资产的集成测试由 CMake 不注册，不能把
  “未执行”解释成通过。
- 当前没有可跨机器复用的产品吞吐、延迟或显存基准；standalone exact-Index probe 只有
  synthetic/available-feature stage 诊断数据，不是产品 backend、支持上限或 realtime SLO。
- Spec 006 的 [R0 asset manifest](006-realtime-voice-conversion/r0-assets.json) 已对 7 个 present
  文件、2 个本地 conversion inputs、3 个 generation scripts 和 `tools/uv.lock` 完成
  path/bytes/SHA-256 核验；FCPE official package/checkpoint provenance、locked project 与
  same-filesystem staged + rollback 事务式发布已固定。qualification gate 现在只因
  provenance-tracked 目标声线 v2/40 kHz
  model，以及与同一份有授权 corpus 匹配的 Flat Index 缺失而按设计失败，不能解释为
  realtime 资产已就绪。
- SDD/格式门禁同日复核：`bash scripts/check_sdd.sh` 的 30 个 required files 全部通过；
  `git diff --check` 退出 0 且无输出。
- ADR-0001 R0–R5 已实施：12 个内部 OBJECT 组件聚合为公共 `voxmutatio` static facade；
  四个主要 CLI 分别使用内部 command-scoped app-support target；`vc_serve` 使用内部 Web
  support 模块。既有 target 名、安装面和公共库 API 未因重构扩大。

## 当前能力矩阵

| 模块 | 当前状态 | 事实边界 |
|---|---|---|
| WAV I/O / 重采样 | 已实现 | WAV 读写、声道下混、线性重采样；不解码 FLAC/MP3 |
| safetensors | 已实现 | C++ 运行时读取；PyTorch 权重仅可离线转换 |
| HuBERT | 已实现、默认 | v1 输出 256D，v2 输出 768D |
| WavLM | 已实现、可选 | 需要与 WavLM 特征匹配训练的 synthesizer；不是默认 RVC 路径 |
| RMVPE | 已实现、默认 | 完整运行时 tensor manifest、有限值/输入边界和失败重初始化均有门禁；持久共享只读 checkpoint 映射 |
| FCPE | 已实现、可选 | 完整 tensor manifest、weight-norm 安全物化、有限值/输入边界和失败重初始化均有门禁；R0 已固定精确来源与 same-filesystem staged + rollback 事务式发布，不据此承诺实时性能 |
| Flat Index | 已实现 | 产品实现为 host 内存映射、精确加权 k-NN；离线 Pipeline 对非零 rate fail-closed；standalone GPU probe 不构成产品 backend |
| F0 VITS v1/v2 | 已实现 | CUDA FP32；v1/256D、v2/768D；当前仅 40 kHz；拒绝 non-F0 |
| 离线变声 pipeline | 已实现 | WAV → HuBERT → RMVPE/可选 FCPE → Index → VITS → WAV |
| 训练 | 实验性、部分完成 | `vc_train` 有 decoder/GAN 路径；旧 `FineTuneTrainer` 不支持 |
| Web/CLI 编排 | 实现存在、验收部分 | `vc_serve` 是唯一受支持 Web 入口；默认 `0.0.0.0` 且无认证；旧嵌入式 `WebServer` 已 deprecated/fail-fast；非实时服务 |
| 声源分离 | runner 实现完成、实验性 | UMX、MelBand-RoFormer、Silero VAD；完整已用权重及公开 finite 边界有资产门禁，质量仍未产品化 |
| 端到端实时变声 | Spec 006 approved / R0 部分完成，产品未实现 | 已有 legacy `infer_stream` characterization、模式感知报告 validator、asset manifest、离线 Index fail-closed 与独立 Index 诊断 probe；仍没有 persistent runtime、session、`vc_realtime` 或 ALSA 产品路径 |

## 离线变声支持契约

| 维度 | 支持 | 明确不支持或会拒绝 |
|---|---|---|
| 设备 | CUDA | CPU 推理 |
| 精度 | FP32 | FP16 |
| 文件格式 | WAV 输入、WAV 输出 | FLAC、MP3 等 |
| 内容编码 | HuBERT（默认）、WavLM（可选） | 与 synthesizer 特征空间不匹配的组合 |
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
| 001 Voice Conversion Engine | 实现基线已具备 / 规格验证部分完成 | 库/pipeline、原子 WAV 和 CLI 严格失败有门禁；真实模型 CLI 成功、`vc_probe` 验收仍缺 |
| 002 Voice Training | 实验性 / 部分完成 | 可运行的 decoder/GAN 实验路径；无生产质量与 resume 保证 |
| 003 WebUI | 实现存在 / 验证部分完成 | 默认开放、可选 token、严格参数、step work+marker/训练三产物事务、FD 隔离与四类 CLI 失败传播有 HTTP 门禁；资源/恶意输入矩阵仍缺 |
| 004 Source Separation | runner 实现完成 / 实验性 | 模型集成、CLI model/task 拒绝和 Web step 事务有门禁；五任务质量/资产覆盖仍不完整 |
| 006 Realtime Voice Conversion | Approved / R0 evidence 部分完成 / 产品未实现 | ALSA、v2/HuBERT/FCPE/40k + 有效 Index、250/350 ms SLO、drop-oldest 已批准；独立 probe 只证明 exact retrieval 的诊断可行性。目标 model+matching Index/licensed corpus 缺失，产品 realtime/质量/SLO 仍未资格化 |

## 下一步

1. **P0：完成实时 R0 qualification**——ADR-0002 已 Accepted；`infer_stream` characterization、
   FCPE provenance/locked project/same-filesystem staged + rollback 发布、报告 validator、
   asset manifest、离线 Index fail-closed
   与独立 exact-retrieval probe 已完成。下一步取得有授权 corpus 及匹配的目标声线 v2/40k
   model+Flat Index pair，再把它们写入现有 R0 manifest，测量 target Index 的 rows/query latency、
   内存及 Index-enabled/no-index 质量 A/B，由硬数据固化 `index_rate`、rows 上限、backend、
   fallback trigger 和质量门槛。不得用 no-index 结果替代 qualified SLO，也不得先接 ALSA 再补 runtime。
2. **P1：训练产品化**——补齐持久化 checkpoint/optimizer state、断点续训、大数据集
   稳定性、失败恢复和人工/客观音色质量验证；决定是否替换或删除旧
   `FineTuneTrainer` 门面。
3. **P1：分离验证与发布**——明确权重来源、许可证、资产清单和真实数据质量基准。
4. **P1：Web 资源与作业生命周期加固**——补请求上限/slow-client、multipart 恶意输入、
   material 双目录 commit、上传/preview 竞态的真实 HTTP 门禁；另行设计子进程 timeout/
   cancel/process-group、日志总量/轮转和超过 64 MiB 音频的 streaming response，不能随意
   猜测长训练的超时策略。
5. **持续加固**——保持 contract 与 asset-gated integration 两层测试，继续覆盖输入
   上限、原子产物、回滚与服务暴露边界。
