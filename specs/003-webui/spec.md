# Feature Specification: WebUI and HTTP Orchestration

**Feature**: `003-webui`
**Created from implementation**: 2026-08-09
**Updated**: 2026-08-12
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
| `GET /api/train/status` | 返回 stage/running/done/error 和受限 log tail |
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
  失败尝试恢复旧 raw/work，无法完整恢复时保留备份并记录路径。
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

仍缺自动门禁：请求大小/slow-client、multipart 恶意输入、material 双目录 commit 的
中途故障注入、路径穿越/越界 symlink，以及进程信号/启动/wait 失败传播矩阵。

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
