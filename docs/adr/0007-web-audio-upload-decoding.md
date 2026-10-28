# ADR-0007: Web 上传常见音频格式的服务端解码

**Status**: Accepted
**Accepted**: 2026-10-24，维护者明确回复“同意服务器端 FFmpeg 方案（推荐）”。
**Date**: 2026-10-24
**Owners**: Resonantia maintainers
**Governing spec**: [Spec 003](../../specs/003-webui/spec.md#multi-format-upload-web-fr-012)

## Context and decision drivers

用户要求 WebUI 兼容主要音频格式。当前浏览器两处上传筛选和 Web 的 collect_uploads
都只接受 WAV；直接上传的 preprocess/convert 路径也把字节当 WAV 写入。
模型 CLI 及 C++ audio I/O 维持 WAV 契约。仅修改 accept 字段不能完成兼容。
本机安装了 FFmpeg 6.1.1 的 libavcodec/libavformat 等共享库，但 PATH 中没有 ffmpeg
可执行文件。增加服务端解码属于新增部署/运行依赖，按 AGENTS.md 需在实施前审批。

## Decision

- Web material、带文件的 training、直接 preprocess/convert 统一使用上传归一化边界。
  初始范围为 WAV、MP3、M4A（AAC/ALAC）、AAC、FLAC、OGG（Vorbis/Opus）、Opus、
  WMA、AIFF/AIF。实际 decoder 能力须用部署版本和真实短文件验证，不仅检查扩展名。
- 通过独立 FFmpeg 可执行文件在 CPU 上解码；C++ Web 编排使用参数数组启动，不拼 shell。
  不向 CUDA inference/training 库引入 Python、FFmpeg 链接或编码容器解析逻辑。
- WAV 保留现有字节及兼容路径；新增格式在唯一 staging 中生成 PCM16、44.1 kHz、mono
  WAV，交给现有流程。输出仍为 WAV。此规则是新压缩/其他容器输入的归一化契约，
  不宣称无损保留原始采样率、声道布局或 24-bit 文件精度。
- 上传原名保留在 UI；服务端派生 `.wav` basename。不同输入归一化后重名时整次拒绝，
  不覆盖。原有磁盘素材、模型和已发布 raw/work 不作原地转换或迁移。
- 解码只读 staged 本地常规文件，限定音频 demuxer/协议和首个音轨，禁用网络访问、
  视频/字幕/数据轨、播放列表与外部媒体引用；DRM、损坏、无音轨文件明确失败。
  FFmpeg 的协议默认并非 file-only，需主动设置允许列表，见
  [官方协议说明](https://ffmpeg.org/ffmpeg-protocols.html#Protocol-Options)。
- 保留 128 MiB HTTP body 和 256 MiB inflight 限制。新增解码产物限制为每文件 64 MiB、
  每请求累计 128 MiB；达到上限拒绝整次请求，不能把截断产物当成功。每个服务实例
  最多运行一个解码进程，限制 CPU threads、子进程地址空间及 120 s wall deadline；
  限制仅用于上传解码，不改变既有模型进程或训练取消协议。
  使用发行版 coreutils `timeout` 和 util-linux `prlimit` 实施限额，地址空间2 GiB，
  FFmpeg单次分配64 MiB、decoder线程2/filter线程1；TERM后2秒以KILL收尾。
- 不可解码/不支持输入返回 400，产物过大 413，解码 busy 409，依赖不可用 503；
  解码进程失败/超时不得触发素材提交或模型调用。全批解码及校验成功后才调用既有事务
  发布 raw/work/marker；失败保留旧一代且清理暂存文件。
- 缺 FFmpeg 时 WAV 仍可使用，新格式返回可诊断错误；部署时安装/定位可执行文件并核验
  所需 decoder。不能让现有 Python 虚拟环境变成生产服务的隐式依赖。

## Alternatives considered

- 浏览器 Web Audio 解码再上传 WAV：当前已有 decodeAudioData 用于时长估计，能复用，
  但支持范围取决于浏览器/平台，长文件会增加浏览器内存压力，不作为服务器兼容承诺。
- 在核心 C++ I/O 中加入多种 decoder：扩大所有 CLI/库依赖及回归范围，超出此次 Web
  上传需求。自研各 codec 也不合适。

## Migration and rollback

1. 维护者确认本提案后，先补失败回归，再实现服务端归一化及 UI 格式/错误提示。
2. 已有 WAV 请求、raw/work/steps 事务、模型及参数保持可用；归一化后的新素材是标准 WAV，
   旧服务仍能读取。提交前的上传容器仅属于暂存，不改写用户原文件。
3. 独立构建并通过测试后，同步更新 Web 静态文件和 vc_serve；保留当前可执行物及 UI
   备份，确认无活动作业再重启，并检查原内网地址。回滚恢复两者，不删除素材/声线。
4. 验证失败时不切换服务；旧部署继续提供 WAV 功能。整个实施不跑完整音频声线转换。

## Verification and SDD impact

- 各种格式的真实短人声输入完成解码/预览，核验完整时长、sample rate/channels、有限样本、
  nonempty PCM 和命名。音频编解码测试不等于声线转换/质量验收。
- HTTP 覆盖四个入口及大小写扩展名、错扩展名/损坏/空/无音轨、重复 basename、缺依赖、
  超时/产物超限、解码失败后旧素材与 marker 保留、并发 busy 和子进程/临时资源回收。
- UI 覆盖选择/拖拽、训练多文件、转换单文件、上传中与失败后的按钮状态、明确拒绝提示。
- 跑受影响的 Web/进程/事务/UI 门禁及 asset-free contracts；使用隔离测试 repo，不触碰
  用户的 `__convert__` 会话。仅 GET 验证部署后的主页/API，由用户自行做声线转换试听。
- 批准后同步 Spec 003、DESIGN、README、TRACEABILITY、PROJECT_STATE 与部署依赖说明。
  Spec 001/002/004 的模型 CLI WAV 输入、40 kHz 模型输出契约保持原样。
