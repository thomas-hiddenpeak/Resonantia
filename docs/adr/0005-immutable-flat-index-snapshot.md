# ADR-0005: Flat Index 私有快照与原子发布

**Status**: Accepted
**Date**: 2026-08-12
**Accepted**: 2026-08-12（维护者要求实时范围以外的已知可靠性问题全部收口，并批准在不改变
精确检索数值语义的前提下消除可变 mmap）
**Implementation**: 2026-08-12 current tree 已实现 snapshot loader 与 atomic/deterministic
`build_index`；Release/SM87 当前工作树全量 39/39 tests 已通过
**Owners**: Resonantia maintainers
**Governing specs**: [Spec 001](../../specs/001-voice-conversion-engine/spec.md)、
[Spec 002](../../specs/002-voice-training/spec.md)、
[ADR-0004](0004-bounded-offline-file-conversion.md)

## Context and decision drivers

`CudaFlatIndex` 原先直接 `mmap(MAP_PRIVATE)` 调用方给出的 `.index`，随后关闭 fd。该映射并不
是内容快照：无 writer 的 FIFO 会在 `open(O_RDONLY)` 阶段永久阻塞；成功加载后，同 inode
原地 truncate 会让下一次 exact scan 收到进程级 `SIGBUS`，C++ 异常或 `VCResult` 都无法拦截；
同尺寸覆盖还会在初始化校验后静默改变活动检索库。仓库自己的 `build_index` 又直接打开最终
路径写入，因此并发重建不是纯恶意场景。

用户已明确要求保留 Index 效果与竞争力。修复不得把正 `index_rate` 静默降为 no-index，也
不应顺手改写 exact L2、`partial_sort`、RVC 权重或相同距离行为。

## Decision

### 1. 加载为私有 filesystem-backed snapshot

`CudaFlatIndex::load` 以 `O_RDONLY|O_CLOEXEC|O_NONBLOCK` 打开源，先 `fstat` 并拒绝非 regular
file，再校验 little-endian header、精确字节数和 checked shape。加载事务随后：

1. 优先在源目录创建 mode `0600` 的 `O_TMPFILE`；不可用时在 `/tmp` 以
   `mkostemp(O_CLOEXEC)` 创建并立即 unlink；
2. 优先以 `FICLONE` 建立 CoW snapshot；不支持 reflink 时以 1 MiB bounded buffer
   `pread`/`pwrite` 全量复制，并再次逐字节核对源与 snapshot；
3. 复制前后比较源 fd 与当前路径的 device/inode/size/mtime/ctime，任一漂移使事务失败；
4. 只读 mmap 私有、已 unlink 的 snapshot，扫描全部 payload finite 后才提交对象状态。

搜索、重建和 weighted retrieval 继续读取这份 mmap，并保持原 exact float 累加与 top-k 代码。
源路径在成功加载后被替换、truncate 或覆盖，不会改变活动 snapshot；显式重新 `load/init` 才
采用新版本。这样运行中不会因外部 source mutation 崩溃，也不会产出混合版本检索结果。

“filesystem-backed”不等于必然占用物理磁盘：若选中的 filesystem 是 tmpfs，其容量计入该
系统的内存/交换策略。无 reflink 时需要约 `O(Index bytes)` 的临时 filesystem capacity；
ENOSPC、权限或 snapshot 创建失败均使 `load` 返回 false，Index-required pipeline 继续
fail-closed。该方案不创建 index-sized anonymous-memory duplicate；mmap 页仍会按扫描访问
进入进程 RSS/page cache，和旧 exact validator 的访问模式一致。

### 2. `build_index` 原子发布且输入顺序确定

构建器先按 normalized path 排序 WAV，再提取特征；最终 Index 在目标同目录以
`O_CREAT|O_EXCL|O_CLOEXEC`、`0666 + caller umask` 创建唯一 staging。完整 header/payload
写入、`fsync` 和 checked close 成功后才以 rename 替换目标，父目录 fsync 为 best-effort。
rename 前失败清理 staging 并保留旧目标。未知参数和缺失 option value 必须非零退出，不能
进入模型或发布阶段。

## Alternatives considered

- **继续 live mmap，只在搜索前后 fstat**：检查与页访问之间仍能 truncate，无法消除 SIGBUS；
  拒绝。
- **把整个 Index 复制到 anonymous RAM**：简单但给大 Index 增加等量常驻匿名内存，削弱
  Jetson 上可用规模；拒绝。
- **每次检索从源 fd bounded pread，并重写 streaming top-k**：可避免 snapshot filesystem
  容量，但会改变缓存、tie、复杂度和延迟特征，需要独立性能 ADR 与规模基准；本次最小安全
  修复拒绝。
- **只修 build_index 为 atomic rename**：能保护合作发布者，不能阻止其他工具或管理员对
  已 mmap inode truncate；不足以关闭库边界。

## Consequences

优点是 active Index 字节不可外部修改，精确检索数值和热路径保持不变，正 index rate 不会被
静默旁路。代价是 load 多一个 snapshot 事务；reflink 不可用时增加全量复制时间和约一份 Index
的临时 filesystem 占用。只读/空间不足环境可能从旧版可 mmap 变成明确 load failure，部署需
为 snapshot capacity 留预算或提供支持 CoW clone 的可写 filesystem。

该决策只加固现有离线 host Flat Index。它不实现 Spec 006 的 persistent GPU Index、实时
scheduler 或实时 fallback。

## Migration and rollback

公共方法、文件格式、exact retrieval 和 `index_rate` 契约不变；消费者只需重编译当前源码树。
若 snapshot 创建失败，初始化失败且不提交部分对象。若性能或容量回归超过部署预算，可回滚
到上一个已验证对象，但不得恢复 live mutable mmap；后续替代必须同样消除 SIGBUS，并经真实
Index 数值/规模基准批准。

## Verification and SDD impact

- `test_index` 以 re-exec 证明 no-writer FIFO 在 2 秒内拒绝，并把旧 truncate `SIGBUS` 红灯
  转为 active snapshot 继续返回原 exact 结果；同尺寸覆盖后结果也保持不变。
- `test_index_retrieval` 使用真实 HuBERT feature fixture 锁定 weighted retrieval 与 blend。
- `test_build_index_contract` 使用真实 HuBERT/语音，锁定 hard-link 旧 inode 保留、输出格式、
  不同创建顺序下逐字节确定性和 staging 清理。
- Spec 001/002、DESIGN、TRACEABILITY、PROJECT_STATE 与 CLI 文档必须同步 snapshot 容量、
  原子发布和验证边界。reflink fallback 分支若当前 filesystem 未触发，应诚实保留为静态审查
  支持而非已强制注入证据。
