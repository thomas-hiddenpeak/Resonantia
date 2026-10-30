# 引导式局部修音实施记录（2026-10-29）

状态：2026-10-30 技术门禁通过，已部署；维护者对41–49秒真实修正对照反馈“确实改善了”。维护者已批准按此前五步交互方向推进。
架构边界见 [ADR-0011](../../docs/adr/0011-guided-pitch-review.md)。

## 交付与验收

1. 历史转换任务直接读取音高证据，自动标记疑点；普通用户无需先选择算法或半音数。
2. 疑点可带上下文循环听原声/转换结果，曲线清楚区分检测值、建议及无音高区。
3. 可试听单项建议的修正前/后，再采用/忽略；采纳不自动执行整曲。
4. 预览持久保存、可关闭页面，并复用既有任务进度/恢复/资源限制。
5. 原任务与模型不改写；完整参数/输入/程序验证不放宽。缺数据明确说明而非自动昂贵重算。

## 实施顺序与边界

- P1：I/O 只读 inspection + F0 纯 CPU 曲线/疑点分析；先加红测再实现。
- P2：任务只读报告 API、原/转换局部循环、曲线、建议与高级编辑整合。
- P3：原配方兼容校验、独立缓存派生、预览后台任务及双音频结果。
- P4：真实缓存和短模型对拍、完整相关测试、内网验收、部署与回退记录。

预览技术边界：仍以整曲上下文运行已部署的 vc_convert，但仅重算受局部 F0 变化影响的
原合成块；全局 RMS 完成后再裁出最多 15 秒听审音频。不是截短后另算 F0，也不是无条件
重算整曲。最多两个受影响合成块；源缓存不完整或模型/参数不相容时拒绝快速预览。

现有脏工作树包含上一轮已部署的修音入口修改及未提交诊断工具，全部保留。
当前生产计算程序摘要和真实资产需在新服务部署前后核验；不得把替身浏览器测试当作模型验证。

## 验证证据

下列日志保存精确命令对应的最终结果；首次失败另列，不计作通过。

## API 与用户操作

任务中心 → 已完成的转换 → **局部修音**。报告加载只读；选择疑点后可在原输入/成功版人声
之间保持位置切换并循环。近八度短暂误判提供“先试听建议”；不确定换音只有定位提示。
建议试听完成后可听修正结果，再采用、忽略或撤销。高级手动区间折叠保留。表单参数或
手动区间与基线不同时会阻止自动预览/采用，并提供显式恢复成功版设置。

| 接口 | 契约 |
|---|---|
| `GET /api/task/pitch?id=...` | 成功整段转换，或失败派生对应的成功父版；返回 baseline_id、analysis_id、duration、start_time、step=0.01、raw、processed、至多64条 issues（reason、可选 shift suggestion）及 baseline_edits；每条曲线至多61100帧 |
| `POST /api/tasks/pitch-preview` | multipart task_id/source_task/analysis_id/pitch_edits/preview_start/preview_end；只允许局部改动，不接受声线/整体参数/新上传/重复字段；202后由任务中心管理 |
| `GET /api/task/result?id=...&variant=baseline` | 已成功局部试听的修正前片段；摘要校验后支持 Range，修正后沿用默认 result/vocal |

范围单位均为原上传音频秒；输出至多15秒，变化须落在范围内，最多两个受影响原合成块。
新恢复配方为 `audio-pitch-preview-v1`，关闭页面、暂停和重启后仍可查询/继续；预览不能
作为整曲派生来源。坏 ID/字段/范围等返回400，记录不存在404，缺数据/身份漂移409，
任务存储不可用503。缺缓存不会改写原任务，也不会触发整曲退化执行。

没有曲谱或可靠旋律参照时，不能自动判断任意一个半音是不是唱错。当前仅生成启发式
八度短暂跳变建议，不强制拉到音阶，也不把曲线平稳或没有疑点当作音质合格。
本机完整来源预览需验证模型身份并重算完整上下文，耗时不等同于输出8秒的长度。

## 真实证据与失败修复

- 从正式《富士山下》主唱取8秒，仅在隔离 Web 目录调用正式 Tom 模型。
  基线音高报告定位局部3.36–3.54与5.68–5.73秒的低八度疑点。
- 首次独立重算对拍发现试听与结果相差1 LSB；完整生成波形已经一致，差异来自裁剪的
  重复PCM量化。新增精确 PCM16 裁剪后，**320000样本/40000Hz全部一致**。
  `real-parity.json` 保留首次失败，`real-parity-fixed.json` 为修复结果。
- 同一短预览以新恢复配方暂停，停止隔离服务、重新打开后仍为 paused，attempt2 恢复完成；
  `real-preview-paused.json`、`real-preview-after-restart.json`、`real-preview-v2-status.json` 保留全过程。
- 完整《富士山下》父任务副本报告20个疑点，其中43.36–43.54及45.68–45.73秒与前次
  人工诊断对应。只创建41–49秒试听；33个F0块全部复用，9个合成核心中8个payload不变，
  仅 `synth_494720` 重算，输出320000样本。含身份核验墙钟167.631秒；未重新全量执行
  整曲分离或模型转换。`fushi-cache-verification.json` 记录计数与阶段日志。
- 原生产父任务对比试听前建立的独立副本，332份文件摘要全部一致；模型及计算CLI摘要
  单独保留。核验脚本首次误用了Python3.11的file_digest（环境3.10），随后用有界流式
  SHA256完成对照，未将失败核验记作通过。
- 首次全量编译发现 Web 头引入的 VCConfig include 路径破坏独立生命周期测试，改为前向声明；
  测试补充发现音频片段 source_format 未反映PCM16，已修正。相邻上下文两块失效、超过
  两块拒绝、损坏块、变化越界/过期分析、无改动、失败父版回退均有确定性回归。
- 可用性门禁覆盖读取不提交、建议试听、采用/撤销、参数漂移阻止、无建议/缺缓存、带令牌
  音频、页面重新打开、320/390/768/1440布局与axe规则。替身PCM只验证交互，不证明模型质量。

证据根：`runs/Guided_Pitch_Review_20261029/evidence/`（本地生成资产不提交Git）。
维护者确认41–49秒对照有所改善，不代表全曲或通用错音判断通过；随后授权
[页面流程补齐](pitch-review-workflow-20261030.md)。未跑长训练、整曲独立重算、全部模型集成/slow测试或远端CI。


## 最终命令与部署

```sh
cmake --build /tmp/resonantia-web-release-host -j2
ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L contract -LE model --output-on-failure
scripts/check_sdd.sh
git diff --check
```

- host **21/21，5.81秒**；contract（排除model标签）**45/45，35.11秒**；均0失败/skip。
  日志 `verified-host-{build,tests}.log`、`verified-native-build.log`、`verified-contract-tests.log`。
- `test_bounded_pipeline_dsp` 对 filter0/3/5/31、变调及局部修音精确对拍；
  `test_pitch_review_io` 含相邻上下文、最多两块、精确裁剪及API失败边界。
- 四项Node UI契约、工作室/任务中心Chromium、引导式听审Chromium均通过。
  引导式脚本命令（其余两个按相同环境运行已有runner）：

```sh
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules \
CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell \
/tmp/resonantia-web-release-browser/node_modules/node/bin/node \
tests/test_web_pitch_review_browser.cjs \
runs/Guided_Pitch_Review_20261029/webui \
runs/Guided_Pitch_Review_20261029/evidence/browser-release
```

- 2026-10-30 01:02:15 CST，无生产活动任务时，只替换 `build/vc_serve` 与静态页面并重启
  `resonantia-web.service`，PID2600879，active。`vc_convert`、`vc_preprocess`、Tom模型/索引
  SHA256前后相同。未提交Git、未推送或触发远端CI。
- 内网 `http://192.168.0.122:8080/` 真实浏览器验证：20疑点、43.36/45.68定位、原生
  局部播放、390px布局、失败父版回退与显式恢复参数；0写请求，0页面异常。
  `lan-browser.json` 与桌面/移动截图留证。生产没有新增转换任务。
- 回退保留 `runs/Guided_Pitch_Review_20261029/vc_serve.before` 和
  `webui-deployment-swap/`；原记录和音频不删除。旧服务不恢复新的预览配方。
- 本地短试听：`listen_fushi_before_41_49.wav`、`listen_fushi_after_41_49.wav`。
  它们来自隔离任务的精确8秒裁剪，不覆盖用户整曲结果；`fushi-excerpt-exact.json`核验
  修正前/后均与对应完整输出的选中PCM区间一致。
