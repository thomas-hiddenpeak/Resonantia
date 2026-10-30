# 历史任务局部修音入口修复（2026-10-29）

## 问题与范围

局部修音原嵌于 `resultSection`；历史任务派生会隐藏该结果区，导致已有修音设置虽恢复，
用户却无法看到或编辑。《富士山下》的平滑跨无声帧失败又只显示继续入口和原始日志尾部。

本次仅修改静态 Web 页面、浏览器回归和文档：独立修音区、成功任务的直接入口、已知修音
失败版本的设置恢复、可读错误和无效原参数重试提示。不改变 API、计算 CLI、F0 算法、
任务存储或模型。自动错音检测、曲线编辑及局部重生成仍未实现。

失败版本编辑通过读取原成功版本作为 `source_task`，沿用失败版本的完整可编辑转换设置。
点击进入不提交任务；用户明确应用后才创建新 ID。原失败记录、原成功结果与音频不改写。
源版本缺失明确报错，不退回其他上传会话。执行中的任务不开放编辑操作。

## 验证记录

证据目录：`runs/Web_Pitch_Access_20261029/evidence/`。隔离静态目录：
`runs/Web_Pitch_Access_20261029/webui/`；生产页面在验证后再替换。

浏览器依赖为既有锁定 Playwright/axe，Node 与 Chromium 路径如下：

```sh
export NODE_PATH=/tmp/resonantia-web-release-browser/node_modules
export CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell
/tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_task_center_browser.cjs runs/Web_Pitch_Access_20261029/webui runs/Web_Pitch_Access_20261029/evidence/green
/tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/run_web_studio_browser.cjs runs/Web_Pitch_Access_20261029/evidence/studio runs/Web_Pitch_Access_20261029/webui
for test in auth audio joint train_cancel; do
  /tmp/resonantia-web-release-browser/node_modules/node/bin/node "tests/test_web_${test}_ui.cjs" runs/Web_Pitch_Access_20261029/webui || exit
done
```

- 红测 `red.log` 在旧页面断言失败：历史任务的 `addPitchEdit` 不可见。
- 首轮修复 `green.log` 通过：普通派生、直接修音、失败版本刷新后恢复、来源缺失拒绝、
  不自动提交、保留失败参数、新 ID/成功父版本来源、明确应用后的完整修音字段。
- `studio.log` 工作室浏览器流程通过；`ui-contracts.log` 四项既有 UI 契约通过。
- 浏览器使用替身 HTTP/PCM，含 320/390/768/1440 宽度与 axe 检查，不代表声音质量验证。

最终增量 `final-browser.log` 通过，另覆盖已知错误原参数继续被禁用；对应截图在 `final/`。
主机回归命令及结果：

```sh
ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure -j1
scripts/check_sdd.sh
```

host **19/19通过、0失败/skip，4.81秒**（`host.log`）；此轮仅静态页面/测试/文档变更，
没有需要重新编译的目标，也未运行 CUDA/真实模型转换测试。不把浏览器替身提交计作实际转换。

内网验收运行以下命令（同上 Node/Chromium 环境）：

```sh
/tmp/resonantia-web-release-browser/node_modules/node/bin/node runs/Web_Pitch_Access_20261029/evidence/production-browser.cjs
```

`production-browser.json` 通过：实际《富士山下》失败任务刷新后出现“修改局部修音”，
恢复原来的两段平滑及正式参数；成功原版出现“局部修音”且保持空修音列表。验证阻断全部
写请求，实际写请求数0；21份任务记录/日志/音频/正式模型/程序文件摘要与部署前一致。
真实页面桌面/手机无横向溢出、无 JS 错误，未运行模型；没有修改为上一轮讨论的候选修音值。

## 部署与回退

已通过目录原子交换更新静态 `webui/`，服务保持 active/PID2339081，未重启。备份为
`runs/Web_Pitch_Access_20261029/webui-before/`；保留正式 Tom 模型、任务与所有音频。
交换后旧目录亦保留为 `webui-deployment-swap/`。测试候选仍保留在隔离 `webui/`，部署后
逐文件核对与 `tested-ui.json` 一致。回退恢复整套旧静态文件。此次没有提交或推送 Git。
