# 任务重命名与双音频结果（2026-10-29）

归属[WEB-FR-014](spec.md#web-fr-014任务显示名称与双音频结果2026-10-29)。沿用现有任务记录和
阶段音频，提供显示名称修改，以及完整翻唱/转换后无伴奏人声的默认试听与下载。

## 行为和迁移

- 任务详情名称旁可重命名，活动任务、训练、暂停和完成历史均支持；原素材/训练目录名、进度、
  参数、声线和已保存音频不改变。名称有UTF-8/长度/控制字符校验；并发改名冲突可见，重试幂等。
- 歌曲默认输出“完整翻唱 + 干净人声”，成功后直接显示两个播放器与独立下载；不会重复分离或
  推理。显式仅人声与无伴奏输入只展示人声，预处理只展示处理后音频。原始result_url保持兼容。
- 清理混音任务冗余副本保留独立人声。旧任务只要原conversion阶段还在即可展示；已删或损坏时
  提示，不能保证恢复已经删除的人声，也不在后台自动重算。
- 显示名称用既有parameters可选字段；不改变文件格式或编排版本。训练恢复身份排除显示字段。
  旧服务可读新记录，但不显示名称且旧清理规则会删独有人声；回退时不得执行混音任务清理。

## 验证记录

证据根`runs/Web_Task_Results_20261029/evidence/`。真实测试使用用户已完成歌曲的隔离副本，
原任务`web-1fcd7cfce2050a8e3fc686caee99dfb6`，不是新增模型转换或音质验收。

- `red-tests.log`：focused 0/2，旧接口无双结果且清理会删独有人声。`green-tests.log`随后2/2，
  3.49秒；`ui-red.log`旧页面无重命名入口，`ui-green.log`新增交互通过。
- 首次完整contract 43/44、40.60秒及host18/19、7.00秒失败于新增完整训练改名恢复测试。
  原训练身份遍历所有parameters，误含display_name；修正为明确排除显示字段。原门槛不变，
  `training-rename-test.log` focused 1/1、4.01秒通过。初始化配方改用插入，保留准备期间的改名。
- `real-results-1.log`：原生vc_serve和Chromium使用真实完成歌曲副本，双输出逐字节相同、Range206、
  名称刷新及服务重启保留、token保护、清理后双结果保留均通过；原记录/结果/人声摘要不变，模型
  执行0。截图验证桌面和390px布局。该轮发生于训练身份排除修复前，音频路径随后未改变。

核心命令（运行日志和最终结果以本节后续交付记录为准）：

```sh
cmake --build /tmp/resonantia-web-release-host --target test_web_tasks test_task_storage -j2
ctest --test-dir /tmp/resonantia-web-release-host -R '^(test_web_tasks|test_task_storage)$' --output-on-failure
cmake --build /tmp/resonantia-performance-20261024.uemhTG --target test_web_tasks -j2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_web_tasks$' --output-on-failure
cmake --build /tmp/resonantia-web-release-host -j2
ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure -j1
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L contract --no-tests=error --output-on-failure -j1
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell /tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_task_center_browser.cjs webui runs/Web_Task_Results_20261029/evidence/ui-final
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell /tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_task_results_real.cjs /home/rm01/Resonantia /tmp/resonantia-performance-20261024.uemhTG /home/rm01/Resonantia/runs/Web_Task_Results_20261029/real-results-1 web-1fcd7cfce2050a8e3fc686caee99dfb6
bash scripts/check_sdd.sh
git diff --check
```

## 最终验证与部署

完整构建`full-build-final-2.log`成功；`host-final.log` **19/19，7.80秒**、
`contract-final-2.log` **44/44，35.59秒**，均0失败/skip。不是全部76项模型/设备测试重跑。
`ui-final-2.log`覆盖名称草稿、并发改名不覆盖、XSS文本、默认双音频、鉴权、关闭页面/断网恢复、
320/390/768/1440布局与axe。命令与上述相同，仅输出目录改为`evidence/ui-final-2`。
`real-results-2.log`使用最终程序再次完成真实HTTP/浏览器验收；命令同上，隔离根改为
`runs/Web_Task_Results_20261029/real-results-2`。服务重启后显示名称保留，原内部name不变；
mix/vocal下载摘要与原文件一致、Range206、未认证401、清理后双结果保留，模型执行0。

确认无活动任务/计算进程后，于19:08:34 CST更新vc_serve并重启用户服务，PID2339081/active。
`deployment.json`核验测试/部署程序相同，原两个完成任务记录、源/结果/人声/identity、正式模型
及四个计算CLI摘要保持。回退程序/服务配置及经先前部署摘要复核的旧页面位于
`runs/Web_Task_Results_20261029/deployment-backup/`；空闲时停服务，恢复程序和页面，再启动，
不删任务数据，旧版不能执行混音任务副本清理。

部署后原生LAN浏览器`production-browser.log/.json`验证用户已完成歌曲显示两个完整时长播放器、
独立下载和名称编辑入口，桌面/390px无溢出；只打开并取消编辑，生产写请求0，未更改用户名称。
命令：

```sh
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell /tmp/resonantia-web-release-browser/node_modules/node/bin/node runs/Web_Task_Results_20261029/evidence/production-browser.cjs
```

`sdd-final.log`记录32项SDD检查通过，`diff-check-final.log`无空白错误。本轮没有运行新的转换、
长训练、自动重命名生产任务、Git提交或推送。保留全部既有未提交修改与用户资产。

## 任务管理变更提交前验证

2026-10-29 按用户要求将任务生命周期、进度/恢复、参数版本复用、升级兼容、重命名和双音频
结果及其测试、文档、CI一起提交。独立的中文上传文件名修复和私人音频、模型、临时诊断工具
留在工作区，不混入本次提交；不重启生产服务，也不新增转换或训练。

以 Git 暂存区导出的干净源代码副本 `/tmp/resonantia-task-release.mvos41m0` 验证，证据根为
`runs/Git_Task_Management_20261029/evidence/`。这与上节既有工作树的设备构建验收分开记录。

```sh
cmake -S cmake/host-tests -B build-host-ci -DCMAKE_BUILD_TYPE=Release
cmake --build build-host-ci --parallel 2
ctest --test-dir build-host-ci --output-on-failure --parallel 1 --no-tests=error
export NODE_PATH=/tmp/resonantia-web-release-browser/node_modules
export CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell
for test in test_web_auth_ui test_web_audio_ui test_web_joint_ui test_web_train_cancel_ui; do
  /tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/$test.cjs
done
/tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/run_web_studio_browser.cjs /home/rm01/Resonantia/runs/Git_Task_Management_20261029/evidence/studio
/tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_task_center_browser.cjs webui /home/rm01/Resonantia/runs/Git_Task_Management_20261029/evidence/tasks
bash scripts/check_sdd.sh
git diff --cached --check
```

全新 host 配置/构建成功，CTest **19/19，4.63秒，0失败/skip**；4项界面契约、工作室及任务中心
Chromium测试全部通过。SDD检查通过32项。未在该干净副本重跑GPU/真实模型测试，不能把这些
检查或推送时默认跳过的手动CUDA CI算作通过；远端CI状态以本次提交对应的运行记录为准。
