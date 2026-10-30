# 修音页面流程验收（2026-10-30）

维护者确认前次《富士山下》41–49秒对照“确实改善了”，随后要求推进WebUI修改。
本轮补齐既有 [WEB-FR-015](spec.md) 的页面流程，不改变计算程序、模型、API或任务格式。

## 页面行为

- 完成、失败、暂停或中断的短试听任务可“返回修音，继续听审”，定位成功父版的对应疑点。
- 疑点编号与状态、待听审计数、下一处待听审、修正前/后试听、采用/保留原音/撤销形成闭环。
  已采用后选择保留原音会移除该修正，撤销可恢复；键盘焦点不因列表重绘丢失。
- 浏览器保存选中位置、采用/忽略草稿及最多64个试听ID；以来源和分析身份隔离。
  历史列表没有列出试听时仍可查询找回，只按需核实当前疑点；所有结果状态重新读取服务端。
  查询核对来源、分析ID、修正配方和覆盖范围。过期、错配、父版缺失不能采用。
- 已有完成/暂停试听打开原任务，查询失败不重新提交。采用不会自动执行整曲，最后点击
  “生成完整新版（N处修正）”才创建整曲新版。原版和已保存试听保持。
- 草稿仅在同一浏览器持久保留；清除浏览器数据会丢失草稿，后台任务仍可找回。没有增加
  跨设备草稿同步、任意半音错音判断或新的信号处理能力。

## 回归与真实联调

证据根：`runs/Web_Pitch_Workflow_20261030/evidence/`，本地音频、截图与日志不提交Git。
页面先在同run的`webui/`暂存目录验证，原生产快照保留于`webui-before/`。

精确命令（从仓库根目录执行）：

```sh
export NODE_PATH=/tmp/resonantia-web-release-browser/node_modules
export CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell
WEB_TEST_NODE=/tmp/resonantia-web-release-browser/node_modules/node/bin/node
WEB_PITCH_RUN=runs/Web_Pitch_Workflow_20261030
"$WEB_TEST_NODE" tests/test_web_pitch_review_browser.cjs "$WEB_PITCH_RUN/webui" "$WEB_PITCH_RUN/evidence/browser-final"
"$WEB_TEST_NODE" tests/test_web_task_center_browser.cjs "$WEB_PITCH_RUN/webui" "$WEB_PITCH_RUN/evidence/task-browser"
"$WEB_TEST_NODE" tests/run_web_studio_browser.cjs "$WEB_PITCH_RUN/evidence/studio-browser" "$WEB_PITCH_RUN/webui"
for suite in auth audio joint train_cancel; do
  "$WEB_TEST_NODE" "tests/test_web_${suite}_ui.cjs" "$WEB_PITCH_RUN/webui" || exit 1
done
ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure
```

- 引导式浏览器最终通过；任务中心、工作室、4项Node UI通过。包括鉴权音频不泄漏令牌、
  历史页外恢复、完成/暂停复用、过期分析/来源缺失、采用/忽略/撤销、焦点和
  320/390/768/1440布局与axe规则；替身PCM不证明模型质量。
- host **21/21，5.06秒，0失败/skip**，见`host-tests.log`。
- `browser-red.log`首次复现短试听没有直接返回入口；`browser-focus-red.log`复现点击疑点
  后键盘焦点丢失。一次中间运行发现参数不匹配时复用按钮未禁用，收紧条件后回归通过。
- 使用既有8秒真实任务启动只读联调实例：

```sh
build/vc_serve --host 127.0.0.1 --port 18082 \
  --repo /home/rm01/Resonantia/runs/Guided_Pitch_Review_20261029/sandbox \
  --webroot /home/rm01/Resonantia/runs/Web_Pitch_Workflow_20261030/webui
"$WEB_TEST_NODE" "$WEB_PITCH_RUN/evidence/verify_real.cjs"
```

真实`guided-real-preview-8s-v2`返回父版、定位疑点、播放已有修正音频、采用/重开恢复/撤销、
复用同一个试听ID及390px布局通过；**0写请求、0页面异常**。浏览器拦截所有非GET API，
没有创建模型任务。`real-browser.json`、桌面/移动截图留证。首次脚本误把隐藏按钮短暂启用
当作页面加载完成，读取到默认文案；改为等待疑点实际显示后再断言，`real-browser-fixed.log`
通过，不将首次失败计为通过或声称它是产品修复。

本轮未跑整曲转换、长训练、全部模型/slow测试或远端CI；不重复声称前次模型对拍是新结果。

## 部署

2026-10-30 **17:46:46 CST**确认生产页面与事前快照一致后，以Linux `renameat2(RENAME_EXCHANGE)`
原子切换静态目录，旧页面保存于同run的`webui-deployment-swap/`，可原样交换回退。
服务未重启，PID **2600879**保持active；`vc_serve`、`vc_convert`、`vc_preprocess`、正式
Tom模型及索引SHA256前后完全相同，见`deployment.json`。联调18082实例已停止。

```sh
"$WEB_TEST_NODE" "$WEB_PITCH_RUN/evidence/verify_lan.cjs"
curl --fail --silent -o /dev/null -w '%{http_code}\n' http://192.168.0.122:8080/
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L contract -LE model --output-on-failure
scripts/check_sdd.sh
git diff --check
```

- 内网HTTP **200**，真实生产《富士山下》20疑点、43.36/45.68定位、原生试听、390px布局、
  失败任务回到成功父版及显式恢复参数通过；**0写请求、0页面异常**，没有新增生产转换。
  `lan-browser.json`及桌面/移动截图留证。
- 当前树asset-free contract **45/45，38.35秒，0失败/skip**，见`contract-tests.log`。
  本轮是静态JS/HTML/CSS变化，没有C++源码变化或新增二进制构建；宿主和原生CTest使用
  前次已构建的当前原生实现，UI契约读取本轮页面。
- SDD检查及diff空白检查通过。预览草稿仍是浏览器本地状态，不宣称跨设备同步。
- 所有既有未提交修改、诊断工具与录音保留；本轮未执行Git提交/推送或远端CI。

## 2026-10-30 Git发布前复验

维护者随后明确要求将局部修音、页面、测试和文档一起提交并推送main。诊断脚本、录音、
模型、runs证据及构建产物不纳入提交。此前部署记录中的“未提交/未推送”为对应阶段事实。

提交前发现修音浏览器runner依赖本机`NODE_PATH`，且尚未加入CI。去除该环境变量复现
`ERR_INVALID_ARG_TYPE`（`browser-ci-red.log`）；修复后与既有runner一致，从锁定的
`tests/browser/node_modules`定位Playwright和axe，并加入CI的`web-studio-browser`步骤。
未改变部署页面或计算实现。复验命令：

```sh
cmake --build /tmp/resonantia-web-release-host -j2
ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L contract -LE model --output-on-failure
export PATH=/tmp/resonantia-web-release-browser/node_modules/node/bin:$PATH
npm ci --prefix tests/browser --ignore-scripts --no-audit --no-fund
unset NODE_PATH
export CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell
node tests/test_web_pitch_review_browser.cjs webui runs/Pitch_Review_Git_20261030/evidence/pitch-browser
node tests/test_web_task_center_browser.cjs webui runs/Pitch_Review_Git_20261030/evidence/task-browser
node tests/run_web_studio_browser.cjs runs/Pitch_Review_Git_20261030/evidence/studio-browser webui
for suite in auth audio joint train_cancel; do node "tests/test_web_${suite}_ui.cjs" || exit 1; done
scripts/check_sdd.sh
git diff --check
```

两套构建成功；host **21/21，6.15秒**，contract **45/45，36.69秒**，0失败/skip；
全部三个Chromium runner、4项Node UI通过。日志在`runs/Pitch_Review_Git_20261030/evidence/`。
真实模型证据沿用前次已记录结果，提交前没有新增模型推理/训练。远端自动CI仅运行仓库、
host与浏览器门禁，设备测试为手动opt-in，不把未运行的GPU/model测试计为通过。
