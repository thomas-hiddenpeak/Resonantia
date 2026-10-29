# 完成音频任务的参数修订（2026-10-29）

状态：已实现并部署。归属[WEB-FR-014](spec.md#2026-10-29-后台长任务web-fr-014)，沿用
[ADR-0010](../../docs/adr/0010-durable-web-task-lifecycle.md)的不可变任务/阶段产物机制。

## 行为与边界

任务中心“调整参数再生成”恢复成功音频任务的原素材、设置及局部修音，提交新稳定ID，原任务及
旧音频保持不变；详情可返回原版。`source_task`与上传/session互斥，身份/路径/状态错误明确拒绝。

| 修改内容 | 验证通过后的复用 |
|---|---|
| 音高、检索率、动态、保护、平滑、局部修音或声线 | 相同流程的人声分离及主唱清理；重做转换/混音 |
| 仅人声/混音输出 | 分离、清理和已转换主唱；重新组织输出 |
| 清理设置 | 相同歌曲输入的分离；重做清理和转换 |

复用先核验父任务完整依赖及阶段SHA，只导入完成阶段，新任务有独立副本及新身份；不把旧参数的
转换块检查点用于新参数。不改原任务记录，不删除旧结果，不引入共享可变缓存或新持久格式。
已清理的模型快照只在当前同名模型匹配旧身份时补齐；纯人声最终结果可替代已删除的转换副本，
混音结果不能充当独唱。缺失/变化身份或坏缓存明确失败。身份核验仍有耗时，本次不宣称整曲加速比。

页面保留原输入类型，手动清理不可操作旧session；移除历史素材后恢复新上传操作。
恢复合法API参数时保留±48音高、非默认平滑窗口及小数检索率，不因控件范围静默改写。
训练的固定配方/完整检查点恢复没有改变；本增量不支持原地改写训练参数。

## 验证与首次失败

本地证据根：`runs/Web_Task_Revisions_20261029/evidence/`；真实模型fixture为同级`real-revision-1/`。

- `red-test.log`：旧版因没有上传/session而返回“no audio”，完成任务修订失败。
- `green-build.log`：首版路由接线误写变量名导致编译失败；修正后`green-test-2.log`1/1通过。
- `extended-test.log`：参数变化/仅混音/清理设置失效、原任务不变、原音频Range、清理后复用纯人声、
  重复ID/坏ID/失踪或失败父任务、冲突来源、非法参数、损坏源/阶段拒绝、干声无额外重采样均通过。
- `ui-red.log`：旧页面没有调整入口。第一次green未展开全部祖先details，第二次键盘无有效焦点；
  修正测试展开操作后`ui-green-3.log`通过。没有改产品门槛掩盖这些脚本失败。
- `ui-boundary-red.log`实际发现移除历史素材后输入类型仍锁定；修复后`ui-boundary-green.log`通过，
  同时保留原API参数范围与精度。最终`ui-final.log`含来源任务无上传的新ID提交、鉴权原音频、参数/
  修音恢复、父版保留、320/390/1440布局与axe；`studio-final.log`原工作室回归通过。
- `real-revision-1.log`：真实vc_serve/GPU和speech_02前6秒，先完整分离/清理/转换，改pitch=-2后
  仅导入分离/清理，输出与对同一主唱独立运行vc_convert的WAV逐字节一致；再改mix，导入三个阶段，
  没有转换子进程。父元数据/结果及正式Tom SHA不变；没有训练、发布或整曲模型测试。
- `host-tests.log`：**19/19，5.33秒**；`contract-final.log`：**44/44，35.92秒**，均0失败/skip。
  `full-build.log`全目标构建通过；这些不是全部76项模型/设备测试重跑。

核心实际命令：

```sh
cmake --build /tmp/resonantia-web-release-host --target test_web_tasks -j2
ctest --test-dir /tmp/resonantia-web-release-host -R '^test_web_tasks$' --output-on-failure
cmake --build /tmp/resonantia-web-release-host -j2
ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure -j1
cmake -S . -B /tmp/resonantia-performance-20261024.uemhTG -DRESONANTIA_WEB_TEST_UI_DIR=/home/rm01/Resonantia/runs/Web_Task_Revisions_20261029/webui-staged
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L contract --no-tests=error --output-on-failure -j1
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell /tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_task_center_browser.cjs runs/Web_Task_Revisions_20261029/webui-staged runs/Web_Task_Revisions_20261029/evidence/ui-final
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules CHROMIUM_PATH=/home/rm01/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux-arm64/chrome-headless-shell /tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/run_web_studio_browser.cjs runs/Web_Task_Revisions_20261029/evidence/studio-final runs/Web_Task_Revisions_20261029/webui-staged
/tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_audio_revision_real.cjs /home/rm01/Resonantia /tmp/resonantia-performance-20261024.uemhTG /home/rm01/Resonantia/runs/Web_Task_Revisions_20261029/real-revision-1
NODE_PATH=/tmp/resonantia-web-release-browser/node_modules /tmp/resonantia-web-release-browser/node_modules/node/bin/node runs/Web_Task_Revisions_20261029/evidence/production-browser.cjs
```

## 生产交付与回退

2026-10-29 14:04:18 CST确认无活动任务后，备份旧服务/页面/配置于
`runs/Web_Task_Revisions_20261029/deployment-backup/`，仅替换vc_serve及静态页面，用户服务重启为
PID2291009/active。四个计算CLI与此前部署二进制逐一SHA相同；当时据此判断已有任务不会失效，
但随后实际执行暴露Web可执行文件也被误纳入身份，原判断不成立，修复及验收见下节。
入口仍为`http://192.168.0.122:8080/`。

`production-browser.log/.json`通过原生内网GET检查：用户已完成的《我管不住我自己》
（261.576984秒）能载入原素材、Tom声线、原调、检索0.35、动态0及混音输出，生成按钮可用。
生产测试禁止全部写请求，实际写请求0；记录/源音频/结果/正式模型SHA均不变，没有启动整曲转换。

回退先确认空闲，再停用户服务，从备份恢复vc_serve及原静态页面后启动；不删除任何任务/素材。
旧程序仍可读v2记录，新来源标记仅在新界面可见；不依赖父任务的已导入阶段仍保留。
本轮未提交或推送Git，保留此前未提交修改及未跟踪资产。

部署后测试改回实际webui目录，UI契约4/4通过（1.00秒）；`deployment.json`确认已部署程序/页面与
测试版本SHA一致。`bash scripts/check_sdd.sh`通过32项，`git diff --check`无错误，日志分别为
`sdd-final.log`与`diff-check-final.log`。

## 跨Web更新的身份修复（2026-10-29）

用户对261.576984秒完成歌曲创建`-12`半音新版，身份校验报错。只读重算证明：父任务摘要精确
匹配09:27 Web，失败新版摘要精确匹配14:04 Web，其他输入/模型/计算CLI/配方均保持。
Web误用了自动绑定`/proc/self/exe`的OperationIdentity；前次同一二进制测试及生产只读检查
未覆盖跨部署实际执行。根因数据为`runs/Web_Identity_Fix_20261029/evidence/diagnosis.json`。

WebWorkflowIdentity改为显式编排协议，仍核验完整依赖；原生CLI身份不变。只接受代码固定的
两个已审计旧生产者，逐项重算完整旧摘要，不能绕过未知版本/依赖变化/坏缓存。TaskArtifacts
保留匹配身份，不重写原标记；训练外层与嵌套切片也按对应版本匹配。协议/存储边界见Spec003及
ADR-0010，训练数值路径与检查点不变。

证据根：`runs/Web_Identity_Fix_20261029/evidence/`。

- `red-test.log`记录仅改变Web程序即身份失效；`green-test.log` focused **2/2，3.38秒**通过。
- `host-tests.log` **19/19，4.61秒**通过；随后完整构建发现Web头文件泄露新头的include依赖，
  改为前置声明后`full-build-2.log`全目标成功，`contract-tests.log` **44/44，41.08秒**通过。
  均0失败/skip；host计时对应前置声明调整前的同一逻辑，不声称全部76项模型测试重新运行。
- `test_task_artifacts`执行两个SHA不同的程序，Web身份一致、CLI身份不同；独立固定摘要覆盖
  两个旧生产者及嵌套训练，配方/输入/计算程序变化与未知旧版本仍拒绝，旧标记保持不变。
- `upgrade-real-2.log`：两个原生旧Web分别创建真实6秒CPU预处理任务，新Web导入成功；再改变
  新Web可执行文件SHA，后续导入仍成功。原标记/WAV不变，模型执行0。首次`upgrade-real.log`
  因第二个备份尚不存在而失败，补齐前置检查及备份后重验，没有把首轮算作通过。
- `user-probe-2.log`及`user-task-copy-2/evidence.json`：复制用户实际父/失败任务至隔离目录，
  通过Web恢复入口保留`-12`并导入`separation,cleanup`，在第一次模型调用前拦截停止，实际
  模型执行0。不是整曲转换成功的证明。首次辅助程序编译有copy重载歧义；首轮运行又假定attempt1，
  但用户已重试到attempt2，服务正确返回幂等旧状态。改为读取副本实际代次后通过，原生产任务未改。

核心实际命令（省略日志重定向）：

```sh
cmake --build /tmp/resonantia-web-release-host -j2
ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure -j1
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L contract --no-tests=error --output-on-failure -j1
/tmp/resonantia-web-release-browser/node_modules/node/bin/node tests/test_web_identity_upgrade_real.cjs /home/rm01/Resonantia /tmp/resonantia-performance-20261024.uemhTG /home/rm01/Resonantia/runs/Web_Identity_Fix_20261029/real-upgrades-2 /home/rm01/Resonantia/runs/Web_Task_Revisions_20261029/deployment-backup/vc_serve /home/rm01/Resonantia/runs/Web_Identity_Fix_20261029/deployment-backup/vc_serve
runs/Web_Identity_Fix_20261029/evidence/validate_user_task /home/rm01/Resonantia /home/rm01/Resonantia/runs/Web_Identity_Fix_20261029/user-task-copy-2
bash scripts/check_sdd.sh
git diff --check
```

15:18:25 CST确认空闲后仅替换vc_serve，用户服务PID2302330/active。`deployment.json`确认
LAN入口HTTP200、部署与测试二进制相同、计算CLI未变、原父记录/源/结果/正式Tom模型摘要不变，
两个旧workflow标记不变。失败任务仍保留旧错误与attempt2，提供继续入口，由用户主动继续；
没有自动提交、重跑整曲或训练。原页面不需重新上传，刷新后从失败任务选择“继续此任务”。

回退备份为`runs/Web_Identity_Fix_20261029/deployment-backup/`，空闲时停服务、恢复vc_serve后启动；
不改页面或删数据。旧服务仍存在本次身份缺陷，且不能继续新协议任务；再部署修复版可恢复入口。
本增量未提交/推送Git，保留其他未提交工作。
