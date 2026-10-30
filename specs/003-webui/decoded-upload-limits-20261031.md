# 2026-10-31 压缩上传与立体声中间文件大小修复

## 问题与边界

用户报告《山丘.flac》43.4 MB，上传时报“decoded audio exceeds 64 MiB per file or
128 MiB per upload”。未取得用户原文件，不能声称验证其实际时长或编解码内容。
旧代码将翻唱的44.1k双声道PCM16与默认mono都按64MiB限制；双声道约380.4秒就触顶。
7分钟PCM16需74,088,044字节（最小WAV头），FFmpeg另有metadata chunk。

本轮在已有FFmpeg/事务流程内修正：

- 原文件64MiB、HTTP body128MiB/inflight256MiB不变；默认mono解码64MiB不变。
- `preserve_stereo=1`解码≤128MiB且≤600秒；新解码产物批次累计128MiB。
  同步调整FFmpeg大小哨兵及OS文件限额，超限整体拒绝，保留旧raw/work/marker。
- 大PCM16预览在material锁内复制独立快照，解锁后复用64KiB块发送，支持Range；
  不按文件大小分配响应字符串。float32预览转≤128MiB PCM16临时副本，原文件不变。
- 追查发现清理输出float32会使7分钟文件超过128MiB；下一清理阶段和混音都会拒绝。
  仅44.1k双声道float32读取允许256MiB，其他输入保留128MiB，所有立体声仍≤600秒。
  格式、时长与体积先于payload分配验证，不改变已接受输入的采样或模型算法。

默认I/O测试使用稀疏零PCM，仅用于尺寸边界，不是音质证据。可选codec测试循环
`tests/fixtures/speech_00.wav`真实人声到420秒；来源沿用`tests/fixtures/README.md`，
不冒充用户歌曲或完整模型测试。训练、权重、录音和本地诊断工具不改。
Spec003没有`plan.md`；本轮依据现有spec、设计及Accepted ADR-0007，不虚构缺失计划。

## 已记录的红测与修正

```bash
cmake --build /tmp/resonantia-web-release-host --target test_web_audio_upload -j2
ctest --test-dir /tmp/resonantia-web-release-host -R '^test_web_audio_upload$' --output-on-failure
```

新增7分钟测试在旧实现失败：`7-minute stereo FLAC rejected`，错误正是用户报告的64MiB限制。
首次实施构建因新增WAV帧数检查函数仍在匿名namespace而歧义失败，移出后构建通过。
编译尚未结束时的一次CTest误跑了旧二进制，仍为相同红测；未计为修复后结果。

```bash
cmake --build /tmp/resonantia-performance-20261024.uemhTG --target test_audio_io -j2
/tmp/resonantia-performance-20261024.uemhTG/test_audio_io 'Long stereo cleanup*'
```

长float32测试在旧128MiB读取边界失败：7分钟`read_stereo_audio`返回空。日志保留于
`runs/Upload_Decoded_Limit_20261031/evidence/stereo-io-red.log`。
初次SDD检查发现本记录尚未创建导致断链，记录补齐后须重新验证。

## 验证命令与结果

首次Web重点门禁：host上传1/1；native HTTP/上传/真实codec 3/3（17.41秒），0失败/skip。
真实长FLAC编码11,076,334字节、解码74,088,078字节，420秒完整，试听与保存WAV逐字节相同。
四项Node UI契约通过。以上发生在补齐float32中间文件修复之前，最终全量结果另记于下文。

```bash
cmake --build /tmp/resonantia-web-release-host -j2
ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure
cmake --build /tmp/resonantia-performance-20261024.uemhTG -j2
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -L contract -LE model --output-on-failure
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^(test_audio_io|test_web_audio_upload|test_web_audio_codecs)$' --output-on-failure
```

最终两套构建均成功；host **21/21（6.37秒）**、完整asset-free contract
**45/45（36.63秒）**，0失败/skip。原始输出存于上述本地evidence目录的
`host-build-final.log`、`native-build-final.log`、`host-final.log`、`contracts-final.log`。

补齐float32后重点CTest：`test_audio_io`、`test_web_audio_upload`、`test_web_audio_codecs`
**3/3，7.86秒，0失败/skip**，含7分钟浮点预览及600秒读入边界。

实际CLI无模型读写联调：

```bash
/usr/bin/ffmpeg -nostdin -hide_banner -loglevel error -y -stream_loop -1 \
  -i tests/fixtures/speech_00.wav -t 420 -ac 2 -ar 44100 -c:a pcm_s16le \
  runs/Upload_Decoded_Limit_20261031/cli/source.wav
build/vc_preprocess --input runs/Upload_Decoded_Limit_20261031/cli/source.wav \
  --output-dir runs/Upload_Decoded_Limit_20261031/cli/old-first --preserve-stereo --no-slice --sr 44100
build/vc_preprocess --input runs/Upload_Decoded_Limit_20261031/cli/old-first/source.wav \
  --output-dir runs/Upload_Decoded_Limit_20261031/cli/old-second --preserve-stereo --no-slice --sr 44100
/tmp/resonantia-performance-20261024.uemhTG/vc_preprocess \
  --input runs/Upload_Decoded_Limit_20261031/cli/old-first/source.wav \
  --output-dir runs/Upload_Decoded_Limit_20261031/cli/new-second --preserve-stereo --no-slice --sr 44100
cmp runs/Upload_Decoded_Limit_20261031/cli/old-first/source.wav \
  runs/Upload_Decoded_Limit_20261031/cli/new-second/source.wav
```

这里`build/vc_preprocess`为部署前旧程序。第一遍退出0、生成完整float32；旧程序重读
自己的输出退出1并报告128MiB上限；新程序重读退出0，`cmp`退出0，输出逐字节一致。
未设置任何清理模型开关，不构成模型分离或端到端翻唱验证。

Node门禁使用`/tmp/resonantia-web-release-browser/node_modules/node/bin/node`运行
`tests/test_web_audio_ui.cjs`、`test_web_auth_ui.cjs`、`test_web_joint_ui.cjs`及
`test_web_train_cancel_ui.cjs`，均通过。SDD路径检查补齐记录后通过（32必需文件），
`git diff --check`通过。

## 部署与兼容边界

无活动任务时备份并替换`vc_serve`、`vc_preprocess`与静态页面，重启原systemd用户服务。
模型、索引、训练程序、转换程序及历史结果不迁移；部署后只GET确认本机/内网可访问。
由于清理CLI二进制变更，旧任务复用/恢复仍受可执行文件摘要校验；不伪造旧身份，
不将旧块缓存与新CLI混合。若旧任务触发版本校验需新建任务；已完成结果仍可播放。
回滚恢复备份程序和UI。未执行长训练、整曲模型转换、用户实际FLAC验证或远端CI。

实际于2026-10-31 **04:22:27 CST**切换，PID从2600879变为3025559，systemd为active/running。
本机`http://127.0.0.1:8080/`和内网`http://192.168.0.122:8080/`均HTTP200，内网页面含
“翻唱最长10分钟”提示（HTML有空格）。API仍返回Tom Cybersinger；7个历史任务保持
6成功/1失败，active_id为空。没有运行/排队/暂停中的任务被中断。
`sha256sum -c .../evidence/unchanged.sha256`确认`vc_convert`、`vc_train`、正式模型及索引未变。
备份位于`runs/Upload_Decoded_Limit_20261031/backup/`；生产验证只使用GET请求。

## 2026-10-31 Git发布前复核

用户要求仅提交本轮修复、测试和文档，诊断工具与录音继续留在本地，并保持服务运行。
复核时本地main与origin/main一致，变更为17个已跟踪文件及本记录。逐项暂存这些路径，
不包含`CyberTom/`、`tools/probe_*`等诊断脚本、隔离依赖、`runs/`或音频/模型资产。

```bash
cmake --build /tmp/resonantia-web-release-host -j2
ctest --test-dir /tmp/resonantia-web-release-host --output-on-failure
ctest --test-dir /tmp/resonantia-performance-20261024.uemhTG -R '^test_web_audio_codecs$' --output-on-failure
scripts/check_sdd.sh
git diff --check
```

host **21/21（10.75秒）**、真实codec **1/1（4.97秒）**，0失败/skip；构建、SDD及格式
检查通过。日志为本地evidence目录中的`publish-host-build.log`、`publish-host.log`与
`publish-codecs.log`。同一实现的45/45契约结果见上文；本次发布复核不重复占用用户测试中的
GPU，也不启动训练或整曲转换。服务保持PID3025559，未停止/重启或替换运行程序。
远端CI结果以推送后该提交的GitHub Actions记录为准，不把尚未运行的CI记作通过。
