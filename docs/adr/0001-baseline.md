# ADR-0001 重构基线

**冻结日期**：2026-08-10
**Git HEAD**：`4afa1c4d773ede2d619b29296a89d8743d02c7d5`
**工作树**：包含维护者已有的未提交修改与本轮审计修复；重构不得清理、覆盖或据此
假定仓库处于 clean 状态。

本文记录 ADR-0001 R0 的行为和构建基线。它是一次执行证据，不替代当前 Spec、实现或
后续 current-tree 验证。

## 环境

- NVIDIA Jetson Orin NX Engineering Reference Developer Kit，SM87；
- GNU C++ 13.3.0；
- CUDA compiler 13.2.78，`/usr/local/cuda/bin/nvcc`；
- Release、C++20/CUDA20、Ninja；
- 构建目录：`/tmp/resonantia-r0.PLD2LT`。

## 复现命令

```bash
cmake -S . -B /tmp/resonantia-r0.PLD2LT -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVOXMUTATIO_BUILD_TESTS=ON \
  -DVOXMUTATIO_ENABLE_MODEL_TESTS=ON \
  -DVOXMUTATIO_CUDA_ARCHITECTURES=87
cmake --build /tmp/resonantia-r0.PLD2LT --parallel 2
ctest --test-dir /tmp/resonantia-r0.PLD2LT -N
ctest --test-dir /tmp/resonantia-r0.PLD2LT --output-on-failure -L contract
```

结果：fresh configure 成功；192/192 build steps 成功；CTest 注册 27 项，恰为 9 个默认
contract 与 18 个 asset-gated integration；默认 contract 9/9 通过、0 失败。

最终切换后的 27/27 current-tree 资产门禁记录见
[`specs/PROJECT_STATE.md`](../../specs/PROJECT_STATE.md)。该结果已在新构建图上执行；本文件
仍只代表 R0 冻结证据，不能替代后续 current-tree 验证。

## 冻结的可观察面

- 公共目标：`voxmutatio` static library；现有 CLI/测试 target 名保持不变；
- 公共 API：`include/voxmutatio/` 下头文件、默认值、返回与错误语义保持不变；
- CLI：名称、flags、默认值、退出码与脚本依赖的输出布局保持不变；
- HTTP：路由、方法、404/405/409、鉴权范围、默认开放策略与资源上限保持不变；
- 格式：WAV、safetensors、Index 与训练导出格式保持不变；
- 数值：当前 FP32/CUDA 支持矩阵及 Spec 门槛保持不变；
- 测试：9 个 contract、18 个 integration 的名称、label、asset gate、timeout 与 GPU lock
  不得减少或静默改变。

## 本地资产集合

本次配置时，`cmake/ResonantiaTests.cmake` 为 18 个 integration 声明的全部资产均存在，
因此 27 项全部注册。资产集合包括：HuBERT、RMVPE、FCPE、WavLM、F0 VITS v1/v2 G、
v2 D、UMX、dereverb RoFormer、Silero VAD，以及 `tests/fixtures/` 下的真实语音、模型中间
参考、分离和 VAD reference。权重与大型 fixture 不入 Git；精确路径和每个测试的依赖以
当次 CMake 文件中的 `ASSETS` 列表为准。

## 回滚触发器

任一阶段若出现注册数下降、原有 target 无法链接、CLI/HTTP 契约变化、数值门槛失败、
资源上界回退或旧模型/格式无法加载，应撤销该阶段的结构切换，恢复到本基线所描述的
兼容 facade；不得删除用户模型、音频、run 目录或维护者现有工作树修改。
