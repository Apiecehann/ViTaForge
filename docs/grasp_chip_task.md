# Grasp Chip 任务 —— 夹起易碎薯片，夹太用力就"碎"

## 任务是什么

机械臂把桌上一根薄薄的弯薯片夹起来抬到空中。和普通抓取不同的是：**薯片很脆，夹得太用力就算"碎"了 → 失败**。这里不做真的物理断裂，而是用触觉的压入量来代表夹持力——压入越深 = 夹得越狠，超过阈值就当薯片碎了。

所以成功要同时满足三点：**夹起来了、还夹着没掉、全程没夹碎**。

- 任务文件：`envs/grasp_chip.py`
- 配置：`task_config/chip_demo.yml`（单次）、`chip_collect.yml`（采 10 条）
- 资产：`assets/objects/CHIP.usd`（薄弯薯片 82.5 × 18 × 2.4mm）

## 怎么实现的

**动作流程**（全程会录进数据，因为"夹持力"就是这个任务的核心）：

1. 张爪到最大，从上方下扎，夹爪沿薯片长度方向闭合、夹住两端
2. 自适应轻夹，闭到目标压入量就停，不夹碎
3. 一次连续抬到空中，停顿保持

**"力" 与判定**：

- "力" = 触觉 gelpad 的压入量 `pressure = far_plane - get_min_depth()`（mm）。没接触≈0，夹得越狠压入越大。
- 全程记录最大压入量，超过 `MAX_PRESSURE` 就判定"碎"→失败。
- 结束时压入量太小 = 掉了；ee 相对抓取后高度抬升不够 = 没夹起来。

成功判定在 `collect_data.run()`：抬起来了、还夹着、且全程没夹碎，才算成功并存数据。

## 跑 / 录制

```bash
conda activate UniVTAC
# 单次调试
python scripts/collect_data.py grasp_chip chip_demo
# 采一批数据（10 条）
python scripts/collect_data.py grasp_chip chip_collect
# 视频在 data/grasp_chip/<cfg>/video/0_success.mp4（或 _fail）
```

演示"夹太狠就碎"（环境变量，不用改代码）：

```bash
# 夹更紧 + 把碎阈值压低 -> 必触发"碎"fail
CHIP_GRIP_DEPTH=20 CHIP_MAX_PRESSURE=5 python scripts/collect_data.py grasp_chip chip_demo
```

常用可调参数都在 `envs/grasp_chip.py` 顶部（标了 `[tune]`）：`LIFT` 抬升高度、`MAX_PRESSURE` 碎的阈值、`GRASP_Z` 抓取点高度等。薯片尺寸在 `scripts/asset_tools/build_chip.py` 改（改完要重转 USD + 烤 tet）。
