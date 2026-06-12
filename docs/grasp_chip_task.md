# Grasp Chip 任务 —— 夹起易碎薯片，夹太用力就"碎"

一个触觉抓取任务：机械臂把一根薄薄的弯薯片从桌上夹起来。**夹得太用力，薯片就"碎"了 → 失败**。
不需要真的物理断裂——用触觉压入量超阈值来代表"碎"。

- 任务文件：`envs/grasp_chip.py`
- 配置：`task_config/chip_demo.yml`
- 资产：`assets/objects/CHIP.usd`（由 `scripts/asset_tools/build_chip.py` 造网格 + fTetWild 烤 tet）

---

## 目标

把桌上一根**薄弯薯片**（82.5 × 18 × 2.4mm，带弧度）稳稳夹起来抬到空中，
**全程夹持力不能太大**（太大=薯片碎）。

---

## 动作流程（这一整段都会录进数据）

1. 张爪到最大（`open_gripper(1.0)`）——薯片是横着夹两端，跨度大，开口要够宽
2. 从**上方下扎**抓取（`construct_grasp_pose`，`camera_up` 用薯片 Y 轴 → 夹爪沿薯片长度方向闭合，夹两端）
3. **自适应轻夹** `close_gripper()`——闭到目标压入量就停（不夹碎）
4. **一次直接抬到位** `move_by_displacement(z=LIFT)`（不分段，避免中间停顿）
5. **停顿保持** `delay(20, is_save=True)`

> 注意：和 HDMI/插入任务不同，这里抓取动作放在 `_play_once`（**会录**），因为"夹持力"正是这个任务的核心数据。

---

## "力" 与 fail 机制

- **"力" = 触觉 gelpad 压入量** `pressure = tactile_far_plane - get_min_depth()`（mm）。
  没接触时 `get_min_depth() ≈ far_plane(34mm)` → pressure≈0；夹得越狠，gelpad 形变越大 → pressure 越大。
- 任务全程记录**最大压入量** `max_pressure_seen`。
- 正常自适应轻夹约 **7mm**，所以把"碎"的阈值 `MAX_PRESSURE` 设在它之上（默认 **11mm**）留余量。

| 判据 | 条件 |
|---|---|
| **失败（碎）** | 全程 `max_pressure_seen > MAX_PRESSURE` → `check_early_stop` 返回 True，且 `_track` 实时置 `plan_success=False` |
| **失败（掉）** | 结束时 `pressure < HOLD_PRESSURE`（基本没接触）|
| **失败（没夹起）** | ee 相对**抓取后高度**上升不足 `0.7 × LIFT`（注意：要跟抓取后高度比，不是 home 高度——抓取要先下降）|
| **成功** | 抬起来了 **且** 还夹着 **且** 全程没夹碎 |

判定在 `collect_data.run()`：`plan_success and check_success() and not check_early_stop()` 才算成功、存数据、视频命名 `_success`。

---

## 可调参数（`envs/grasp_chip.py` 顶部，都标了 `[tune]`）

| 参数 | 含义 | 当前值 |
|---|---|---|
| `CHIP_REST_Z` | 薯片平躺桌面的体心高度(m) | 0.01 |
| `GRASP_Z` | 抓取点相对薯片体心的 z 偏置 | 0.004 |
| `LIFT` | 一次抬到位的高度(m) | 0.08 |
| `MAX_PRESSURE` | 压入量上限(mm)，超过=碎=fail（可用环境变量 `CHIP_MAX_PRESSURE` 临时改）| 8.5 |
| `CHIP_GRIP_DEPTH` | 环境变量：夹力，闭到的触觉深度(越小夹越紧)，0=默认轻夹 | 0 |
| `HOLD_PRESSURE` | 压入量低于此=掉了 | 1.0 |
| `CHIP_QUAT` | 薯片摆放朝向(绕竖直轴) | — |

薯片尺寸/弧度在 `scripts/asset_tools/build_chip.py` 里改（改完要重新转 USD + 烤 tet，见 `docs/making_a_univtac_task.md`）。

---

## 跑 / 录制

```bash
conda activate UniVTAC
# 单次调试(headless): chip_demo.yml (episode_num:1, max_seed:0)
python scripts/collect_data.py grasp_chip chip_demo
# 采一批数据: chip_collect.yml (episode_num:10) —— 实测 10/11 成功率
python scripts/collect_data.py grasp_chip chip_collect
# 看动作(GUI): 把配置里 render_frequency 设 1, 然后
DISPLAY=:1 python scripts/collect_data.py grasp_chip chip_demo
# 视频: data/grasp_chip/<cfg>/video/0_success.mp4 (或 _fail); 日志打印 [CHIP] 最大压入量 = X mm
```
> 跑前先清残留实例：`pkill -9 -f '[c]ollect_data.py'`（方括号是为了不误杀 pkill 命令自身）。

**演示"夹太狠就碎"触发 fail**（环境变量，不用改代码）：
```bash
# 夹更紧(闭到触觉深度20) + 把碎阈值压到 5mm -> 必触发"碎"fail
CHIP_GRIP_DEPTH=20 CHIP_MAX_PRESSURE=5 python scripts/collect_data.py grasp_chip chip_demo
```

> **刚性薄片的"力"范围很窄**：gelpad 压在刚性薄片上物理上最多压进 ~9mm，所以正常夹 ~7mm、死夹 ~8.7mm，
> 区间不大。默认 `MAX_PRESSURE=8.5` 卡在中间（正常成功、死夹判碎）。要更宽的"力"范围、更像真实易碎物，
> 应把薯片做成**软体(FEM, StableNeoHookean)** 而非刚体——gelpad 能压进更多、力的层次更丰富。

---

## 调试要点（踩过的坑，换别的薄/弯物体时有用）

- **抓弯曲物体的两端会"挤飞"**：从两端往里挤一个拱形，接触面是斜的、挤压有向上分量 → 物体被弹出。
  这次靠"把抓取点 `GRASP_Z` 往下压到位 + 一次连续抬起"解决；夹更平的中段或夹宽度方向也能避免。
- **抓取点太高只会蹭一下**：gelpad 没真正夹到物体（或蹭到桌面），一抬就掉。压入量在"夹的瞬间有、抬起归 0"
  就是这个症状——把 `GRASP_Z` 往下调。
- **分两段抬 → 中间停顿 + 容易掉**：两个 `move_by_displacement` 是两段独立规划。合成一次连续抬升更稳更顺。
- **手持件要 `planner_ignore_actors`**：否则 curobo 把手里的薯片当静态障碍 → 规划失败。
- **"力"是 gelpad 压入量、不是牛顿力**，且只在 `_update_render()` 里刷新；`render_frequency` 也决定它多久更新一次。
- **"抬起来了"判据要跟抓取后高度比，不能跟 home 比**：抓取要先下降到桌面，即使抬了 8cm，ee 仍比 home 低 →
  跟 home 比会把"明明夹起来了"误判成 fail。`check_success` 里用 `self.grasp_ee_z`（抬升前记录）。
- 想**故意夹太狠**触发 fail：跑前设环境变量 `CHIP_GRIP_DEPTH=20`（闭到更深的触觉深度=夹更紧，
  压入量会超过 `MAX_PRESSURE` → 判碎 fail），或直接把 `MAX_PRESSURE` 调到正常 7mm 以下。
