---
name: make-univtac-task
description: 从一个输入 URDF/网格资产端到端做出一个 UniVTAC 触觉插入任务 —— 重建简化版 peg+socket、用 fTetWild 烤 tet、导入 Isaac 人工检查、按描述搭新任务场景、给出并人工复核 motion planning、迭代修正、产出录制结果。当用户想"用某个 URDF/mesh 做一个新的 UniVTAC 插入任务"时使用。
---

# Make a UniVTAC insertion task（URDF → 可跑可录的任务）

把外部 URDF/网格变成 UniVTAC 里能仿真+采数据的插入任务。**这是一个带人工检查点的半自动流程**：
两处必须停下来让用户肉眼确认（导入后看资产、跑通后看 motion）。不要跳过检查点。

## 适用前提 / 先问清楚的输入
开始前确认这几项（缺了就用 AskUserQuestion 问）：
1. **输入资产路径**（URDF / zip / obj 目录），例如 `/home/ubuntu/Downloads/xxx.zip`。
2. **任务描述**：哪个件插进哪个件、插口/孔大致尺寸、成功判据（对正+插到底+姿态阈值）。
3. **抓取范式**：开局已握住（简单稳）/ 桌面抓取（同上游 HDMI）。默认"开局已握住"。

## 环境（每条命令都要用对的 python）
```bash
conda activate UniVTAC                 # 必须: 才挂 libcuda 软链 + OMNI_KIT_ACCEPT_EULA
PY=~/miniconda3/envs/UniVTAC/bin/python
EXT=~/miniconda3/envs/UniVTAC/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-*
# 读/写 USD 的 pxr 不在标准路径, 用 extscache:
#   PYTHONPATH=$EXT LD_LIBRARY_PATH=$EXT/bin:$EXT/pxr/.libs $PY <脚本>
```
现成脚本（参考/复用，必要时按新资产改尺寸）：
`scripts/asset_tools/{build_usb.py,build_usb_slot.py,bake_tet.py,write_tet_to_usd.py}`、
`scripts/{convert.py,view_obj.py,view_usd.py,view_task.py,collect_data.py}`、
范例任务 `envs/insert_USB.py`、范例配置 `task_config/usb_demo.yml`、详解 `docs/urdf_import_pipeline.md`。

---

## 步骤

### Step 1 — 解析 URDF，挑出要保留的部件
- 解压资产；看 `result.json`/`semantics.txt` 的命名 obj 分组。
- **删掉所有可动件**（UIPC 不消费 URDF 关节）。只留要插的主体。
- 渲染对照图（trimesh，给每组上色）确认哪组是哪个，必要时给用户看。

### Step 2 — 重建简化版实体（peg + socket）
- **优先重建参数化长方体**（干净水密、tet 又快又准），别硬啃扫描网格。
- 拿实测尺寸：peg = 机身长方体 + 矩形插口（`trimesh.boolean.union`），**插口朝 local −Z**，居中原点，mm→m。
- **按真实标准定标**（如 USB-A 插口宽 12mm）反推 scale，别拍脑袋。
- socket = 外块 `difference` 主孔 + 导入倒角；孔 = 插口截面 + **单边间隙(必须 > IPC d_hat 0.5mm，建议 1.5mm)**，孔口朝 +Z。
- 用 `build_usb.py`/`build_usb_slot.py` 当模板改尺寸。输出到 `/tmp/<name>_assets/`。

### Step 2.5 — 🔴 人工检查点①：三维查看简化后的网格
- 用 `python scripts/view_obj.py /tmp/<name>_assets/PEG.obj`（pyglet 交互窗口，鼠标可转）逐个看 peg 和 socket。
- **停下来用 AskUserQuestion 让用户确认**：简化后的形状对不对、尺寸/比例对不对、peg 插口和 socket 孔对不对得上。
- 这一步是在**转 USD/算 tet 之前**就抓问题（这里改网格最便宜）。不过就回 Step 2 改尺寸，过了再继续。

### Step 3 — 转 USD + 烤 tet + 校验
> 只走一条路线：**fTetWild 离线算 tet 写进 USD**（不用 convert 自带的 tetgen，也不靠加载时现算——太慢）。
> fTetWild(Fast Tetrahedral Meshing in the Wild) = 把任意三角表面鲁棒转成四面体体网格的算法，
> 靠误差包络(envelope)+绕数(winding number)，能吃脏/非水密网格；用其 Python 绑定 `wildmeshing`。
```bash
# 3a OBJ -> surface USD
SKIP_TET=1 python scripts/convert.py -i /tmp/<name>_assets -o assets/objects \
  --collision-approximation convexDecomposition --headless
# 3b 烤 tet(每个资产单独进程! 同进程多次 tetrahedralize 会 double-free)
$PY scripts/asset_tools/bake_tet.py /tmp/<name>_assets/PEG.obj  /tmp/PEG_tet.npz  <elr> <eps>
$PY scripts/asset_tools/bake_tet.py /tmp/<name>_assets/SLOT.obj /tmp/SLOT_tet.npz <elr> <eps>
# 3c tet 写进 USD
PYTHONPATH=$EXT LD_LIBRARY_PATH=$EXT/bin:$EXT/pxr/.libs $PY scripts/asset_tools/write_tet_to_usd.py assets/objects/PEG.usd  /tmp/PEG_tet.npz
PYTHONPATH=$EXT LD_LIBRARY_PATH=$EXT/bin:$EXT/pxr/.libs $PY scripts/asset_tools/write_tet_to_usd.py assets/objects/SLOT.usd /tmp/SLOT_tet.npz
```
- tet 旋钮：`edge_length_r`(密度/仿真开销，~1.5mm 边长一档，对齐 HDMI ~3.4k tets)；`epsilon=0.001`(紧包络→表面光滑)。
- 用 extscache 的 pxr 校验：`/<name>/mesh` 上 4 个 tet 属性都在、bbox 是真实毫米尺寸。

### Step 4 — 🔴 人工检查点②：导入 Isaac 看资产（带 tet 的最终效果）
- 渲染最终网格（`view_obj.py`）或在 Isaac 里看（`view_usd.py <usd>`，DISPLAY=:1）。把图/窗口给用户。
- **停下来用 AskUserQuestion 让用户确认**：形状对不对、尺寸/比例对不对、表面是否够光滑、peg 和 socket 孔对不对得上。
- 不通过就回 Step 2/3 调（尺寸/间隙/epsilon），通过再继续。

### Step 5 — 按描述搭新任务场景
- 复制 `envs/insert_HDMI.py` 或 `envs/insert_USB.py` 为 `envs/<task>.py`，换资产名、改关键常量
  （`SLOT_HEIGHT`/`HOLE_DEPTH`、peg 几何偏置、抓取/孔口位姿）。
- 写好 `create_actors`(slot density=1e5 当固定座 / peg)、`_reset_actors`、`pre_move`、`_play_once`、`check_success`。
- 复制 `task_config/usb_demo.yml` 为 `<task>.yml`，录制参数**对齐 HDMI 的 demo.yml**
  （`save_frequency:2 / video_frequency:2 / render_frequency:0`，完整 observations）。

### Step 6 — Motion planning 方案
- 给出明确的动作序列并说明每段意图，例如：
  `reset 把 peg 放到夹爪中心+近闭合` → `pre_move: close_gripper + place_actor 到孔口上方(留间隙)`
  → `_play_once: place_actor 精对准 + move_by_displacement 分段下压(~1.5×孔深) 柔顺插入`。
- 关键参数：抓取/孔口的 z 偏置、`pre_dis/dis`、下压行程、`constraint_pose`(放松绕 z)。

### Step 7 — 🔴 人工检查点③：跑通 + 看 motion
```bash
# 调试单次(GUI, 看动作): DISPLAY=:1 python scripts/collect_data.py <task> <cfg>   (cfg 里 max_seed:0)
# 录视频(headless, render_frequency:0): python scripts/collect_data.py <task> <cfg>
# 视频: data/<task>/<cfg>/video/0_success.mp4  (只录 play_once = 已握住→插入)
```
- 把视频/截图给用户。**停下来用 AskUserQuestion 让用户确认** motion 哪段有问题：
  抓取？移动到孔口？对准？插到底？卡不卡？

### Step 8 — 迭代修正（按反馈）
对照症状改：
- 规划失败(planning failed) → 任务里设 `self.planner_ignore_actors = {'<peg>'}`（手持件别当障碍）。
- "World is not valid" → IPC 初始穿透：peg 别预插孔里/别和夹爪重叠，悬空留间隙。
- 没夹住 → reset 阶段就闭合夹爪（趁 peg 仍被 set_pose 约束）。
- 插入卡死 → 加大孔间隙(>0.5mm)/加导入倒角/加大下压行程/检查对准。
- 表面糙 → 收紧 `epsilon` 重烤。
- 改完回 Step 7 重看，直到用户满意。

### Step 9 — 产出结果
交付清单：`assets/objects/<PEG>.usd`+`<SLOT>.usd`、`envs/<task>.py`、`task_config/<task>.yml`、
录制视频路径；并把新增/改动的关键点记到 `docs/` 或 memory。

---

## 必踩坑速查（贯穿全程）
| 现象 | 解法 |
|---|---|
| `errno=28 / No space left on device` | inotify watch 超限(非磁盘)：`sudo sysctl -w fs.inotify.max_user_watches=1048576`(需 `!` 跑) |
| `Could not load libcuda.so` | 没 `conda activate UniVTAC` |
| 加载卡几分钟、GPU 空闲 | USD 没烤 tet，加载时在现算 → 补 Step 3 |
| 表面糙/波浪 | `epsilon` 太松 → 0.001 |
| 渲染像低模 | 渲染网格加载时被 tet 表面覆盖(本就是 tet 分辨率) |
| 插入卡死 | 孔间隙太小(>0.5mm)/没对准/下压不够 |
| 手持长 peg 规划必失败 | `planner_ignore_actors` |
| 开局 "World is not valid" | IPC 初始穿透 |
| 烤 tet `double free` | 每个资产单独进程 |
| pxr `ImportError: libtf.so` | 用 extscache 的 `omni.usd.libs-*`，`LD_LIBRARY_PATH` 指到它的 `bin/` |
| 数据 `.cache/0 does not exist` | 录数据要 `video_frequency>0` + 完整 observations |
| 视频卡顿 | 帧率上限(VideoHandler `-framerate 10`)+ 采样稀；和 HDMI 对齐 `render_frequency:0`，或调高 fps/`save_frequency:1` |

## 重要纪律
- 三处 🔴 人工检查点（①简化网格 ②带 tet 的 USD 在 Isaac ③motion 视频）**必须停下来问用户**（用 AskUserQuestion 或贴图/开窗），不要自顾自往下冲。
- 每次只改少量参数再重看，不要一把改一堆。
- 跑任何 Isaac 命令前先 `pkill -9 -f collect_data.py` 清掉残留实例（多实例会抢 GPU 变卡）。
- 清理仓库时 `video/` 目录单独保留，别连锅端。
