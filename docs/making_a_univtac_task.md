# 做一个 UniVTAC 插入任务（URDF → 可跑可录）

把一个 URDF/网格做成 UniVTAC 里能仿真、采数据的插入任务（peg 插进带孔 socket）。
以 USB 为例，流程对任何 peg/socket 通用。`envs/insert_HDMI.py`、`insert_USB.py` 是可参考的范例。

## 前提

- UniVTAC 物体是 UIPC 刚体/软体，仿真跑在**四面体(tet)网格**上。能用的 USD 必须在 `/<名>/mesh` 上带
  `tet_points / tet_indices / tet_surf_points / tet_surf_indices` 四个属性，且为**真实尺寸(米)**、**单刚体**（URDF 关节不支持）。
- tet 用 **fTetWild** 算（`wildmeshing` 库）：把任意三角表面鲁棒转成四面体体网格，靠误差包络+绕数判内外，能吃脏/非水密网格。
  我们**离线烤好写进 USD**（否则加载时现算，密网格要几分钟）。
- 环境：`conda activate UniVTAC`（挂 libcuda + 接受 EULA，不激活 Isaac 起不来）。

## 流程

**1. 挑部件**：解析 `result.json`，删掉所有可动件（UIPC 不要关节），只留要插的主体。

**2. 重建简化实体**（推荐，比啃扫描网格省事）：trimesh 拼长方体。
- peg：机身+矩形插口，布尔并集，**插口朝 −Z**，居中，mm→m（`build_usb.py` 改尺寸）。按真实标准定标。
- socket：外块挖孔+导入倒角，孔=插口+**单边间隙(必须 >0.5mm，建议 1.5mm，否则卡死)**（`build_usb_slot.py`）。

**🔴 检查点①**：`python scripts/view_obj.py /tmp/asset/PEG.obj`（鼠标转）。确认形状/尺寸/孔对得上，不对回第 2 步。

**3. 转 USD + 烤 tet**：
```bash
# OBJ -> surface USD (SKIP_TET=1: 跳过自带 tetgen; -i 可给目录批量; --headless 不开界面)
SKIP_TET=1 python scripts/convert.py -i /tmp/asset -o assets/objects \
  --collision-approximation convexDecomposition --headless

# 烤 tet -> npz  (每个资产单独进程; 参数: <in.obj> <out.npz> <edge_length_r> <epsilon>)
PY=~/miniconda3/envs/UniVTAC/bin/python
$PY scripts/asset_tools/bake_tet.py /tmp/asset/PEG.obj  /tmp/PEG_tet.npz  0.026 0.001
$PY scripts/asset_tools/bake_tet.py /tmp/asset/SLOT.obj /tmp/SLOT_tet.npz 0.055 0.001

# 写进 USD  (pxr 借 Isaac 的; 参数: <目标.usd> <tet.npz>)
EXT=~/miniconda3/envs/UniVTAC/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-*
PYTHONPATH=$EXT LD_LIBRARY_PATH=$EXT/bin:$EXT/pxr/.libs \
  $PY scripts/asset_tools/write_tet_to_usd.py assets/objects/PEG.usd /tmp/PEG_tet.npz
```
tet 两个旋钮：`edge_length_r` 控制 **tet 数量(=仿真开销)**；`epsilon` 控制 **表面光滑度**（糙就收到 0.001）。

**🔴 检查点②**：`DISPLAY=:1 python scripts/view_usd.py assets/objects/PEG.usd`。看仿真里真实效果（表面光滑、tet 够细）。

**4. 搭任务**：复制一个 `envs/insert_*.py` 改资产名+尺寸常量（`create_actors/_reset_actors/pre_move/_play_once/check_success`）；
复制 `task_config/*.yml` 当配置（录视频要 `video_frequency>0` + camera/tactile 观测）。
> 数据从"已握住"开始录：抓取在 `pre_move`(reset 阶段不录)，`_play_once` 才录。

**5. Motion planning**：`place_actor` 会把物体局部 +Z 对齐孔口 +Z → 插口自动朝下。
手持长 peg 必须在任务里设 `self.planner_ignore_actors = {'peg'}`（否则规划器把它当障碍物，规划必失败）。

**🔴 检查点③**：跑通看 motion：
```bash
pkill -9 -f collect_data.py     # 先清残留实例(多个 Isaac 抢 GPU 会卡)
python scripts/collect_data.py <task> <cfg>          # <task>=envs/名, <cfg>=task_config/名
# 带 DISPLAY=:1 弹 GUI 实时看; cfg 里 render_frequency:0 = headless 纯录制; max_seed:0 = 只试一次
# 视频: data/<task>/<cfg>/video/0_success.mp4  (只录 play_once)
```

## 坑速查

| 现象 | 解法 |
|---|---|
| `errno=28 / No space left on device` | inotify 超限(非磁盘)：`sudo sysctl -w fs.inotify.max_user_watches=1048576` |
| `Could not load libcuda.so` | 没 `conda activate UniVTAC` |
| 加载卡几分钟、GPU 空闲 | USD 没烤 tet → 补第 3 步 |
| 表面糙/波浪 | `epsilon` 太松 → 0.001 重烤 |
| 插入卡死 | 孔间隙太小(>0.5mm)/没对准/下压不够 |
| `Arm motion planning failed` | 手持 peg 当障碍 → `planner_ignore_actors` |
| `World is not valid` | IPC 初始穿透：peg 别预插孔里/别和夹爪重叠 |
| 烤 tet `double free` | 每个资产单独进程 |
| pxr `ImportError: libtf.so` | `LD_LIBRARY_PATH` 指到 extscache 的 `bin/` |

## 脚本

| 脚本 | 作用 |
|---|---|
| `scripts/asset_tools/build_usb.py` | 生成简化 peg（模板，改尺寸） |
| `scripts/asset_tools/build_usb_slot.py` | 生成带孔 socket（孔=插口+间隙+倒角） |
| `scripts/asset_tools/bake_tet.py` | 烤 tet（fTetWild）→ npz |
| `scripts/asset_tools/write_tet_to_usd.py` | tet 写进 USD |
| `scripts/view_obj.py` / `view_usd.py` / `view_task.py` | 看 obj / 看 USD / 看任务场景 |
| `scripts/convert.py` / `collect_data.py` | OBJ→USD / 跑任务·采数据·录视频 |
