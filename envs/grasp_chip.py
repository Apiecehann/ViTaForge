from ._base_task import *
import os
import numpy as np
import torch

# ============================================================================
# Grasp Chip —— 桌面抓起一根易碎薄薯片, 夹太用力就 "碎"(fail)
#   chip : CHIP.usd  (薄弯细条刚体, 82.5x18x2.4mm 带弧度)
#   流程(录数据): 张爪 -> 上方下扎抓取 -> 自适应轻夹 -> 一次直接抬起 -> 停顿保持
#   "力" = 触觉 gelpad 压入量 pressure = far_plane - get_min_depth()(mm), 夹越狠越大。
#   全程最大压入量 > MAX_PRESSURE => 夹太狠 => 薯片"碎" => 失败(不需要真断裂)。
#   正常自适应轻夹约 7mm, 所以 MAX_PRESSURE 设在其之上留余量。
# ----------------------------------------------------------------------------
#   gsmini: tactile_far_plane=34mm(无接触), 自适应轻夹目标 depth=27.5(≈6.5mm 压入)
# ============================================================================

CHIP_REST_Z   = 0.01      # [tune] 桌面体心高度(chip7那版)
CHIP_QUAT     = [-0.70710678, 0.0, 0.0, 0.70710678]  # 绕竖直轴的摆放朝向
GRASP_Z       = 0.004     # [tune] 抓取点相对薯片体心的 z 偏置(夹弯起的两端附近)
LIFT          = 0.08      # 一次直接抬到位的高度(m)
MAX_PRESSURE  = float(os.environ.get('CHIP_MAX_PRESSURE', '8.5'))  # [tune] 压入量(mm)上限; 超过=夹太狠=碎=fail (正常约7mm, 死夹~8.7mm)
HOLD_PRESSURE = 1.0       # [tune] 压入量(mm)低于此=基本没接触=掉了
# 夹力开关: 0=默认自适应轻夹; >0=闭到指定触觉深度(越小夹越紧). 用于故意夹太狠: CHIP_GRIP_DEPTH=20
GRIP_DEPTH    = float(os.environ.get('CHIP_GRIP_DEPTH', '0'))


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.74, 0.0, 0.066), rot=(0.512, 0.512, 0.487, 0.487), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
            ),
            width=480, height=270, update_period=1/120,
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480, height=270, update_period=1/120,
        ),
    ]
    step_lim = 600


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)
        # 抓起后薯片随夹爪移动 -> 规划时当随动件(否则手持件盖住夹爪导致规划失败)
        self.planner_ignore_actors = {'chip'}
        self.max_pressure_seen = 0.0   # 全程最大压入量(=最大夹持力)

    # ---------------------------------------------------------------- 力
    def _pressure(self):
        return self.cfg.robot.tactile_far_plane - torch.min(self._tactile_manager.get_min_depth()).item()

    def _track(self):
        p = self._pressure()
        self.max_pressure_seen = max(self.max_pressure_seen, p)
        if p > MAX_PRESSURE:           # 夹太狠 -> 薯片碎 -> 本回合失败
            self.plan_success = False

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        self.chip = self._actor_manager.add_from_usd_file(
            name='chip', asset_path="CHIP.usd",
            pose=Pose([0.5, 0.0, CHIP_REST_Z], CHIP_QUAT),
        )

    def _reset_actors(self):
        offset = self.create_noise([0.005, 0.005, 0.0])   # 落点 xy ±5mm 随机
        self.chip.set_pose(Pose([0.5, 0.0, CHIP_REST_Z], CHIP_QUAT).add_offset(offset))

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(10)
        self.start_ee_z = self._robot_manager.get_ee_pose()[2]
        self.max_pressure_seen = 0.0

    def _play_once(self):
        # 张爪开到最大(夹的是长度方向, 跨度大)
        self.move(self.atom.open_gripper(1.0))
        # 上方下扎抓取(camera_up 用薯片 Y 轴 -> 夹两端)
        grasp_rotate = self.rng.uniform(-np.pi / 18, np.pi / 18)
        target_pose = self.chip.get_pose().add_bias([0.0, 0.0, GRASP_Z]).add_rotation([0.0, grasp_rotate, 0.0])
        tmat = target_pose.to_transformation_matrix()
        cpose = construct_grasp_pose(target_pose.p, tmat[:3, 2], tmat[:3, 1])
        cid = self.chip.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(self.chip, contact_point_id=cid, is_close=False)); self._track()
        # 自适应夹: 默认轻夹(不夹碎); 若设了 CHIP_GRIP_DEPTH 则闭到更深=夹更紧(用于故意夹太狠)
        if GRIP_DEPTH > 0:
            self.move(self.atom.close_gripper(depth_threshold=GRIP_DEPTH)); self._track()
        else:
            self.move(self.atom.close_gripper()); self._track()
        # 记录抬升前(抓取后)的 ee 高度: 判"有没有从抓取点升起来"(抓取要先下降, 不能跟 home 比)
        self.grasp_ee_z = self._robot_manager.get_ee_pose()[2]
        # 一次直接抬到位, 再停顿保持
        self.move(self.atom.move_by_displacement(z=LIFT, xyz_coord='world')); self._track()
        self.delay(20, is_save=True); self._track()
        print(f"[CHIP] 最大压入量 = {self.max_pressure_seen:.2f} mm  (上限 MAX_PRESSURE={MAX_PRESSURE}, 超过=fail)", flush=True)

    # ---------------------------------------------------------------- success / fail
    def check_early_stop(self):
        # 全程压入量超上限 -> 夹碎 -> 失败
        return self.max_pressure_seen > MAX_PRESSURE

    def check_success(self):
        p = self._pressure()
        ee_z = self._robot_manager.get_ee_pose()[2]
        self.metadata['max_pressure_mm'] = self.max_pressure_seen
        held = p > HOLD_PRESSURE                              # 还夹着(没掉)
        not_crushed = self.max_pressure_seen <= MAX_PRESSURE  # 全程没夹碎
        lifted = (ee_z - self.grasp_ee_z) > LIFT * 0.7        # 从抓取点升起来了(不跟 home 比)
        return held and not_crushed and lifted
