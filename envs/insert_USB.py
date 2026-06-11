from ._base_task import *
import numpy as np
import torch

# ============================================================================
# Insert USB —— 纯 peg-in-hole: 开局夹爪已握住 USB, 直接对准插入孔
#   peg : USB.usd      (刚体, 体 19.9x9.5mm, 全长 63mm, 插口 12x4.5mm 朝 -z, 突出 11mm)
#   slot: USBSlot.usd  (带孔底座, 外形 26x16x13mm, 孔 15x7.5mm 深 11mm, 孔口在顶面, density=1e5)
#   流程: _reset_actors 把 USB 摆到夹爪中心(插口朝下)+瞬间近闭合 -> pre_move 夹紧并移到孔口上方
#         -> _play_once 精对准 + 下压插入
# ----------------------------------------------------------------------------
#   USB 局部几何(原点在体心附近): 插口尖 -31.5mm, 机身底 -20.5mm, 机身顶 +31.5mm
#   => 原点->插口尖 0.0315m; 原点->机身中心 +0.0055m
# ============================================================================

SLOT_HEIGHT = 0.013      # 底座高(m), 孔口在顶面
HOLE_DEPTH  = 0.011      # 孔深(m)
USB_TIP     = 0.0315     # USB 原点 -> 插口尖(m)
BODY_CENTER = 0.0055     # USB 原点 -> 机身中心(m, 沿 +z)
# [tune] 夹爪握持: 把机身中心放到夹爪中心 -> USB 原点在夹爪中心下方 BODY_CENTER
GRIP_Z      = -BODY_CENTER


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
        # USB 全程握在手里, 让规划器把它当随动件而非静态障碍物(否则长 peg 盖住夹爪->规划必失败)
        self.planner_ignore_actors = {'usb'}

    # ---------------------------------------------------------------- actors
    def create_actors(self):
        # 底座: 重刚体当"固定插座"
        self.slot = self._actor_manager.add_from_usd_file(
            name='slot', asset_path="USBSlot.usd",
            pose=Pose([0.55, 0.0, 0.002], [1, 0, 0, 0]), density=1e5,
        )
        # USB: 先放在安全空位, _reset_actors 再精确摆到夹爪中心(避免初始穿透)
        self.usb = self._actor_manager.add_from_usd_file(
            name='usb', asset_path="USB.usd",
            pose=Pose([0.45, 0.0, 0.25], [1, 0, 0, 0]),
        )

    # ---------------------------------------------------------------- reset
    def _reset_actors(self):
        # 底座 xy ±5mm 随机
        base_offset = self.create_noise([0.005, 0.005, 0.0])
        base_pose = Pose([0.55, 0.0, self.slot.get_pose()[2]], [1, 0, 0, 0]).add_offset(base_offset)
        self.slot.set_pose(base_pose)

        # USB 摆到夹爪中心: home 位姿夹爪朝下, identity 朝向即插口朝下
        gc = self._robot_manager.get_gripper_center_pose()
        usb_pose = Pose(gc.p, [1, 0, 0, 0]).add_bias([0.0, 0.0, GRIP_Z])
        self.usb.set_pose(usb_pose)
        # 让夹爪"开局即近闭合": 先 force 瞬间把爪收到贴近 USB(留极小缝, 不穿透 -> 不触发 IPC 报错),
        # 消除从 0.02 大张口缓慢闭合的可见过程; 再把目标设为 0, 趁 USB 仍被约束时完成最后挤压建立夹持。
        n, dev = self.num_envs, self.device
        self._robot_manager.set_gripper(torch.full((n, 2), 0.006, device=dev), force=True)
        self._robot_manager.set_gripper(torch.zeros((n, 2), device=dev), force=False)

    # ---------------------------------------------------------------- script
    def pre_move(self):
        self.delay(10)
        # 闭合夹爪夹紧已就位的 USB
        self.move(self.atom.close_gripper())

        # 孔口目标(给 place_actor 的是 USB 原点目标): 插口尖到孔口 -> 原点在孔口上方 USB_TIP
        self.hole_pose = self.slot.get_pose().add_bias([0.0, 0.0, SLOT_HEIGHT + USB_TIP])
        noise = self.create_noise([0.004, 0.004, 0.0])
        self.noise_pose = self.hole_pose.add_offset(noise)

        # 移到孔口上方(带噪声, 插口尖悬在孔口上方 ~6cm, 留足间隙避免长 peg 撞底座导致规划失败)
        self.move(self.atom.place_actor(
            self.usb, target_pose=self.noise_pose.add_bias([0.0, 0.0, 0.06]),
            pre_dis=0.05, dis=0.02, is_open=False,
        ))

    def _play_once(self):
        # 精对准到孔口
        self.move(self.atom.place_actor(
            self.usb, target_pose=self.hole_pose,
            pre_dis=0.02, dis=0.004, is_open=False,
        ), time_dilation_factor=0.5)
        # 柔顺下压插入(放松绕 z 约束), 比之前下压更深(总~1.5x孔深=插到底+轻压)
        self.move(self.atom.move_by_displacement(
            z=HOLE_DEPTH * 0.9, xyz_coord='local',
        ), time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
        self.move(self.atom.move_by_displacement(
            z=HOLE_DEPTH * 0.6, xyz_coord='local',
        ), time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
        self.delay(20, is_save=True)

    # ---------------------------------------------------------------- success
    def check_success(self, z_threshold=0.006):
        # 成功 = USB 相对孔口对正且插入到位
        usb_pose = self.usb.get_pose().rebase(self.hole_pose)
        self.metadata['rel_pose'] = usb_pose.tolist()
        return np.all(np.abs(usb_pose.p[0:2]) < np.array([0.005, 0.005])) \
            and usb_pose.p[2] < z_threshold \
            and np.dot(usb_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.95
