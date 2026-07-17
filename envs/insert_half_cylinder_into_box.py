from ._base_task import *
import numpy as np

# 半圆柱体蓝色木块：assets/objects/Blue_half_cylinder.usd，尺寸为：半径22.8mm，高度30mm
# 木箱：assets/objects/wooden_box_semicircle_hole.usd，半圆柱孔尺寸：半径24mm

BOX_SIZE = 0.1000
BOX_WALL_THICKNESS = 0.0050
HALF_CYLINDER_HEIGHT = 0.0300

BOX_BASE_POSE = Pose([0.4, -0.05, 0.002], [1, 0, 0, 0])
HALF_CYLINDER_BASE_POSE = Pose([0.4, 0.15, 0.002], [1, 0, 0, 0])
# 抓取后先移动到盒子上方的中间高度，避免直接横移时碰到盒壁。
LIFT_TARGET_POSE = Pose([0.4, 0.05, 0.112], [1, 0, 0, 0])

# reset 时盒子和半圆柱都只加 xy 平面小扰动，z 保持不变，避免初始状态离开桌面。
XY_NOISE = (0.005, 0.005, 0.0)
GRASP_ROTATE_NOISE = np.deg2rad(10.0)
# 抓取点位于半圆柱半高附近，并加入小幅高度和旋转随机性，提高演示覆盖范围。
GRASP_HEIGHT = HALF_CYLINDER_HEIGHT * 0.5
GRASP_HEIGHT_NOISE = 0.003
PRE_INSERT_CLEARANCE = 0.002
INSERT_DEPTH = 0.010
PRE_PLACE_DISTANCE = 0.050

# 盒子局部坐标系下的有效内部范围，用于判断半圆柱是否真正落在盒内。
INNER_X_MIN = -BOX_SIZE * 0.5 + BOX_WALL_THICKNESS
INNER_X_MAX = BOX_SIZE * 0.5
INNER_Y_MIN = -BOX_SIZE * 0.5 + BOX_WALL_THICKNESS
INNER_Y_MAX = BOX_SIZE * 0.5 - BOX_WALL_THICKNESS
INNER_Z_MIN = BOX_WALL_THICKNESS
INNER_Z_MAX = BOX_SIZE - BOX_WALL_THICKNESS


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.7, 0.0, 0.16), rot=(0.555057, 0.465748, 0.443006, 0.527954), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.6, focus_distance=1.0, horizontal_aperture=2.4, clipping_range=(0.1, 100.0)
            ),
            width=480,
            height=270,
            update_period=1/120,
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480,
            height=270,
            update_period=1/120,
        ),
    ]
    step_lim = 300


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        # 插入和放置任务对接触稳定性敏感，提高摩擦可以减少抓取后滑动或放置时弹出。
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        # 创建带半圆孔的木盒作为近似固定基座，蓝色半圆柱作为可抓取/插入物体。
        self.wooden_box = self._actor_manager.add_from_usd_file(
            name="wooden_box_semicircle_hole",
            asset_path="wooden_box_semicircle_hole.usd",
            pose=BOX_BASE_POSE,
            density=1e6,
        )
        self.blue_half_cylinder = self._actor_manager.add_from_usd_file(
            name="Blue_half_cylinder",
            asset_path="Blue_half_cylinder.usd",
            pose=HALF_CYLINDER_BASE_POSE,
            density=1e3,
        )

    def _reset_actors(self):
        # 每个 episode 随机化盒子和半圆柱的 xy 位置，让放置目标和抓取目标都有轻微分布变化。
        box_offset = self.create_noise(list(XY_NOISE))
        half_cylinder_offset = self.create_noise(list(XY_NOISE))
        box_pose = BOX_BASE_POSE.add_offset(box_offset)
        half_cylinder_pose = HALF_CYLINDER_BASE_POSE.add_offset(half_cylinder_offset)

        self.wooden_box.set_pose(box_pose)
        self.blue_half_cylinder.set_pose(half_cylinder_pose)

        # 保存 reset 后的真实初始位姿，便于离线排查放置失败和复现实验。
        self.metadata["box_xy_noise"] = box_offset.p.tolist()
        self.metadata["half_cylinder_xy_noise"] = half_cylinder_offset.p.tolist()
        self.metadata["wooden_box_pose"] = box_pose.tolist()
        self.metadata["blue_half_cylinder_pose"] = half_cylinder_pose.tolist()

    def pre_move(self):
        # 正式动作前等待物体稳定，再打开夹爪准备抓取蓝色半圆柱。
        self.delay(10)
        self.move(self.atom.open_gripper(0.5), tag="open_gripper_for_blue_half_cylinder")

        half_cylinder_pose = self.blue_half_cylinder.get_pose()
        # 在半圆柱半高附近采样抓取点，并绕局部 y 轴加入轻微随机旋转。
        grasp_rotate = self.rng.uniform(-GRASP_ROTATE_NOISE, GRASP_ROTATE_NOISE)
        grasp_height = GRASP_HEIGHT + self.rng.uniform(-GRASP_HEIGHT_NOISE, GRASP_HEIGHT_NOISE)
        grasp_target_pose = half_cylinder_pose.add_bias([0.0, 0.0, grasp_height]).add_rotation([0.0, grasp_rotate, 0.0])
        target_mat = grasp_target_pose.to_transformation_matrix()
        # 根据抓取点、接近方向和夹爪横向方向，构造最终末端抓取姿态。
        grasp_pose = construct_grasp_pose(
            grasp_target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        )
        # 注册 contact point 后，atom.grasp_actor 会先到预抓取距离，再靠近该局部抓取点。
        contact_point_id = self.blue_half_cylinder.register_point(grasp_pose, type="contact")

        self.move(self.atom.grasp_actor(
            self.blue_half_cylinder,
            contact_point_id=contact_point_id,
            is_close=False,
            pre_dis=0.05,
        ), tag="approach_blue_half_cylinder")
        self.move(self.atom.close_gripper(), tag="close_blue_half_cylinder")

        # 记录抓取随机量和抓取姿态，方便分析数据中的抓取分布与失败样本。
        self.metadata["grasp_rotate_rad"] = float(grasp_rotate)
        self.metadata["grasp_rotate_deg"] = float(np.rad2deg(grasp_rotate))
        self.metadata["grasp_height"] = float(grasp_height)
        self.metadata["grasp_pose"] = grasp_pose.tolist()

    def _sample_pre_insert_pose(self):
        box_pose = self.wooden_box.get_pose()
        # 预插入点位于木盒中心正上方，z 为盒顶高度再加 2mm clearance。
        pre_insert_pose = Pose(
            [
                box_pose.p[0],
                box_pose.p[1],
                box_pose.p[2] + BOX_SIZE + PRE_INSERT_CLEARANCE,
            ],
            box_pose.q,
        )
        self.metadata["pre_insert_pose"] = pre_insert_pose.tolist()
        return pre_insert_pose

    def _play_once(self):
        # 先把半圆柱带到盒子上方的中间位姿，减少横移时碰撞盒壁的概率。
        self.move(self.atom.place_actor(
            self.blue_half_cylinder,
            target_pose=LIFT_TARGET_POSE,
            pre_dis=PRE_PLACE_DISTANCE,
            dis=0.0,
            is_open=False,
        ), tag="lift_blue_half_cylinder", time_dilation_factor=0.5)

        pre_insert_pose = self._sample_pre_insert_pose()
        # 再移动到当前随机化木盒正上方的预插入点，保持夹爪闭合不释放物体。
        self.move(self.atom.place_actor(
            self.blue_half_cylinder,
            target_pose=pre_insert_pose,
            pre_dis=PRE_PLACE_DISTANCE,
            dis=0.0,
            is_open=False,
        ), tag="move_blue_half_cylinder_to_pre_insert", time_dilation_factor=0.5)

        # 从盒口上方向下插入 1cm，使半圆柱进入盒内有效空间。
        self.move(
            self.atom.move_by_displacement(z=-INSERT_DEPTH),
            tag="insert_blue_half_cylinder_into_box",
            time_dilation_factor=0.5,
        )
        self.move(self.atom.open_gripper(0.5), tag="release_blue_half_cylinder")
        # 释放后等待较长时间但不保存，给物体足够时间在盒内稳定下来再做 success 检查。
        self.delay(420, is_save=False)

    def _get_success_diagnostics(self):
        box_pose = self.wooden_box.get_pose()
        half_cylinder_pose = self.blue_half_cylinder.get_pose()
        # 将半圆柱位姿转换到木盒坐标系下，直接判断其中心点是否落在盒内有效范围。
        half_cylinder_in_box = half_cylinder_pose.rebase(box_pose)

        x_ok = bool(INNER_X_MIN <= half_cylinder_in_box.p[0] <= INNER_X_MAX)
        y_ok = bool(INNER_Y_MIN <= half_cylinder_in_box.p[1] <= INNER_Y_MAX)
        z_ok = bool(INNER_Z_MIN <= half_cylinder_in_box.p[2] <= INNER_Z_MAX)

        return {
            "wooden_box_pose": box_pose.tolist(),
            "blue_half_cylinder_pose": half_cylinder_pose.tolist(),
            "half_cylinder_pose_in_box": half_cylinder_in_box.tolist(),
            "half_cylinder_xyz_in_box": half_cylinder_in_box.p.tolist(),
            "inner_x_range": [float(INNER_X_MIN), float(INNER_X_MAX)],
            "inner_y_range": [float(INNER_Y_MIN), float(INNER_Y_MAX)],
            "inner_z_range": [float(INNER_Z_MIN), float(INNER_Z_MAX)],
            "x_ok": x_ok,
            "y_ok": y_ok,
            "z_ok": z_ok,
            "xy_ok": bool(x_ok and y_ok),
        }

    def check_success(self):
        # 成功要求半圆柱中心在盒内有效 xyz 范围内；完整诊断写入 metadata 便于离线分析。
        diagnostics = self._get_success_diagnostics()
        self.metadata["success_diagnostics"] = diagnostics
        return diagnostics["xy_ok"] and diagnostics["z_ok"]
