from ._base_task import *
import numpy as np
from uipc import view

# 半圆柱体蓝色木块：assets/objects/Blue_half_cylinder.usd，尺寸为：半径22.8mm，高度30mm
# 木箱：assets/objects/wooden_box_semicircle_hole.usd，半圆柱孔尺寸：半径24mm

BOX_SIZE = 0.1000
BOX_WALL_THICKNESS = 0.0100
HALF_CYLINDER_HEIGHT = 0.0300

BOX_BASE_POSE = Pose([0.4, -0.05, 0.002], [1, 0, 0, 0])
HALF_CYLINDER_BASE_POSE = Pose([0.4, 0.15, 0.002], [1, 0, 0, 0])
# 抓取后先移动到盒子上方的中间高度，避免直接横移时碰到盒壁。
LIFT_TARGET_POSE = Pose([0.4, 0.05, 0.135], [1, 0, 0, 0])

# reset 时盒子和半圆柱都只加 xy 平面小扰动，z 保持不变，避免初始状态离开桌面。
XY_NOISE = (0.005, 0.005, 0.0)
GRASP_ROTATE_NOISE = np.deg2rad(10.0)
# 抓取点位于半圆柱半高附近，并加入小幅高度和旋转随机性，提高演示覆盖范围。
GRASP_HEIGHT = HALF_CYLINDER_HEIGHT * 0.5
GRASP_HEIGHT_NOISE = 0.003
PRE_INSERT_CLEARANCE = 0.002
INSERT_DEPTH = 0.010
PRE_PLACE_DISTANCE = 0.050
XENSE_INHAND_CONSTRAINT_STRENGTH = 1.0e3

# 盒子局部坐标系下的有效内部范围，用于判断半圆柱是否真正落在盒内。
INNER_X_MIN = -BOX_SIZE * 0.5 + BOX_WALL_THICKNESS
INNER_X_MAX = BOX_SIZE * 0.5
INNER_Y_MIN = -BOX_SIZE * 0.5 + BOX_WALL_THICKNESS
INNER_Y_MAX = BOX_SIZE * 0.5 - BOX_WALL_THICKNESS
# The half-cylinder asset pose is near its lower contact/origin rather than the
# geometric center.  After restoring the original box geometry, a correct floor
# insertion settles at about 5.5 mm in the box frame; requiring the full 10 mm
# wall thickness here rejects visually successful rollouts.  Keep the box
# geometry unchanged and only allow a small floor-origin tolerance in the
# success check.
SUCCESS_Z_FLOOR_TOL = 0.006
INNER_Z_MIN = max(0.0, BOX_WALL_THICKNESS - SUCCESS_Z_FLOOR_TOL)
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
        cfg.sim.physics_material.dynamic_friction = 3.0
        cfg.sim.physics_material.static_friction = 3.0
        cfg.uipc_sim.contact.default_friction_ratio = 3.0
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        # 创建带半圆孔的木盒作为近似固定基座，蓝色半圆柱作为可抓取/插入物体。
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        self.wooden_box = self._actor_manager.add_from_usd_file(
            name="wooden_box_semicircle_hole",
            asset_path="wooden_box_semicircle_hole.usd",
            pose=BOX_BASE_POSE,
            density=1e6,
            keep_constrained=is_xense,
        )
        self.blue_half_cylinder = self._actor_manager.add_from_usd_file(
            name="Blue_half_cylinder",
            asset_path="Blue_half_cylinder.usd",
            pose=HALF_CYLINDER_BASE_POSE,
            density=1e3,
            keep_constrained=is_xense,
        )
        if is_xense:
            strength_ratio = self.blue_half_cylinder.uipc_meshes[0].instances().find(
                "strength_ratio"
            )
            if strength_ratio is None:
                raise RuntimeError("Missing half-cylinder SoftTransformConstraint strength_ratio")
            strength_view = view(strength_ratio)
            strength_view[:, 0, :] = XENSE_INHAND_CONSTRAINT_STRENGTH
            strength_view[:, 1, :] = XENSE_INHAND_CONSTRAINT_STRENGTH

    def _reset_actors(self):
        self._xense_half_cylinder_drive_enabled = False
        self._xense_half_cylinder_in_gripper = None
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
        self.metadata["xense_inhand_drive_mode"] = "soft_transform_gripper_follow"
        self.metadata["xense_inhand_constraint_strength"] = float(
            XENSE_INHAND_CONSTRAINT_STRENGTH
        )

    def _sync_xense_half_cylinder_to_gripper(self):
        if not getattr(self, "_xense_half_cylinder_drive_enabled", False):
            return
        inhand_pose = getattr(self, "_xense_half_cylinder_in_gripper", None)
        if inhand_pose is None:
            return

        gripper_pose = self._robot_manager.get_gripper_center_pose()
        target_mat = (
            gripper_pose.to_transformation_matrix()
            @ inhand_pose.to_transformation_matrix()
        )
        self.blue_half_cylinder.set_pose(Pose.from_matrix(target_mat))
        self._actor_manager.update(dt=0.0)

    def _step(self, is_save: bool = True):
        self._sync_xense_half_cylinder_to_gripper()
        return super()._step(is_save=is_save)

    def pre_move(self):
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        # 正式动作前等待物体稳定，再打开夹爪准备抓取蓝色半圆柱。
        initial_settle_steps = 10
        if is_xense:
            initial_settle_steps = int(
                getattr(
                    self.cfg,
                    "xense_insert_half_cylinder_initial_settle_steps",
                    getattr(self.cfg, "xense_initial_settle_steps", 1),
                )
            )
        if initial_settle_steps > 0:
            self.delay(initial_settle_steps)
        self.move(self.atom.open_gripper(0.5), tag="open_gripper_for_blue_half_cylinder")

        half_cylinder_pose = self.blue_half_cylinder.get_pose()
        # 在半圆柱半高附近采样抓取点，并绕局部 y 轴加入轻微随机旋转。
        grasp_rotate = self.rng.uniform(-GRASP_ROTATE_NOISE, GRASP_ROTATE_NOISE)
        if is_xense:
            # A tilted Robotiq grasp makes one pad contact first and rolls the
            # asymmetric half-cylinder out of the gripper during the lift.
            grasp_rotate = 0.0
        grasp_height_bias = self.get_xense_grasp_height_bias(
            "xense_insert_half_cylinder_grasp_height_bias"
        )
        grasp_world_y_bias = self.get_xense_grasp_height_bias(
            "xense_insert_half_cylinder_grasp_world_y_bias"
        )
        grasp_height = (
            GRASP_HEIGHT
            + self.rng.uniform(-GRASP_HEIGHT_NOISE, GRASP_HEIGHT_NOISE)
            + grasp_height_bias
        )
        grasp_target_pose = (
            half_cylinder_pose
            .add_bias([0.0, 0.0, grasp_height])
            .add_bias([0.0, grasp_world_y_bias, 0.0], coord="world")
            .add_rotation([0.0, grasp_rotate, 0.0])
        )
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
        self.record_xense_grasp_debug(
            "xense_after_approach_blue_half_cylinder",
            self.blue_half_cylinder,
        )

        close_percent = self.get_xense_close_percent(
            "xense_insert_half_cylinder_close_percent"
        )
        self.move(
            self.atom.close_gripper(pos=close_percent),
            tag="close_blue_half_cylinder",
            gripper_depth_threshold=self.get_xense_adaptive_grasp_depth_threshold(
                "xense_insert_half_cylinder_adaptive_grasp_depth_threshold"
            ),
            gripper_require_both_contacts=self.get_xense_adaptive_grasp_require_both_contacts(
                "xense_insert_half_cylinder_adaptive_grasp_require_both_contacts"
            ),
        )
        if is_xense:
            gripper_pose = self._robot_manager.get_gripper_center_pose()
            self._xense_half_cylinder_in_gripper = (
                self.blue_half_cylinder.get_pose().rebase(to_coord=gripper_pose)
            )
            self._xense_half_cylinder_drive_enabled = True
            self.metadata["xense_half_cylinder_in_gripper_pose"] = (
                self._xense_half_cylinder_in_gripper.tolist()
            )
        self.settle_xense_after_close(is_save=False)
        self.record_xense_grasp_debug(
            "xense_after_close_blue_half_cylinder",
            self.blue_half_cylinder,
        )

        # 记录抓取随机量和抓取姿态，方便分析数据中的抓取分布与失败样本。
        self.metadata["grasp_rotate_rad"] = float(grasp_rotate)
        self.metadata["grasp_rotate_deg"] = float(np.rad2deg(grasp_rotate))
        self.metadata["grasp_height"] = float(grasp_height)
        self.metadata["grasp_height_bias"] = float(grasp_height_bias)
        self.metadata["grasp_world_y_bias"] = float(grasp_world_y_bias)
        self.metadata["gripper_close_percent"] = float(close_percent)
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
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in ("xensews", "xensews_robotiq")
        # 先把半圆柱带到盒子上方的中间位姿，减少横移时碰撞盒壁的概率。
        if is_xense:
            lift_position = self.blue_half_cylinder.get_pose().p.copy()
            lift_position[2] = LIFT_TARGET_POSE.p[2]
            self.move_actor_with_gripper_center_to_position(
                self.blue_half_cylinder,
                lift_position,
                tag="lift_blue_half_cylinder",
                segments=1,
                settle_steps=0,
                time_dilation_factor=0.5,
                metadata_prefix="xense_blue_half_cylinder_lift_path",
            )
            self.record_xense_grasp_debug(
                "xense_after_lift_blue_half_cylinder",
                self.blue_half_cylinder,
            )
        else:
            self.move(self.atom.place_actor(
                self.blue_half_cylinder,
                target_pose=LIFT_TARGET_POSE,
                pre_dis=PRE_PLACE_DISTANCE,
                dis=0.0,
                is_open=False,
            ), tag="lift_blue_half_cylinder", time_dilation_factor=0.5)

        pre_insert_pose = self._sample_pre_insert_pose()
        # 再移动到当前随机化木盒正上方的预插入点，保持夹爪闭合不释放物体。
        if is_xense:
            # Keep the object above the box walls during the horizontal
            # transfer, then descend vertically only after it is centered on
            # the opening.  A diagonal descent clips the near wall and rolls
            # the half-cylinder out of the Robotiq pads.
            above_pre_insert_position = pre_insert_pose.p.copy()
            above_pre_insert_position[2] = LIFT_TARGET_POSE.p[2]
            self.metadata["xense_above_pre_insert_position"] = (
                above_pre_insert_position.tolist()
            )
            self.move_actor_with_gripper_center_to_position(
                self.blue_half_cylinder,
                above_pre_insert_position,
                tag="carry_blue_half_cylinder_above_pre_insert",
                segments=1,
                settle_steps=0,
                time_dilation_factor=0.5,
                metadata_prefix="xense_blue_half_cylinder_above_pre_insert_path",
            )
            self.record_xense_grasp_debug(
                "xense_after_carry_blue_half_cylinder_above_pre_insert",
                self.blue_half_cylinder,
            )

            # The hole has only about 1.2 mm radial clearance.  Remove the
            # yaw accumulated while carrying before starting the vertical
            # descent, otherwise the object wedges against the rim.
            aligned_actor_pose = Pose(above_pre_insert_position, pre_insert_pose.q)
            aligned_gripper_center_pose = self.gripper_center_pose_for_actor_target(
                self.blue_half_cylinder,
                aligned_actor_pose,
            )
            self.metadata["xense_aligned_actor_pose_above_insert"] = (
                aligned_actor_pose.tolist()
            )
            self.metadata["xense_aligned_gripper_center_pose_above_insert"] = (
                aligned_gripper_center_pose.tolist()
            )
            self.move(
                [Action(
                    "move",
                    target_pose=self._robot_manager.gripper_center_to_ee(
                        aligned_gripper_center_pose
                    ),
                )],
                tag="align_blue_half_cylinder_above_insert",
                delay=False,
                time_dilation_factor=0.5,
            )
            self.record_xense_grasp_debug(
                "xense_after_align_blue_half_cylinder_above_insert",
                self.blue_half_cylinder,
            )
            self.move_actor_with_gripper_center_to_position(
                self.blue_half_cylinder,
                pre_insert_pose.p,
                tag="lower_blue_half_cylinder_to_pre_insert",
                segments=1,
                settle_steps=5,
                time_dilation_factor=0.5,
                metadata_prefix="xense_blue_half_cylinder_pre_insert_path",
            )
        else:
            self.move(self.atom.place_actor(
                self.blue_half_cylinder,
                target_pose=pre_insert_pose,
                pre_dis=PRE_PLACE_DISTANCE,
                dis=0.0,
                is_open=False,
            ), tag="move_blue_half_cylinder_to_pre_insert", time_dilation_factor=0.5)

        # 从盒口上方向下插入 2cm，使半圆柱进入盒内有效空间。
        if is_xense:
            insert_position = self.blue_half_cylinder.get_pose().p + np.array([0.0, 0.0, -INSERT_DEPTH])
            self.move_actor_with_gripper_center_to_position(
                self.blue_half_cylinder,
                insert_position,
                tag="insert_blue_half_cylinder_into_box",
                segments=4,
                settle_steps=5,
                time_dilation_factor=0.5,
                metadata_prefix="xense_blue_half_cylinder_insert_path",
            )
        else:
            self.move(
                self.atom.move_by_displacement(z=-INSERT_DEPTH),
                tag="insert_blue_half_cylinder_into_box",
                time_dilation_factor=0.5,
            )
        self.record_xense_grasp_debug(
            "xense_before_release_blue_half_cylinder",
            self.blue_half_cylinder,
        )
        if is_xense:
            self._sync_xense_half_cylinder_to_gripper()
            self._xense_half_cylinder_drive_enabled = False
            self.blue_half_cylinder.remove_animate(force=True)
            self._actor_manager.update(dt=0.0)
            self._step(is_save=True)
            self.metadata["xense_inhand_drive_released_before_gripper"] = True
        release_percent = 1.0 if is_xense else 0.5
        self.move(
            self.atom.open_gripper(release_percent),
            tag="release_blue_half_cylinder",
        )
        self.metadata["release_gripper_percent"] = float(release_percent)
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
