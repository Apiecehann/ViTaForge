from ._base_task import *
import numpy as np
from uipc import view


is_random = True
TASK_INSTRUCTION = "Grasp the red gear and rotate it to turn the gear pair."
GEAR_PHASE_OFFSET_DEG = 6.0
GEAR_RED_Q = (1.0, 0.0, 0.0, 0.0)
GEAR_BLUE_Q = (
    float(np.cos(np.deg2rad(GEAR_PHASE_OFFSET_DEG) * 0.5)),
    0.0,
    0.0,
    float(np.sin(np.deg2rad(GEAR_PHASE_OFFSET_DEG) * 0.5)),
)
BASE_INITIAL_POSITION = np.array([0.45, 0.0, 0.002])
GEAR_CENTER_DISTANCE = 0.072
RED_GEAR_INITIAL_POSITION = np.array([0.45 - 0.036, 0.0, 0.010])
BLUE_GEAR_INITIAL_POSITION = np.array([0.45 + 0.036, 0.0, 0.010])
RESET_XY_NOISE = 0.030
GEAR_PLATE_HEIGHT = 0.010
GEAR_SHAFT_TOP = 0.030
GEAR_SHAFT_CENTER_HEIGHT = 0.5 * (GEAR_PLATE_HEIGHT + GEAR_SHAFT_TOP)
GEAR_ASSET_ROOT = "task_0724/turn_gear_pair"
XENSE_GEAR_PHYSICS_ASSET_PATH = f"{GEAR_ASSET_ROOT}/gear_physics_proxy.usda"
XENSE_GEAR_BASE_PHYSICS_ASSET_PATH = (
    f"{GEAR_ASSET_ROOT}/gear_base_physics_proxy.usda"
)
XENSE_GEAR_TRANSLATION_STRENGTH = 5.0e3
GEAR_ROTATION_STRENGTH = 5.0e3
GEAR_DRIVE_MIN_YAW_STEP = 1.0e-6


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        # CameraCfg(
        #     name="head",
        #     prim_path="/World/envs/env_.*/Camera",
        #     offset=CameraCfg.OffsetCfg(pos=(0.74, 0.0, 0.066), rot=(0.512, 0.512, 0.487, 0.487), convention="opengl"),
        #     data_types=["rgb", "depth"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
        #     ),
        #     width=480,
        #     height=270,
        #     update_period=1/120
        # ),
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.7, 0.14, 0.11), rot=(0.370608, 0.310977, 0.562556, 0.670428), convention="opengl"),
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
            update_period=1 / 120,
        ),
    ]
    step_lim = 500


class Task(BaseTask):
    @staticmethod
    def _is_xense_cfg(cfg):
        return getattr(cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )

    def __init__(self, cfg: BaseTaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 1.0
        cfg.sim.physics_material.static_friction = 1.0
        cfg.uipc_sim.contact.default_friction_ratio = 1.0
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        is_xense = self._is_xense_cfg(self.cfg)
        # base 中心放在 x=0.45；底座局部 z=0 是底面，所以世界 z=0.002 表示放在桌面上方 2mm。
        base_pose = Pose(BASE_INITIAL_POSITION, GEAR_RED_Q)
        # 两个齿轮中心距取 0.072m，所以红色在 0.45-0.036，蓝色在 0.45+0.036。
        # 齿轮局部 z=0 是底面；底板 5mm + 小方柱 2mm，顶面约 0.002+0.007=0.009m。
        # 齿轮底面设 0.010m，留约 1mm 初始间隙。
        red_pose = Pose(RED_GEAR_INITIAL_POSITION, GEAR_RED_Q)
        blue_pose = Pose(BLUE_GEAR_INITIAL_POSITION, GEAR_BLUE_Q)

        self.base = self._actor_manager.add_from_usd_file(
            name="base",
            asset_path=(
                XENSE_GEAR_BASE_PHYSICS_ASSET_PATH
                if is_xense
                else f"{GEAR_ASSET_ROOT}/gear_pair_base.usd"
            ),
            pose=base_pose,
            density=1e5,
            visual_asset_path=(
                f"{GEAR_ASSET_ROOT}/gear_pair_base.usd" if is_xense else None
            ),
            show_physics_mesh=not is_xense,
            keep_constrained=is_xense,
        )
        self.red_gear = self._actor_manager.add_from_usd_file(
            name="red_gear",
            asset_path=(
                XENSE_GEAR_PHYSICS_ASSET_PATH
                if is_xense
                else f"{GEAR_ASSET_ROOT}/gear_pair_red.usd"
            ),
            pose=red_pose,
            density=1e3,
            visual_asset_path=(
                f"{GEAR_ASSET_ROOT}/gear_pair_red.usd" if is_xense else None
            ),
            show_physics_mesh=not is_xense,
            keep_constrained=True,
        )
        self.blue_gear = self._actor_manager.add_from_usd_file(
            name="blue_gear",
            asset_path=(
                XENSE_GEAR_PHYSICS_ASSET_PATH
                if is_xense
                else f"{GEAR_ASSET_ROOT}/gear_pair_blue.usd"
            ),
            pose=blue_pose,
            density=1e3,
            visual_asset_path=(
                f"{GEAR_ASSET_ROOT}/gear_pair_blue.usd" if is_xense else None
            ),
            show_physics_mesh=not is_xense,
            keep_constrained=True,
        )

        # Keep the real gear meshes shape-stable without replacing their
        # collision geometry with simplified shafts.
        translation_strength = (
            XENSE_GEAR_TRANSLATION_STRENGTH if is_xense else 5.0e3
        )
        for gear in (self.red_gear, self.blue_gear):
            strength_ratio = gear.uipc_meshes[0].instances().find("strength_ratio")
            if strength_ratio is None:
                raise RuntimeError("Missing SoftTransformConstraint strength_ratio")
            strength_view = view(strength_ratio)
            strength_view[:, 0, :] = translation_strength
            strength_view[:, 1, :] = GEAR_ROTATION_STRENGTH

    def _reset_actors(self):
        self._gear_drive_enabled = False
        self._gear_drive_angle = 0.0
        self._gear_drive_prev_gripper_pose = None
        self._gear_drive_pose_updates = 0

        # 三个物体整体在 xy 平面内随机平移，范围是 +/-5mm，保持相对位置不变。
        xy_offset = (
            self.rng.uniform(-RESET_XY_NOISE, RESET_XY_NOISE, size=2)
            if is_random
            else np.zeros(2)
        )
        offset = np.array([xy_offset[0], xy_offset[1], 0.0])

        base_pose = Pose(BASE_INITIAL_POSITION + offset, GEAR_RED_Q)
        red_pose = Pose(RED_GEAR_INITIAL_POSITION + offset, GEAR_RED_Q)
        blue_pose = Pose(BLUE_GEAR_INITIAL_POSITION + offset, GEAR_BLUE_Q)
        self.base.set_pose(base_pose)
        self.red_gear.set_pose(red_pose)
        self.blue_gear.set_pose(blue_pose)
        self._reset_red_pose = red_pose
        self._reset_blue_pose = blue_pose

        self.metadata["base_pose"] = base_pose.tolist()
        self.metadata["reset_xy_offset"] = xy_offset.tolist()
        self.metadata["gear_center_distance"] = GEAR_CENTER_DISTANCE
        self.metadata["gear_phase_offset_deg"] = GEAR_PHASE_OFFSET_DEG
        self.metadata["gear_collision_asset"] = (
            "xense_stepped_plate_and_shaft_proxy_with_real_visual"
            if self._is_xense_cfg(self.cfg)
            else "real_gear_mesh_usd"
        )
        self.metadata["gear_proxy_geometry"] = {
            "plate_height_m": GEAR_PLATE_HEIGHT,
            "shaft_top_m": GEAR_SHAFT_TOP,
            "shaft_radius_m": 0.015,
            "center_hole_radius_m": 0.007075,
        }
        self.metadata["gear_constraint_translation_strength"] = float(
            XENSE_GEAR_TRANSLATION_STRENGTH
            if self._is_xense_cfg(self.cfg)
            else 5.0e3
        )
        self.metadata["gear_constraint_rotation_strength"] = float(
            GEAR_ROTATION_STRENGTH
        )
        self.metadata["red_gear_pose"] = red_pose.tolist()
        self.metadata["blue_gear_pose"] = blue_pose.tolist()

    @staticmethod
    def _rotate_pose_about_world_z(pose, angle):
        pose_mat = pose.to_transformation_matrix()
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        world_rotation = np.array(
            [
                [cos_angle, -sin_angle, 0.0],
                [sin_angle, cos_angle, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        target_mat = pose_mat.copy()
        target_mat[:3, :3] = world_rotation @ pose_mat[:3, :3]
        return Pose.from_matrix(target_mat)

    def _sync_driven_gears_to_gripper(self):
        if not getattr(self, "_gear_drive_enabled", False):
            return

        current_gripper_pose = self._robot_manager.get_gripper_center_pose()
        previous_gripper_pose = self._gear_drive_prev_gripper_pose
        self._gear_drive_prev_gripper_pose = current_gripper_pose
        if previous_gripper_pose is None:
            return

        current_rotation = current_gripper_pose.to_transformation_matrix()[:3, :3]
        previous_rotation = previous_gripper_pose.to_transformation_matrix()[:3, :3]
        relative_rotation = current_rotation @ previous_rotation.T
        yaw_step = float(
            np.arctan2(relative_rotation[1, 0], relative_rotation[0, 0])
        )
        if abs(yaw_step) <= GEAR_DRIVE_MIN_YAW_STEP:
            return

        self._gear_drive_angle += yaw_step
        self.red_gear.set_pose(
            self._rotate_pose_about_world_z(self.initial_red_pose, self._gear_drive_angle)
        )
        self.blue_gear.set_pose(
            self._rotate_pose_about_world_z(self.initial_blue_pose, -self._gear_drive_angle)
        )
        self._actor_manager.update(dt=0.0)
        self._gear_drive_pose_updates += 1

    def _stop_gear_drive(self):
        self._gear_drive_enabled = False
        self._gear_drive_prev_gripper_pose = None
        self.metadata["gear_drive_enabled_final"] = False
        self.metadata["gear_drive_pose_updates"] = int(
            self._gear_drive_pose_updates
        )

    def _step(self, is_save: bool = True):
        # The full-mesh UIPC revolute joint is prohibitively slow with Xense
        # contact. Drive real-gear transform targets from measured wrist motion.
        self._sync_driven_gears_to_gripper()
        return super()._step(is_save=is_save)

    def _record_initial_gear_poses(self):
        if self._is_xense_cfg(self.cfg) and hasattr(self, "_reset_red_pose"):
            self.initial_red_pose = self._reset_red_pose
            self.initial_blue_pose = self._reset_blue_pose
            initial_pose_source = "reset_target_pose"
        else:
            self.initial_red_pose = self.red_gear.get_pose()
            self.initial_blue_pose = self.blue_gear.get_pose()
            initial_pose_source = "settled_actor_pose"
        self.metadata["initial_red_pose"] = self.initial_red_pose.tolist()
        self.metadata["initial_blue_pose"] = self.initial_blue_pose.tolist()
        self.metadata["initial_gear_pose_source"] = initial_pose_source

    def reset(
        self,
        seed: int = -1,
        instructions: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ):
        ret = super().reset(seed=seed, instructions=instructions, options=options)
        if self.cfg.skip_pre_move or (
            options is not None and not options.get("run_pre_move", True)
        ):
            self._record_initial_gear_poses()
        return ret

    def pre_move(self):
        self._record_initial_gear_poses()
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        open_percent = 1.0 if is_xense else 0.5
        self.move(self.atom.open_gripper(open_percent), delay=False)
        self.metadata["gripper_open_percent"] = float(open_percent)
        if is_xense:
            self._approach_red_gear()
            self._update_render()

    def _approach_red_gear(self):
        if not hasattr(self, "initial_red_pose"):
            self._record_initial_gear_poses()
        red_pose = self.red_gear.get_pose()
        # The proxy follows the real two-level mesh: a 10mm plate and a
        # 10-30mm shaft. Use the shaft center as the geometric reference; the
        # Xense-specific bias accounts for the pad origin and keeps the pads
        # clear of the toothed plate.
        grasp_height_bias = self.get_xense_grasp_height_bias("xense_gear_grasp_height_bias")
        grasp_world_y_bias = self.get_xense_grasp_height_bias("xense_gear_grasp_world_y_bias")
        target_pose = (
            red_pose
            .add_bias(
                [0.0, 0.0, GEAR_SHAFT_CENTER_HEIGHT + grasp_height_bias],
                coord="world",
            )
            .add_bias([0.0, grasp_world_y_bias, 0.0], coord="world")
        )
        # grasp_from=[0,0,1] 表示从目标点上方沿 -Z 方向接近，爪子姿态尽量竖直向下。
        grasp_pose = construct_grasp_pose(
            target_pose.p,
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        )

        self.metadata["gear_grasp_height_ratio"] = (
            GEAR_SHAFT_CENTER_HEIGHT / GEAR_SHAFT_TOP
        )
        self.metadata["gear_grasp_height"] = GEAR_SHAFT_CENTER_HEIGHT
        self.metadata["gear_grasp_target_local_z"] = (
            GEAR_SHAFT_CENTER_HEIGHT + grasp_height_bias
        )
        self.metadata["gear_grasp_height_bias"] = float(grasp_height_bias)
        self.metadata["gear_grasp_world_y_bias"] = float(grasp_world_y_bias)
        self.metadata["pre_grasp_dis"] = 0.050
        self.metadata["red_grasp_pose"] = grasp_pose.tolist()

        cid = self.red_gear.register_point(grasp_pose, type="contact")
        # pre_dis=0.050 会让 planner 先到抓取点正上方 5cm，再沿竖直方向下压到抓取高度。
        approach_actions = self.atom.grasp_actor(
            self.red_gear,
            contact_point_id=cid,
            pre_dis=0.050,
            dis=0.0,
            is_close=False,
        )
        self.move(
            approach_actions,
            tag="grasp_red_gear",
            delay=False,
        )
        self.record_xense_grasp_debug("xense_after_approach_red_gear", self.red_gear)

    def _close_red_gear(self):
        close_percent = self.get_xense_close_percent("xense_gear_close_percent")
        self.move(
            self.atom.close_gripper(pos=close_percent),
            tag="close_gripper",
            delay=False,
            gripper_depth_threshold=self.get_xense_adaptive_grasp_depth_threshold(
                "xense_gear_adaptive_grasp_depth_threshold"
            ),
            gripper_require_both_contacts=self.get_xense_adaptive_grasp_require_both_contacts(
                "xense_gear_adaptive_grasp_require_both_contacts"
            ),
        )
        self._gear_drive_angle = 0.0
        self._gear_drive_prev_gripper_pose = self._robot_manager.get_gripper_center_pose()
        self._gear_drive_enabled = True
        self.metadata["gear_drive_mode"] = "wrist_soft_joint_real_mesh_1_to_1"
        self.settle_xense_after_close(is_save=False)
        self.record_xense_grasp_debug("xense_after_close_red_gear", self.red_gear)
        self.metadata["gripper_close_percent"] = float(close_percent)

    def _grasp_red_gear(self):
        self._approach_red_gear()
        self._close_red_gear()

    def _play_once(self):
        if self._is_xense_cfg(self.cfg):
            self._close_red_gear()
        else:
            self._grasp_red_gear()
        # One 90-degree wrist turn is sufficient. A second turn reaches the
        # +/-180-degree yaw wrap and cannot provide a reliable direction check.
        for turn_idx in range(1):
            self.move(
                self.atom.move_by_displacement(
                    rpy=[0.0, 0.0, np.pi / 2],
                    xyz_coord="local",
                ),
                tag=f"turn_red_gear_{turn_idx + 1}",
                delay=False,
                time_dilation_factor=0.5,
                constraint_pose=[0, 0, 0, 1, 1, 1],
            )
            # Apply the final gripper pose, which otherwise arrives one UIPC
            # frame after the last arm control target.
            self._sync_driven_gears_to_gripper()
            self._step(is_save=True)
            result = self._get_gear_pair_turn_result()
            self._record_turn_result(result, suffix=f"_after_turn_{turn_idx + 1}")
            self._save_metadata()
            print(
                "[turn_gear_pair] "
                f"turn {turn_idx + 1}, red yaw: {result['red_yaw_signed_deg']:.2f} deg, "
                f"blue yaw: {result['blue_yaw_signed_deg']:.2f} deg, "
                f"red drift: {result['red_center_delta']:.4f} m, "
                f"blue drift: {result['blue_center_delta']:.4f} m",
                flush=True,
            )
            if self._is_physical_success(result):
                break

        # The last set_pose targets remain active through the existing soft
        # transform constraints. Stop following the wrist so later render or
        # settle steps cannot re-inject the same pose or rotate the pair again.
        self._stop_gear_drive()
        result = self._get_gear_pair_turn_result()
        self._record_turn_result(result)
        self.metadata["gear_drive_angle_deg"] = float(
            np.rad2deg(self._gear_drive_angle)
        )
        self.record_xense_grasp_debug("xense_after_turn_red_gear", self.red_gear)
        print(
            "[turn_gear_pair] "
            f"red yaw: {result['red_yaw_signed_deg']:.2f} deg, "
            f"blue yaw: {result['blue_yaw_signed_deg']:.2f} deg, "
            f"red drift: {result['red_center_delta']:.4f} m, "
            f"blue drift: {result['blue_center_delta']:.4f} m"
        )

    @staticmethod
    def _get_actor_turn_result(actor, initial_pose):
        if initial_pose is None:
            return float("inf"), float("inf"), float("inf"), 0.0

        actor_pose = actor.get_pose()
        init_mat = initial_pose.to_transformation_matrix()
        curr_mat = actor_pose.to_transformation_matrix()
        init_x = init_mat[:3, 0].copy()
        curr_x = curr_mat[:3, 0].copy()
        init_x[2] = 0.0
        curr_x[2] = 0.0
        init_x /= np.linalg.norm(init_x) + 1e-8
        curr_x /= np.linalg.norm(curr_x) + 1e-8

        yaw_delta = np.arctan2(
            np.dot(np.cross(init_x, curr_x), np.array([0.0, 0.0, 1.0])),
            np.dot(init_x, curr_x),
        )
        position_delta = actor_pose.p - initial_pose.p
        center_delta = np.linalg.norm(position_delta)
        center_xy_delta = np.linalg.norm(position_delta[:2])
        vertical_delta = abs(float(position_delta[2]))
        return (
            float(center_delta),
            float(center_xy_delta),
            vertical_delta,
            float(np.rad2deg(yaw_delta)),
        )

    def _get_gear_pair_turn_result(self):
        (
            red_center_delta,
            red_center_xy_delta,
            red_vertical_delta,
            red_yaw_signed_deg,
        ) = self._get_actor_turn_result(
            self.red_gear, getattr(self, "initial_red_pose", None)
        )
        (
            blue_center_delta,
            blue_center_xy_delta,
            blue_vertical_delta,
            blue_yaw_signed_deg,
        ) = self._get_actor_turn_result(
            self.blue_gear, getattr(self, "initial_blue_pose", None)
        )
        return {
            "red_center_delta": red_center_delta,
            "blue_center_delta": blue_center_delta,
            "red_center_xy_delta": red_center_xy_delta,
            "blue_center_xy_delta": blue_center_xy_delta,
            "red_vertical_delta": red_vertical_delta,
            "blue_vertical_delta": blue_vertical_delta,
            "red_yaw_signed_deg": red_yaw_signed_deg,
            "blue_yaw_signed_deg": blue_yaw_signed_deg,
        }

    def _record_turn_result(self, result, suffix=""):
        for key, value in result.items():
            self.metadata[f"{key}{suffix}"] = float(value)
        self.metadata[f"red_yaw_delta_deg{suffix}"] = abs(
            float(result["red_yaw_signed_deg"])
        )
        self.metadata[f"blue_yaw_delta_deg{suffix}"] = abs(
            float(result["blue_yaw_signed_deg"])
        )

    def _is_physical_success(self, result):
        red_yaw = result["red_yaw_signed_deg"]
        blue_yaw = result["blue_yaw_signed_deg"]
        rotation_success = (
            abs(red_yaw) >= 45.0
            and abs(blue_yaw) >= 30.0
            and red_yaw * blue_yaw < 0.0
        )
        if self._is_xense_cfg(self.cfg):
            position_success = (
                result["red_center_xy_delta"] <= 0.003
                and result["blue_center_xy_delta"] <= 0.003
                and result["red_vertical_delta"] <= 0.005
                and result["blue_vertical_delta"] <= 0.005
            )
        else:
            position_success = (
                result["red_center_delta"] <= 0.003
                and result["blue_center_delta"] <= 0.003
            )
        return rotation_success and position_success

    def check_success(self):
        result = self._get_gear_pair_turn_result()
        self._record_turn_result(result)
        return self._is_physical_success(result)
