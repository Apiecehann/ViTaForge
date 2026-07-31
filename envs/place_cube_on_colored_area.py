from ._base_task import *
import numpy as np

# Frame 尺寸：外框 100mm x 100mm，内框 90mm x 90mm，厚度 2mm

FRAME_OUTER_SIZE = 0.1000
FRAME_INNER_SIZE = 0.0900
FRAME_THICKNESS = 0.0020
CUBE_SIZE = 0.0300

LEFT_FRAME_BASE_POSE = Pose([0.5, -0.08, 0.002], [1, 0, 0, 0])
RIGHT_FRAME_BASE_POSE = Pose([0.5, 0.08, 0.002], [1, 0, 0, 0])
CUBE_BASE_POSE = Pose([0.38, 0.0, 0.002], [1, 0, 0, 0])

# reset 时只在 xy 平面加入扰动，z 维保持不变，避免物体初始高度偏离桌面。
FRAME_XY_NOISE = (0.01, 0.01, 0.0)
CUBE_XY_NOISE = (0.02, 0.04, 0.0)
PLACE_XY_NOISE = 0.02
PLACE_TARGET_Z = 0.020
RELEASE_TARGET_Z = 0.010
GRASP_ROTATE_NOISE = np.deg2rad(10.0)
GRASP_HEIGHT = CUBE_SIZE * 0.5
GRASP_HEIGHT_NOISE = 0.003
PRE_GRASP_HEIGHT = 0.0200
LIFT_HEIGHT = 0.0300
SUCCESS_TABLE_HEIGHT_TOLERANCE = 0.003
SUCCESS_MIN_GRIPPER_OPEN_RATIO = 0.4
TARGET_AREA_COLORS = ("yellow", "blue")
FRAME_ORDERS = ("yellow_left", "blue_left")
TASK_INSTRUCTION = "Place the red cube on the blue area."
TASK_INITIAL_JOINT_POS = {
    "panda_joint1": -0.010809095,
    "panda_joint2": 0.096037410,
    "panda_joint3": 0.000734462,
    "panda_joint4": -2.433035851,
    "panda_joint5": 0.035354517,
    "panda_joint6": 2.500859022,
    "panda_joint7": 0.741,
    "panda_finger.*": 0.02,
}


@configclass
class TaskCfg(BaseTaskCfg):
    # "random" 表示每个 episode 随机选择目标框，也可设为 "yellow" 或 "blue"。
    target_area: Literal["random", "yellow", "blue"] = "blue"
    # 左右按照 head 相机视野定义：世界 -Y 为左，+Y 为右。
    frame_order: Literal["yellow_left", "blue_left"] = "yellow_left"

    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.8, 0.0, 0.15), rot=(0.54167522, 0.454519478, 0.454519478, 0.54167522), convention="opengl"),
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
    step_lim = 260


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        if cfg.target_area not in ("random", *TARGET_AREA_COLORS):
            raise ValueError(
                f"target_area must be one of ('random', 'yellow', 'blue'), got {cfg.target_area!r}"
            )
        if cfg.frame_order not in FRAME_ORDERS:
            raise ValueError(
                f"frame_order must be one of ('yellow_left', 'blue_left'), got {cfg.frame_order!r}"
            )
        self.frame_order = cfg.frame_order
        if self.frame_order == "yellow_left":
            self.yellow_frame_base_pose = LEFT_FRAME_BASE_POSE
            self.blue_frame_base_pose = RIGHT_FRAME_BASE_POSE
        else:
            self.yellow_frame_base_pose = RIGHT_FRAME_BASE_POSE
            self.blue_frame_base_pose = LEFT_FRAME_BASE_POSE
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        # 仅覆盖本任务的默认关节位置，不影响 robot_cfg.py 中其他任务的配置。
        cfg.robot.robot.init_state.joint_pos.update(TASK_INITIAL_JOINT_POS)
        return cfg

    def create_actors(self):
        self.yellow_area = self._actor_manager.add_from_usd_file(
            name="yellow_area",
            asset_path="square_frame_yellow.usd",
            pose=self.yellow_frame_base_pose,
            density=1e6,
        )
        self.blue_area = self._actor_manager.add_from_usd_file(
            name="blue_area",
            asset_path="square_frame_blue.usd",
            pose=self.blue_frame_base_pose,
            density=1e6,
        )
        self.wooden_cube = self._actor_manager.add_from_usd_file(
            name="wooden_cube",
            asset_path="Cube_red.usd",
            pose=CUBE_BASE_POSE,
            density=1e3,
        )

    def _select_target_area(self):
        if self.cfg.target_area == "random":
            self.target_area_color = str(self.rng.choice(TARGET_AREA_COLORS))
        else:
            self.target_area_color = self.cfg.target_area
        self.target_area = {
            "yellow": self.yellow_area,
            "blue": self.blue_area,
        }[self.target_area_color]

    def _reset_actors(self):
        # 两个框共用同一 xy 扰动，保持相对位置和朝向不变。
        frame_offset = self.create_noise(list(FRAME_XY_NOISE))
        cube_offset = self.create_noise(list(CUBE_XY_NOISE))
        yellow_pose = self.yellow_frame_base_pose.add_offset(frame_offset)
        blue_pose = self.blue_frame_base_pose.add_offset(frame_offset)
        cube_pose = CUBE_BASE_POSE.add_offset(cube_offset)

        self.yellow_area.set_pose(yellow_pose)
        self.blue_area.set_pose(blue_pose)
        self.wooden_cube.set_pose(cube_pose)
        self._select_target_area()

        self.metadata["yellow_area_xy_noise"] = frame_offset.p.tolist()
        self.metadata["blue_area_xy_noise"] = frame_offset.p.tolist()
        self.metadata["cube_xy_noise"] = cube_offset.p.tolist()
        self.metadata["yellow_area_pose"] = yellow_pose.tolist()
        self.metadata["blue_area_pose"] = blue_pose.tolist()
        self.metadata["wooden_cube_pose"] = cube_pose.tolist()
        self.metadata["target_area_color"] = self.target_area_color
        self.metadata["frame_order"] = self.frame_order

    def _get_local_z_bounds(self, actor, fallback_half_height):
        # 优先使用 actor 缓存的表面点估计局部 z 上下界。
        if hasattr(actor, "origin_surf_pts"):
            local_points = np.asarray(actor.origin_surf_pts)
            return float(np.min(local_points[:, 2])), float(np.max(local_points[:, 2]))
        return -fallback_half_height, fallback_half_height

    def _get_world_z_bounds(self, actor, actor_pose, fallback_half_height):
        # 使用最终位姿将局部表面点变换到世界坐标，倾斜时也能得到真实最低表面高度。
        if hasattr(actor, "origin_surf_pts"):
            local_points = np.asarray(actor.origin_surf_pts)
            world_points = local_points @ actor_pose.R.T + actor_pose.p
            return float(np.min(world_points[:, 2])), float(np.max(world_points[:, 2]))
        return (
            float(actor_pose.p[2] - fallback_half_height),
            float(actor_pose.p[2] + fallback_half_height),
        )

    def _sample_place_pose(self):
        area_pose = self.target_area.get_pose()
        cube_min_z, cube_max_z = self._get_local_z_bounds(self.wooden_cube, CUBE_SIZE * 0.5)
        area_min_z, area_max_z = self._get_local_z_bounds(self.target_area, FRAME_THICKNESS * 0.5)
        place_xy_noise = self.rng.uniform(-PLACE_XY_NOISE, PLACE_XY_NOISE, size=2)
        place_pose = Pose(
            [
                area_pose.p[0] + place_xy_noise[0],
                area_pose.p[1] + place_xy_noise[1],
                PLACE_TARGET_Z,
            ],
            [1, 0, 0, 0],
        )
        self.metadata["place_xy_noise"] = place_xy_noise.tolist()
        self.metadata["target_area_local_z_bounds"] = [area_min_z, area_max_z]
        self.metadata["cube_local_z_bounds"] = [cube_min_z, cube_max_z]
        self.metadata["target_place_pose"] = place_pose.tolist()
        return place_pose

    def pre_move(self):
        self.delay(10)
        self.move(self.atom.open_gripper(0.5), tag="open_gripper_for_cube")

        cube_pose = self.wooden_cube.get_pose()
        grasp_rotate = self.rng.uniform(-GRASP_ROTATE_NOISE, GRASP_ROTATE_NOISE)
        grasp_height = GRASP_HEIGHT + self.rng.uniform(-GRASP_HEIGHT_NOISE, GRASP_HEIGHT_NOISE)
        target_pose = cube_pose.add_bias([0.0, 0.0, grasp_height]).add_rotation([0.0, grasp_rotate, 0.0])
        target_mat = target_pose.to_transformation_matrix()
        grasp_pose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        )
        pre_grasp_gripper_center_pose = Pose(
            grasp_pose.p + np.array([0.0, 0.0, PRE_GRASP_HEIGHT]),
            grasp_pose.q,
        )
        pre_grasp_ee_pose = self._robot_manager.gripper_center_to_ee(pre_grasp_gripper_center_pose)
        self.move(
            self.atom.move_to_pose(pre_grasp_ee_pose),
            tag="move_above_wooden_cube",
        )

        contact_point_id = self.wooden_cube.register_point(grasp_pose, type="contact")
        self.move(self.atom.grasp_actor(
            self.wooden_cube,
            contact_point_id=contact_point_id,
            is_close=False,
            pre_dis=0.01,
            dis=0.0,
        ), tag="approach_wooden_cube")
        self.move(self.atom.close_gripper(), tag="close_wooden_cube")

        self.metadata["grasp_rotate_rad"] = float(grasp_rotate)
        self.metadata["grasp_rotate_deg"] = float(np.rad2deg(grasp_rotate))
        self.metadata["grasp_height"] = float(grasp_height)
        self.metadata["pre_grasp_height"] = float(PRE_GRASP_HEIGHT)
        self.metadata["pre_grasp_gripper_center_pose"] = pre_grasp_gripper_center_pose.tolist()
        self.metadata["pre_grasp_ee_pose"] = pre_grasp_ee_pose.tolist()
        self.metadata["grasp_pose"] = grasp_pose.tolist()

    def _play_once(self):
        self.move(self.atom.move_by_displacement(z=LIFT_HEIGHT), tag="lift_wooden_cube")

        place_pose = self._sample_place_pose()
        self.move(self.atom.place_actor(
            self.wooden_cube,
            target_pose=place_pose,
            pre_dis=0.02,
            dis=0.00,
            is_open=False,
        ), tag=f"place_wooden_cube_on_{self.target_area_color}_area", time_dilation_factor=0.5)

        cube_pose_before_descent = self.wooden_cube.get_pose()
        descent_z = RELEASE_TARGET_Z - cube_pose_before_descent.p[2]
        self.move(
            self.atom.move_by_displacement(z=descent_z),
            tag="lower_wooden_cube_to_release_height"
        )
        self.metadata["release_target_z"] = float(RELEASE_TARGET_Z)
        self.metadata["cube_pose_before_descent"] = cube_pose_before_descent.tolist()
        self.metadata["descent_z"] = float(descent_z)
        self.metadata["cube_pose_before_release"] = self.wooden_cube.get_pose().tolist()

        self.move(self.atom.open_gripper(0.5), tag="release_wooden_cube")
        self.delay(20, is_save=False)

    def _get_success_diagnostics(self):
        target_area_pose = self.target_area.get_pose()
        cube_pose = self.wooden_cube.get_pose()
        # 只相对于本轮指定区域检查位置，放入另一个框不会判定成功。
        cube_in_target_area = cube_pose.rebase(target_area_pose)
        inner_half_size = FRAME_INNER_SIZE * 0.5
        cube_half_size = CUBE_SIZE * 0.5
        center_x_ok = bool(abs(cube_in_target_area.p[0]) <= inner_half_size)
        center_y_ok = bool(abs(cube_in_target_area.p[1]) <= inner_half_size)
        footprint_x_ok = bool(abs(cube_in_target_area.p[0]) + cube_half_size <= inner_half_size)
        footprint_y_ok = bool(abs(cube_in_target_area.p[1]) + cube_half_size <= inner_half_size)
        cube_min_world_z, cube_max_world_z = self._get_world_z_bounds(
            self.wooden_cube,
            cube_pose,
            cube_half_size,
        )
        table_height = float(self.cfg.uipc_sim.ground_height)
        cube_table_height_error = float(cube_min_world_z - table_height)
        on_table_ok = bool(abs(cube_table_height_error) <= SUCCESS_TABLE_HEIGHT_TOLERANCE)

        gripper_qpos = float(self._robot_manager.get_gripper_qpos())
        gripper_max_qpos = float(self._robot_manager.gripper_max_qpos)
        gripper_open_ratio = gripper_qpos / gripper_max_qpos if gripper_max_qpos > 0.0 else 0.0
        gripper_open_ok = bool(gripper_open_ratio >= SUCCESS_MIN_GRIPPER_OPEN_RATIO)
        return {
            "target_area_color": self.target_area_color,
            "target_area_pose": target_area_pose.tolist(),
            "yellow_area_pose": self.yellow_area.get_pose().tolist(),
            "blue_area_pose": self.blue_area.get_pose().tolist(),
            "wooden_cube_pose": cube_pose.tolist(),
            "cube_pose_in_target_area": cube_in_target_area.tolist(),
            "cube_xy_in_target_area": cube_in_target_area.p[:2].tolist(),
            "frame_outer_half_size": float(FRAME_OUTER_SIZE * 0.5),
            "frame_inner_half_size": float(inner_half_size),
            "cube_half_size": float(cube_half_size),
            "center_x_ok": center_x_ok,
            "center_y_ok": center_y_ok,
            "footprint_x_ok": footprint_x_ok,
            "footprint_y_ok": footprint_y_ok,
            "xy_ok": bool(footprint_x_ok and footprint_y_ok),
            "table_height": table_height,
            "cube_min_world_z": cube_min_world_z,
            "cube_max_world_z": cube_max_world_z,
            "cube_table_height_error": cube_table_height_error,
            "table_height_tolerance": float(SUCCESS_TABLE_HEIGHT_TOLERANCE),
            "on_table_ok": on_table_ok,
            "gripper_qpos": gripper_qpos,
            "gripper_max_qpos": gripper_max_qpos,
            "gripper_open_ratio": float(gripper_open_ratio),
            "min_gripper_open_ratio": float(SUCCESS_MIN_GRIPPER_OPEN_RATIO),
            "gripper_open_ok": gripper_open_ok,
        }

    def check_success(self):
        diagnostics = self._get_success_diagnostics()
        self.metadata["success_diagnostics"] = diagnostics
        return bool(
            diagnostics["xy_ok"]
            and diagnostics["on_table_ok"]
            and diagnostics["gripper_open_ok"]
        )
