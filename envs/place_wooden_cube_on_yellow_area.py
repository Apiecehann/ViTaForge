from ._base_task import *
import numpy as np

# Frame 尺寸：外框 100mm x 100mm，内框 90mm x 90mm，厚度 2mm

FRAME_OUTER_SIZE = 0.1000
FRAME_INNER_SIZE = 0.0900
FRAME_THICKNESS = 0.0020
CUBE_SIZE = 0.0300

# Keep the thin frame clear of the UIPC ground contact band. It remains
# constrained after reset, so this offset is stable throughout the episode.
FRAME_BASE_POSE = Pose([0.4, 0.0, 0.003], [1, 0, 0, 0])
CUBE_BASE_POSE = Pose([0.35, 0.25, 0.002], [1, 0, 0, 0])

# reset 时只在 xy 平面加小扰动，z 维保持 0，避免物体初始高度被随机噪声带离桌面。
XY_NOISE = (0.01, 0.01, 0.0)
PLACE_XY_NOISE = 0.01
GRASP_ROTATE_NOISE = np.deg2rad(10.0)
GRASP_HEIGHT = CUBE_SIZE * 0.5
GRASP_HEIGHT_NOISE = 0.003
LIFT_HEIGHT = 0.0300


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.8, 0.0, 0.15), rot=(0.555057, 0.465748, 0.443006, 0.527954), convention="opengl"),
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
        if getattr(cfg, "tactile_sensor_type", "") in ("xensews", "xensews_robotiq"):
            # Advancing the constrained frame and cube together can make the
            # first UIPC reset step take minutes. pre_move starts with ten
            # unsaved settling steps after the cube constraint is released.
            cfg.reset_first_frame_steps = 0
            cfg.reset_after_actor_steps = 0
            cfg.reset_final_steps = 0
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        self.yellow_area = self._actor_manager.add_from_usd_file(
            name="yellow_area",
            asset_path="yellow_square_frame.usd",
            pose=FRAME_BASE_POSE,
            density=1e5,
            keep_constrained=True,
        )
        self.wooden_cube = self._actor_manager.add_from_usd_file(
            name="wooden_cube",
            asset_path="wooden_cube.usd",
            pose=CUBE_BASE_POSE,
            density=1e3
        )

    def _reset_actors(self):
        # 每个 episode 都重新采样黄色区域和方块的初始 xy 位置，让策略看到轻微的位姿变化。
        area_offset = self.create_noise(list(XY_NOISE))
        cube_offset = self.create_noise(list(XY_NOISE))
        area_pose = FRAME_BASE_POSE.add_offset(area_offset)
        cube_pose = CUBE_BASE_POSE.add_offset(cube_offset)

        self.yellow_area.set_pose(area_pose)
        self.wooden_cube.set_pose(cube_pose)

        self.metadata["area_xy_noise"] = area_offset.p.tolist()
        self.metadata["cube_xy_noise"] = cube_offset.p.tolist()
        self.metadata["yellow_area_pose"] = area_pose.tolist()
        self.metadata["wooden_cube_pose"] = cube_pose.tolist()

    def _get_local_z_bounds(self, actor, fallback_half_height):
        # 优先使用 actor 预先缓存的表面点估计局部 z 上下界；这样比硬编码高度更适配非规则几何。
        if hasattr(actor, "origin_surf_pts"):
            local_points = np.asarray(actor.origin_surf_pts)
            return float(np.min(local_points[:, 2])), float(np.max(local_points[:, 2]))
        # 如果没有表面点，就退回到传入的半高估计，保持采样逻辑可用。
        return -fallback_half_height, fallback_half_height

    def _sample_place_pose(self):
        area_pose = self.yellow_area.get_pose()
        cube_min_z, cube_max_z = self._get_local_z_bounds(self.wooden_cube, CUBE_SIZE * 0.5)
        area_min_z, area_max_z = self._get_local_z_bounds(self.yellow_area, FRAME_THICKNESS * 0.5)
        # 放置点在黄色区域中心附近随机 +/-5mm，避免每条轨迹完全重合。
        place_xy_noise = self.rng.uniform(-PLACE_XY_NOISE, PLACE_XY_NOISE, size=2)
        # 目标 z 让方块底面刚好落在黄色区域表面：area_z + area_top - cube_bottom。
        place_z = area_pose.p[2] + area_max_z - cube_min_z
        place_pose = Pose(
            [
                area_pose.p[0] + place_xy_noise[0],
                area_pose.p[1] + place_xy_noise[1],
                place_z,
            ],
            [1, 0, 0, 0],
        )
        self.metadata["place_xy_noise"] = place_xy_noise.tolist()
        self.metadata["area_local_z_bounds"] = [area_min_z, area_max_z]
        self.metadata["cube_local_z_bounds"] = [cube_min_z, cube_max_z]
        self.metadata["target_place_pose"] = place_pose.tolist()
        return place_pose

    def pre_move(self):
        # 初始先等待若干仿真步，让物体在物理引擎中稳定，再打开夹爪准备抓取。
        self.delay(10)
        self.move(self.atom.open_gripper(0.5), tag="open_gripper_for_cube")

        cube_pose = self.wooden_cube.get_pose()
        # 抓取姿态以方块中心为基准，上移到半高附近；绕局部 y 轴加少量随机旋转，提高数据多样性。
        grasp_rotate = self.rng.uniform(-GRASP_ROTATE_NOISE, GRASP_ROTATE_NOISE)
        grasp_height_bias = self.get_xense_grasp_height_bias("xense_cube_grasp_height_bias")
        grasp_world_y_bias = self.get_xense_grasp_height_bias("xense_cube_grasp_world_y_bias")
        grasp_height = (
            GRASP_HEIGHT
            + self.rng.uniform(-GRASP_HEIGHT_NOISE, GRASP_HEIGHT_NOISE)
            + grasp_height_bias
        )
        target_pose = (
            cube_pose
            .add_bias([0.0, 0.0, grasp_height])
            .add_bias([0.0, grasp_world_y_bias, 0.0], coord="world")
            .add_rotation([0.0, grasp_rotate, 0.0])
        )
        target_mat = target_pose.to_transformation_matrix()
        # construct_grasp_pose 需要抓取点、接近方向和夹爪横向方向；这里沿目标局部 z 方向接近。
        grasp_pose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        )
        # register_point 会把抓取点注册到物体局部坐标，用于后续 atom.grasp_actor 生成接近轨迹。
        contact_point_id = self.wooden_cube.register_point(grasp_pose, type="contact")

        approach_actions = self.atom.grasp_actor(
            self.wooden_cube,
            contact_point_id=contact_point_id,
            is_close=False,
            pre_dis=0.05,
        )
        self.move(approach_actions, tag="approach_wooden_cube")
        self.record_xense_grasp_debug("xense_after_approach_wooden_cube", self.wooden_cube)

        close_percent = self.get_xense_close_percent("xense_cube_close_percent")
        self.move(self.atom.close_gripper(pos=close_percent), tag="close_wooden_cube")
        self.settle_xense_after_close(is_save=False)
        self.record_xense_grasp_debug("xense_after_close_wooden_cube", self.wooden_cube)

        self.metadata["grasp_rotate_rad"] = float(grasp_rotate)
        self.metadata["grasp_rotate_deg"] = float(np.rad2deg(grasp_rotate))
        self.metadata["grasp_height"] = float(grasp_height)
        self.metadata["grasp_height_bias"] = float(grasp_height_bias)
        self.metadata["grasp_world_y_bias"] = float(grasp_world_y_bias)
        self.metadata["gripper_close_percent"] = float(close_percent)
        self.metadata["grasp_pose"] = grasp_pose.tolist()

    def _play_once(self):
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in ("xensews", "xensews_robotiq")
        place_pose = self._sample_place_pose()

        if is_xense:
            cube_pose = self.wooden_cube.get_pose()
            lift_position = cube_pose.p + np.array([0.0, 0.0, LIFT_HEIGHT])
            self.move_actor_by_world_displacement_to_position(
                self.wooden_cube,
                lift_position,
                tag="lift_wooden_cube",
                metadata_prefix="xense_wooden_cube_lift_path",
            )
            pre_place_position = place_pose.p + np.array([0.0, 0.0, 0.04])
            self.move_actor_by_world_displacement_to_position(
                self.wooden_cube,
                pre_place_position,
                tag="carry_wooden_cube_above_yellow_area",
                metadata_prefix="xense_wooden_cube_carry_path",
            )
            self.move_actor_by_world_displacement_to_position(
                self.wooden_cube,
                place_pose.p,
                tag="place_wooden_cube_on_yellow_area",
                metadata_prefix="xense_wooden_cube_place_path",
            )
        else:
            # 抓牢后先竖直上提 3cm，给方块离开桌面和越过黄色区域边框留出余量。
            self.move(self.atom.move_by_displacement(z=LIFT_HEIGHT), tag="lift_wooden_cube")
            self.move(self.atom.place_actor(
                self.wooden_cube,
                target_pose=place_pose,
                pre_dis=0.02,
                dis=0.005,
                is_open=False,
            ), tag="place_wooden_cube_on_yellow_area", time_dilation_factor=0.5)
        self.move(self.atom.open_gripper(0.5), tag="release_wooden_cube")
        # 松爪后不保存等待帧，避免把纯稳定过程混入动作数据，同时让 success 检查读到更稳定的物理状态。
        self.delay(20, is_save=False)

    def _get_success_diagnostics(self):
        area_pose = self.yellow_area.get_pose()
        cube_pose = self.wooden_cube.get_pose()
        # 将方块位姿转到黄色区域坐标系下，后续 xy 判断就可以直接和方框内半径比较。
        cube_in_area = cube_pose.rebase(area_pose)
        inner_half_size = FRAME_INNER_SIZE * 0.5
        cube_half_size = CUBE_SIZE * 0.5
        # center_* 只检查方块中心是否在框内；footprint_* 进一步要求方块外轮廓也完全落在内框里。
        center_x_ok = bool(abs(cube_in_area.p[0]) <= inner_half_size)
        center_y_ok = bool(abs(cube_in_area.p[1]) <= inner_half_size)
        footprint_x_ok = bool(abs(cube_in_area.p[0]) + cube_half_size <= inner_half_size)
        footprint_y_ok = bool(abs(cube_in_area.p[1]) + cube_half_size <= inner_half_size)
        return {
            "yellow_area_pose": area_pose.tolist(),
            "wooden_cube_pose": cube_pose.tolist(),
            "cube_pose_in_area": cube_in_area.tolist(),
            "cube_xy_in_area": cube_in_area.p[:2].tolist(),
            "frame_outer_half_size": float(FRAME_OUTER_SIZE * 0.5),
            "frame_inner_half_size": float(inner_half_size),
            "cube_half_size": float(cube_half_size),
            "center_x_ok": center_x_ok,
            "center_y_ok": center_y_ok,
            "footprint_x_ok": footprint_x_ok,
            "footprint_y_ok": footprint_y_ok,
            "xy_ok": bool(footprint_x_ok and footprint_y_ok),
        }

    def check_success(self):
        # 成功判定只依赖 xy footprint 是否完全位于黄色方框内；完整诊断写入 metadata 便于离线排查。
        diagnostics = self._get_success_diagnostics()
        self.metadata["success_diagnostics"] = diagnostics
        return diagnostics["xy_ok"]
