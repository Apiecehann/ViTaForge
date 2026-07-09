from ._base_task import *
import numpy as np


GEAR_RED_Q = (1.0, 0.0, 0.0, 0.0)
GEAR_BLUE_Q = (
    float(np.cos(np.deg2rad(6.0) * 0.5)),
    0.0,
    0.0,
    float(np.sin(np.deg2rad(6.0) * 0.5)),
)


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
    def __init__(self, cfg: BaseTaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 1.0
        cfg.sim.physics_material.static_friction = 1.0
        cfg.uipc_sim.contact.default_friction_ratio = 1.0
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        # base 中心放在 x=0.45；底座局部 z=0 是底面，所以世界 z=0.002 表示放在桌面上方 2mm。
        base_pose = Pose([0.45, 0.0, 0.002], GEAR_RED_Q)
        # 两个齿轮中心距取 0.072m，所以红色在 0.45-0.036，蓝色在 0.45+0.036。
        # 齿轮局部 z=0 是底面；底板 5mm + 小方柱 2mm，顶面约 0.002+0.007=0.009m。
        # 齿轮底面设 0.010m，留约 1mm 初始间隙。
        red_pose = Pose([0.45 - 0.036, 0.0, 0.010], GEAR_RED_Q)
        blue_pose = Pose([0.45 + 0.036, 0.0, 0.010], GEAR_BLUE_Q)

        self.base = self._actor_manager.add_from_usd_file(
            name="base",
            asset_path="gear_pair_base.usd",
            pose=base_pose,
            density=1e5,
        )
        self.red_gear = self._actor_manager.add_from_usd_file(
            name="red_gear",
            asset_path="gear_pair_red.usd",
            pose=red_pose,
            density=1e3,
        )
        self.blue_gear = self._actor_manager.add_from_usd_file(
            name="blue_gear",
            asset_path="gear_pair_blue.usd",
            pose=blue_pose,
            density=1e3,
        )

    def _reset_actors(self):
        # 三个物体整体在 xy 平面内随机平移，范围是 +/-5mm，保持相对位置不变。
        xy_offset = self.rng.uniform(-0.005, 0.005, size=2)
        offset = np.array([xy_offset[0], xy_offset[1], 0.0])

        base_pose = Pose(np.array([0.45, 0.0, 0.002]) + offset, GEAR_RED_Q)
        red_pose = Pose(np.array([0.45 - 0.036, 0.0, 0.010]) + offset, GEAR_RED_Q)
        blue_pose = Pose(np.array([0.45 + 0.036, 0.0, 0.010]) + offset, GEAR_BLUE_Q)
        self.base.set_pose(base_pose)
        self.red_gear.set_pose(red_pose)
        self.blue_gear.set_pose(blue_pose)

        self.metadata["base_pose"] = base_pose.tolist()
        self.metadata["reset_xy_offset"] = xy_offset.tolist()
        self.metadata["gear_center_distance"] = 0.072
        self.metadata["gear_phase_offset_deg"] = 6.0
        self.metadata["red_gear_pose"] = red_pose.tolist()
        self.metadata["blue_gear_pose"] = blue_pose.tolist()

    def pre_move(self):
        self.initial_red_pose = self.red_gear.get_pose()
        self.metadata["initial_red_pose"] = self.initial_red_pose.tolist()

        self.move(self.atom.open_gripper(0.5), delay=False)

        red_pose = self.red_gear.get_pose()
        # 齿轮高度是 30mm，抓 66% 高度处：0.030*0.66=0.0198m，尽量抓在齿轮中上部。
        target_pose = red_pose.add_bias([0.0, 0.0, 0.030 * 0.66], coord="world")
        # grasp_from=[0,0,1] 表示从目标点上方沿 -Z 方向接近，爪子姿态尽量竖直向下。
        grasp_pose = construct_grasp_pose(
            target_pose.p,
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        )

        self.metadata["gear_grasp_height_ratio"] = 0.66
        self.metadata["gear_grasp_height"] = 0.030 * 0.66
        self.metadata["pre_grasp_dis"] = 0.050
        self.metadata["red_grasp_pose"] = grasp_pose.tolist()

        cid = self.red_gear.register_point(grasp_pose, type="contact")
        # pre_dis=0.050 会让 planner 先到抓取点正上方 5cm，再沿竖直方向下压到抓取高度。
        self.move(
            self.atom.grasp_actor(
                self.red_gear,
                contact_point_id=cid,
                pre_dis=0.050,
                dis=0.0,
                is_close=False,
            ),
            tag="grasp_red_gear",
            delay=False,
        )
        self.move(self.atom.close_gripper(), tag="close_gripper", delay=False)

    def _play_once(self):
        # 抓住红色齿轮后绕夹爪本地 Z 轴旋转 90deg；实际成功只要求红齿轮转过 45deg。
        self.move(
            self.atom.move_by_displacement(rpy=[0.0, 0.0, np.pi / 2], xyz_coord="local"),
            tag="turn_red_gear",
            delay=False,
            time_dilation_factor=0.5,
            constraint_pose=[0, 0, 0, 1, 1, 1],
        )

        center_delta, yaw_delta_deg = self._get_red_turn_result()
        self.metadata["red_center_delta"] = float(center_delta)
        self.metadata["red_yaw_delta_deg"] = float(yaw_delta_deg)
        print(f"[turn_gear_pair] red yaw delta: {yaw_delta_deg:.2f} deg, center delta: {center_delta:.4f} m")

    def _get_red_turn_result(self):
        if not hasattr(self, "initial_red_pose"):
            return float("inf"), 0.0

        red_pose = self.red_gear.get_pose()
        init_mat = self.initial_red_pose.to_transformation_matrix()
        curr_mat = red_pose.to_transformation_matrix()
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
        center_delta = np.linalg.norm(red_pose.p - self.initial_red_pose.p)
        return center_delta, abs(float(np.rad2deg(yaw_delta)))

    def check_success(self):
        center_delta, yaw_delta_deg = self._get_red_turn_result()
        self.metadata["red_center_delta"] = float(center_delta)
        self.metadata["red_yaw_delta_deg"] = float(yaw_delta_deg)
        return yaw_delta_deg >= 45.0
