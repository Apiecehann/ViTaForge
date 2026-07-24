from ._base_task import *
import numpy as np


TASK_INSTRUCTION = "Grasp the red gear and rotate it to turn the gear pair."
GEAR_PHASE_OFFSET_DEG = 6.0
GEAR_RED_Q = (1.0, 0.0, 0.0, 0.0)
GEAR_BLUE_Q = (
    float(np.cos(np.deg2rad(GEAR_PHASE_OFFSET_DEG) * 0.5)),
    0.0,
    0.0,
    float(np.sin(np.deg2rad(GEAR_PHASE_OFFSET_DEG) * 0.5)),
)

# 齿轮组物体初始化位姿。齿轮局部 z=0 是底面，底座局部 z=0 也是底面。
BASE_INITIAL_POSITION = (0.45, 0.0, 0.002)
GEAR_CENTER_DISTANCE = 0.072
RED_GEAR_INITIAL_POSITION = (
    BASE_INITIAL_POSITION[0] - GEAR_CENTER_DISTANCE * 0.5,
    BASE_INITIAL_POSITION[1],
    0.010,
)
BLUE_GEAR_INITIAL_POSITION = (
    BASE_INITIAL_POSITION[0] + GEAR_CENTER_DISTANCE * 0.5,
    BASE_INITIAL_POSITION[1],
    0.010,
)
RESET_XY_NOISE = (0.03, 0.03, 0.0)

# 红色齿轮高度为 30mm；抓取点位于中上部，减少夹爪与底座碰撞的概率。
RED_GEAR_HEIGHT = 0.030
GEAR_GRASP_HEIGHT_RATIO = 0.66
PRE_GRASP_DISTANCE = 0.050

# 与 insert_USB_0724 任务保持一致的机械臂初始状态。
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
    step_lim = 200


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 1.0
        cfg.sim.physics_material.static_friction = 1.0
        cfg.uipc_sim.contact.default_friction_ratio = 1.0
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        cfg.robot.robot.init_state.joint_pos.update(TASK_INITIAL_JOINT_POS)
        return cfg

    def create_actors(self):
        base_pose = Pose(BASE_INITIAL_POSITION, GEAR_RED_Q)
        red_pose = Pose(RED_GEAR_INITIAL_POSITION, GEAR_RED_Q)
        blue_pose = Pose(BLUE_GEAR_INITIAL_POSITION, GEAR_BLUE_Q)

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
        # 三个物体整体共享同一世界坐标系下的 XY 偏移，保持齿轮啮合关系不变。
        offset = self.create_noise(list(RESET_XY_NOISE))

        base_pose = Pose(BASE_INITIAL_POSITION, GEAR_RED_Q).add_offset(offset, coord="world")
        red_pose = Pose(RED_GEAR_INITIAL_POSITION, GEAR_RED_Q).add_offset(offset, coord="world")
        blue_pose = Pose(BLUE_GEAR_INITIAL_POSITION, GEAR_BLUE_Q).add_offset(offset, coord="world")
        self.base.set_pose(base_pose)
        self.red_gear.set_pose(red_pose)
        self.blue_gear.set_pose(blue_pose)

        self.metadata["base_pose"] = base_pose.tolist()
        self.metadata["reset_xy_offset"] = offset.p[:2].tolist()
        self.metadata["gear_center_distance"] = GEAR_CENTER_DISTANCE
        self.metadata["gear_phase_offset_deg"] = GEAR_PHASE_OFFSET_DEG
        self.metadata["red_gear_pose"] = red_pose.tolist()
        self.metadata["blue_gear_pose"] = blue_pose.tolist()

    def _record_initial_red_pose(self):
        self.initial_red_pose = self.red_gear.get_pose()
        self.metadata["initial_red_pose"] = self.initial_red_pose.tolist()

    def reset(self, seed: int = -1, instructions: list[str] | None = None, options: dict[str, Any] | None = None):
        ret = super().reset(seed=seed, instructions=instructions, options=options)
        # eval_from_start 会跳过 pre_move；此时在 reset 和物理稳定完成后单独记录成功判定基准。
        if self.cfg.skip_pre_move:
            self._record_initial_red_pose()
        return ret

    def pre_move(self):
        self._record_initial_red_pose()

        self.move(self.atom.open_gripper(0.5), delay=False)

        red_pose = self.red_gear.get_pose()
        target_pose = red_pose.add_bias(
            [0.0, 0.0, RED_GEAR_HEIGHT * GEAR_GRASP_HEIGHT_RATIO],
            coord="world",
        )
        # grasp_from=[0,0,1] 表示从目标点上方沿 -Z 方向接近，爪子姿态尽量竖直向下。
        grasp_pose = construct_grasp_pose(
            target_pose.p,
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        )

        self.metadata["gear_grasp_height_ratio"] = GEAR_GRASP_HEIGHT_RATIO
        self.metadata["gear_grasp_height"] = RED_GEAR_HEIGHT * GEAR_GRASP_HEIGHT_RATIO
        self.metadata["pre_grasp_dis"] = PRE_GRASP_DISTANCE
        self.metadata["red_grasp_pose"] = grasp_pose.tolist()

        cid = self.red_gear.register_point(grasp_pose, type="contact")
        self.move(
            self.atom.grasp_actor(
                self.red_gear,
                contact_point_id=cid,
                pre_dis=PRE_GRASP_DISTANCE,
                dis=0.0,
                is_close=False,
            ),
            tag="grasp_red_gear",
            delay=False,
        )
        self.move(self.atom.close_gripper(), tag="close_gripper", delay=False)

    def _play_once(self):
        # 抓住红色齿轮后绕夹爪本地 Z 轴旋转 45deg；实际成功只要求红齿轮转过 30deg。
        self.move(
            self.atom.move_by_displacement(rpy=[0.0, 0.0, np.pi / 4], xyz_coord="local"),
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
        return yaw_delta_deg >= 30.0
