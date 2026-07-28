from ._base_task import *
import numpy as np

# True 时给初始物体位置和若干 motion target 加小扰动; False 时所有随机量为 0。
is_random = True

# 三个杯子的初始位置。通过修改各颜色对应的 Pose 可交换初始位置。
CUP_BASE_POSES = {
    "green": Pose([0.33, 0.00, 0.002], (1.0, 0.0, 0.0, 0.0)),
    "yellow": Pose([0.45, 0.00, 0.002], (1.0, 0.0, 0.0, 0.0)),
    "blue": Pose([0.57, 0.00, 0.002], (1.0, 0.0, 0.0, 0.0)),
}
CUP_ASSET_PATHS = {
    "yellow": "cup_yellow.usd",
    "green": "cup_green.usd",
    "blue": "cup_blue.usd",
}

# 任务选择：left 对应世界 +Y，right 对应世界 -Y。
TARGET_CUP = "yellow"
REFERENCE_CUP = "blue"
PLACEMENT_SIDE = "left"
TASK_INSTRUCTION = f"Move the {TARGET_CUP} cup to the {PLACEMENT_SIDE} of the {REFERENCE_CUP} cup."

CUP_RESET_XY_NOISE = (0.030, 0.030, 0.0)
CUP_MIN_RESET_XY_DISTANCE = 0.075
MAX_RESET_SAMPLE_ATTEMPTS = 100
PRE_PLACE_XYZ_NOISE = 0.010
PRE_PLACE_Y_DISTANCE = 0.120
PRE_PLACE_WORLD_Z = 0.120

TASK_INITIAL_ARM_JOINT_POS = {
    "panda_joint1": -0.010809095,
    "panda_joint2": 0.096037410,
    "panda_joint3": 0.000734462,
    "panda_joint4": -2.433035851,
    "panda_joint5": 0.035354517,
    "panda_joint6": 2.500859022,
    "panda_joint7": 0.741,
}

# 成功判定参数。
SUCCESS_TABLE_HEIGHT_TOLERANCE = 0.010
SUCCESS_UPRIGHT_SCORE_THRESHOLD = 0.95
SUCCESS_MAX_X_ERROR = 0.030
SUCCESS_GRIPPER_OPEN_RATIO = 0.90


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.9, 0.0, 0.25),
                rot=(0.593537, 0.415599, 0.395306, 0.564555),
                convention="opengl",
            ),
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
    step_lim = 400


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        if TARGET_CUP not in CUP_BASE_POSES:
            raise ValueError(f"TARGET_CUP must be one of {tuple(CUP_BASE_POSES)}, got {TARGET_CUP!r}")
        if REFERENCE_CUP not in CUP_BASE_POSES:
            raise ValueError(f"REFERENCE_CUP must be one of {tuple(CUP_BASE_POSES)}, got {REFERENCE_CUP!r}")
        if TARGET_CUP == REFERENCE_CUP:
            raise ValueError("TARGET_CUP and REFERENCE_CUP must be different cups")
        if PLACEMENT_SIDE not in ("left", "right"):
            raise ValueError(f"PLACEMENT_SIDE must be 'left' or 'right', got {PLACEMENT_SIDE!r}")

        # 三个杯子都用较高摩擦, 减少竖直抓取和搬运时的滑动。
        cfg.sim.physics_material.dynamic_friction = 3.0
        cfg.sim.physics_material.static_friction = 3.0
        cfg.uipc_sim.contact.default_friction_ratio = 3.0
        self.target_cup_name = TARGET_CUP
        self.reference_cup_name = REFERENCE_CUP
        self.placement_side = PLACEMENT_SIDE
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        cfg.robot.robot.init_state.joint_pos.update(TASK_INITIAL_ARM_JOINT_POS)
        cfg.robot.robot.init_state.joint_pos.update({
            "panda_finger.*": cfg.robot.gripper_max_qpos,
        })
        return cfg

    def create_actors(self):
        self.cups = {}
        for cup_name, base_pose in CUP_BASE_POSES.items():
            self.cups[cup_name] = self._actor_manager.add_from_usd_file(
                name=f"{cup_name}_cup",
                asset_path=CUP_ASSET_PATHS[cup_name],
                pose=base_pose,
                density=1e3,
            )

    def _reset_actors(self):
        # reset 时三个杯子分别绕各自基准位置做 xy +/-3cm 随机; z 仍保持桌面高度。
        reset_noise = CUP_RESET_XY_NOISE if is_random else (0.0, 0.0, 0.0)
        self.cup_poses, cup_offsets = self._sample_cup_reset_poses(reset_noise)
        self.cup_xy_noises = {
            cup_name: cup_offset.p.tolist() for cup_name, cup_offset in cup_offsets.items()
        }
        for cup_name, cup_pose in self.cup_poses.items():
            self.cups[cup_name].set_pose(cup_pose)

        self.target_cup = self.cups[self.target_cup_name]
        self.reference_cup = self.cups[self.reference_cup_name]
        self.metadata["is_random"] = bool(is_random)
        self.metadata["target_cup"] = self.target_cup_name
        self.metadata["reference_cup"] = self.reference_cup_name
        self.metadata["placement_side"] = self.placement_side
        self.metadata["cup_xy_noises"] = self.cup_xy_noises
        self.metadata["cup_poses"] = {
            cup_name: cup_pose.tolist() for cup_name, cup_pose in self.cup_poses.items()
        }

    def _sample_cup_reset_poses(self, reset_noise):
        for _ in range(MAX_RESET_SAMPLE_ATTEMPTS):
            cup_offsets = {
                cup_name: self.create_noise(list(reset_noise))
                for cup_name in CUP_BASE_POSES
            }
            cup_poses = {
                cup_name: CUP_BASE_POSES[cup_name].add_offset(cup_offset)
                for cup_name, cup_offset in cup_offsets.items()
            }
            cup_names = tuple(cup_poses)
            cups_are_separated = all(
                np.linalg.norm(cup_poses[cup_names[i]].p[:2] - cup_poses[cup_names[j]].p[:2])
                >= CUP_MIN_RESET_XY_DISTANCE
                for i in range(len(cup_names))
                for j in range(i + 1, len(cup_names))
            )
            if cups_are_separated:
                return cup_poses, cup_offsets
        raise RuntimeError(
            f"Could not sample non-overlapping cup poses after {MAX_RESET_SAMPLE_ATTEMPTS} attempts"
        )

    def pre_move(self):
        # 等待任务初始姿态稳定。
        self.delay(10)
        self.metadata["gripper_open_qpos"] = float(self._robot_manager.gripper_max_qpos)
        self.metadata["gripper_open_ratio"] = 1.0
        self.metadata["gripper_qpos_after_init_delay"] = float(self._robot_manager.get_gripper_qpos())

    def _random_vec(self, scale: float, size: int = 3):
        if not is_random:
            return np.zeros(size)
        return self.rng.uniform(-scale, scale, size=size)

    def _random_scalar(self, scale: float):
        if not is_random:
            return 0.0
        return float(self.rng.uniform(-scale, scale))

    def _placement_y_direction(self):
        return 1.0 if self.placement_side == "left" else -1.0

    def _play_once(self):
        target_cup_pose = self.target_cup.get_pose()
        current_eef_pose = self._robot_manager.get_ee_pose()

        # 竖直向下抓杯口中心: x/y 不偏; z 为杯高 92.5mm - 卷边 11mm 的一半, 再下降 1cm。
        grasp_pos = target_cup_pose.p + np.array([0.0, 0.0, 0.0925 - 0.5 * 0.011 - 0.010])
        # 竖直抓取时只绕 z 轴随机 yaw +/-10deg, 改变两指朝向但不改变向下抓取方向。
        grasp_yaw_noise = self._random_scalar(np.deg2rad(10.0))
        grasp_q = Pose(current_eef_pose.p, current_eef_pose.q).add_rotation([0.0, 0.0, grasp_yaw_noise]).q

        # 先移动到目标杯抓取点上方 5cm, 再向下到抓取点。
        pre_grasp_gripper_center_pose = Pose(grasp_pos + np.array([0.0, 0.0, 0.050]), grasp_q)
        pre_grasp_ee_pose = self._robot_manager.gripper_center_to_ee(pre_grasp_gripper_center_pose)
        self.metadata["grasp_yaw_noise_rad"] = float(grasp_yaw_noise)
        self.metadata["grasp_yaw_noise_deg"] = float(np.rad2deg(grasp_yaw_noise))
        self.metadata["pre_grasp_gripper_center_pose"] = pre_grasp_gripper_center_pose.tolist()
        self.metadata["pre_grasp_ee_pose"] = pre_grasp_ee_pose.tolist()
        self.move(
            self.atom.move_to_pose(pre_grasp_ee_pose),
            tag=f"move_above_{self.target_cup_name}_cup",
            time_dilation_factor=0.5,
        )

        # 向下抓取目标在 xyz 上随机 +/-0.5cm。
        grasp_noise = self._random_vec(0.005)
        grasp_pose = Pose(grasp_pos + grasp_noise, grasp_q)
        self.metadata["grasp_noise"] = grasp_noise.tolist()
        self.metadata["target_cup_grasp_pose"] = grasp_pose.tolist()
        cid = self.target_cup.register_point(grasp_pose, type="contact")
        self.move(self.atom.grasp_actor(
            self.target_cup,
            contact_point_id=cid,
            pre_dis=0.0,
            dis=0.0,
            is_close=False,
        ), tag=f"approach_{self.target_cup_name}_cup", time_dilation_factor=0.5)
        self.move(self.atom.close_gripper(), tag=f"close_{self.target_cup_name}_cup")

        target_cup_pose = self.target_cup.get_pose()

        # 抓住后先上提 12cm。
        lifted_target_cup_pose = Pose(
            target_cup_pose.p + np.array([0.0, 0.0, 0.120]),
            target_cup_pose.q,
        )
        self.metadata["lifted_target_cup_pose"] = lifted_target_cup_pose.tolist()
        self.move(self.atom.place_actor(
            self.target_cup,
            target_pose=lifted_target_cup_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False,
            constrain="free",
        ), tag=f"lift_{self.target_cup_name}_cup", time_dilation_factor=0.5)

        # 移到参考杯指定侧 12cm 的预放置点；其世界 z 固定为 12cm。
        reference_cup_pose = self.reference_cup.get_pose()
        pre_place_noise = self._random_vec(PRE_PLACE_XYZ_NOISE)
        pre_place_target_cup_pose = Pose(
            np.array([
                reference_cup_pose.p[0],
                reference_cup_pose.p[1] + self._placement_y_direction() * PRE_PLACE_Y_DISTANCE,
                PRE_PLACE_WORLD_Z,
            ]) + pre_place_noise,
            self.target_cup.get_pose().q,
        )
        self.metadata["pre_place_noise"] = pre_place_noise.tolist()
        self.metadata["reference_cup_pose_for_pre_place"] = reference_cup_pose.tolist()
        self.metadata["pre_place_target_cup_pose"] = pre_place_target_cup_pose.tolist()
        self.move(self.atom.place_actor(
            self.target_cup,
            target_pose=pre_place_target_cup_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False,
            constrain="free",
        ), tag=f"move_{self.target_cup_name}_cup_to_pre_place", time_dilation_factor=0.5)

        # 最后向下放到参考杯所在的桌面高度, 然后打开夹爪。
        pre_place_target_cup_pose = self.target_cup.get_pose()
        reference_cup_pose = self.reference_cup.get_pose()
        placed_target_cup_pose = Pose(
            np.array([
                pre_place_target_cup_pose.p[0],
                pre_place_target_cup_pose.p[1],
                reference_cup_pose.p[2],
            ]),
            pre_place_target_cup_pose.q,
        )
        self.metadata["reference_cup_pose_for_place_height"] = reference_cup_pose.tolist()
        self.metadata["placed_target_cup_pose"] = placed_target_cup_pose.tolist()
        self.move(self.atom.place_actor(
            self.target_cup,
            target_pose=placed_target_cup_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False,
            constrain="free",
        ), tag=f"place_{self.target_cup_name}_cup_down", time_dilation_factor=0.5)
        self.move(self.atom.open_gripper(1.0), tag=f"release_{self.target_cup_name}_cup")
        # 松爪后不保存, 等杯子物理姿态稍微稳定一下再做 success 检查。
        self.delay(5, is_save=False)

    def check_success(self):
        target_pose = self.target_cup.get_pose()
        reference_pose = self.reference_cup.get_pose()
        gripper_qpos = float(self._robot_manager.get_gripper_qpos())
        gripper_open_threshold = float(
            self._robot_manager.gripper_max_qpos * SUCCESS_GRIPPER_OPEN_RATIO
        )
        self.metadata["final_target_cup_pose"] = target_pose.tolist()
        self.metadata["final_reference_cup_pose"] = reference_pose.tolist()

        target_table_height_error = float(abs(target_pose.p[2] - reference_pose.p[2]))
        target_on_table_ok = target_table_height_error <= SUCCESS_TABLE_HEIGHT_TOLERANCE

        target_up_dir = target_pose.to_transformation_matrix()[:3, :3] @ np.array([0.0, 0.0, 1.0])
        target_upright_score = float(np.dot(target_up_dir, np.array([0.0, 0.0, 1.0])))
        target_upright_ok = target_upright_score > SUCCESS_UPRIGHT_SCORE_THRESHOLD

        target_y_gap = float(target_pose.p[1] - reference_pose.p[1])
        signed_y_gap = self._placement_y_direction() * target_y_gap
        target_on_selected_side_ok = signed_y_gap > 0.0
        target_x_error = float(abs(target_pose.p[0] - reference_pose.p[0]))
        target_x_ok = target_x_error <= SUCCESS_MAX_X_ERROR
        gripper_open_ok = gripper_qpos >= gripper_open_threshold

        self.metadata["success_target_up_dir"] = target_up_dir.tolist()
        self.metadata["success_target_upright_score"] = target_upright_score
        self.metadata["success_checks"] = {
            "target_cup": self.target_cup_name,
            "reference_cup": self.reference_cup_name,
            "placement_side": self.placement_side,
            "target_on_table_ok": bool(target_on_table_ok),
            "target_table_height_error": target_table_height_error,
            "table_height_tolerance": float(SUCCESS_TABLE_HEIGHT_TOLERANCE),
            "target_upright_ok": bool(target_upright_ok),
            "upright_score_threshold": float(SUCCESS_UPRIGHT_SCORE_THRESHOLD),
            "target_on_selected_side_ok": bool(target_on_selected_side_ok),
            "target_y": float(target_pose.p[1]),
            "reference_y": float(reference_pose.p[1]),
            "target_y_gap": target_y_gap,
            "signed_y_gap": signed_y_gap,
            "target_x_ok": bool(target_x_ok),
            "target_x_error": target_x_error,
            "max_x_error": float(SUCCESS_MAX_X_ERROR),
            "gripper_open_ok": bool(gripper_open_ok),
            "gripper_qpos": gripper_qpos,
            "gripper_open_threshold": gripper_open_threshold,
        }
        return bool(
            target_on_table_ok
            and target_upright_ok
            and target_on_selected_side_ok
            and target_x_ok
            and gripper_open_ok
        )
