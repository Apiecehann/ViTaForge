from ._base_task import *
import numpy as np

# python scripts/collect_data.py swap_cup_order swap_cup_order_gui --episode_num 1 --start_seed 0 --max_seed 0 --gpu 0

# True 时给初始物体位置和若干 motion target 加小扰动; False 时所有随机量为 0。
is_random = True


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.9, 0.0, 0.2),
                rot=(0.555057, 0.465748, 0.443006, 0.527954),
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
    step_lim = 350


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        # 两个杯子都用较高摩擦, 减少竖直抓取和搬运时的滑动。
        cfg.sim.physics_material.dynamic_friction = 3.0
        cfg.sim.physics_material.static_friction = 3.0
        cfg.uipc_sim.contact.default_friction_ratio = 3.0
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        # arm joint 保持 robot_cfg 默认; 这里只把夹爪初始开口设为最大值, 对应 open_gripper(1.0)。
        cfg.robot.robot.init_state.joint_pos.update({
            "panda_finger.*": cfg.robot.gripper_max_qpos,
        })
        return cfg

    def create_actors(self):
        self.yellow_cup = self._actor_manager.add_from_usd_file(
            name="yellow_cup",
            asset_path="cup_yellow.usd",
            pose=Pose([0.50, 0.00, 0.020], (1.0, 0.0, 0.0, 0.0)),
            density=1e3,
        )
        self.blue_cup = self._actor_manager.add_from_usd_file(
            name="blue_cup",
            asset_path="cup_blue.usd",
            pose=Pose([0.50, -0.12, 0.020], (1.0, 0.0, 0.0, 0.0)),
            density=1e3,
        )

    def _reset_actors(self):
        # reset 时两个杯子分别绕各自基准位置做 xy +/-1cm 随机; z 仍保持桌面高度。
        yellow_cup_xy_noise = self._random_vec(0.010, size=2)
        blue_cup_xy_noise = self._random_vec(0.010, size=2)
        yellow_cup_pose = Pose([0.50 + yellow_cup_xy_noise[0], 0.00 + yellow_cup_xy_noise[1], 0.020], (1.0, 0.0, 0.0, 0.0))
        blue_cup_pose = Pose([0.50 + blue_cup_xy_noise[0], -0.12 + blue_cup_xy_noise[1], 0.020], (1.0, 0.0, 0.0, 0.0))
        self.yellow_cup.set_pose(yellow_cup_pose)
        self.blue_cup.set_pose(blue_cup_pose)
        self.metadata["is_random"] = bool(is_random)
        self.metadata["yellow_cup_xy_noise"] = yellow_cup_xy_noise.tolist()
        self.metadata["blue_cup_xy_noise"] = blue_cup_xy_noise.tolist()
        self.metadata["yellow_cup_pose"] = yellow_cup_pose.tolist()
        self.metadata["blue_cup_pose"] = blue_cup_pose.tolist()

    def pre_move(self):
        # 默认 robot_cfg 给竖直向下的手臂姿态; 夹爪已在 init_state 里设成最大开口。
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

    def _play_once(self):
        cup_pose = self.blue_cup.get_pose()
        current_eef_pose = self._robot_manager.get_ee_pose()

        # 竖直向下抓杯口中心: x/y 不偏; z 为杯高 92.5mm - 卷边 11mm 的一半, 再下降 1cm。
        grasp_pos = cup_pose.p + np.array([0.0, 0.0, 0.0925 - 0.5 * 0.011 - 0.010])
        # 竖直抓取时只绕 z 轴随机 yaw +/-10deg, 改变两指朝向但不改变向下抓取方向。
        grasp_yaw_noise = self._random_scalar(np.deg2rad(10.0))
        grasp_q = Pose(current_eef_pose.p, current_eef_pose.q).add_rotation([0.0, 0.0, grasp_yaw_noise]).q

        # 先移动到蓝杯抓取点上方 5cm, 再向下到抓取点。
        pre_grasp_gripper_center_pose = Pose(grasp_pos + np.array([0.0, 0.0, 0.050]), grasp_q)
        pre_grasp_ee_pose = self._robot_manager.gripper_center_to_ee(pre_grasp_gripper_center_pose)
        self.metadata["grasp_yaw_noise_rad"] = float(grasp_yaw_noise)
        self.metadata["grasp_yaw_noise_deg"] = float(np.rad2deg(grasp_yaw_noise))
        self.metadata["pre_grasp_gripper_center_pose"] = pre_grasp_gripper_center_pose.tolist()
        self.metadata["pre_grasp_ee_pose"] = pre_grasp_ee_pose.tolist()
        self.move(
            self.atom.move_to_pose(pre_grasp_ee_pose),
            tag="move_above_blue_cup",
            time_dilation_factor=0.5,
        )

        # 向下抓取目标在 xyz 上随机 +/-0.5cm。
        grasp_noise = self._random_vec(0.005)
        grasp_pose = Pose(grasp_pos + grasp_noise, grasp_q)
        self.metadata["grasp_noise"] = grasp_noise.tolist()
        self.metadata["blue_cup_grasp_pose"] = grasp_pose.tolist()
        cid = self.blue_cup.register_point(grasp_pose, type="contact")
        self.move(self.atom.grasp_actor(
            self.blue_cup,
            contact_point_id=cid,
            pre_dis=0.0,
            dis=0.0,
            is_close=False,
        ), tag="approach_blue_cup", time_dilation_factor=0.5)
        self.move(self.atom.close_gripper(), tag="close_blue_cup")

        blue_cup_pose = self.blue_cup.get_pose()

        # 抓住后先上提 12cm; 抬起目标 xyz 随机 +/-1cm。
        lift_noise = self._random_vec(0.010)
        lifted_blue_cup_pose = Pose(
            blue_cup_pose.p + np.array([0.0, 0.0, 0.120]) + lift_noise,
            blue_cup_pose.q,
        )
        self.metadata["lift_noise"] = lift_noise.tolist()
        self.metadata["lifted_blue_cup_pose"] = lifted_blue_cup_pose.tolist()
        self.move(self.atom.place_actor(
            self.blue_cup,
            target_pose=lifted_blue_cup_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False,
            constrain="free",
        ), tag="lift_blue_cup", time_dilation_factor=0.5)

        # 保持高度, 沿 +Y 方向移动到 y=+12cm 的排序位置; 目标 xyz 随机 +/-1cm。
        lifted_blue_cup_pose = self.blue_cup.get_pose()
        sorted_noise = self._random_vec(0.010)
        y_sorted_blue_cup_pos = np.array([lifted_blue_cup_pose.p[0], 0.12, lifted_blue_cup_pose.p[2]]) + sorted_noise
        y_sorted_blue_cup_pose = Pose(
            y_sorted_blue_cup_pos,
            lifted_blue_cup_pose.q,
        )
        self.metadata["sorted_noise"] = sorted_noise.tolist()
        self.metadata["y_sorted_blue_cup_pose"] = y_sorted_blue_cup_pose.tolist()
        self.move(self.atom.place_actor(
            self.blue_cup,
            target_pose=y_sorted_blue_cup_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False,
            constrain="free",
        ), tag="move_blue_cup_to_sorted_y", time_dilation_factor=0.5)

        # 最后向下放到和黄色杯子一样的 z 高度, 然后打开夹爪。
        y_sorted_blue_cup_pose = self.blue_cup.get_pose()
        yellow_cup_pose_for_place = self.yellow_cup.get_pose()
        place_xy_noise = self._random_vec(0.010, size=2)
        placed_blue_cup_pose = Pose(
            np.array([
                y_sorted_blue_cup_pose.p[0] + place_xy_noise[0],
                y_sorted_blue_cup_pose.p[1] + place_xy_noise[1],
                yellow_cup_pose_for_place.p[2],
            ]),
            y_sorted_blue_cup_pose.q,
        )
        self.metadata["yellow_cup_pose_for_place_height"] = yellow_cup_pose_for_place.tolist()
        self.metadata["place_xy_noise"] = place_xy_noise.tolist()
        self.metadata["placed_blue_cup_pose"] = placed_blue_cup_pose.tolist()
        self.move(self.atom.place_actor(
            self.blue_cup,
            target_pose=placed_blue_cup_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False,
            constrain="free",
        ), tag="place_blue_cup_down", time_dilation_factor=0.5)
        self.move(self.atom.open_gripper(1.0), tag="release_blue_cup")
        # 松爪后不保存, 等杯子物理姿态稍微稳定一下再做 success 检查。
        self.delay(5, is_save=False)

    def check_success(self):
        yellow_pose = self.yellow_cup.get_pose()
        blue_pose = self.blue_cup.get_pose()
        self.metadata["final_yellow_cup_pose"] = yellow_pose.tolist()
        self.metadata["final_blue_cup_pose"] = blue_pose.tolist()
        blue_up_dir = blue_pose.to_transformation_matrix()[:3, :3] @ np.array([0.0, 0.0, 1.0])
        blue_upright_score = float(np.dot(blue_up_dir, np.array([0.0, 0.0, 1.0])))
        blue_upright_ok = blue_upright_score > 0.95
        blue_after_yellow_ok = blue_pose.p[1] > yellow_pose.p[1]
        self.metadata["success_blue_up_dir"] = blue_up_dir.tolist()
        self.metadata["success_blue_upright_score"] = blue_upright_score
        self.metadata["success_checks"] = {
            "blue_upright_ok": bool(blue_upright_ok),
            "blue_after_yellow_ok": bool(blue_after_yellow_ok),
            "blue_y": float(blue_pose.p[1]),
            "yellow_y": float(yellow_pose.p[1]),
        }
        return bool(blue_upright_ok and blue_after_yellow_ok)
