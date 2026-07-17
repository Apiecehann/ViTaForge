from ._base_task import *
import numpy as np
import transforms3d as t3d

# python scripts/collect_data.py pour_ball_to_cup pour_ball_to_cup_gui --episode_num 1 --start_seed 0 --max_seed 0 --gpu 0


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.9, 0.0, 0.2), rot=(0.555057, 0.465748, 0.443006, 0.527954), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.6, focus_distance=1.0, horizontal_aperture=2.4, clipping_range=(0.1, 100.0)
            ),
            width=480,
            height=270,
            update_period=1/120,
        ),
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
        #     update_period=1/120,
        # ),
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
    step_lim = 2000
    # # 抓蓝杯时, cuRobo 不把被抓物和杯内球当避障物; 黄杯和桌面仍参与避障。
    # planner_ignore_actors = ["yellow_cup", "red_ball", "blue_cup"]
    pass


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        # 塑料杯和球先给较高摩擦, 减少抓取和倒球时的滑动。
        cfg.sim.physics_material.dynamic_friction = 3.0
        cfg.sim.physics_material.static_friction = 3.0
        cfg.uipc_sim.contact.default_friction_ratio = 3.0
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)

        # 这组是当前手调后的侧向平抬初始姿态, 用来避开开局的大幅姿态规划。
        # 这个覆盖只影响 pour_ball_to_cup, 不改 envs/robot/robot_cfg.py 的全局默认姿态。
        tilted_home_joint_pos = {
            # "panda_joint1": 1.229010820,
            # "panda_joint2": 0.878662646,
            # "panda_joint3": -1.072338104,
            # "panda_joint4": -2.460769415,
            # "panda_joint5": -2.891249657,
            # "panda_joint6": 1.975600839,
            # "panda_joint7": -1.683924437,
            "panda_joint1": 1.458306313,
            "panda_joint2": 0.806513369,
            "panda_joint3": -1.137000442,
            "panda_joint4": -2.767615318,
            "panda_joint5": -2.809691191,
            "panda_joint6": 1.849624872,
            "panda_joint7": -1.680763006,
            # init_state 里写的是真实 qpos, 不是 open_gripper() 的比例; 0.039m 对应 open ratio 1.0。
            "panda_finger.*": cfg.robot.gripper_max_qpos,
        }
        self._use_tilted_home_joint_pos = bool(tilted_home_joint_pos)
        if self._use_tilted_home_joint_pos:
            cfg.robot.robot.init_state.joint_pos.update(tilted_home_joint_pos)
        return cfg

    def create_actors(self):
        self.blue_cup = self._actor_manager.add_from_usd_file(
            name="blue_cup",
            asset_path="cup_blue.usd",
            # 蓝色杯子创建时放在 x=50cm, y=10cm。
            pose=Pose([0.50, 0.10, 0.020], (1.0, 0.0, 0.0, 0.0)),
            density=1e3,
        )
        self.yellow_cup = self._actor_manager.add_from_usd_file(
            name="yellow_cup",
            asset_path="cup_yellow.usd",
            # 黄色杯子创建时放在 x=50cm, y=-10cm 作为接球杯。
            pose=Pose([0.50, -0.10, 0.020], (1.0, 0.0, 0.0, 0.0)),
            density=1e5,
        )
        self.red_ball = self._actor_manager.add_from_usd_file(
            name="red_ball",
            asset_path="ball_red.usd",
            # 红球跟蓝杯同 x/y; 球心高度 0.038m = 杯底 z 0.020m + 杯底厚约 0.002m + 预留 0.001m + 半径 0.015m。
            pose=Pose([0.50, 0.10, 0.038], (1.0, 0.0, 0.0, 0.0)),
            density=200,
        )

    def _reset_actors(self):
        # reset 时固定摆位, 不加随机噪声; 当前实际运行会重置到 x=50cm, y=+/-10cm。
        blue_cup_pose = Pose([0.50, 0.10, 0.020], (1.0, 0.0, 0.0, 0.0))
        yellow_cup_pose = Pose([0.50, -0.10, 0.020], (1.0, 0.0, 0.0, 0.0))
        red_ball_pose = Pose([0.50, 0.10, 0.038], (1.0, 0.0, 0.0, 0.0))
        self.blue_cup.set_pose(blue_cup_pose)
        self.yellow_cup.set_pose(yellow_cup_pose)
        self.red_ball.set_pose(red_ball_pose)
        self.metadata["blue_cup_pose"] = blue_cup_pose.tolist()
        self.metadata["yellow_cup_pose"] = yellow_cup_pose.tolist()
        self.metadata["red_ball_pose"] = red_ball_pose.tolist()
        self.metadata["ball_bottom_z"] = 0.023

    def pre_move(self):
        # 先等待 10 step 看初始 settling。夹爪开口已经在 init_state 里设成最大值, 不再额外规划 open_gripper。
        self.delay(10)
        # open_gripper(1.0) 对应的真实 qpos 是 gripper_max_qpos: 单侧约 3.9cm, 总开口约 7.8cm。
        self.metadata["gripper_open_qpos"] = float(self._robot_manager.gripper_max_qpos)
        self.metadata["gripper_open_ratio"] = 1.0
        self.metadata["gripper_qpos_after_init_delay"] = float(self._robot_manager.get_gripper_qpos())

    def _rotate_last_arm_joint(self, delta: float, steps: int = 160, tag: str = "rotate_last_joint", is_save: bool = True):
        # 不走 cuRobo IK, 只做 panda_joint7 的关节空间插值; 旋转角度由 delta 指定。
        self.atom_id += 1
        self.atom_tag = tag
        arm_ids = self._robot_manager._arm_ids
        start_qpos = self._robot_manager.robot.data.joint_pos[0, arm_ids].clone()
        target_qpos = start_qpos.clone()
        target_qpos[-1] += delta

        self.metadata[f"{tag}_start_arm_qpos"] = start_qpos.detach().cpu().tolist()
        self.metadata[f"{tag}_target_arm_qpos"] = target_qpos.detach().cpu().tolist()
        self.metadata[f"{tag}_delta_rad"] = float(delta)
        self.metadata[f"{tag}_delta_deg"] = float(np.rad2deg(delta))

        last_qpos = start_qpos
        for i in range(1, steps + 1):
            alpha = i / steps
            qpos = start_qpos + (target_qpos - start_qpos) * alpha
            vel = (qpos - last_qpos) / self.cfg.sim.dt
            self._robot_manager.set_arm(qpos, vel)
            self._step(is_save)
            last_qpos = qpos
        self._update_render()
        return True

    def _rotate_last_arm_joint_with_ee_translation(
        self,
        delta: float,
        translation: np.ndarray,
        steps: int = 160,
        tag: str = "rotate_last_joint_with_translation",
        is_save: bool = True,
        time_dilation_factor: float = 0.5,
    ):
        # 同一个动作里做两件事:
        # 1. 前 6 个关节沿 cuRobo 规划出的 EEF 平移轨迹走;
        # 2. 第 7 个关节不用 cuRobo 结果, 手动插值旋转 delta 来倒杯。
        translation = np.array(translation, dtype=float)
        current_ee_pose = self._robot_manager.get_ee_pose()
        target_ee_pose = Pose(current_ee_pose.p + translation, current_ee_pose.q)
        self.metadata[f"{tag}_target_ee_pose"] = target_ee_pose.tolist()
        self.metadata[f"{tag}_translation"] = translation.tolist()
        self.metadata[f"{tag}_joint7_delta_rad"] = float(delta)
        self.metadata[f"{tag}_joint7_delta_deg"] = float(np.rad2deg(delta))
        if np.linalg.norm(translation) < 1e-6:
            self.metadata[f"{tag}_translation_plan_status"] = "SkippedZeroTranslation"
            return self._rotate_last_arm_joint(delta, steps=steps, tag=f"{tag}_joint7_only", is_save=is_save)

        arm_plan = self._robot_manager.plan_arm(
            target_ee_pose,
            time_dilation_factor=time_dilation_factor,
        )
        if arm_plan["status"] == "Fail":
            self.metadata[f"{tag}_translation_plan_status"] = "Fail"
            self._rotate_last_arm_joint(delta, steps=steps, tag=f"{tag}_joint7_only_fallback", is_save=is_save)
            return False

        self.atom_id += 1
        self.atom_tag = tag
        arm_ids = self._robot_manager._arm_ids
        start_qpos = self._robot_manager.robot.data.joint_pos[0, arm_ids].clone()
        planned_positions = arm_plan["position"]
        plan_steps = int(planned_positions.shape[0])
        total_steps = max(steps, plan_steps)

        self.metadata[f"{tag}_translation_plan_status"] = "Success"
        self.metadata[f"{tag}_translation_plan_steps"] = plan_steps
        self.metadata[f"{tag}_start_arm_qpos"] = start_qpos.detach().cpu().tolist()
        target_qpos = start_qpos.clone()
        target_qpos[:6] = planned_positions[-1, :6]
        target_qpos[-1] = start_qpos[-1] + delta
        self.metadata[f"{tag}_target_arm_qpos"] = target_qpos.detach().cpu().tolist()

        last_qpos = start_qpos
        for i in range(1, total_steps + 1):
            alpha = i / total_steps
            plan_idx = min(int(round(alpha * (plan_steps - 1))), plan_steps - 1)
            qpos = start_qpos.clone()
            qpos[:6] = planned_positions[plan_idx, :6]
            qpos[-1] = start_qpos[-1] + delta * alpha
            vel = (qpos - last_qpos) / self.cfg.sim.dt
            self._robot_manager.set_arm(qpos, vel)
            self._step(is_save)
            last_qpos = qpos

        self._update_render()
        return True

    def _wait_ball_until_still(
        self,
        max_steps: int = 600,
        pos_tol: float = 1e-4,
        quat_tol: float = 1e-3,
        tag: str = "wait_ball_still",
    ):
        # 倒杯后每次 delay(1), 比较 delay 前后的红球 pose; 不再变化就结束。
        # max_steps 是兜底, 防止球持续微小抖动导致无限循环。
        prev_pose = self.red_ball.get_pose()
        self.metadata[f"{tag}_start_pose"] = prev_pose.tolist()

        for step in range(1, max_steps + 1):
            self.delay(1, is_save=True)
            curr_pose = self.red_ball.get_pose()
            pos_delta = float(np.linalg.norm(curr_pose.p - prev_pose.p))
            quat_delta = float(min(
                np.linalg.norm(curr_pose.q - prev_pose.q),
                np.linalg.norm(curr_pose.q + prev_pose.q),
            ))

            if pos_delta <= pos_tol and quat_delta <= quat_tol:
                self.metadata[f"{tag}_steps"] = step
                self.metadata[f"{tag}_end_pose"] = curr_pose.tolist()
                self.metadata[f"{tag}_last_pos_delta"] = pos_delta
                self.metadata[f"{tag}_last_quat_delta"] = quat_delta
                self.metadata[f"{tag}_stopped_by"] = "still"
                return True

            prev_pose = curr_pose

        self.metadata[f"{tag}_steps"] = max_steps
        self.metadata[f"{tag}_end_pose"] = prev_pose.tolist()
        self.metadata[f"{tag}_stopped_by"] = "max_steps"
        return False

    def _is_ball_in_yellow_cup(self):
        # 黄杯 pose 是杯底中心; 红球转到黄杯局部坐标后,
        # x/y 在杯口半径 +/-3.4cm 内, z 在 0~9.6cm 内就认为球在黄杯中。
        ball_pose = self.red_ball.get_pose()
        yellow_cup_pose = self.yellow_cup.get_pose()
        ball_local_pose = ball_pose.rebase(to_coord=yellow_cup_pose)
        ball_local_p = ball_local_pose.p

        cup_radius = 0.068 * 0.5
        cup_height = 0.096
        in_x = -cup_radius <= ball_local_p[0] <= cup_radius
        in_y = -cup_radius <= ball_local_p[1] <= cup_radius
        in_z = 0.0 <= ball_local_p[2] <= cup_height
        success = bool(in_x and in_y and in_z)

        self.metadata["success_ball_world_pose"] = ball_pose.tolist()
        self.metadata["success_yellow_cup_pose"] = yellow_cup_pose.tolist()
        self.metadata["success_ball_pose_in_yellow_cup"] = ball_local_pose.tolist()
        self.metadata["success_cup_radius"] = cup_radius
        self.metadata["success_cup_height"] = cup_height
        self.metadata["success_checks"] = {
            "in_x": bool(in_x),
            "in_y": bool(in_y),
            "in_z": bool(in_z),
        }
        return success

    def _play_once(self):
        # 黄色杯子是接球杯, 这里先抓蓝色起始杯; 如果要抓 yellow, 把 self.blue_cup 换成 self.yellow_cup。
        cup_pose = self.blue_cup.get_pose()
        # 抓卷边中线高度: 杯高 92.5mm - 卷边 11mm 的一半 = 杯底上方 87mm。
        # gripper center/gel pad 不再额外下调; 横向抓杯时让目标点沿 +X 前伸 1cm, 让垫片吃住更多杯沿。
        grasp_pos = cup_pose.p + np.array([0.01, 0.0, 0.0925 - 0.5 * 0.011])

        # 初始 joint 已经是 gripper_flat_lift_debug 里平抬后的姿态。
        # 这里不再额外旋转夹爪, 直接沿用当前 EEF 姿态做后续平移和下探。
        current_eef_pose = self._robot_manager.get_ee_pose()
        tilted_eef_q = current_eef_pose.q
        self.metadata["initial_grasp_eef_pose"] = current_eef_pose.tolist()
        self.metadata["use_tilted_home_joint_pos"] = bool(getattr(self, "_use_tilted_home_joint_pos", False))

        # 保持当前侧向姿态, 平移到蓝杯抓取点上方 5cm。
        # 这里先写的是 gripper center 目标, move_to_pose 需要 EEF 目标, 所以下面要转换一次。
        pre_grasp_gripper_center_pose = Pose(grasp_pos + np.array([0.0, 0.0, 0.050]), tilted_eef_q)
        pre_grasp_ee_pose = self._robot_manager.gripper_center_to_ee(pre_grasp_gripper_center_pose)
        self.metadata["pre_grasp_gripper_center_pose"] = pre_grasp_gripper_center_pose.tolist()
        self.metadata["pre_grasp_ee_pose"] = pre_grasp_ee_pose.tolist()
        self.move(
            self.atom.move_to_pose(pre_grasp_ee_pose),
            tag="move_above_blue_cup_rim",
            time_dilation_factor=0.5,
        )

        # 再从预抓取位直接平移到最终抓取位, 姿态保持当前侧向姿态。
        grasp_pose = Pose(grasp_pos, tilted_eef_q)

        self.metadata["rim_grasp_world_z"] = float(grasp_pos[2])
        self.metadata["rim_grasp_pose"] = grasp_pose.tolist()
        self.metadata["target_gripper_center_pose"] = grasp_pose.tolist()
        self.metadata["target_gripper_minus_z"] = (
            grasp_pose.to_transformation_matrix()[:3, :3] @ np.array([0.0, 0.0, -1.0])
        ).tolist()

        cid = self.blue_cup.register_point(grasp_pose, type="contact")
        self.move(self.atom.grasp_actor(
            self.blue_cup,
            contact_point_id=cid,
            pre_dis=0.0,
            dis=0.0,
            is_close=False,
        ), tag="approach_blue_cup_rim", time_dilation_factor=0.5)

        # 模仿 insert_USB.py: 不手写闭合宽度, close_gripper() 交给 adaptive/tactile 接触逻辑停止。
        self.move(self.atom.close_gripper(), tag="close_blue_cup_rim")

        yellow_cup_pose = self.yellow_cup.get_pose()
        blue_cup_pose = self.blue_cup.get_pose()
        target_blue_cup_pos = yellow_cup_pose.p + np.array([0.0, 0.030, 0.120])
        # 搬运拆成两步: 先只抬高蓝杯到目标高度, 再保持高度沿 y 方向移到黄杯旁边。
        # 这样每次 IK 只处理单纯位移, 比一次性斜向搬运更容易规划。
        lift_blue_cup_pose = Pose(
            np.array([blue_cup_pose.p[0], blue_cup_pose.p[1], target_blue_cup_pos[2]]),
            blue_cup_pose.q,
        )
        target_blue_cup_pose = Pose(target_blue_cup_pos, blue_cup_pose.q)
        self.metadata["yellow_cup_pose_before_pour"] = yellow_cup_pose.tolist()
        self.metadata["blue_cup_pose_before_pour"] = blue_cup_pose.tolist()
        self.metadata["lift_blue_cup_pose"] = lift_blue_cup_pose.tolist()
        self.metadata["target_blue_cup_carry_pose"] = target_blue_cup_pose.tolist()
        self.move(self.atom.place_actor(
            self.blue_cup,
            target_pose=lift_blue_cup_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False,
            constrain="free",
        ), tag="lift_blue_cup_before_pour", time_dilation_factor=0.5)
        self.move(self.atom.place_actor(
            self.blue_cup,
            target_pose=target_blue_cup_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False,
            constrain="free",
        ), tag="move_blue_cup_y_near_yellow", time_dilation_factor=0.5)

        # 第二步不再给蓝杯目标 pose 做反解。当前 translation 为 0, 会跳过 EEF 平移规划,
        # 只旋转最后一个关节 panda_joint7 来倒杯。
        self.metadata["blue_cup_pose_before_wrist_pour"] = self.blue_cup.get_pose().tolist()
        self._rotate_last_arm_joint_with_ee_translation(
            delta=np.deg2rad(120.0),
            translation=np.array([0.0, 0.0, 0.00]),
            steps=160,
            tag="translate_yz_and_rotate_panda_joint7_to_pour",
            is_save=True,
        )
        self._wait_ball_until_still(tag="wait_red_ball_still_after_pour")
        self.metadata["blue_cup_pose_after_wrist_pour"] = self.blue_cup.get_pose().tolist()
        self.metadata["red_ball_pose_after_wait"] = self.red_ball.get_pose().tolist()


    def check_success(self):
        return self._is_ball_in_yellow_cup()
