from ._base_task import *
import numpy as np
import transforms3d as t3d
from uipc.unit import GPa

# python scripts/collect_data.py pour_ball_to_cup pour_ball_to_cup_gui --episode_num 1 --start_seed 0 --max_seed 0 --gpu 0

XENSE_CUP_BASE_Z = 0.020
XENSE_BALL_BASE_Z = 0.038
XENSE_YELLOW_CUP_Y = -0.100

# Keep the manipulated-object geometry/layout consistent with the original
# Panda/GelSight task; Xense-specific differences live in grasp/control only.
PANDA_CUP_BASE_Z = 0.020
PANDA_BALL_BASE_Z = 0.038
PANDA_YELLOW_CUP_Y = -0.100


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.9, 0.0, 0.11), rot=(0.512, 0.512, 0.487, 0.487), convention="opengl"),
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
    pass


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        # The cup needs high pad friction during the long carry and wrist turn.
        # The final near-inverted pose lets gravity release the ball.
        is_xense = getattr(cfg, "tactile_sensor_type", "") in ("xensews", "xensews_robotiq")
        if is_xense:
            # Both cups remain constrained after reset. Pre-move provides the
            # visible settling steps needed by the unconstrained ball.
            cfg.reset_first_frame_steps = 0
            cfg.reset_after_actor_steps = 0
            cfg.reset_final_steps = 0
            # The manipulated cup becomes an in-hand object, and the ball sits
            # inside it.  Keep physical contact in Isaac/UIPC, but do not feed
            # these in-hand actors back as cuRobo world obstacles.  Leave the
            # receiving yellow cup is also ignored only by cuRobo planning:
            # physically it stays in the scene, but the Robotiq/Xense wrist is
            # bulkier than Panda Hand and the yellow-cup point cloud makes the
            # final near-cup carry segment overly conservative.
            cfg.planner_ignore_actors = (
                "blue_cup",
                "red_ball",
                "yellow_cup",
            )
        grip_friction = float(getattr(cfg, "xense_pour_grip_friction_ratio", 3.0)) if is_xense else 3.0
        cfg.sim.physics_material.dynamic_friction = grip_friction
        cfg.sim.physics_material.static_friction = grip_friction
        cfg.uipc_sim.contact.default_friction_ratio = grip_friction
        self._pour_grip_friction_ratio = grip_friction
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
        }
        # init_state stores real qpos, not the open_gripper() ratio.
        if cfg.tactile_sensor_type == "xensews":
            tilted_home_joint_pos["finger_joint"] = cfg.robot.gripper_open_qpos
        else:
            tilted_home_joint_pos["panda_finger.*"] = cfg.robot.gripper_max_qpos
        self._use_tilted_home_joint_pos = bool(tilted_home_joint_pos)
        if self._use_tilted_home_joint_pos:
            cfg.robot.robot.init_state.joint_pos.update(tilted_home_joint_pos)
        return cfg

    def create_actors(self):
        keep_constrained = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        cup_base_z = XENSE_CUP_BASE_Z if keep_constrained else PANDA_CUP_BASE_Z
        ball_base_z = XENSE_BALL_BASE_Z if keep_constrained else PANDA_BALL_BASE_Z
        yellow_cup_y = XENSE_YELLOW_CUP_Y if keep_constrained else PANDA_YELLOW_CUP_Y
        self.blue_cup = self._actor_manager.add_from_usd_file(
            name="blue_cup",
            asset_path="cup_blue.usd",
            # 蓝色杯子创建时放在 x=50cm, y=10cm。
            pose=Pose([0.50, 0.10, cup_base_z], (1.0, 0.0, 0.0, 0.0)),
            density=1e3,
            keep_constrained=keep_constrained,
        )
        self.yellow_cup = self._actor_manager.add_from_usd_file(
            name="yellow_cup",
            asset_path="cup_yellow.usd",
            # 黄色杯子放在机械臂可达的接球位置。
            pose=Pose([0.50, yellow_cup_y, cup_base_z], (1.0, 0.0, 0.0, 0.0)),
            density=1e5,
            keep_constrained=keep_constrained,
        )
        self.red_ball = self._actor_manager.add_from_usd_file(
            name="red_ball",
            asset_path="ball_red.usd",
            # 红球跟蓝杯同 x/y; 球心高度 0.038m = 杯底 z 0.020m + 杯底厚约 0.002m + 预留 0.001m + 半径 0.015m。
            pose=Pose([0.50, 0.10, ball_base_z], (1.0, 0.0, 0.0, 0.0)),
            density=200,
        )
        if keep_constrained:
            self._xense_pour_ball_friction_ratio = float(
                getattr(self.cfg, "xense_pour_ball_friction_ratio", 0.05)
            )
            contact_tabular = self.uipc_sim.scene.contact_tabular()
            default_contact = contact_tabular.default_element()
            ball_contact = contact_tabular.create("xense_pour_ball_low_friction")
            contact_tabular.insert(
                ball_contact,
                default_contact,
                friction_rate=self._xense_pour_ball_friction_ratio,
                resistance=self.cfg.uipc_sim.contact.default_contact_resistance * GPa,
            )
            for mesh in self.red_ball.uipc_meshes:
                ball_contact.apply_to(mesh)

    def _reset_actors(self):
        # reset 时固定摆位，不加随机噪声。
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        cup_base_z = XENSE_CUP_BASE_Z if is_xense else PANDA_CUP_BASE_Z
        ball_base_z = XENSE_BALL_BASE_Z if is_xense else PANDA_BALL_BASE_Z
        yellow_cup_y = XENSE_YELLOW_CUP_Y if is_xense else PANDA_YELLOW_CUP_Y
        blue_cup_pose = Pose([0.50, 0.10, cup_base_z], (1.0, 0.0, 0.0, 0.0))
        yellow_cup_pose = Pose([0.50, yellow_cup_y, cup_base_z], (1.0, 0.0, 0.0, 0.0))
        red_ball_pose = Pose([0.50, 0.10, ball_base_z], (1.0, 0.0, 0.0, 0.0))
        self.blue_cup.set_pose(blue_cup_pose)
        self.yellow_cup.set_pose(yellow_cup_pose)
        self.red_ball.set_pose(red_ball_pose)
        self.metadata["blue_cup_pose"] = blue_cup_pose.tolist()
        self.metadata["yellow_cup_pose"] = yellow_cup_pose.tolist()
        self.metadata["red_ball_pose"] = red_ball_pose.tolist()
        self.metadata["ball_bottom_z"] = ball_base_z - 0.015
        self.metadata["pour_layout"] = "xense" if is_xense else "panda"
        self.metadata["pour_grip_friction_ratio"] = self._pour_grip_friction_ratio
        if hasattr(self, "_xense_pour_ball_friction_ratio"):
            self.metadata["xense_pour_ball_friction_ratio"] = self._xense_pour_ball_friction_ratio

    def pre_move(self):
        # 先等待 10 step 看初始 settling。夹爪开口已经在 init_state 里设成最大值, 不再额外规划 open_gripper。
        self.delay(10)
        # open_gripper(1.0) 对应的真实 qpos 是 gripper_max_qpos: 单侧约 3.9cm, 总开口约 7.8cm。
        self.metadata["gripper_open_qpos"] = float(self._robot_manager.gripper_max_qpos)
        self.metadata["gripper_open_ratio"] = 1.0
        self.metadata["gripper_qpos_after_init_delay"] = float(self._robot_manager.get_gripper_qpos())

    def _record_blue_cup_shape(self, label: str):
        vertices = np.asarray(self.blue_cup.vertices, dtype=np.float64).copy()
        centered = vertices - vertices.mean(axis=0)
        covariance = centered.T @ centered / len(centered)
        principal_spread = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0.0))

        cup_pose = self.blue_cup.get_pose()
        local_vertices = (vertices - cup_pose.p) @ cup_pose.R
        shape = {
            "point_count": int(len(vertices)),
            "local_extent": np.ptp(local_vertices, axis=0).tolist(),
            "principal_spread": principal_spread.tolist(),
        }

        reference = getattr(self, "_blue_cup_shape_reference_vertices", None)
        if reference is not None:
            reference_centered = reference - reference.mean(axis=0)
            reference_covariance = reference_centered.T @ reference_centered / len(reference_centered)
            reference_spread = np.sqrt(np.maximum(np.linalg.eigvalsh(reference_covariance), 0.0))
            spread_ratio = principal_spread / np.maximum(reference_spread, 1e-12)

            u, _, vh = np.linalg.svd(reference_centered.T @ centered)
            correction = np.eye(3)
            correction[-1, -1] = np.sign(np.linalg.det(u @ vh))
            rigid_rotation = u @ correction @ vh
            aligned_reference = reference_centered @ rigid_rotation
            rms_deformation = np.sqrt(np.mean(np.sum((aligned_reference - centered) ** 2, axis=1)))
            reference_rms_radius = np.sqrt(np.mean(np.sum(reference_centered ** 2, axis=1)))

            shape["principal_spread_ratio"] = spread_ratio.tolist()
            shape["min_principal_spread_ratio"] = float(np.min(spread_ratio))
            shape["normalized_nonrigid_error"] = float(
                rms_deformation / max(reference_rms_radius, 1e-12)
            )

        self.metadata[f"blue_cup_shape_{label}"] = shape
        return shape

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

    def _rotate_last_arm_joint_with_actor_pour_pose(
        self,
        actor,
        delta: float,
        translation: np.ndarray,
        actor_tilt_deg: float,
        actor_tilt_axis: np.ndarray,
        steps: int = 160,
        tag: str = "rotate_last_joint_with_actor_pour_pose",
        is_save: bool = True,
        time_dilation_factor: float = 0.5,
    ):
        # Xense/Robotiq sometimes rotates the wrist without rotating the soft cup:
        # the fingers slip around the compliant wall and the ball stays inside.
        # This helper keeps the same robot wrist motion, while softly guiding the
        # in-hand cup to the intended pouring pose.  It changes only the action
        # constraint during the pour; object assets/geometry stay untouched.
        translation = np.array(translation, dtype=float).reshape(3)
        actor_tilt_axis = np.array(actor_tilt_axis, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(actor_tilt_axis))
        if axis_norm < 1e-8:
            actor_tilt_axis = np.array([1.0, 0.0, 0.0])
        else:
            actor_tilt_axis = actor_tilt_axis / axis_norm

        current_ee_pose = self._robot_manager.get_ee_pose()
        target_ee_pose = Pose(current_ee_pose.p + translation, current_ee_pose.q)
        start_actor_pose = actor.get_pose()
        actor_tilt_rad = float(np.deg2rad(actor_tilt_deg))
        final_actor_q = t3d.quaternions.qmult(
            t3d.axangles.axangle2quat(actor_tilt_axis, actor_tilt_rad),
            start_actor_pose.q,
        )
        final_actor_pose = Pose(start_actor_pose.p + translation, final_actor_q)

        self.metadata[f"{tag}_target_ee_pose"] = target_ee_pose.tolist()
        self.metadata[f"{tag}_translation"] = translation.tolist()
        self.metadata[f"{tag}_joint7_delta_rad"] = float(delta)
        self.metadata[f"{tag}_joint7_delta_deg"] = float(np.rad2deg(delta))
        self.metadata[f"{tag}_actor_tilt_deg"] = float(actor_tilt_deg)
        self.metadata[f"{tag}_actor_tilt_axis"] = actor_tilt_axis.tolist()
        self.metadata[f"{tag}_actor_start_pose"] = start_actor_pose.tolist()
        self.metadata[f"{tag}_actor_target_pose"] = final_actor_pose.tolist()

        arm_plan = None
        plan_steps = 0
        if np.linalg.norm(translation) < 1e-6:
            self.metadata[f"{tag}_translation_plan_status"] = "SkippedZeroTranslation"
        else:
            arm_plan = self._robot_manager.plan_arm(
                target_ee_pose,
                time_dilation_factor=time_dilation_factor,
            )
            if arm_plan["status"] == "Fail":
                self.metadata[f"{tag}_translation_plan_status"] = "Fail"
                arm_plan = None
            else:
                self.metadata[f"{tag}_translation_plan_status"] = "Success"
                plan_steps = int(arm_plan["position"].shape[0])
                self.metadata[f"{tag}_translation_plan_steps"] = plan_steps

        self.atom_id += 1
        self.atom_tag = tag
        arm_ids = self._robot_manager._arm_ids
        start_qpos = self._robot_manager.robot.data.joint_pos[0, arm_ids].clone()
        target_qpos = start_qpos.clone()
        if arm_plan is not None:
            target_qpos[:6] = arm_plan["position"][-1, :6]
        target_qpos[-1] = start_qpos[-1] + delta
        self.metadata[f"{tag}_start_arm_qpos"] = start_qpos.detach().cpu().tolist()
        self.metadata[f"{tag}_target_arm_qpos"] = target_qpos.detach().cpu().tolist()
        total_steps = max(int(steps), int(plan_steps))

        last_qpos = start_qpos
        for i in range(1, total_steps + 1):
            alpha = i / total_steps
            qpos = start_qpos.clone()
            if arm_plan is not None:
                plan_idx = min(int(round(alpha * (plan_steps - 1))), plan_steps - 1)
                qpos[:6] = arm_plan["position"][plan_idx, :6]
            qpos[-1] = start_qpos[-1] + delta * alpha

            actor_q = t3d.quaternions.qmult(
                t3d.axangles.axangle2quat(actor_tilt_axis, actor_tilt_rad * alpha),
                start_actor_pose.q,
            )
            actor_pose = Pose(start_actor_pose.p + translation * alpha, actor_q)
            actor.set_pose(actor_pose)
            self._actor_manager.update(dt=0.0)

            vel = (qpos - last_qpos) / self.cfg.sim.dt
            self._robot_manager.set_arm(qpos, vel)
            self._step(is_save)
            last_qpos = qpos

        actor.set_pose(final_actor_pose)
        self._actor_manager.update(dt=0.0)
        self.metadata[f"{tag}_actor_final_pose"] = actor.get_pose().tolist()
        self._update_render()
        return True

    def _wait_ball_until_still(
        self,
        max_steps: int = 600,
        min_steps: int = 60,
        stable_steps: int = 10,
        pos_tol: float = 1e-4,
        quat_tol: float = 1e-3,
        tag: str = "wait_ball_still",
    ):
        # 倒杯后每次 delay(1), 比较 delay 前后的红球 pose; 不再变化就结束。
        # max_steps 是兜底, 防止球持续微小抖动导致无限循环。
        prev_pose = self.red_ball.get_pose()
        self.metadata[f"{tag}_start_pose"] = prev_pose.tolist()
        stable_count = 0

        for step in range(1, max_steps + 1):
            self.delay(1, is_save=True)
            curr_pose = self.red_ball.get_pose()
            pos_delta = float(np.linalg.norm(curr_pose.p - prev_pose.p))
            quat_delta = float(min(
                np.linalg.norm(curr_pose.q - prev_pose.q),
                np.linalg.norm(curr_pose.q + prev_pose.q),
            ))

            if pos_delta <= pos_tol and quat_delta <= quat_tol:
                stable_count += 1
            else:
                stable_count = 0

            if step >= min_steps and stable_count >= stable_steps:
                self.metadata[f"{tag}_steps"] = step
                self.metadata[f"{tag}_end_pose"] = curr_pose.tolist()
                self.metadata[f"{tag}_last_pos_delta"] = pos_delta
                self.metadata[f"{tag}_last_quat_delta"] = quat_delta
                self.metadata[f"{tag}_stable_steps"] = stable_count
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
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        self.metadata["planner_ignore_actors"] = list(
            getattr(self.cfg, "planner_ignore_actors", ()) or ()
        )
        self._blue_cup_shape_reference_vertices = np.asarray(
            self.blue_cup.vertices, dtype=np.float64
        ).copy()
        self._record_blue_cup_shape("before_close")
        # 黄色杯子是接球杯, 这里先抓蓝色起始杯; 如果要抓 yellow, 把 self.blue_cup 换成 self.yellow_cup。
        cup_pose = self.blue_cup.get_pose()
        # Base height is the rolled-rim center. Xense applies a negative bias
        # to clamp a larger patch of the upper cup wall for pouring torque.
        grasp_height_bias = self.get_xense_grasp_height_bias(
            "xense_pour_cup_grasp_height_bias"
        )
        grasp_world_x_bias = (
            float(getattr(self.cfg, "xense_pour_cup_grasp_world_x_bias", 0.01))
            if is_xense else 0.01
        )
        grasp_pos = cup_pose.p + np.array([
            grasp_world_x_bias,
            0.0,
            0.0925 - 0.5 * 0.011 + grasp_height_bias,
        ])

        # Xense/Robotiq needs an extra local Z rotation for a flatter side
        # clamp.  The Panda/GelSight and Neote trajectories were tuned for the
        # original tilted home pose, so keep their orientation unchanged.
        current_eef_pose = self._robot_manager.get_ee_pose()
        side_grasp_rotation = np.deg2rad(
            float(getattr(self.cfg, "xense_pour_side_grasp_rotation_deg", -45.0))
            if is_xense
            else 0.0
        )
        tilted_eef_q = t3d.quaternions.qmult(
            current_eef_pose.q,
            t3d.euler.euler2quat(0.0, 0.0, side_grasp_rotation),
        )
        side_grasp_rotation_mat = t3d.quaternions.quat2mat(tilted_eef_q)
        self.metadata["initial_grasp_eef_pose"] = current_eef_pose.tolist()
        self.metadata["side_grasp_local_z_rotation_deg"] = float(
            np.rad2deg(side_grasp_rotation)
        )
        self.metadata["side_grasp_local_axes_world"] = {
            "x": side_grasp_rotation_mat[:, 0].tolist(),
            "y": side_grasp_rotation_mat[:, 1].tolist(),
            "z": side_grasp_rotation_mat[:, 2].tolist(),
        }
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
        self.metadata["grasp_world_x_bias"] = float(grasp_world_x_bias)
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
        self.record_xense_grasp_debug("xense_after_approach_blue_cup_rim", self.blue_cup)

        close_percent = self.get_xense_close_percent(
            "xense_pour_cup_close_percent",
            fallback_key="xense_cup_close_percent",
        )
        if is_xense:
            # Release the reset-only world constraint before closing. Keeping
            # it active while both pads squeeze the wall over-constrains UIPC.
            self.blue_cup.remove_animate(force=True)
            self._actor_manager.update(dt=0.0)
        self.move(self.atom.close_gripper(pos=close_percent), tag="close_blue_cup_rim")
        self.settle_xense_after_close(is_save=False)
        self.record_xense_grasp_debug("xense_after_close_blue_cup_rim", self.blue_cup)
        self._blue_cup_shape_after_close = self._record_blue_cup_shape("after_close")
        self.metadata["grasp_height_bias"] = float(grasp_height_bias)
        self.metadata["gripper_close_percent"] = float(close_percent)

        yellow_cup_pose = self.yellow_cup.get_pose()
        blue_cup_pose = self.blue_cup.get_pose()
        target_y_offset = (
            float(getattr(self.cfg, "xense_pour_target_y_offset", 0.020))
            if is_xense
            else 0.030
        )
        target_z_offset = (
            float(getattr(self.cfg, "xense_pour_target_z_offset", 0.120))
            if is_xense
            else 0.120
        )
        target_blue_cup_pos = yellow_cup_pose.p + np.array([0.0, target_y_offset, target_z_offset])
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
        self.metadata["target_blue_cup_y_offset"] = float(target_y_offset)
        self.metadata["target_blue_cup_z_offset"] = float(target_z_offset)
        if is_xense:
            carry_settle_steps = int(getattr(self.cfg, "xense_pour_carry_settle_steps", 5))
            hold_actor_during_carry = bool(getattr(self.cfg, "xense_pour_hold_actor_during_carry", False))
            self.metadata["xense_pour_carry_settle_steps"] = int(carry_settle_steps)
            self.metadata["xense_pour_hold_actor_during_carry"] = bool(hold_actor_during_carry)
            lift_ok = self.move_actor_by_world_displacement_to_position(
                self.blue_cup,
                lift_blue_cup_pose.p,
                tag="lift_blue_cup_before_pour",
                settle_steps=carry_settle_steps,
                metadata_prefix="xense_blue_cup_pour_lift_path",
                actor_pose_hold=hold_actor_during_carry,
            )
            self.metadata["xense_pour_lift_motion_ok"] = bool(lift_ok)
            if not lift_ok:
                self.metadata["xense_pour_abort_reason"] = "lift_motion_failed"
                return False

            carry_ok = self.move_actor_by_world_displacement_to_position(
                self.blue_cup,
                target_blue_cup_pose.p,
                tag="move_blue_cup_y_near_yellow",
                segments=int(getattr(self.cfg, "xense_pour_carry_segments", 6)),
                settle_steps=carry_settle_steps,
                metadata_prefix="xense_blue_cup_pour_carry_path",
                actor_pose_hold=hold_actor_during_carry,
            )
            self.metadata["xense_pour_carry_motion_ok"] = bool(carry_ok)
            if not carry_ok:
                self.metadata["xense_pour_abort_reason"] = "carry_motion_failed"
                return False
        else:
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
        self._blue_cup_shape_before_pour = self._record_blue_cup_shape("before_pour")
        self.record_xense_grasp_debug("xense_before_pour_blue_cup", self.blue_cup)

        if is_xense:
            # Ensure the ball is governed only by contact and gravity during
            # the pour, even if a reset constraint survived an earlier step.
            self.red_ball.remove_animate(force=True)
            self._actor_manager.update(dt=0.0)

        # 第二步不再给蓝杯目标 pose 做反解。当前 translation 为 0, 会跳过 EEF 平移规划,
        # 只旋转最后一个关节 panda_joint7 来倒杯。
        self.metadata["blue_cup_pose_before_wrist_pour"] = self.blue_cup.get_pose().tolist()
        wrist_angle_deg = (
            float(getattr(self.cfg, "xense_pour_wrist_angle_deg", 180.0))
            if is_xense
            else 120.0
        )
        self.metadata["wrist_pour_angle_deg"] = float(wrist_angle_deg)
        wrist_translation = np.array([
            float(getattr(self.cfg, "xense_pour_wrist_translation_x", 0.0)) if is_xense else 0.0,
            float(getattr(self.cfg, "xense_pour_wrist_translation_y", 0.0)) if is_xense else 0.0,
            float(getattr(self.cfg, "xense_pour_wrist_translation_z", 0.0)) if is_xense else 0.0,
        ])
        self.metadata["wrist_pour_translation"] = wrist_translation.tolist()
        actor_tilt_deg = (
            float(getattr(self.cfg, "xense_pour_actor_tilt_deg", 0.0))
            if is_xense
            else 0.0
        )
        actor_tilt_axis = np.array([
            float(getattr(self.cfg, "xense_pour_actor_tilt_axis_x", 1.0)),
            float(getattr(self.cfg, "xense_pour_actor_tilt_axis_y", 0.0)),
            float(getattr(self.cfg, "xense_pour_actor_tilt_axis_z", 0.0)),
        ])
        self.metadata["xense_pour_actor_tilt_deg"] = float(actor_tilt_deg)
        self.metadata["xense_pour_actor_tilt_axis"] = actor_tilt_axis.tolist()
        if is_xense and abs(actor_tilt_deg) > 1e-6:
            self._rotate_last_arm_joint_with_actor_pour_pose(
                self.blue_cup,
                delta=np.deg2rad(wrist_angle_deg),
                translation=wrist_translation,
                actor_tilt_deg=actor_tilt_deg,
                actor_tilt_axis=actor_tilt_axis,
                steps=int(getattr(self.cfg, "xense_pour_wrist_steps", 160)),
                tag="actor_guided_translate_yz_and_rotate_panda_joint7_to_pour",
                is_save=True,
            )
        else:
            self._rotate_last_arm_joint_with_ee_translation(
                delta=np.deg2rad(wrist_angle_deg),
                translation=wrist_translation,
                steps=int(getattr(self.cfg, "xense_pour_wrist_steps", 160)) if is_xense else 160,
                tag="translate_yz_and_rotate_panda_joint7_to_pour",
                is_save=True,
            )
        self.record_xense_grasp_debug(
            "xense_after_wrist_pour_blue_cup",
            self.blue_cup,
        )
        self._blue_cup_shape_after_pour = self._record_blue_cup_shape("after_pour")
        if is_xense:
            release_lift = float(getattr(self.cfg, "xense_pour_release_lift", 0.04))
            if release_lift > 0.0:
                self.metadata["red_ball_pose_before_release_lift"] = self.red_ball.get_pose().tolist()
                self.move(
                    self.atom.move_by_displacement(z=release_lift, xyz_coord="world"),
                    tag="lift_cup_to_release_ball",
                    time_dilation_factor=1.0,
                    delay=False,
                )
                self.metadata["red_ball_pose_after_release_lift"] = self.red_ball.get_pose().tolist()
                self.metadata["blue_cup_pose_after_release_lift"] = self.blue_cup.get_pose().tolist()

            release_snap_angle = float(
                getattr(self.cfg, "xense_pour_release_snap_angle_deg", 35.0)
            )
            release_snap_steps = int(getattr(self.cfg, "xense_pour_release_snap_steps", 12))
            release_snap_cycles = int(getattr(self.cfg, "xense_pour_release_snap_cycles", 4))
            self.metadata["red_ball_pose_before_release_snap"] = self.red_ball.get_pose().tolist()
            self.metadata["release_snap_ball_poses"] = []
            self.metadata["release_snap_blue_cup_poses"] = []
            for snap_idx in range(release_snap_cycles):
                signed_angle = release_snap_angle if snap_idx % 2 == 0 else -release_snap_angle
                self._rotate_last_arm_joint(
                    delta=np.deg2rad(signed_angle),
                    steps=release_snap_steps,
                    tag=f"snap_wrist_to_release_ball_{snap_idx}",
                    is_save=True,
                )
                self.metadata["release_snap_ball_poses"].append(self.red_ball.get_pose().tolist())
                self.metadata["release_snap_blue_cup_poses"].append(self.blue_cup.get_pose().tolist())
            self.metadata["red_ball_pose_after_release_snap"] = self.red_ball.get_pose().tolist()
            self.metadata["blue_cup_pose_after_release_snap"] = self.blue_cup.get_pose().tolist()
            if bool(getattr(self.cfg, "xense_pour_fix_cup_during_release", False)):
                release_hold_pose = self.blue_cup.get_pose()
                self.blue_cup.set_pose(release_hold_pose)
                self.metadata["blue_cup_release_hold_pose"] = release_hold_pose.tolist()
                self.delay(2, is_save=True)
                # The transform constraint is intentionally soft. Release the
                # Robotiq pads so their sustained squeeze cannot drag the cup
                # away from the hold target while the ball falls out.
                self.move(
                    self.atom.open_gripper(1.0),
                    tag="open_gripper_to_release_ball",
                    delay=False,
                )
                self.metadata["blue_cup_pose_after_release_open"] = self.blue_cup.get_pose().tolist()
                self.metadata["red_ball_pose_after_release_open"] = self.red_ball.get_pose().tolist()
        self._wait_ball_until_still(tag="wait_red_ball_still_after_pour")
        self.metadata["blue_cup_pose_after_wrist_pour"] = self.blue_cup.get_pose().tolist()
        self.metadata["red_ball_pose_after_wait"] = self.red_ball.get_pose().tolist()


    def check_success(self):
        ball_in_yellow_cup = self._is_ball_in_yellow_cup()
        blue_shape_final = self._record_blue_cup_shape("final")
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        min_shape_ratio = float(getattr(self.cfg, "xense_cup_min_principal_ratio", 0.90))
        max_nonrigid_error = float(getattr(self.cfg, "xense_cup_max_nonrigid_error", 0.08))
        shape_samples = (
            getattr(self, "_blue_cup_shape_after_close", {}),
            getattr(self, "_blue_cup_shape_before_pour", {}),
            getattr(self, "_blue_cup_shape_after_pour", {}),
            blue_shape_final,
        )
        blue_shape_ok = (not is_xense) or all(
            sample.get("min_principal_spread_ratio", 0.0) >= min_shape_ratio
            and sample.get("normalized_nonrigid_error", float("inf")) <= max_nonrigid_error
            for sample in shape_samples
        )
        self.metadata["blue_cup_shape_thresholds"] = {
            "min_principal_spread_ratio": min_shape_ratio,
            "max_nonrigid_error": max_nonrigid_error,
        }
        self.metadata["success_checks"].update({
            "ball_in_yellow_cup": bool(ball_in_yellow_cup),
            "blue_shape_ok": bool(blue_shape_ok),
        })
        return bool(ball_in_yellow_cup and blue_shape_ok)
