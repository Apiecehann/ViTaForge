from ._base_task import *
import numpy as np


# 下面的尺寸来自 assets/objects/*.usd 的网格顶点，用于把 USB 和插槽在 z 方向上对齐。
USB_PLUG_HEIGHT = 0.0124
USB_BODY_HEIGHT = 0.0500
# 抓取点设置在插头以上、USB 本体中部附近，再上移 12mm，避免夹取位置过低。
USB_GRASP_HEIGHT = USB_PLUG_HEIGHT + USB_BODY_HEIGHT * 0.5 + 0.007
USB_GRASP_HEIGHT_NOISE = 0.0
SLOT_HEIGHT = 0.0200
SLOT_HOLE_BOTTOM = 0.0050
# USB 插入后的目标 z：至少高于插槽孔底，同时考虑插头自身高度。
USB_INSERT_Z = max(SLOT_HOLE_BOTTOM, SLOT_HEIGHT - USB_PLUG_HEIGHT)

# episode 级随机化和脚本动作幅度。xy 噪声只扰动插槽平面位置，不扰动高度。
SLOT_XY_NOISE = (0.0, 0.0, 0.0)
LIFT_HEIGHT = 0.0500
LIFT_HEIGHT_NOISE = 0.0
# 预插入位姿相对插槽开口只保留很小 clearance，后续 _play_once 再竖直下插。
SLOT_APPROACH_CLEARANCE = 0.040
SLOT_APPROACH_CLEARANCE_NOISE = 0.0
# _play_once 下插前使用的精确预插入高度，确保 USB 先对准到槽口上方0mm。
PLAY_PRE_INSERT_CLEARANCE = 0.012
# Empirical world-frame bias aligning the scalar gripper center to the Xense pad midpoint.
XENSE_GRASP_CENTER_BIAS = (0.0, 0.0, 0.0)
# The Robotiq/Xense grasp under-tracks held USB placement in x/y; keep success checks on the true slot.
XENSE_INSERT_MOTION_BIAS = (0.0, 0.0, 0.0)
XENSE_PLAY_INSERT_MOTION_BIAS = (0.0050, -0.0012, 0.0)
XENSE_INSERT_EXTRA_DEPTH = 0.014
XENSE_INSERT_STAGE_MAX_STEP = 0.004
XENSE_INSERT_XY_CORRECTION_LIMIT = 0.0015
XENSE_INSERT_XY_CORRECTION_DEADBAND = 0.0004
POST_INSERT_XY_CORRECTION_LIMIT = 0.08
POST_RELEASE_RETREAT_HEIGHT = 0.04
POST_OPEN_SETTLE_STEPS = 60
POST_TACTILE_RESET_SETTLE_STEPS = 20
# Robotiq/Xense needs extra settle time to actually reach the 45 deg close target before lifting.
POST_CLOSE_SETTLE_STEPS = 80
# close_gripper uses an open fraction: 0.0 -> full close qpos, 1.0 -> open qpos.
# Diagnostic: keep the pads less compressed while still attempting to hold the USB.
XENSE_USB_CLOSE_PERCENT = 0.185


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 2200
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.74, 0.0, 0.066), rot=(0.512, 0.512, 0.487, 0.487), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
            ),
            width=480,
            height=270,
            update_period=1/120
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/XenseWristCamera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.48, 0.0, 0.14),
                rot=(0.827725, 0.528309, 0.101743, 0.159405),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.2, focus_distance=0.45, horizontal_aperture=3.6, clipping_range=(0.01, 100.0)
            ),
            width=480,
            height=270,
            update_period=1/120,
        )
    ]
    step_lim = 2200


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        # USB 插入任务对接触稳定性敏感，提高摩擦可以减少夹取和下插过程中的打滑。
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        # Keep the UIPC Newton solver at the GelSight/default quality. A tiny
        # diagnostic cap here makes gel contact under-converged and looks like an
        # unrealistically soft pad.
        # Keep adaptive grasp enabled for Xense depth/contact diagnostics. The
        # threshold is resolved from the shared Python sensor/task defaults.
        cfg.use_adaptive_grasp = getattr(cfg, "use_adaptive_grasp", True)
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        cfg.reset_time_limit = max(float(cfg.reset_time_limit), 300.0)
        return cfg

    def _usb_pose_in_slot(self, slot_pose: Pose):
        # 给定插槽位姿，计算 USB 完成插入时应处于的目标位姿。
        return slot_pose.add_bias([0.0, 0.0, USB_INSERT_Z])

    def _update_insert_reference_poses(self):
        # 每次都读取当前插槽位姿；该位姿已经包含 reset 时采样到的 xy 噪声。
        self.target_pose = self._usb_pose_in_slot(self.slot.get_pose())
        # opening_pose 表示插槽开口上表面位置，用作预插入位姿的基准。
        self.opening_pose = self.slot.get_pose().add_bias([0.0, 0.0, SLOT_HEIGHT])

    def _update_pre_insert_pose(self, clearance):
        # 预插入位姿位于随机化后的槽口正上方，clearance 控制离槽口表面的安全高度。
        self.approach_clearance = float(clearance)
        self.pre_insert_pose = self.opening_pose.add_bias([0.0, 0.0, self.approach_clearance])

    def create_actors(self):
        # start_slot 是 USB 初始放置槽，slot 是目标插入槽；USB 初始就放在起始槽内。
        start_slot_pose = Pose([0.4, 0.0, 0.002], [1, 0, 0, 0])
        target_slot_pose = Pose([0.58, 0.0, 0.002], [1, 0, 0, 0])
        usb_pose = self._usb_pose_in_slot(start_slot_pose)

        self.start_slot = self._actor_manager.add_from_usd_file(
            name='start_slot',
            asset_path="USB_slot_start.usd",
            pose=start_slot_pose,
            density=1e6
        )

        self.slot = self._actor_manager.add_from_usd_file(
            name='slot',
            asset_path="USB_slot_target.usd",
            pose=target_slot_pose,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(kinematic=True),
            density=1e6
        )

        self.prism = self._actor_manager.add_from_usd_file(
            name='prism',
            asset_path="usb.usd",
            pose=usb_pose
        )

    def _reset_actors(self):
        # Pose.create_noise 会修改输入向量，因此每次传入新的 list，避免跨 episode 污染随机范围。
        start_offset = self.create_noise(list(SLOT_XY_NOISE))
        target_offset = self.create_noise(list(SLOT_XY_NOISE))
        start_slot_pose = Pose([0.4, 0.0, self.start_slot.get_pose()[2]], [1, 0, 0, 0]).add_offset(start_offset)
        target_slot_pose = Pose([0.58, 0.0, self.slot.get_pose()[2]], [1, 0, 0, 0]).add_offset(target_offset)

        self.start_slot.set_pose(start_slot_pose)
        self.slot.set_pose(target_slot_pose)
        self.prism.set_pose(self._usb_pose_in_slot(start_slot_pose))
        self._update_insert_reference_poses()
        # 保存 reset 后的实际槽位，方便离线复现或诊断插入失败样本。
        self.metadata['start_slot_pose'] = start_slot_pose.tolist()
        self.metadata['target_slot_pose'] = target_slot_pose.tolist()

    def _ee_pose_for_held_usb_target(self, target_pose: Pose):
        gripper_center_pose = self._robot_manager.get_gripper_center_pose()
        usb_in_gripper = self.prism.get_pose().rebase(to_coord=gripper_center_pose)
        desired_gripper_center_mat = (
            target_pose.to_transformation_matrix()
            @ np.linalg.inv(usb_in_gripper.to_transformation_matrix())
        )
        desired_gripper_center_pose = Pose.from_matrix(desired_gripper_center_mat)
        return self._robot_manager.gripper_center_to_ee(desired_gripper_center_pose)

    def _move_held_usb_to_pose(self, target_pose: Pose, tag: str, time_dilation_factor=None):
        ee_pose = self._ee_pose_for_held_usb_target(target_pose)
        return self.move(
            self.atom.move_to_pose(ee_pose),
            tag=tag,
            time_dilation_factor=time_dilation_factor,
        )

    def _move_held_usb_by_translation(self, target_pose: Pose, tag: str, time_dilation_factor=None, constraint_pose=None):
        delta = target_pose.p - self.prism.get_pose().p
        return self.move(
            self.atom.move_by_displacement(
                x=float(delta[0]),
                y=float(delta[1]),
                z=float(delta[2]),
                xyz_coord='world',
            ),
            tag=tag,
            time_dilation_factor=time_dilation_factor,
            constraint_pose=constraint_pose,
        )

    def _record_xense_debug_pose(self, label):
        debug = self.metadata.setdefault('xense_debug_poses', {})
        entry = {
            'prism_pose': self.prism.get_pose().tolist(),
            'slot_pose': self.slot.get_pose().tolist(),
            'target_pose': self._usb_pose_in_slot(self.slot.get_pose()).tolist(),
            'ee_pose': self._robot_manager.get_ee_pose().tolist(),
            'gripper_center_pose': self._robot_manager.get_gripper_center_pose().tolist(),
            'gripper_qpos': float(self._robot_manager.get_gripper_qpos()),
        }
        try:
            entry['prism_in_slot'] = self.prism.get_pose().rebase(self._usb_pose_in_slot(self.slot.get_pose())).tolist()
            entry['prism_in_gripper_center'] = self.prism.get_pose().rebase(self._robot_manager.get_gripper_center_pose()).tolist()
            prism_vertices = torch.as_tensor(self.prism.vertices, dtype=torch.float32, device=self._robot_manager.device)
            for tact_name, tact in self._tactile_manager.tactiles.items():
                gel_vertices = tact.gelpad._data.nodal_pos_w.to(dtype=torch.float32)
                dist_mat = torch.cdist(prism_vertices, gel_vertices)
                min_flat = torch.argmin(dist_mat)
                min_i = torch.div(min_flat, dist_mat.shape[1], rounding_mode='floor')
                min_j = min_flat % dist_mat.shape[1]
                prism_closest = prism_vertices[min_i]
                gel_closest = gel_vertices[min_j]
                min_vec = gel_closest - prism_closest
                entry[f'usb_to_{tact_name}_gelpad_min_dist_m'] = float(dist_mat.flatten()[min_flat].item())
                entry[f'usb_to_{tact_name}_gelpad_min_vec_m'] = [float(x) for x in min_vec.detach().cpu().tolist()]
                entry[f'usb_to_{tact_name}_usb_closest_w'] = [float(x) for x in prism_closest.detach().cpu().tolist()]
                entry[f'usb_to_{tact_name}_gelpad_closest_w'] = [float(x) for x in gel_closest.detach().cpu().tolist()]
                attach_idx = tact.attachment.attachment_points_idx
                entry[f'{tact_name}_attachment_count'] = int(len(attach_idx))
                entry[f'{tact_name}_attachment_body_pose'] = tact.get_attach_pose().tolist()
                if len(attach_idx) > 0:
                    gel_attach_points = gel_vertices[attach_idx]
                    aim_points = tact.attachment._compute_aim_positions()
                    aim_points = torch.as_tensor(
                        aim_points, dtype=torch.float32, device=gel_attach_points.device
                    )
                    attach_error = torch.linalg.norm(gel_attach_points - aim_points, dim=1)
                    entry[f'{tact_name}_gel_to_attachment_aim_mean_m'] = float(attach_error.mean().item())
                    entry[f'{tact_name}_gel_to_attachment_aim_max_m'] = float(attach_error.max().item())
                    entry[f'{tact_name}_gel_attach_centroid_w'] = [
                        float(x) for x in gel_attach_points.mean(dim=0).detach().cpu().tolist()
                    ]
                    entry[f'{tact_name}_aim_centroid_w'] = [
                        float(x) for x in aim_points.mean(dim=0).detach().cpu().tolist()
                    ]
                    entry[f'{tact_name}_gel_centroid_w'] = [
                        float(x) for x in gel_vertices.mean(dim=0).detach().cpu().tolist()
                    ]
        except Exception as exc:
            entry['pose_debug_error'] = repr(exc)
        debug[label] = entry

    def pre_move(self):
        # 正式动作前等待 10 个仿真步，让 reset 后的物体接触状态先稳定下来。
        self.delay(10)
        if hasattr(self._tactile_manager, "reset_reference"):
            self.metadata['xense_reference_reset_result'] = self._tactile_manager.reset_reference()
            self.metadata['xense_reference_reset_step'] = int(self.step_count)
            self.delay(POST_TACTILE_RESET_SETTLE_STEPS, is_save=False)
            self._record_xense_debug_pose('after_initial_tactile_reset')

        self.move(self.atom.open_gripper(1.0), tag="open_gripper_for_usb")
        self.delay(POST_OPEN_SETTLE_STEPS, is_save=False)
        self._record_xense_debug_pose('after_open_for_usb')
        # 抓取姿态保留少量俯仰角和高度噪声，让演示覆盖轻微抓取误差。
        grasp_rotate = 0.0
        grasp_height = USB_GRASP_HEIGHT + self.rng.uniform(
            -USB_GRASP_HEIGHT_NOISE,
            USB_GRASP_HEIGHT_NOISE
        )
        # Robotiq/Xense uses a fixed world-aligned vertical grasp. Do not inherit
        # the USB actor rotation here: after reset the USB can be slightly tilted,
        # and following that pose makes the gripper approach diagonally.
        target_p = self.prism.get_pose().p.copy()
        target_p[:2] = self.start_slot.get_pose().p[:2]
        target_p[2] = self.start_slot.get_pose().p[2] + USB_INSERT_Z + grasp_height
        target_p = target_p + np.array(XENSE_GRASP_CENTER_BIAS, dtype=float)
        # Use world Y as the in-plane reference so Robotiq closes across
        # the USB narrow face instead of filling the Xense tactile window.
        cpose = construct_grasp_pose(
            target_p,
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0])
        )
        # 将抓取点注册到 USB actor 上，后续 grasp_actor 会按照该局部 contact point 规划接近轨迹。
        cid = self.prism.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.prism,
            contact_point_id=cid,
            is_close=False
        ), tag="approach_usb")
        self._record_xense_debug_pose('after_approach_usb')
        usb_close_percent = float(getattr(self.cfg, "xense_usb_close_percent", XENSE_USB_CLOSE_PERCENT))
        usb_close_percent = float(np.clip(usb_close_percent, 0.0, 1.0))
        post_close_settle_steps = int(getattr(self.cfg, "xense_post_close_settle_steps", POST_CLOSE_SETTLE_STEPS))
        close_target_qpos = float(self._robot_manager.gripper_percent2qpos(usb_close_percent))
        self.metadata['usb_close_percent'] = float(usb_close_percent)
        self.metadata['usb_close_target_qpos'] = close_target_qpos
        self.metadata['post_close_settle_steps'] = int(post_close_settle_steps)
        self.metadata['gripper_open_qpos'] = float(self._robot_manager.gripper_open_qpos)
        self.metadata['gripper_close_qpos'] = float(self._robot_manager.gripper_close_qpos)
        self.move(self.atom.close_gripper(pos=usb_close_percent), tag="close_usb")
        self.delay(post_close_settle_steps, is_save=True)
        self._record_xense_debug_pose('after_close')

        # 抓住后先上提约 3cm，并加入少量 z 噪声，给 USB 离开起始槽和桌面留出安全余量。
        lift_height = LIFT_HEIGHT + self.rng.uniform(
            -LIFT_HEIGHT_NOISE,
            LIFT_HEIGHT_NOISE
        )
        self.move(self.atom.move_by_displacement(z=lift_height), tag="lift_usb")
        self._record_xense_debug_pose('after_lift')

        self._update_insert_reference_poses()
        # 移动到随机化后的目标槽口正上方；真正的竖直下插放在 _play_once 中执行。
        approach_clearance = SLOT_APPROACH_CLEARANCE + self.rng.uniform(
            -SLOT_APPROACH_CLEARANCE_NOISE,
            SLOT_APPROACH_CLEARANCE_NOISE
        )
        self._update_pre_insert_pose(approach_clearance)
        motion_pre_insert_pose = self.pre_insert_pose.add_bias(XENSE_INSERT_MOTION_BIAS, coord='world')
        # Keep the post-grasp in-hand orientation while moving to the slot approach.
        # place_actor recomputes a target grasp orientation and can twist the USB sideways.
        self._move_held_usb_by_translation(
            motion_pre_insert_pose,
            tag="move_usb_to_pre_insert",
        )
        self._record_xense_debug_pose('after_pre_insert')

        # 记录本轮脚本采样到的关键随机量，便于复现实验和分析失败数据。
        self.metadata['grasp_rotate'] = float(grasp_rotate)
        self.metadata['grasp_height'] = float(grasp_height)
        self.metadata['grasp_center_bias'] = list(XENSE_GRASP_CENTER_BIAS)
        self.metadata['insert_motion_bias'] = list(XENSE_INSERT_MOTION_BIAS)
        self.metadata['play_insert_motion_bias'] = list(XENSE_PLAY_INSERT_MOTION_BIAS)
        self.metadata['lift_height'] = float(lift_height)
        self.metadata['approach_clearance'] = float(approach_clearance)

    def _play_once(self):
        self._update_insert_reference_poses()
        play_pre_insert_pose = self.opening_pose.add_bias([0.0, 0.0, PLAY_PRE_INSERT_CLEARANCE])
        motion_play_pre_insert_pose = play_pre_insert_pose.add_bias(XENSE_PLAY_INSERT_MOTION_BIAS, coord='world')
        self._move_held_usb_by_translation(
            motion_play_pre_insert_pose,
            tag="move_usb_to_play_pre_insert",
            time_dilation_factor=0.5,
        )
        self._record_xense_debug_pose('after_play_pre_insert')
        self.metadata['play_pre_insert_clearance'] = PLAY_PRE_INSERT_CLEARANCE
        # Compute vertical insertion distance from the current USB height.
        # Use short stages so metadata shows when contact starts to tip the USB.
        insert_distance = max(0.0, float(self.prism.get_pose().p[2] - self.target_pose.p[2] + XENSE_INSERT_EXTRA_DEPTH))
        self.metadata['insert_distance'] = insert_distance
        self.metadata['insert_extra_depth'] = XENSE_INSERT_EXTRA_DEPTH
        self.metadata['insert_stage_max_step'] = XENSE_INSERT_STAGE_MAX_STEP

        remaining_insert_distance = insert_distance
        insert_stage_idx = 0
        xy_corrections = []
        while remaining_insert_distance > 1e-6:
            insert_stage_idx += 1
            dz = min(float(XENSE_INSERT_STAGE_MAX_STEP), remaining_insert_distance)
            self.move(self.atom.move_by_displacement(
                z=-dz,
                xyz_coord='world'
            ), tag=f"insert_usb_into_slot_stage_{insert_stage_idx}", time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
            self.delay(10, is_save=True)
            self._record_xense_debug_pose(f'after_insert_stage_{insert_stage_idx}_raw')

            rel_pose = self.prism.get_pose().rebase(self.target_pose)
            correction_local = np.array([-float(rel_pose.p[0]), -float(rel_pose.p[1]), 0.0], dtype=float)
            correction_norm = float(np.linalg.norm(correction_local[:2]))
            applied_correction = [0.0, 0.0, 0.0]
            if correction_norm > XENSE_INSERT_XY_CORRECTION_DEADBAND:
                correction_scale = min(correction_norm, float(XENSE_INSERT_XY_CORRECTION_LIMIT)) / correction_norm
                correction_local = correction_local * correction_scale
                target_rot = self.target_pose.to_transformation_matrix()[:3, :3]
                correction_world = target_rot @ correction_local
                applied_correction = [float(correction_world[0]), float(correction_world[1]), 0.0]
                self.move(self.atom.move_by_displacement(
                    x=applied_correction[0],
                    y=applied_correction[1],
                    z=0.0,
                    xyz_coord='world'
                ), tag=f"insert_xy_correct_stage_{insert_stage_idx}", time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
                self.delay(5, is_save=True)

            xy_corrections.append({
                'stage': insert_stage_idx,
                'raw_xy_error': [float(rel_pose.p[0]), float(rel_pose.p[1])],
                'applied_world_correction': applied_correction,
            })
            self._record_xense_debug_pose(f'after_insert_stage_{insert_stage_idx}')
            remaining_insert_distance -= dz

        self.metadata['insert_xy_correction_limit'] = XENSE_INSERT_XY_CORRECTION_LIMIT
        self.metadata['insert_xy_correction_deadband'] = XENSE_INSERT_XY_CORRECTION_DEADBAND
        self.metadata['insert_xy_corrections'] = xy_corrections

        self.metadata['insert_stage_count'] = insert_stage_idx
        self._record_xense_debug_pose('after_insert')

        self.delay(20, is_save=True)

    def _get_success_diagnostics(self, xy_threshold=0.002, z_threshold=0.003):
        cached_target_pose = self.target_pose
        # 重新计算目标位姿，避免使用过期缓存；同时保留 cached_target_pose 便于诊断差异。
        target_pose = self._usb_pose_in_slot(self.slot.get_pose())
        self.target_pose = target_pose

        # 将 USB 位姿变换到目标插槽坐标系下，xy/z 误差就可以直接作为成功判定输入。
        prism_pose = self.prism.get_pose().rebase(target_pose)
        ee_pose = self._robot_manager.get_ee_pose()
        # 使用 USB 局部 z 轴和目标竖直 z 轴的夹角，判断插入后是否明显歪斜。
        prism_z_axis = prism_pose.to_transformation_matrix()[:3, 2]
        target_z_axis = np.array([0, 0, 1])
        z_axis_dot = float(np.dot(prism_z_axis, target_z_axis))
        tilt_angle_deg = float(np.degrees(np.arccos(np.clip(z_axis_dot, -1.0, 1.0))))
        xy_error = float(np.linalg.norm(prism_pose.p[:2]))
        pos_error = float(np.linalg.norm(prism_pose.p))

        return {
            'target_pose': target_pose.tolist(),
            'cached_target_pose': cached_target_pose.tolist(),
            'rel_pose': prism_pose.tolist(),
            'rel_xyz': prism_pose.p.tolist(),
            'xy_error': xy_error,
            'z_error': float(prism_pose.p[2]),
            'abs_z_error': float(np.abs(prism_pose.p[2])),
            'pos_error': pos_error,
            'tilt_angle_deg': tilt_angle_deg,
            'z_axis_dot': z_axis_dot,
            'ee_z': float(ee_pose[2]),
            'xy_ok': bool(np.all(np.abs(prism_pose.p[:2]) < np.array([xy_threshold, xy_threshold]))),
            'z_ok': bool(np.abs(prism_pose.p[2]) < z_threshold),
            'ee_z_ok': bool(ee_pose[2] > 0.145),
            'angle_ok': bool(z_axis_dot > 0.965), # 约 15 度以内认为姿态足够竖直。
        }

    def _print_failure_diagnostics(self, diagnostics):
        print(
            "\n[insert_USB_0701 failure diagnostics]\n"
            f"  rel_xyz(m): {diagnostics['rel_xyz']}\n"
            f"  xy_error: {diagnostics['xy_error']:.6f} m\n"
            f"  z_error: {diagnostics['z_error']:.6f} m "
            f"(abs {diagnostics['abs_z_error']:.6f} m)\n"
            f"  pos_error: {diagnostics['pos_error']:.6f} m\n"
            f"  tilt_angle: {diagnostics['tilt_angle_deg']:.3f} deg\n"
            f"  z_axis_dot: {diagnostics['z_axis_dot']:.6f}\n"
            f"  ee_z: {diagnostics['ee_z']:.6f} m\n"
            f"  checks: xy={diagnostics['xy_ok']}, z={diagnostics['z_ok']}, "
            f"ee_z={diagnostics['ee_z_ok']}, angle={diagnostics['angle_ok']}"
        )

    def check_success(self, xy_threshold=0.002, z_threshold=0.003):
        # 成功需要 USB 在目标槽坐标系下 xy/z 误差足够小、末端仍处在合理高度、姿态没有明显倾斜。
        diagnostics = self._get_success_diagnostics(xy_threshold, z_threshold)
        self.metadata['rel_pose'] = diagnostics['rel_pose']
        self.metadata['success_diagnostics'] = diagnostics
        success = diagnostics['xy_ok'] and diagnostics['z_ok'] \
            and diagnostics['ee_z_ok'] and diagnostics['angle_ok']
        # 采集/测试失败时只打印一次详细诊断，避免终端被重复日志刷屏。
        if not success and self.mode in ['collect', 'eval_test'] \
                and not self.metadata.get('failure_diagnostics_printed', False):
            self._print_failure_diagnostics(diagnostics)
            self.metadata['failure_diagnostics_printed'] = True
        return success
