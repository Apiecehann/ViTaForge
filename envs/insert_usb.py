from ._base_task import *
import numpy as np

# USB尺寸：
# assets/objects/USB.usd, 横截面：12.0mm x 4.5mm，插头长度：12.4mm，USB本体长度：45mm
# assets/objects/usb.usd, 横截面：12.4mm x 4.4mm，插头长度：12.4mm，USB本体长度：50mm
# USB.usd对应的Visual mesh: assets/objects/USB03_visual.usd

# USB孔：
# assets/objects/USB_slot_start.usd, 孔：13.5mm x 6.0mm
# assets/objects/USB_slot_target.usd, 孔：13.2mm x 5.7mm
# assets/objects/USB_slot_start_001.usd, 孔：13.9mm x 5.9mm
# assets/objects/USB_slot_target_001.usd, 孔：13.4mm x 5.4mm

# 下面的尺寸来自 assets/objects/*.usd 的网格顶点，用于把 USB 和插槽在 z 方向上对齐。
USB_PLUG_HEIGHT = 0.0124
USB_BODY_HEIGHT = 0.0450
# USB_BODY_HEIGHT = 0.0500
# 抓取点设置在插头以上、USB 本体中部附近，再略微上移，避免夹爪碰到插槽或桌面。
USB_GRASP_HEIGHT = USB_PLUG_HEIGHT + USB_BODY_HEIGHT * 0.5 + 0.007
USB_GRASP_HEIGHT_NOISE = 0.003
SLOT_HEIGHT = 0.0200
SLOT_HOLE_BOTTOM = 0.0050
# USB 插入后的目标 z：至少高于插槽孔底，同时考虑插头自身高度。
USB_INSERT_Z = max(SLOT_HOLE_BOTTOM, SLOT_HEIGHT - USB_PLUG_HEIGHT)

# episode 级随机化和脚本动作幅度。xy 噪声只扰动插槽平面位置，不扰动高度。
SLOT_XY_NOISE = (0.005, 0.005, 0.0)
LIFT_HEIGHT = 0.0200
LIFT_HEIGHT_NOISE = 0.0050
# pre_move 先移到槽口上方 10 mm，并在插槽平面内叠加小幅位置噪声。
SLOT_APPROACH_CLEARANCE = 0.010
SLOT_APPROACH_XY_NOISE = (0.002, 0.002, 0.0)
# _play_once 下插前将 USB 精确移到槽口高度。
PLAY_PRE_INSERT_CLEARANCE = 0.0


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.65, 0.14, 0.11), rot=(0.370608, 0.310977, 0.562556, 0.670428), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.6, focus_distance=1.0, horizontal_aperture=2.4, clipping_range=(0.1, 100.0)
            ),
            width=480,
            height=270,
            update_period=1/120
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None, # use existing camera
            width=480,
            height=270,
            update_period=1/120,
        )
    ]
    step_lim = 300


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        # USB 插入任务对接触稳定性敏感，提高摩擦可以减少夹取和下插过程中的打滑。
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

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
        target_slot_pose = Pose([0.5, 0.0, 0.002], [1, 0, 0, 0])
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
            density=1e6
        )

        self.prism = self._actor_manager.add_from_usd_file(
            name='prism',
            asset_path="USB.usd",
            visual_asset_path="USB03_visual.usd", 
            pose=usb_pose,
            show_physics_mesh=False
        )
    

    def _reset_actors(self):
        # Pose.create_noise 会修改输入向量，因此每次传入新的 list，避免跨 episode 污染随机范围。
        start_offset = self.create_noise(list(SLOT_XY_NOISE))
        target_offset = self.create_noise(list(SLOT_XY_NOISE))
        start_slot_pose = Pose([0.4, 0.0, self.start_slot.get_pose()[2]], [1, 0, 0, 0]).add_offset(start_offset)
        target_slot_pose = Pose([0.5, 0.0, self.slot.get_pose()[2]], [1, 0, 0, 0]).add_offset(target_offset)

        self.start_slot.set_pose(start_slot_pose)
        self.slot.set_pose(target_slot_pose)
        self.prism.set_pose(self._usb_pose_in_slot(start_slot_pose))
        self._update_insert_reference_poses()
        # 保存 reset 后的实际槽位，方便离线复现或诊断插入失败样本。
        self.metadata['start_slot_pose'] = start_slot_pose.tolist()
        self.metadata['target_slot_pose'] = target_slot_pose.tolist()

    def pre_move(self):
        # 正式动作前等待 10 个仿真步，让 reset 后的物体接触状态先稳定下来。
        self.delay(10)

        self.move(self.atom.open_gripper(0.5), tag="open_gripper_for_usb")
        # 抓取姿态保留少量俯仰角和高度噪声，让演示覆盖轻微抓取误差。
        grasp_rotate = self.rng.uniform(-np.pi/18, np.pi/18)
        grasp_height = USB_GRASP_HEIGHT + self.rng.uniform(
            -USB_GRASP_HEIGHT_NOISE,
            USB_GRASP_HEIGHT_NOISE
        )
        target_pose = self.prism.get_pose().add_bias([0, 0, grasp_height]).add_rotation([0, grasp_rotate, 0])
        target_mat = target_pose.to_transformation_matrix()
        # construct_grasp_pose 使用抓取点、接近方向和夹爪横向方向生成末端抓取姿态。
        cpose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0]
        )
        # 将抓取点注册到 USB actor 上，后续 grasp_actor 会按照该局部 contact point 规划接近轨迹。
        cid = self.prism.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.prism,
            contact_point_id=cid,
            is_close=False
        ), tag="approach_usb")
        self.move(self.atom.close_gripper(), tag="close_usb")

        # 抓住后先上提约 3cm，并加入少量 z 噪声，给 USB 离开起始槽和桌面留出安全余量。
        lift_height = LIFT_HEIGHT + self.rng.uniform(
            -LIFT_HEIGHT_NOISE,
            LIFT_HEIGHT_NOISE
        )
        self.move(self.atom.move_by_displacement(z=lift_height), tag="lift_usb")

        self._update_insert_reference_poses()
        # 移动到槽口上方 10 mm，并在精确位姿上叠加 XY 各 ±2 mm 噪声。
        approach_clearance = SLOT_APPROACH_CLEARANCE
        self._update_pre_insert_pose(approach_clearance)
        approach_offset = self.create_noise(list(SLOT_APPROACH_XY_NOISE))
        approach_pose = self.pre_insert_pose.add_offset(approach_offset)
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=approach_pose,
            pre_dis=0.02,
            dis=0.004,
            is_open=False
        ), tag="move_usb_to_pre_insert")

        # 记录本轮脚本采样到的关键随机量，便于复现实验和分析失败数据。
        self.metadata['grasp_rotate'] = float(grasp_rotate)
        self.metadata['grasp_height'] = float(grasp_height)
        self.metadata['lift_height'] = float(lift_height)
        self.metadata['approach_clearance'] = float(approach_clearance)
        self.metadata['approach_xy_noise'] = approach_offset.p.tolist()

    def _play_once(self):
        self._update_insert_reference_poses()
        # 正式下插前去掉 XY 噪声，将 USB 参考原点精确移到槽口高度。
        play_pre_insert_pose = self.opening_pose.add_bias([0.0, 0.0, PLAY_PRE_INSERT_CLEARANCE])
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=play_pre_insert_pose,
            pre_dis=0.01,
            dis=0.0,
            is_open=False
        ), tag="move_usb_to_play_pre_insert", time_dilation_factor=0.5)
        self.metadata['play_pre_insert_clearance'] = PLAY_PRE_INSERT_CLEARANCE
        # 根据 USB 当前实际高度和目标插入高度计算下插距离，避免假设预插入高度完全等于期望值。
        insert_distance = max(0.0, float(self.prism.get_pose().p[2] - self.target_pose.p[2]))
        self.metadata['insert_distance'] = insert_distance
        self.move(self.atom.move_by_displacement(
            z=-insert_distance,
            xyz_coord='world'
        ), tag="insert_usb_into_slot", time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
        # 下插后保存一段稳定观测，便于 success 检查和离线数据回放看到最终状态。
        self.delay(40, is_save=True)

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
