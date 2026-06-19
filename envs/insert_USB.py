from ._base_task import *
import numpy as np


# Dimensions read from assets/objects/*.usd mesh vertices.
USB_PLUG_HEIGHT = 0.0124
USB_BODY_HEIGHT = 0.0500
USB_GRASP_HEIGHT = USB_PLUG_HEIGHT + USB_BODY_HEIGHT * 0.5 + 0.006
SLOT_HEIGHT = 0.0200
SLOT_HOLE_BOTTOM = 0.0050
USB_INSERT_Z = max(SLOT_HOLE_BOTTOM, SLOT_HEIGHT - USB_PLUG_HEIGHT)
PRE_INSERT_CLEARANCE = 0.0100


@configclass
class TaskCfg(BaseTaskCfg):
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
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None, # use existing camera
            width=480,
            height=270,
            update_period=1/120,
        )
    ]
    step_lim = 600


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def _usb_pose_in_slot(self, slot_pose: Pose):
        return slot_pose.add_bias([0.0, 0.0, USB_INSERT_Z])

    def create_actors(self):
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
            density=1e6
        )

        self.prism = self._actor_manager.add_from_usd_file(
            name='prism',
            asset_path="usb.usd",
            pose=usb_pose
        )
    
    def _reset_actors(self):
        start_slot_pose = Pose([0.4, 0.0, self.start_slot.get_pose()[2]], [1, 0, 0, 0])
        target_offset = self.create_noise([0.005, 0.005, 0.0])
        target_slot_pose = Pose([0.58, 0.0, self.slot.get_pose()[2]], [1, 0, 0, 0]).add_offset(target_offset)

        self.start_slot.set_pose(start_slot_pose)
        self.slot.set_pose(target_slot_pose)
        self.prism.set_pose(self._usb_pose_in_slot(start_slot_pose))

    def pre_move(self):
        self.delay(10)

        self.move(self.atom.open_gripper(0.5))
        grasp_rotate = self.rng.uniform(-np.pi/18, np.pi/18)
        target_pose = self.prism.get_pose().add_bias([0, 0, USB_GRASP_HEIGHT]).add_rotation([0, grasp_rotate, 0])
        target_mat = target_pose.to_transformation_matrix()
        cpose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0]
        )
        cid = self.prism.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.prism,
            contact_point_id=cid,
            is_close=False
        ))
        self.move(self.atom.close_gripper())
        self.move(self.atom.move_by_displacement(z=0.03))

        self.target_pose = self._usb_pose_in_slot(self.slot.get_pose())
        self.opening_pose = self.slot.get_pose().add_bias([0.0, 0.0, SLOT_HEIGHT])
        self.pre_insert_pose = self.opening_pose.add_bias([0.0, 0.0, PRE_INSERT_CLEARANCE])
        noise = self.create_noise([0.005, 0.005, 0.0])
        self.noise_pose = self.pre_insert_pose.add_offset(noise)
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=self.noise_pose,
            pre_dis=0.02,
            dis=0.008,
            is_open=False
        ))

    def _play_once(self):
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=self.pre_insert_pose,
            pre_dis=0.01,
            dis=0.002,
            is_open=False
        ), time_dilation_factor=0.5)
        self.move(self.atom.move_by_displacement(
            z=-(PRE_INSERT_CLEARANCE + USB_PLUG_HEIGHT),
            xyz_coord='world'
        ), time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
        self.delay(20, is_save=True)

    def _get_success_diagnostics(self, xy_threshold=0.002, z_threshold=0.003):
        prism_pose = self.prism.get_pose().rebase(self.target_pose)
        ee_pose = self._robot_manager.get_ee_pose()
        prism_z_axis = prism_pose.to_transformation_matrix()[:3, 2]
        target_z_axis = np.array([0, 0, 1])
        z_axis_dot = float(np.dot(prism_z_axis, target_z_axis))
        tilt_angle_deg = float(np.degrees(np.arccos(np.clip(z_axis_dot, -1.0, 1.0))))
        xy_error = float(np.linalg.norm(prism_pose.p[:2]))
        pos_error = float(np.linalg.norm(prism_pose.p))

        return {
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
            'angle_ok': bool(z_axis_dot > 0.965), # 15°
        }

    def _print_failure_diagnostics(self, diagnostics):
        print(
            "\n[insert_USB failure diagnostics]\n"
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
        diagnostics = self._get_success_diagnostics(xy_threshold, z_threshold)
        self.metadata['rel_pose'] = diagnostics['rel_pose']
        self.metadata['success_diagnostics'] = diagnostics
        success = diagnostics['xy_ok'] and diagnostics['z_ok'] \
            and diagnostics['ee_z_ok'] and diagnostics['angle_ok']
        if not success and self.mode in ['collect', 'eval_test'] \
                and not self.metadata.get('failure_diagnostics_printed', False):
            self._print_failure_diagnostics(diagnostics)
            self.metadata['failure_diagnostics_printed'] = True
        return success
