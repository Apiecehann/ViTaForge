from ._base_task import *
import numpy as np


TASK_INSTRUCTION = "Grasp the soft blue cube, lift it by 5 cm, then open the gripper to release it."

BLOCK_HEIGHT = 0.0400
SOFT_CUBE_POSE = Pose([0.4, 0.0, 0.002], [1, 0, 0, 0])
GRASP_HEIGHT = BLOCK_HEIGHT * 0.5
GRASP_HEIGHT_NOISE = 0.002
GRASP_ROTATE_NOISE = np.deg2rad(5.0)
LIFT_HEIGHT = 0.0500
SUCCESS_MIN_LIFT = 0.0450
POST_RELEASE_DELAY = 30

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
    step_lim = 220
    adaptive_grasp_depth_threshold = 28.4
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.84, 0.0, 0.20),
                rot=(0.579228, 0.405580, 0.405580, 0.579228),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5,
                focus_distance=1.0,
                horizontal_aperture=2.4,
                clipping_range=(0.1, 100.0),
            ),
            width=480,
            height=270,
            update_period=1 / 120,
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


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        cfg.robot.robot.init_state.joint_pos.update(TASK_INITIAL_JOINT_POS)
        return cfg

    def create_actors(self):
        self.soft_cube = self._actor_manager.add_from_usd_file(
            name="soft_blue_cube",
            asset_path="block_blue_cube.usd",
            pose=SOFT_CUBE_POSE,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=0.02,
                poisson_rate=0.45,
            ),
            density=300,
        )

    def _reset_actors(self):
        self.soft_cube.set_pose(SOFT_CUBE_POSE)
        self.initial_pose = SOFT_CUBE_POSE
        self.pre_release_pose = None
        self.release_done = False
        self.metadata["soft_cube_initial_pose"] = self.initial_pose.tolist()

    def pre_move(self):
        self.delay(10)
        self.move(self.atom.open_gripper(0.5), tag="open_gripper_for_soft_cube")

        cube_pose = self.soft_cube.get_pose()
        grasp_rotate = self.rng.uniform(-GRASP_ROTATE_NOISE, GRASP_ROTATE_NOISE)
        grasp_height = GRASP_HEIGHT + self.rng.uniform(-GRASP_HEIGHT_NOISE, GRASP_HEIGHT_NOISE)
        grasp_target_pose = cube_pose.add_bias([0.0, 0.0, grasp_height]).add_rotation([0.0, grasp_rotate, 0.0])
        target_mat = grasp_target_pose.to_transformation_matrix()
        grasp_pose = construct_grasp_pose(
            grasp_target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        )
        contact_point_id = self.soft_cube.register_point(grasp_pose, type="contact")

        self.move(
            self.atom.grasp_actor(
                self.soft_cube,
                contact_point_id=contact_point_id,
                is_close=False,
                pre_dis=0.05,
                dis=0.0,
            ),
            tag="approach_soft_cube",
        )
        self.move(self.atom.close_gripper(), tag="close_soft_cube")

        self.metadata["grasp_rotate_rad"] = float(grasp_rotate)
        self.metadata["grasp_rotate_deg"] = float(np.rad2deg(grasp_rotate))
        self.metadata["grasp_height"] = float(grasp_height)
        self.metadata["grasp_pose"] = grasp_pose.tolist()

    def _play_once(self):
        self.move(
            self.atom.move_by_displacement(z=LIFT_HEIGHT, xyz_coord="world"),
            tag="lift_soft_cube_5cm",
            time_dilation_factor=0.5,
        )
        self.pre_release_pose = self.soft_cube.get_pose()
        self.metadata["soft_cube_pre_release_pose"] = self.pre_release_pose.tolist()
        self.metadata["lift_height_before_release"] = float(self.pre_release_pose.p[2] - self.initial_pose.p[2])

        self.move(self.atom.open_gripper(0.8), tag="release_soft_cube")
        self.release_done = True
        self.delay(POST_RELEASE_DELAY, is_save=False)

    def _get_success_diagnostics(self):
        lift_height = None
        if self.pre_release_pose is not None:
            lift_height = float(self.pre_release_pose.p[2] - self.initial_pose.p[2])

        return {
            "soft_cube_initial_pose": self.initial_pose.tolist(),
            "soft_cube_pre_release_pose": None if self.pre_release_pose is None else self.pre_release_pose.tolist(),
            "soft_cube_final_pose": self.soft_cube.get_pose().tolist(),
            "lift_height_before_release": lift_height,
            "success_min_lift": float(SUCCESS_MIN_LIFT),
            "release_done": bool(self.release_done),
            "height_ok": bool(lift_height is not None and lift_height >= SUCCESS_MIN_LIFT),
        }

    def check_success(self):
        diagnostics = self._get_success_diagnostics()
        self.metadata["success_diagnostics"] = diagnostics
        return diagnostics["height_ok"] and diagnostics["release_done"]
