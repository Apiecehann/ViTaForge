from ._base_task import *
import numpy as np


BLOCK_HEIGHT = 0.0400
BLOCK_XY_NOISE = (0.020, 0.020, 0.0)
GRASP_ROTATE_NOISE = np.deg2rad(10.0)
GRASP_HEIGHT = BLOCK_HEIGHT * 0.5
GRASP_HEIGHT_NOISE = 0.003
LIFT_HEIGHT = 0.1000
SUCCESS_MIN_LIFT = 0.0500
SUCCESS_MAX_LIFT = 0.1500

BLOCK_BASE_POSES = (
    Pose([0.44, 0.0, 0.002], [1, 0, 0, 0]),
    Pose([0.36, -0.09, 0.002], [1, 0, 0, 0]),
    Pose([0.36, 0.09, 0.002], [1, 0, 0, 0]),
    Pose([0.44, -0.12, 0.002], [1, 0, 0, 0]),
    Pose([0.44, 0.12, 0.002], [1, 0, 0, 0]),
    Pose([0.54, -0.08, 0.002], [1, 0, 0, 0]),
    Pose([0.53, 0.08, 0.002], [1, 0, 0, 0]),
    # Pose([0.53, 0.12, 0.002], [1, 0, 0, 0]),
)

BLOCK_SPECS = (
    {
        "name": "block_blue_half_cylinder",
        "shape": "half_cylinder",
        "asset_path": "block_blue_half_cylinder.usd",
    },
    {
        "name": "block_blue_quarter_cylinder",
        "shape": "quarter_cylinder",
        "asset_path": "block_blue_quarter_cylinder.usd",
    },
    {
        "name": "block_blue_star_prism",
        "shape": "star_prism",
        "asset_path": "block_blue_star_prism.usd",
    },
    {
        "name": "block_red_ellipse_cylinder",
        "shape": "ellipse_cylinder",
        "asset_path": "block_red_ellipse_cylinder.usd",
    },
    {
        "name": "block_red_hexagonal_prism",
        "shape": "hexagonal_prism",
        "asset_path": "block_red_hexagonal_prism.usd",
    },
    {
        "name": "block_yellow_cylinder",
        "shape": "cylinder",
        "asset_path": "block_yellow_cylinder.usd",
    },
    {
        "name": "block_yellow_triangular_prism",
        "shape": "triangular_prism",
        "asset_path": "block_yellow_triangular_prism.usd",
    },
    # {
    #     "name": "block_blue_cube",
    #     "shape": "cube",
    #     "asset_path": "block_blue_cube.usd",
    # },
)
TARGET_BLOCKS = tuple(spec["name"] for spec in BLOCK_SPECS)
DEFAULT_TARGET_BLOCK = "block_yellow_cylinder"
TASK_INSTRUCTION = "Grasp the yellow cylinder from the clutter and lift it up."
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
    target_block: str = DEFAULT_TARGET_BLOCK
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.87, 0.0, 0.20),
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
    step_lim = 200


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        if cfg.target_block not in TARGET_BLOCKS:
            raise ValueError(f"target_block must be one of {TARGET_BLOCKS}, got {cfg.target_block!r}")

        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        self.target_block_name = cfg.target_block
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        cfg.robot.robot.init_state.joint_pos.update(TASK_INITIAL_JOINT_POS)
        return cfg

    def _sample_distinct_block_poses(self):
        pose_indices = self.rng.choice(len(BLOCK_BASE_POSES), size=len(BLOCK_SPECS), replace=False)
        return {
            spec["name"]: (int(pose_index), BLOCK_BASE_POSES[int(pose_index)])
            for spec, pose_index in zip(BLOCK_SPECS, pose_indices)
        }

    def create_actors(self):
        self.initial_block_pose_assignments = self._sample_distinct_block_poses()
        self.blocks = {}
        self.block_specs = {spec["name"]: spec for spec in BLOCK_SPECS}

        for spec in BLOCK_SPECS:
            _, initial_pose = self.initial_block_pose_assignments[spec["name"]]
            self.blocks[spec["name"]] = self._actor_manager.add_from_usd_file(
                name=spec["name"],
                asset_path=spec["asset_path"],
                pose=initial_pose,
                density=1e3,
            )

    def _reset_actors(self):
        self.block_poses = {}
        self.block_base_pose_indices = {}
        self.block_xy_noises = {}

        for name, (pose_index, base_pose) in self.initial_block_pose_assignments.items():
            offset = self.create_noise(list(BLOCK_XY_NOISE))
            pose = base_pose.add_offset(offset)
            self.blocks[name].set_pose(pose)
            self.block_poses[name] = pose
            self.block_base_pose_indices[name] = pose_index
            self.block_xy_noises[name] = offset.p.tolist()

        self.target_block = self.blocks[self.target_block_name]
        self.target_initial_pose = self.block_poses[self.target_block_name]
        self.metadata["target_shape"] = self.block_specs[self.target_block_name]["shape"]
        self.metadata["target_block_name"] = self.target_block_name
        self.metadata["block_base_pose_indices"] = self.block_base_pose_indices
        self.metadata["block_xy_noises"] = self.block_xy_noises
        self.metadata["block_poses"] = {
            name: pose.tolist() for name, pose in self.block_poses.items()
        }

    def pre_move(self):
        self.delay(10)
        self.move(self.atom.open_gripper(0.6), tag=f"open_gripper_for_{self.target_block_name}")

        target_pose = self.target_block.get_pose()
        grasp_rotate = self.rng.uniform(-GRASP_ROTATE_NOISE, GRASP_ROTATE_NOISE)
        grasp_height = GRASP_HEIGHT + self.rng.uniform(-GRASP_HEIGHT_NOISE, GRASP_HEIGHT_NOISE)
        grasp_target_pose = target_pose.add_bias([0.0, 0.0, grasp_height]).add_rotation([0.0, grasp_rotate, 0.0])
        target_mat = grasp_target_pose.to_transformation_matrix()
        grasp_pose = construct_grasp_pose(
            grasp_target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        )
        contact_point_id = self.target_block.register_point(grasp_pose, type="contact")

        self.move(
            self.atom.grasp_actor(
                self.target_block,
                contact_point_id=contact_point_id,
                is_close=False,
                pre_dis=0.05,
            ),
            tag=f"approach_{self.target_block_name}",
        )
        self.move(self.atom.close_gripper(), tag=f"close_{self.target_block_name}")

        self.metadata["grasp_rotate_rad"] = float(grasp_rotate)
        self.metadata["grasp_rotate_deg"] = float(np.rad2deg(grasp_rotate))
        self.metadata["grasp_height"] = float(grasp_height)
        self.metadata["grasp_pose"] = grasp_pose.tolist()

    def _play_once(self):
        self.move(self.atom.move_by_displacement(z=LIFT_HEIGHT), tag=f"lift_{self.target_block_name}")
        self.delay(20, is_save=False)

    def _get_success_diagnostics(self):
        target_pose = self.target_block.get_pose()
        lifted_height = target_pose.p[2] - self.target_initial_pose.p[2]
        height_ok = bool(SUCCESS_MIN_LIFT <= lifted_height <= SUCCESS_MAX_LIFT)

        return {
            "target_block_name": self.target_block_name,
            "target_initial_pose": self.target_initial_pose.tolist(),
            "target_final_pose": target_pose.tolist(),
            "lifted_height": float(lifted_height),
            "success_min_lift": float(SUCCESS_MIN_LIFT),
            "success_max_lift": float(SUCCESS_MAX_LIFT),
            "height_ok": height_ok,
        }

    def check_success(self):
        diagnostics = self._get_success_diagnostics()
        self.metadata["success_diagnostics"] = diagnostics
        return diagnostics["height_ok"]
