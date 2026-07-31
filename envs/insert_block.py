from ._base_task import *
import os
import numpy as np


# Set TaskCfg.target_block, or TARGET_BLOCK, to select the block to grasp.
TARGET_BLOCKS = ("cube", "half_cylinder", "hexagon")
DEFAULT_TARGET_BLOCK = os.environ.get("TARGET_BLOCK", "cube")
TASK_INSTRUCTION = f"Insert the {DEFAULT_TARGET_BLOCK.replace('_', ' ')} into the matching hole in the yellow box."

BOX_SIZE = 0.1500
BOX_WALL_THICKNESS = 0.0060
BLOCK_HEIGHT = 0.0400

BOX_BASE_POSE = Pose([0.48, -0.12, 0.002], [1, 0, 0, 0])
BLOCK_BASE_POSES = (
    Pose([0.45, 0.15, 0.002], [1, 0, 0, 0]),
    Pose([0.37, 0.08, 0.002], [1, 0, 0, 0]),
    Pose([0.36, 0.24, 0.002], [1, 0, 0, 0]),
    Pose([0.53, 0.08, 0.002], [1, 0, 0, 0]),
    Pose([0.54, 0.24, 0.002], [1, 0, 0, 0]),
)
LIFT_TARGET_POSE = Pose([0.45, 0.02, 0.162], [1, 0, 0, 0])

BOX_XY_NOISE = (0.010, 0.010, 0.0)
BLOCK_XY_NOISE = (0.020, 0.020, 0.0)
GRASP_ROTATE_NOISE = np.deg2rad(10.0)
GRASP_HEIGHT = BLOCK_HEIGHT * 0.5
GRASP_HEIGHT_NOISE = 0.003
POST_GRASP_LIFT_HEIGHT = 0.030
PRE_INSERT_CLEARANCE = 0.002
INSERT_DEPTH = 0.010
PRE_PLACE_DISTANCE = 0.020

# The box mesh has 6 mm thick walls and bottom/top plates.  Success is based
# only on the selected block origin in this local interior volume.
INNER_X_MIN = -BOX_SIZE * 0.5 + BOX_WALL_THICKNESS
# The +X side of box_with_holes_yellow is open.
INNER_X_MAX = BOX_SIZE * 0.5
INNER_Y_MIN = -BOX_SIZE * 0.5 + BOX_WALL_THICKNESS
INNER_Y_MAX = BOX_SIZE * 0.5 - BOX_WALL_THICKNESS
INNER_Z_MIN = BOX_WALL_THICKNESS
INNER_Z_MAX = BOX_SIZE - BOX_WALL_THICKNESS

# These are the hole cross-section centroids in the box local frame.  The
# half-cylinder centre follows that asset's semicircle area centroid.
BLOCK_SPECS = {
    "cube": {
        "actor_name": "block_blue_cube",
        "asset_path": "block_blue_cube.usd",
        "hole_center": np.array([-0.035000, 0.030000]),
    },
    "half_cylinder": {
        "actor_name": "block_blue_half_cylinder",
        "asset_path": "block_blue_half_cylinder.usd",
        "hole_center": np.array([0.034000, -0.018117]),
    },
    "hexagon": {
        "actor_name": "block_red_hexagonal_prism",
        "asset_path": "block_red_hexagonal_prism.usd",
        "hole_center": np.array([-0.026000, -0.036000]),
    },
}

TASK_INITIAL_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": 0.0,
    "panda_joint3": 0.0,
    "panda_joint4": -2.46,
    "panda_joint5": 0.0,
    "panda_joint6": 2.5,
    "panda_joint7": 0.741,
    "panda_finger.*": 0.02,
}


@configclass
class TaskCfg(BaseTaskCfg):
    # Set TARGET_BLOCK, or change this variable, to choose a target.
    target_block: str = DEFAULT_TARGET_BLOCK
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            # Original orientation pitched 10 degrees further downward around
            # the camera-local X axis.
            offset=CameraCfg.OffsetCfg(
                pos=(0.8, -0.02, 0.26),
                rot=(0.627501, 0.362287, 0.344597, 0.596861),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.6, focus_distance=1.0, horizontal_aperture=2.4, clipping_range=(0.1, 100.0)
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
    step_lim = 400


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        if cfg.target_block not in TARGET_BLOCKS:
            raise ValueError(f"target_block must be one of {TARGET_BLOCKS}, got {cfg.target_block!r}")

        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        self.target_block_key = cfg.target_block
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        cfg.robot.robot.init_state.joint_pos.update(TASK_INITIAL_JOINT_POS)
        return cfg

    def _sample_distinct_block_poses(self):
        pose_indices = self.rng.choice(len(BLOCK_BASE_POSES), size=len(TARGET_BLOCKS), replace=False)
        return {
            key: (int(pose_index), BLOCK_BASE_POSES[int(pose_index)])
            for key, pose_index in zip(TARGET_BLOCKS, pose_indices)
        }

    def create_actors(self):
        self.wooden_box = self._actor_manager.add_from_usd_file(
            name="box_with_holes_yellow",
            asset_path="box_with_holes_yellow.usd",
            pose=BOX_BASE_POSE,
            density=1e6,
        )

        # UIPC validates the scene before the first reset.  Sample three
        # distinct valid poses before creating the three block actors, then
        # reuse this base-pose assignment on every reset.
        self.initial_block_pose_assignments = self._sample_distinct_block_poses()
        self.blocks = {}
        for key, spec in BLOCK_SPECS.items():
            _, initial_pose = self.initial_block_pose_assignments[key]
            self.blocks[key] = self._actor_manager.add_from_usd_file(
                name=spec["actor_name"],
                asset_path=spec["asset_path"],
                pose=initial_pose,
                density=1e3,
            )

    def _reset_actors(self):
        box_offset = self.create_noise(list(BOX_XY_NOISE))
        box_pose = BOX_BASE_POSE.add_offset(box_offset)
        self.wooden_box.set_pose(box_pose)

        self.block_poses = {}
        self.block_base_pose_indices = {}
        self.block_xy_noises = {}
        for key, (pose_index, base_pose) in self.initial_block_pose_assignments.items():
            block_offset = self.create_noise(list(BLOCK_XY_NOISE))
            block_pose = base_pose.add_offset(block_offset)
            self.blocks[key].set_pose(block_pose)
            self.block_poses[key] = block_pose
            self.block_base_pose_indices[key] = pose_index
            self.block_xy_noises[key] = block_offset.p.tolist()

        self.selected_block = self.blocks[self.target_block_key]
        self.selected_hole_center = BLOCK_SPECS[self.target_block_key]["hole_center"]

        self.metadata["target_block"] = self.target_block_key
        self.metadata["target_hole_center_xy"] = self.selected_hole_center.tolist()
        self.metadata["box_xy_noise"] = box_offset.p.tolist()
        self.metadata["wooden_box_pose"] = box_pose.tolist()
        self.metadata["block_base_pose_indices"] = self.block_base_pose_indices
        self.metadata["block_xy_noises"] = self.block_xy_noises
        self.metadata["block_poses"] = {
            key: pose.tolist() for key, pose in self.block_poses.items()
        }

    def pre_move(self):
        self.delay(10)
        self.move(self.atom.open_gripper(0.6), tag=f"open_gripper_for_{self.target_block_key}")

        selected_pose = self.selected_block.get_pose()
        grasp_rotate = self.rng.uniform(-GRASP_ROTATE_NOISE, GRASP_ROTATE_NOISE)
        grasp_height = GRASP_HEIGHT + self.rng.uniform(-GRASP_HEIGHT_NOISE, GRASP_HEIGHT_NOISE)
        grasp_target_pose = selected_pose.add_bias([0.0, 0.0, grasp_height]).add_rotation([0.0, grasp_rotate, 0.0])
        target_mat = grasp_target_pose.to_transformation_matrix()
        grasp_pose = construct_grasp_pose(
            grasp_target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        )
        contact_point_id = self.selected_block.register_point(grasp_pose, type="contact")

        self.move(
            self.atom.grasp_actor(
                self.selected_block,
                contact_point_id=contact_point_id,
                is_close=False,
                pre_dis=PRE_PLACE_DISTANCE,
                dis=0.0,
            ),
            tag=f"approach_{self.target_block_key}",
        )
        self.move(self.atom.close_gripper(), tag=f"close_{self.target_block_key}")
        self.move(
            self.atom.move_by_displacement(
                z=POST_GRASP_LIFT_HEIGHT,
                xyz_coord="world",
            ),
            tag=f"lift_{self.target_block_key}_after_grasp",
        )

        self.metadata["grasp_rotate_rad"] = float(grasp_rotate)
        self.metadata["grasp_rotate_deg"] = float(np.rad2deg(grasp_rotate))
        self.metadata["grasp_height"] = float(grasp_height)
        self.metadata["grasp_pose"] = grasp_pose.tolist()

    def _sample_pre_insert_pose(self):
        box_pose = self.wooden_box.get_pose()
        pre_insert_pose = box_pose.add_bias([
            float(self.selected_hole_center[0]),
            float(self.selected_hole_center[1]),
            BOX_SIZE + PRE_INSERT_CLEARANCE,
        ])
        self.metadata["pre_insert_pose"] = pre_insert_pose.tolist()
        return pre_insert_pose

    def _get_success_diagnostics(self):
        box_pose = self.wooden_box.get_pose()
        selected_pose = self.selected_block.get_pose()
        selected_in_box = selected_pose.rebase(box_pose)
        selected_xyz_in_box = selected_in_box.p

        inside_x = bool(INNER_X_MIN <= selected_xyz_in_box[0] <= INNER_X_MAX)
        inside_y = bool(INNER_Y_MIN <= selected_xyz_in_box[1] <= INNER_Y_MAX)
        inside_z = bool(INNER_Z_MIN <= selected_xyz_in_box[2] <= INNER_Z_MAX)
        origin_inside_box = bool(inside_x and inside_y and inside_z)

        return {
            "target_block": self.target_block_key,
            "wooden_box_pose": box_pose.tolist(),
            "selected_block_pose": selected_pose.tolist(),
            "selected_block_pose_in_box": selected_in_box.tolist(),
            "selected_block_xyz_in_box": selected_xyz_in_box.tolist(),
            "inner_x_range": [float(INNER_X_MIN), float(INNER_X_MAX)],
            "inner_y_range": [float(INNER_Y_MIN), float(INNER_Y_MAX)],
            "inner_z_range": [float(INNER_Z_MIN), float(INNER_Z_MAX)],
            "inside_x": inside_x,
            "inside_y": inside_y,
            "inside_z": inside_z,
            "origin_inside_box": origin_inside_box,
        }

    def _play_once(self):
        self.move(
            self.atom.place_actor(
                self.selected_block,
                target_pose=LIFT_TARGET_POSE,
                pre_dis=PRE_PLACE_DISTANCE,
                dis=0.0,
                is_open=False,
            ),
            tag=f"lift_{self.target_block_key}",
            time_dilation_factor=0.5,
        )

        pre_insert_pose = self._sample_pre_insert_pose()
        self.move(
            self.atom.place_actor(
                self.selected_block,
                target_pose=pre_insert_pose,
                pre_dis=PRE_PLACE_DISTANCE,
                dis=0.0,
                is_open=False,
            ),
            tag=f"move_{self.target_block_key}_to_pre_insert",
            time_dilation_factor=0.5,
        )
        self.move(
            self.atom.move_by_displacement(z=-INSERT_DEPTH),
            tag=f"insert_{self.target_block_key}_into_box",
            time_dilation_factor=0.5,
        )
        self.move(
            self.atom.open_gripper(0.6),
            tag=f"release_{self.target_block_key}",
            delay=False,
        )

    def check_success(self):
        diagnostics = self._get_success_diagnostics()
        self.metadata["success_diagnostics"] = diagnostics
        return diagnostics["origin_inside_box"]
