from ._base_task import *
import os
import numpy as np


# Set TaskCfg.target_block, or TARGET_BLOCK, to select the block to grasp.
TARGET_BLOCKS = ("cube", "half_cylinder", "hexagon")
DEFAULT_TARGET_BLOCK = os.environ.get("TARGET_BLOCK", "half_cylinder")

BOX_SIZE = 0.1500
BOX_WALL_THICKNESS = 0.0060
BLOCK_HEIGHT = 0.0400

BOX_BASE_POSE = Pose([0.45, -0.12, 0.002], [1, 0, 0, 0])
TARGET_BLOCK_BASE_POSE = Pose([0.45, 0.15, 0.002], [1, 0, 0, 0])
DISTRACTOR_BLOCK_BASE_POSES = (
    Pose([0.37, 0.08, 0.002], [1, 0, 0, 0]),
    Pose([0.36, 0.24, 0.002], [1, 0, 0, 0]),
    Pose([0.54, 0.08, 0.002], [1, 0, 0, 0]),
    Pose([0.55, 0.24, 0.002], [1, 0, 0, 0]),
)
LIFT_TARGET_POSE = Pose([0.45, 0.02, 0.162], [1, 0, 0, 0])

BOX_XY_NOISE = (0.020, 0.020, 0.0)
BLOCK_XY_NOISE = (0.020, 0.020, 0.0)
GRASP_ROTATE_NOISE = np.deg2rad(10.0)
GRASP_HEIGHT = BLOCK_HEIGHT * 0.5
GRASP_HEIGHT_NOISE = 0.003
PRE_INSERT_CLEARANCE = 0.002
INSERT_DEPTH = 0.010
PRE_PLACE_DISTANCE = 0.020
HOLE_POSITION_TOLERANCE = 0.012
CONTAINMENT_TOLERANCE = 1e-4

# These are the hole cross-section centroids in the box local frame.  The
# half-cylinder reference is its semicircle area centroid, matching that
# asset's bottom-face origin rather than the full-circle center.
BLOCK_SPECS = {
    "cube": {
        "actor_name": "block_wooden_cube",
        "asset_path": "block_wooden_cube.usd",
        "hole_center": np.array([-0.035000, 0.030000]),
    },
    "half_cylinder": {
        "actor_name": "block_blue_half_cylinder",
        "asset_path": "block_blue_half_cylinder.usd",
        "hole_center": np.array([0.034000, -0.018117]),
    },
    "hexagon": {
        "actor_name": "block_yellow_hexagonal_prism",
        "asset_path": "block_yellow_hexagonal_prism.usd",
        "hole_center": np.array([-0.026000, -0.036000]),
    },
}

INNER_X_MIN = -BOX_SIZE * 0.5 + BOX_WALL_THICKNESS
INNER_X_MAX = BOX_SIZE * 0.5
INNER_Y_MIN = -BOX_SIZE * 0.5 + BOX_WALL_THICKNESS
INNER_Y_MAX = BOX_SIZE * 0.5 - BOX_WALL_THICKNESS
INNER_Z_MIN = BOX_WALL_THICKNESS
INNER_Z_MAX = BOX_SIZE - BOX_WALL_THICKNESS


@configclass
class TaskCfg(BaseTaskCfg):
    # Set TARGET_BLOCK, or change this variable, to choose a target.
    target_block: str = DEFAULT_TARGET_BLOCK
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.84, -0.02, 0.26), rot=(0.593538, 0.415599, 0.395306, 0.564556), convention="opengl"),
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
    step_lim = 300


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        if cfg.target_block not in TARGET_BLOCKS:
            raise ValueError(f"target_block must be one of {TARGET_BLOCKS}, got {cfg.target_block!r}")

        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        self.target_block_key = cfg.target_block
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        self.wooden_box = self._actor_manager.add_from_usd_file(
            name="wooden_box_three_holes_open_x",
            asset_path="wooden_box_three_holes_open_x.usd",
            pose=BOX_BASE_POSE,
            density=1e6,
        )
        # UIPC validates the scene before the first reset.  Give every block a
        # separate valid pose here; _reset_actors then applies the requested
        # randomized episode layout.
        initial_block_poses = {self.target_block_key: TARGET_BLOCK_BASE_POSE}
        initial_distractor_keys = [key for key in TARGET_BLOCKS if key != self.target_block_key]
        for key, pose in zip(initial_distractor_keys, DISTRACTOR_BLOCK_BASE_POSES):
            initial_block_poses[key] = pose

        self.blocks = {}
        for key, spec in BLOCK_SPECS.items():
            self.blocks[key] = self._actor_manager.add_from_usd_file(
                name=spec["actor_name"],
                asset_path=spec["asset_path"],
                pose=initial_block_poses[key],
                density=1e3,
            )

    def _reset_actors(self):
        box_offset = self.create_noise(list(BOX_XY_NOISE))
        box_pose = BOX_BASE_POSE.add_offset(box_offset)
        self.wooden_box.set_pose(box_pose)

        target_offset = self.create_noise(list(BLOCK_XY_NOISE))
        target_pose = TARGET_BLOCK_BASE_POSE.add_offset(target_offset)
        self.selected_block = self.blocks[self.target_block_key]
        self.selected_block.set_pose(target_pose)
        self.selected_hole_center = BLOCK_SPECS[self.target_block_key]["hole_center"]

        distractor_keys = [key for key in TARGET_BLOCKS if key != self.target_block_key]
        distractor_pose_indices = self.rng.choice(
            len(DISTRACTOR_BLOCK_BASE_POSES),
            size=2,
            replace=False,
        )
        self.distractor_poses = {}
        for key, pose_index in zip(distractor_keys, distractor_pose_indices):
            distractor_offset = self.create_noise(list(BLOCK_XY_NOISE))
            distractor_pose = DISTRACTOR_BLOCK_BASE_POSES[int(pose_index)].add_offset(
                distractor_offset
            )
            self.blocks[key].set_pose(distractor_pose)
            self.distractor_poses[key] = distractor_pose

        self.metadata["target_block"] = self.target_block_key
        self.metadata["target_hole_center_xy"] = self.selected_hole_center.tolist()
        self.metadata["box_xy_noise"] = box_offset.p.tolist()
        self.metadata["target_block_xy_noise"] = target_offset.p.tolist()
        self.metadata["wooden_box_pose"] = box_pose.tolist()
        self.metadata["target_block_pose"] = target_pose.tolist()
        self.metadata["distractor_block_poses"] = {
            key: pose.tolist() for key, pose in self.distractor_poses.items()
        }

    def pre_move(self):
        self.delay(10)
        self.move(self.atom.open_gripper(0.5), tag=f"open_gripper_for_{self.target_block_key}")

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
            tag=f"insert_{self.target_block_key}_into_three_hole_box",
            time_dilation_factor=0.5,
        )
        self.move(self.atom.open_gripper(0.5), tag=f"release_{self.target_block_key}")
        self.delay(20)

    def _get_success_diagnostics(self):
        box_pose = self.wooden_box.get_pose()
        selected_pose = self.selected_block.get_pose()
        selected_in_box = selected_pose.rebase(box_pose)
        # 检查完整的物块表面，而不是只检查原点，避免物块还卡在孔口时被误判成功。
        vertices_world = np.asarray(self.selected_block.vertices)
        vertices_in_box = (vertices_world - box_pose.p) @ box_pose.R
        bounds_min = np.min(vertices_in_box, axis=0)
        bounds_max = np.max(vertices_in_box, axis=0)

        lower_bound = np.array([INNER_X_MIN, INNER_Y_MIN, INNER_Z_MIN])
        upper_bound = np.array([INNER_X_MAX, INNER_Y_MAX, INNER_Z_MAX])
        axis_inside = (bounds_min >= lower_bound - CONTAINMENT_TOLERANCE) & (
            bounds_max <= upper_bound + CONTAINMENT_TOLERANCE
        )
        fully_inside_box = bool(np.all(axis_inside))

        # 孔中心偏差仅用于离线分析，不参与成功判定。
        hole_center_offset_xy = selected_in_box.p[:2] - self.selected_hole_center
        hole_center_offset_xy_norm = float(np.linalg.norm(hole_center_offset_xy))

        return {
            "target_block": self.target_block_key,
            "wooden_box_pose": box_pose.tolist(),
            "selected_block_pose": selected_pose.tolist(),
            "selected_block_pose_in_box": selected_in_box.tolist(),
            "selected_block_xyz_in_box": selected_in_box.p.tolist(),
            "target_hole_center_xy": self.selected_hole_center.tolist(),
            "hole_center_offset_xy": hole_center_offset_xy.tolist(),
            "hole_center_offset_xy_norm": hole_center_offset_xy_norm,
            "hole_position_tolerance": float(HOLE_POSITION_TOLERANCE),
            "selected_block_bounds_in_box": {
                "min": bounds_min.tolist(),
                "max": bounds_max.tolist(),
            },
            "inner_x_range": [float(INNER_X_MIN), float(INNER_X_MAX)],
            "inner_y_range": [float(INNER_Y_MIN), float(INNER_Y_MAX)],
            "inner_z_range": [float(INNER_Z_MIN), float(INNER_Z_MAX)],
            "containment_tolerance": float(CONTAINMENT_TOLERANCE),
            "inside_x": bool(axis_inside[0]),
            "inside_y": bool(axis_inside[1]),
            "inside_z": bool(axis_inside[2]),
            "fully_inside_box": fully_inside_box,
        }

    def check_success(self):
        diagnostics = self._get_success_diagnostics()
        self.metadata["success_diagnostics"] = diagnostics
        return diagnostics["fully_inside_box"]
