from ._base_task import *
import numpy as np


TASK_INSTRUCTION = "Fold the patterned towel by grasping one side, lifting it, moving it over the center line, and releasing it."

TOWEL_LENGTH = 0.40
TOWEL_WIDTH = 0.25
TABLE_TOP_Z = 0.0025
TOWEL_POSE = Pose([0.42, 0.0, TABLE_TOP_Z + 0.006], [1, 0, 0, 0])
GRASP_Y_NOISE = 0.025
INIT_LIFT_HEIGHT = 0.060
INIT_LIFT_X_WIDTH = 0.090
INIT_LIFT_Y_RADIUS = 0.065
INIT_CONTACT_Z = TABLE_TOP_Z + 0.0015
GRASP_Z_BIAS = -0.004
GRASP_PRE_DIS = 0.060
GRIPPER_CLOSE_PERCENT = -0.1
POST_GRASP_HOLD_DELAY = 30
LIFT_HEIGHT = 0.070
FOLD_DISPLACEMENT_X = -0.21
FOLD_CLEARANCE_Z = 0.020
SETTLE_DELAY = 40

TASK_INITIAL_JOINT_POS = {
    "panda_joint1": -0.010809095,
    "panda_joint2": 0.096037410,
    "panda_joint3": 0.000734462,
    "panda_joint4": -2.433035851,
    "panda_joint5": 0.035354517,
    "panda_joint6": 2.500859022,
    "panda_joint7": 0.741,
    "panda_finger.*": 0.039,
}


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 340
    use_adaptive_grasp = False
    reset_time_limit = 500.0
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(1.08, 0.0, 0.38),
                rot=(0.579228, 0.405580, 0.405580, 0.579228),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.0,
                focus_distance=1.0,
                horizontal_aperture=2.8,
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
        cfg.use_adaptive_grasp = False
        cfg.adaptive_grasp_depth_threshold = None
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        cfg.uipc_sim.contact.d_hat = 0.0005
        cfg.uipc_sim.ground_height = TABLE_TOP_Z
        cfg.uipc_sim.newton.max_iter = 1024
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        cfg.robot.robot.init_state.joint_pos.update(TASK_INITIAL_JOINT_POS)
        return cfg

    def create_actors(self):
        self.towel = self._actor_manager.add_from_usd_file(
            name="patterned_towel",
            asset_path="cloth_task/patterned_towel.usda",
            pose=TOWEL_POSE,
            constitution_cfg=UipcObjectCfg.NeoHookeanShellCfg(
                youngs_modulus=0.01,
                poisson_rate=0.499,
                thickness=0.002,
                enable_bending=True,
                bending_stiffness=10.0,
                render_offset=(0.0, 0.0, 0.0),
            ),
            density=200,
        )

    def _reset_actors(self):
        self.towel.remove_animate()
        self.initial_vertices = None
        self.pre_release_vertices = None
        self.release_done = False
        self._flat_reset_vertices = self._get_towel_vertices().copy()
        self._lift_center_y = self.rng.uniform(-GRASP_Y_NOISE, GRASP_Y_NOISE)
        self._raised_grasp_point = None
        self._raised_vertex_mask = None
        self._apply_raised_towel_state(hold=False)
        self.metadata["towel_initial_pose"] = TOWEL_POSE.tolist()
        self.metadata["towel_init_mode"] = "right_edge_lift"

    def _get_towel_vertices(self):
        vertices = self.towel.vertices
        if vertices.shape[0] == 0:
            vertices = self.towel.init_vertex_pos.detach().cpu().numpy()
        return np.asarray(vertices, dtype=np.float64)

    def _make_raised_towel_vertices(self):
        vertices = self._flat_reset_vertices.copy()
        vertices[:, 2] = np.maximum(vertices[:, 2], INIT_CONTACT_Z)

        x = vertices[:, 0]
        y = vertices[:, 1]
        x_max = float(x.max())
        x_weight = np.clip(1.0 - (x_max - x) / INIT_LIFT_X_WIDTH, 0.0, 1.0)
        y_weight = np.exp(-((y - self._lift_center_y) ** 2) / (2.0 * INIT_LIFT_Y_RADIUS ** 2))
        lift_weight = x_weight * y_weight
        vertices[:, 2] += INIT_LIFT_HEIGHT * lift_weight

        mask = lift_weight > 0.35
        if int(mask.sum()) < 4:
            top_ids = np.argsort(lift_weight)[-4:]
            mask = np.zeros(vertices.shape[0], dtype=bool)
            mask[top_ids] = True

        lifted_vertices = vertices[mask]
        top_count = min(6, lifted_vertices.shape[0])
        top_vertices = lifted_vertices[np.argsort(lifted_vertices[:, 2])[-top_count:]]
        grasp_point = top_vertices.mean(axis=0)
        grasp_point[2] += GRASP_Z_BIAS
        return vertices, mask, grasp_point

    def _apply_raised_towel_state(self, hold: bool):
        vertices, mask, grasp_point = self._make_raised_towel_vertices()
        vertex_tensor = torch.tensor(vertices, dtype=torch.float64, device=self.towel.init_vertex_pos.device)
        self.towel.write_vertex_positions_to_sim(vertex_tensor)
        if hold:
            self.towel.set_vertex_targets(vertices, mask=mask)

        self._raised_grasp_point = grasp_point
        self._raised_vertex_mask = mask
        self.metadata["raised_towel"] = {
            "lift_center_y": float(self._lift_center_y),
            "lift_height": INIT_LIFT_HEIGHT,
            "held_vertex_count": int(mask.sum()),
            "grasp_point": grasp_point.tolist(),
        }

    def _make_lifted_grasp_pose(self) -> Pose:
        world_point = self._raised_grasp_point
        if world_point is None:
            _, _, world_point = self._make_raised_towel_vertices()
        return construct_grasp_pose(world_point, grasp_from=[0, 0, 1], camera_up=[1, 0, 0])

    def pre_move(self):
        self._apply_raised_towel_state(hold=True)
        self.delay(5, is_save=False)
        self.initial_vertices = self.towel.vertices.copy()

        grasp_pose = self._make_lifted_grasp_pose()
        pre_grasp_pose = grasp_pose.add_bias([0, 0, -GRASP_PRE_DIS])
        self.move(
            self.atom.move_to_pose(self.atom.robot.gripper_center_to_ee(pre_grasp_pose)),
            tag="approach_lifted_towel_edge",
        )
        self.move(self.atom.move_to_pose(self.atom.robot.gripper_center_to_ee(grasp_pose)), tag="touch_lifted_towel_edge")
        self.move(
            self.atom.close_gripper(GRIPPER_CLOSE_PERCENT, depth_threshold=None),
            tag="pinch_towel_edge",
        )
        self.towel.remove_animate()
        self.delay(POST_GRASP_HOLD_DELAY, is_save=False)

        self.metadata["grasp_pose"] = grasp_pose.tolist()

    def _play_once(self):
        self.move(
            self.atom.move_by_displacement(z=LIFT_HEIGHT, xyz_coord="world"),
            tag="lift_towel_edge",
            time_dilation_factor=0.5,
        )
        self.move(
            self.atom.move_by_displacement(x=FOLD_DISPLACEMENT_X, z=FOLD_CLEARANCE_Z, xyz_coord="world"),
            tag="fold_towel_over_center",
            time_dilation_factor=0.5,
        )
        self.move(
            self.atom.move_by_displacement(z=-(LIFT_HEIGHT + FOLD_CLEARANCE_Z - 0.010), xyz_coord="world"),
            tag="lower_folded_towel",
            time_dilation_factor=0.5,
        )
        self.pre_release_vertices = self.towel.vertices.copy()
        self.move(self.atom.open_gripper(0.8), tag="release_towel")
        self.release_done = True
        self.delay(SETTLE_DELAY, is_save=False)

    def _get_success_diagnostics(self):
        vertices = self.towel.vertices
        initial_vertices = self.initial_vertices if self.initial_vertices is not None else vertices

        initial_min_x = float(initial_vertices[:, 0].min())
        initial_max_x = float(initial_vertices[:, 0].max())
        current_min_x = float(vertices[:, 0].min())
        current_max_x = float(vertices[:, 0].max())
        initial_width = initial_max_x - initial_min_x
        current_width = current_max_x - current_min_x
        width_ratio = current_width / initial_width if initial_width > 1e-6 else 1.0

        return {
            "release_done": bool(self.release_done),
            "initial_x_range": [initial_min_x, initial_max_x],
            "current_x_range": [current_min_x, current_max_x],
            "initial_width": float(initial_width),
            "current_width": float(current_width),
            "width_ratio": float(width_ratio),
            "right_edge_reduction": float(initial_max_x - current_max_x),
            "max_height": float(vertices[:, 2].max()),
        }

    def check_success(self):
        diagnostics = self._get_success_diagnostics()
        self.metadata["success_diagnostics"] = diagnostics
        return (
            diagnostics["release_done"]
            and diagnostics["width_ratio"] < 0.85
            and diagnostics["right_edge_reduction"] > 0.035
        )
