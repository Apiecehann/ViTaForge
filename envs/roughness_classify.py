from ._base_task import *
import numpy as np

TASK_INSTRUCTION = "Classify the block surface by touch: place the smooth block on the blue plate and the rough block on the yellow plate."

GRIPPER_CLOSE_QPOS_RANGE = (0.0035, 0.0040)
BLOCK_RESET_XY_NOISE = 0.01
LIFT_HEIGHT = 0.05
LIFT_HEIGHT_NOISE = 0.01
TARGET_XY_NOISE = 0.01


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.9, 0.0, 0.15), rot=(0.5, 0.5, 0.5, 0.5), convention="opengl"),
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
            spawn=None,
            width=480,
            height=270,
            update_period=1/120,
        )
    ]
    use_adaptive_grasp = False


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        self.blue_plate = self._actor_manager.add_from_usd_file(
            name='blue_plate',
            asset_path="roughness_task/blue_square_plate.usd",
            pose=Pose([0.45, 0.08, 0.002], [1, 0, 0, 0]),
        )
        self.yellow_plate = self._actor_manager.add_from_usd_file(
            name='yellow_plate',
            asset_path="roughness_task/yellow_square_plate.usd",
            pose=Pose([0.45, -0.08, 0.002], [1, 0, 0, 0]),
        )

        self.smooth_block = self._actor_manager.add_from_usd_file(
            name='smooth_block',
            asset_path="roughness_task/black_smooth_cuboid.usd",
            pose=Pose([0.38, 1.0, 0.002], [1, 0, 0, 0]),
        )
        self.rough_block = self._actor_manager.add_from_usd_file(
            name='rough_block',
            asset_path="roughness_task/black_rough_cuboid.usd",
            pose=Pose([0.38, -1.0, 0.002], [1, 0, 0, 0]),
        )

    def _reset_actors(self):
        self.choice = self.rng.choice(['smooth', 'rough'])
        start_pose = Pose([
            0.38 + self.rng.uniform(-BLOCK_RESET_XY_NOISE, BLOCK_RESET_XY_NOISE),
            self.rng.uniform(-BLOCK_RESET_XY_NOISE, BLOCK_RESET_XY_NOISE),
            0.002
        ], [1, 0, 0, 0])
        if self.choice == 'smooth':
            self.block = self.smooth_block
            self.target = self.blue_plate
            self.other_target = self.yellow_plate
        else:
            self.block = self.rough_block
            self.target = self.yellow_plate
            self.other_target = self.blue_plate
        self.block.set_pose(start_pose)

    def pre_move(self):
        self.delay(10)

        self.move(self.atom.open_gripper(0.5))

        grasp_rotate = self.rng.uniform(-np.pi / 36, np.pi / 36)
        target_pose = self.block.get_pose().add_bias([0.0, 0.0, 0.04 + 0.01 * self.rng.random()])\
            .add_rotation([0, grasp_rotate, 0])
        target_mat = target_pose.to_transformation_matrix()
        cpose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0]
        )
        cid = self.block.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.block, contact_point_id=cid, pre_dis=0.04, dis=0.0, is_close=False
        ))
        gripper_qpos = self.rng.uniform(*GRIPPER_CLOSE_QPOS_RANGE) / 0.039
        self.move(self.atom.close_gripper(gripper_qpos))
        lift_height = LIFT_HEIGHT + self.rng.uniform(-LIFT_HEIGHT_NOISE, LIFT_HEIGHT_NOISE)
        self.move(self.atom.move_by_displacement(z=lift_height))

        self.target_pose = self.target.get_pose().add_bias([
            self.rng.uniform(-TARGET_XY_NOISE, TARGET_XY_NOISE),
            self.rng.uniform(-TARGET_XY_NOISE, TARGET_XY_NOISE),
            0.01
        ])

    def _play_once(self):
        self.move(self.atom.place_actor(
            self.block,
            target_pose=self.target_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False
        ), time_dilation_factor=0.5)
        self.delay(20, is_save=False)

    def check_success(self):
        block_pose = self.block.get_pose().rebase(self.target_pose)
        xy_threshold = 0.035
        z_threshold = 0.01
        return np.all(np.abs(block_pose.p) < np.array([xy_threshold, xy_threshold, z_threshold])) and \
            np.dot(block_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.965
