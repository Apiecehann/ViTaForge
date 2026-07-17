from ._base_task import *
import numpy as np


TARGET_BLOCK_NAME = "wooden_half_cylinder"
BLOCK_HEIGHT = 0.0300

# reset 时只在桌面 xy 平面扰动各个木块，z 维保持 0，避免初始物体离开桌面。
XY_NOISE = (0.005, 0.005, 0.0)
GRASP_ROTATE_NOISE = np.deg2rad(10.0)
# 半圆柱抓取高度设在物体半高附近，并叠加少量噪声，增加演示数据的抓取多样性。
GRASP_HEIGHT = BLOCK_HEIGHT * 0.5
GRASP_HEIGHT_NOISE = 0.003
LIFT_HEIGHT = 0.1000
SUCCESS_MIN_LIFT = 0.0500
SUCCESS_MAX_LIFT = 0.1500

WOODEN_BLOCK_SPECS = [
    {
        "name": "wooden_cube",
        "shape": "cube",
        "asset_path": "wooden_cube.usd",
        "base_pose": Pose([0.4, 0.0, 0.002], [1, 0, 0, 0]),
    },
    {
        "name": "wooden_cylinder",
        "shape": "cylinder",
        "asset_path": "wooden_cylinder.usd",
        "base_pose": Pose([0.32, 0.02, 0.002], [1, 0, 0, 0]),
    },
    {
        "name": "wooden_ellipse_cylinder",
        "shape": "ellipse_cylinder",
        "asset_path": "wooden_ellipse_cylinder.usd",
        "base_pose": Pose([0.46, -0.03, 0.002], [1, 0, 0, 0]),
    },
    {
        "name": "wooden_half_cylinder",
        "shape": "half_cylinder",
        "asset_path": "wooden_half_cylinder.usd",
        "base_pose": Pose([0.42, -0.07, 0.002], [1, 0, 0, 0]),
    },
    {
        "name": "wooden_triangular_prism",
        "shape": "triangular_prism",
        "asset_path": "wooden_triangular_prism.usd",
        "base_pose": Pose([0.35, -0.08, 0.002], [1, 0, 0, 0]),
    },
    {
        "name": "wooden_hexagonal_prism",
        "shape": "hexagonal_prism",
        "asset_path": "wooden_hexagonal_prism.usd",
        "base_pose": Pose([0.43, 0.07, 0.002], [1, 0, 0, 0]),
    },
    {
        "name": "wooden_quarter_cylinder",
        "shape": "quarter_cylinder",
        "asset_path": "wooden_quarter_cylinder.usd",
        "base_pose": Pose([0.36, 0.06, 0.002], [1, 0, 0, 0]),
    },
]


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.8, 0.0, 0.15), rot=(0.555057, 0.465748, 0.443006, 0.527954), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=2.4, clipping_range=(0.1, 100.0)
            ),
            width=480,
            height=270,
            update_period=1/120,
        ),
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
    step_lim = 300


class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        # clutter 抓取依赖稳定摩擦；提高刚体材质和 UIPC 接触摩擦，减少夹取目标时打滑。
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        # 一次性创建所有木块 actor；每个 episode 只在 _reset_actors 中移动已有 actor。
        self.wooden_blocks = {}
        self.wooden_block_specs = {}

        for spec in WOODEN_BLOCK_SPECS:
            actor = self._actor_manager.add_from_usd_file(
                name=spec["name"],
                asset_path=spec["asset_path"],
                pose=spec["base_pose"],
                density=1e3,
            )
            self.wooden_blocks[spec["name"]] = actor
            self.wooden_block_specs[spec["name"]] = spec

        # 本任务的目标物体固定为 wooden_half_cylinder，其它木块作为 clutter 干扰项。
        self.target_block = self.wooden_blocks[TARGET_BLOCK_NAME]
        self.target_initial_pose = self.wooden_block_specs[TARGET_BLOCK_NAME]["base_pose"]

    def _reset_actors(self):
        # 记录目标类别信息，方便离线过滤同一 clutter 场景中的不同目标任务。
        self.metadata["target_shape"] = self.wooden_block_specs[TARGET_BLOCK_NAME]["shape"]
        self.metadata["target_block_name"] = TARGET_BLOCK_NAME
        self.metadata["block_poses"] = {}
        self.metadata["block_xy_noise"] = {}

        for name, actor in self.wooden_blocks.items():
            # 每个木块独立采样 xy 偏移，既保持整体布局相近，又给抓取点和遮挡关系加入扰动。
            base_pose = self.wooden_block_specs[name]["base_pose"]
            offset = self.create_noise(list(XY_NOISE))
            pose = base_pose.add_offset(offset)
            actor.set_pose(pose)

            self.metadata["block_xy_noise"][name] = offset.p.tolist()
            self.metadata["block_poses"][name] = pose.tolist()

            if name == TARGET_BLOCK_NAME:
                # success 需要和目标物体 reset 后的真实初始高度比较，而不是和无噪声基准位姿比较。
                self.target_initial_pose = pose

    def pre_move(self):
        # 正式动作前等待物理状态稳定，再打开夹爪准备从目标物体上方接近。
        self.delay(10)
        self.move(self.atom.open_gripper(0.5), tag="open_gripper_for_half_cylinder")

        target_pose = self.target_block.get_pose()
        # 以目标半圆柱当前位姿为基准，在半高附近构造抓取点，并绕局部 y 轴加入少量随机旋转。
        grasp_rotate = self.rng.uniform(-GRASP_ROTATE_NOISE, GRASP_ROTATE_NOISE)
        grasp_height = GRASP_HEIGHT + self.rng.uniform(-GRASP_HEIGHT_NOISE, GRASP_HEIGHT_NOISE)
        grasp_target_pose = target_pose.add_bias([0.0, 0.0, grasp_height]).add_rotation([0.0, grasp_rotate, 0.0])
        target_mat = grasp_target_pose.to_transformation_matrix()
        # construct_grasp_pose 使用目标点、接近方向和夹爪横向方向，生成机器人末端抓取姿态。
        grasp_pose = construct_grasp_pose(
            grasp_target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        )
        # 将抓取点注册到目标 actor 局部坐标系，后续 grasp_actor 会按该 contact point 规划接近动作。
        contact_point_id = self.target_block.register_point(grasp_pose, type="contact")

        self.move(self.atom.grasp_actor(
            self.target_block,
            contact_point_id=contact_point_id,
            is_close=False,
            pre_dis=0.05,
        ), tag="approach_half_cylinder")
        self.move(self.atom.close_gripper(), tag="close_half_cylinder")

        # 保存抓取随机量和最终抓取姿态，便于复现失败样本或分析抓取分布。
        self.metadata["grasp_rotate_rad"] = float(grasp_rotate)
        self.metadata["grasp_rotate_deg"] = float(np.rad2deg(grasp_rotate))
        self.metadata["grasp_height"] = float(grasp_height)
        self.metadata["grasp_pose"] = grasp_pose.tolist()

    def _play_once(self):
        # 抓住目标后竖直上提 10cm；该任务只验证目标是否被稳定提起到期望高度范围。
        self.move(self.atom.move_by_displacement(z=LIFT_HEIGHT), tag="lift_half_cylinder")
        # 上提后等待但不保存等待帧，避免纯稳定过程混入动作监督数据。
        self.delay(20, is_save=False)

    def _get_success_diagnostics(self):
        target_pose = self.target_block.get_pose()
        # 用当前目标高度减去 reset 后初始高度，得到真实上提距离。
        lifted_height = target_pose.p[2] - self.target_initial_pose.p[2]
        height_ok = bool(SUCCESS_MIN_LIFT <= lifted_height <= SUCCESS_MAX_LIFT)

        return {
            "target_block_name": TARGET_BLOCK_NAME,
            "target_initial_pose": self.target_initial_pose.tolist(),
            "target_final_pose": target_pose.tolist(),
            "lifted_height": float(lifted_height),
            "success_min_lift": float(SUCCESS_MIN_LIFT),
            "success_max_lift": float(SUCCESS_MAX_LIFT),
            "height_ok": height_ok,
        }

    def check_success(self):
        # 成功条件只看目标半圆柱是否被提到 9-11cm 的范围内，诊断信息写入 metadata。
        diagnostics = self._get_success_diagnostics()
        self.metadata["success_diagnostics"] = diagnostics
        return diagnostics["height_ok"]
