import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay future demonstration qpos targets.")
parser.add_argument("task_name")
parser.add_argument("task_config")
parser.add_argument("hdf5_path")
parser.add_argument("output_dir")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
parser.add_argument("--step-limit", type=int, default=120)
parser.add_argument("--action-repeat", type=int, default=2)
parser.add_argument("--restore-scene", action="store_true")
parser.add_argument(
    "--force-control",
    action=argparse.BooleanOptionalAction,
    default=False,
)
help_requested = any(argument in ("-h", "--help") for argument in sys.argv[1:])
if help_requested:
    parser.print_help()
    raise SystemExit(0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.num_envs = 1
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from policy.RL.task_factory import create_task
from envs.utils.transforms import Pose


def demonstration_state():
    with h5py.File(args.hdf5_path, "r") as hdf5_file:
        phase = hdf5_file["phase/id"][()]
        policy_indices = np.flatnonzero(phase == 1)
        first_index = int(policy_indices[0])
        qpos = hdf5_file["embodiment/joint"][policy_indices, :8].astype(np.float32)
        actors = {
            name: hdf5_file[f"actor/{name}"][first_index].astype(np.float64)
            for name in hdf5_file["actor"].keys()
        }
        return qpos, actors


def restore_scene(task, qpos, actors):
    for name, pose in actors.items():
        task._actor_manager.actors[name].set_pose(Pose(pose[:3], pose[3:]))
    task._actor_manager.update(dt=0.0)
    task._robot_manager.set_arm(qpos[0, :7], force=True)
    task._robot_manager.set_gripper(qpos[0, 7], force=True)
    for _ in range(2):
        task._step(is_save=False)
    task._actor_manager.remove_animate()
    task._actor_manager.update(dt=0.0)
    task.target_initial_pose = task.target_block.get_pose()


def main():
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qpos, actors = demonstration_state()
    task = create_task(
        args.task_name,
        args.task_config,
        save_dir=output_dir,
        video_frequency=0,
        step_limit=args.step_limit,
    )
    results = []
    try:
        for horizon in args.horizons:
            task.reset(seed=args.seed)
            if args.restore_scene:
                restore_scene(task, qpos, actors)
            terminated = truncated = False
            rewards = []
            tracking_errors = []
            last_info = {"metrics": task.get_rl_metrics()}
            for step in range(args.step_limit):
                target_index = min(step + horizon, len(qpos) - 1)
                observation, reward, terminated, truncated, last_info = task.env_step(
                    qpos[target_index],
                    action_type="qpos",
                    force=args.force_control,
                    action_repeat=args.action_repeat,
                )
                actual = observation["embodiment"]["joint"][:8]
                if isinstance(actual, torch.Tensor):
                    actual = actual.detach().cpu().numpy()
                reference_index = min(step + 1, len(qpos) - 1)
                tracking_errors.append(
                    float(np.sqrt(np.mean(np.square(actual - qpos[reference_index]))))
                )
                rewards.append(float(reward))
                if terminated or truncated:
                    break
            result = {
                "horizon": horizon,
                "success": bool(last_info.get("success", False)),
                "steps": len(rewards),
                "reward": float(sum(rewards)),
                "mean_tracking_rmse": float(np.mean(tracking_errors)),
                "final_tracking_rmse": float(tracking_errors[-1]),
                "metrics": last_info.get("metrics", {}),
            }
            print(json.dumps(result), flush=True)
            results.append(result)
            task.clean_cache(result="success" if result["success"] else "failed")
        serialized = json.dumps({"results": results}, indent=2)
        (output_dir / "horizon_calibration.json").write_text(
            serialized,
            encoding="utf-8",
        )
    finally:
        task.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
