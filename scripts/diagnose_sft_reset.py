import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Compare online reset and HDF5 SFT observations.")
parser.add_argument("task_name")
parser.add_argument("task_config")
parser.add_argument("checkpoint")
parser.add_argument("hdf5_path")
parser.add_argument("output_dir")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--image-size", type=int, default=128)
parser.add_argument(
    "--control-mode",
    choices=("direct", "residual"),
    default="residual",
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

from policy.RL.dataset import ActionPhaseDataset
from policy.RL.gym_env import TactileControlEnv
from policy.RL.task_factory import create_task


def image_metrics(online, offline):
    online = np.asarray(online, dtype=np.float32)
    offline = np.asarray(offline, dtype=np.float32)
    candidates = {
        "same": online,
        "channel_reverse": online[::-1],
        "horizontal_flip": online[:, :, ::-1],
        "vertical_flip": online[:, ::-1, :],
        "rotate_180": online[:, ::-1, ::-1],
    }
    errors = {
        name: float(np.abs(candidate - offline).mean())
        for name, candidate in candidates.items()
    }
    return {
        "online_mean": online.mean(axis=(1, 2)).tolist(),
        "offline_mean": offline.mean(axis=(1, 2)).tolist(),
        "mae": errors,
        "best_alignment": min(errors, key=errors.get),
    }


def array_list(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).tolist()


def main():
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ActionPhaseDataset([args.hdf5_path], image_size=args.image_size)
    offline_observation, offline_action = dataset[0]
    task = create_task(
        args.task_name,
        args.task_config,
        save_dir=output_dir,
        video_frequency=0,
        step_limit=120,
    )
    environment = TactileControlEnv(
        task,
        args.checkpoint,
        image_size=args.image_size,
        action_repeat=2,
        control_gripper=True,
        control_mode=args.control_mode,
        seed=args.seed,
        device="cuda:0",
    )
    try:
        online_observation, _ = environment.reset(seed=args.seed)
        raw_observation = task._get_observations()
        with h5py.File(args.hdf5_path, "r") as hdf5_file:
            phase = hdf5_file["phase/id"][()]
            first_policy_index = int(np.flatnonzero(phase == 1)[0])
            offline_actors = {
                key: hdf5_file[f"actor/{key}"][first_policy_index].tolist()
                for key in hdf5_file["actor"].keys()
            }
        online_actors = {
            key: array_list(value)
            for key, value in raw_observation.get("actor", {}).items()
        }
        image_keys = environment.camera_keys + environment.tactile_keys
        comparison = {
            "seed": args.seed,
            "online_qpos": online_observation["qpos"].tolist(),
            "offline_qpos": offline_observation["qpos"].tolist(),
            "qpos_delta": (
                online_observation["qpos"] - offline_observation["qpos"].numpy()
            ).tolist(),
            "online_policy_step": online_observation["policy_step"].tolist(),
            "offline_policy_step": offline_observation["policy_step"].tolist(),
            "target_block_name": task.target_block_name,
            "online_actors": online_actors,
            "offline_actors": offline_actors,
            "images": {
                key: image_metrics(
                    online_observation[key],
                    offline_observation[key].numpy(),
                )
                for key in image_keys
            },
        }
        batch = {
            key: value.unsqueeze(0).to(environment.device)
            for key, value in offline_observation.items()
        }
        with torch.no_grad():
            comparison["offline_policy_action"] = (
                environment.bc_model.forward_policy_action(batch)[0].cpu().tolist()
            )
        comparison["online_policy_action"] = environment.sft_action(
            online_observation
        ).tolist()
        comparison["offline_target_qpos"] = offline_action.tolist()
        serialized = json.dumps(comparison, indent=2)
        (output_dir / "comparison.json").write_text(
            serialized,
            encoding="utf-8",
        )
        print(serialized, flush=True)
    finally:
        environment.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
