from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from stable_baselines3 import SAC

from policy.RL.bc import load_bc_checkpoint
from policy.RL.dataset import ActionPhaseDataset, split_episode_paths


def _batch(samples):
    observation_keys = samples[0][0]
    observation = {
        key: torch.stack([sample[0][key] for sample in samples])
        for key in observation_keys
    }
    action = torch.stack([sample[1] for sample in samples])
    return observation, action


def _rmse(total_squared_error, total_values):
    return float(np.sqrt(total_squared_error / total_values))


def main():
    parser = argparse.ArgumentParser(
        description="Measure deterministic SAC drift from its SFT initializer."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("bc_checkpoint")
    parser.add_argument("sac_checkpoint")
    parser.add_argument("output")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    paths, _ = split_episode_paths(args.dataset_root)
    dataset = ActionPhaseDataset(paths, image_size=args.image_size)
    selected_indices = np.linspace(
        0,
        len(dataset) - 1,
        num=min(args.samples, len(dataset)),
        dtype=np.int64,
    )
    sft_model, _ = load_bc_checkpoint(args.bc_checkpoint, device="cpu")
    sac_model = SAC.load(args.sac_checkpoint, device="cpu")
    sac_model.policy.set_training_mode(False)

    target_squared_error = {"sft": 0.0, "sac": 0.0, "sac_vs_sft": 0.0}
    max_abs_difference = 0.0
    value_count = 0
    with torch.no_grad():
        for start_index in range(0, len(selected_indices), args.batch_size):
            batch_indices = selected_indices[start_index : start_index + args.batch_size]
            observation, next_qpos = _batch([dataset[int(index)] for index in batch_indices])
            target = torch.clamp(
                (next_qpos - observation["qpos"] - sft_model.delta_mean)
                / sft_model.action_scale,
                min=-1.0,
                max=1.0,
            )
            sft_action = sft_model.forward_policy_action(observation)
            sac_action = sac_model.policy.actor(observation, deterministic=True)
            target_squared_error["sft"] += float(((sft_action - target) ** 2).sum())
            target_squared_error["sac"] += float(((sac_action - target) ** 2).sum())
            target_squared_error["sac_vs_sft"] += float(
                ((sac_action - sft_action) ** 2).sum()
            )
            max_abs_difference = max(
                max_abs_difference,
                float((sac_action - sft_action).abs().max()),
            )
            value_count += int(target.numel())

    summary = {
        "samples": len(selected_indices),
        "dataset_records": len(dataset),
        "sft_target_rmse": _rmse(target_squared_error["sft"], value_count),
        "sac_target_rmse": _rmse(target_squared_error["sac"], value_count),
        "sac_vs_sft_rmse": _rmse(
            target_squared_error["sac_vs_sft"],
            value_count,
        ),
        "sac_vs_sft_max_abs": max_abs_difference,
    }
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
